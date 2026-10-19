"""Warm standby: prewarm opens the held socket early, and a lost socket reopens on its own."""

import asyncio
import contextlib
import logging

from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectOptions

import livekit_slng as slng


def make(**kw):
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def test_default_on_and_pool_gone():
    t = make()
    assert t.warm_standby_enabled is True
    assert make(warm_standby_enabled=False).warm_standby_enabled is False
    for name in (
        "_standby",
        "_standby_lock",
        "_standby_task",
        "_standby_close_tasks",
        "_ws_connection_timings",
        "_standby_epoch",
    ):
        assert not hasattr(t, name), name
    assert not hasattr(slng.tts, "_WarmStandbyConnection")


async def test_prewarm_opens_the_held_connection_before_text(caplog):
    t = make()
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.05)
    assert len(session.created) == 1
    assert t._held is not None and t._held.opened_by == "prewarm"
    with caplog.at_level(logging.INFO, logger="livekit_slng"):
        await drive_reply(t, sample_deltas())
    seg = next(r for r in caplog.records if r.getMessage() == "[TTS] segment timing")
    assert seg.ws_source == "reused" and seg.ws_connect_ms is None
    assert len(session.created) == 1
    await t.aclose()


async def test_prewarm_only_active_candidate():
    t = slng.TTS(api_key="k", connections=["deepgram/aura:2", "cartesia/sonic:3.5"], voice="v")
    p, f = t._candidate_tts
    ps, fs = install_fake_session(p), install_fake_session(f)
    t.prewarm()
    await asyncio.sleep(0.05)
    assert len(ps.created) == 1 and len(fs.created) == 0
    await t.aclose()


async def test_idle_close_reopens_in_background_when_enabled_every_time():
    t = make()
    session = install_fake_session(t, lambda: FakeWs(close_after_reply=True))
    for _ in range(3):  # quick successive replies: served sockets are not throttled
        await drive_reply(t, sample_deltas())
        await asyncio.sleep(0.05)
        assert t._held is not None and t._held.opened_by == "reconnect"
    assert len(session.created) == 4
    await t.aclose()


async def test_never_served_socket_reopen_is_throttled():
    t = make()
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.05)
    session.created[0].push_close()  # gateway closes a prewarmed, never-used socket
    await asyncio.sleep(0.1)
    assert len(session.created) == 1  # throttled: no immediate second attempt
    await t.aclose()


async def test_idle_close_does_not_reopen_when_disabled():
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t, lambda: FakeWs(close_after_reply=True))
    await drive_reply(t, sample_deltas())
    await asyncio.sleep(0.05)
    assert len(session.created) == 1 and t._held is None
    await t.aclose()


async def test_update_options_reopens_with_new_init_when_enabled():
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    t.update_options(voice="other")
    await asyncio.sleep(0.05)
    assert len(session.created) == 2 and session.created[1].sent[0]["voice"] == "other"
    assert t._held is not None and t._held.opened_by == "options_changed"
    await t.aclose()


async def test_disabled_never_second_socket_and_aclose_closes():
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.02)
    assert session.created == []
    await drive_reply(t, sample_deltas())
    await t.aclose()
    assert len(session.created) == 1 and session.created[0].closed


# --- prewarm and background connect ---------------------------


async def test_prewarmed_first_reply_is_visible_in_the_segment_log(caplog):
    """ws_source says "reused" for a prewarmed socket, so it needs its own fields."""
    t = make()
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.05)
    with caplog.at_level(logging.INFO, logger="livekit_slng"):
        await drive_reply(t, sample_deltas())
        await drive_reply(t, sample_deltas())
    segments = [r for r in caplog.records if r.getMessage() == "[TTS] segment timing"]
    assert len(segments) == 2
    assert segments[0].ws_opened_by == "prewarm" and segments[0].ws_first_use is True
    assert segments[1].ws_opened_by == "prewarm" and segments[1].ws_first_use is False
    assert len(session.created) == 1
    await t.aclose()


async def test_interrupted_first_reply_still_reopens_in_background():
    """Warm standby must survive a call whose first reply is interrupted."""
    t = make()
    session = install_fake_session(
        t, lambda: FakeWs(on_cancel="silent", audio_chunks_per_reply=40, chunk_interval_s=0.01)
    )
    t.prewarm()
    await asyncio.sleep(0.05)
    assert len(session.created) == 1
    # Interrupted before it ever completed a reply: served_segments stays 0.
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    await asyncio.sleep(slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S + 0.4)
    assert session.created[0].closed, "an unacknowledged clear must not keep the socket"
    assert t._held is not None, "warm standby was abandoned after one interrupt"
    assert len(session.created) == 2
    await t.aclose()


async def test_options_change_during_a_pending_connect_reopens_with_the_new_voice():
    """The reschedule from inside the connect task must not be dropped."""
    t = make()
    session = install_fake_session(t, connect_delay_s=0.15)
    t.prewarm()
    await asyncio.sleep(0.05)  # connect in flight, nothing installed yet
    assert t._held is None
    t.update_options(voice="other")
    await asyncio.sleep(slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S + 0.2)
    assert t._held is not None, "no socket was reopened after the options changed"
    assert session.created[-1].sent[0]["voice"] == "other"
    await t.aclose()


async def test_cancelled_background_connect_does_not_cancel_a_waiting_reply(caplog):
    """A prewarm cancelled by aclose/failover must not look like a barge-in."""
    t = make()
    install_fake_session(t, connect_delay_s=0.4)
    t.prewarm()
    await asyncio.sleep(0.05)
    connect_task = t._connect_task
    assert connect_task is not None

    with caplog.at_level(logging.INFO, logger="livekit_slng"):
        reply = asyncio.create_task(drive_reply(t, sample_deltas()))
        await asyncio.sleep(0.15)  # the reply is parked on that connect
        connect_task.cancel()
        assert await reply > 0, "the reply died with the background connect"
    segments = [r for r in caplog.records if r.getMessage() == "[TTS] segment timing"]
    assert segments, "no segment was logged"
    assert [s.outcome for s in segments] == ["completed"], (
        "a cancelled connect was logged as a barge-in"
    )
    await t.aclose()


async def test_pending_connect_wait_is_bounded_by_the_caller_timeout():
    """A stalled prewarm must not hold the first reply for its own timeout."""
    t = make()
    install_fake_session(t, connect_delay_s=3.0)
    t.prewarm()
    await asyncio.sleep(0.02)
    started = asyncio.get_running_loop().time()
    reply = asyncio.create_task(
        drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    )
    with contextlib.suppress(Exception):
        await reply
    waited = asyncio.get_running_loop().time() - started
    assert waited < 2.0, f"the reply waited {waited:.2f}s on the background connect"
    await t.aclose()


def test_prewarm_without_a_running_loop_warns(caplog):
    """Warm standby silently off is the failure nobody can diagnose."""
    t = make()
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        t.prewarm()
    assert any("no running event loop" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
