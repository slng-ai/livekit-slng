"""Detaching from an agent session, and server error text in logs.

A TTS no session uses any more closes its socket after a grace period and holds one again
once a session attaches; server error text is logged only under the lk.pii.error key.
"""

import asyncio
import logging

import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectOptions, APIError

import livekit_slng as slng


def make(**kw):
    kw.setdefault("warm_standby_enabled", True)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


class Activity:
    """What an AgentSession activity does to its TTS: listen on start, stop on close."""

    def __init__(self, t) -> None:
        self.t = t

    def _on_metrics(self, _m) -> None:
        pass

    def _on_error(self, _e) -> None:
        pass

    def start(self) -> None:
        self.t.prewarm()  # AgentActivity.start() prewarms before _start_session attaches
        self.t.on("metrics_collected", self._on_metrics)
        self.t.on("error", self._on_error)

    def close(self) -> None:
        self.t.off("metrics_collected", self._on_metrics)
        self.t.off("error", self._on_error)


@pytest.fixture
def short_grace(monkeypatch):
    monkeypatch.setattr(slng.tts, "_DETACH_GRACE_S", 0.1, raising=False)


# --- replaced TTS releases its socket ----------------------------------------


async def test_a_replaced_tts_closes_its_socket_and_stays_closed(short_grace):
    old = make()
    session = install_fake_session(old)
    a = Activity(old)
    a.start()
    await asyncio.sleep(0.05)
    assert old._held is not None
    a.close()  # handoff to an agent with its own TTS; nothing closes this one
    await asyncio.sleep(0.2)
    assert session.created[0].closed and old._held is None
    await asyncio.sleep(0.3)
    assert len(session.created) == 1, "a detached instance reopened in the background"
    await old.aclose()


async def test_a_shared_tts_survives_a_handoff(short_grace):
    t = make()
    session = install_fake_session(t)
    a, b = Activity(t), Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    held = t._held
    a.close()
    await asyncio.sleep(0.05)  # inside the grace window, e.g. MCP servers connecting
    b.start()
    await asyncio.sleep(0.2)
    assert t._held is held and not held.ws.closed and len(session.created) == 1
    await t.aclose()


async def test_reattaching_after_a_detach_reopens(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    a.close()
    await asyncio.sleep(0.2)
    assert t._held is None
    Activity(t).start()  # e.g. update_agent back to the agent that owns this TTS
    await asyncio.sleep(0.05)
    assert t._held is not None and len(session.created) == 2
    await t.aclose()


async def test_a_reply_in_flight_at_detach_closes_its_socket_when_done(short_grace):
    t = make()
    session = install_fake_session(
        t, lambda: FakeWs(audio_chunks_per_reply=30, chunk_interval_s=0.01)
    )
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    reply = asyncio.create_task(drive_reply(t, sample_deltas()))
    await asyncio.sleep(0.05)
    a.close()
    assert await reply > 0
    await asyncio.sleep(0.2)
    assert t._held is None and session.created[0].closed
    await t.aclose()


async def test_a_new_stream_after_a_detach_holds_again(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    a.close()
    await asyncio.sleep(0.2)
    assert await drive_reply(t, sample_deltas()) > 0
    assert t._held is not None and len(session.created) == 2
    await t.aclose()


async def test_a_tts_used_without_a_session_is_left_alone(short_grace):
    t = make()
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.3)
    assert t._held is not None and not session.created[0].closed
    await t.aclose()


async def test_every_candidate_is_released(short_grace):
    t = slng.TTS(api_key="k", connections=["deepgram/aura:2", "cartesia/sonic:3.5"], voice="v")
    primary, fallback = t._candidate_tts
    ps, _fs = install_fake_session(primary), install_fake_session(fallback)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    assert primary._held is not None
    a.close()
    await asyncio.sleep(0.2)
    assert primary._held is None and fallback._held is None and ps.created[0].closed
    await t.aclose()


# --- server error text is marked as PII in logs --------------------------------


def _records_with(caplog, message):
    return [r for r in caplog.records if r.getMessage() == message]


async def test_idle_error_text_is_marked_pii(caplog):
    t = make()
    install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        t._held.ws.push_json(
            {"type": "error", "code": "invalid_pronunciation", "message": "refused: my card 4242"}
        )
        await asyncio.sleep(0.05)
    (rec,) = _records_with(caplog, "[TTS] error on idle connection")
    assert getattr(rec, "lk.pii.error") == "refused: my card 4242"
    assert not hasattr(rec, "error")
    await t.aclose()


class RefusingWs(FakeWs):
    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "text":
            self.push_json({"type": "error", "code": "config_error", "message": "refused: 4242"})


async def test_segment_log_keeps_the_error_type_and_marks_the_message(caplog):
    t = make(warm_standby_enabled=False)
    install_fake_session(t, RefusingWs)
    with caplog.at_level(logging.INFO, logger="livekit_slng"):
        with pytest.raises(APIError):
            await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0))
    (rec,) = _records_with(caplog, "[TTS] segment timing")
    assert rec.outcome == "error" and rec.error_type == "APIStatusError"
    assert "4242" in getattr(rec, "lk.pii.error")
    assert not hasattr(rec, "error")
    await t.aclose()


async def test_the_redactor_drops_the_marked_fields():
    from livekit.agents.telemetry.pii import is_pii_attribute

    assert is_pii_attribute("lk.pii.error")
    assert not is_pii_attribute("error_type") and not is_pii_attribute("error_status")


# --- a pending detach must not undo a reattach -------------


async def test_a_reattach_before_the_cleanup_runs_keeps_the_socket(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    held = t._held
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()  # the grace timer fires...
    Activity(t).start()  # ...and a session attaches before the cleanup task has run
    await asyncio.sleep(0.2)
    assert t._hold_allowed, "the reattached session was left unable to hold a socket"
    assert t._held is held and not held.ws.closed and len(session.created) == 1
    await t.aclose()


async def test_a_reattach_while_the_cleanup_waits_keeps_the_socket(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    held = t._held
    a.close()
    t._detach_handle.cancel()
    await t._ws_lock.acquire()  # something else owns the lock for a moment
    t._detach_if_unused()
    await asyncio.sleep(0.02)  # the cleanup is now waiting for the lock
    Activity(t).start()
    t._ws_lock.release()
    await asyncio.sleep(0.2)
    assert t._hold_allowed and t._held is held and not held.ws.closed
    assert len(session.created) == 1
    await t.aclose()


async def test_a_stream_before_the_cleanup_runs_keeps_its_socket(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()
    stream = t.stream()
    await stream.__aenter__()  # the stream starts before the cleanup task has run
    for d in sample_deltas():
        stream.push_text(d)
    stream.end_input()
    total = 0
    async for ev in stream:
        total += len(ev.frame.data)
    await stream.aclose()
    assert total > 0
    await asyncio.sleep(0.1)
    assert t._held is not None and not t._held.ws.closed
    assert len(session.created) == 1, "the stream's socket was dropped by a stale detach"
    await t.aclose()


async def test_a_stream_after_the_cleanup_holds_a_new_socket(short_grace):
    t = make()
    session = install_fake_session(t)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.05)
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()
    await asyncio.sleep(0.05)  # the cleanup has closed the old socket
    assert session.created[0].closed
    assert await drive_reply(t, sample_deltas()) > 0
    assert t._held is not None and not t._held.ws.closed and len(session.created) == 2
    await t.aclose()


async def test_a_connect_in_flight_at_detach_installs_only_if_reattached(short_grace):
    t = make()
    session = install_fake_session(t, connect_delay_s=0.1)
    a = Activity(t)
    a.start()  # prewarm connect in flight
    await asyncio.sleep(0.01)
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()
    await asyncio.sleep(0.2)
    assert t._held is None and session.created[0].closed, "a detached instance installed a socket"
    Activity(t).start()
    await asyncio.sleep(0.2)
    assert t._held is not None and not t._held.ws.closed
    await t.aclose()


class SlowCloseWs(FakeWs):
    async def close(self) -> bool:
        await asyncio.sleep(0.15)
        return await super().close()


async def test_a_reattach_while_a_refused_connect_closes_still_prewarms(short_grace):
    """The refused connect still counts as in flight while it closes."""
    t = make()
    session = install_fake_session(t, SlowCloseWs, connect_delay_s=0.1)
    a = Activity(t)
    a.start()  # prewarm connect in flight
    await asyncio.sleep(0.01)
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()
    await asyncio.sleep(0.15)  # the connect landed, was refused, and is closing its socket
    assert t._connect_task is not None and not t._connect_task.done()
    Activity(t).start()  # this prewarm is turned away: a connect is "in flight"
    await asyncio.sleep(0.5)
    assert t._held is not None and not t._held.ws.closed, "the reattached session lost its prewarm"
    assert session.created[0].closed and len(session.created) == 2
    await t.aclose()


async def test_a_refused_connect_on_a_still_detached_instance_does_not_reopen(short_grace):
    t = make()
    session = install_fake_session(t, SlowCloseWs, connect_delay_s=0.1)
    a = Activity(t)
    a.start()
    await asyncio.sleep(0.01)
    a.close()
    t._detach_handle.cancel()
    t._detach_if_unused()
    await asyncio.sleep(0.6)
    assert t._held is None and len(session.created) == 1
    await t.aclose()
