"""Connection reuse: one socket across replies, a clean socket after an interrupt, and replays.

Also covers the connection lifecycle around cancels, idle errors, keepalives and reconnects.
"""

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

import pytest
from fakews import AUDIO_B64, FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectionError, APIConnectOptions

import livekit_slng as slng


# Socket-count assertions here assume only replies connect. Warm standby is what
# holds a socket across replies, so it stays on, but its background reopen is
# switched off unless a test asks for warm_standby_enabled explicitly.
def make(**kw):
    explicit = "warm_standby_enabled" in kw
    kw.setdefault("warm_standby_enabled", True)
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)
    if not explicit:
        t._schedule_connect = lambda **_: None
    return t


# The cancel drain runs for the settle window plus the quiet grace; every test
# that waits for it to finish derives the wait from the plugin's own constants.
SETTLED_S = slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S + 0.2


def slow_ws(on_cancel="terminal"):
    return lambda: FakeWs(on_cancel=on_cancel, audio_chunks_per_reply=40, chunk_interval_s=0.01)


async def baseline_bytes(chunks: int = 40) -> int:
    """Audio bytes of one uninterrupted reply on a fresh TTS (what a clean reply yields)."""
    t = make()
    install_fake_session(t, lambda: FakeWs(audio_chunks_per_reply=chunks, chunk_interval_s=0.01))
    total = await drive_reply(t, sample_deltas())
    await t.aclose()
    return total


async def test_three_replies_share_one_socket():
    t = make()
    session = install_fake_session(t)
    for _ in range(3):
        assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1
    ws = session.created[0]
    assert ws.count("init") == 1 and ws.count("flush") == 3
    assert session.connect_kwargs[0].get("heartbeat") == 20.0
    await t.aclose()
    assert ws.sent[-1] == {"type": "close"} and ws.closed


async def test_a_close_after_each_reply_reconnects():
    t = make()
    session = install_fake_session(t, lambda: FakeWs(close_after_reply=True))
    for _ in range(3):
        assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 3
    await t.aclose()


async def test_dead_reused_socket_replays_segment():
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    session.created[0].kill()  # peer vanished, plugin still thinks it is open
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 2
    assert session.created[1].count("flush") == 1
    await t.aclose()


async def test_update_options_reconnects_once_with_new_voice():
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    t.update_options(voice="other")
    await asyncio.sleep(0.05)
    await drive_reply(t, sample_deltas())
    assert len(session.created) == 2 and session.created[0].closed
    assert session.created[1].sent[0]["voice"] == "other"
    await t.aclose()


@pytest.mark.parametrize(
    "on_cancel,expect_new_socket",
    [
        # The gateway acknowledges and goes quiet: the socket is clean, so it
        # is kept and the next reply costs no connect.
        ("terminal", False),
        # Nothing comes back at all, so the socket's state is unknown.
        ("silent", True),
        # The gateway ignores the interrupt and is still streaming the
        # cancelled reply when the window closes.
        ("streaming", True),
        # Audio behind the terminal frame: the next reply would hear it.
        ("terminal_then_more", True),
    ],
)
async def test_interrupt_never_bleeds(on_cancel, expect_new_socket):
    # "streaming" has to outlast the settle window to stay unresolved.
    chunks = 40 if on_cancel != "streaming" else int(slng.tts._CANCEL_SETTLE_S / 0.01) + 100
    expected = await baseline_bytes(chunks)
    t = make()
    session = install_fake_session(
        t, lambda: FakeWs(on_cancel=on_cancel, audio_chunks_per_reply=chunks, chunk_interval_s=0.01)
    )
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    await asyncio.sleep(SETTLED_S)
    assert session.created[0].count("clear") == 1
    assert session.created[0].count("cancel") == 0, "cancel is never acknowledged"
    total = await drive_reply(t, sample_deltas())
    assert total == expected  # exactly this reply's audio, nothing from the cancelled one
    assert (len(session.created) == 2) is expect_new_socket
    await t.aclose()


async def test_reply_during_drain_uses_private_socket_and_closes_it():
    expected = await baseline_bytes()
    t = make()
    session = install_fake_session(t, slow_ws("silent"))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    # no sleep: the cancel drain is still running when the next reply starts
    assert await drive_reply(t, sample_deltas()) == expected
    await asyncio.sleep(SETTLED_S)
    assert len(session.created) == 2
    assert session.created[1].closed  # private socket closed at release
    await t.aclose()


async def test_interrupt_before_any_text_keeps_socket():
    t = make()
    session = install_fake_session(t, slow_ws())
    await drive_reply(t, sample_deltas())  # establishes the held socket
    stream = t.stream()
    await asyncio.sleep(0.01)  # acquire happens on first text; none pushed yet
    await stream.aclose()
    await asyncio.sleep(0.05)
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1 and session.created[0].count("clear") == 0
    await t.aclose()


async def test_fallback_opens_own_socket_and_closes_primary():
    t = slng.TTS(
        api_key="k",
        connections=["deepgram/aura:2", "cartesia/sonic:3.5"],
        voice="v",
        warm_standby_enabled=False,
    )
    primary, fallback = t._candidate_tts
    ps, fs = install_fake_session(primary), install_fake_session(fallback)
    await drive_reply(t, sample_deltas())
    assert len(ps.created) == 1 and len(fs.created) == 0
    ps.created[0].kill()

    async def refuse(url, **kw):
        raise ConnectionRefusedError("primary down")

    ps.ws_connect = refuse  # type: ignore[method-assign]
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(fs.created) == 1 and ps.created[0].closed
    await t.aclose()


async def test_aclose_waits_for_all_background_tasks():
    t = make()
    session = install_fake_session(t, slow_ws("silent"))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    started = asyncio.get_running_loop().time()
    await t.aclose()  # must not hang, must not leave tasks running
    assert asyncio.get_running_loop().time() - started < 1.5
    await asyncio.sleep(0)
    assert not t._background_tasks and all(ws.closed for ws in session.created)


def _text_of(ws) -> str:
    return "".join(m["text"] for m in ws.texts())


def only_first_dies(fail_after_sends: int):
    """First socket dies after N writes; every later socket is healthy."""
    made = []

    def factory():
        ws = FakeWs(fail_after_sends=fail_after_sends if not made else None)
        made.append(ws)
        return ws

    return factory


async def _reuse_then_die(mode: str, fail_after_sends: int) -> None:
    """Reply 1 succeeds; the reused socket dies mid reply 2. Reply 2 must arrive whole."""
    from fakews import SAMPLE

    t = make(text_chunking=mode)
    session = install_fake_session(t, only_first_dies(fail_after_sends))
    assert await drive_reply(t, ["Short first reply. "]) > 0
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 2, "expected exactly one reconnect"
    replayed = _text_of(session.created[1])
    assert replayed.split() == SAMPLE.split(), replayed
    assert session.created[1].count("flush") == 1
    await t.aclose()


async def test_reconnect_replays_whole_reply_sentence_mode():
    # init + reply 1 (1 frame) = 2 sends; reply 2 fails on its second frame.
    await _reuse_then_die("sentence", fail_after_sends=3)


async def test_reconnect_replays_whole_reply_word_mode():
    await _reuse_then_die("word", fail_after_sends=6)


async def test_reconnect_replays_whole_reply_phrase_mode():
    await _reuse_then_die("phrase", fail_after_sends=4)


# --- connection lifecycle -------------------------------------


async def _drain(stream) -> None:
    with contextlib.suppress(Exception):
        async for _ in stream:
            pass


async def test_cancel_inside_first_send_still_clears():
    """A barge-in during send_str must not leave the gateway holding text."""
    t = make()
    session = install_fake_session(
        t, lambda: FakeWs(send_delay_s=0.2, audio_chunks_per_reply=40, chunk_interval_s=0.01)
    )
    await drive_reply(t, ["Warm up the socket with a long enough sentence. "])
    ws = session.created[0]
    assert ws.count("clear") == 0

    stream = t.stream()
    stream.push_text("This reply is interrupted while its frame is being written. ")
    stream.end_input()
    consume = asyncio.create_task(_drain(stream))
    await asyncio.sleep(0.05)  # inside send_str's delay, before it returns
    await stream.aclose()
    await consume
    await asyncio.sleep(0.8)

    assert ws.count("clear") == 1, ws.sent
    await t.aclose()


async def test_cancel_while_waiting_for_lock_closes_new_socket():
    """A cancel between connect and install must not orphan the socket."""
    t = make()
    session = install_fake_session(t, connect_delay_s=0.2)
    # max_retry=0: the fallback wrapper must not open a second socket behind
    # the cancelled attempt, which is the one this test inspects.
    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
    stream.push_text("A reply that is cancelled while its socket is installed. ")
    stream.end_input()
    consume = asyncio.create_task(_drain(stream))
    await asyncio.sleep(0.05)  # past the first lock section, inside the connect

    async with t._ws_lock:  # the install below now blocks here
        await asyncio.sleep(0.3)
        assert len(session.created) == 1, "expected the socket to be open already"
        assert t._held is None, "expected the install to still be blocked"
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.05)
    await closing
    await consume
    await asyncio.sleep(0.05)

    assert t._held is None
    assert all(ws.closed for ws in session.created), "a socket was left open"
    await t.aclose()


async def test_cancel_send_failure_reports_send_failed(caplog):
    """A socket that was already dead is not reported as a drain failure."""
    t = make()
    install_fake_session(t)
    conn = await t._acquire_connection(timeout=5.0)
    conn.ws.kill()  # dead before the cancel is written
    with caplog.at_level(logging.INFO, logger="livekit_slng"):
        with pytest.raises(ConnectionResetError):
            await t._cancel_and_settle(conn)
    settled = [r for r in caplog.records if r.getMessage() == "[TTS] cancel settled"]
    assert settled and settled[-1].cancel_outcome == "send_failed"
    await t.aclose()


async def test_late_audio_after_terminal_closes_socket():
    """The quiet window has a floor of _CANCEL_QUIET_S.

    The window is deadline + quiet, which is never less than quiet after the
    terminal frame. This pins that, with the ack landing just inside the
    deadline (the worst case).
    """
    t = make()
    # The ack lands at 0.28s, just inside the 0.3s settle deadline: the
    # narrowest quiet window the rule ever gives.
    session = install_fake_session(
        t,
        lambda: FakeWs(
            on_cancel="terminal_then_late",
            audio_chunks_per_reply=40,
            chunk_interval_s=0.01,
            cancel_ack_delay_s=0.28,
            late_audio_delay_s=0.06,
        ),
    )
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    await asyncio.sleep(1.2)
    assert session.created[0].closed, "socket kept despite audio after the terminal frame"
    await t.aclose()


async def test_idle_reader_receive_error_clears_held_socket():
    """A read failure still runs the bookkeeping instead of killing the task."""
    t = make()
    install_fake_session(t)
    await drive_reply(t, sample_deltas())
    held = t._held
    assert held is not None
    await t._stop_idle_reader(held)
    held.ws.raise_on_idle_receive = True
    t._start_idle_reader(held)
    await asyncio.sleep(0.05)
    assert t._held is None, "a socket whose reader died stayed installed"
    assert held.ws.closed
    await t.aclose()


async def test_idle_error_frame_emits_plugin_event():
    """A gateway error during a silence reaches the caller."""
    t = make()
    install_fake_session(t)
    events: list[Any] = []
    t.on("slng_event", events.append)
    await drive_reply(t, sample_deltas())
    held = t._held
    assert held is not None
    held.ws.push_json({"type": "error", "code": "rate_limit", "message": "slow down"})
    await asyncio.sleep(0.05)
    names = [e.name for e in events]
    assert "tts.idle_error" in names, names
    await t.aclose()


async def test_stray_idle_audio_is_logged_and_reported(caplog):
    """A frame between replies is the one bleed signal, so it is logged and reported."""
    t = make()
    install_fake_session(t)
    events: list[Any] = []
    t.on("slng_event", events.append)
    await drive_reply(t, sample_deltas())
    held = t._held
    assert held is not None
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        held.ws.push_json({"type": "audio_chunk", "data": AUDIO_B64})
        await asyncio.sleep(0.05)
        assert any(
            r.getMessage() == "[TTS] stray frame on idle connection" for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
    assert "tts.idle_stray_frame" in [e.name for e in events]
    await t.aclose()


async def test_candidate_switch_retires_a_socket_still_in_use():
    """The candidate left behind must not keep and reopen its socket."""
    t = slng.TTS(
        api_key="k",
        connections=["deepgram/aura:2", "cartesia/sonic:3.5"],
        voice="v",
    )
    primary, fallback = t._candidate_tts
    ps = install_fake_session(primary)
    install_fake_session(fallback)
    conn = await primary._acquire_connection(timeout=5.0)
    assert primary._held is conn and conn.in_use
    await t._drop_inactive_candidate_connections(fallback)
    assert primary._hold_allowed is False
    await primary._release_connection(conn, keep=True)
    await asyncio.sleep(0.05)
    assert primary._held is None, "the replaced candidate kept its socket"
    assert ps.created[0].closed
    assert len(ps.created) == 1, "the replaced candidate reopened a socket"
    await t.aclose()


async def test_idle_error_frame_keeps_a_socket_the_gateway_kept():
    """An error frame is not a terminator, so it must not close the socket."""
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    held = t._held
    assert held is not None
    for _ in range(3):
        held.ws.push_json(
            {"type": "error", "code": "invalid_pronunciation", "message": "no such entry"}
        )
        await asyncio.sleep(0.02)
    assert t._held is held, "the plugin tore down a connection the gateway kept"
    assert not held.ws.closed
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1, "no reconnect loop"
    await t.aclose()


async def test_idle_socket_is_kept_alive_with_a_keepalive_frame(monkeypatch):
    """The gateway's idle timer only counts text frames, not transport pings."""
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    ws = session.created[0]
    before = ws.count("keepalive")
    await asyncio.sleep(0.16)
    assert ws.count("keepalive") > before, ws.sent
    assert t._held is not None and not ws.closed
    await t.aclose()


async def test_transport_error_mid_reply_is_logged_as_a_failure(caplog):
    """WSMsgType.ERROR carries an exception; a clean close does not."""
    caplog.set_level(logging.DEBUG, logger="livekit_slng")

    class ErrorWs(FakeWs):
        async def _stream_reply(self) -> None:
            self.push_audio()
            self.push_transport_error(ConnectionResetError("invalid frame from gateway"))

    t = make()
    install_fake_session(t, ErrorWs)
    # A reply cut off by a transport failure is an error, not a short reply.
    with pytest.raises(APIConnectionError, match="invalid frame from gateway"):
        await drive_reply(t, sample_deltas())
    failures = [r for r in caplog.records if "websocket transport failed" in r.getMessage()]
    assert failures, "a transport error was indistinguishable from a clean close"
    assert "invalid frame from gateway" in str(getattr(failures[0], "ws_error", ""))
    await t.aclose()


async def test_close_without_audio_end_is_reported_as_such(caplog):
    """A socket that closes mid-reply is not 'terminal model output'."""
    caplog.set_level(logging.INFO, logger="livekit_slng")

    class TruncatingWs(FakeWs):
        async def _stream_reply(self) -> None:
            self.push_audio()
            self.push_close()

    t = make()
    install_fake_session(t, TruncatingWs)
    assert await drive_reply(t, sample_deltas()) > 0
    segments = [r for r in caplog.records if r.getMessage() == "[TTS] segment timing"]
    assert [s.outcome for s in segments] == ["closed_no_audio_end"]
    assert not any("terminal model output" in r.getMessage() for r in caplog.records)
    await t.aclose()


async def test_a_throttled_reconnect_still_happens(monkeypatch):
    """A socket that dies inside the throttle window must not end warm standby."""
    monkeypatch.setattr(slng.tts, "_BACKGROUND_CONNECT_MIN_INTERVAL_S", 0.1)
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.02)
    assert t._held is not None
    t._held.ws.push_close()  # gateway drops it at once, before any text
    await asyncio.sleep(0.05)
    assert t._held is None and len(session.created) == 1, "the retry fired too early"
    await asyncio.sleep(0.2)
    assert t._held is not None, "warm standby stopped for the rest of the call"
    assert len(session.created) == 2
    await t.aclose()


async def test_a_connection_that_never_works_is_eventually_abandoned(monkeypatch, caplog):
    """The deferred retry must not turn a permanent config error into a loop."""
    monkeypatch.setattr(slng.tts, "_BACKGROUND_CONNECT_MIN_INTERVAL_S", 0.05)
    caplog.set_level(logging.DEBUG, logger="livekit_slng")

    class RefusedAtInit(FakeWs):
        """The gateway accepts the socket and closes it as soon as it reads init."""

        async def send_str(self, data: str) -> None:
            await super().send_str(data)
            if json.loads(data)["type"] == "init":
                self.push_close(1008, "invalid voice")

    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    session = install_fake_session(t, RefusedAtInit)
    t.prewarm()
    # 0.05 + 0.10 + 0.20 of backoff, then the give-up.
    await asyncio.sleep(0.8)
    attempts = len(session.created)
    assert attempts <= slng.tts._MAX_BACKGROUND_REOPENS + 1, attempts
    await asyncio.sleep(0.8)
    assert len(session.created) == attempts, "it kept reopening a connection that never works"
    assert any("giving up on reopening" in r.getMessage() for r in caplog.records)
    await t.aclose()


async def test_a_socket_that_carried_text_starts_the_backoff_over():
    """A working connection proves the settings are fine, so failures start again at one."""
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    install_fake_session(t)
    await drive_reply(t, sample_deltas())
    held = t._held
    assert held is not None and held.text_sent
    t._failed_reopens = 3
    t._on_connection_lost(held, reason="reconnect")
    assert t._failed_reopens == 0
    await t.aclose()


async def test_a_reply_at_the_end_of_a_drain_waits_and_reuses_the_socket():
    """A reply arriving as the drain finishes takes the socket back, not a new one."""
    expected = await baseline_bytes()
    t = make()
    session = install_fake_session(t, slow_ws("terminal"))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    # Inside the hand-back window: the drain is about to release the socket.
    await asyncio.sleep(
        slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S - slng.tts._SETTLE_HANDBACK_WAIT_S / 2
    )
    started = time.monotonic()
    assert await drive_reply(t, sample_deltas()) == expected
    assert len(session.created) == 1, "it opened a private socket instead of waiting"
    assert time.monotonic() - started < 2.0
    await t.aclose()


async def test_a_reply_early_in_a_drain_does_not_wait():
    """Before the acknowledgement, the drain is far from done: open a private socket."""
    expected = await baseline_bytes()
    t = make()
    # The gateway takes half a second to acknowledge, and sends nothing after.
    session = install_fake_session(
        t,
        lambda: FakeWs(
            on_cancel="terminal_then_late",
            cancel_ack_delay_s=0.5,
            late_audio_delay_s=30.0,
            audio_chunks_per_reply=40,
            chunk_interval_s=0.01,
        ),
    )
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    started = time.monotonic()
    stream = t.stream()
    stream.push_text("Hello ")
    await asyncio.sleep(0.05)
    acquired = time.monotonic() - started
    stream.end_input()
    total = 0
    async for ev in stream:
        total += len(ev.frame.data)
    await stream.aclose()
    assert total == expected
    assert len(session.created) == 2, "the held socket was still draining"
    assert acquired < slng.tts._SETTLE_HANDBACK_WAIT_S + 0.05, (
        f"the reply waited {acquired:.3f}s for a drain that had just started"
    )
    await asyncio.sleep(slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S)
    await t.aclose()


async def test_the_quiet_window_runs_from_the_acknowledgement():
    """An immediate ack frees the socket ~_CANCEL_QUIET_S later, not at the deadline."""
    t = make()
    session = install_fake_session(t, slow_ws("terminal"))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    settle = t._settle
    assert settle is not None
    started = time.perf_counter()
    await asyncio.wait({settle.task}, timeout=2.0)
    took = time.perf_counter() - started
    assert took < slng.tts._CANCEL_QUIET_S + 0.1, f"drain took {took:.3f}s"
    assert t._held.ws is session.created[0]
    assert t._settle is None, "a finished drain is not left behind"
    await t.aclose()


async def test_a_reply_right_after_the_ack_waits_for_the_handback():
    """Once acknowledged, the drain is about to hand the socket back: wait, reuse it."""
    expected = await baseline_bytes()
    t = make()
    session = install_fake_session(t, slow_ws("terminal"))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    await asyncio.sleep(0.01)  # the ack is in, the quiet window is running
    assert t._settle is not None and t._settle.ends_at - time.perf_counter() <= 0.1
    assert await drive_reply(t, sample_deltas()) == expected
    assert len(session.created) == 1, "opened a private socket instead of waiting"
    await t.aclose()


async def test_a_drain_on_another_socket_never_delays_a_reply():
    """A private socket is closed at once, with no drain for a reply to wait on."""
    t = make()
    session = install_fake_session(t, slow_ws("terminal"))
    held = await t._acquire_connection(timeout=5.0)
    private = await t._acquire_connection(timeout=5.0)
    assert t._held is held and private is not held, "expected a private second socket"
    await t._release_connection(held, keep=True)  # idle and reusable from here

    t._release_after_cancel(private, in_flight=True, tasks_done=True)
    assert t._settle is None, "only the held socket's drain is recorded"
    started = time.monotonic()
    conn = await t._acquire_connection(timeout=5.0)
    waited = time.monotonic() - started
    assert conn is held, "the idle held socket was there to be reused"
    assert waited < slng.tts._SETTLE_HANDBACK_WAIT_S / 2
    await asyncio.sleep(0.05)
    assert private.ws.closed and private.ws.count("clear") == 0, "closed without a drain"
    assert len(session.created) == 2, "no third socket"
    await t._release_connection(conn, keep=True)
    await t.aclose()
