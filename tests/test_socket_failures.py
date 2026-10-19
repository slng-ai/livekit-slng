"""Socket failures around a reply.

A transport failure after audio fails the reply, an error before any text retires a prewarmed
socket, acquisition keeps one deadline, and stray audio on an idle socket retires it.
"""

import asyncio
import time

import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectionError, APIConnectOptions, APIError

import livekit_slng as slng


def make(**kw):
    kw.setdefault("warm_standby_enabled", True)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


async def warm(t) -> None:
    t.prewarm()
    await asyncio.sleep(0.05)


# --- transport failures after audio are errors ----------------------------


class ResetAfterAudioWs(FakeWs):
    """Two chunks, then the connection drops with no close frame (CLOSED, 1006)."""

    async def _stream_reply(self) -> None:
        import aiohttp
        from fakews import _msg

        self.push_audio()
        self.push_audio()
        await asyncio.sleep(0.05)
        # What aiohttp hands a waiting reader when the connection is lost.
        self.kill()
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))


class ErrorAfterAudioWs(FakeWs):
    async def _stream_reply(self) -> None:
        self.push_audio()
        self.push_transport_error(ConnectionResetError("reset by peer"))


class CloseAfterAudioWs(FakeWs):
    """A model that ends the turn by closing, with a close frame."""

    async def _stream_reply(self) -> None:
        self.push_audio()
        self.push_close(1000)


@pytest.mark.parametrize("factory", [ResetAfterAudioWs, ErrorAfterAudioWs])
async def test_stream_raises_when_the_connection_fails_mid_reply(factory):
    t = make()
    install_fake_session(t, factory)
    started = time.perf_counter()
    with pytest.raises(APIConnectionError, match="failed after"):
        await drive_reply(t, sample_deltas())
    assert time.perf_counter() - started < 2.0, "failed by timeout, not by the transport check"
    await t.aclose()


async def test_stream_still_accepts_a_close_after_audio():
    t = make()
    install_fake_session(t, CloseAfterAudioWs)
    assert await drive_reply(t, sample_deltas()) > 0
    await t.aclose()


@pytest.mark.parametrize("factory", [ResetAfterAudioWs, ErrorAfterAudioWs])
async def test_synthesize_raises_when_the_connection_fails_mid_reply(factory):
    t = make()
    install_fake_session(t, factory)
    started = time.perf_counter()
    with pytest.raises(APIError, match="partway through synthesis"):
        await t.synthesize("Hello there, this is a test.").collect()
    assert time.perf_counter() - started < 2.0, "failed by timeout, not by the transport check"
    await t.aclose()


async def test_synthesize_still_accepts_a_close_after_audio():
    t = make()
    install_fake_session(t, CloseAfterAudioWs)
    frame = await t.synthesize("Hello there, this is a test.").collect()
    assert frame.samples_per_channel > 0
    await t.aclose()


# --- an error before any text retires a prewarmed socket ------------------


class RefusedInitWs(FakeWs):
    """The provider refuses the init, and the socket stays open."""

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "init":
            self.push_json({"type": "error", "code": "config_error", "message": "unknown voice"})
        elif self.sent[-1]["type"] == "text":
            self.push_json({"type": "error", "code": "config_error", "message": "unknown voice"})


async def test_an_error_on_a_prewarmed_socket_retires_it():
    t = make()
    session = install_fake_session(t, RefusedInitWs)
    await warm(t)
    first = session.created[0]
    assert first.closed and t._held is None, "a failed session stayed installed"
    assert t._failed_reopens == 1, "the reopen goes through the backoff"
    assert t._reconnect_handle is not None, "and is throttled, not immediate"
    started = time.perf_counter()
    with pytest.raises(APIError, match="unknown voice"):
        await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0))
    assert time.perf_counter() - started < 2.0, "the reply waited for a timeout"
    await t.aclose()


async def test_an_error_after_a_reply_keeps_the_socket():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    held = t._held
    held.ws.push_json({"type": "error", "code": "invalid_pronunciation", "message": "bad ref"})
    await asyncio.sleep(0.05)
    assert t._held is held and not held.ws.closed
    assert len(session.created) == 1
    await t.aclose()


# --- one deadline for the whole acquisition -------------------------------


async def test_acquisition_never_outlasts_its_timeout(monkeypatch):
    monkeypatch.setattr(slng.tts, "_PENDING_CONNECT_WAIT_S", 0.3)
    t = make()
    session = install_fake_session(t)
    real_connect = session.ws_connect
    calls = 0

    async def connect(url, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(30)  # the prewarm stalls
        await asyncio.sleep(0.35)  # the reply's own connect is slow
        return await real_connect(url, **kw)

    session.ws_connect = connect  # type: ignore[method-assign]
    t.prewarm()
    await asyncio.sleep(0.01)
    started = time.perf_counter()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await t._acquire_connection(timeout=0.5)
    took = time.perf_counter() - started
    assert took < 0.5 + 0.1, f"acquisition took {took:.2f}s against a 0.5s timeout"
    await t.aclose()


async def test_a_timely_prewarm_is_still_waited_for():
    t = make()
    session = install_fake_session(t, connect_delay_s=0.2)
    t.prewarm()
    await asyncio.sleep(0.01)
    conn = await t._acquire_connection(timeout=5.0)
    assert len(session.created) == 1 and conn.opened_by == "prewarm"
    await t._release_connection(conn, keep=True)
    await t.aclose()


# --- stray audio on an idle socket retires it -----------------------------


@pytest.mark.parametrize("kind", ["binary", "json"])
async def test_stray_audio_on_an_idle_socket_retires_it(kind):
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    first = t._held
    if kind == "binary":
        first.ws.push_audio()
    else:
        first.ws.binary_audio = False
        first.ws.push_audio()  # {"type": "audio_chunk", ...}
    await asyncio.sleep(0.05)
    assert first.ws.closed, "a socket with a turn still producing audio was kept"
    assert t._held is not None and t._held is not first
    assert t._held.opened_by == "stray_audio" and t._failed_reopens == 0
    # the next reply runs on the clean socket and hears only its own audio
    before = len(session.created)
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == before
    await t.aclose()


async def test_a_stray_non_audio_frame_keeps_the_socket():
    t = make()
    install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    held = t._held
    held.ws.push_json({"type": "Metadata", "request_id": "x"})
    await asyncio.sleep(0.05)
    assert t._held is held and not held.ws.closed
    await t.aclose()
