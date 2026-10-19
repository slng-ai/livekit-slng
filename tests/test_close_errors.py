"""Every error raised for a closed connection names the close code and the server's reason.

The STT failover logs keep a provider's error text out of the log message, under `lk.pii.error`.
"""

import asyncio
import logging
from types import SimpleNamespace

import aiohttp
import pytest
from fakews import FakeWs, _msg, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError
from test_gateway_errors import AUDIO, SttWs, envelope, stt_stream
from test_request_ids import read_until_error

import livekit_slng as slng

ONCE = APIConnectOptions(max_retry=0, timeout=2.0)


def make():
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=False)


class ClosesOnTextWs(FakeWs):
    """Closes the connection with ``code`` and ``reason`` instead of answering text."""

    def __init__(self, code: int = 1008, reason: str = "invalid voice") -> None:
        super().__init__()
        self.code = code
        self.reason = reason

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "text":
            self.push_close(self.code, self.reason)


class ClosesAfterAudioWs(FakeWs):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__()
        self.code = code
        self.reason = reason

    async def _stream_reply(self) -> None:
        self.push_audio()
        self.push_close(self.code, self.reason)


class ResetAfterAudioWs(FakeWs):
    async def _stream_reply(self) -> None:
        self.push_audio()
        await asyncio.sleep(0.02)
        self.kill()
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))


# --- TTS ---------------------------------------------------------------------------


async def test_a_close_before_audio_names_its_code_and_reason():
    t = make()
    install_fake_session(t, ClosesOnTextWs)
    with pytest.raises(APIStatusError, match="close code 1008, reason: invalid voice"):
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    await t.aclose()


async def test_a_fault_close_after_audio_names_its_reason():
    t = make()
    install_fake_session(t, lambda: ClosesAfterAudioWs(1011, "provider reset"))
    with pytest.raises(APIConnectionError, match="close code 1011, reason: provider reset"):
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    await t.aclose()


async def test_a_lost_connection_names_its_close_code():
    t = make()
    install_fake_session(t, ResetAfterAudioWs)
    with pytest.raises(APIConnectionError, match="failed after .*close code 1006"):
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    await t.aclose()


async def test_synthesize_names_the_close_code_and_reason():
    t = make()
    install_fake_session(t, lambda: ClosesOnTextWs(1013, "try again later"))
    with pytest.raises(APIStatusError, match="close code 1013, reason: try again later"):
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    await t.aclose()


async def test_synthesize_names_the_close_code_of_a_lost_connection():
    t = make()
    install_fake_session(t, ResetAfterAudioWs)
    with pytest.raises(APIError, match="partway through synthesis.*close code 1006"):
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    await t.aclose()


async def test_a_long_reason_is_cut_short():
    t = make()
    install_fake_session(t, lambda: ClosesOnTextWs(1008, "x" * 500))
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    assert "x" * 200 in raised.value.message and "x" * 201 not in raised.value.message
    await t.aclose()


# --- STT ---------------------------------------------------------------------------


def _close(code, reason):
    return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=code, extra=reason)


class ClosesOnInitWs(SttWs):
    async def send_str(self, msg: str) -> None:
        self.sent.append({"type": "init"})
        self._q.put_nowait(_close(1008, "unknown model"))


class ClosesOnAudioWs(SttWs):
    async def send_bytes(self, payload: bytes) -> None:
        self._q.put_nowait(_close(1011, "provider went away"))


def _stt_with(monkeypatch, factory, *, max_retry=0):
    stream, session = stt_stream({"deepgram/nova:3": {}}, max_retry=max_retry)

    async def ws_connect(url, **kw):
        session.connects.append(url)
        return factory()

    monkeypatch.setattr(session, "ws_connect", ws_connect)
    return stream


async def test_stt_closing_before_ready_names_the_code_and_reason(monkeypatch):
    stream = _stt_with(monkeypatch, lambda: ClosesOnInitWs({}))
    exc = await read_until_error(stream, APIConnectionError)
    assert "close code 1008, reason: unknown model" in str(exc)


async def test_stt_closing_mid_utterance_names_the_code_and_reason(monkeypatch):
    stream = _stt_with(monkeypatch, lambda: ClosesOnAudioWs({"type": "ready"}))
    exc = await read_until_error(stream)
    assert "close code 1011, reason: provider went away" in exc.message


async def test_stt_closing_repeatedly_names_the_last_close(monkeypatch):
    class ClosesAfterAnEmptyFinalWs(SttWs):
        """Ready, then on audio an empty final and a close, so nothing is left to replay."""

        async def send_bytes(self, payload: bytes) -> None:
            if not getattr(self, "_closed", False):
                self._closed = True
                self._put({"type": "final_transcript", "transcript": ""})
                self._q.put_nowait(_close(1008, "session limit"))

    stream = _stt_with(monkeypatch, lambda: ClosesAfterAnEmptyFinalWs({"type": "ready"}))

    async def feed():
        while True:
            await asyncio.sleep(0.02)
            stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))

    feeder = asyncio.create_task(feed())
    try:
        exc = await read_until_error(stream)
    finally:
        feeder.cancel()
    assert "repeatedly" in exc.message
    assert "close code 1008, reason: session limit" in exc.message


# --- STT failover logs carry no provider text -------------------------------------

SECRET = "the caller said secretive things"


async def test_stt_failover_logs_keep_provider_text_out_of_the_message(caplog):
    frame = envelope("provider_unavailable", message=f"provider refused: {SECRET}")
    stream, _ = stt_stream({"deepgram/nova:3": frame, "soniox/stt-rt:v3": frame}, max_retry=0)
    with caplog.at_level(logging.DEBUG, logger="livekit_slng"):
        await read_until_error(stream)
    records = [r for r in caplog.records if r.name.startswith("livekit_slng")]
    failures = [r for r in records if "STT" in r.getMessage() and r.levelno >= logging.WARNING]
    assert failures, "the failover is still logged"
    for record in records:
        assert "secretive" not in record.getMessage()
        if record.exc_info:
            assert "secretive" not in logging.Formatter().formatException(record.exc_info)
    assert all("secretive" in getattr(r, "lk.pii.error", "") for r in failures)
    assert all(getattr(r, "error_type", None) == "APIStatusError" for r in failures)


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
