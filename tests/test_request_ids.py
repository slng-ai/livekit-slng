"""SLNG's request ID on every raised error, so SLNG support can find the request.

The ID comes from the error frame, the connection's `ready`, or the `X-Slng-Request-Id` header of
a refused upgrade. It is never a local ID: support cannot look one up.
"""

import asyncio
import logging
from types import SimpleNamespace

import aiohttp
import pytest
import yarl
from fakews import FakeWs, _msg, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError
from multidict import CIMultiDict, CIMultiDictProxy
from test_gateway_errors import AUDIO, ErrorOnTextWs, SttWs, envelope, stt_stream

import livekit_slng as slng

ONCE = APIConnectOptions(max_retry=0, timeout=0.3)
RETRY3 = APIConnectOptions(max_retry=3, retry_interval=0.01)
READY_ID = "req"  # what FakeWs's `ready` carries
LEGACY_ERROR = {"type": "error", "code": "provider_error", "message": "upstream failed"}


def make():
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=False)


def handshake_error(status: int, request_id: str | None) -> aiohttp.WSServerHandshakeError:
    url = yarl.URL("wss://us-east.api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2")
    headers = CIMultiDict({"x-slng-request-id": request_id} if request_id else {})
    return aiohttp.WSServerHandshakeError(
        aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url),
        (),
        status=status,
        message="Invalid response status",
        headers=CIMultiDictProxy(headers),
    )


class RefusingSession:
    """Refuses every WebSocket upgrade, as the server does an unknown key."""

    def __init__(self, status: int, request_id: str | None) -> None:
        self.status = status
        self.request_id = request_id
        self.closed = False
        self.attempts = 0

    async def ws_connect(self, url: str, **kw):
        self.attempts += 1
        raise handshake_error(self.status, self.request_id)


class ResetAfterAudioWs(FakeWs):
    async def _stream_reply(self) -> None:
        self.push_audio()
        await asyncio.sleep(0.02)
        self.kill()
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))


class SilentWs(FakeWs):
    """Owes the reply's audio and never sends it."""

    async def _stream_reply(self) -> None:
        return


# --- TTS streaming ---------------------------------------------------------------


async def test_an_error_frame_reports_its_own_request_id():
    t = make()
    install_fake_session(t, lambda: ErrorOnTextWs(envelope("provider_error")))
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    assert raised.value.request_id == "gw-req-1"
    assert "request_id=gw-req-1" in str(raised.value)
    await t.aclose()


async def test_an_error_frame_without_one_reports_the_ready_id():
    t = make()
    install_fake_session(t, lambda: ErrorOnTextWs(LEGACY_ERROR))
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    assert raised.value.request_id == READY_ID
    await t.aclose()


async def test_a_refused_upgrade_reports_the_header_id():
    t = make()
    session = RefusingSession(401, "gw-hs-1")
    t._session = session
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert raised.value.status_code == 401 and raised.value.request_id == "gw-hs-1"
    assert session.attempts == 1, "a refused key is not retried"
    await t.aclose()


async def test_no_gateway_id_means_no_request_id():
    t = make()
    t._session = RefusingSession(503, None)
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    assert raised.value.request_id is None, "a local id is no use to SLNG support"
    await t.aclose()


async def test_a_connection_error_names_the_request_id():
    t = make()
    install_fake_session(t, ResetAfterAudioWs)
    with pytest.raises(APIConnectionError, match=f"SLNG request id: {READY_ID}"):
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    await t.aclose()


async def test_a_timeout_names_the_request_id():
    t = make()
    install_fake_session(t, SilentWs)
    with pytest.raises(APITimeoutError, match=f"SLNG request id: {READY_ID}"):
        await drive_reply(t, sample_deltas(), conn_options=ONCE)
    await t.aclose()


async def test_a_missed_first_audio_deadline_names_the_request_id():
    t = slng.TTS(
        api_key="k",
        model="deepgram/aura:2",
        voice="v",
        warm_standby_enabled=False,
        first_audio_timeout_s=0.2,
    )
    install_fake_session(t, SilentWs)
    with pytest.raises(APITimeoutError, match=f"0.2 s \\(SLNG request id: {READY_ID}\\)"):
        await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0))
    await t.aclose()


# --- TTS synthesize() --------------------------------------------------------------


async def test_synthesize_reports_the_error_frames_id():
    t = make()
    install_fake_session(t, lambda: ErrorOnTextWs(envelope("provider_error")))
    with pytest.raises(APIStatusError) as raised:
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    assert raised.value.request_id == "gw-req-1"
    await t.aclose()


async def test_synthesize_reports_the_ready_id():
    t = make()
    install_fake_session(t, lambda: ErrorOnTextWs(LEGACY_ERROR))
    with pytest.raises(APIStatusError) as raised:
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    assert raised.value.request_id == READY_ID
    await t.aclose()


@pytest.mark.parametrize("request_id", ["gw-hs-2", None])
async def test_synthesize_reports_the_header_id(request_id):
    t = make()
    t._session = RefusingSession(401, request_id)
    with pytest.raises(APIStatusError) as raised:
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    assert raised.value.status_code == 401 and raised.value.request_id == request_id
    await t.aclose()


async def test_synthesize_names_the_request_id_on_a_connection_error():
    t = make()
    install_fake_session(t, ResetAfterAudioWs)
    with pytest.raises(APIConnectionError, match=f"SLNG request id: {READY_ID}"):
        await t.synthesize("Hello there.", conn_options=ONCE).collect()
    await t.aclose()


# --- STT --------------------------------------------------------------------------


async def read_until_error(stream, error_type=APIStatusError):
    async def read():
        async for _ in stream:
            pass

    with pytest.raises(error_type) as raised:
        await asyncio.wait_for(read(), 3)
    await stream.aclose()
    return raised.value


async def test_stt_reports_the_error_frames_id():
    stream, _ = stt_stream({"deepgram/nova:3": envelope("provider_error")}, max_retry=0)
    exc = await read_until_error(stream)
    assert exc.request_id == "gw-req-1"


class ErrorOnAudioWs(SttWs):
    """Ready, then a legacy error frame, with no ID, once audio arrives."""

    async def send_bytes(self, payload: bytes) -> None:
        if not getattr(self, "_failed", False):
            self._failed = True
            self._put(LEGACY_ERROR)


async def test_stt_reports_the_ready_id_for_an_error_without_one(monkeypatch):
    stream, session = stt_stream({"deepgram/nova:3": {}}, max_retry=0)

    async def ws_connect(url, **kw):
        session.connects.append(url)
        return ErrorOnAudioWs({"type": "ready", "slng_request_id": "gw-ready-1"})

    monkeypatch.setattr(session, "ws_connect", ws_connect)
    exc = await read_until_error(stream)
    assert exc.request_id == "gw-ready-1"


async def test_stt_reports_the_header_id_of_a_refused_upgrade():
    session = RefusingSession(401, "gw-hs-3")
    stt_obj = slng.STT(api_key="k", model="deepgram/nova:3", http_session=session)  # type: ignore[arg-type]
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=2))
    for _ in range(3):
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))
    exc = await read_until_error(stream)
    assert exc.status_code == 401 and exc.request_id == "gw-hs-3"
    assert session.attempts == 1, "a refused key is not retried"


async def test_stt_still_retries_a_refused_upgrade_that_is_transient():
    session = RefusingSession(503, "gw-hs-4")
    stt_obj = slng.STT(api_key="k", model="deepgram/nova:3", http_session=session)  # type: ignore[arg-type]
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=2))
    for _ in range(3):
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))
    exc = await read_until_error(stream)
    assert exc.status_code == 503 and session.attempts == 3


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
