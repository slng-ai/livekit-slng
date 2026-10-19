"""The gateway's error contract, in both of its formats.

An error frame's status decides retry and failover: a credential, quota or request error is not
retried on the same model, while rate limits, restarts and outages are.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import aiohttp
import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectOptions, APIStatusError

import livekit_slng as slng
from livekit_slng.gateway_adapter import extract_error_status

RETRY3 = APIConnectOptions(max_retry=3, retry_interval=0.01)
TWO = ["deepgram/aura:2", "cartesia/sonic:3.5"]


def envelope(code: str, **extra: object) -> dict[str, object]:
    """An error frame in the gateway's canonical format."""
    return {
        "type": "error",
        "code": code,
        "message": "refused",
        "slng_request_id": "gw-req-1",
        "session_closing": True,
        "recovery": "unknown",
        **extra,
    }


# --- the status an error frame stands for ------------------------------------

CANONICAL_CODES = {
    "provider_invalid_request": 400,
    "provider_credential_rejected": 401,
    "provider_quota_exceeded": 402,
    "provider_rate_limited": 429,
    "provider_unavailable": 503,
    "internal_error": 500,
    "processing_error": 500,
    "idle_timeout": 408,
    "session_duration_exceeded": 408,
    "service_restart": 503,
    "unsupported_watermark_format": 400,
    "watermarking_unavailable": 503,
    "watermarking_error": 500,
    # already known before the canonical format
    "config_error": 400,
    "payload_too_large": 413,
    "invalid_pronunciation": 400,
    "pronunciation_not_found": 404,
    "pronunciation_unauthenticated": 401,
    "rate_limit_exceeded": 429,
    "pronunciation_unavailable": 503,
    "provider_error": 502,
}


@pytest.mark.parametrize(("code", "status"), CANONICAL_CODES.items())
def test_every_canonical_code_carries_a_status(code, status):
    assert extract_error_status(envelope(code)) == status


@pytest.mark.parametrize("upstream", [401, 403, 413, 429])
def test_the_gateways_code_wins_over_the_providers_status(upstream):
    # The gateway keeps upstream_status on a provider_error it chose not to
    # classify further, a provider's 413 among them: that limit is the
    # provider's own, so the request still fails over to the next model.
    assert extract_error_status(envelope("provider_error", upstream_status=upstream)) == 502


def test_a_code_we_do_not_know_yet_follows_its_recovery():
    assert extract_error_status(envelope("a_future_code", recovery="fix_request")) == 400
    for recovery in ("backoff", "reconnect", "unknown"):
        assert extract_error_status(envelope("a_future_code", recovery=recovery)) is None


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "error", "code": "provider_error", "status_code": 400},
        {"type": "error", "code": "internal_error", "data": {"status_code": 400}},
    ],
)
def test_an_explicit_status_in_the_older_format_keeps_its_place(frame):
    assert extract_error_status(frame) == 400


@pytest.mark.parametrize("upstream", [0, 99, 600, 1011, "429", True, None, 4.29])
def test_an_upstream_status_that_is_not_one_is_ignored(upstream):
    assert extract_error_status(envelope("provider_error", upstream_status=upstream)) == 502


@pytest.mark.parametrize("provider_code", [401, 429, "401", "rate_limited"])
def test_the_providers_own_code_is_never_read(provider_code):
    frame = envelope("not_a_code_we_know", provider_code=provider_code)
    assert extract_error_status(frame) is None


@pytest.mark.parametrize(("code", "status"), [(429, 429), ("503", 503), (413, 413)])
def test_a_numeric_code_in_the_older_format_is_a_status(code, status):
    assert extract_error_status({"type": "error", "code": code, "message": "x"}) == status


@pytest.mark.parametrize("code", [1011, 4001, "1011", "4001", 0, -1, 99, 600])
def test_a_number_outside_the_http_range_is_not_a_status(code):
    assert extract_error_status({"type": "error", "code": code, "message": "x"}) is None


# --- TTS: what is retried on the same model -----------------------------------


class ErrorOnTextWs(FakeWs):
    """Answers every text frame with ``frame``."""

    def __init__(self, frame: dict[str, object]) -> None:
        super().__init__()
        self.frame = frame

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "text":
            self.push_json(self.frame)


NOT_RETRIED = [
    envelope("provider_credential_rejected", upstream_status=401, recovery="fix_request"),
    envelope("provider_quota_exceeded", upstream_status=402, recovery="fix_request"),
    envelope("provider_invalid_request", upstream_status=422, recovery="fix_request"),
    envelope("config_error", recovery="fix_request"),
    envelope("a_future_code", recovery="fix_request"),
]
RETRIED = [
    envelope("provider_error", upstream_status=413),
    envelope("provider_error", upstream_status=403),
    envelope("a_future_code", recovery="backoff"),
    envelope("provider_rate_limited", upstream_status=429, recovery="backoff"),
    envelope("provider_unavailable", upstream_status=503, recovery="backoff"),
    envelope("rate_limit_exceeded", recovery="backoff"),
    envelope("service_restart", recovery="reconnect"),
    envelope("idle_timeout", recovery="reconnect"),
    envelope("internal_error"),
    {"type": "error", "code": 1011, "message": "legacy close code as an error code"},
]


def make_two(primary_frame):
    t = slng.TTS(api_key="k", connections=TWO, voice="v", warm_standby_enabled=False)
    first, second = t._candidate_tts
    return (
        t,
        install_fake_session(first, lambda: ErrorOnTextWs(primary_frame)),
        install_fake_session(second, FakeWs),
    )


@pytest.mark.parametrize("frame", NOT_RETRIED, ids=lambda f: str(f["code"]))
async def test_tts_fails_over_without_retrying_a_request_error(frame):
    t, primary, fallback = make_two(frame)
    assert await drive_reply(t, sample_deltas(), conn_options=RETRY3) > 0
    assert len(primary.created) == 1, "retried on the model that refused it"
    assert fallback.created and t._active_candidate_index == 1
    await t.aclose()


@pytest.mark.parametrize("frame", RETRIED, ids=lambda f: str(f["code"]))
async def test_tts_retries_a_transient_error_before_failing_over(frame):
    t, primary, fallback = make_two(frame)
    assert await drive_reply(t, sample_deltas(), conn_options=RETRY3) > 0
    assert len(primary.created) == 1 + RETRY3.max_retry
    assert fallback.created
    await t.aclose()


async def test_tts_raises_the_status_of_a_credential_error():
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=False)
    session = install_fake_session(t, lambda: ErrorOnTextWs(NOT_RETRIED[0]))
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert raised.value.status_code == 401
    assert len(session.created) == 1
    with pytest.raises(APIStatusError) as raised:
        await t.synthesize("Hello there.", conn_options=RETRY3).collect()
    assert raised.value.status_code == 401
    assert len(session.created) == 2, "synthesize retried it too"
    await t.aclose()


async def test_a_websocket_code_in_an_error_frame_is_not_reported_as_a_status():
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=False)
    install_fake_session(t, lambda: ErrorOnTextWs({"type": "error", "code": 1011}))
    with pytest.raises(APIStatusError) as raised:
        await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0))
    assert raised.value.status_code == -1
    await t.aclose()


# --- STT: what is retried on the same endpoint ---------------------------------

STT_TWO = ["deepgram/nova:3", "soniox/stt-rt:v3"]
AUDIO = b"\x00\x01" * 1600


class SttWs:
    """Answers init with ``answer``; a working one answers finalize and close too."""

    def __init__(self, answer: dict[str, object]) -> None:
        self.answer = answer
        self.sent: list[dict] = []
        self._q: asyncio.Queue = asyncio.Queue()

    def _put(self, payload: dict) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)))

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent.append(frame)
        if frame["type"] == "init":
            self._put(self.answer)
        elif frame["type"] == "finalize":
            self._put({"type": "final_transcript", "transcript": "hello", "confidence": 1.0})
        elif frame["type"] == "close":
            self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def send_bytes(self, payload: bytes) -> None:
        pass

    async def receive(self, timeout=None):
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


class SttSession:
    """Opens an ``SttWs`` per connect, answering init with the endpoint's frame."""

    def __init__(self, answers: dict[str, dict[str, object]]) -> None:
        self.answers = answers
        self.closed = False
        self.connects: list[str] = []

    async def ws_connect(self, url: str, **kw):
        self.connects.append(url)
        model = next(m for m in self.answers if url.endswith(m))
        return SttWs(self.answers[model])


def stt_stream(answers, *, max_retry=2):
    session = SttSession(answers)
    stt_obj = slng.STT(
        api_key="k",
        connections=list(answers),
        http_session=session,  # type: ignore[arg-type]
    )
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=max_retry))
    for _ in range(5):
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))
    return stream, session


async def run_until_error(stream):
    async def read():
        async for _ in stream:
            pass

    with pytest.raises(APIStatusError) as raised:
        await asyncio.wait_for(read(), 3)
    await stream.aclose()
    return raised.value


@pytest.mark.parametrize("frame", NOT_RETRIED, ids=lambda f: str(f["code"]))
async def test_stt_does_not_retry_a_request_error(frame):
    stream, session = stt_stream({"deepgram/nova:3": frame})
    exc = await run_until_error(stream)
    assert 400 <= exc.status_code < 500
    assert len(session.connects) == 1, "retried on the endpoint that refused it"


@pytest.mark.parametrize("frame", RETRIED, ids=lambda f: str(f["code"]))
async def test_stt_retries_a_transient_error(frame):
    stream, session = stt_stream({"deepgram/nova:3": frame})
    await run_until_error(stream)
    assert len(session.connects) == 3, "one attempt and two retries"


async def test_stt_fails_over_on_a_credential_error_without_retrying():
    stream, session = stt_stream(
        {
            "deepgram/nova:3": NOT_RETRIED[0],
            "soniox/stt-rt:v3": {"type": "ready", "slng_request_id": "gw-req-2"},
        }
    )

    async def read():
        async for _ in stream:
            pass

    reader = asyncio.create_task(read())
    while len(session.connects) < 2:
        await asyncio.sleep(0.01)
    stream.end_input()
    await asyncio.wait_for(reader, 3)
    assert [url.rsplit("/", 2)[-2] for url in session.connects] == ["deepgram", "soniox"]


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
