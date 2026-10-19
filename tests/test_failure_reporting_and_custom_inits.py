"""Failure reporting, replies with nothing to say, and custom init messages.

A failed reply reaches the session as one unrecoverable error and a reply with nothing to say
is not a failure; custom TTS and STT inits keep their fields but take the instance's settings.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import aiohttp
import pytest
from fakews import (
    FakeWs,
    ListSentenceTokenizer,
    _msg,
    drive_reply,
    install_fake_session,
    sample_deltas,
)
from livekit.agents import APIConnectionError, APIConnectOptions, APIError
from livekit.agents.stt import SpeechEventType

import livekit_slng as slng

RETRY3 = APIConnectOptions(max_retry=3, retry_interval=0.01)
TWO = ["deepgram/aura:2", "cartesia/sonic:3.5"]


def make(**kw):
    kw.setdefault("warm_standby_enabled", False)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def make_two(primary=FakeWs, fallback=FakeWs, **kw):
    kw.setdefault("warm_standby_enabled", False)
    t = slng.TTS(api_key="k", connections=TWO, voice="v", **kw)
    first, second = t._candidate_tts
    return t, install_fake_session(first, primary), install_fake_session(second, fallback)


def record_errors(t):
    events = []
    t.on("error", lambda e: events.append(e))
    return events


def unrecoverable(events):
    return [e for e in events if not e.recoverable]


def sent_texts(*sessions):
    return [
        m for s in sessions for ws in s.created for m in ws.sent if m["type"] in ("text", "flush")
    ]


class ProviderErrorWs(FakeWs):
    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "text":
            self.push_json({"type": "error", "code": "provider_error", "message": "upstream 503"})


class TooLargeWs(FakeWs):
    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.sent[-1]["type"] == "text":
            self.push_json({"type": "error", "code": 413, "message": "too large"})


class ResetAfterAudioWs(FakeWs):
    async def _stream_reply(self) -> None:
        self.push_audio()
        self.push_audio()
        await asyncio.sleep(0.02)
        self.kill()
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.CLOSED))


class NoAudioWs(FakeWs):
    """Ends every reply with audio_end and no audio at all."""

    def __init__(self, **kw):
        super().__init__(audio_chunks_per_reply=0, **kw)


# --- one failed reply, one unrecoverable error -----------------------------


async def test_a_reply_failing_on_every_attempt_reports_one_unrecoverable_error():
    t = make()
    install_fake_session(t, ProviderErrorWs)
    events = record_errors(t)
    with pytest.raises(APIError):
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert len(unrecoverable(events)) == 1, [e.recoverable for e in events]
    assert len(events) > 1, "the retried attempts are still reported, as recoverable"
    await t.aclose()


async def test_a_reply_saved_by_the_fallback_reports_no_unrecoverable_error():
    t, _, fallback = make_two(primary=ProviderErrorWs)
    events = record_errors(t)
    assert await drive_reply(t, sample_deltas(), conn_options=RETRY3) > 0
    assert unrecoverable(events) == []
    assert fallback.created, "the fallback model spoke the reply"
    await t.aclose()


async def test_every_model_failing_reports_one_unrecoverable_error():
    t, _, _ = make_two(primary=ProviderErrorWs, fallback=ProviderErrorWs)
    events = record_errors(t)
    with pytest.raises(APIError):
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert len(unrecoverable(events)) == 1
    await t.aclose()


async def test_a_connection_lost_after_audio_reports_one_unrecoverable_error():
    t = make()
    install_fake_session(t, ResetAfterAudioWs)
    events = record_errors(t)
    with pytest.raises(APIConnectionError):
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert len(unrecoverable(events)) == 1
    await t.aclose()


async def test_a_payload_too_large_reports_one_unrecoverable_error():
    t, _, fallback = make_two(primary=TooLargeWs)
    events = record_errors(t)
    with pytest.raises(APIError):
        await drive_reply(t, sample_deltas(), conn_options=RETRY3)
    assert len(unrecoverable(events)) == 1
    assert not fallback.created, "413 does not walk the chain"
    await t.aclose()


async def test_a_failed_synthesize_reports_one_unrecoverable_error():
    t = make()
    install_fake_session(t, ProviderErrorWs)
    events = record_errors(t)
    with pytest.raises(APIError):
        await t.synthesize("Hello there.", conn_options=RETRY3).collect()
    assert len(unrecoverable(events)) == 1
    await t.aclose()


async def test_a_synthesize_saved_by_the_fallback_reports_no_unrecoverable_error():
    t, _, _ = make_two(primary=ProviderErrorWs)
    events = record_errors(t)
    frame = await t.synthesize("Hello there.", conn_options=RETRY3).collect()
    assert frame.samples_per_channel > 0
    assert unrecoverable(events) == []
    await t.aclose()


async def test_an_interrupted_reply_reports_no_error():
    t = make()
    install_fake_session(t, lambda: FakeWs(audio_chunks_per_reply=30, chunk_interval_s=0.01))
    events = record_errors(t)
    assert await drive_reply(t, sample_deltas(), interrupt_after_events=2) > 0
    await asyncio.sleep(0.05)
    assert events == []
    await t.aclose()


# --- a reply with nothing to say -------------------------------------------

NOTHING_TO_SAY = ["\n\n", "   ", "...", "?!", "…", " — ", "\U0001f44d"]


@pytest.mark.parametrize("text", NOTHING_TO_SAY)
async def test_a_stream_with_nothing_to_say_ends_quietly(text):
    t, first, second = make_two()
    events = record_errors(t)
    assert await drive_reply(t, [text], conn_options=RETRY3) == 0
    assert not first.created and not second.created, "nothing to say opened a connection"
    assert events == [] and t._active_candidate_index == 0
    await t.aclose()


@pytest.mark.parametrize("text", NOTHING_TO_SAY)
async def test_a_synthesize_with_nothing_to_say_ends_quietly(text):
    t, first, second = make_two()
    events = record_errors(t)
    frames = [ev async for ev in t.synthesize(text, conn_options=RETRY3)]
    assert frames == []
    assert not first.created and not second.created
    assert events == [] and t._active_candidate_index == 0
    await t.aclose()


async def test_a_held_socket_sends_nothing_for_a_reply_with_nothing_to_say():
    t = make(warm_standby_enabled=True)
    session = install_fake_session(t)
    t.prewarm()
    await asyncio.sleep(0.05)
    events = record_errors(t)
    assert await drive_reply(t, ["\n", "\n"]) == 0
    assert sent_texts(session) == [] and events == []
    assert await drive_reply(t, sample_deltas()) > 0, "the next reply still speaks"
    assert len(session.created) == 1
    await t.aclose()


async def test_leading_punctuation_goes_out_with_the_speech_after_it():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["(", "423", ")"]) > 0
    assert [m.get("text") for m in sent_texts(session)] == ["(423) ", None]
    await t.aclose()


async def test_leading_whitespace_goes_out_with_the_speech_after_it():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["\n\n", "Hello there."]) > 0
    texts = [m["text"] for m in sent_texts(session) if m["type"] == "text"]
    assert "".join(texts).strip() == "Hello there."
    await t.aclose()


async def test_a_number_alone_is_still_spoken():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["4200."]) > 0
    assert [m.get("text") for m in sent_texts(session)] == ["4200. ", None]
    await t.aclose()


async def test_trailing_punctuation_is_not_sent_as_a_frame_of_its_own():
    # One sentence per pushed string, so "..." reaches the batcher on its own.
    # slng.SentenceTokenizer keeps "Hello there. ..." as one sentence anyway.
    t = make(word_tokenizer=ListSentenceTokenizer(), text_chunking="sentence")
    session = install_fake_session(t)
    assert await drive_reply(t, ["Hello there. ", "..."]) > 0
    sent = sent_texts(session)
    assert [m["text"] for m in sent if m["type"] == "text"] == ["Hello there. "]
    assert [m["type"] for m in sent].count("flush") == 1
    await t.aclose()


async def test_a_reply_with_text_and_no_audio_still_fails_over():
    t, _, fallback = make_two(primary=NoAudioWs)
    assert await drive_reply(t, sample_deltas(), conn_options=RETRY3) > 0
    assert t._active_candidate_index == 1 and fallback.created
    await t.aclose()


# --- a custom TTS init --------------------------------------------------------


def _custom_tts_init():
    return {
        "type": "init",
        "model": "deepgram/aura:2",
        "voice": "old-voice",
        "language": "old-language",
        "custom_top_level": "keep",
        "config": {"speed": 0.5, "language": "old-language", "custom_config": "keep"},
    }


async def test_update_options_applies_over_a_custom_tts_init():
    t = make(runtime_init=_custom_tts_init(), warm_standby_enabled=True)
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    t.update_options(voice="new-voice", language="es", speed=1.25)
    assert await drive_reply(t, sample_deltas()) > 0
    init = [m for ws in session.created for m in ws.sent if m["type"] == "init"][-1]
    assert init["voice"] == "new-voice"
    assert init["language"] == "es" and init["config"]["language"] == "es"
    assert init["config"]["speed"] == 1.25
    assert init["custom_top_level"] == "keep" and init["config"]["custom_config"] == "keep"
    await t.aclose()


async def test_a_custom_tts_init_keeps_what_the_instance_leaves_unset():
    t = make(runtime_init=_custom_tts_init())
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    (init,) = [m for ws in session.created for m in ws.sent if m["type"] == "init"]
    assert init["language"] == "old-language" and init["config"]["language"] == "old-language"
    assert init["config"]["speed"] == 0.5
    assert init["voice"] == "v", "the instance's voice wins"
    assert init["config"]["encoding"] == "linear16", "the plugin decodes linear16 itself"
    assert init["config"]["sample_rate"] == t.sample_rate
    await t.aclose()


async def test_a_connection_init_is_merged_too():
    config = slng.TTSConnectionConfig(
        endpoint="wss://us-east.api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2",
        init=_custom_tts_init(),
    )
    t = slng.TTS(api_key="k", connections=[config], voice="v", warm_standby_enabled=False)
    session = install_fake_session(t)
    t.update_options(voice="new-voice")
    assert await drive_reply(t, sample_deltas()) > 0
    (init,) = [m for ws in session.created for m in ws.sent if m["type"] == "init"]
    assert init["voice"] == "new-voice" and init["custom_top_level"] == "keep"
    await t.aclose()


# --- STT ---------------------------------------------------------------


async def _wait_until(predicate, timeout=2.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


class _InitWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_str(self, msg: str) -> None:
        self.sent.append(json.loads(msg))

    async def receive(self, timeout=None):
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps({"type": "ready"}))

    async def close(self) -> None:
        pass


class _InitSession:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.closed = False

    async def ws_connect(self, url, **kw):
        return self.ws


STT_ENDPOINT = "wss://us-east.api.slng.ai/v1/bridges/unmute/stt/deepgram/nova:3"


def _stt_with_init(ws, init):
    return slng.STT(
        api_key="k",
        connections=[slng.STTConnectionConfig(endpoint=STT_ENDPOINT, init=init)],
        http_session=_InitSession(ws),  # type: ignore[arg-type]
    )


async def _stt_init_sent(stt_obj, **stream_kw):
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0), **stream_kw)
    return stream


CUSTOM_STT_INIT = {
    "type": "init",
    "custom_top_level": "keep",
    "config": {"language": "hi", "enable_partials": False, "custom_config": "keep"},
}


async def test_a_custom_stt_init_is_sent_as_given_without_updates():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, CUSTOM_STT_INIT)
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    assert ws.sent[0] == CUSTOM_STT_INIT, "the plugin's defaults must not replace the init"


async def test_update_options_applies_over_a_custom_stt_init():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, CUSTOM_STT_INIT)
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.update_options(language="id", enable_partial_transcripts=True)
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    sent = ws.sent[0]
    assert sent["config"]["language"] == "id" and sent["config"]["enable_partials"] is True
    assert sent["custom_top_level"] == "keep" and sent["config"]["custom_config"] == "keep"
    assert "vad_threshold" not in sent["config"], "only what was updated is applied"


async def test_an_stt_level_update_reaches_later_streams_over_a_custom_init():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, CUSTOM_STT_INIT)
    stt_obj.update_options(language="ta")
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    assert ws.sent[0]["config"]["language"] == "ta"


async def test_a_stream_language_applies_over_a_custom_stt_init():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, CUSTOM_STT_INIT)
    stream = stt_obj.stream(language="bn", conn_options=APIConnectOptions(max_retry=0))
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    assert ws.sent[0]["config"]["language"] == "bn"


class _SilentSttWs:
    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_payloads: list[bytes] = []
        self._q: asyncio.Queue = asyncio.Queue()
        self.closed = asyncio.Event()

    async def send_str(self, msg: str) -> None:
        self.sent_texts.append(json.loads(msg))

    async def send_bytes(self, payload: bytes) -> None:
        self.audio_payloads.append(payload)

    async def receive(self, timeout=None):
        return await self._q.get()

    async def close(self) -> None:
        self.closed.set()
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


class _DyingSttWs:
    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_payloads: list[bytes] = []
        self._die = asyncio.Event()

    async def send_str(self, msg: str) -> None:
        self.sent_texts.append(json.loads(msg))

    async def send_bytes(self, payload: bytes) -> None:
        self.audio_payloads.append(payload)
        self._die.set()

    async def receive(self, timeout=None):
        await self._die.wait()
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)

    async def close(self) -> None:
        self._die.set()


class _AnsweringSttWs:
    def __init__(self) -> None:
        self.sent_texts: list[dict] = []
        self.audio_payloads: list[bytes] = []
        self.audio_received = asyncio.Event()
        self._q: asyncio.Queue = asyncio.Queue()

    def _put(self, payload: dict) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)))

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent_texts.append(frame)
        if frame.get("type") == "finalize":
            self._put({"type": "final_transcript", "transcript": "hello world", "confidence": 1.0})
        if frame.get("type") == "close":
            self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def send_bytes(self, payload: bytes) -> None:
        self.audio_payloads.append(payload)
        self.audio_received.set()

    async def receive(self, timeout=None):
        return await self._q.get()

    async def close(self) -> None:
        self._q.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))


async def test_an_options_change_reconnect_drops_the_audio_it_abandoned(monkeypatch):
    abandoned = b"\x01\x02" * 1024
    active = b"\x03\x04" * 1024
    ws1, ws2, ws3 = _SilentSttWs(), _DyingSttWs(), _AnsweringSttWs()
    sockets: list = [ws1, ws2, ws3]

    async def fake_connect(self, *, model_endpoint, model):
        return sockets.pop(0)

    monkeypatch.setattr(slng.stt.SpeechStream, "_connect_ws", fake_connect)
    async with aiohttp.ClientSession() as session:
        stt_obj = slng.STT(api_key="k", model="deepgram/nova:3", http_session=session)
        stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=2))
        got_final = asyncio.Event()

        async def read():
            async for ev in stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    got_final.set()

        reader = asyncio.create_task(read())
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(abandoned)))
        await _wait_until(lambda: bool(ws1.audio_payloads))
        stream.update_options(language="id")
        await asyncio.wait_for(ws1.closed.wait(), 3)
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(active)))
        await asyncio.wait_for(ws3.audio_received.wait(), 3)
        stt_obj.notify_user_state("listening")
        await asyncio.wait_for(got_final.wait(), 3)
        stream.end_input()
        await asyncio.wait_for(reader, 3)

    assert b"".join(ws2.audio_payloads) == active
    assert b"".join(ws3.audio_payloads) == active, "the replay carried audio from before the change"


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
