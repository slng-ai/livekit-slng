"""Log privacy, recovery after a failed reply, and assorted tokenizer and init cases.

Logs carry no reply text and a failed reply's held socket reopens in the background; the
module also covers the first-audio deadline, titles after list numbers and custom STT inits.
"""

import asyncio
import logging
import re

import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectOptions, APIError
from test_failure_reporting_and_custom_inits import (
    CUSTOM_STT_INIT,
    STT_ENDPOINT,
    NoAudioWs,
    ProviderErrorWs,
    _InitWs,
    _stt_with_init,
    _wait_until,
)

import livekit_slng as slng

TOK = slng.SentenceTokenizer()


def make(**kw):
    kw.setdefault("warm_standby_enabled", False)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


async def streamed(text: str) -> tuple[list[str], int]:
    stream = TOK.stream()
    out: list[str] = []

    async def read() -> None:
        async for token in stream:
            out.append(token.token)

    reader = asyncio.create_task(read())
    for push in re.findall(r"\s*\S+|\s+", text):
        stream.push_text(push)
        await asyncio.sleep(0)
    before = len(out)
    stream.end_input()
    await reader
    return out, before


# --- the tokenizer ------------------------------------------------------

TOKENIZER_CASES = {
    # a title after a list number, an opening quote or a bracket stays with its name
    "Here are your options:\n1. Dr. Smith on Monday.\n2. Dr. Patel on Tuesday.": [
        "Here are your options:",
        "1. Dr. Smith on Monday.",
        "2. Dr. Patel on Tuesday.",
    ],
    'He left. "Dr. Smith is here," she said.': ["He left.", '"Dr. Smith is here," she said.'],
    "He left. (Dr. Smith stayed.) Bye.": ["He left.", "(Dr. Smith stayed.)", "Bye."],
    "Order status\nDr. Smith is here. Bye.": ["Order status", "Dr. Smith is here.", "Bye."],
    "1. No. I don't think so.": ["1. No.", "I don't think so."],
    # a short sentence ending in "I", and a stop inside a CJK word
    "So do I. Then we left.": ["So do I.", "Then we left."],
    "Neither can I. Maybe later.": ["Neither can I.", "Maybe later."],
    "文件名是报告.pdf，请查收。": ["文件名是报告.pdf，请查收。"],
    "请访问百度.com 查看。谢谢。": ["请访问百度.com 查看。", "谢谢。"],
    "你好.我是小明.很高兴认识你.": ["你好.", "我是小明.", "很高兴认识你."],
    "Sie können z. B. per Karte zahlen. Das geht schnell.": [
        "Sie können z. B. per Karte zahlen.",
        "Das geht schnell.",
    ],
    # a letterless last piece rides with the sentence before it
    "Your code is below.\n4200": ["Your code is below. 4200"],
    "Call us at\n555-0100.": ["Call us at 555-0100."],
    "Totals:\n4200.\n5300.": ["Totals: 4200. 5300."],
    "Your code is below.\n4200\nAnything else?": ["Your code is below.", "4200", "Anything else?"],
    "4200.": ["4200."],
    "Sure. Let me check.": ["Sure.", "Let me check."],
}


@pytest.mark.parametrize(("text", "expected"), TOKENIZER_CASES.items())
async def test_tokenizer_cases_batch_and_streamed(text, expected):
    assert TOK.tokenize(text) == expected
    out, before = await streamed(text)
    assert out == expected
    assert before == len(expected) - 1


async def test_the_sentence_before_a_letterless_tail_waits_for_a_letter():
    stream = TOK.stream()
    out: list[str] = []

    async def read() -> None:
        async for token in stream:
            out.append(token.token)

    reader = asyncio.create_task(read())
    stream.push_text("Your code is below.\n4200")
    await asyncio.sleep(0)
    assert out == [], "held while what follows it has no letter"
    stream.push_text("\nAnything")
    await asyncio.sleep(0)
    assert out == ["Your code is below.", "4200"]
    stream.end_input()
    await reader
    assert out == ["Your code is below.", "4200", "Anything"]


def test_many_list_numbers_on_one_line_stay_linear():
    import time

    text = "Item 1. and item 2. and " * 3000
    started = time.perf_counter()
    TOK.tokenize(text)
    assert time.perf_counter() - started < 2.0


# --- letterless tails end to end: the frames on the wire ------------------------


async def test_a_letterless_last_line_is_sent_with_the_sentence_before_it():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["Your code", " is below.", "\n", "4200"]) > 0
    ws = session.created[0]
    assert [m["text"] for m in ws.texts()] == ["Your code is below. 4200 "]
    assert ws.count("flush") == 1
    await t.aclose()


# --- the first-audio deadline ----------------------------------------------------


async def test_whitespace_before_reading_does_not_start_the_deadline():
    t = make(first_audio_timeout_s=0.3)
    install_fake_session(t)
    errors: list = []
    t.on("error", errors.append)
    stream = t.stream()
    stream.push_text("\n\n")

    async def consume():
        return [ev async for ev in stream]

    reader = asyncio.create_task(consume())
    await asyncio.sleep(0.6)
    stream.end_input()
    assert await reader == []
    assert errors == []
    await t.aclose()


async def test_the_deadline_starts_with_the_first_letter_or_digit():
    t = make(first_audio_timeout_s=5.0)
    install_fake_session(t)
    stream = t.stream()
    await stream.__aenter__()
    stream.push_text("\n\n")
    assert stream._deadline is None
    stream.push_text("4")
    assert stream._deadline is not None
    stream.end_input()
    async for _ in stream:
        pass
    await t.aclose()


# --- collect() on text with nothing to say --------------------------------------


async def test_synthesize_of_nothing_to_say_collects_an_empty_frame():
    t = make()
    install_fake_session(t)
    frame = await t.synthesize("...").collect()
    assert frame.samples_per_channel == 0
    assert frame.sample_rate == t.sample_rate and frame.num_channels == t.num_channels
    await t.aclose()


# --- no reply text in the logs --------------------------------------------

SECRET = "my account number is secretive"


def _leaks(records, text):
    return [
        r
        for r in records
        if text in r.getMessage()
        or (r.exc_info and text in logging.Formatter().formatException(r.exc_info))
    ]


async def test_a_failed_attempt_logs_no_reply_text(caplog):
    t = make()
    install_fake_session(t, NoAudioWs)
    with caplog.at_level(logging.DEBUG, logger="livekit_slng"):
        with pytest.raises(APIError):
            await drive_reply(
                t,
                [f"Sure, {SECRET}."],
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01),
            )
    records = [r for r in caplog.records if r.name.startswith("livekit_slng")]
    assert not _leaks(records, "secretive"), [r.getMessage() for r in _leaks(records, "secretive")]
    failures = [
        r
        for r in records
        if "attempt failed" in r.getMessage() or "every candidate" in r.getMessage()
    ]
    assert failures, "the failure is still logged"
    assert all(r.exc_info is None for r in failures)
    assert any("secretive" in getattr(r, "lk.pii.error", "") for r in failures)
    assert all(getattr(r, "error_type", None) for r in failures)
    await t.aclose()


async def test_a_successful_reply_logs_no_reply_text(caplog):
    t = make()
    install_fake_session(t)
    with caplog.at_level(logging.DEBUG, logger="livekit_slng"):
        assert await drive_reply(t, [f"Sure, {SECRET}. Anything else?"]) > 0
    records = [r for r in caplog.records if r.name.startswith("livekit_slng")]
    assert not _leaks(records, "secretive")
    await t.aclose()


# --- encoding as a model option --------------------------------------------------


async def test_encoding_linear16_as_a_model_option_is_accepted():
    t = make(encoding="linear16")
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    (init,) = [m for ws in session.created for m in ws.sent if m["type"] == "init"]
    assert init["config"]["encoding"] == "linear16"
    await t.aclose()


def test_another_encoding_as_a_model_option_raises():
    with pytest.raises(ValueError, match="linear16"):
        make(encoding="mp3")


# --- control frames on an idle socket --------------------------------------------


async def test_control_frames_on_an_idle_socket_are_not_stray():
    t = make(warm_standby_enabled=True)
    session = install_fake_session(t)
    events: list = []
    t.on("slng_event", events.append)
    assert await drive_reply(t, sample_deltas()) > 0
    ws = session.created[0]
    for frame in (
        {"type": "Metadata"},
        {"type": "cleared"},
        {"type": "audio_end"},
        {"type": "control_ack"},
    ):
        ws.push_json(frame)
    await asyncio.sleep(0.05)
    assert not [e for e in events if e.name == "tts.idle_stray_frame"]
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1, "the socket was kept"
    await t.aclose()


async def test_json_audio_on_an_idle_socket_still_retires_it():
    t = make(warm_standby_enabled=True)
    session = install_fake_session(t)
    events: list = []
    t.on("slng_event", events.append)
    assert await drive_reply(t, sample_deltas()) > 0
    ws = session.created[0]
    ws.binary_audio = False
    ws.push_audio()
    await _wait_until(lambda: ws.closed)
    assert [e for e in events if e.name == "tts.idle_stray_frame"]
    await t.aclose()


# --- reopen after a failed reply -------------------------------------------------


async def test_a_failed_reply_reopens_the_held_socket_in_the_background():
    t = make(warm_standby_enabled=True)
    install_fake_session(t, ProviderErrorWs)
    with pytest.raises(APIError):
        await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0))
    await _wait_until(lambda: t._held is not None)
    assert not t._held.ws.closed
    await t.aclose()


class _FailOnceSession:
    """First socket answers text with an error, every later one works."""

    def __init__(self) -> None:
        self.closed = False
        self.created: list[FakeWs] = []

    async def ws_connect(self, url, **kw):
        ws = ProviderErrorWs() if not self.created else FakeWs()
        self.created.append(ws)
        return ws


async def test_a_retry_takes_the_reopened_socket_rather_than_opening_another():
    t = make(warm_standby_enabled=True)
    session = _FailOnceSession()
    t._session = session
    assert (
        await drive_reply(
            t, sample_deltas(), conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01)
        )
        > 0
    )
    await asyncio.sleep(0.05)
    assert len(session.created) == 2, "one failed socket and one replacement, no third"
    await t.aclose()


async def test_a_switch_to_the_fallback_leaves_the_primary_with_no_socket():
    t = slng.TTS(
        api_key="k",
        connections=["deepgram/aura:2", "cartesia/sonic:3.5"],
        voice="v",
        warm_standby_enabled=True,
    )
    first, second = t._candidate_tts
    install_fake_session(first, ProviderErrorWs)
    install_fake_session(second, FakeWs)
    assert await drive_reply(t, sample_deltas(), conn_options=APIConnectOptions(max_retry=0)) > 0
    await asyncio.sleep(0.1)
    assert first._held is None and (first._connect_task is None or first._connect_task.done())
    await t.aclose()


# --- a custom STT init ----------------------------------------------------------


async def test_a_custom_stt_init_without_config_is_sent_exactly_as_written():
    ws = _InitWs()
    init = {"type": "init", "language": "en"}
    stt_obj = _stt_with_init(ws, init)
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    assert ws.sent[0] == init


async def test_an_update_reaches_a_top_level_copy_in_a_custom_stt_init():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, {"type": "init", "language": "en", "custom": "keep"})
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.update_options(language="es")
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    sent = ws.sent[0]
    assert sent["language"] == "es" and sent["config"]["language"] == "es"
    assert sent["custom"] == "keep"
    assert set(sent["config"]) == {"language"}, "only what was updated is added"


async def test_an_stt_init_with_config_is_still_sent_as_written():
    ws = _InitWs()
    stt_obj = _stt_with_init(ws, CUSTOM_STT_INIT)
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream._connect_ws(model_endpoint=STT_ENDPOINT, model="deepgram/nova:3")
    await stream.aclose()
    assert ws.sent[0] == CUSTOM_STT_INIT


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
