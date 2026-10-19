"""The held socket's lifecycle, and the defaults around it.

Covers warm standby off, idle and age limits, stalled prewarms, a closed HTTP session and
pings, plus the "auto" chunking default, openings in CJK text and blingfire failure logs.
"""

import asyncio
import logging
import time

import pytest
from fakews import FakeWs, drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIError, tokenize

import livekit_slng as slng
from livekit_slng import sentence_tokenizer as st

LONG_OPENING = (
    "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami "
    "dan statusnya sedang dalam proses pengiriman. Terima kasih."
)
CJK = "我们需要非常仔细地考虑所有可能出现的问题和各种细节，然后再根据实际情况做出最终的决定。"


def make(**kw):
    kw.setdefault("warm_standby_enabled", True)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def words(text: str) -> list[str]:
    return [w + " " for w in text.split(" ")]


def chars(text: str, n: int = 3) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


async def warm(t) -> None:
    t.prewarm()
    await asyncio.sleep(0.05)


# --- "auto" with a WordTokenizer -------------------------------------


def test_auto_with_a_word_tokenizer_is_phrase_batching():
    t = make(word_tokenizer=tokenize.basic.WordTokenizer())
    assert t._opts.text_chunking == "phrase"
    assert make()._opts.text_chunking == "sentence"
    assert make(word_tokenizer=slng.SentenceTokenizer())._opts.text_chunking == "sentence"
    with pytest.raises(ValueError, match="SentenceTokenizer"):
        make(text_chunking="sentence", word_tokenizer=tokenize.basic.WordTokenizer())


async def test_auto_with_a_word_tokenizer_still_speaks():
    t = make(word_tokenizer=tokenize.basic.WordTokenizer(ignore_punctuation=False))
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created[0].texts()) > 1  # phrase-sized frames
    await t.aclose()


# --- warm standby off means one socket per reply ---------------------


async def test_standby_off_holds_nothing_between_replies():
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t)
    for _ in range(3):
        assert await drive_reply(t, sample_deltas()) > 0
        assert t._held is None
    await asyncio.sleep(0.05)
    assert len(session.created) == 3 and all(ws.closed for ws in session.created)
    assert all(ws.count("keepalive") == 0 for ws in session.created)
    await t.aclose()


async def test_standby_off_interrupt_closes_without_a_drain():
    t = make(warm_standby_enabled=False)
    session = install_fake_session(
        t, lambda: FakeWs(audio_chunks_per_reply=40, chunk_interval_s=0.01)
    )
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    await asyncio.sleep(0.05)
    ws = session.created[0]
    assert ws.closed and ws.count("clear") == 0
    assert t._settle is None
    await t.aclose()


# --- an end of reply before the flush --------------------------------


class EarlyEndWs(FakeWs):
    """Ends the turn after the first text frame, as when the provider closes cleanly."""

    async def send_str(self, data: str) -> None:
        await super().send_str(data)
        if self.texts() and len(self.texts()) == 1 and self.sent[-1]["type"] == "text":
            self.push_audio()
            self.push_json({"type": "audio_end"})


async def test_an_end_before_the_flush_closes_the_socket(caplog):
    t = make()
    t._schedule_connect = lambda **_: None
    session = install_fake_session(t, EarlyEndWs)
    stream = t.stream()
    total = 0

    async def drain():
        nonlocal total
        async for ev in stream:
            total += len(ev.frame.data)

    reader = asyncio.create_task(drain())
    stream.push_text("First sentence here. Second one follows. ")
    await asyncio.sleep(0.1)  # the first frame is out and its turn has "ended"
    stream.push_text("Third sentence at the end.")
    stream.end_input()
    # The rest of the reply was never voiced, so the reply fails rather than
    # passing as complete.
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        with pytest.raises(APIError, match="before all of its text was sent"):
            await asyncio.wait_for(reader, 5.0)
    await stream.aclose()
    assert total > 0, "the audio that did arrive is still played"
    ws = session.created[0]
    assert ws.closed and t._held is None, "a socket with an unread turn was kept"
    assert any("before its flush" in r.message for r in caplog.records)
    await t.aclose()


async def test_a_normal_reply_is_not_flagged(caplog):
    t = make()
    session = install_fake_session(t)
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        assert await drive_reply(t, sample_deltas()) > 0
    assert t._held is not None and t._held.ws is session.created[0]
    assert not any("before its flush" in r.message for r in caplog.records)
    await t.aclose()


# --- the public tokenizer hands out whole sentences by default -------


async def test_stream_default_emits_no_partial_opening():
    tok = slng.SentenceTokenizer()
    stream = tok.stream()
    for w in words(LONG_OPENING):
        stream.push_text(w)
    stream.end_input()
    got = [ev async for ev in stream]
    assert not any(st.is_partial_head(ev) for ev in got)
    assert [ev.token for ev in got] == tok.tokenize(LONG_OPENING)


async def test_stream_adapter_gets_whole_sentences():
    """What a caller wrapping a non-streaming TTS with this tokenizer would see."""
    tok = slng.SentenceTokenizer()
    stream = tok.stream()  # exactly what tts.StreamAdapter calls
    for w in words(LONG_OPENING):
        stream.push_text(w)
    stream.end_input()
    tokens = [ev.token async for ev in stream]
    assert tokens[0].endswith("pengiriman."), tokens


# --- idle and age limits --------------------------------------


async def test_an_idle_socket_is_retired_and_not_reopened(monkeypatch):
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    monkeypatch.setattr(slng.tts, "_MAX_IDLE_S", 0.2)
    t = make()
    session = install_fake_session(t)
    await warm(t)
    first = session.created[0]
    await asyncio.sleep(0.5)
    assert first.closed and first.sent[-1] == {"type": "close"}
    assert t._held is None and len(session.created) == 1, "an idle instance reopened"
    # a later reply still works, and holds a socket again
    assert await drive_reply(t, sample_deltas()) > 0
    assert t._held is not None and len(session.created) == 2
    await t.aclose()


async def test_a_reply_restarts_the_idle_clock(monkeypatch):
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    monkeypatch.setattr(slng.tts, "_MAX_IDLE_S", 0.3)
    t = make()
    session = install_fake_session(t)
    await warm(t)
    for _ in range(3):
        await asyncio.sleep(0.2)
        assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1 and not session.created[0].closed
    await t.aclose()


async def test_an_old_socket_is_replaced_in_the_background(monkeypatch):
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    t = make()
    install_fake_session(t)
    await warm(t)
    first = t._held
    first.opened_at -= slng.tts._MAX_CONNECTION_AGE_S  # it has been open too long
    await asyncio.sleep(0.2)
    assert first.ws.closed
    assert t._held is not None and t._held is not first and t._held.opened_by == "max_age"
    assert t._failed_reopens == 0, "a planned replacement is not a failure"
    await t.aclose()


async def test_an_old_socket_is_retired_after_the_reply_that_used_it():
    t = make()
    install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    held = t._held
    held.opened_at -= slng.tts._MAX_CONNECTION_AGE_S
    assert await drive_reply(t, sample_deltas()) > 0  # reply lands on a fresh socket
    assert held.ws.closed and held.ws.count("text") > 0
    await asyncio.sleep(0.05)
    assert t._held is not None and t._held is not held
    await t.aclose()


# --- a stalled background connect ------------------------------------


async def test_a_stalled_prewarm_costs_the_reply_at_most_the_wait_budget(monkeypatch):
    monkeypatch.setattr(slng.tts, "_PENDING_CONNECT_WAIT_S", 0.3)
    t = make()
    session = install_fake_session(t)
    stalled = asyncio.Event()
    real_connect = session.ws_connect

    async def connect(url, **kw):
        if not session.created and not stalled.is_set():
            stalled.set()
            await asyncio.sleep(30)  # the prewarm never finishes
        return await real_connect(url, **kw)

    session.ws_connect = connect  # type: ignore[method-assign]
    t.prewarm()
    await asyncio.sleep(0.01)
    started = time.perf_counter()
    assert await drive_reply(t, sample_deltas()) > 0
    took = time.perf_counter() - started
    assert took < 0.3 + 0.3, f"the reply waited {took:.2f}s on a stalled connect"
    await t.aclose()


# --- update_options is not a gateway failure ------------------------


async def test_update_options_after_prewarm_is_not_counted(monkeypatch):
    t = make()
    install_fake_session(t)
    await warm(t)
    for voice in ("a", "b", "c", "d", "e"):
        t.update_options(voice=voice)
        await asyncio.sleep(0.05)
    assert t._failed_reopens == 0
    assert t._held is not None and t._held.ws.sent[0]["voice"] == "e"
    assert t._reconnect_handle is None, "an options change was throttled"
    await t.aclose()


# --- shutdown closes the HTTP session under the instance -------------


async def test_no_reconnect_once_the_http_session_is_closed(caplog):
    t = make()
    session = install_fake_session(t)
    await warm(t)
    session.closed = True
    with caplog.at_level(logging.WARNING, logger="livekit_slng"):
        session.created[0].push_close(1006)
        await asyncio.sleep(0.1)
    assert len(session.created) == 1
    assert t._connect_task is None or t._connect_task.done()
    assert not any("background connect failed" in r.message for r in caplog.records)
    await t.aclose()


# --- an opening in a script written without spaces -----------------


async def test_cjk_opening_rejoins_without_a_space():
    tok = slng.SentenceTokenizer()
    stream = tok.stream(partial_head=True)
    for piece in chars(CJK):
        stream.push_text(piece)
    stream.end_input()
    got = [ev async for ev in stream]
    assert st.is_partial_head(got[0]) and st.separator_after(got[0]) == ""
    joined = got[0].token + st.separator_after(got[0]) + got[1].token
    assert joined == CJK


async def test_latin_opening_keeps_its_space():
    tok = slng.SentenceTokenizer()
    stream = tok.stream(partial_head=True)
    for w in words(LONG_OPENING):
        stream.push_text(w)
    stream.end_input()
    got = [ev async for ev in stream]
    assert st.is_partial_head(got[0]) and st.separator_after(got[0]) == " "


async def test_cjk_opening_on_the_wire_has_no_inner_space():
    t = make()
    session = install_fake_session(t, lambda: FakeWs(accepts_partial_text=True))
    await warm(t)
    assert await drive_reply(t, chars(CJK)) > 0
    texts = session.created[0].texts()
    assert texts[0].get("partial") is True, texts
    assert "".join(m["text"] for m in texts).rstrip() == CJK
    await t.aclose()


async def test_cjk_opening_merged_on_a_fresh_socket_has_no_inner_space():
    t = make()
    t._partial_text_supported = True  # learned earlier; this socket has not said so yet
    session = install_fake_session(t, lambda: FakeWs(accepts_partial_text=True))
    assert await drive_reply(t, chars(CJK)) > 0
    texts = session.created[0].texts()
    assert [m["text"].rstrip() for m in texts] == [CJK], texts
    await t.aclose()


# --- reuse metrics --------------------------------------------------


async def test_metrics_report_reuse_and_acquire_time():
    t = make()
    install_fake_session(t)
    seen: list = []
    t.on("metrics_collected", lambda m: seen.append(m))
    await drive_reply(t, sample_deltas())
    await drive_reply(t, sample_deltas())
    await asyncio.sleep(0.05)
    assert len(seen) >= 2
    assert seen[0].connection_reused is False and seen[0].acquire_time > 0
    assert seen[-1].connection_reused is True
    await t.aclose()


# --- blingfire failure logging --------------------------------------


def test_blingfire_failure_warns_once_then_debugs(monkeypatch, caplog):
    monkeypatch.setattr(st, "_blingfire_warned", False)

    def boom(text):
        raise UnicodeEncodeError("utf-8", text, 0, 1, "surrogates not allowed")

    monkeypatch.setattr(st.blingfire, "text_to_sentences_with_offsets", boom)
    with caplog.at_level(logging.DEBUG, logger="livekit_slng"):
        assert st._blingfire_ends("a. b.") is None
        assert st._blingfire_ends("c. d.") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "blingfire" in r.message]
    assert len(warnings) == 1 and "rest of this process" not in warnings[0].message
    assert len(debugs) == 1


# --- control traffic must not defer the idle checks -------


class PingingWs(FakeWs):
    """Pings every 20 ms, answered inside receive() with its timeout restarted, as aiohttp does."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._pinger = asyncio.get_running_loop().create_task(self._ping())

    async def _ping(self) -> None:
        import aiohttp
        from fakews import _msg

        while not self.closed:
            await asyncio.sleep(0.02)
            self._rx.put_nowait(_msg(aiohttp.WSMsgType.PING, b""))

    async def receive(self, timeout=None):
        import aiohttp

        while True:
            msg = await super().receive(timeout)
            if msg.type is aiohttp.WSMsgType.PING:
                continue  # autoping: answered internally, the timeout starts over
            return msg

    async def close(self) -> bool:
        self._pinger.cancel()
        return await super().close()


async def test_pings_do_not_defer_the_keepalive_or_the_idle_limit(monkeypatch):
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    monkeypatch.setattr(slng.tts, "_MAX_IDLE_S", 0.3)
    t = make()
    session = install_fake_session(t, PingingWs)
    await warm(t)
    ws = session.created[0]
    await asyncio.sleep(0.5)
    assert ws.count("keepalive") >= 3, ws.sent
    assert ws.closed and t._held is None, "pings kept an idle socket alive past its limit"
    await t.aclose()


async def test_pings_do_not_defer_the_age_limit(monkeypatch):
    monkeypatch.setattr(slng.tts, "_IDLE_KEEPALIVE_S", 0.05)
    t = make()
    install_fake_session(t, PingingWs)
    await warm(t)
    first = t._held
    first.opened_at -= slng.tts._MAX_CONNECTION_AGE_S
    await asyncio.sleep(0.2)
    assert first.ws.closed and t._held is not first
    await t.aclose()


async def test_pings_do_not_stretch_the_quiet_window():
    t = make()
    t._schedule_connect = lambda **_: None
    install_fake_session(t, lambda: PingingWs(audio_chunks_per_reply=40, chunk_interval_s=0.01))
    await drive_reply(t, sample_deltas(), interrupt_after_events=2)
    settle = t._settle
    assert settle is not None
    started = time.perf_counter()
    await asyncio.wait({settle.task}, timeout=2.0)
    assert time.perf_counter() - started < slng.tts._CANCEL_QUIET_S + 0.1
    await t.aclose()
