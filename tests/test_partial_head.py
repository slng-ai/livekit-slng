"""The early sentence opening (`partial: true`): when the TTS sends one, and when it does not."""

import asyncio

import pytest
from fakews import FakeWs, drive_reply, install_fake_session

import livekit_slng as slng

LONG_OPENING = (
    "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami "
    "dan statusnya sedang dalam proses pengiriman. Terima kasih ya kak."
)
SHORT_OPENING = "Hello there. How can I help you today?"
# The openings of LONG_OPENING: its first comma gives a short one, so a second
# follows at the next.
HEAD = "Baik kak, untuk pesanan yang kakak sebutkan tadi,"
HEADS = ["Baik kak,", "untuk pesanan yang kakak sebutkan tadi,"]


def deltas(text: str) -> list[str]:
    return [w + " " for w in text.split(" ")]


def make(**kw):
    kw.setdefault("warm_standby_enabled", True)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def supporting_ws() -> FakeWs:
    return FakeWs(accepts_partial_text=True)


async def warm(tts_obj) -> None:
    """Open the socket and read `ready`, as a session start does."""
    tts_obj.prewarm()
    await asyncio.sleep(0.05)


async def test_a_long_opening_is_released_early_when_the_gateway_takes_it():
    t = make()
    session = install_fake_session(t, supporting_ws)
    await warm(t)
    assert t._partial_text_supported is True
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0

    texts = session.created[0].texts()
    partials = [m for m in texts if m.get("partial")]
    assert [m["text"].strip() for m in partials] == [HEAD], texts
    assert partials == texts[:1], "the opening is the first frame of the reply"
    assert not texts[-1].get("partial"), "the reply's last frame is never an opening"

    # what reaches the gateway is exactly what sentence framing would have sent
    sent = " ".join(m["text"].strip() for m in texts)
    assert sent == " ".join(slng.SentenceTokenizer().tokenize(LONG_OPENING))
    await t.aclose()


async def test_a_short_opening_sends_no_partial_frame():
    t = make()
    session = install_fake_session(t, supporting_ws)
    await warm(t)
    assert await drive_reply(t, deltas(SHORT_OPENING)) > 0
    texts = session.created[0].texts()
    assert not any(m.get("partial") for m in texts), texts
    assert [m["text"].strip() for m in texts] == slng.SentenceTokenizer().tokenize(SHORT_OPENING)
    await t.aclose()


async def test_a_gateway_that_does_not_advertise_it_never_sees_partial():
    t = make()
    session = install_fake_session(t)  # ready has no accepts_partial_text
    await warm(t)
    assert t._partial_text_supported is False
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    texts = session.created[0].texts()
    assert not any("partial" in m for m in texts), texts
    assert [m["text"].strip() for m in texts] == slng.SentenceTokenizer().tokenize(LONG_OPENING)
    await t.aclose()


async def test_a_reply_before_ready_has_been_read_sends_whole_sentences():
    """Cold start with standby off: the capability is unknown, so nothing is risked."""
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t, supporting_ws)
    assert t._partial_text_supported is None
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    texts = session.created[0].texts()
    assert not any("partial" in m for m in texts), texts
    await t.aclose()


async def test_the_caller_can_turn_it_off():
    t = make(word_tokenizer=slng.SentenceTokenizer(partial_head=False))
    session = install_fake_session(t, supporting_ws)
    await warm(t)
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    assert not any("partial" in m for m in session.created[0].texts())
    await t.aclose()


async def test_an_interrupt_during_the_opening_leaves_nothing_behind():
    t = make()
    session = install_fake_session(
        t,
        lambda: FakeWs(accepts_partial_text=True, audio_chunks_per_reply=40, chunk_interval_s=0.01),
    )
    await warm(t)
    await drive_reply(t, deltas(LONG_OPENING), interrupt_after_events=1)
    await asyncio.sleep(slng.tts._CANCEL_SETTLE_S + slng.tts._CANCEL_QUIET_S + 0.2)

    before = {id(ws): len(ws.texts()) for ws in session.created}

    second = "Selamat pagi. Ada yang bisa saya bantu?"
    assert await drive_reply(t, deltas(second)) > 0
    after: list[str] = []
    for ws in session.created:
        after += [m["text"].strip() for m in ws.texts()[before.get(id(ws), 0) :]]
    assert after == slng.SentenceTokenizer().tokenize(second), after
    assert "Baik" not in " ".join(after), "text from the interrupted reply came back"
    assert session.created[0].count("clear") == 1
    await t.aclose()


@pytest.mark.parametrize("chunking", ["phrase", "word"])
async def test_other_chunking_modes_are_untouched(chunking):
    t = make(text_chunking=chunking)
    session = install_fake_session(t, supporting_ws)
    await warm(t)
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    assert not any("partial" in m for m in session.created[0].texts())
    await t.aclose()


async def test_a_fresh_socket_never_gets_a_bare_opening():
    """`ready` is unread while the first frames go out, so the opening must merge back."""
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t, supporting_ws)
    # The first reply reads `ready` while it runs, so the instance learns the
    # gateway supports openings, and the next reply's stream is allowed one.
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    assert t._partial_text_supported is True
    assert t._held is None  # standby off: nothing is held, the next reply connects

    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    fresh = session.created[-1]
    assert fresh.count("init") == 1, "expected a new socket"
    texts = [m["text"].strip() for m in fresh.texts()]
    assert not any("partial" in m for m in fresh.texts()), fresh.sent
    assert texts == slng.SentenceTokenizer().tokenize(LONG_OPENING), texts
    assert texts[0] != HEAD, "an opening went out on its own, with nothing marking it"
    await t.aclose()


async def test_an_opening_that_ends_the_reply_is_sent_plainly():
    """Nothing follows it, so it is the whole reply and goes out as an ordinary frame."""
    t = make(warm_standby_enabled=False)
    session = install_fake_session(t, supporting_ws)
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0
    assert t._held is None  # standby off: nothing is held, the next reply connects
    # A reply that stops mid-sentence: the opening is all there is.
    assert await drive_reply(t, [HEAD + " "]) > 0
    fresh = session.created[-1]
    texts = fresh.texts()
    assert len(texts) == 1 and not any("partial" in m for m in texts), fresh.sent
    assert texts[0]["text"].strip() == HEAD
    assert fresh.count("flush") == 1
    await t.aclose()


async def test_a_reply_that_is_nothing_but_an_opening_is_still_sent():
    """The opening took the whole reply, so there is no remainder to merge it into.

    The sender has to consume the opening while the input is still open, which
    is what puts it in the queue rather than in the end-of-reply batch.
    """
    from livekit.agents import APIConnectOptions

    t = make(warm_standby_enabled=False)
    session = install_fake_session(t, supporting_ws)
    assert await drive_reply(t, deltas(LONG_OPENING)) > 0  # the instance learns the capability
    assert t._held is None  # standby off: nothing is held, the next reply connects

    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2.0))
    total = 0

    async def drain() -> None:
        nonlocal total
        async for ev in stream:
            total += len(ev.frame.data)

    reader = asyncio.create_task(drain())
    stream.push_text("Baik kak, untuk pesanan yang kakak ")  # no clause long enough yet
    await asyncio.sleep(0.2)  # the socket is acquired and the sender is waiting
    stream.push_text("sebutkan tadi, ")  # the opening, consumed while the input is open
    await asyncio.sleep(0.2)
    stream.end_input()  # and the reply ends there
    await asyncio.wait_for(reader, timeout=5.0)
    await stream.aclose()

    fresh = session.created[-1]
    texts = fresh.texts()
    assert [m["text"].strip() for m in texts] == [HEAD], fresh.sent
    assert not any("partial" in m for m in texts), fresh.sent
    assert fresh.count("flush") == 1, fresh.sent
    assert total > 0
    await t.aclose()


async def test_an_opening_that_ends_the_reply_on_a_supporting_socket_is_terminated():
    """It went out marked as an opening, so the flush behind it has to close the turn."""
    from livekit.agents import APIConnectOptions

    t = make()
    session = install_fake_session(t, supporting_ws)
    await warm(t)  # ready read at prewarm, so this socket takes openings

    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2.0))
    total = 0

    async def drain() -> None:
        nonlocal total
        async for ev in stream:
            total += len(ev.frame.data)

    reader = asyncio.create_task(drain())
    stream.push_text("Baik kak, untuk pesanan yang kakak ")
    await asyncio.sleep(0.2)
    stream.push_text("sebutkan tadi, ")
    await asyncio.sleep(0.2)
    stream.end_input()
    await asyncio.wait_for(reader, timeout=5.0)
    await stream.aclose()

    ws = session.created[0]
    texts = ws.texts()
    # An opening needs more of its line after it, so a reply that ends at the
    # clause mark leaves as one ordinary frame, and the flush follows it.
    assert [m["text"].strip() for m in texts] == [HEAD], ws.sent
    assert not any(m.get("partial") for m in texts), ws.sent
    assert ws.sent[-1]["type"] == "flush", "and the turn still has to be terminated"
    assert total > 0
    await t.aclose()
