"""Tokenizer edge cases: contractions and possessives, letterless tails, and wrapped lines.

A contraction or possessive ends its sentence, a letterless tail of up to max_chars joins the
sentence before it, and a line that starts in lowercase continues the sentence above it.
"""

import asyncio
import re

import pytest
from fakews import drive_reply, install_fake_session

import livekit_slng as slng

TOK = slng.SentenceTokenizer()


def make(**kw):
    kw.setdefault("warm_standby_enabled", False)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def words(text: str) -> list[str]:
    return re.findall(r"\s*\S+|\s+", text)


async def streamed(pushes: list[str], tok: slng.SentenceTokenizer = TOK) -> tuple[list[str], int]:
    stream = tok.stream()
    out: list[str] = []

    async def read() -> None:
        async for token in stream:
            out.append(token.token)

    reader = asyncio.create_task(read())
    for push in pushes:
        stream.push_text(push)
        await asyncio.sleep(0)
    before = len(out)
    stream.end_input()
    await reader
    return out, before


CASES = {
    # contractions and possessives
    "I can't. She won't. We don't. Try again.": [
        "I can't.",
        "She won't.",
        "We don't.",
        "Try again.",
    ],
    "We went to John's. Then we left.": ["We went to John's.", "Then we left."],
    "We ate at McDonald's. It was fine.": ["We ate at McDonald's.", "It was fine."],
    "It was the 1990's. Then it changed.": ["It was the 1990's.", "Then it changed."],
    "No, it doesn't. Anything else?": ["No, it doesn't.", "Anything else?"],
    "I can’t. Try again.": ["I can’t.", "Try again."],
    "I can‘t. Try again.": ["I can‘t.", "Try again."],
    # one-letter words behind an opener still wait for the next word
    'He said "J. Smith is here." Then left.': ['He said "J. Smith is here."', "Then left."],
    "He called. (J. Smith stayed.) Bye.": ["He called.", "(J. Smith stayed.)", "Bye."],
    "Sie können z. B. per Karte zahlen. Das geht schnell.": [
        "Sie können z. B. per Karte zahlen.",
        "Das geht schnell.",
    ],
    "Ευχαριστώ για την υπομονή σας. Ο κ. Παπαδόπουλος θα σας καλέσει.": [
        "Ευχαριστώ για την υπομονή σας.",
        "Ο κ. Παπαδόπουλος θα σας καλέσει.",
    ],
    "So do I. Then we left.": ["So do I.", "Then we left."],
    # a lowercase line continues a wrapped sentence
    "Your order shipped yesterday and\nshould arrive on Friday. Anything else?": [
        "Your order shipped yesterday and should arrive on Friday.",
        "Anything else?",
    ],
    "Call 555-1234\nor 555-9999\nanytime.": ["Call 555-1234 or 555-9999 anytime."],
    "Welcome to Acme support. I can help with orders,\n    returns and billing. What can I do?": [
        "Welcome to Acme support.",
        "I can help with orders, returns and billing.",
        "What can I do?",
    ],
    "Η παραγγελία σας\nθα φτάσει αύριο. Ευχαριστώ.": [
        "Η παραγγελία σας θα φτάσει αύριο.",
        "Ευχαριστώ.",
    ],
    "Ваш заказ\nприбудет завтра. Спасибо.": ["Ваш заказ прибудет завтра.", "Спасибо."],
    # the documented price: a lowercase list runs together after its first item
    "You will need:\nyour ID\na utility bill": ["You will need:", "your ID a utility bill"],
    # still cut after a colon, before a list letter, before a capital, in CJK
    "Run this command:\npip install app\nThen restart the service.": [
        "Run this command:",
        "pip install app",
        "Then restart the service.",
    ],
    "Options:\na) the basic plan\nb) the premium plan": [
        "Options:",
        "a) the basic plan",
        "b) the premium plan",
    ],
    "Steps:\ni. open the app\nii. tap settings": ["Steps:", "i. open the app", "ii. tap settings"],
    "Order status\nYour order shipped.": ["Order status", "Your order shipped."],
    "To reset your password:\n1. Open the app.\n2. Tap Settings.": [
        "To reset your password:",
        "1. Open the app.",
        "2. Tap Settings.",
    ],
    "请按以下步骤操作：\n打开应用。\n点击设置。": [
        "请按以下步骤操作：",
        "打开应用。",
        "点击设置。",
    ],
    # a tail that fits still joins the sentence before it
    "Your code is:\n4200": ["Your code is: 4200"],
}


@pytest.mark.parametrize(("text", "expected"), CASES.items())
async def test_cases_batch_and_streamed(text, expected):
    assert TOK.tokenize(text) == expected
    for pushes in (words(text), list(text)):
        out, before = await streamed(pushes)
        assert out == expected
        assert before == len(expected) - 1


# --- the letterless tail stays within max_chars ----------------------------------

NUMBERS = "Here are the numbers:\n" + "1234 5678\n" * 60


async def test_a_long_letterless_tail_stays_within_max_chars_and_streams():
    pieces = TOK.tokenize(NUMBERS)
    assert pieces[0] == "Here are the numbers:"
    assert all(len(piece) <= 200 for piece in pieces)
    out, before = await streamed(words(NUMBERS))
    assert out == pieces
    assert before == len(pieces) - 1, "the sentence and the lines before the last leave early"


async def test_a_long_letterless_run_keeps_the_stream_buffer_bounded():
    stream = TOK.stream()
    drained: list[str] = []

    async def read() -> None:
        async for token in stream:
            drained.append(token.token)

    reader = asyncio.create_task(read())
    longest = 0
    for push in words("Here are the numbers:\n" + "1234 5678\n" * 400):
        stream.push_text(push)
        longest = max(longest, len(stream._buf))
        await asyncio.sleep(0)
    stream.end_input()
    await reader
    assert longest <= 450, longest  # a held sentence plus up to max_chars of tail
    assert "".join(drained).count("1234 5678") == 400


def test_a_tail_longer_than_a_smaller_max_chars_is_not_merged():
    tok = slng.SentenceTokenizer(max_chars=40)
    assert tok.tokenize("Your code is:\n4200") == ["Your code is: 4200"]
    # 29 characters of tail join, although the piece then passes 40.
    assert tok.tokenize("Your code is:\n1234 5678\n1234 5678\n1234 5678") == [
        "Your code is: 1234 5678 1234 5678 1234 5678"
    ]
    # 49 characters of tail do not.
    assert (
        tok.tokenize("Your code is:\n" + "1234 5678\n" * 5) == ["Your code is:"] + ["1234 5678"] * 5
    )


LONG_SENTENCE = "word " * 38 + "number:"  # 197 characters, one piece under max_chars


def test_a_short_tail_joins_a_sentence_of_any_length():
    assert len(LONG_SENTENCE) == 197
    assert TOK.tokenize(LONG_SENTENCE + "\n4200") == [LONG_SENTENCE + " 4200"]


# --- contractions end to end: the frames on the wire -------------------------------------------


async def test_contractions_leave_as_frames_of_their_own():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["I can't.", " She won't."]) > 0
    ws = session.created[0]
    assert [m["text"] for m in ws.texts()] == ["I can't. ", "She won't. "]
    assert ws.count("flush") == 1
    await t.aclose()


# --- tails end to end: no frame without a letter -----------------------------------------


@pytest.mark.parametrize(
    "deltas",
    [
        [LONG_SENTENCE, "\n", "4200"],
        ["Here are your codes:\n"] + ["1234 5678\n"] * 15,
    ],
)
async def test_a_letterless_tail_within_max_chars_never_leaves_a_frame_without_a_letter(deltas):
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, deltas) > 0
    ws = session.created[0]
    texts = [m["text"] for m in ws.texts()]
    assert len(texts) == 1 and any(c.isalpha() for c in texts[0]), texts
    assert ws.count("flush") == 1
    await t.aclose()
