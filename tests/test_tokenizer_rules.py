"""Eight rules of ``slng.SentenceTokenizer`` for lines, lists, scripts and quotes.

A line break ends a piece, a list number stays with its line, an ASCII stop after a CJK
character ends a sentence, a title opening a sentence is put to blingfire again, a ";" after a
Greek letter is a question mark, a one-letter word before a stop stays with the next word, a
quote closing mid-sentence is not a sentence end, and an ellipsis before a capital is.
"""

import asyncio
import re

import pytest
from livekit.agents.voice.transcription.filters import filter_markdown

import livekit_slng as slng
from livekit_slng import sentence_tokenizer as st

TOK = slng.SentenceTokenizer()


async def streamed(pushes: list[str]) -> tuple[list[str], int]:
    """Pieces from ``pushes``, and how many of them left before ``end_input()``."""
    stream = TOK.stream()
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


def words(text: str) -> list[str]:
    """Word pushes, each carrying its leading whitespace."""
    return re.findall(r"\s*\S+|\s+", text)


NEW = {
    "To reset your password:\n1. Open the app.\n2. Tap Settings.\n3. Choose Reset Password.": [
        "To reset your password:",
        "1. Open the app.",
        "2. Tap Settings.",
        "3. Choose Reset Password.",
    ],
    "Steps:\n1. Open the app\n2. Tap Settings": ["Steps:", "1. Open the app", "2. Tap Settings"],
    "Order status\nYour order shipped yesterday. It should arrive on Friday.": [
        "Order status",
        "Your order shipped yesterday.",
        "It should arrive on Friday.",
    ],
    "Here are your options:\nStandard shipping, five days\nExpress shipping, two days\nPickup in store": [
        "Here are your options:",
        "Standard shipping, five days",
        "Express shipping, two days",
        "Pickup in store",
    ],
    "你好.我是小明.很高兴认识你.": ["你好.", "我是小明.", "很高兴认识你."],
    "你好吗?我很好!谢谢.": ["你好吗?", "我很好!", "谢谢."],
    "Γεια σας, πώς μπορώ να βοηθήσω; Η παραγγελία σας θα φτάσει αύριο.": [
        "Γεια σας, πώς μπορώ να βοηθήσω;",
        "Η παραγγελία σας θα φτάσει αύριο.",
    ],
    "Ευχαριστώ για την υπομονή σας. Ο κ. Παπαδόπουλος θα σας καλέσει. Καλή σας μέρα.": [
        "Ευχαριστώ για την υπομονή σας.",
        "Ο κ. Παπαδόπουλος θα σας καλέσει.",
        "Καλή σας μέρα.",
    ],
    "जी हाँ। डॉ. शर्मा कल उपलब्ध होंगे।": ["जी हाँ।", "डॉ. शर्मा कल उपलब्ध होंगे।"],
    "بالتأكيد. د. أحمد متاح يوم الاثنين.": ["بالتأكيد.", "د. أحمد متاح يوم الاثنين."],
    'He said "Stop!" and left. Nobody followed him.': [
        'He said "Stop!" and left.',
        "Nobody followed him.",
    ],
    '"Can you hold?" she asked. "Of course," I said.': [
        '"Can you hold?" she asked.',
        '"Of course," I said.',
    ],
    "One moment… Your order is ready.": ["One moment…", "Your order is ready."],
    "让我查一下……好的，找到了。": ["让我查一下……", "好的，找到了。"],
    "Thanks for waiting. Dr. Smith will see you now.": [
        "Thanks for waiting.",
        "Dr. Smith will see you now.",
    ],
    "Of course. Mrs. Patel is on the line.": ["Of course.", "Mrs. Patel is on the line."],
    "Great news \ud83c the price is 3.14 today. Dr. Smith agrees. Bye.": [
        "Great news \ud83c the price is 3.14 today.",
        "Dr. Smith agrees.",
        "Bye.",
    ],
}

KEEP = {
    "Sure. Let me check that for you. One moment.": [
        "Sure.",
        "Let me check that for you.",
        "One moment.",
    ],
    "Okay. I have booked it for Monday.": ["Okay.", "I have booked it for Monday."],
    "No. I don't think so.": ["No.", "I don't think so."],
    "Dr. Smith will see you now. Please take a seat.": [
        "Dr. Smith will see you now.",
        "Please take a seat.",
    ],
    "Pi is about 3.14. That is close enough.": ["Pi is about 3.14.", "That is close enough."],
    "Visit example.com for details. You can also call us.": [
        "Visit example.com for details.",
        "You can also call us.",
    ],
    "Your order shipped.\n\nIs there anything else I can help with?": [
        "Your order shipped.",
        "Is there anything else I can help with?",
    ],
    "价格是 3.5 元。请确认。": ["价格是 3.5 元。", "请确认。"],
    "请访问 www.example.com 查看详情。谢谢！": ["请访问 www.example.com 查看详情。", "谢谢！"],
    "Sie können z. B. per Karte zahlen. Das geht schnell.": [
        "Sie können z. B. per Karte zahlen.",
        "Das geht schnell.",
    ],
    "Llegará el martes, entre las 9 y las 12 h. ¿Necesita algo más?": [
        "Llegará el martes, entre las 9 y las 12 h.",
        "¿Necesita algo más?",
    ],
    'He shouted "Stop!" Then he ran.': ['He shouted "Stop!"', "Then he ran."],
    "Hmm… that is odd. Let me look again.": ["Hmm… that is odd.", "Let me look again."],
    "「はい。」と言った。それから帰った。": ["「はい。」と言った。", "それから帰った。"],
    "本当？！ そうです。": ["本当？！", "そうです。"],
    "नमस्ते, आप कैसे हैं। मैं ठीक हूँ।": ["नमस्ते, आप कैसे हैं।", "मैं ठीक हूँ।"],
    "Your balance is $5. Mrs. Davis called. Dr. Smith will see you now.": [
        "Your balance is $5.",
        "Mrs. Davis called.",
        "Dr. Smith will see you now.",
    ],
    "He is John Smith Jr. He lives in Ohio.": ["He is John Smith Jr.", "He lives in Ohio."],
}
THAI = ("สวัสดีครับ วันนี้อากาศดีมาก " * 12).strip()
KEEP[THAI] = [
    ("สวัสดีครับ วันนี้อากาศดีมาก " * 7).strip(),
    ("สวัสดีครับ วันนี้อากาศดีมาก " * 5).strip(),
]


@pytest.mark.parametrize(("text", "expected"), [*NEW.items(), *KEEP.items()])
async def test_batch_and_word_stream_agree_and_hold_only_the_last(text, expected):
    assert TOK.tokenize(text) == expected
    out, before = await streamed(words(text))
    assert out == expected, "stream() and tokenize() must agree"
    assert before == len(expected) - 1, "every piece but the last leaves before end_input()"


@pytest.mark.parametrize(
    ("pushes", "expected", "before"),
    [
        (
            ["Sie können z. ", "B", ". per Karte zahlen. Das geht schnell."],
            ["Sie können z. B. per Karte zahlen.", "Das geht schnell."],
            1,
        ),
        (
            ["Thanks for waiting. Dr. Smith will see you now. Bye."],
            ["Thanks for waiting.", "Dr. Smith will see you now.", "Bye."],
            2,
        ),
        (["Order status\n"], ["Order status"], 0),
        (
            ["Hello. I met Dr.", " S", "mith yesterday."],
            ["Hello.", "I met Dr. Smith yesterday."],
            1,
        ),
        (
            ["Thanks for waiting. Dr.", " S", "mith will see you now. Bye."],
            ["Thanks for waiting.", "Dr. Smith will see you now.", "Bye."],
            2,
        ),
        (
            ["Thanks for waiting. Dr.", " ", "Smith will see you now. Bye."],
            ["Thanks for waiting.", "Dr. Smith will see you now.", "Bye."],
            2,
        ),
        (
            ["Thanks for waiting. Mr.", " L", "ee is here. Bye."],
            ["Thanks for waiting.", "Mr. Lee is here.", "Bye."],
            2,
        ),
        (['He said "Stop!"', " ", "and left. Bye."], ['He said "Stop!" and left.', "Bye."], 1),
        (
            ["One moment…", " ", "Your order is ready. Bye."],
            ["One moment…", "Your order is ready.", "Bye."],
            2,
        ),
        (
            ["Order status\nDr. Smith is here. Bye."],
            ["Order status", "Dr. Smith is here.", "Bye."],
            2,
        ),
        (["Sure.", " Let me check. Bye."], ["Sure.", "Let me check.", "Bye."], 2),
        (["Sure. ", "L", "et me check. Bye."], ["Sure.", "Let me check.", "Bye."], 2),
    ],
)
async def test_explicit_pushes_never_commit_to_a_boundary_early(pushes, expected, before):
    out, got_before = await streamed(pushes)
    assert out == expected
    assert got_before == before
    assert out == TOK.tokenize("".join(pushes)), "stream() and tokenize() must agree"
    assert not any(piece.endswith(("Dr.", "Mr.")) for piece in out)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A sentence ending in a one-letter word waits for the next one.
        (
            "Take vitamin C. It helps your immune system.",
            ["Take vitamin C. It helps your immune system."],
        ),
        # A lowercase brand after a closing quote reads as the sentence going on.
        ('He said "done." iPhones are next.', ['He said "done." iPhones are next.']),
        # A title before a name blingfire reads as a sentence opener.
        (
            "Your balance is $5. Mrs. Davis called. Dr. Who is a show.",
            ["Your balance is $5.", "Mrs. Davis called.", "Dr.", "Who is a show."],
        ),
        # German ordinals: no language-agnostic rule found.
        (
            "Wir liefern am 3. Oktober. Passt Ihnen das?",
            ["Wir liefern am 3.", "Oktober.", "Passt Ihnen das?"],
        ),
    ],
)
def test_documented_costs_and_residuals(text, expected):
    assert TOK.tokenize(text) == expected


async def test_the_markdown_filter_leaves_bare_lines_and_list_numbers():
    """What the line-break and list-number rules rest on."""

    async def source():
        for chunk in ["## Order status\n", "- Standard shipping\n", "1. Open the app\n"]:
            yield chunk

    filtered = "".join([chunk async for chunk in filter_markdown(source())])
    assert filtered == "Order status\nStandard shipping\n1. Open the app\n"
    assert TOK.tokenize(filtered) == ["Order status", "Standard shipping", "1. Open the app"]


async def test_a_finished_greek_question_gets_no_early_opening():
    stream = slng.SentenceTokenizer().stream(partial_head=True)
    out = []

    async def read():
        async for token in stream:
            out.append(token)

    reader = asyncio.create_task(read())
    # Long enough for an opening, with the space after ";" that would let the
    # clause scan take the question mark for a clause mark.
    stream.push_text("Γεια σας, πώς μπορώ να σας βοηθήσω σήμερα το πρωί; ")
    await asyncio.sleep(0)
    stream.end_input()
    await reader
    assert [token.token for token in out] == ["Γεια σας, πώς μπορώ να σας βοηθήσω σήμερα το πρωί;"]
    assert not any(st.is_partial_head(token) for token in out)


def test_a_mark_after_a_line_break_is_kept_once():
    # Malformed, and a mark after a line feed starts its own grapheme cluster
    # (UAX #29, GB4), so the cut may fall in front of it; it must not be lost.
    pieces = TOK.tokenize("Order status\ńYour order shipped.")
    assert pieces == ["Order status", "́Your order shipped."]


def test_many_line_breaks_stay_linear():
    import time

    text = "Line of text\n" * 5000
    started = time.perf_counter()
    assert len(TOK.tokenize(text)) == 5000
    assert time.perf_counter() - started < 1.0
