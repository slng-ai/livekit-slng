"""The language-agnostic sentence tokenizer.

Covers sentence ends in many scripts, the length budget, grapheme clusters at a cut, and the
early opening of a long first sentence.
"""

import asyncio
import unicodedata

import pytest
from fakews import drive_reply, install_fake_session
from livekit.agents import tokenize

import livekit_slng as slng
from livekit_slng.sentence_tokenizer import (
    _CLAUSE_MARKS,
    _HEAD_MIN_CLAUSE_CHARS,
    _SHORT_HEAD_CHARS,
    _STERM,
    _head_cut,
    _split,
)

# (text, expected sentence count). Whitespace-separated so the streamed variant
# can feed it word by word.
MATRIX = {
    "English": ("Hello there friend. How are you today? Your order ships tomorrow.", 3),
    "Bahasa": (
        "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami, "
        "statusnya sedang dalam proses pengiriman dan diperkirakan sampai besok sore. "
        "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak.",
        2,
    ),
    "Hindi": ("नमस्ते, आप कैसे हैं। मैं ठीक हूँ। आपका ऑर्डर कल पहुंच जाएगा।", 3),
    "Bengali": ("নমস্কার। আপনি কেমন আছেন। আমি ভালো আছি।", 3),
    "Tamil": ("வணக்கம். நீங்கள் எப்படி இருக்கிறீர்கள்? நான் நன்றாக இருக்கிறேன்.", 3),
    "Chinese": ("你好，我很好。 你的订单明天到达。 谢谢。 你还需要别的吗。", 4),
    "Japanese": ("「はい。」と言った。 それから帰った。 注文は明日届きます。", 3),
    "Arabic": ("كيف حالك؟ أنا بخير. سيصل طلبك غدا.", 3),
    "Burmese": ("မင်္ဂလာပါ။ နေကောင်းလား။ ကောင်းပါတယ်။", 3),
    "Khmer": ("សួស្តី។ អ្នកសុខសប្បាយទេ។ ខ្ញុំសុខសប្បាយ។", 3),
    "Armenian": ("Բարև։ Ինչպես ես։ Լավ եմ։", 3),
}


def squash(s: str) -> str:
    return "".join(s.split())


@pytest.mark.parametrize("name", list(MATRIX))
def test_tokenize_matrix(name):
    text, expected = MATRIX[name]
    tk = slng.SentenceTokenizer()
    sentences = tk.tokenize(text)
    assert len(sentences) == expected, sentences
    assert squash("".join(sentences)) == squash(text), "text was dropped or duplicated"


async def stream_words(
    tk: tokenize.SentenceTokenizer, text: str, *, partial_head: bool = False
) -> tuple[list[str], int]:
    """Feed word by word; return (all tokens, tokens emitted before end_input)."""
    st = tk.stream(partial_head=partial_head)
    got: list[str] = []

    async def reader() -> None:
        async for t in st:
            got.append(t.token)

    task = asyncio.create_task(reader())
    for w in text.split(" "):
        st.push_text(w + " ")
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    early = len(got)
    st.end_input()
    await task
    return got, early


@pytest.mark.parametrize("name", list(MATRIX))
async def test_stream_matrix_emits_before_end_of_input(name):
    text, expected = MATRIX[name]
    got, early = await stream_words(slng.SentenceTokenizer(), text)
    assert len(got) == expected, got
    assert squash("".join(got)) == squash(text)
    assert early == expected - 1, "every sentence but the held last one should leave early"


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith is here.",
        "See e.g. the docs.",
        "The U.S. economy grew.",
        "Pi is 3.14 roughly.",
        "Visit example.com today.",
        "Mail me at first.last@email.com please.",
        "Wait for it... there it is.",
        "It cost Rp 2.500.000 total.",
    ],
)
def test_latin_edge_cases_stay_whole(text):
    assert slng.SentenceTokenizer().tokenize(text) == [text]


def test_quotation_closing_mid_sentence_does_not_break():
    """「はい。」と言った。 is one sentence; a break after the bracket is a pause inside it."""
    assert slng.SentenceTokenizer().tokenize("「はい。」と言った。それから帰った。") == [
        "「はい。」と言った。",
        "それから帰った。",
    ]
    # but a quotation that really does end the sentence still breaks
    assert slng.SentenceTokenizer().tokenize("彼は「はい。」 それから帰った。") == [
        "彼は「はい。」",
        "それから帰った。",
    ]
    assert slng.SentenceTokenizer().tokenize('Ia berkata "halo." Lalu pergi.') == [
        'Ia berkata "halo."',
        "Lalu pergi.",
    ]


def test_run_of_terminators_is_one_boundary():
    assert slng.SentenceTokenizer().tokenize("本当？！ そうです。") == ["本当？！", "そうです。"]


def test_thai_is_cut_at_a_space_once_past_the_budget():
    phrase = "สวัสดีครับ วันนี้อากาศดีมาก "
    text = (phrase * 12).strip()  # ~330 chars, no terminator anywhere
    assert len(text) > 200
    sentences = slng.SentenceTokenizer().tokenize(text)
    assert len(sentences) == 2
    assert len(sentences[0]) <= 200
    # the cut landed at a space, so the first piece ends on a whole phrase
    assert text[len(sentences[0])] == " "
    assert squash("".join(sentences)) == squash(text)


# Clusters whose marks have canonical combining class 0, which is why an
# earlier version of the guard (unicodedata.combining) let the cut split them.
CLUSTERS = {
    "thai sara i": "\u0e17\u0e21\u0e34" * 200,  # THO + MO + SARA I (ccc 0)
    "thai tone": "\u0e17\u0e35\u0e48" * 200,  # THO + SARA II + MAI EK
    "devanagari vowel": "\u0915" + "\u0915\u093f" * 200,  # KA + VOWEL SIGN I (Mc, ccc 0)
    "devanagari virama": "\u0915\u094d\u0937" * 150,  # KA + VIRAMA + SSA conjunct
    "zwj": ("\u0915\u200d\u0937") * 150,
    "emoji zwj": "\U0001f468\u200d\U0001f469\u200d\U0001f467 " * 40,
}


@pytest.mark.parametrize("name", list(CLUSTERS))
def test_hard_cut_never_splits_a_cluster(name):
    text = CLUSTERS[name]
    pieces = slng.SentenceTokenizer().tokenize(text)
    assert len(pieces) > 1, "expected the budget to cut this text"
    for piece in pieces[1:]:
        head = piece[0]
        assert not unicodedata.category(head).startswith("M"), (
            f"{name}: a frame starts with the mark {head!r}, stranded from its base"
        )
        assert head not in "\u200c\u200d", f"{name}: a frame starts with a joiner"
    for piece in pieces[:-1]:
        tail = piece[-1]
        assert tail not in "\u200c\u200d", f"{name}: a frame ends on a dangling joiner"
        assert unicodedata.combining(tail) != 9, f"{name}: a frame ends on a virama"
    assert squash("".join(pieces)) == squash(text)


def test_budget_gives_way_rather_than_splitting_a_cluster():
    """When the only cut inside the budget would split a cluster, overshoot instead."""
    text = "\u0915" + "\u0915\u093f" * 100  # 201 chars, the 200th is a vowel sign
    pieces = slng.SentenceTokenizer().tokenize(text)
    assert pieces == [text], "should keep it whole rather than strand the mark"


def test_budget_is_configurable():
    text = "one two three four five six seven eight nine ten"
    assert len(slng.SentenceTokenizer(max_chars=20).tokenize(text)) == 3
    with pytest.raises(ValueError):
        slng.SentenceTokenizer(max_chars=0)


def test_a_line_break_ends_a_piece_and_whitespace_only_is_nothing():
    tk = slng.SentenceTokenizer()
    assert tk.tokenize("First line\nContinues here. Then more.") == [
        "First line",
        "Continues here.",
        "Then more.",
    ]
    # A line starting in lowercase goes on with the sentence above it.
    assert tk.tokenize("First line\ncontinues here. Then more.") == [
        "First line continues here.",
        "Then more.",
    ]
    assert tk.tokenize("   \n  ") == []
    assert _split("", max_chars=200) == []


def test_offsets_are_contiguous_and_advance():
    text = "Hello there. नमस्ते। 你好。 Done."
    pieces = _split(text, max_chars=200)
    assert pieces[0][1] == 0 and pieces[-1][2] == len(text)
    for (_, _, end), (_, start, _) in zip(pieces, pieces[1:], strict=False):
        assert end == start, "spans must be contiguous so the stream buffer is fully consumed"


def test_ascii_terminators_are_not_in_the_vendored_set():
    assert not (_STERM & set(".!?")), "ASCII terminators belong to blingfire"
    assert len(_STERM) == 169
    for ch in "।॥。！？؟۔။។։።":
        assert ch in _STERM, ch


async def test_last_sentence_is_held_until_more_text_or_end_of_input():
    """The sender's inline flush depends on the tokenizer retaining its last sentence."""
    st = slng.SentenceTokenizer().stream()
    got: list[str] = []

    async def reader() -> None:
        async for t in st:
            got.append(t.token)

    task = asyncio.create_task(reader())
    st.push_text("नमस्ते, आप कैसे हैं। ")
    await asyncio.sleep(0)
    assert got == [], "a complete sentence must be held until the next one starts"
    st.push_text("मैं ")
    await asyncio.sleep(0)
    assert got == ["नमस्ते, आप कैसे हैं।"]
    st.end_input()
    await task
    assert got == ["नमस्ते, आप कैसे हैं।", "मैं"]


async def test_hindi_reply_end_to_end_flushes_on_the_last_frame():
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=False)
    assert isinstance(t._opts.word_tokenizer, slng.SentenceTokenizer)
    session = install_fake_session(t)
    text, _ = MATRIX["Hindi"]
    assert await drive_reply(t, [w + " " for w in text.split(" ")]) > 0
    ws = session.created[0]
    texts = ws.texts()
    assert [m["text"].strip() for m in texts] == slng.SentenceTokenizer().tokenize(text)
    assert not any(m.get("flush") for m in texts)
    assert ws.count("flush") == 1 and ws.sent[-1]["type"] == "flush"
    await t.aclose()


def test_a_lone_surrogate_does_not_lose_the_reply(caplog):
    """Blingfire raises on text it cannot encode; lone surrogates are now swapped out first."""
    import livekit_slng.sentence_tokenizer as st

    st._blingfire_warned = False
    text = "A reply with a split emoji \ud83d in it. And a second sentence."
    pieces = slng.SentenceTokenizer().tokenize(text)
    assert len(pieces) == 2, pieces
    assert squash("".join(pieces)) == squash(text)
    assert not any("blingfire" in r.getMessage() for r in caplog.records)
    st._blingfire_warned = False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Great work‼️ Keep going.", ["Great work‼️", "Keep going."]),
        ("Really⁉️ Yes.", ["Really⁉️", "Yes."]),
    ],
)
def test_a_terminator_cut_keeps_its_variation_selector(text, expected):
    """The cluster guard has to apply to sentence boundaries, not only the budget."""
    assert slng.SentenceTokenizer().tokenize(text) == expected


THAI_CLUSTERS = {
    # SARA AM and SARA AA are category Lo, so the general category misses them.
    "spacing vowel": ("กขคงจฉชญฎกำ" * 6, 10),
    # A leading vowel is stored before the consonant it is pronounced after.
    "leading vowel": ("กขคงเดินทา" * 6, 10),
}


@pytest.mark.parametrize("name", list(THAI_CLUSTERS))
def test_thai_budget_cuts_keep_vowels_with_their_consonant(name):
    text, max_chars = THAI_CLUSTERS[name]
    pieces = slng.SentenceTokenizer(max_chars=max_chars).tokenize(text)
    assert len(pieces) > 1
    for piece in pieces[1:]:
        assert piece[0] not in "าำາຳ", f"{name}: a spacing vowel was orphaned"
        assert not unicodedata.category(piece[0]).startswith("M")
    for piece in pieces[:-1]:
        assert piece[-1] not in "เแโใไ", (
            f"{name}: a frame ends on a leading vowel, whose consonant is in the next one"
        )
    assert squash("".join(pieces)) == squash(text)


def test_flags_and_skin_tones_survive_a_budget_cut():
    """Regional indicators pair up, and a skin tone binds to the emoji before it."""
    flags = "\U0001f1e9\U0001f1ea\U0001f1eb\U0001f1f7\U0001f1ef\U0001f1f5" * 6
    for max_chars in (5, 7, 9, 11):
        pieces = slng.SentenceTokenizer(max_chars=max_chars).tokenize(flags)
        assert all(len(p) % 2 == 0 for p in pieces), (max_chars, pieces)
    waving = "\U0001f44b\U0001f3fd " * 10
    pieces = slng.SentenceTokenizer(max_chars=6).tokenize(waving)
    assert all(not p.startswith("\U0001f3fd") for p in pieces)


def test_the_budget_applies_to_a_long_punctuated_sentence_too():
    """Documented behaviour, and the reason max_chars is not a Thai-only knob."""
    text = "word " * 60 + "end."
    pieces = slng.SentenceTokenizer().tokenize(text)
    assert len(pieces) == 2, "a 305-character sentence is cut by the budget"
    assert all(len(p) <= 200 for p in pieces)


# (text, expected openings). A short first opening is followed by a second.
HEADS = {
    "clause first": (
        "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem "
        "kami dan statusnya sedang dalam proses pengiriman. Terima kasih.",
        # the first comma leaves too short an opening, so the clause runs on
        ["Baik kak, untuk pesanan yang kakak sebutkan tadi,"],
    ),
    "no clause, long": (
        "The quick brown fox jumps over the lazy dog while the sun sets behind the "
        "distant mountains tonight. Then it sleeps.",
        ["The quick brown fox jumps over the lazy dog while the sun sets"],
    ),
    "short opening": ("Hello there. How can I help you today?", []),
    "short opening with a clause": ("Sure, I can help. What is your order number?", []),
    "decimal comma is not a clause": (
        "The total is 2,5 million rupiah for the whole order including delivery today. Thanks.",
        ["The total is 2,5 million rupiah for the whole order including"],
    ),
}


@pytest.mark.parametrize("name", list(HEADS))
async def test_the_head_policy(name):
    text, expected = HEADS[name]
    pieces, _early = await stream_words(slng.SentenceTokenizer(), text, partial_head=True)
    plain, _ = await stream_words(slng.SentenceTokenizer(), text)
    assert pieces[: len(expected)] == expected, pieces
    assert len(pieces) == len(plain) + len(expected), pieces
    assert " ".join(pieces) == " ".join(plain)


async def heads_and_pieces(text: str) -> tuple[list[str], list[str]]:
    """The pieces with and without the early opening, for the same input."""
    st = slng.SentenceTokenizer().stream(partial_head=True)
    got: list[tuple[str, bool]] = []

    async def reader() -> None:
        async for t in st:
            got.append((t.token, slng.sentence_tokenizer.is_partial_head(t)))

    task = asyncio.create_task(reader())
    for word in text.split(" "):
        st.push_text(word + " ")
        await asyncio.sleep(0)
    st.end_input()
    await task
    plain, _ = await stream_words(slng.SentenceTokenizer(), text)
    return [token for token, _ in got], plain


@pytest.mark.parametrize(
    "text",
    [
        "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami. Terima kasih ya kak.",
        "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak, terima kasih. Sampai jumpa.",
        "First line\ncontinues here for quite a while with plenty of words in it, and then it ends.\nThen more.",
        "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains. Then it sleeps.",
        "नमस्ते, आप कैसे हैं और आपका दिन कैसा जा रहा है यह जानना चाहता हूँ। मैं ठीक हूँ।",
    ],
)
async def test_head_plus_remainder_is_the_sentence_output(text):
    """The hazard: _split normalizes, so the head must come from raw offsets."""
    with_head, plain = await heads_and_pieces(text)
    assert " ".join(with_head) == " ".join(plain), "the text sent to the gateway changed"
    assert with_head[1:] != plain or len(with_head) == len(plain)


async def test_no_word_is_duplicated_or_dropped_across_the_head():
    text = (
        "Thank you  for  calling  today,  your  order  of  Rp 2.500.000  is  on  its  "
        "way  to  you  now.  It  arrives  tomorrow."
    )
    with_head, plain = await heads_and_pieces(text)
    # A run of whitespace at the cut is trimmed, as at any other piece edge, so
    # compare by word rather than by byte here.
    assert " ".join(with_head).split() == " ".join(plain).split()
    assert len(with_head) == len(plain) + 1


async def test_a_reply_gets_one_opening_by_default_and_two_with_a_lowered_minimum():
    text = (
        "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya cek dulu ya. Untuk pesanan "
        "kakak yang kedua, statusnya sedang dalam proses pengiriman sekarang. Terima kasih "
        "banyak atas kesabarannya, sampai jumpa lagi."
    )
    pieces, _ = await stream_words(slng.SentenceTokenizer(), text, partial_head=True)
    plain, _ = await stream_words(slng.SentenceTokenizer(), text)
    assert pieces[:2] == ["Baik kak, untuk pesanan yang kakak sebutkan tadi,", "saya cek dulu ya."]
    assert len(pieces) == len(plain) + 1, pieces
    assert " ".join(pieces) == " ".join(plain)
    lowered = slng.SentenceTokenizer(partial_head_min_chars=8)
    pieces, _ = await stream_words(lowered, text, partial_head=True)
    assert pieces[:3] == [
        "Baik kak,",
        "untuk pesanan yang kakak sebutkan tadi,",
        "saya cek dulu ya.",
    ]
    assert len(pieces) == len(plain) + 2, pieces


async def test_the_tokenizer_can_be_told_not_to_release_openings():
    tk = slng.SentenceTokenizer(partial_head=False)
    stream = tk.stream(partial_head=True)
    assert isinstance(stream, tokenize.SentenceStream)
    assert stream._head_allowed is False, "the constructor setting has the last word"


async def test_a_short_first_sentence_passes_the_opening_to_the_second():
    """ "Hello there." plays for under a second, so the next sentence may open early."""
    text = (
        "Hello there. Now let me explain the whole situation to you in some detail, "
        "because there is quite a lot of it to cover today. Thanks for waiting."
    )
    pieces, _ = await stream_words(slng.SentenceTokenizer(), text, partial_head=True)
    assert pieces == [
        "Hello there.",
        "Now let me explain the whole situation to you in some detail,",
        "because there is quite a lot of it to cover today.",
        "Thanks for waiting.",
    ]


async def test_a_first_sentence_of_25_characters_keeps_later_ones_whole():
    text = (
        "Thanks for waiting on me. Now let me explain the whole situation to you in some "
        "detail, because there is quite a lot of it to cover today. Bye."
    )
    pieces, _ = await stream_words(slng.SentenceTokenizer(), text, partial_head=True)
    plain, _ = await stream_words(slng.SentenceTokenizer(), text)
    assert pieces == plain, "a later sentence was split although audio was already playing"


# ---- clause marks in every script, and the minimum opening length ----------


def by_words(text: str) -> list[str]:
    return [w + " " for w in text.split(" ")]


def by_chars(text: str, size: int = 2) -> list[str]:
    """How an LLM streams a script written without spaces: a few characters a token."""
    return [text[i : i + size] for i in range(0, len(text), size)]


async def pieces_for(chunks: list[str], *, partial_head: bool) -> list[tuple[str, bool]]:
    st = slng.SentenceTokenizer().stream(partial_head=partial_head)
    got: list[tuple[str, bool]] = []

    async def reader() -> None:
        async for t in st:
            got.append((t.token, slng.sentence_tokenizer.is_partial_head(t)))

    task = asyncio.create_task(reader())
    for chunk in chunks:
        st.push_text(chunk)
        await asyncio.sleep(0)
    st.end_input()
    await task
    return got


# (chunks, the opening it must release). Each one checks head + remainder
# against the same input in sentence mode, byte for byte.
SPACED = {
    # A line break ends a piece, so the opening cannot span one; this puts one
    # after the opening instead.
    "newline": (
        "Thank you for calling us today, your order is on its way\nand should arrive soon. Bye.",
        "Thank you for calling us today,",
    ),
    "double spaces inside the pieces": (
        "Thank  you  for  calling  us  today, your  order  is  on  its  way  and  should  "
        "arrive  soon.  Bye.",
        "Thank  you  for  calling  us  today,",
    ),
    "Rp 2.500.000": (
        "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak, terima kasih "
        "banyak. Sampai jumpa.",
        # no clause mark in the first 60 characters, so the word fallback
        "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim",
    ),
    "Arabic comma": (
        "مرحبا بك في خدمة العملاء لدينا، لقد راجعت طلبك وهو في الطريق إليك الآن. هل هناك شيء آخر؟",
        "مرحبا بك في خدمة العملاء لدينا،",
    ),
    "spaced em dash": (
        "I have checked your order for you just now — it left our warehouse this morning "
        "and arrives tomorrow. Anything else?",
        "I have checked your order for you just now —",
    ),
    "spaced en dash": (
        "Your refund has been approved by our team – it should reach your account within "
        "three days. Anything else?",
        "Your refund has been approved by our team –",
    ),
    # Each of these has a mark past character 25 that must NOT count, so it is
    # the whitespace rule that holds it, not the minimum length.
    "2,5": (
        "The total for your order is 2,5 million rupiah, and it is charged today. Thanks.",
        "The total for your order is 2,5 million rupiah,",
    ),
    "10:30": (
        "Your visit is booked for 10:30 at the clinic; please arrive early today. Thanks.",
        "Your visit is booked for 10:30 at the clinic;",
    ),
    "well-known": (
        "We have seen this exact well-known problem with older routers, and a restart "
        "fixes it. Try that.",
        "We have seen this exact well-known problem with older routers,",
    ),
    "10-20": (
        "Delivery usually takes about 10-20 minutes in your area, depending on the "
        "traffic. Thanks.",
        "Delivery usually takes about 10-20 minutes in your area,",
    ),
}


@pytest.mark.parametrize("name", list(SPACED))
async def test_head_plus_remainder_is_exactly_the_sentence_output(name):
    text, head = SPACED[name]
    with_head = await pieces_for(by_words(text), partial_head=True)
    plain = await pieces_for(by_words(text), partial_head=False)
    assert [t for t, partial in with_head if partial] == [head], with_head
    assert with_head[0] == (head, True), "the opening is the reply's first piece"
    assert " ".join(t for t, _ in with_head) == " ".join(t for t, _ in plain)


# Scripts written without spaces. The batcher's trailing space adds one at the
# join, which TTS ignores, so these compare with whitespace removed.
UNSPACED = {
    "Chinese fullwidth comma": (
        "您好，感谢您致电我们的客服中心，我已经查到了您的订单信息，它今天早上已经从仓库发出了，"
        "预计明天下午送达。还有其他问题吗？",
        ["您好，感谢您致电我们的客服中心，我已经查到了您的订单信息，"],
    ),
    "Japanese ideographic comma": (
        "お電話ありがとうございます、ご注文の状況を確認いたしましたところ、本日発送済みでございます。"
        "他にご質問はございますか？",
        ["お電話ありがとうございます、ご注文の状況を確認いたしましたところ、"],
    ),
}


@pytest.mark.parametrize("name", list(UNSPACED))
async def test_head_plus_remainder_matches_in_scripts_without_spaces(name):
    text, heads = UNSPACED[name]
    with_head = await pieces_for(by_chars(text), partial_head=True)
    plain = await pieces_for(by_chars(text), partial_head=False)
    assert [t for t, partial in with_head if partial] == heads, with_head
    assert squash("".join(t for t, _ in with_head)) == squash("".join(t for t, _ in plain))
    assert len(with_head) == len(plain) + 1


@pytest.mark.parametrize(
    "text",
    [
        "Hi, thanks for calling. How can I help?",
        "Sure, I can check that booking for you now. One moment please.",
        # the comma is at character 7, just short of the minimum
        "Got it, we are checking your booking for you right now. Please hold.",
        "你好，我很好。你的订单明天到达。",
        "مرحبا، كيف حالك اليوم؟ أنا بخير.",
    ],
)
async def test_an_early_clause_mark_does_not_split(text):
    chunks = by_chars(text) if " " not in text else by_words(text)
    with_head = await pieces_for(chunks, partial_head=True)
    plain = await pieces_for(chunks, partial_head=False)
    assert with_head == plain, "a short opening must stay whole, and stay cacheable"


def test_the_minimum_counts_the_mark():
    assert _HEAD_MIN_CLAUSE_CHARS == 25
    assert _head_cut("x" * 23 + ", more words") is None, "a 24-character opening"
    assert _head_cut("x" * 24 + ", more words") == 26, "a 25-character opening"
    assert _head_cut("x" * 7 + ", more words", 8) == 9, "an 8-character one, when lowered"


@pytest.mark.parametrize(
    "chunks",
    [
        by_words(t)
        for t in (
            "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek. Terima kasih.",
            "Okay, so, well, yes, let me pull up the details of your booking right now. Done.",
            "One — two — three — four — five — six — seven — eight — nine. Ten.",
        )
    ]
    + [by_chars("好，好，好，好，好，好，好，好，好，好，好，好，好，好，好，好。完。")],
)
async def test_no_opening_is_ever_shorter_than_the_minimum(chunks):
    with_head = await pieces_for(chunks, partial_head=True)
    heads = [t for t, partial in with_head if partial]
    assert 1 <= len(heads) <= 2, with_head
    assert all(len(h) >= _HEAD_MIN_CLAUSE_CHARS for h in heads), heads
    assert len(heads) == 1 or len(heads[0]) < _SHORT_HEAD_CHARS, "only a short one gets a second"


def test_a_non_ascii_mark_needs_no_space_but_keeps_its_closer():
    text = "我们已经收到了您的订单并且会尽快为您安排发货和配送，”他说道"
    cut = _head_cut(text)
    assert cut is not None and text[:cut].endswith("，”"), "the quote stays with the clause"
    # a mark that is the last character so far waits for the next one
    assert _head_cut("我们已经收到了您的订单并且会尽快为您安排发货和配送，") is None


def test_a_dash_counts_only_when_spaced_and_not_a_bullet():
    opening = "Here is what I found for you today"
    assert _head_cut(opening + " — the first option") == len(opening) + 3
    assert _head_cut(opening + " \u2013 the first option") == len(opening) + 3
    assert _head_cut(opening + " - the first option") is None, "an ASCII hyphen is often a range"
    assert _head_cut(opening + "—the first option") is None
    assert _head_cut(opening + " —the first option") is None
    assert _head_cut(opening + "\n- the first option") is None, "a list bullet"
    assert _head_cut(opening + "\n  - the first option") is None, "an indented bullet"


def test_the_vendored_clause_table():
    assert len(_CLAUSE_MARKS) == 118
    assert _CLAUSE_MARKS.isdisjoint(_STERM), "a sentence terminator is not a clause mark"
    assert all(ord(ch) > 0x7F for ch in _CLAUSE_MARKS), "ASCII marks are handled apart"
    for ch in "，、；：،؛፣፤፥᠂､":
        assert ch in _CLAUSE_MARKS, hex(ord(ch))
