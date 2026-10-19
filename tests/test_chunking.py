"""Text chunking modes: sentence, phrase and word framing, and the frame batcher."""

import pytest
from fakews import (
    ListSentenceTokenizer,
    drive_reply,
    install_fake_session,
    sample_deltas,
)
from livekit.agents import tokenize

import livekit_slng as slng
from livekit_slng.tts import _FrameBatcher


def make(**kw):
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def test_default_is_sentence_with_sentence_tokenizer():
    t = make()
    assert t._opts.text_chunking == "sentence"
    assert isinstance(t._opts.word_tokenizer, tokenize.SentenceTokenizer)


def test_phrase_and_word_keep_word_tokenizer():
    for mode in ("phrase", "word"):
        t = make(text_chunking=mode)
        assert t._opts.text_chunking == mode
        assert isinstance(t._opts.word_tokenizer, tokenize.WordTokenizer)


def test_explicit_sentence_tokenizer_accepted():
    t = make(word_tokenizer=tokenize.blingfire.SentenceTokenizer())
    assert isinstance(t._opts.word_tokenizer, tokenize.SentenceTokenizer)


def test_invalid_value_raises():
    with pytest.raises(ValueError):
        make(text_chunking="clause")


@pytest.mark.parametrize("mode,expected", [("sentence", 2), ("phrase", 7)])
async def test_frame_count_per_mode(mode, expected):
    t = make(text_chunking=mode)
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    texts = session.created[0].texts()
    assert len(texts) == expected, [x["text"] for x in texts]
    if mode == "sentence":
        assert "Rp 2.500.000" in texts[1]["text"]
    await t.aclose()


async def test_word_mode_one_frame_per_word():
    from fakews import SAMPLE

    t = make(text_chunking="word")
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    texts = [m["text"] for m in session.created[0].texts()]
    # One frame per spoken word; a letterless token ("2.500.000") rides with its
    # neighbour, so the count is close to but not exactly the whitespace split.
    assert len(texts) >= len(SAMPLE.split()) - 2
    assert all(any(c.isalpha() for c in f) for f in texts)
    assert "".join(texts).split() == SAMPLE.split()
    await t.aclose()


# --- sentence framing -----------------------------------------


@pytest.mark.parametrize(
    "sentences",
    [
        ['He said "hello world."', "Then I left.", "(An aside.)", "Next one here."],
        ["She said “go.”", "I went."],
        ["こんにちは。", "元気ですか。"],
        ["Wait for it…", "There it is."],
    ],
)
def test_sentence_mode_is_one_frame_per_token(sentences):
    """Clause punctuation must not decide sentence boundaries."""
    batcher = _FrameBatcher(mode="sentence", max_chars=60)
    frames = [f for f in (batcher.push(s) for s in sentences) if f is not None]
    last = batcher.finish()
    if last is not None:
        frames.append(last)
    assert [f.strip() for f in frames] == sentences


def test_sentence_mode_ignores_phrase_max_chars():
    """A sentence longer than phrase_max_chars is still one frame."""
    long_sentence = "This sentence is comfortably longer than sixty characters in total."
    assert len(long_sentence) > 60
    batcher = _FrameBatcher(mode="sentence", max_chars=60)
    assert batcher.push(long_sentence).strip() == long_sentence


def test_phrase_mode_still_batches_at_punctuation():
    """The old behaviour is unchanged where it is still selected."""
    batcher = _FrameBatcher(mode="phrase", max_chars=60)
    frames = [f for f in (batcher.push(w) for w in ["Hello", "there,", "friend"]) if f is not None]
    assert [f.strip() for f in frames] == ["Hello there,"]


def test_finish_returns_a_letterless_tail():
    """At the batcher level, nothing is left behind at the end of a reply."""
    batcher = _FrameBatcher(mode="sentence", max_chars=60)
    assert batcher.push("4200") is None
    assert batcher.finish().strip() == "4200"


async def test_sentence_mode_end_to_end_one_frame_per_sentence():
    """With a tokenizer that does not merge, every sentence leaves on its own."""
    t = make(word_tokenizer=ListSentenceTokenizer())
    session = install_fake_session(t)
    sentences = ['He said "hello world."', "Then I left.", "(An aside.)", "Next one here."]
    await drive_reply(t, sentences)
    texts = session.created[0].texts()
    assert [m["text"].strip() for m in texts] == sentences
    assert not any(m.get("flush") for m in texts)
    assert session.created[0].count("flush") == 1
    await t.aclose()


def test_sentence_mode_rejects_a_word_tokenizer():
    """Silently framing per word while reporting "sentence" is worse than failing."""
    with pytest.raises(ValueError, match="SentenceTokenizer"):
        make(text_chunking="sentence", word_tokenizer=tokenize.basic.WordTokenizer())
    # "auto" with a WordTokenizer is a caller from before sentence mode: phrase batching
    assert make(word_tokenizer=tokenize.basic.WordTokenizer())._opts.text_chunking == "phrase"
    # the other modes still accept one
    assert make(text_chunking="word", word_tokenizer=tokenize.basic.WordTokenizer())


def test_sentence_mode_rejects_a_tokenizer_that_is_neither():
    """The check is 'can it produce sentences', not 'is it a word tokenizer'."""

    class Neither:
        def stream(self, *, language=None):  # pragma: no cover - never reached
            raise AssertionError("should not be used")

    with pytest.raises(ValueError, match="SentenceTokenizer"):
        make(word_tokenizer=Neither())
