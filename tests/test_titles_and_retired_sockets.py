"""Titles that open a sentence, one-pass splitting, lone surrogates, and retired sockets.

A title such as "Dr." stays with the sentence it opens while "No." still ends one, and a
socket the idle reader decided to retire is never handed to a reply.
"""

import asyncio
import logging
import time

import pytest
from fakews import drive_reply, install_fake_session, sample_deltas

import livekit_slng as slng
from livekit_slng import sentence_tokenizer as st

TOK = slng.SentenceTokenizer()


async def streamed(text, *, one_push=False):
    stream = TOK.stream()
    out: list[str] = []

    async def read():
        async for token in stream:
            out.append(token.token)

    reader = asyncio.create_task(read())
    if one_push:
        stream.push_text(text)
    else:
        for word in text.split(" "):
            stream.push_text(word + " ")
            await asyncio.sleep(0)
    stream.end_input()
    await reader
    return out


# --- titles --------------------------------------------------------------------

TITLE_CASES = [
    (
        "Your balance is $5. Mrs. Davis called. Dr. Smith will see you now.",
        ["Your balance is $5.", "Mrs. Davis called.", "Dr. Smith will see you now."],
    ),
    (
        "Pay $5. Mr. Lee will call you back tomorrow.",
        ["Pay $5.", "Mr. Lee will call you back tomorrow."],
    ),
]


@pytest.mark.parametrize(("text", "expected"), TITLE_CASES)
async def test_a_title_opening_a_sentence_stays_with_it(text, expected):
    assert TOK.tokenize(text) == expected
    assert await streamed(text) == expected, "tokenize() and stream() must agree"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("He is John Smith Jr. He lives in Ohio.", ["He is John Smith Jr.", "He lives in Ohio."]),
        ("Got it. Thanks for waiting.", ["Got it.", "Thanks for waiting."]),
        ("No! I will not do that.", ["No!", "I will not do that."]),
        ("Yes? Tell me more.", ["Yes?", "Tell me more."]),
        ("I met Dr. Smith today. He said hi.", ["I met Dr. Smith today.", "He said hi."]),
        ("नमस्ते। आप कैसे हैं?", ["नमस्ते।", "आप कैसे हैं?"]),
        ("好的。我明白了。", ["好的。", "我明白了。"]),
    ],
)
async def test_other_cuts_are_unchanged(text, expected):
    assert TOK.tokenize(text) == expected
    assert await streamed(text) == expected


async def test_a_one_word_sentence_of_that_shape_still_ends():
    # The shape alone cannot tell "No." from "Dr.", so blingfire is asked again
    # with what follows. "No." keeps its own frame; the documented loss is a
    # title before a name blingfire reads as a sentence opener ("Dr. Who").
    assert TOK.tokenize("No. I don't think so.") == ["No.", "I don't think so."]


# --- one large push ------------------------------------------------------------


async def test_one_large_push_is_split_in_one_pass():
    base = "This is a normal sentence with some words in it. "
    text = (base * (80_000 // len(base) + 1))[:80_000]
    started = time.perf_counter()
    out = await streamed(text, one_push=True)
    took = time.perf_counter() - started
    assert out == TOK.tokenize(text)
    assert took < 2.0, f"one push of 80k characters took {took:.1f}s"


async def test_an_opening_released_early_is_not_repeated_by_the_one_pass_split():
    stream = TOK.stream(partial_head=True)
    out: list[str] = []

    async def read():
        async for token in stream:
            out.append(token.token)

    reader = asyncio.create_task(read())
    stream.push_text("Well, as I was saying to the team earlier this morning, ")
    await asyncio.sleep(0)
    stream.push_text("the plan is fine. The second one follows. And a third.")
    stream.end_input()
    await reader
    joined = " ".join(out)
    assert joined.count("the team") == 1 and joined.count("second") == 1
    assert out[-2:] == ["The second one follows.", "And a third."]


# --- blingfire and lone surrogates -------------------------------------------


def test_a_lone_surrogate_does_not_stop_blingfire(caplog):
    text = "3.14 see example.com and email a@b.co now \ud83d. Next one."
    with caplog.at_level(logging.DEBUG, logger="livekit_slng"):
        assert TOK.tokenize(text) == [
            "3.14 see example.com and email a@b.co now \ud83d.",
            "Next one.",
        ]
    assert not [r for r in caplog.records if "could not use blingfire" in r.getMessage()]


def test_without_blingfire_a_stop_only_cuts_before_whitespace(monkeypatch):
    def refuse(text):
        raise TypeError("refused")

    monkeypatch.setattr(st.blingfire, "text_to_sentences_with_offsets", refuse)
    assert TOK.tokenize("Pi is 3.14 on example.com today. Next one.") == [
        "Pi is 3.14 on example.com today.",
        "Next one.",
    ]


# --- a socket the idle reader retired --------------------------------------------


def make():
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", warm_standby_enabled=True)


async def test_a_socket_the_reader_is_retiring_is_not_handed_to_a_reply(monkeypatch):
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    contaminated = t._held
    real_stop = t._stop_idle_reader

    async def stop_after_stray_audio(conn):
        # The interleaving under test: while the reply holds the lock,
        # stray audio reaches the reader, which decides to retire the socket
        # and waits for the lock; only then does the reply cancel it.
        if conn is contaminated:
            conn.ws.push_audio()
            await asyncio.sleep(0.05)
        await real_stop(conn)

    monkeypatch.setattr(t, "_stop_idle_reader", stop_after_stray_audio)
    conn = await t._acquire_connection(timeout=5.0)
    try:
        assert conn is not contaminated, "the reply was handed the socket the reader retired"
    finally:
        await t._release_connection(conn, keep=True)
    await asyncio.sleep(0.05)
    assert contaminated.ws.closed
    assert len(session.created) == 2
    await t.aclose()


async def test_a_retired_socket_is_not_kept_after_a_reply():
    t = make()
    install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    conn = t._held
    conn.retired = True
    assert not t._may_keep(conn)
    await t.aclose()


async def test_a_healthy_idle_socket_is_still_reused():
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, sample_deltas()) > 0
    assert await drive_reply(t, sample_deltas()) > 0
    assert len(session.created) == 1
    await t.aclose()


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
