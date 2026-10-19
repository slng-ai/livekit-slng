"""The init message and the reply terminator.

Covers which fields the init carries, the single standalone flush that ends a reply, and the
status attached to permanent gateway errors.
"""

import pytest
from fakews import drive_reply, install_fake_session, sample_deltas
from livekit.agents import APIConnectOptions

import livekit_slng as slng
from livekit_slng.gateway_adapter import build_tts_init_payload


def make(**kw):
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


def test_init_payload_omits_unset_fields():
    p = build_tts_init_payload(model="m", voice="v", sample_rate=24000, encoding="linear16")
    assert p["config"] == {"encoding": "linear16", "sample_rate": 24000}
    assert "language" not in p and "speed" not in p["config"]


def test_init_payload_includes_set_fields_and_passthrough():
    p = build_tts_init_payload(
        model="m",
        voice="v",
        sample_rate=24000,
        encoding="linear16",
        language="id",
        speed=1.2,
        model_options={"segment": "immediate", "speed": 0.9},
    )
    assert p["language"] == "id" and p["config"]["language"] == "id"
    assert p["config"]["speed"] == 0.9  # model_options wins, passthrough untouched
    assert p["config"]["segment"] == "immediate"


async def test_default_construction_sends_no_language_or_speed():
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, ["Hello there friend. "])
    init = session.created[0].sent[0]
    assert init["type"] == "init"
    assert "language" not in init
    assert "language" not in init["config"] and "speed" not in init["config"]
    assert init["config"]["encoding"] == "linear16" and init["config"]["sample_rate"] == 24000
    await t.aclose()


async def test_set_language_and_speed_are_sent():
    t = make(language="id", speed=1.1)
    session = install_fake_session(t)
    await drive_reply(t, ["Hello there friend. "])
    init = session.created[0].sent[0]
    assert init["config"]["language"] == "id" and init["config"]["speed"] == 1.1
    await t.aclose()


async def test_reply_is_terminated_by_one_standalone_flush():
    """The terminator is its own message, and there is exactly one of it."""
    t = make()
    session = install_fake_session(t)
    await drive_reply(t, sample_deltas())
    ws = session.created[0]
    assert ws.count("flush") == 1
    assert ws.sent[-1]["type"] == "flush", "the flush must follow the last text frame"
    assert not any(m.get("flush") for m in ws.texts()), "no inline flag on any frame"
    await t.aclose()


async def test_letterless_tail_still_flushes():
    t = make(text_chunking="phrase")
    session = install_fake_session(t)
    await drive_reply(t, ["Sampai jumpa lagi besok. ", "12345."])
    ws = session.created[0]
    assert ws.count("flush") == 1 and ws.sent[-1]["type"] == "flush"
    await t.aclose()


async def test_letterless_reply_is_sent_and_voiced():
    """A reply of "4200" is a whole answer; dropping it leaves the caller with silence."""
    t = make()
    session = install_fake_session(t)
    assert await drive_reply(t, ["4200."]) > 0
    ws = session.created[0]
    assert [m["text"].strip() for m in ws.texts()] == ["4200."]
    assert ws.count("flush") == 1 and ws.sent[-1]["type"] == "flush"
    await t.aclose()


async def test_whitespace_only_reply_sends_nothing():
    """Nothing to say is nothing to send, and not a failure."""
    t = make()
    session = install_fake_session(t)
    errors: list = []
    t.on("error", errors.append)
    assert await drive_reply(t, ["   "], conn_options=APIConnectOptions(max_retry=0)) == 0
    assert all(ws.texts() == [] and ws.count("flush") == 0 for ws in session.created)
    assert errors == []
    await t.aclose()


def test_encoding_and_sample_rate_cannot_be_smuggled_through_model_options():
    """The decode path owns these two, so a silent override means noise."""
    with pytest.raises(ValueError, match="encoding"):
        slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", encoding="mp3")
    # the named argument still works, and other model options are untouched
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", sample_rate=16000, modelId="coda")
    assert t.sample_rate == 16000 and t._opts.model_options == {"modelId": "coda"}


def test_a_language_model_option_reaches_both_places_it_is_read():
    """Models disagree about which copy wins, so the two copies must not disagree."""
    from livekit_slng.gateway_adapter import build_tts_init_payload

    payload = build_tts_init_payload(
        model="m",
        voice="v",
        sample_rate=24000,
        encoding="linear16",
        language="en",
        model_options={"language": "fr"},
    )
    assert payload["config"]["language"] == "fr"
    assert payload["language"] == "fr"


def test_permanent_gateway_errors_carry_a_status():
    """A code with no status is retried on every candidate, forever."""
    from livekit_slng.gateway_adapter import extract_error_status

    for code, status in (
        ("invalid_pronunciation", 400),
        ("pronunciation_not_found", 404),
        ("unsupported_watermark_format", 400),
        ("watermarking_unavailable", 503),
        ("configuration_error", 400),
    ):
        assert extract_error_status({"type": "error", "code": code}) == status, code
