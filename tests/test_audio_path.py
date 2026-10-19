"""Decoding audio frames: an emitter failure is not reported as an encoding problem."""

import base64

import pytest
from fakews import FakeWs, drive_reply, install_fake_session
from livekit.agents import APIConnectOptions

import livekit_slng as slng


def make(**kw):
    kw.setdefault("warm_standby_enabled", False)
    return slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", **kw)


class PassthroughWs(FakeWs):
    """Replies in the raw provider form: {"audio": ..., "isFinal": ...}."""

    async def _stream_reply(self) -> None:
        audio = base64.b64encode(b"\x00\x01" * 480).decode()
        self.push_json({"audio": audio})
        self.push_json({"audio": audio, "isFinal": True})


async def test_emitter_failure_is_not_swallowed_as_a_base64_error(monkeypatch, caplog):
    t = make()
    install_fake_session(t, PassthroughWs)

    from livekit.agents.tts import tts as tts_base

    real_push = tts_base.AudioEmitter.push

    def exploding_push(self, data):
        raise RuntimeError("resampler exploded")

    monkeypatch.setattr(tts_base.AudioEmitter, "push", exploding_push)
    with pytest.raises(Exception):  # noqa: B017 - any failure will do, as long as it is not swallowed
        await drive_reply(
            t, ["Hello there friend. "], conn_options=APIConnectOptions(max_retry=0, timeout=5.0)
        )
    monkeypatch.setattr(tts_base.AudioEmitter, "push", real_push)
    assert not any("invalid base64" in r.getMessage() for r in caplog.records), (
        "an emitter failure was logged as an encoding problem"
    )
    await t.aclose()


async def test_corrupt_base64_is_still_reported_and_skipped(caplog):
    class CorruptWs(FakeWs):
        async def _stream_reply(self) -> None:
            self.push_json({"audio": "!!!not base64!!!"})
            self.push_json({"audio": base64.b64encode(b"\x00\x01" * 480).decode(), "isFinal": True})

    t = make()
    install_fake_session(t, CorruptWs)
    assert await drive_reply(t, ["Hello there friend. "]) > 0
    assert any("invalid base64" in r.getMessage() for r in caplog.records)
    await t.aclose()
