"""The tracking headers: agent, session and project IDs, each sent only when set."""

import asyncio
from types import SimpleNamespace

import pytest
from fakews import drive_reply, install_fake_session, sample_deltas
from test_gateway_errors import AUDIO, SttSession

import livekit_slng as slng
from livekit_slng.gateway_adapter import build_external_tracking_headers

PROJECT = "0b7f2c1e-4d3a-4f6b-9a8e-2c5d1e7f3a90"
TRACKING = ("X-SLNG-Agent-Id", "X-SLNG-Session-Id", "X-SLNG-Project-Id")


def test_nothing_is_sent_when_no_id_is_set():
    assert build_external_tracking_headers(external_agent_id=None, external_session_id=None) == {}
    t = slng.TTS(api_key="k", model="deepgram/aura:2", voice="v")
    s = slng.STT(api_key="k", model="deepgram/nova:3")
    assert not set(TRACKING) & set(t._opts.extra_headers)
    assert not set(TRACKING) & set(s._extra_headers)


def test_the_project_id_is_sent_as_a_canonical_uuid():
    headers = build_external_tracking_headers(
        external_agent_id="agent-1",
        external_session_id="call-1",
        project_id=f" {PROJECT.upper()} ",
    )
    assert headers == {
        "X-SLNG-Agent-Id": "agent-1",
        "X-SLNG-Session-Id": "call-1",
        "X-SLNG-Project-Id": PROJECT,
    }


@pytest.mark.parametrize(
    ("project_id", "error"),
    [("", "must not be empty"), ("not-a-uuid", "must be a UUID"), ("a,b", "commas")],
)
def test_a_bad_project_id_fails_at_construction(project_id, error):
    with pytest.raises(ValueError, match=error):
        slng.TTS(api_key="k", model="deepgram/aura:2", voice="v", project_id=project_id)
    with pytest.raises(ValueError, match=error):
        slng.STT(api_key="k", model="deepgram/nova:3", project_id=project_id)


async def test_tts_sends_the_project_id_on_every_model_it_may_use():
    t = slng.TTS(
        api_key="k",
        connections=["deepgram/aura:2", "cartesia/sonic:3.5"],
        voice="v",
        project_id=PROJECT,
        warm_standby_enabled=False,
    )
    assert all(c._opts.extra_headers["X-SLNG-Project-Id"] == PROJECT for c in t._candidate_tts)
    session = install_fake_session(t._candidate_tts[0])
    assert await drive_reply(t, sample_deltas()) > 0
    assert session.connect_kwargs[0]["headers"]["X-SLNG-Project-Id"] == PROJECT
    await t.aclose()


async def test_stt_sends_the_project_id_on_connect():
    session = SttSession({"deepgram/nova:3": {"type": "ready"}})
    sent: list[dict] = []
    real_connect = session.ws_connect

    async def connect(url, **kw):
        sent.append(kw.get("headers") or {})
        return await real_connect(url, **kw)

    session.ws_connect = connect
    s = slng.STT(
        api_key="k",
        model="deepgram/nova:3",
        project_id=PROJECT,
        http_session=session,  # type: ignore[arg-type]
    )
    stream = s.stream()
    for _ in range(5):
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))

    async def read():
        async for _ in stream:
            pass

    reader = asyncio.create_task(read())
    while not sent:
        await asyncio.sleep(0.01)
    stream.end_input()
    await asyncio.wait_for(reader, 3)
    assert sent[0]["X-SLNG-Project-Id"] == PROJECT
