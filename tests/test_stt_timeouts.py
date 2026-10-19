"""STT connect and `ready` timeouts are timeouts.

They raise `APITimeoutError` naming the timeout, fail over with the reason "timeout", and are
retried and failed over exactly as before.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from livekit.agents import APIConnectOptions, APITimeoutError
from test_gateway_errors import AUDIO, SttSession, SttWs
from test_request_ids import read_until_error

import livekit_slng as slng

TWO = ["deepgram/nova:3", "soniox/stt-rt:v3"]


class StalledSession(SttSession):
    """The upgrade never completes for the models in ``stalled``."""

    def __init__(self, stalled: set[str], answers: dict) -> None:
        super().__init__(answers)
        self.stalled = stalled

    async def ws_connect(self, url: str, **kw):
        if any(url.endswith(model) for model in self.stalled):
            self.connects.append(url)
            await asyncio.sleep(30)
        return await super().ws_connect(url, **kw)


class NoReadyWs(SttWs):
    """Accepts the init and never answers it."""

    async def send_str(self, msg: str) -> None:
        self.sent.append({"type": "init"})


def stream_over(session, models, *, max_retry=0, timeout=0.2):
    stt_obj = slng.STT(api_key="k", connections=models, http_session=session)  # type: ignore[arg-type]
    events: list = []
    stt_obj.on("slng_event", events.append)
    stream = stt_obj.stream(conn_options=APIConnectOptions(max_retry=max_retry, timeout=timeout))
    for _ in range(5):
        stream._input_ch.send_nowait(SimpleNamespace(data=memoryview(AUDIO)))
    return stream, events


def reasons(events, name):
    return [e.data["reason"] for e in events if e.name == name]


async def test_a_connect_timeout_raises_a_timeout_naming_it():
    session = StalledSession({"deepgram/nova:3"}, {"deepgram/nova:3": {}})
    stream, events = stream_over(session, ["deepgram/nova:3"])
    exc = await read_until_error(stream, APITimeoutError)
    assert "0.2 s" in str(exc)
    assert reasons(events, "fallback.exhausted") == ["timeout"]


async def test_a_ready_timeout_raises_a_timeout_naming_it(monkeypatch):
    session = SttSession({"deepgram/nova:3": {}})

    async def ws_connect(url, **kw):
        session.connects.append(url)
        return NoReadyWs({})

    monkeypatch.setattr(session, "ws_connect", ws_connect)
    stream, events = stream_over(session, ["deepgram/nova:3"])
    exc = await read_until_error(stream, APITimeoutError)
    assert "ready" in str(exc) and "0.2 s" in str(exc)
    assert reasons(events, "fallback.exhausted") == ["timeout"]


async def test_a_timeout_is_still_retried_on_the_same_endpoint():
    session = StalledSession({"deepgram/nova:3"}, {"deepgram/nova:3": {}})
    stream, _ = stream_over(session, ["deepgram/nova:3"], max_retry=2, timeout=0.1)
    await read_until_error(stream, APITimeoutError)
    assert len(session.connects) == 3, "one attempt and two retries"


async def test_a_timeout_still_fails_over_and_says_why():
    session = StalledSession(
        {"deepgram/nova:3"},
        {"deepgram/nova:3": {}, "soniox/stt-rt:v3": {"type": "ready"}},
    )
    stream, events = stream_over(session, TWO, max_retry=1, timeout=0.1)

    async def read():
        async for _ in stream:
            pass

    reader = asyncio.create_task(read())
    while not any(url.endswith("soniox/stt-rt:v3") for url in session.connects):
        await asyncio.sleep(0.01)
    stream.end_input()
    await asyncio.wait_for(reader, 3)
    assert [url.rsplit("/", 2)[-2] for url in session.connects] == [
        "deepgram",
        "deepgram",
        "soniox",
    ]
    assert reasons(events, "fallback.attempt_failed") == ["timeout"]
    (failed,) = [e for e in events if e.name == "fallback.attempt_failed"]
    assert failed.data["timeout_s"] == 0.1


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("livekit_slng").setLevel(logging.CRITICAL)
    yield
    logging.getLogger("livekit_slng").setLevel(logging.NOTSET)
