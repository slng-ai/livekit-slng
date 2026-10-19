"""Reading one frame by a deadline: a cancellation always stops the reader.

A reply that takes over the held socket cancels its idle reader and waits for it to
stop. Before Python 3.12, ``asyncio.wait_for`` returned the frame instead of
raising when the cancellation and the frame arrived together, so the reader read on
and the reply waited for it forever.
"""

import asyncio
import json
import time

import pytest
from fakews import FakeWs

from livekit_slng.tts import _receive_by


async def test_a_reader_cancelled_as_its_frame_arrives_still_stops():
    ws = FakeWs()
    reader = asyncio.ensure_future(_receive_by(ws, time.perf_counter() + 30))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    ws.push_json({"type": "ready"})
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader


async def test_a_cancelled_reader_leaves_no_read_behind():
    ws = FakeWs()
    reader = asyncio.ensure_future(_receive_by(ws, time.perf_counter() + 30))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    ws.push_json({"type": "ready"})
    msg = await _receive_by(ws, time.perf_counter() + 1)  # type: ignore[arg-type]
    assert json.loads(msg.data)["type"] == "ready", "a read left running took the frame"


async def test_a_read_past_its_deadline_times_out_and_ends():
    ws = FakeWs()
    with pytest.raises(asyncio.TimeoutError):
        await _receive_by(ws, time.perf_counter() + 0.01)  # type: ignore[arg-type]
    ws.push_json({"type": "ready"})
    msg = await _receive_by(ws, time.perf_counter() + 1)  # type: ignore[arg-type]
    assert json.loads(msg.data)["type"] == "ready"
