"""Test doubles for the SLNG TTS: a scripted WebSocket, a fake HTTP session and helpers."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, Literal

import aiohttp
from livekit.agents import tokenize

AUDIO_BYTES = b"\x00\x01" * 480
AUDIO_B64 = base64.b64encode(AUDIO_BYTES).decode()


def _msg(type_: aiohttp.WSMsgType, data: Any = None, extra: Any = None) -> SimpleNamespace:
    # aiohttp's WSMessage is (type, data, extra); a CLOSE carries the code in
    # data and the reason in extra, and an ERROR carries the exception in data.
    return SimpleNamespace(type=type_, data=data, extra=extra)


class FakeWs:
    """Scripted stand-in for aiohttp.ClientWebSocketResponse.

    Modelled on what the two runtimes really do:

    - init -> queues {"type":"ready"}
    - a STANDALONE flush -> queues `audio_chunks_per_reply` audio frames spaced
      `chunk_interval_s` apart, then audio_end, then a CLOSE frame when
      close_after_reply (a model that ends the turn by closing). A `flush` flag
      on a text frame is ignored, as most models do on the legacy host.
    - clear -> per on_cancel: "terminal" stops streaming and queues `cleared`,
      "silent" stops streaming and queues nothing, "streaming" keeps streaming,
      "terminal_then_more" queues `cleared` and then two more audio frames
    - cancel -> aborts nothing and is never acknowledged, like the bridge
    - audio is binary unless `binary_audio=False`, because that is the only
      form the bridge emits
    """

    def __init__(
        self,
        *,
        close_after_reply: bool = False,
        on_cancel: Literal[
            "terminal", "silent", "streaming", "terminal_then_more", "terminal_then_late"
        ] = "terminal",
        audio_chunks_per_reply: int = 3,
        chunk_interval_s: float = 0.0,
        fail_after_sends: int | None = None,
        send_delay_s: float = 0.0,
        raise_on_idle_receive: bool = False,
        late_audio_delay_s: float = 0.08,
        cancel_ack_delay_s: float = 0.0,
        binary_audio: bool = True,
        accepts_partial_text: bool = False,
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._exception: BaseException | None = None
        self.close_after_reply = close_after_reply
        self.on_cancel = on_cancel
        self.audio_chunks_per_reply = audio_chunks_per_reply
        self.chunk_interval_s = chunk_interval_s
        # After this many successful sends, every later write fails (a socket the
        # peer dropped without a close frame).
        self.fail_after_sends = fail_after_sends
        # send_str suspends this long before recording, so a cancel can land
        # inside the write the way aiohttp's drain does under backpressure.
        self.send_delay_s = send_delay_s
        # The next receive() raises once, like a transport error or aiohttp's
        # "Concurrent call to receive()" RuntimeError.
        self.raise_on_idle_receive = raise_on_idle_receive
        self.late_audio_delay_s = late_audio_delay_s
        # How long the gateway takes to acknowledge a cancel. A slow ack lands
        # the terminal frame near the settle deadline, which is where a quiet
        # window anchored to the deadline gives almost no quiet at all.
        self.cancel_ack_delay_s = cancel_ack_delay_s
        self.binary_audio = binary_audio
        self.accepts_partial_text = accepts_partial_text
        self._rx: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
        self._reply_task: asyncio.Task[None] | None = None
        self._late_task: asyncio.Task[None] | None = None

    # test controls
    def push_json(self, payload: dict[str, Any]) -> None:
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.TEXT, json.dumps(payload)))

    def push_audio(self) -> None:
        if self.binary_audio:
            self._rx.put_nowait(_msg(aiohttp.WSMsgType.BINARY, AUDIO_BYTES))
        else:
            self.push_json({"type": "audio_chunk", "data": AUDIO_B64})

    def push_transport_error(self, exc: Exception | None = None) -> None:
        """A WSMsgType.ERROR frame: aiohttp carries the exception in msg.data."""
        self._exception = exc or ConnectionResetError("transport failed")
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.ERROR, self._exception))

    def push_close(self, code: int = 1000, reason: str = "") -> None:
        self._rx.put_nowait(_msg(aiohttp.WSMsgType.CLOSE, code, reason))

    def kill(self) -> None:
        """Peer dropped the TCP connection: writes fail, reads return CLOSED."""
        self.closed = True
        self.close_code = 1006

    def exception(self) -> BaseException | None:
        return self._exception

    def texts(self) -> list[dict[str, Any]]:
        return [m for m in self.sent if m["type"] == "text"]

    def count(self, type_: str) -> int:
        return sum(1 for m in self.sent if m["type"] == type_)

    # aiohttp surface used by the plugin
    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError("fake socket closed")
        if self.fail_after_sends is not None and len(self.sent) >= self.fail_after_sends:
            self.closed = True
            raise ConnectionResetError("peer went away mid-reply")
        if self.send_delay_s:
            # A cancel landing here may find the frame already on the wire,
            # even though send_str never returned.
            await asyncio.sleep(self.send_delay_s)
        msg = json.loads(data)
        self.sent.append(msg)
        if msg["type"] == "init":
            ready = {"type": "ready", "session_id": "sess", "slng_request_id": "req"}
            if self.accepts_partial_text:
                ready["accepts_partial_text"] = True
            self.push_json(ready)
        elif msg["type"] == "flush":
            self._reply_task = asyncio.create_task(self._stream_reply())
        elif msg["type"] == "clear":
            if self._reply_task is not None and self.on_cancel != "streaming":
                self._reply_task.cancel()
            if self.on_cancel in ("terminal", "terminal_then_more"):
                self.push_json({"type": "cleared"})
            if self.on_cancel == "terminal_then_more":
                for _ in range(2):
                    self.push_audio()
            if self.on_cancel == "terminal_then_late":
                self._late_task = asyncio.create_task(self._late_audio())

    async def _late_audio(self) -> None:
        """A slow terminal frame, then audio behind it inside the quiet window."""
        if self.cancel_ack_delay_s:
            await asyncio.sleep(self.cancel_ack_delay_s)
        self.push_json({"type": "cleared"})
        await asyncio.sleep(self.late_audio_delay_s)
        self.push_audio()

    async def _stream_reply(self) -> None:
        for _ in range(self.audio_chunks_per_reply):
            if self.chunk_interval_s:
                await asyncio.sleep(self.chunk_interval_s)
            self.push_audio()
        self.push_json({"type": "audio_end"})
        if self.close_after_reply:
            self.push_close()

    async def receive(self, timeout: float | None = None) -> SimpleNamespace:
        if self.raise_on_idle_receive:
            self.raise_on_idle_receive = False
            raise RuntimeError("Concurrent call to receive() is not allowed")
        if self.closed:
            return _msg(aiohttp.WSMsgType.CLOSED)
        if timeout:
            msg = await asyncio.wait_for(self._rx.get(), timeout)
        else:
            msg = await self._rx.get()
        if msg.type == aiohttp.WSMsgType.CLOSE:
            self.closed = True
            self.close_code = msg.data
        return msg

    async def close(self) -> bool:
        self.closed = True
        for task in (self._reply_task, self._late_task):
            if task is not None:
                task.cancel()
        return True


class FakeSession:
    """Stands in for aiohttp.ClientSession so the real _connect_ws (and its init) runs."""

    def __init__(self, factory: Any, *, connect_delay_s: float = 0.0) -> None:
        self.factory = factory
        self.closed = False
        self.created: list[FakeWs] = []
        self.connect_kwargs: list[dict[str, Any]] = []
        # A gateway that accepts the TCP connection but is slow to hand back a
        # WebSocket, which is what makes a pending background connect visible.
        self.connect_delay_s = connect_delay_s

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWs:
        self.connect_kwargs.append(kwargs)
        if self.connect_delay_s:
            await asyncio.sleep(self.connect_delay_s)
        ws = self.factory()
        self.created.append(ws)
        return ws


def install_fake_session(
    tts_obj: Any, factory: Any = FakeWs, *, connect_delay_s: float = 0.0
) -> FakeSession:
    """Route tts_obj's connects through FakeSession; returns it (created sockets in .created)."""
    session = FakeSession(factory, connect_delay_s=connect_delay_s)
    tts_obj._session = session
    return session


class ListSentenceTokenizer(tokenize.SentenceTokenizer):
    """Emits each pushed string as exactly one sentence, with no merging.

    The real blingfire tokenizer merges short spans and holds its last token,
    which hides what the frame batcher does. This isolates the batcher.
    """

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [text] if text.strip() else []

    def stream(self, *, language: str | None = None) -> Any:
        return _ListSentenceStream()


class _ListSentenceStream(tokenize.SentenceStream):
    def __init__(self) -> None:
        super().__init__()

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if text.strip():
            self._event_ch.send_nowait(tokenize.TokenData(token=text.strip(), segment_id="seg"))

    def flush(self) -> None:
        self._check_not_closed()

    def end_input(self) -> None:
        self._check_not_closed()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()


async def drive_reply(
    tts_obj: Any,
    deltas: list[str],
    *,
    interrupt_after_events: int | None = None,
    conn_options: Any = None,
) -> int:
    """Push LLM-style deltas into one SynthesizeStream; return total audio bytes received.

    Bytes, not event count: the AudioEmitter re-chunks audio, so the number of
    SynthesizedAudio events is not the number of audio_chunk messages.
    """
    stream = tts_obj.stream(conn_options=conn_options) if conn_options else tts_obj.stream()
    for d in deltas:
        stream.push_text(d)
        await asyncio.sleep(0)
    stream.end_input()
    total = 0
    events = 0
    async for ev in stream:
        events += 1
        total += len(ev.frame.data)
        if interrupt_after_events is not None and events >= interrupt_after_events:
            await stream.aclose()
            break
    return total


SAMPLE = (
    "Baik kak, untuk pesanan yang kakak sebutkan tadi, saya sudah cek di sistem kami, "
    "statusnya sedang dalam proses pengiriman dan diperkirakan sampai besok sore. "
    "Total tagihannya Rp 2.500.000 dan sudah termasuk ongkos kirim ya kak."
)


def sample_deltas(n: int = 4) -> list[str]:
    words = SAMPLE.split(" ")
    return [
        " ".join(words[i : i + n]) + (" " if i + n < len(words) else "")
        for i in range(0, len(words), n)
    ]
