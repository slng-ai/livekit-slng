# Changelog

All notable changes to `livekit-slng` are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0]

First release as `livekit-slng`. It replaces `livekit-plugins-slng`, the SLNG
plugin published with LiveKit Agents (last version 1.8.8). See "Moving from
livekit-plugins-slng" in the README for the three steps to switch.

### Changed

- The package is `livekit-slng` and the import is `livekit_slng`
  (`import livekit_slng as slng`). The logger is `livekit_slng`.
- `slng_base_url` has no default on STT or TTS. Pass your region's host, for
  example `us-east.api.slng.ai`, or set `SLNG_BASE_URL`. Without either,
  construction raises a `ValueError`. Connections given as full endpoint URLs
  need neither.
- Whitespace around the host or the API key (`slng_base_url`, `SLNG_BASE_URL`,
  `api_key`, `SLNG_API_KEY`), such as an env file's trailing newline, is
  trimmed, and a blank value counts as missing.
- TTS holds one WebSocket for the whole call instead of one per reply.
  `warm_standby_enabled` defaults to True: `prewarm()` opens the connection,
  and it counts as one concurrent SLNG session for the whole call, silences of
  up to five idle minutes included. It is replaced every 20 minutes, between
  replies. Set `warm_standby_enabled=False` for a connection per reply.
- TTS `text_chunking="auto"` sends one frame per sentence instead of
  clause-sized frames, because a model that renders each frame as it arrives
  put a closing intonation on every clause. A `WordTokenizer` keeps
  clause-sized frames.
- The TTS init carries `language` and `speed` only when you set them, so a
  model's catalog defaults apply otherwise.
- Interruptions use the acknowledged `clear` message instead of `cancel`, so a
  barge-in settles in about 220 to 300 ms and keeps the socket.
- A failed reply reaches the session as one unrecoverable error, however many
  attempts and models it went through. Retried and recovered attempts are
  reported as recoverable, as livekit's own streams do. Before, a single
  failed reply could end the call.
- A connection that fails after a reply's audio has started raises
  `APIConnectionError`, where the reply used to end early as if complete. So
  does a close after the audio with a code other than 1000 or 1001.
- A reply the server ends before all of its text was sent raises
  `APIConnectionError` once more speech turns up, and nothing more is written
  to that socket.
- The TTS receive timeout counts only time the server owes audio. A slow LLM,
  for example during a tool call, no longer fails the reply.
- A missed `first_audio_timeout_s` raises `APITimeoutError` naming the
  configured value.
- A reply with nothing to say, such as whitespace or punctuation alone, sends
  nothing and ends without audio. A reply of only a number is still spoken.
- A connection's own `init` no longer freezes its settings. On TTS its fields
  are kept and the plugin's settings win where both set one. On STT only the
  options changed with `update_options` are laid over it.
- A reconnect caused by STT `update_options` drops the audio kept for
  replaying the abandoned utterance.
- A TTS `encoding` keyword argument other than `linear16` raises a
  `ValueError`.
- Errors carry SLNG's request ID, which SLNG support can look up, instead of a
  local one: from the error frame, the connection's `ready`, or a refused
  upgrade's `X-Slng-Request-Id` header. It is the `request_id` of an
  `APIStatusError` (None when the server gave none), and ends the message of
  an `APIConnectionError` or `APITimeoutError`. A refused STT upgrade raises
  `APIStatusError` instead of aiohttp's `WSServerHandshakeError`, and a refused
  upgrade's message names the service, the endpoint and the HTTP status.
- Every error raised for a closed connection names the close code and the
  server's close reason, cut to 200 characters. The STT failover logs put the
  error's text under `lk.pii.error` instead of in the message.
- An STT connect or `ready` that misses its timeout raises `APITimeoutError`
  naming the timeout, instead of `APIConnectionError`, and its failover is
  reported with the reason `"timeout"`. It is retried and failed over as before.

### Added

- `slng.SentenceTokenizer`, the default for sentence frames. It ends a
  sentence at any script's terminator (Unicode `Sentence_Terminal`), at a line
  break, and at a space past `max_chars` (200), keeps list numbers, titles,
  contractions and Indonesian honorifics with their sentence, and joins a
  letterless last line (a number) to the sentence before it.
- Early openings: where the gateway advertises `accepts_partial_text`, the
  opening clause of a long first sentence is sent as soon as it exists, marked
  `partial: true`, so models that can start early speak about 450 ms sooner.
  `slng.SentenceTokenizer(partial_head=False)` turns this off, and
  `partial_head_min_chars` (default 25) sets the shortest opening.
- `TTSMetrics` report `connection_reused` and `acquire_time` for every reply.
- livekit's `tts.FallbackAdapter` can wrap `slng.TTS`.
- `project_id` on STT and TTS: an SLNG project UUID, sent as the
  `X-SLNG-Project-Id` header to select the project that usage is billed and
  limited under. It is optional, and nothing is sent when it is omitted.
- The gateway's canonical error codes decide retry and failover, in both of
  its error formats: a provider's refused request, rejected credential or
  used-up quota moves on to the next model without a retry, while rate limits,
  outages, restarts and session timeouts are retried. A code the plugin does
  not know yet follows the error's `recovery` hint. A number outside 100-599 in
  an error frame, such as a WebSocket close code, is no longer reported as an
  HTTP status.
