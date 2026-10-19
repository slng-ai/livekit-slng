# AGENTS.md

`livekit-slng` is SLNG's STT and TTS plugin for LiveKit Agents, published on PyPI and imported as `livekit_slng`. It talks to SLNG's gateway over WebSockets. It started as `livekit-plugins-slng` inside livekit/agents.

## Layout

- `src/livekit_slng/`: the package. `tts.py` holds the TTS and its connection handling, `stt.py` the STT, `sentence_tokenizer.py` the sentence tokenizer, `gateway_adapter.py` the gateway message formats, `connection.py` the connection configs and host resolution.
- `tests/`: unit tests. `fakews.py` is a scripted WebSocket double; no test reaches the network.

## Commands

```bash
uv sync                                               # install with the dev tools
uv run pytest                                         # tests
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy                                           # strict type check of src/
```

## Conventions

- Python 3.10 compatible: catch `asyncio.TimeoutError` as well as `TimeoutError`.
- Strict mypy, Google-style docstrings, 100-character lines.
- Use only public livekit-agents APIs, so new LiveKit releases keep working. CI tests the oldest allowed versions and the newest ones, and runs weekly.
- Never log reply text or transcripts in a log message; put provider text under the `lk.pii.*` extra keys.
- Every user-facing change gets a line under `[Unreleased]` in `CHANGELOG.md`.

## Releasing

1. Set the new version in `src/livekit_slng/version.py` and move the `[Unreleased]` entries in `CHANGELOG.md` under it, in a PR.
2. After it merges, publish a GitHub release whose tag is `v` plus that version, for example `v2.0.1`.
3. The Release workflow tests, builds and uploads to PyPI after a maintainer approves the `pypi` environment. It refuses a tag that does not match `version.py`.
