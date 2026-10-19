from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from livekit.agents import APIStatusError

_EXTERNAL_TRACKING_ID_RE = re.compile(r"[\x00-\x1f\x7f,]")
_MODEL_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_RETRYABLE_CLIENT_STATUS_CODES = {408, 409, 425, 429}
_ERROR_MESSAGE_MAX_LEN = 500
_CLOSE_REASON_MAX_LEN = 200

# The gateway's error codes, mapped to the HTTP status the retry logic reads.
# A code missing here yields no status, and a status-less error is retried on
# every candidate in turn, so a permanent failure (a malformed pronunciation
# reference, say) costs a full walk of the chain on every reply. Keep this in
# step with the codes the gateway emits on a websocket error frame. Both of its
# formats are here: the older codes, and the canonical ones its error envelope
# carries. A 4xx other than 408, 409, 425 and 429 is not retried on the same
# model; anything else is.
BRIDGE_ERROR_CODE_STATUS: dict[str, int] = {
    "auth_error": 401,
    "config_error": 400,
    "configuration_error": 400,
    "invalid_request": 400,
    "payload_too_large": 413,
    "rate_limit": 429,
    "rate_limit_exceeded": 429,
    "idle_timeout_exceeded": 408,
    "max_connection_duration_exceeded": 408,
    "not_ready": 503,
    "backpressure": 503,
    "stt_metering_unavailable": 503,
    "translation_error": 500,
    "provider_error": 502,
    "backend_error": 502,
    "backend_connection_failed": 502,
    # Pronunciation: a bad reference is permanent, an unreachable dictionary
    # service is not.
    "invalid_pronunciation": 400,
    "pronunciation_unauthenticated": 401,
    "pronunciation_not_found": 404,
    "pronunciation_unavailable": 503,
    # Watermarking, TTS only.
    "unsupported_watermark_format": 400,
    "watermarking_error": 500,
    "watermarking_unavailable": 503,
    # The canonical envelope. The provider's own failures, classified by the
    # gateway from the provider's HTTP status: a request or a credential the
    # provider refuses, or a quota it has used up, fails the same way on
    # every retry, so it moves on to the next model.
    "provider_invalid_request": 400,
    "provider_credential_rejected": 401,
    "provider_quota_exceeded": 402,
    "provider_rate_limited": 429,
    "provider_unavailable": 503,
    # The gateway's own failures.
    "internal_error": 500,
    "processing_error": 500,
    # The gateway ended the session: a new connection is all it takes.
    "idle_timeout": 408,
    "session_duration_exceeded": 408,
    "service_restart": 503,
}


def bridge_error_status(code: object) -> int | None:
    if isinstance(code, str):
        return BRIDGE_ERROR_CODE_STATUS.get(code.strip().lower())
    return None


def _http_status(value: object) -> int | None:
    """``value`` as an HTTP status: an int, or a numeric string, in 100-599."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return None
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    return None


def extract_error_status(frame: Mapping[str, object]) -> int | None:
    """The HTTP status an error frame stands for, which retry and failover read.

    The provider's own status (``upstream_status``) comes first, then the
    gateway's string ``code`` through ``BRIDGE_ERROR_CODE_STATUS``, then a
    numeric code in the older format. A number outside 100-599, such as a
    WebSocket close code, is not a status. ``provider_code`` is the provider's
    native code, in whatever scheme the provider uses, so it is never read.
    """
    upstream = frame.get("upstream_status")
    if isinstance(upstream, int) and not isinstance(upstream, bool) and 100 <= upstream <= 599:
        return upstream
    candidates: list[object] = []
    nested = frame.get("data")
    if isinstance(nested, Mapping):
        candidates.extend(nested.get(key) for key in ("status_code", "code", "err_code"))
    candidates.extend(frame.get(key) for key in ("status_code", "code", "err_code"))
    for value in candidates:
        status = bridge_error_status(value)
        if status is not None:
            return status
    for value in candidates:
        status = _http_status(value)
        if status is not None:
            return status
    return None


def request_id_from_frame(frame: Mapping[str, object]) -> str | None:
    """SLNG's request ID from a `ready` or an error frame, if it carries one."""
    value = frame.get("slng_request_id")
    return value if isinstance(value, str) and value else None


def request_id_from_handshake(exc: BaseException) -> str | None:
    """SLNG's request ID from a refused WebSocket upgrade's ``X-Slng-Request-Id`` header."""
    headers = getattr(exc, "headers", None)
    value = headers.get("X-Slng-Request-Id") if isinstance(headers, Mapping) else None
    return value if isinstance(value, str) and value else None


def with_request_id(message: str, request_id: str | None) -> str:
    """``message`` naming SLNG's request ID, for an error type with no field for it."""
    return f"{message} (SLNG request id: {request_id})" if request_id else message


def describe_close(code: object, reason: object) -> str:
    """A WebSocket close for an error message: its code, and the reason the server gave.

    The server writes the reason, not the caller, so it carries no reply text.
    It is cut short all the same, as an error message is no place for a long one.
    """
    text = f"close code {code}" if code is not None else "no close code"
    if isinstance(reason, str) and reason.strip():
        reason = reason.strip()
        if len(reason) > _CLOSE_REASON_MAX_LEN:
            reason = f"{reason[:_CLOSE_REASON_MAX_LEN]}..."
        text += f", reason: {reason}"
    return text


def gateway_status_error(
    message: str, frame: Mapping[str, object], *, request_id: str | None
) -> APIStatusError:
    """An error frame as the error to raise, with its status and SLNG's request ID.

    The frame's own ``slng_request_id`` wins over ``request_id``, which is the
    one the connection's `ready` gave. A frame with no status keeps
    ``APIStatusError``'s default.
    """
    status = extract_error_status(frame)
    request_id = request_id_from_frame(frame) or request_id
    if status is None:
        return APIStatusError(message, request_id=request_id)
    return APIStatusError(message, status_code=status, request_id=request_id)


def validate_model_identifier(model: str) -> str:
    if not isinstance(model, str) or not _MODEL_IDENTIFIER_RE.fullmatch(model):
        raise ValueError("model must be a provider/model identifier, for example 'deepgram/nova:3'")
    return model


def build_tts_init_payload(
    *,
    model: str,
    voice: str,
    sample_rate: int,
    encoding: str,
    language: str | None = None,
    speed: float | None = None,
    model_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Unmute TTS ``init`` message.

    ``encoding`` and ``sample_rate`` are always present in the payload: the
    plugin needs them to decode the audio it receives. ``language`` and
    ``speed`` are included only when the caller set them, so the model's
    catalog defaults apply otherwise. ``model_options`` are passed through
    verbatim into ``config`` and are applied last, so a key repeated there
    wins.

    ``language`` is also sent at the top level, where some models read it in
    preference to ``config``. Both copies carry the same value, including when
    ``model_options`` overrode it, so the two cannot disagree.
    """
    config: dict[str, Any] = {"encoding": encoding, "sample_rate": sample_rate}
    if language is not None:
        config["language"] = language
    if speed is not None:
        config["speed"] = speed
    config.update(dict(model_options or {}))
    payload: dict[str, Any] = {"type": "init", "model": model, "voice": voice, "config": config}
    if config.get("language") is not None:
        payload["language"] = config["language"]
    return payload


def build_stt_init_payload(
    *,
    model: str | None,
    language: str,
    sample_rate: int,
    encoding: str,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    enable_diarization: bool,
    enable_partial_transcripts: bool,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del model
    config: dict[str, Any] = {
        "language": language,
        "sample_rate": sample_rate,
        "encoding": "linear16" if encoding == "pcm_s16le" else encoding,
        "vad_threshold": vad_threshold,
        "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
        "vad_speech_pad_ms": vad_speech_pad_ms,
        "enable_diarization": enable_diarization,
        "enable_partials": enable_partial_transcripts,
        "enable_partial_transcripts": enable_partial_transcripts,
        **dict(model_options or {}),
    }
    if min_speakers is not None:
        config["min_speakers"] = min_speakers
    if max_speakers is not None:
        config["max_speakers"] = max_speakers
    partials = config.get(
        "enable_partials",
        config.get("enable_partial_transcripts", enable_partial_transcripts),
    )
    config["enable_partials"] = partials
    config["enable_partial_transcripts"] = partials
    return {"type": "init", "config": config}


def merge_init_payload(
    configured: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Lay ``current`` over a caller's own init message, one level deep.

    Every field of ``configured`` is kept unless ``current`` sets it too, in
    which case ``current`` wins, both at the top level and inside ``config``.
    """
    merged = dict(configured)
    configured_config = configured.get("config")
    current_config = current.get("config")
    if isinstance(current_config, Mapping):
        base = configured_config if isinstance(configured_config, Mapping) else {}
        merged["config"] = {**base, **current_config}
    merged.update({key: value for key, value in current.items() if key != "config"})
    return merged


def normalize_region_override(region_override: str | list[str] | None) -> str | None:
    if region_override is None:
        return None
    raw_values = (
        region_override.split(",")
        if isinstance(region_override, str)
        else [str(value) for value in region_override]
    )
    values = [value.strip().lower() for value in raw_values if value.strip()]
    return ", ".join(values) or None


def normalize_world_part_override(world_part_override: str | None) -> str | None:
    if world_part_override is None:
        return None
    value = world_part_override.strip().lower()
    return value or None


def _validate_external_tracking_id(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    if len(cleaned) > 128:
        raise ValueError(f"{name} must be 128 characters or fewer")
    if _EXTERNAL_TRACKING_ID_RE.search(cleaned):
        raise ValueError(f"{name} must not contain commas or control characters")
    return cleaned


def build_external_tracking_headers(
    *,
    external_agent_id: str | None,
    external_session_id: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    agent_id = _validate_external_tracking_id(
        external_agent_id,
        name="external_agent_id",
    )
    session_id = _validate_external_tracking_id(
        external_session_id,
        name="external_session_id",
    )
    if agent_id is not None:
        headers["X-SLNG-Agent-Id"] = agent_id
    if session_id is not None:
        headers["X-SLNG-Session-Id"] = session_id
    return headers


def _safe_error_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def http_status_from_exception(exc: BaseException | None) -> int | None:
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "status"):
            value = getattr(current, attr, None)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
                return value
        current = current.__cause__ or current.__context__
    return None


def is_non_retryable_client_error(exc: BaseException | None) -> bool:
    status = http_status_from_exception(exc)
    return (
        status is not None and 400 <= status < 500 and status not in _RETRYABLE_CLIENT_STATUS_CODES
    )


def is_payload_too_large(exc: BaseException | None) -> bool:
    # 413 is terminal for the whole chain: every candidate receives the same
    # oversized request body, so retrying or failing over cannot help.
    return http_status_from_exception(exc) == 413


def error_details(exc: BaseException | None) -> dict[str, Any]:
    if exc is None:
        return {"error_type": None, "error_code": None, "error_message": None}
    message = str(exc).strip() or type(exc).__name__
    return {
        "error_type": type(exc).__name__,
        "error_code": http_status_from_exception(exc) or _safe_error_code(exc),
        "error_message": message[:_ERROR_MESSAGE_MAX_LEN],
    }
