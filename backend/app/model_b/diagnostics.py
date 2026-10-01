"""Development-only diagnostics for the Model B provider boundary.

Exists because of one specific failure mode: a structured-output request that
returns HTTP 200 and a perfectly well-formed envelope whose assistant content is
not JSON. Every layer above the transport behaves correctly in that situation —
the client extracts text, the parser rejects it, the job fails with a typed
error — so nothing in the stack is obviously at fault and the only way to find
out what the model actually said is to look at it.

**Off by default.** Nothing is captured unless `MODEL_B_DIAGNOSTICS` is
explicitly truthy, and when it is off the cost is one boolean read.

**Redaction is unconditional, not gated on the flag.** A diagnostic that leaked a
credential when the flag was on would be a diagnostic nobody could safely turn
on, which is the same as no diagnostic. So redaction runs on every path, and the
captured content is bounded.

What is deliberately never captured:

* the API key, in any form, including if the model echoes it back;
* the `Authorization` header value;
* the request image, or its base64 encoding;
* unbounded model output.

Header *names* and sizes are reported, values are not, because "was the token
actually attached" is a real question and "what was the token" is not.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

#: Captured model content is truncated to this many characters. Long enough to
#: see a prose preamble or a truncated JSON prefix, short enough that a runaway
#: response cannot flood a terminal or a log.
MAX_CONTENT_CHARS = 2000

# Credential shapes. Applied to any captured text, including model output,
# because a model can be induced to echo request material and the one thing
# this module must never do is forward that.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # `Authorization: Bearer <token>` in any casing.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <redacted>"),
    # OpenRouter keys, and any `*_API_KEY` / `*_KEY` assignment.
    (re.compile(r"sk-[A-Za-z0-9\-_]{8,}"), "<redacted-key>"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{10,}"), "<redacted-key>"),
    (
        re.compile(r"(?i)\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\b\s*[=:]\s*\S+"),
        "<redacted-assignment>",
    ),
    # A data URL carries the worksheet itself; keep the type, drop the payload.
    (re.compile(r"(data:[a-z0-9.+/\-]+;base64,)[A-Za-z0-9+/=]{16,}", re.IGNORECASE),
     r"\1<redacted-image>"),
)


def enabled() -> bool:
    """Whether diagnostics are switched on for this process."""
    return (os.getenv("MODEL_B_DIAGNOSTICS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "debug",
    }


def redact(text: str) -> str:
    """Strip credential-shaped material from text bound for a terminal or log.

    Applied unconditionally. This is the reason the module is safe to enable.
    """
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def summarize_request(
    body: dict[str, Any], headers: dict[str, str], endpoint: str
) -> str:
    """Describe what was sent, without any of its payload or secrets.

    Reports sizes and shapes, because the question being asked is almost always
    "was the thing included at all", which a length answers without disclosing
    content.
    """
    lines = [f"endpoint          : {endpoint}"]

    messages = body.get("messages") or []
    lines.append(f"messages          : {len(messages)}")
    for index, message in enumerate(messages):
        role = message.get("role", "?") if isinstance(message, dict) else "?"
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            lines.append(f"  [{index}] role={role} text_chars={len(content)}")
            continue
        if isinstance(content, list):
            kinds = [
                part.get("type", "?")
                for part in content
                if isinstance(part, dict)
            ]
            text_chars = sum(
                len(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            image_parts = sum(1 for kind in kinds if kind == "image_url")
            lines.append(
                f"  [{index}] role={role} parts={kinds} "
                f"text_chars={text_chars} image_parts={image_parts}"
            )
            continue
        lines.append(f"  [{index}] role={role} content_type={type(content).__name__}")

    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        block = response_format.get("json_schema")
        lines.append(f"response_format   : type={kind}")
        if isinstance(block, dict):
            schema = block.get("schema")
            lines.append(
                f"  name={block.get('name')!r} strict={block.get('strict')} "
                f"schema_chars={len(json.dumps(schema)) if schema is not None else 0}"
            )
    else:
        lines.append(f"response_format   : {response_format!r}")

    for key in ("temperature", "max_tokens", "stream"):
        if key in body:
            lines.append(f"{key:17s}: {body[key]!r}")

    lines.append(f"model             : {body.get('model')!r}")
    # Names only. Never values: `Authorization` is in this list.
    lines.append(f"header names      : {sorted(headers)}")
    lines.append(
        f"authorization set : {any(k.lower() == 'authorization' for k in headers)} "
        "(value withheld)"
    )
    return redact("\n".join(lines))


def summarize_response(
    *,
    status: Any,
    raw: bytes,
    envelope: dict[str, Any] | None,
    content: str,
    content_source: str,
) -> str:
    """Describe what came back, including the model's own words, redacted."""
    lines: list[str] = []
    lines.append(f"http status       : {status}")
    lines.append(f"response bytes    : {len(raw)}")

    if envelope is None:
        lines.append("envelope          : not JSON (top-level body is not an object)")
        return redact("\n".join(lines))

    lines.append(f"envelope keys     : {sorted(envelope)}")
    resolved = envelope.get("model")
    lines.append(f"resolved model    : {resolved!r}")
    lines.append(f"upstream provider : {envelope.get('provider')!r}")
    lines.append(f"request id        : {envelope.get('id')!r}")
    lines.append(f"error envelope    : {'error' in envelope}")

    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        lines.append("choices           : absent or empty")
        return redact("\n".join(lines))

    lines.append(f"choices count     : {len(choices)}")
    first = choices[0] if isinstance(choices[0], dict) else {}
    lines.append(f"choice keys       : {sorted(first)}")
    lines.append(f"finish_reason     : {first.get('finish_reason')!r}")

    message = first.get("message")
    if isinstance(message, dict):
        lines.append(f"message keys      : {sorted(message)}")
        raw_content = message.get("content")
        lines.append(f"content type      : {type(raw_content).__name__}")
        if isinstance(raw_content, list):
            part_types = [
                part.get("type", "?")
                for part in raw_content
                if isinstance(part, dict)
            ]
            lines.append(f"content parts     : {part_types}")
        # Structured-output implementations that route through a reasoning
        # wrapper can return the answer somewhere other than `content`.
        extra = sorted(set(message) - {"role", "content", "refusal"})
        if extra:
            lines.append(f"other message keys: {extra}")
        refusal = message.get("refusal")
        if refusal:
            lines.append(f"refusal           : {str(refusal)[:400]!r}")
    else:
        lines.append(f"message           : {type(message).__name__}")

    usage = envelope.get("usage")
    if isinstance(usage, dict):
        lines.append(f"usage             : {usage}")

    lines.append(f"extracted from    : {content_source}")
    lines.append(f"content chars     : {len(content)}")
    lines.append(f"content is JSON   : {_looks_like_json(content)}")
    truncated = content[:MAX_CONTENT_CHARS]
    suffix = "\n  ...[truncated]" if len(content) > MAX_CONTENT_CHARS else ""
    lines.append("--- raw model content (redacted) ---")
    lines.append(truncated + suffix)
    return redact("\n".join(lines))


def _looks_like_json(text: str) -> bool:
    """Whether the content parses as a JSON object.

    This is a *diagnostic* helper and deliberately stricter than the production
    parser: it answers "was this JSON at all", which is the question being asked
    when a parse failed. The real parser remains the only thing that decides
    whether content is acceptable.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return isinstance(json.loads(stripped), dict)
        except json.JSONDecodeError:
            return False
    return False
