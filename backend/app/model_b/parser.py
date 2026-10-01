"""Recover a JSON document from the model's raw text.

`response_json_schema` makes well-formed output the overwhelmingly likely case,
but not a guaranteed one: a truncated response, a reasoning model that wrapped
its answer in a fence, or a safety-stopped candidate can all produce something
that is not directly parseable. This module degrades that gracefully instead of
reporting a bare JSONDecodeError to a teacher.
"""

from __future__ import annotations

import json
import re

from .errors import ModelBMalformedResponse, ModelBValidationError

# ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict:
    """Parse `text` into a JSON object, tolerating fences and stray prose.

    Order of attempts:
      1. Whole string as JSON.
      2. Contents of the first markdown fence.
      3. The outermost balanced brace span.

    Raises:
        ModelBValidationError: the string is empty or is not JSON.
        ModelBMalformedResponse: JSON was found but the object was cut short,
            which is a truncation signal the caller surfaces rather than hides.
    """
    stripped = text.strip()
    if not stripped:
        raise ModelBValidationError("The model provider returned an empty response body.")

    direct = _try_load(stripped)
    if direct is not None:
        return direct

    for match in _FENCE_RE.finditer(stripped):
        candidate = _try_load(match.group("body"))
        if candidate is not None:
            return candidate

    span = _outermost_object_span(stripped)
    if span is not None:
        start, end = span
        candidate = _try_load(stripped[start:end])
        if candidate is not None:
            return candidate

    # An object that was opened and never closed is a budget problem, and the
    # teacher deserves to be told that rather than shown a generic parse error.
    # Checked even when no balanced span exists, since a truncated response has
    # no closing brace to find.
    if _looks_truncated(stripped):
        raise ModelBMalformedResponse(
            "The model response was cut off before the JSON document closed."
        )

    raise ModelBValidationError(
        "The model response did not contain a readable JSON object."
    )


def _try_load(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _outermost_object_span(text: str) -> tuple[int, int] | None:
    """Brace-matched span of the first top-level `{...}`, string-aware.

    String-awareness matters: a brace inside an evidence string such as
    "{90 deg}" would otherwise desynchronise the depth counter and produce a
    span that ends in the wrong place.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    return None


def _looks_truncated(fragment: str) -> bool:
    """True when the text opens an object it never closes."""
    depth = 0
    in_string = False
    escaped = False
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return depth > 0
