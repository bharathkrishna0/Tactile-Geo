"""Content-addressed cache key for Model B results.

The same image analysed under the same prompt, response schema and model
configuration gets the same advisory answer, so a repeat request (a teacher
re-opening a worksheet, or a second teacher uploading the same page) reuses the
stored result instead of paying for another provider call. Changing the prompt,
the schema version or the configured model changes the key.
"""
from __future__ import annotations

import hashlib

from app.model_b.prompts import build_prompt
from app.model_b.schema import SCHEMA_VERSION
from app.model_b.service import ModelBSettings

_PROMPT_FINGERPRINT = hashlib.sha256(build_prompt().encode()).hexdigest()[:16]


def model_b_cache_key(image_bytes: bytes, settings: ModelBSettings) -> str:
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{image_digest}:{SCHEMA_VERSION}:{_PROMPT_FINGERPRINT}:{settings.provider}:{settings.model}"
