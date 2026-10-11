"""Fuse readings of the same text region from independent OCR engines.

Each engine reads the same crop. Readings are compared, not concatenated: two
engines agreeing on a plausible label is strong evidence; a single engine must
clear its own confidence bar; implausible strings (``"TA"``, ``"01"``) are
rejected however confident the engine is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrReading:
    text: str
    confidence: float
    provider: str


# A short label recovered from loose glyphs: a vertex letter, a letter with a
# subscript digit, or a small integer (an axis tick, a side length).
SHORT_LABEL = re.compile(r"^(?:[A-Za-z]|[A-Z][0-9]|-?(?:0|[1-9][0-9]{0,2}))$")

# Per-engine bar for accepting an unconfirmed reading.
SOLO_MIN_CONFIDENCE = {"easyocr": 0.75, "tesseract": 0.85}
AGREEMENT_MIN_CONFIDENCE = 0.2


# Single glyphs the engines confuse; each maps to the digit it resembles.
LOOKALIKE_DIGIT = {
    "0": "0", "O": "0", "o": "0",
    "1": "1", "I": "1", "l": "1", "i": "1",
    "5": "5", "S": "5", "s": "5",
    "8": "8", "B": "8",
    "2": "2", "Z": "2", "z": "2",
}
LETTER_FORM = {"o": "O", "i": "I", "l": "I"}


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _group(text: str) -> str:
    key = _key(text)
    return LOOKALIKE_DIGIT.get(key, key) if len(key) == 1 else key


def _resolve_lookalike(group: list[OcrReading], prefer_letters: bool | None) -> str:
    texts = [_key(r.text) for r in sorted(group, key=lambda r: -r.confidence)]
    if prefer_letters is None or len(set(texts)) == 1:
        return texts[0]
    pick = next((t for t in texts if t.isalpha() == prefer_letters), texts[0])
    return LETTER_FORM.get(pick, pick) if pick.isalpha() else pick


def prefers_letters(texts: list[str]) -> bool | None:
    """Whether the short labels already read on a page are mostly letters, digits, or neither."""
    letters = sum(1 for t in texts if _key(t).isalpha() and len(_key(t)) <= 3)
    digits = sum(1 for t in texts if _key(t).lstrip("-").isdigit())
    if letters == digits:
        return None
    return letters > digits


def plausible_short_label(text: str) -> bool:
    return bool(SHORT_LABEL.match(_key(text)))


def fuse_short_label(readings: list[OcrReading], prefer_letters: bool | None = None) -> OcrReading | None:
    """Best reading of a short label crop, or None when no engine is convincing.

    Engines that disagree only on a lookalike pair (``O``/``0``, ``I``/``1``)
    agree on the glyph; the page context (``prefer_letters``) picks the form.
    """
    candidates = [r for r in readings if plausible_short_label(r.text)]
    if not candidates:
        return None
    by_text: dict[str, list[OcrReading]] = {}
    for reading in candidates:
        by_text.setdefault(_group(reading.text), []).append(reading)
    agreed = [
        group for group in by_text.values()
        if len({r.provider for r in group}) > 1 and max(r.confidence for r in group) >= AGREEMENT_MIN_CONFIDENCE
    ]
    if agreed:
        group = max(agreed, key=lambda g: max(r.confidence for r in g))
        providers = "+".join(sorted({r.provider for r in group}))
        text = _resolve_lookalike(group, prefer_letters)
        return OcrReading(text=text, confidence=max(r.confidence for r in group), provider=providers)
    solo = [r for r in candidates if r.confidence >= SOLO_MIN_CONFIDENCE.get(r.provider, 1.0)]
    if not solo:
        return None
    best = max(solo, key=lambda r: r.confidence)
    text = _resolve_lookalike(by_text[_group(best.text)], prefer_letters)
    return OcrReading(text=text, confidence=best.confidence, provider=best.provider)
