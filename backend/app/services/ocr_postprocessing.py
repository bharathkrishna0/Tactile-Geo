"""OCR postprocessing: deduplication, confidence filtering, text validation.

Sits between raw EasyOCR detections and label mapping to produce cleaner,
more reliable text labels for the tactile pipeline.
"""
from __future__ import annotations

import re
from dataclasses import replace
from math import hypot

import numpy as np

from app.services.ocr import OcrDetection


_MIN_CONFIDENCE = 0.15
_MAX_TEXT_LENGTH = 20
_MIN_TEXT_LENGTH = 1
_DUPLICATE_DISTANCE_PX = 30.0
_VALID_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9\s\.\,\;\:\!\?\-\+\/\=\(\)\[\]]*$")


def postprocess_detections(detections: list[OcrDetection]) -> list[OcrDetection]:
    """Clean raw OCR detections: filter, deduplicate, and validate."""
    filtered = _filter_by_confidence(detections)
    filtered = _filter_by_text_quality(filtered)
    deduplicated = _merge_duplicates(filtered)
    return deduplicated


def _filter_by_confidence(detections: list[OcrDetection]) -> list[OcrDetection]:
    return [d for d in detections if d.confidence >= _MIN_CONFIDENCE]


def _filter_by_text_quality(detections: list[OcrDetection]) -> list[OcrDetection]:
    result: list[OcrDetection] = []
    for detection in detections:
        text = detection.text.strip()
        if len(text) < _MIN_TEXT_LENGTH or len(text) > _MAX_TEXT_LENGTH:
            continue
        if not _VALID_TEXT_PATTERN.match(text):
            continue
        result.append(detection)
    return result


def _bbox_center(detection: OcrDetection) -> tuple[float, float]:
    xs = [p[0] for p in detection.bbox]
    ys = [p[1] for p in detection.bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _merge_duplicates(detections: list[OcrDetection]) -> list[OcrDetection]:
    """Merge detections with the same text that are spatially close."""
    if not detections:
        return []
    used: set[int] = set()
    merged: list[OcrDetection] = []
    for i, a in enumerate(detections):
        if i in used:
            continue
        group = [a]
        used.add(i)
        ca = _bbox_center(a)
        for j, b in enumerate(detections):
            if j in used:
                continue
            if a.text.strip().lower() != b.text.strip().lower():
                continue
            cb = _bbox_center(b)
            if hypot(ca[0] - cb[0], ca[1] - cb[1]) <= _DUPLICATE_DISTANCE_PX:
                group.append(b)
                used.add(j)
        best = max(group, key=lambda d: d.confidence)
        merged.append(best)
    return merged


# OCR has no square-root glyph and reads "\u221a17" as "V17", "Vi7" or "VT7". A
# radical is recognised visually: its vinculum is one unbroken ink run across
# the top of the label, whereas the arms of a letter V leave two short runs.
_RADICAL_CANDIDATE = re.compile(r"^[Vv]([0-9iIlT|OoSs]+)$")
_DIGIT_LOOKALIKES = str.maketrans({"i": "1", "I": "1", "l": "1", "T": "1", "|": "1", "O": "0", "o": "0", "S": "5", "s": "5"})
_VINCULUM_MIN_RUN_FRACTION = 0.35
_VINCULUM_BAND_FRACTION = 1 / 3
_INK_LEVEL = 128


def _longest_run(row: np.ndarray) -> int:
    best = current = 0
    for value in row:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _has_vinculum(gray: np.ndarray, bbox: list[tuple[int, int]]) -> bool:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    x0, y0 = max(min(xs), 0), max(min(ys), 0)
    x1, y1 = min(max(xs), gray.shape[1]), min(max(ys), gray.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 3:
        return False
    band = gray[y0:y0 + max(1, round((y1 - y0) * _VINCULUM_BAND_FRACTION)), x0:x1] < _INK_LEVEL
    return max(_longest_run(row) for row in band) >= (x1 - x0) * _VINCULUM_MIN_RUN_FRACTION


def restore_radicals(detections: list[OcrDetection], gray: np.ndarray) -> list[OcrDetection]:
    """Rewrite "V<digits>" detections as "\u221a<digits>" when a vinculum is drawn over them."""
    restored: list[OcrDetection] = []
    for detection in detections:
        match = _RADICAL_CANDIDATE.match(detection.text.strip())
        digits = match.group(1).translate(_DIGIT_LOOKALIKES) if match else ""
        if digits.isdigit() and _has_vinculum(gray, detection.bbox):
            detection = replace(detection, text="\u221a" + digits)
        restored.append(detection)
    return restored
