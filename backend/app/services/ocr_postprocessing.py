"""OCR postprocessing: deduplication, confidence filtering, text validation.

Sits between raw EasyOCR detections and label mapping to produce cleaner,
more reliable text labels for the tactile pipeline.
"""
from __future__ import annotations

import re
from math import hypot

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
