"""OCR postprocessing: deduplication, confidence filtering, text validation.

Sits between raw EasyOCR detections and label mapping to produce cleaner,
more reliable text labels for the tactile pipeline.
"""
from __future__ import annotations

import re
from dataclasses import replace
from math import hypot

import cv2
import numpy as np

from app.services.ocr import VALID_TEXT_PATTERN, OcrDetection


_MIN_CONFIDENCE = 0.15
_MAX_TEXT_LENGTH = 20
_MIN_TEXT_LENGTH = 1
_DUPLICATE_DISTANCE_PX = 30.0


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
        if not VALID_TEXT_PATTERN.match(text):
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


# OCR has no reliable degree sign either: "70\u00b0" comes back as "70?", "708",
# "702", "70o" or plain "70". The sign is recognised visually: the last glyph of
# the label is a small mark whose ink ends in the upper part of the digits.
_DEGREE_CANDIDATE = re.compile("^([0-9]{1,3})([?8o0O2\u00ba*\u00b0\"']?)$")
_DEGREE_MAX_HEIGHT_FRACTION = 0.6
_DEGREE_MAX_BOTTOM_FRACTION = 0.65
# A drawn degree sign is a ring about a third of the digit height; a speck is not.
_DEGREE_MIN_SIZE_FRACTION = 0.2
_DEGREE_MIN_SIZE_PX = 2
_DEGREE_BOX_PAD_PX = 2
# Pieces of one glyph broken by thresholding overlap horizontally.
_GLYPH_OVERLAP_PX = 1


def _ink_glyphs(gray: np.ndarray, bbox: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    x0 = max(min(xs) - _DEGREE_BOX_PAD_PX, 0)
    y0 = max(min(ys) - _DEGREE_BOX_PAD_PX, 0)
    x1 = min(max(xs) + _DEGREE_BOX_PAD_PX, gray.shape[1])
    y1 = min(max(ys) + _DEGREE_BOX_PAD_PX, gray.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    crop = gray[y0:y1, x0:x1]
    _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    pieces = sorted(
        (int(x), int(y), int(x + w), int(y + h))
        for x, y, w, h, _ in stats[1:count]
        if 0 < x and x + w < x1 - x0 and 0 < y and y + h < y1 - y0
    )
    glyphs: list[list[int]] = []
    for left, top, right, bottom in pieces:
        if glyphs and left < glyphs[-1][2] - _GLYPH_OVERLAP_PX:
            glyph = glyphs[-1]
            glyph[:] = [min(glyph[0], left), min(glyph[1], top), max(glyph[2], right), max(glyph[3], bottom)]
        else:
            glyphs.append([left, top, right, bottom])
    return [(left, top, right - left, bottom - top) for left, top, right, bottom in glyphs]


def _digits_before_degree_sign(gray: np.ndarray, bbox: list[tuple[int, int]]) -> int | None:
    """How many glyphs precede a trailing degree sign, or None without one."""
    glyphs = sorted(_ink_glyphs(gray, bbox))
    if len(glyphs) < 2:
        return None
    digits, last = glyphs[:-1], glyphs[-1]
    top = min(y for _, y, _, _ in digits)
    height = max(y + h for _, y, _, h in digits) - top
    raised = (
        height > 0
        and last[0] >= max(x + w for x, _, w, _ in digits) - 1
        and last[3] <= _DEGREE_MAX_HEIGHT_FRACTION * height
        and min(last[2], last[3]) >= max(_DEGREE_MIN_SIZE_PX, _DEGREE_MIN_SIZE_FRACTION * height)
        and last[1] + last[3] - top <= _DEGREE_MAX_BOTTOM_FRACTION * height
    )
    return len(digits) if raised else None


def restore_degrees(detections: list[OcrDetection], gray: np.ndarray) -> list[OcrDetection]:
    """Rewrite a number whose last glyph is a raised small mark as "<digits>\u00b0".

    The sign may have been read as an extra character or dropped; the glyph
    count of the ink says which.
    """
    restored: list[OcrDetection] = []
    for detection in detections:
        text = detection.text.strip()
        match = _DEGREE_CANDIDATE.match(text)
        glyph_digits = _digits_before_degree_sign(gray, detection.bbox) if match else None
        if match and glyph_digits is not None:
            digits = match.group(1)
            if not match.group(2) and len(digits) == glyph_digits + 1:
                digits = digits[:-1]
            if len(digits) == glyph_digits:
                detection = replace(detection, text=digits + "\u00b0")
        restored.append(detection)
    degrees = [d for d in restored if d.text.endswith("\u00b0")]
    return [d for d in restored if d in degrees or not any(_inside(d, label) for label in degrees)]


def _inside(detection: OcrDetection, label: OcrDetection) -> bool:
    cx, cy = _bbox_center(detection)
    xs = [point[0] for point in label.bbox]
    ys = [point[1] for point in label.bbox]
    return min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys)
