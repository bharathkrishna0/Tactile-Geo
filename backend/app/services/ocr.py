import math
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OcrDetection:
    text: str
    bbox: list[tuple[int, int]]
    confidence: float


class OcrProvider(Protocol):
    def detect(self, image: np.ndarray) -> list[OcrDetection]: ...


# EasyOCR's detector needs glyphs roughly 30px tall to be reliable. Worksheet
# labels are often ~10px, and at that size the detector returns nothing at all, so
# the image is upscaled before recognition and the boxes are scaled back afterwards.
OCR_MIN_SHORT_SIDE_PX = 960
OCR_MAX_LONG_SIDE_PX = 2400

_READER_LOCK = threading.Lock()
_READER_CACHE: dict[tuple[str, ...], object] = {}


def ocr_scale_factor(height: int, width: int) -> int:
    """Integer upscale factor needed to make OCR labels legible."""
    short_side = min(height, width)
    if short_side <= 0 or short_side >= OCR_MIN_SHORT_SIDE_PX:
        return 1
    scale = math.ceil(OCR_MIN_SHORT_SIDE_PX / short_side)
    long_side = max(height, width)
    while scale > 1 and long_side * scale > OCR_MAX_LONG_SIDE_PX:
        scale -= 1
    return max(1, scale)


def _get_reader(languages: list[str]):
    import easyocr

    key = tuple(languages)
    with _READER_LOCK:
        reader = _READER_CACHE.get(key)
        if reader is None:
            # Loading the model is expensive; share one reader per language set
            # across requests instead of reloading it on every pipeline run.
            reader = easyocr.Reader(list(languages), gpu=False)
            _READER_CACHE[key] = reader
        return reader


class EasyOcrProvider:
    """EasyOCR implementation, deliberately isolated behind OcrProvider."""
    def __init__(self, languages: list[str] | None = None, reader=None, recover_glyphs: bool | None = None) -> None:
        self.languages = languages or ["en"]
        self._reader = reader
        # Glyph recovery needs EasyOCR's ``recognize``; an injected reader only
        # promises ``readtext``, so it is opted in explicitly.
        self.recover_glyphs = reader is None if recover_glyphs is None else recover_glyphs

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        try:
            import cv2
            import easyocr  # noqa: F401
        except ImportError as error:
            raise RuntimeError("EasyOCR is not installed. Install backend requirements before processing labels.") from error
        reader = self._reader or _get_reader(self.languages)

        scale = ocr_scale_factor(*image.shape[:2])
        working = image
        if scale > 1:
            working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        detections = reader.readtext(working, detail=1, paragraph=False)
        found = [
            OcrDetection(
                text=text.strip(),
                bbox=[(round(x / scale), round(y / scale)) for x, y in bbox],
                confidence=float(confidence),
            )
            for bbox, text, confidence in detections
            if text.strip()
        ]
        if not self.recover_glyphs:
            return found
        return found + recognize_isolated_glyphs(reader, image, found)


# EasyOCR's text detector rarely boxes a lone one- or two-character label (a
# vertex letter, a single digit). Such labels are recovered from the ink
# directly: small, unfilled, letter-sized connected components outside every
# detected text box are grouped into short runs, each run is cropped, enlarged
# and recognised against a restricted alphanumeric character set.
GLYPH_MIN_HEIGHT_PX = 8
GLYPH_MAX_HEIGHT_PX = 36
GLYPH_MIN_WIDTH_PX = 3
GLYPH_MIN_PIXELS = 12
GLYPH_MAX_FILL = 0.75
GLYPH_MAX_ASPECT = 2.0
GLYPH_GROUP_GAP = 0.6
GLYPH_MAX_CHARS = 3
GLYPH_CROP_HEIGHT_PX = 64
GLYPH_MIN_CONFIDENCE = 0.75
GLYPH_MIN_THICKNESS = 0.3
GLYPH_CLEARANCE_PX = 2
MAX_GLYPH_CANDIDATES = 80
GLYPH_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _box_bounds(bbox: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def glyph_candidates(gray: np.ndarray, existing: list[OcrDetection]) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h) of short ink runs that look like an undetected label."""
    import cv2

    binary = cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    large = np.zeros(count, dtype=bool)
    large[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) > GLYPH_MAX_HEIGHT_PX
    taken = [_box_bounds(d.bbox) for d in existing]
    glyphs = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not (GLYPH_MIN_HEIGHT_PX <= h <= GLYPH_MAX_HEIGHT_PX and GLYPH_MIN_WIDTH_PX <= w <= GLYPH_MAX_HEIGHT_PX):
            continue
        if area < GLYPH_MIN_PIXELS or area > GLYPH_MAX_FILL * w * h or w > GLYPH_MAX_ASPECT * h:
            continue
        # Thin strokes are dashes, ticks and hatching far more often than an "I" or "1".
        if min(w, h) < GLYPH_MIN_THICKNESS * max(w, h):
            continue
        # A piece almost touching a long stroke is a broken part of the drawing.
        c = GLYPH_CLEARANCE_PX
        if large[labels[max(0, y - c):y + h + c, max(0, x - c):x + w + c]].any():
            continue
        pad = 3
        if any(x < bx1 + pad and x + w > bx0 - pad and y < by1 + pad and y + h > by0 - pad for bx0, by0, bx1, by1 in taken):
            continue
        glyphs.append([x, y, x + w, y + h])
    glyphs.sort()
    groups: list[list[int]] = []
    for x0, y0, x1, y1 in glyphs:
        for group in groups:
            height = max(group[3] - group[1], y1 - y0)
            v_overlap = min(group[3], y1) - max(group[1], y0)
            gap = x0 - group[2]
            if v_overlap >= 0.5 * min(group[3] - group[1], y1 - y0) and -height < gap <= GLYPH_GROUP_GAP * height:
                merged = [min(group[0], x0), min(group[1], y0), max(group[2], x1), max(group[3], y1)]
                if merged[2] - merged[0] <= GLYPH_MAX_CHARS * (merged[3] - merged[1]):
                    group[:] = merged
                    break
        else:
            groups.append([x0, y0, x1, y1])
    boxes = [(g[0], g[1], g[2] - g[0], g[3] - g[1]) for g in groups]
    return boxes[:MAX_GLYPH_CANDIDATES]


def recognize_isolated_glyphs(reader, image: np.ndarray, existing: list[OcrDetection]) -> list[OcrDetection]:
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found: list[OcrDetection] = []
    height, width = gray.shape[:2]
    for x, y, w, h in glyph_candidates(gray, existing):
        pad = max(4, h // 2)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        crop = gray[y0:y1, x0:x1]
        factor = GLYPH_CROP_HEIGHT_PX / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        for _, text, confidence in reader.recognize(crop, allowlist=GLYPH_ALLOWLIST, detail=1):
            text = text.strip()
            if 1 <= len(text) <= GLYPH_MAX_CHARS and float(confidence) >= GLYPH_MIN_CONFIDENCE:
                found.append(OcrDetection(
                    text=text,
                    bbox=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                    confidence=float(confidence),
                ))
    return _drop_dash_runs(found)


DASH_RUN_MIN = 3


def _drop_dash_runs(detections: list[OcrDetection]) -> list[OcrDetection]:
    """Remove runs of identical glyphs spaced along a line: the dashes of a dashed line."""
    centres = [((b[0][0] + b[2][0]) / 2, (b[0][1] + b[2][1]) / 2, b[2][1] - b[0][1]) for b in (d.bbox for d in detections)]
    dashed: set[int] = set()
    for i, (xi, yi, hi) in enumerate(centres):
        same = [
            j for j, (xj, yj, _) in enumerate(centres)
            if j != i and detections[j].text == detections[i].text and math.dist((xi, yi), (xj, yj)) <= 3 * hi
        ]
        for a in same:
            for b in same:
                if a < b:
                    va = (centres[a][0] - xi, centres[a][1] - yi)
                    vb = (centres[b][0] - xi, centres[b][1] - yi)
                    if va[0] * vb[0] + va[1] * vb[1] < -0.9 * math.hypot(*va) * math.hypot(*vb):
                        dashed.update({i, a, b})
    return [d for k, d in enumerate(detections) if k not in dashed]
