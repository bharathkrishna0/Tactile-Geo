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
    def __init__(self, languages: list[str] | None = None, reader=None) -> None:
        self.languages = languages or ["en"]
        self._reader = reader

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
        return [
            OcrDetection(
                text=text.strip(),
                bbox=[(round(x / scale), round(y / scale)) for x, y in bbox],
                confidence=float(confidence),
            )
            for bbox, text, confidence in detections
            if text.strip()
        ]
