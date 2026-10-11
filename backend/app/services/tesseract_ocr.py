"""Tesseract as a second, independent reader of label crops.

EasyOCR stays the primary engine. Tesseract reads the same small crops (a lone
vertex letter, a short number) so the two readings can be fused; it is optional,
and when the binary is missing the pipeline runs on EasyOCR alone.
"""
from __future__ import annotations

import shutil

import cv2
import numpy as np

from app.services.ocr_fusion import OcrReading

try:
    import pytesseract
except ImportError:  # optional dependency
    pytesseract = None

TESSERACT_BORDER_PX = 12
TESSERACT_TIMEOUT_S = 2


def tesseract_available() -> bool:
    return pytesseract is not None and shutil.which("tesseract") is not None


class TesseractCropReader:
    provider = "tesseract"

    def __init__(self, allowlist: str | None = None) -> None:
        self.allowlist = allowlist

    def read(self, crop: np.ndarray, single_char: bool) -> list[OcrReading]:
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        binary = cv2.copyMakeBorder(binary, *(TESSERACT_BORDER_PX,) * 4, cv2.BORDER_CONSTANT, value=255)
        config = f"--psm {10 if single_char else 7}"
        if self.allowlist:
            config += f" -c tessedit_char_whitelist={self.allowlist}"
        try:
            data = pytesseract.image_to_data(
                binary, config=config, output_type=pytesseract.Output.DICT, timeout=TESSERACT_TIMEOUT_S,
            )
        except RuntimeError:
            return []
        words = [(text.strip(), float(conf)) for text, conf in zip(data["text"], data["conf"]) if text.strip()]
        if not words:
            return []
        text = " ".join(word for word, _ in words)
        confidence = min(conf for _, conf in words) / 100.0
        return [OcrReading(text=text, confidence=max(0.0, confidence), provider=self.provider)]
