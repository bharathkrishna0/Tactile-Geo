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


class EasyOcrProvider:
    """EasyOCR implementation, deliberately isolated behind OcrProvider."""
    def __init__(self, languages: list[str] | None = None, reader=None) -> None:
        self.languages = languages or ["en"]
        self._reader = reader

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("EasyOCR is not installed. Install backend requirements before processing labels.") from error
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=False)
        detections = self._reader.readtext(image, detail=1, paragraph=False)
        return [
            OcrDetection(
                text=text.strip(),
                bbox=[(round(x), round(y)) for x, y in bbox],
                confidence=float(confidence),
            )
            for bbox, text, confidence in detections
            if text.strip()
        ]
