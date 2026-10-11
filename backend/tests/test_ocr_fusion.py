"""Second OCR engine fusion and text-first glyph candidates."""
import cv2
import numpy as np

from app.services.ocr import EasyOcrProvider, OcrDetection, glyph_candidates, recognize_isolated_glyphs
from app.services.ocr_fusion import OcrReading, fuse_short_label, prefers_letters


def _r(text, confidence, provider):
    return OcrReading(text=text, confidence=confidence, provider=provider)


def test_agreeing_engines_accept_a_label_neither_would_alone():
    fused = fuse_short_label([_r("Q", 0.6, "easyocr"), _r("Q", 0.7, "tesseract")])
    assert fused == OcrReading("Q", 0.7, "easyocr+tesseract")


def test_one_unconfirmed_low_confidence_reading_is_rejected():
    assert fuse_short_label([_r("C", 0.6, "easyocr"), _r("Cc", 0.0, "tesseract")]) is None


def test_one_confident_engine_is_enough():
    assert fuse_short_label([_r("P", 0.95, "easyocr")]).text == "P"
    assert fuse_short_label([_r("8", 0.13, "easyocr"), _r("P", 0.9, "tesseract")]).text == "P"


def test_implausible_strings_are_rejected_however_confident():
    for text in ("TA", "01", "tH", "Ae"):
        assert fuse_short_label([_r(text, 0.99, "easyocr"), _r(text, 0.99, "tesseract")]) is None


def test_fusion_never_returns_two_labels_for_one_crop():
    fused = fuse_short_label([_r("R", 0.95, "easyocr"), _r("F", 0.9, "tesseract")])
    assert fused is not None and fused.text in {"R", "F"} and fused.provider in {"easyocr", "tesseract"}


def test_lookalikes_are_resolved_by_page_context():
    readings = [_r("0", 1.0, "easyocr"), _r("O", 0.0, "tesseract")]
    assert fuse_short_label(readings, prefer_letters=True).text == "O"
    assert fuse_short_label(readings, prefer_letters=False).text == "0"
    assert fuse_short_label(readings).text == "0"
    one = [_r("1", 0.93, "easyocr"), _r("i", 0.96, "tesseract")]
    assert fuse_short_label(one, prefer_letters=False).text == "1"
    assert fuse_short_label(one, prefer_letters=True).text == "I"


def test_page_context_counts_letters_and_digits():
    assert prefers_letters(["A", "B", "3"]) is True
    assert prefers_letters(["1", "2", "-3", "x"]) is False
    assert prefers_letters(["A", "1"]) is None
    assert prefers_letters([]) is None


def test_dashed_line_pieces_are_not_glyph_candidates():
    gray = np.full((300, 500), 255, dtype=np.uint8)
    for y in range(20, 280, 24):
        cv2.rectangle(gray, (100, y), (104, y + 14), 0, -1)
        cv2.circle(gray, (102, y + 7), 1, 255, -1)
    cv2.putText(gray, "B", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    boxes = glyph_candidates(gray, [])
    assert len(boxes) == 1 and 295 <= boxes[0][0] <= 305


def test_tick_labels_along_an_axis_are_kept():
    gray = np.full((200, 500), 255, dtype=np.uint8)
    for i, text in enumerate("1234"):
        cv2.putText(gray, text, (60 + 90 * i, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    assert len(glyph_candidates(gray, [])) == 4


class _Recognizer:
    def __init__(self, text, confidence):
        self.text, self.confidence = text, confidence

    def recognize(self, crop, allowlist=None, detail=1):
        return [(None, self.text, self.confidence)]


class _SecondReader:
    provider = "tesseract"

    def __init__(self, text, confidence):
        self.text, self.confidence = text, confidence

    def read(self, crop, single_char):
        return [OcrReading(self.text, self.confidence, self.provider)]


def _letter_image():
    gray = np.full((300, 500), 255, dtype=np.uint8)
    cv2.putText(gray, "Q", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return gray


def test_glyph_recovery_records_both_providers_when_they_agree():
    found = recognize_isolated_glyphs(_Recognizer("Q", 0.4), _letter_image(), [], _SecondReader("Q", 0.8))
    assert [(d.text, d.provider) for d in found] == [("Q", "easyocr+tesseract")]


def test_glyph_recovery_without_a_second_reader_keeps_easyocr_bar():
    assert recognize_isolated_glyphs(_Recognizer("Q", 0.4), _letter_image(), []) == []
    found = recognize_isolated_glyphs(_Recognizer("Q", 0.95), _letter_image(), [])
    assert [(d.text, d.provider) for d in found] == [("Q", "easyocr")]


def test_glyph_recovery_uses_page_letters_to_read_o():
    page = [OcrDetection("A", [(10, 10), (20, 10), (20, 22), (10, 22)], 0.9),
            OcrDetection("B", [(400, 10), (410, 10), (410, 22), (400, 22)], 0.9)]
    found = recognize_isolated_glyphs(_Recognizer("0", 1.0), _letter_image(), page, _SecondReader("O", 0.1))
    assert [d.text for d in found] == ["O"]


def test_injected_reader_does_not_pick_up_tesseract():
    assert EasyOcrProvider(reader=object()).second_reader is None


def test_detection_provider_defaults_to_easyocr():
    assert OcrDetection("A", [(0, 0)], 0.9).provider == "easyocr"
