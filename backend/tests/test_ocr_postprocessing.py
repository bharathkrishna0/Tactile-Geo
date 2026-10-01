"""Tests for Milestone 2: OCR preprocessing/postprocessing."""
import cv2

from app.services.ocr import EasyOcrProvider
from app.services.ocr_postprocessing import postprocess_detections
from app.services.pipeline import build_full_analysis, build_preview
from app.services.braille import LouisBrailleTranslator


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class NoisyReader:
    """Returns low-confidence noise plus valid labels."""
    def readtext(self, image, detail, paragraph):
        return [
            ([[0, 0], [8, 0], [8, 8], [0, 8]], "IIII", 0.05),
            ([[5, 5], [12, 5], [12, 12], [5, 12]], "llll", 0.02),
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class DuplicateReader:
    """Returns the same text in overlapping positions."""
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[115, 10], [130, 10], [130, 30], [115, 30]], "A", 0.90),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.80),
        ]


class InvalidTextReader:
    """Returns invalid text content."""
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 202]], "~~~", 0.95),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "This is a very long label that exceeds the maximum", 0.99),
        ]


class FakeLouis:
    def translateString(self, tables, text):
        return {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉"}.get(text, text)


def _detection(bbox, text, confidence):
    from app.services.ocr import OcrDetection
    return OcrDetection(text=text, bbox=bbox, confidence=confidence)


# --- basic filtering ---

def test_postprocess_filters_low_confidence():
    detections = [
        _detection([(0, 0), (8, 0), (8, 8), (0, 8)], "noise", 0.05),
        _detection([(10, 10), (20, 10), (20, 20), (10, 20)], "A", 0.95),
    ]
    result = postprocess_detections(detections)
    assert [d.text for d in result] == ["A"]


def test_postprocess_filters_bad_chars():
    detections = [
        _detection([(0, 0), (8, 0), (8, 8), (0, 8)], "~~~", 0.95),
        _detection([(10, 10), (20, 10), (20, 20), (10, 20)], "A", 0.95),
    ]
    result = postprocess_detections(detections)
    assert [d.text for d in result] == ["A"]


def test_postprocess_filters_long_text():
    detections = [
        _detection([(0, 0), (8, 0), (8, 8), (0, 8)], "X" * 30, 0.95),
        _detection([(10, 10), (20, 10), (20, 20), (10, 20)], "A", 0.95),
    ]
    result = postprocess_detections(detections)
    assert [d.text for d in result] == ["A"]


def test_postprocess_preserves_valid():
    detections = [
        _detection([(0, 0), (8, 0), (8, 8), (0, 8)], "A", 0.98),
        _detection([(10, 10), (20, 10), (20, 20), (10, 20)], "BC", 0.90),
    ]
    result = postprocess_detections(detections)
    assert [d.text for d in result] == ["A", "BC"]


# --- duplicate merging ---

def test_postprocess_merges_duplicates():
    detections = [
        _detection([(112, 8), (128, 8), (128, 28), (112, 28)], "A", 0.98),
        _detection([(115, 10), (130, 10), (130, 30), (115, 30)], "A", 0.90),
        _detection([(8, 202), (25, 202), (25, 222), (8, 222)], "B", 0.80),
    ]
    result = postprocess_detections(detections)
    # Same text, overlapping position → merge into one (keep the higher confidence)
    assert len(result) == 2
    assert result[0].text == "A"
    assert result[0].confidence == 0.98


def test_postprocess_keeps_distant_same_text():
    detections = [
        _detection([(10, 10), (20, 10), (20, 20), (10, 20)], "A", 0.9),
        _detection([(300, 300), (310, 300), (310, 310), (300, 310)], "A", 0.8),
    ]
    result = postprocess_detections(detections)
    assert len(result) == 2


# --- pipeline integration ---

def test_pipeline_filters_noisy_ocr(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    result = build_full_analysis(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=NoisyReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    # Only the valid A,B,C labels survive; noise removed
    assert len(result.labels) == 3


def test_pipeline_merges_duplicate_ocr(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    result = build_full_analysis(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=DuplicateReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    # A detected twice in same spot → merged to one; only B remains
    assert len(result.labels) == 2


def test_pipeline_filters_invalid_text(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    result = build_full_analysis(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=InvalidTextReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    # Only "A" is valid; "~~~" and the too-long text are removed
    assert len(result.labels) == 1
    assert result.labels[0]["text"] == "A"


def test_build_preview_uses_postprocessing(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    _, _, labels = build_preview(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=NoisyReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    assert len(labels) == 3
