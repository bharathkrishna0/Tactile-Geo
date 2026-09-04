import cv2

from app.services.diagram_analysis import analyze_diagram
from app.services.image_quality import assess_image_quality
from app.services.ocr import EasyOcrProvider
from app.services.braille import LouisBrailleTranslator
from app.services.pipeline import build_full_analysis, build_preview
from app.services.tactile_qa import run_tactile_qa


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class FakeLouis:
    def translateString(self, tables, text):
        return {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉"}[text]


def test_full_analysis_returns_all_stages(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    result = build_full_analysis(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    assert result.preview_svg.startswith("<svg")
    assert result.quality_report.passes_gate is True
    assert result.semantic_geometry.element_count > 0
    assert result.simplified_geometry.elements
    assert result.qa_report.passes is True
    assert len(result.labels) == 3


def test_build_preview_still_works_backward_compatible(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    svg, shapes, labels = build_preview(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    assert svg.startswith("<svg")
    assert shapes
    assert len(labels) == 3


def test_full_analysis_produces_tactile_text_labels(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    result = build_full_analysis(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    text_labels = [e for e in result.semantic_geometry.elements if e.type.value == "text_label"]
    assert len(text_labels) == 3
