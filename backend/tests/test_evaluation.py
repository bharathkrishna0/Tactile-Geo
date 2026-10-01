"""Tests for Milestone 5: Evaluation framework.

Metrics are computed ONLY from known ground truth over real fixtures. No
numbers are fabricated; the test asserts metric *relationships* and that
labeled images produce reproducible results.
"""
from app.services.evaluation import (
    GroundTruth,
    LabelEvalResult,
    ShapeMatchResult,
    evaluate_image,
    run_evaluation,
)
from app.services.ocr import EasyOcrProvider
from app.services.braille import LouisBrailleTranslator


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class FakeLouis:
    def translateString(self, tables, text):
        return {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉"}.get(text, text)


# --- metric helpers ---

def test_safe_ratio_handles_zero_denominator():
    from app.services.evaluation import _safe_ratio
    assert _safe_ratio(0, 0) is None
    assert _safe_ratio(3, 0) is None
    assert _safe_ratio(3, 6) == 0.5


def test_shape_result_properties():
    r = ShapeMatchResult(true_positive=8, false_positive=2, false_negative=4)
    assert r.true_positive == 8
    assert r.false_positive == 2
    assert r.false_negative == 4


# --- evaluate_image with hand-built semantic ---

def _make_semantic(elements):
    from app.models.geometry import SemanticGeometry
    return SemanticGeometry(elements=elements)


def test_evaluate_image_perfect_match():
    from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType

    triangle = DetectedElement(id="t1", type=GeometryType.TRIANGLE, geometry={"points": [(0, 0), (10, 0), (5, 10)]},
                               confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="contour")
    label = DetectedElement(id="l1", type=GeometryType.TEXT_LABEL, geometry={"text": "A", "position": [5, 5]},
                            confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="ocr")

    gt = GroundTruth(image="x.png", shape_types=["triangle"], labels=["A"])
    semantic = _make_semantic([triangle, label])
    shape_result, label_result = evaluate_image(semantic, gt)

    assert shape_result.true_positive == 1
    assert shape_result.false_positive == 0
    assert shape_result.false_negative == 0
    assert label_result.true_positive == 1
    assert label_result.false_positive == 0
    assert label_result.false_negative == 0


def test_evaluate_image_partial_match():
    from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType

    triangle = DetectedElement(id="t1", type=GeometryType.TRIANGLE, geometry={"points": [(0, 0), (10, 0), (5, 10)]},
                               confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="contour")
    circle = DetectedElement(id="c1", type=GeometryType.CIRCLE, geometry={"center": [50, 50], "radius": 20},
                             confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="contour")
    label = DetectedElement(id="l1", type=GeometryType.TEXT_LABEL, geometry={"text": "A", "position": [5, 5]},
                            confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="ocr")

    # Ground truth expects triangle only (missing circle), labels A,B.
    gt = GroundTruth(image="x.png", shape_types=["triangle"], labels=["A", "B"])
    semantic = _make_semantic([triangle, circle, label])
    shape_result, label_result = evaluate_image(semantic, gt)

    assert shape_result.true_positive == 1   # triangle matches
    assert shape_result.false_positive == 1  # circle not expected
    assert shape_result.false_negative == 0  # nothing expected missing
    assert label_result.true_positive == 1   # A matches
    assert label_result.false_positive == 0
    assert label_result.false_negative == 1  # B missing


# --- run_evaluation over real fixtures ---

def test_run_evaluation_over_labeled_fixture(fixture_directory):
    ground_truths = [
        GroundTruth(
            image="triangle_worksheet.png",
            shape_types=["line_segment", "triangle"],
            labels=[],
        ),
    ]
    result = run_evaluation(
        fixture_directory,
        ground_truths,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    assert result.total_images == 1
    assert result.unlabeled_images == []
    # Triangle detected as triangle; contour yields a triangle element.
    assert result.shapes.true_positive >= 1


def test_run_evaluation_skips_unlabeled_images():
    ground_truths = [
        GroundTruth(image="triangle_worksheet.png", shape_types=[], labels=[]),
    ]
    result = run_evaluation(
        ".",
        ground_truths,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )
    assert result.total_images == 0
    assert "triangle_worksheet.png" in result.unlabeled_images


def test_run_evaluation_reports_missing_fixture_as_unlabeled(fixture_directory):
    ground_truths = [
        GroundTruth(image="does_not_exist.png", shape_types=["triangle"], labels=["A"]),
    ]
    result = run_evaluation(
        fixture_directory,
        ground_truths,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )
    assert "does_not_exist.png" in result.unlabeled_images


def test_eval_result_metrics_derived_not_fabricated():
    from app.services.evaluation import EvalResult

    result = EvalResult(
        shapes=ShapeMatchResult(true_positive=9, false_positive=1, false_negative=0),
        labels=LabelEvalResult(true_positive=3, false_positive=0, false_negative=0, total_detected=3, total_expected=3),
        total_images=2,
    )
    assert result.shape_precision == 0.9
    assert result.shape_recall == 1.0
    assert result.shape_f1 is not None and abs(result.shape_f1 - 0.947) < 0.001
    assert result.label_accuracy == 1.0
