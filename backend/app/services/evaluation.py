"""Evaluation framework for the conversion pipeline.

Computes accuracy, precision, recall, and F1 from *known ground truth* test
cases only. Metrics are never fabricated: if no ground truth is available for
an image, that image is excluded from the aggregate and reported as "unlabeled".

Ground-truth format (a Python dict list) for each test case::

    {
        "image": "fixture_name.png",
        "shapes": [  # expected geometry features (non-label)
            {"type": "line_segment", "center": (x, y)},
            {"type": "triangle",     "center": (x, y)},
        ],
        "labels": ["A", "B", "C"],
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.geometry import GeometryType
from app.services.pipeline import build_full_analysis
from app.services.ocr import OcrProvider
from app.services.braille import LouisBrailleTranslator


@dataclass
class GroundTruth:
    image: str
    shape_types: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass
class ShapeMatchResult:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0


@dataclass
class LabelEvalResult:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    total_detected: int = 0
    total_expected: int = 0


@dataclass
class EvalResult:
    shapes: ShapeMatchResult = field(default_factory=ShapeMatchResult)
    labels: LabelEvalResult = field(default_factory=LabelEvalResult)
    unlabeled_images: list[str] = field(default_factory=list)
    total_images: int = 0

    @property
    def shape_precision(self) -> float | None:
        return _safe_ratio(self.shapes.true_positive, self.shapes.true_positive + self.shapes.false_positive)

    @property
    def shape_recall(self) -> float | None:
        return _safe_ratio(self.shapes.true_positive, self.shapes.true_positive + self.shapes.false_negative)

    @property
    def shape_f1(self) -> float | None:
        p, r = self.shape_precision, self.shape_recall
        if p is None or r is None:
            return None
        return _safe_ratio(2 * p * r, p + r)

    @property
    def label_accuracy(self) -> float | None:
        expected = self.labels.total_expected
        if expected == 0:
            return None
        correct = self.labels.true_positive
        # Accuracy over the expected label set.
        return correct / expected

    @property
    def label_precision(self) -> float | None:
        return _safe_ratio(self.labels.true_positive, self.labels.true_positive + self.labels.false_positive)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _ground_truth_shape_types(shapes: list[dict]) -> set[str]:
    # Ground truth uses "type" matching the semantic GeometryType values.
    return {s["type"] for s in shapes}


def _detected_shape_types(elements) -> set[str]:
    result: set[str] = set()
    for element in elements:
        if element.type is GeometryType.TEXT_LABEL:
            continue
        result.add(element.type.value)
    return result


def evaluate_image(semantic, ground_truth: GroundTruth) -> tuple[ShapeMatchResult, LabelEvalResult]:
    """Compare one analyzed image against known ground truth.

    Shape matching is by type set (a coarse but reproducible proxy). For precise
    location-aware matching a future enhancement can compare bounding boxes.
    """
    expected_types = _ground_truth_shape_types([{"type": t} for t in ground_truth.shape_types])
    detected_types = _detected_shape_types(semantic.elements)

    shape_result = ShapeMatchResult()
    for d_type in detected_types:
        if d_type in expected_types:
            shape_result.true_positive += 1
        else:
            shape_result.false_positive += 1
    for expected_type in expected_types:
        if expected_type not in detected_types:
            shape_result.false_negative += 1

    # Labels: compare detected OCR text against expected labels.
    detected_labels = {e.geometry.get("text") for e in semantic.elements if e.type is GeometryType.TEXT_LABEL}
    expected_labels = set(ground_truth.labels)

    label_result = LabelEvalResult(
        total_detected=len(detected_labels),
        total_expected=len(expected_labels),
    )
    for text in detected_labels:
        if text in expected_labels:
            label_result.true_positive += 1
        else:
            label_result.false_positive += 1
    for expected in expected_labels:
        if expected not in detected_labels:
            label_result.false_negative += 1

    return shape_result, label_result


def run_evaluation(
    fixtures_dir,
    ground_truths: list[GroundTruth],
    edge_sensitivity: int = 100,
    ocr_provider: OcrProvider | None = None,
    braille_translator: LouisBrailleTranslator | None = None,
) -> EvalResult:
    """Run the pipeline over labeled fixtures and aggregate metrics.

    Only images with ground truth contribute to metrics; others are listed
    under ``unlabeled_images`` and do not skew the numbers.
    """
    result = EvalResult(total_images=len(ground_truths))
    known: list[GroundTruth] = []
    for gt in ground_truths:
        if not gt.shape_types and not gt.labels:
            result.unlabeled_images.append(gt.image)
            continue
        known.append(gt)
    result.total_images = len(known)
    result.unlabeled_images = [gt.image for gt in ground_truths if not gt.shape_types and not gt.labels]

    for gt in known:
        image_path = fixtures_dir / gt.image
        if not image_path.exists():
            # Missing fixture: do not fabricate; report unlabeled.
            result.unlabeled_images.append(gt.image)
            continue
        pipeline = build_full_analysis(
            image_path.read_bytes(),
            edge_sensitivity=edge_sensitivity,
            ocr_provider=ocr_provider,
            braille_translator=braille_translator,
        )
        shape_result, label_result = evaluate_image(pipeline.semantic_geometry, gt)
        result.shapes.true_positive += shape_result.true_positive
        result.shapes.false_positive += shape_result.false_positive
        result.shapes.false_negative += shape_result.false_negative
        result.labels.true_positive += label_result.true_positive
        result.labels.false_positive += label_result.false_positive
        result.labels.false_negative += label_result.false_negative
        result.labels.total_detected += label_result.total_detected
        result.labels.total_expected += label_result.total_expected

    return result
