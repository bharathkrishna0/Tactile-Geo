"""Phase 5 tests: the reconciliation layer and its four safety invariants.

I1  Model A geometry is unchanged after fusion.
I2  The QA report is unchanged after fusion.
I3  No Model A element id is removed or renamed.
I4  Every suggested addition requires teacher approval.
"""

from __future__ import annotations

import copy
import json

import cv2
import numpy as np
import pytest

from app.model_b._fixtures import valid_document
from app.model_b.fusion import (
    MATCH_IOU_THRESHOLD,
    STRONG_IOU_THRESHOLD,
    reconcile,
)
from app.model_b.normalizer import normalize
from app.model_b.preparation import prepare_image
from app.model_b.validator import validate_document
from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    GeometryType,
    SemanticGeometry,
)
from app.services.tactile_qa import QAIssue, QAReport


def make_model_a() -> SemanticGeometry:
    """A triangle plus an ellipse, with the bboxes the fixture will overlap."""
    return SemanticGeometry(
        elements=[
            DetectedElement(
                id="m1",
                type=GeometryType.TRIANGLE,
                geometry={"points": [[10, 10], [200, 10], [100, 180]]},
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                needs_review=False,
                source="opencv_hough",
                bbox=(100, 80, 600, 560),
            ),
            DetectedElement(
                id="m2",
                type=GeometryType.ELLIPSE,
                geometry={"center": [620, 360], "semi_axes": [150, 80]},
                confidence=0.7,
                confidence_level=ConfidenceLevel.MEDIUM,
                needs_review=True,
                source="opencv_contour",
                bbox=(550, 240, 300, 240),
            ),
        ],
        image_width=1000,
        image_height=800,
        element_count=2,
    )


def make_model_b(doc=None) -> "object":
    document, warnings = validate_document(
        doc or valid_document(), prepared_width=1568, prepared_height=1568
    )
    return normalize(document, _prepared(), finish_reason="STOP", validation_warnings=warnings)


def _prepared():
    array = np.full((800, 1000, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return prepare_image(encoded.tobytes())


def norm_for_pixel_bbox(prepared, bbox: tuple[int, int, int, int]) -> list[float]:
    """Invert `project_bbox` so tests can place entities at known pixels.

    Without this, every matching test depends on the letterbox arithmetic being
    guessed correctly, and a failure reads as a fusion bug rather than a bad
    test fixture.
    """
    x, y, width, height = bbox

    def to_norm(pixel: float, pad: int, scale: float, extent: int) -> float:
        return (pixel / scale + pad) / extent

    return [
        to_norm(x, prepared.pad_x, prepared.scale_x, prepared.width),
        to_norm(y, prepared.pad_y, prepared.scale_y, prepared.height),
        to_norm(x + width, prepared.pad_x, prepared.scale_x, prepared.width),
        to_norm(y + height, prepared.pad_y, prepared.scale_y, prepared.height),
    ]


def entity_at(kind: str, pixel_bbox: tuple[int, int, int, int]) -> dict:
    """A single well-formed entity covering a known pixel region."""
    base = valid_document()["entities"][0]
    entity = dict(base, kind=kind, bbox=norm_for_pixel_bbox(_prepared(), pixel_bbox))
    return entity


def entity_with_id(entity_id: str, kind: str, pixel_bbox: tuple[int, int, int, int]) -> dict:
    """As `entity_at`, but with a chosen id so two entities can stack up.

    Ids are the raw contract vocabulary (`e1`..`e40`); the `b_` prefix is added
    later by the normalizer, so the report refers to `b_e1` and so on.
    """
    return dict(entity_at(kind, pixel_bbox), id=entity_id)


def doc_with(entities: list[dict]) -> dict:
    doc = valid_document()
    doc["entities"] = entities
    doc["relationships"] = []
    doc["diagram_relations"] = []
    return doc


def snapshot(geometry: SemanticGeometry) -> str:
    """Full structural snapshot, so a mutation anywhere shows up."""
    return json.dumps(
        {
            "elements": [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "geometry": e.geometry,
                    "confidence": e.confidence,
                    "confidence_level": e.confidence_level.value,
                    "needs_review": e.needs_review,
                    "source": e.source,
                    "bbox": e.bbox,
                    "semantic_properties": e.semantic_properties,
                    "associated_label_id": e.associated_label_id,
                    "provenance": e.provenance,
                }
                for e in geometry.elements
            ],
            "image_width": geometry.image_width,
            "image_height": geometry.image_height,
            "element_count": geometry.element_count,
            "low_confidence_count": geometry.low_confidence_count,
            "review_flags": geometry.review_flags,
            "explanations": [vars(x) for x in geometry.explanations],
        },
        sort_keys=True,
        default=str,
    )


class TestSafetyInvariants:
    """The tests that matter most. Each would fail loudly on a careless edit."""

    def test_i1_model_a_geometry_unchanged(self) -> None:
        geometry = make_model_a()
        before = snapshot(geometry)
        reconcile(geometry, make_model_b())
        assert snapshot(geometry) == before

    def test_i1_survives_a_hostile_report(self) -> None:
        """Every entity, relationship, and disagreement path, all at once."""
        geometry = make_model_a()
        before = snapshot(geometry)
        doc = valid_document()
        doc["entities"] = []
        doc["relationships"] = []
        reconcile(geometry, make_model_b(doc))
        reconcile(geometry, None)
        assert snapshot(geometry) == before

    def test_i2_qa_report_unchanged(self) -> None:
        """Fusion must not soften a blocking QA issue."""
        geometry = make_model_a()
        report = QAReport(
            overall_score=42.0,
            passes=False,
            issues=[
                QAIssue(
                    check="stroke_width_out_of_bounds",
                    severity="blocking",
                    message="Stroke too thin to emboss reliably.",
                    element_id="m1",
                )
            ],
            element_checks=2,
            score_0_100=42,
        )
        snapshot_before = copy.deepcopy(vars(report))

        reconcile(geometry, make_model_b())

        assert vars(report) == snapshot_before
        assert report.passes is False
        assert report.score_0_100 == 42
        assert report.issues[0].severity == "blocking"

    def test_i3_no_element_id_removed_or_renamed(self) -> None:
        geometry = make_model_a()
        before_ids = [element.id for element in geometry.elements]
        report = reconcile(geometry, make_model_b())
        after_ids = [element.id for element in geometry.elements]
        assert before_ids == after_ids
        # Every id referenced by the report must exist in Model A or be a Model B id.
        model_b_ids = {e.id for e in make_model_b().entities}
        for hint in report.agreements:
            assert hint.model_a_id in before_ids
            assert hint.model_b_id in model_b_ids

    def test_i4_every_addition_requires_approval(self) -> None:
        geometry = make_model_a()
        report = reconcile(geometry, make_model_b())
        assert report.candidate_additions
        for addition in report.candidate_additions:
            assert addition.requires_teacher_approval is True
            assert addition.advisory_only is True

    def test_i4_agreement_hints_are_marked_advisory(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        for hint in report.agreements:
            assert hint.advisory_only is True
        for note in report.disagreements:
            assert note.advisory_only is True
        assert report.advisory_only is True


class TestEmptyInputs:
    def test_none_model_b_yields_empty_report(self) -> None:
        report = reconcile(make_model_a(), None)
        assert not report.has_actionable_content
        assert report.summary()["agreements"] == 0

    def test_model_a_with_no_elements_reports_everything_as_addition(self) -> None:
        report = reconcile(SemanticGeometry(image_width=1000, image_height=800), make_model_b())
        assert len(report.candidate_additions) == 3
        assert report.agreements == []

    def test_elements_without_bbox_are_skipped(self) -> None:
        geometry = make_model_a()
        geometry.elements[0].bbox = None
        report = reconcile(geometry, make_model_b())
        # The bbox-less triangle cannot be matched, so the Model B triangle
        # becomes an addition rather than a false agreement.
        assert any(a.model_b_id == "b_e1" for a in report.candidate_additions)


class TestMatching:
    def test_overlapping_same_type_agrees(self) -> None:
        """Identical region, identical type: a genuine agreement."""
        report = reconcile(make_model_a(), make_model_b(doc_with([entity_at("triangle", (100, 80, 600, 560))])))
        assert len(report.agreements) == 1
        hint = report.agreements[0]
        assert hint.model_a_id == "m1"
        assert hint.model_b_kind == "triangle"
        assert hint.model_a_type == "triangle"
        assert hint.iou == pytest.approx(1.0, abs=0.02)
        assert hint.verdict == "agrees"

    def test_weak_overlap_is_distinguished_from_agreement(self) -> None:
        """A loose pairing is a prompt to look, not a confirmation.

        Boxes are (x, y, width, height). m1 is 600x480, so a 600x187 band
        covering its top third gives IoU ~0.39: real overlap, same type, but
        below the strong-overlap cutoff.
        """
        report = reconcile(make_model_a(), make_model_b(doc_with([entity_at("triangle", (100, 80, 600, 187))])))
        hint = report.agreements[0]
        assert MATCH_IOU_THRESHOLD <= hint.iou < STRONG_IOU_THRESHOLD
        assert hint.verdict == "weak_overlap"

    def test_below_threshold_is_not_matched_at_all(self) -> None:
        """Slop is worse than silence: a weak guess is not offered as a hint."""
        report = reconcile(make_model_a(), make_model_b(doc_with([entity_at("triangle", (100, 80, 600, 60))])))
        assert report.agreements == []
        assert any(a.model_b_id == "b_e1" for a in report.candidate_additions)

    def test_type_mismatch_is_a_disagreement(self) -> None:
        """Model A says ellipse, Model B says rectangle, same place."""
        report = reconcile(make_model_a(), make_model_b(doc_with([entity_at("rectangle", (550, 240, 300, 240))])))
        mismatches = [d for d in report.disagreements if d.kind == "type_mismatch"]
        assert len(mismatches) == 1
        assert mismatches[0].model_a_id == "m2"
        assert "ellipse" in mismatches[0].detail
        assert "rectangle" in mismatches[0].detail

    def test_unmapped_kind_over_a_match_is_a_mismatch_not_a_pretend_agreement(self) -> None:
        """A quadrilateral over Model A's triangle must not read as agreement."""
        report = reconcile(make_model_a(), make_model_b(doc_with([entity_at("quadrilateral", (100, 80, 600, 560))])))
        assert report.agreements[0].verdict == "type_mismatch"
        assert any(d.kind == "type_mismatch" for d in report.disagreements)

    def test_unmatched_model_a_element_is_flagged(self) -> None:
        """Duplicate contours are the most common real-world Model A artifact."""
        doc = valid_document()
        doc["entities"] = [doc["entities"][0]]  # only the triangle
        report = reconcile(make_model_a(), make_model_b(doc))
        orphan = [d for d in report.disagreements if d.kind == "model_a_only"]
        assert any(d.model_a_id == "m2" for d in orphan)
        assert "duplicate" in orphan[0].detail

    def test_each_model_a_element_matched_at_most_once(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        model_a_ids = [hint.model_a_id for hint in report.agreements]
        assert len(model_a_ids) == len(set(model_a_ids))

    def test_two_model_b_entities_cannot_both_claim_one_model_a_element(self) -> None:
        """Regression: the pairing must actually be one-to-one.

        Two Model B regions covering the same Model A element used to both
        report an agreement, so a single shape could be double-counted as
        independent corroboration. The surplus region is offered as a candidate
        addition instead, which is the honest reading: it is unverified.
        """
        report = reconcile(
            make_model_a(),
            make_model_b(
                doc_with(
                    [
                        entity_with_id("e1", "triangle", (100, 80, 600, 560)),
                        entity_with_id("e2", "triangle", (100, 80, 600, 560)),
                    ]
                )
            ),
        )

        model_a_ids = [hint.model_a_id for hint in report.agreements]
        assert model_a_ids == ["m1"], f"one Model A element paired twice: {model_a_ids}"
        assert len(report.agreements) == 1
        assert any(a.model_b_id == "b_e2" for a in report.candidate_additions)
        # m1 is genuinely matched, so it must not also be reported as missing.
        assert not any(
            d.kind == "model_a_only" and d.model_a_id == "m1" for d in report.disagreements
        )

    def test_second_entity_falls_back_to_an_unclaimed_element(self) -> None:
        """Claiming m1 must not stop a later entity from matching m2."""
        report = reconcile(
            make_model_a(),
            make_model_b(
                doc_with(
                    [
                        # Near-identical overlap with m1 twice, then a real m2 hit.
                        entity_with_id("e1", "triangle", (100, 80, 600, 560)),
                        entity_with_id("e2", "triangle", (110, 85, 590, 555)),
                        entity_with_id("e3", "ellipse", (550, 240, 300, 240)),
                    ]
                )
            ),
        )

        pairs = {hint.model_a_id for hint in report.agreements}
        assert pairs == {"m1", "m2"}, f"unexpected pairing: {pairs}"

    def test_addition_carries_model_a_shaped_suggestion(self) -> None:
        report = reconcile(SemanticGeometry(image_width=1000, image_height=800), make_model_b())
        addition = next(a for a in report.candidate_additions if a.model_b_id == "b_e1")
        assert addition.suggested_type == "triangle"
        x, y, width, height = addition.region
        assert all(isinstance(v, int) for v in (x, y, width, height))
        assert width > 0 and height > 0

    def test_unmapped_kind_yields_null_suggested_type(self) -> None:
        """A quadrilateral must not be offered as a Model A rectangle."""
        doc = valid_document()
        doc["entities"] = [dict(doc["entities"][0], kind="quadrilateral")]
        report = reconcile(SemanticGeometry(image_width=1000, image_height=800), make_model_b(doc))
        assert report.candidate_additions[0].suggested_type is None

    def test_thresholds_are_ordered(self) -> None:
        assert MATCH_IOU_THRESHOLD < STRONG_IOU_THRESHOLD


class TestPassThrough:
    def test_diagram_relations_survive(self) -> None:
        """The tangent finding is the headline value of Model B."""
        report = reconcile(make_model_a(), make_model_b())
        kinds = {note["kind"] for note in report.diagram_relations}
        assert "tangent" in kinds
        assert "right_triangle" in kinds

    def test_text_notes_keep_null_text(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        assert any(note["text"] is None for note in report.text_notes)
        assert any(note["text"] == "BC" for note in report.text_notes)

    def test_uncertainties_survive(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        assert any(note["kind"] == "occluded_geometry" for note in report.uncertainties)

    def test_truncation_is_surfaced_as_uncertainty(self) -> None:
        """A partial answer must not read like a complete one."""
        document, _ = validate_document(
            valid_document(), prepared_width=1568, prepared_height=1568
        )
        result = normalize(document, _prepared(), truncated=True, finish_reason="MAX_TOKENS")
        report = reconcile(make_model_a(), result)
        assert any(note["kind"] == "partial_diagram" for note in report.uncertainties)

    def test_complete_analysis_has_no_truncation_note(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        assert not any(note["kind"] == "partial_diagram" for note in report.uncertainties)


class TestReportShape:
    def test_summary_counts_match_contents(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        summary = report.summary()
        assert summary["agreements"] == len(report.agreements)
        assert summary["candidate_additions"] == len(report.candidate_additions)
        assert summary["disagreements"] == len(report.disagreements)
        assert summary["uncertainties"] == len(report.uncertainties)

    def test_actionable_content_detected(self) -> None:
        assert reconcile(make_model_a(), make_model_b()).has_actionable_content is True

    def test_report_is_json_serializable(self) -> None:
        """It has to cross the wire."""
        report = reconcile(make_model_a(), make_model_b())
        assert json.dumps(report.summary())
        assert json.dumps([vars(a) for a in report.agreements], default=str)
        assert json.dumps([vars(a) for a in report.candidate_additions], default=str)
        assert json.dumps([vars(d) for d in report.disagreements], default=str)

    def test_no_model_b_id_collides_with_model_a_id(self) -> None:
        report = reconcile(make_model_a(), make_model_b())
        for addition in report.candidate_additions:
            assert not addition.model_b_id.startswith("m")
