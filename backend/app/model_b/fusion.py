"""Reconcile Model B's advisory regions against Model A's geometry.

The single most important property of this module is what it does NOT do. It
never writes to Model A. It does not move a point, add a vertex, delete an
element, change a confidence score, touch the QA report, or rename an id. It is
a pure function from (Model A geometry, Model B result) to a `FusionReport`, and
the report is a list of things a human might want to look at.

Why that constraint, stated this strongly: Model A's output is embossed. A
tactile output that a blind student cannot question is worse than one that is
merely incomplete, because the student has no way to detect that a confident
model disagreed with the page. So Model B's job is to raise questions, and the
only mechanism by which it can do that is by not being able to change anything.

Four invariants, each enforced by a test in `test_model_b_fusion.py`:

    I1  Model A geometry is byte-identical after fusion.
    I2  The QA report is byte-identical after fusion.
    I3  No Model A element id is removed or renamed.
    I4  Every suggested addition requires explicit teacher approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.geometry import DetectedElement, SemanticGeometry
from ..models.model_b_result import BoundingRegion, ModelBResult, SuggestedEntity

# Overlap required before two regions are considered the same object.
#
# This is a heuristic for "should a human look here", not a measurement, and it
# is deliberately loose. A tight threshold would silence exactly the cases Model
# B exists to catch, where Model A has drawn a triangle three times as large as
# the real one. A loose threshold costs a false pairing in the agreement list,
# which is cheap; a tight one hides a real disagreement, which is not.
MATCH_IOU_THRESHOLD = 0.25

# Below this the pair is too weak to even call a hint. Kept separate from
# MATCH_IOU_THRESHOLD so the two decisions can be tuned independently.
STRONG_IOU_THRESHOLD = 0.5

Verdict = str  # "agrees" | "weak_overlap" | "type_mismatch" | "model_b_only" | "model_a_only"


@dataclass
class AgreementHint:
    """Model B and Model A both point at the same region."""

    model_b_id: str
    model_a_id: str
    model_b_kind: str
    model_a_type: str
    iou: float
    verdict: Verdict
    advisory_only: bool = True


@dataclass
class CandidateAddition:
    """Something Model B saw that Model A did not.

    Never applied. Carries a Model A-shaped suggestion so the teacher can accept
    it through the existing element-editing path rather than a special case.
    """

    model_b_id: str
    suggested_type: str | None
    region: tuple[int, int, int, int]
    reason: str
    requires_teacher_approval: bool = True
    advisory_only: bool = True


@dataclass
class Disagreement:
    """A place where the two models appear to conflict."""

    kind: str
    model_b_id: str
    model_a_id: str | None
    detail: str
    advisory_only: bool = True


@dataclass
class FusionReport:
    agreements: list[AgreementHint] = field(default_factory=list)
    candidate_additions: list[CandidateAddition] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    # Whole-diagram statements Model A has no concept of. Passed through as
    # context; the tangent/right-triangle findings are the headline value.
    diagram_relations: list[dict] = field(default_factory=list)
    text_notes: list[dict] = field(default_factory=list)
    uncertainties: list[dict] = field(default_factory=list)
    advisory_only: bool = True

    @property
    def has_actionable_content(self) -> bool:
        return bool(
            self.candidate_additions
            or self.disagreements
            or self.uncertainties
            or self.diagram_relations
        )

    def summary(self) -> dict[str, int]:
        return {
            "agreements": len(self.agreements),
            "candidate_additions": len(self.candidate_additions),
            "disagreements": len(self.disagreements),
            "diagram_relations": len(self.diagram_relations),
            "text_notes": len(self.text_notes),
            "uncertainties": len(self.uncertainties),
        }


def reconcile(
    model_a: SemanticGeometry, model_b: ModelBResult | None
) -> FusionReport:
    """Compare the two models. Returns a report; mutates nothing.

    A `None` Model B result yields an empty report, so the caller needs no
    special case and the panel renders an honest "not analysed" state.
    """
    report = FusionReport()
    if model_b is None:
        return report

    elements = [element for element in model_a.elements if element.bbox]
    matched_a: set[str] = set()

    for entity in model_b.entities:
        best = _best_overlap(entity.region, elements, matched_a)
        if best is None:
            report.candidate_additions.append(
                _candidate(entity, "Model B found a region Model A did not report.")
            )
            continue

        element, iou = best
        matched_a.add(element.id)
        report.agreements.append(
            AgreementHint(
                model_b_id=entity.id,
                model_a_id=element.id,
                model_b_kind=entity.kind,
                model_a_type=element.type.value,
                iou=iou,
                verdict=_verdict(entity, element, iou),
            )
        )
        if _verdict(entity, element, iou) == "type_mismatch":
            report.disagreements.append(
                Disagreement(
                    kind="type_mismatch",
                    model_b_id=entity.id,
                    model_a_id=element.id,
                    detail=(
                        f"Model A read this region as {element.type.value}; "
                        f"Model B read it as {entity.kind}."
                    ),
                )
            )

    for element in elements:
        if element.id not in matched_a:
            report.disagreements.append(
                Disagreement(
                    kind="model_a_only",
                    model_b_id="",
                    model_a_id=element.id,
                    detail=(
                        f"Model A reported a {element.type.value} that Model B "
                        "did not mention. This is often a duplicate or an artifact."
                    ),
                )
            )

    report.diagram_relations = [
        {
            "id": note.id,
            "kind": note.kind,
            "statement": note.statement,
            "evidence": note.evidence,
            "subject_ids": note.subject_ids,
            "confidence": note.confidence,
            "needs_review": note.needs_review,
            "detection_kind": note.detection_kind,
        }
        for note in model_b.diagram_relations
    ]
    report.text_notes = [
        {
            "id": item.id,
            "text": item.text,
            "role": item.role,
            "bbox": item.region.as_model_a_bbox(),
            "needs_review": item.needs_review,
            "evidence": item.evidence,
        }
        for item in model_b.text_items
    ]
    report.uncertainties = [
        {
            "id": note.id,
            "kind": note.kind,
            "note": note.note,
            "severity": note.severity,
            "subject_ids": note.subject_ids,
        }
        for note in model_b.uncertainties
    ]

    if model_b.truncated:
        report.uncertainties.append(
            {
                "id": "truncation",
                "kind": "partial_diagram",
                "note": (
                    "The Model B response was cut off, so this analysis is "
                    "incomplete and may be missing findings."
                ),
                "severity": "worth_review",
                "subject_ids": [],
            }
        )

    return report


def _best_overlap(
    region: BoundingRegion,
    elements: list[DetectedElement],
    claimed: set[str] | None = None,
) -> tuple[DetectedElement, float] | None:
    """Highest-IoU element above threshold, ignoring ones already paired.

    A Model A element can back at most one Model B entity. Without excluding
    claimed ids, two Model B regions over the same shape would both report an
    agreement against a single Model A element and the counts would overstate
    how well the two models line up.
    """
    best: tuple[DetectedElement, float] | None = None
    for element in elements:
        if claimed is not None and element.id in claimed:
            continue
        x, y, width, height = element.bbox  # type: ignore[misc]
        candidate = BoundingRegion(norm=(), x=x, y=y, width=width, height=height)
        overlap = region.iou(candidate)
        if overlap >= MATCH_IOU_THRESHOLD and (best is None or overlap > best[1]):
            best = (element, overlap)
    return best


def _candidate(entity: SuggestedEntity, reason: str) -> CandidateAddition:
    return CandidateAddition(
        model_b_id=entity.id,
        suggested_type=entity.geometry_type.value if entity.geometry_type else None,
        region=entity.region.as_model_a_bbox(),
        reason=reason,
    )


def _verdict(entity: SuggestedEntity, element: DetectedElement, iou: float) -> Verdict:
    """Label a pairing. Never resolves it; a human resolves it."""
    if entity.geometry_type is None or entity.geometry_type is not element.type:
        return "type_mismatch"
    return "agrees" if iou >= STRONG_IOU_THRESHOLD else "weak_overlap"
