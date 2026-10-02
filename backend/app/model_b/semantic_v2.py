"""Semantic Geometry v2: Model A geometry plus teacher-accepted Model B findings.

Pipeline position::

    Model A ─┐
             ├─ fusion (reconcile) ─ teacher decisions ─ Semantic Geometry v2
    Model B ─┘                                              │
                         simplification ─ Braille layout ─ QA ─ teacher export

`reconcile()` stays read-only. This module is the only place a Model B finding
can change what is embossed, and it does so under three rules:

1. Only findings a teacher explicitly accepted are applied.
2. Only findings that correspond to an existing Model A element are applied.
   Coordinates, sizes and positions always come from Model A; a Model B region
   is never turned into geometry. Accepted Model B-only regions are reported as
   not applied.
3. Every change is reversible. The element records what it looked like before,
   so withdrawing an acceptance and applying again restores Model A's values,
   unless the teacher has since edited the element by hand.

Three kinds of change exist:

* ``set_type``: Model B's type for a contradicted element, only when Model A's
  own geometry already has what that type needs (a triangle needs exactly
  three Model A vertices, a circle a Model A centre and radius, and so on).
* ``attach_label``: Model B read a label Model A did not attach, and Model A
  has an unattached text label with the same text. The two Model A elements
  are associated; no label is created.
* ``confirm``: the teacher confirmed a Model A element Model B flagged (weak
  overlap, uncertainty, or a relationship). The element becomes a teacher
  override, which protects it from density simplification.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field

from ..models.geometry import ConfidenceLevel, DetectedElement, GeometryType, SemanticGeometry, TransformationExplanation
from ..models.model_b_result import ModelBResult, SuggestedEntity
from .fusion import EntityReview, reconcile

MARKER_KEY = "model_b_fusion"
STAGE = "model_b_fusion"


@dataclass
class FindingOutcome:
    model_b_id: str
    applied: bool
    change: str | None
    model_a_id: str | None
    detail: str


@dataclass
class SemanticGeometryV2:
    semantic: SemanticGeometry
    outcomes: list[FindingOutcome] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)

    @property
    def applied(self) -> list[FindingOutcome]:
        return [outcome for outcome in self.outcomes if outcome.applied]

    @property
    def not_applied(self) -> list[FindingOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.applied]


def model_a_baseline(semantic: SemanticGeometry, job_id: str) -> SemanticGeometry:
    """A copy of ``semantic`` with this job's applied findings undone."""
    baseline = copy.deepcopy(semantic)
    _revert_job(baseline, job_id)
    return baseline


def build_semantic_geometry_v2(
    semantic: SemanticGeometry,
    result: ModelBResult,
    decisions: dict[str, str],
    job_id: str,
) -> SemanticGeometryV2:
    """Apply accepted findings to a copy of ``semantic``. Never mutates inputs.

    ``decisions`` maps Model B id to the latest teacher decision. Applying is
    idempotent: earlier changes from the same job are undone first, so the
    output depends only on the current decisions.
    """
    v2 = copy.deepcopy(semantic)
    reverted = _revert_job(v2, job_id)
    report = reconcile(v2, result)
    entities = {entity.id: entity for entity in result.entities}
    reviews = {review.model_b_id: review for review in report.entity_reviews}
    id_counts = Counter(entity.id for entity in result.entities)
    repeated = {model_b_id for model_b_id, count in id_counts.items() if count > 1}
    outcomes: list[FindingOutcome] = []

    for model_b_id, decision in sorted(decisions.items()):
        if decision != "accept":
            continue
        if model_b_id in repeated:
            outcomes.append(FindingOutcome(
                model_b_id, False, None, None,
                f"Model B reported more than one finding as {model_b_id}, so this decision "
                "cannot be tied to one region. Correct it in the geometry view.",
            ))
            continue
        review = reviews.get(model_b_id)
        entity = entities.get(model_b_id)
        if review is None or entity is None:
            outcomes.append(FindingOutcome(model_b_id, False, None, None, "Recorded as context only; it is not a shape that can be embossed."))
            continue
        if review.model_a_id is None:
            outcomes.append(FindingOutcome(
                model_b_id, False, None, None,
                "Model A has no geometry here, so nothing can be embossed from this finding. Redraw or re-photograph this part of the worksheet.",
            ))
            continue
        element = next(e for e in v2.elements if e.id == review.model_a_id)
        outcomes.append(_apply(v2, element, entity, review, job_id))

    for model_b_id in reverted:
        if decisions.get(model_b_id) != "accept":
            v2.explanations.append(TransformationExplanation(
                stage=STAGE, element_id=None, message=f"Withdrew Model B finding {model_b_id}; restored the Model A values.",
            ))
    return SemanticGeometryV2(semantic=v2, outcomes=outcomes, reverted=reverted)


def _apply(semantic: SemanticGeometry, element: DetectedElement, entity: SuggestedEntity, review: EntityReview, job_id: str) -> FindingOutcome:
    existing = element.semantic_properties.get(MARKER_KEY)
    if existing is not None:
        return FindingOutcome(
            entity.id, False, None, element.id,
            f"Element {element.id} was already changed by Model B finding {existing['model_b_id']}.",
        )

    label: DetectedElement | None = None
    if review.contradicts_model_a:
        target = entity.geometry_type
        if target is None:
            return FindingOutcome(entity.id, False, None, element.id, f"Model B's '{entity.kind}' has no Model A equivalent.")
        if GeometryType.TEXT_LABEL in (element.type, target):
            return FindingOutcome(
                entity.id, False, None, element.id,
                f"{element.id} would switch between a Braille label and a raised shape, "
                "which would drop or invent a label; correct it in the geometry view.",
            )
        if not geometry_supports(target, element.geometry):
            return FindingOutcome(
                entity.id, False, None, element.id,
                f"Model A's geometry for {element.id} cannot be drawn as a {target.value}; correct it in the geometry view.",
            )
        change = "set_type"
    else:
        label = _matching_label(semantic, element, entity)
        change = "attach_label" if label is not None else "confirm"

    marker = {
        "job_id": job_id,
        "model_b_id": entity.id,
        "change": change,
        "previous": _tracked(element),
    }
    if change == "set_type":
        element.type = entity.geometry_type  # type: ignore[assignment]
        detail = f"Element {element.id} is now a {element.type.value}, keeping Model A's coordinates."
    elif change == "attach_label" and label is not None:
        marker["label_id"] = label.id
        marker["previous_label_association"] = label.associated_label_id
        element.associated_label_id = label.id
        label.associated_label_id = element.id
        detail = f"Attached Model A label {label.id} ('{entity.label}') to {element.id}."
    else:
        detail = f"Teacher confirmed {element.id}; it is protected from simplification."

    element.confidence = 1.0
    element.confidence_level = ConfidenceLevel.HIGH
    element.needs_review = False
    element.source = "teacher_override"
    element.provenance = f"Teacher accepted Model B finding {entity.id}. {detail}"
    marker["applied"] = _tracked(element)
    element.semantic_properties[MARKER_KEY] = marker
    semantic.explanations.append(TransformationExplanation(stage=STAGE, element_id=element.id, message=element.provenance))
    return FindingOutcome(entity.id, True, change, element.id, detail)


def geometry_supports(target: GeometryType, geometry: dict) -> bool:
    """Whether Model A's coordinates are enough to draw ``target``."""
    points = geometry.get("points") or []
    if target is GeometryType.TRIANGLE:
        return len(points) == 3
    if target is GeometryType.RECTANGLE:
        return len(points) == 4
    if target is GeometryType.POLYGON:
        return len(points) >= 3
    if target is GeometryType.CIRCLE:
        return "center" in geometry and "radius" in geometry
    if target is GeometryType.ELLIPSE:
        return "center" in geometry and "semi_axes" in geometry
    if target in (GeometryType.LINE_SEGMENT, GeometryType.RAY, GeometryType.ARROW, GeometryType.AXES):
        return ("start" in geometry and "end" in geometry) or len(points) == 2
    if target is GeometryType.POINT:
        return "position" in geometry
    return False


def _matching_label(semantic: SemanticGeometry, element: DetectedElement, entity: SuggestedEntity) -> DetectedElement | None:
    if not entity.label or element.associated_label_id is not None:
        return None
    wanted = entity.label.strip().casefold()
    for candidate in semantic.elements:
        if (
            candidate.type is GeometryType.TEXT_LABEL
            and candidate.associated_label_id is None
            and str(candidate.geometry.get("text", "")).strip().casefold() == wanted
        ):
            return candidate
    return None


def _tracked(element: DetectedElement) -> dict:
    return {
        "type": element.type.value,
        "confidence": element.confidence,
        "confidence_level": element.confidence_level.value,
        "needs_review": element.needs_review,
        "source": element.source,
        "provenance": element.provenance,
        "associated_label_id": element.associated_label_id,
    }


def _revert_job(semantic: SemanticGeometry, job_id: str) -> list[str]:
    """Undo this job's changes in place. Returns the Model B ids undone.

    A field is restored only while it still holds the value Model B set; a later
    hand edit by the teacher wins.
    """
    by_id = {element.id: element for element in semantic.elements}
    reverted: list[str] = []
    for element in semantic.elements:
        marker = element.semantic_properties.get(MARKER_KEY)
        if not marker or marker.get("job_id") != job_id:
            continue
        del element.semantic_properties[MARKER_KEY]
        reverted.append(marker["model_b_id"])
        applied, previous = marker["applied"], marker["previous"]
        if _tracked(element) != applied:
            continue
        element.type = GeometryType(previous["type"])
        element.confidence = previous["confidence"]
        element.confidence_level = ConfidenceLevel(previous["confidence_level"])
        element.needs_review = previous["needs_review"]
        element.source = previous["source"]
        element.provenance = previous["provenance"]
        element.associated_label_id = previous["associated_label_id"]
        label = by_id.get(marker.get("label_id", ""))
        if label is not None and label.associated_label_id == element.id:
            label.associated_label_id = marker.get("previous_label_association")
    return reverted


__all__ = [
    "FindingOutcome",
    "MARKER_KEY",
    "SemanticGeometryV2",
    "build_semantic_geometry_v2",
    "geometry_supports",
    "model_a_baseline",
]
