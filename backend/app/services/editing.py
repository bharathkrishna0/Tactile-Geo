"""Teacher-in-the-loop editing over the semantic geometry.

The semantic geometry is the single source of truth. Teacher corrections are
applied directly to :class:`~app.models.geometry.DetectedElement` instances,
then the downstream tactile pipeline (simplification -> QA -> SVG) is
re-generated. The raw CV vector (`preview_svg`) and the original image are
never touched, so a correction never restarts the upload/process flow.
"""
from __future__ import annotations

from collections.abc import Callable

from app.models.geometry import ConfidenceLevel, DetectedElement, ElementRelationship, GeometryType, SemanticGeometry, TransformationExplanation, classify_confidence
from app.services.geometry_relations import infer_relationships
from app.services.tactile_qa import QAReport, run_tactile_qa
from app.services.tactile_simplification import SimplifiedGeometry, simplify_geometry
from app.services.tactile_svg import render_tactile_svg


class ElementNotFoundError(Exception):
    """Raised when an edit targets an element that is not in the geometry."""


class InvalidEditError(Exception):
    """Raised when an edit is malformed or would produce invalid geometry."""


def semantic_from_dict(data: dict) -> SemanticGeometry:
    """Rebuild a SemanticGeometry model from the session's plain-dict form."""
    data = data or {}
    elements = [
        DetectedElement(
            id=item["id"],
            type=GeometryType(item["type"]),
            geometry=dict(item.get("geometry") or {}),
            confidence=float(item.get("confidence", 0.5)),
            confidence_level=ConfidenceLevel(item.get("confidence_level", "medium")),
            needs_review=bool(item.get("needs_review", False)),
            source=item.get("source", "unknown"),
            bbox=_coerce_bbox(item.get("bbox")),
            semantic_properties=dict(item.get("semantic_properties") or {}),
            associated_label_id=item.get("associated_label_id"),
        )
        for item in data.get("elements", [])
    ]
    relationships = [
        ElementRelationship(
            id=rel["id"],
            type=rel["type"],
            element_ids=list(rel.get("element_ids") or []),
            confidence=float(rel.get("confidence", 0.0)),
            confidence_level=ConfidenceLevel(rel.get("confidence_level", "medium")),
            needs_review=bool(rel.get("needs_review", False)),
            explanation=rel.get("explanation"),
            properties=dict(rel.get("properties") or {}),
        )
        for rel in data.get("relationships", [])
    ]
    return SemanticGeometry(
        elements=elements,
        relationships=relationships,
        image_width=int(data.get("image_width", 0)),
        image_height=int(data.get("image_height", 0)),
        element_count=int(data.get("element_count", len(elements))),
        low_confidence_count=int(data.get("low_confidence_count", 0)),
        review_flags=list(data.get("review_flags") or []),
        explanations=[
            TransformationExplanation(
                stage=ex.get("stage", ""),
                element_id=ex.get("element_id"),
                message=ex.get("message", ""),
            )
            for ex in data.get("explanations", [])
        ],
    )


def _coerce_bbox(bbox):
    if not bbox:
        return None
    return tuple(int(value) for value in bbox)


def apply_edit(semantic: SemanticGeometry, element_id: str, edit: dict, translator: Callable[[str], str] | None = None) -> list[TransformationExplanation]:
    """Apply a single correction to ``semantic``.

    ``translator`` converts label text to Braille; only needed for
    ``set_label`` edits. Returns the transformation explanations produced.
    """
    element = _find_element(semantic.elements, element_id)
    if element is None:
        raise ElementNotFoundError(f"No element with id {element_id!r}.")
    action = edit.get("action")
    if action == "delete":
        return [_delete_element(semantic, element)]
    if action == "set_type":
        return [_set_type(element, edit)]
    if action == "set_label":
        return [_set_label(element, edit, translator)]
    if action == "set_confidence":
        return [_set_confidence(element, edit)]
    if action == "set_association":
        return _set_association(semantic, element, edit)
    if action == "nudge":
        return [_nudge(element, edit, semantic.image_width, semantic.image_height)]
    raise InvalidEditError(f"Unsupported edit action {action!r}.")


def apply_batch_edits(semantic: SemanticGeometry, edits: list[dict], translator: Callable[[str], str] | None = None) -> list[TransformationExplanation]:
    """Apply many edits, each dict including an ``element_id``.

    Mutations happen on a caller-controlled ``semantic``; if any edit fails the
    caller simply does not persist, giving atomicity at the session-store level.
    """
    explanations: list[TransformationExplanation] = []
    for edit in edits:
        element_id = edit.get("element_id")
        if not element_id:
            raise InvalidEditError("Each batch edit must carry an element_id.")
        explanations.extend(apply_edit(semantic, element_id, edit, translator=translator))
    return explanations


def regenerate(semantic: SemanticGeometry, translator=None) -> tuple[SimplifiedGeometry, QAReport, str]:
    """Re-run the downstream tactile stages from the corrected semantic model.

    Returns ``(simplified, qa, tactile_svg)``. Braille text is written directly
    into label geometry by the caller/apply_edit, so ``translator`` is only a
    hook for consistency.
    """
    simplified = simplify_geometry(semantic)
    qa_report = run_tactile_qa(simplified.elements, semantic.image_width, semantic.image_height)
    tactile_svg = render_tactile_svg(simplified.elements, semantic.image_width, semantic.image_height)
    return simplified, qa_report, tactile_svg


def refresh_semantic_fields(semantic: SemanticGeometry) -> SemanticGeometry:
    """Recompute relationships and summary counters after edits.

    The teacher's label-association overrides are preserved (relationships are
    re-inferred from the edited elements; associations are not overwritten).
    """
    semantic.relationships = _dedupe_relationships(infer_relationships(semantic.elements))
    semantic.element_count = len(semantic.elements)
    semantic.low_confidence_count = sum(
        1 for element in semantic.elements if element.needs_review or element.confidence_level is ConfidenceLevel.LOW
    )
    semantic.review_flags = [
        f"Low-confidence {element.type.value} detected — may be noise. Review before export."
        for element in semantic.elements if element.needs_review
    ]
    return semantic


def _dedupe_relationships(relationships: list[ElementRelationship]) -> list[ElementRelationship]:
    seen: set[tuple] = set()
    result: list[ElementRelationship] = []
    for rel in relationships:
        key = (rel.type.value, tuple(sorted(rel.element_ids)))
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


def _find_element(elements: list[DetectedElement], element_id: str) -> DetectedElement | None:
    return next((element for element in elements if element.id == element_id), None)


def _delete_element(semantic: SemanticGeometry, element: DetectedElement) -> list[TransformationExplanation]:
    semantic.elements.remove(element)
    _clear_references(semantic.elements, element)
    return [TransformationExplanation(
        stage="teacher_edit",
        element_id=element.id,
        message=f"Teacher removed element {element.id} ({element.type.value}).",
    )]


def _clear_references(elements: list[DetectedElement], removed: DetectedElement) -> None:
    """Drop stale association pointers that referenced the removed element."""
    for element in elements:
        if element.id != removed.id and element.associated_label_id == removed.id:
            element.associated_label_id = None


def _set_type(element: DetectedElement, edit: dict) -> TransformationExplanation:
    raw_type = edit.get("type")
    if not raw_type:
        raise InvalidEditError("set_type requires a 'type' value.")
    try:
        new_type = GeometryType(raw_type)
    except ValueError as error:
        raise InvalidEditError(f"Unknown geometry type {raw_type!r}.") from error
    if element.type is GeometryType.TEXT_LABEL or new_type is GeometryType.TEXT_LABEL:
        raise InvalidEditError("A label cannot be converted to geometry or vice versa.")
    element.type = new_type
    element.confidence = 1.0
    element.confidence_level = ConfidenceLevel.HIGH
    element.needs_review = False
    element.source = "teacher_override"
    _normalize_geometry_for_type(element)
    return TransformationExplanation(
        stage="teacher_edit",
        element_id=element.id,
        message=f"Teacher overrode element {element.id} as {new_type.value}.",
    )


def _normalize_geometry_for_type(element: DetectedElement) -> None:
    points = element.geometry.get("points")
    # Point
    if element.type is GeometryType.POINT and "position" not in element.geometry and points:
        element.geometry["position"] = list(points[0])
    # Line-like shapes need start/end.
    if element.type in (GeometryType.LINE_SEGMENT, GeometryType.RAY) and ("start" not in element.geometry or "end" not in element.geometry) and points and len(points) >= 2:
        element.geometry["start"] = list(points[0])
        element.geometry["end"] = list(points[-1])


def _set_label(element: DetectedElement, edit: dict, translator: Callable[[str], str] | None) -> TransformationExplanation:
    if element.type is not GeometryType.TEXT_LABEL:
        raise InvalidEditError("Only label elements have editable text.")
    text = str(edit.get("text", "")).strip()
    element.geometry["text"] = text
    element.geometry["braille"] = translator(text) if translator else element.geometry.get("braille", text)
    element.confidence = 1.0
    element.confidence_level = ConfidenceLevel.HIGH
    element.needs_review = False
    element.source = "teacher_override"
    return TransformationExplanation(
        stage="teacher_edit",
        element_id=element.id,
        message=f"Teacher edited label text to {text!r}.",
    )


def _set_confidence(element: DetectedElement, edit: dict) -> TransformationExplanation:
    confidence = edit.get("confidence")
    if confidence is None:
        raise InvalidEditError("set_confidence requires a 'confidence' value.")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise InvalidEditError("confidence must be a number between 0 and 1.") from error
    if not (0.0 <= confidence <= 1.0):
        raise InvalidEditError("confidence must be between 0 and 1.")
    element.confidence = confidence
    element.confidence_level = classify_confidence(confidence)
    element.needs_review = confidence < 0.5
    element.source = "teacher_override"
    verb = "accepted" if confidence >= 0.8 else "flagged as uncertain"
    return TransformationExplanation(
        stage="teacher_edit",
        element_id=element.id,
        message=f"Teacher {verb} element {element.id} (confidence {confidence:.2f}).",
    )


def _set_association(semantic: SemanticGeometry, element: DetectedElement, edit: dict) -> list[TransformationExplanation]:
    label_id = edit.get("association_label_id")
    target_id = edit.get("association_target_id")
    target: DetectedElement | None = None
    label: DetectedElement | None = None

    if element.type is GeometryType.TEXT_LABEL:
        # element is the label; association_target_id names the geometry.
        label, target = element, _find_element(semantic.elements, target_id) if target_id else None
    else:
        # element is geometry; association_label_id names the label.
        target, label = element, _find_element(semantic.elements, label_id) if label_id else None

    if target_id and target is None:
        raise InvalidEditError(f"Association target element {target_id!r} not found.")
    if label_id and label is None:
        raise InvalidEditError(f"Association label element {label_id!r} not found.")

    if target is not None and target.type is GeometryType.TEXT_LABEL:
        raise InvalidEditError("A label cannot be associated with another label.")
    if label is not None and label.type is not GeometryType.TEXT_LABEL:
        raise InvalidEditError("Only a text label may associate to geometry.")

    # Clear prior associations pointing at either element.
    for other in semantic.elements:
        if other.id != label.id and other.associated_label_id == label.id:
            other.associated_label_id = None

    if label is not None:
        label.associated_label_id = target.id if target else None
        if target is not None:
            target.associated_label_id = label.id

    explanation = (
        f"Teacher associated label {label.id} with {target.id}."
        if label is not None and target is not None
        else f"Teacher cleared the label association for {element.id}."
    )
    return [TransformationExplanation(stage="teacher_edit", element_id=element.id, message=explanation)]


def _nudge(element: DetectedElement, edit: dict, image_width: int, image_height: int) -> TransformationExplanation:
    offset = edit.get("offset")
    if not offset or len(offset) != 2:
        raise InvalidEditError("nudge requires a 2-value 'offset' (dx, dy).")
    dx = int(offset[0])
    dy = int(offset[1])
    if dx == 0 and dy == 0:
        raise InvalidEditError("nudge offset must be non-zero.")
    modified = _offset_geometry(element.geometry, dx, dy, image_width, image_height)
    if not modified:
        raise InvalidEditError("This element type has no movable coordinates.")
    return TransformationExplanation(
        stage="teacher_edit",
        element_id=element.id,
        message=f"Teacher moved element {element.id} by ({dx}, {dy}).",
    )


def _offset_geometry(geometry: dict, dx: int, dy: int, image_width: int, image_height: int) -> bool:
    moved = False

    def _shift(coord) -> list[int] | None:
        x = coord[0] + dx
        y = coord[1] + dy
        if x < 0 or y < 0 or x > image_width or y > image_height:
            raise InvalidEditError("Nudged element would leave the image bounds.")
        return [x, y]

    for key in ("points", "arms"):
        if key in geometry:
            updated = []
            for coord in geometry[key]:
                shifted = _shift(coord)
                if shifted is None:
                    return False
                updated.append(shifted)
            geometry[key] = updated
            moved = True
    for key in ("start", "end", "center", "position", "vertex"):
        if key in geometry:
            shifted = _shift(geometry[key])
            if shifted is None:
                return False
            geometry[key] = shifted
            moved = True
    return moved