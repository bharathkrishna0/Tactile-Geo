"""Validate a parsed Model B document.

Two layers, and the split matters:

* Structural validation is Pydantic's job (`schema.py`). Failures there are hard
  errors: the document does not match the contract we sent.
* Semantic validation happens here and produces *warnings*, not errors. A
  relationship pointing at a missing entity, or a reported image size that
  disagrees with the size we sent, is a recoverable defect. Dropping the whole
  analysis because one id dangles would throw away a correct tangent-circle
  finding over a bookkeeping mistake.

Warnings are surfaced to the teacher rather than swallowed, because "this result
is internally inconsistent" is information a person can act on.
"""

from __future__ import annotations

from pydantic import ValidationError

from .errors import ModelBValidationError
from .schema import ModelBDocument

_MAX_WARNINGS = 25


def validate_document(
    payload: dict, *, prepared_width: int, prepared_height: int
) -> tuple[ModelBDocument, list[str]]:
    """Validate structure, then check referential integrity.

    Returns:
        (document, warnings)

    Raises:
        ModelBValidationError: the document violates the schema. Field paths are
            attached to `.issues` for the server log and are not sent to clients.
    """
    try:
        document = ModelBDocument.model_validate(payload)
    except ValidationError as exc:
        issues = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        raise ModelBValidationError(
            "The model response did not match the Model B contract.",
            issues=issues[:_MAX_WARNINGS],
        ) from exc

    return document, _semantic_warnings(document, prepared_width, prepared_height)


def _semantic_warnings(
    document: ModelBDocument, prepared_width: int, prepared_height: int
) -> list[str]:
    warnings: list[str] = []
    entity_ids = {entity.id for entity in document.entities}
    seen_entity_ids: set[str] = set()

    for entity in document.entities:
        if entity.id in seen_entity_ids:
            warnings.append(f"duplicate entity id {entity.id}")
        seen_entity_ids.add(entity.id)
        if entity.kind == "circle":
            _warn_if_not_squareish(entity.bbox, "circle", entity.id, warnings)
        if entity.occluded and entity.confidence == "certain":
            warnings.append(
                f"entity {entity.id} is occluded but reported as certain"
            )

    for relationship in document.relationships:
        for role, ref in (("from_id", relationship.from_id), ("to_id", relationship.to_id)):
            if ref not in entity_ids:
                warnings.append(
                    f"relationship {relationship.id} {role}={ref} has no matching entity"
                )
        if relationship.from_id == relationship.to_id:
            warnings.append(
                f"relationship {relationship.id} references itself"
            )

    for note in document.diagram_relations:
        for ref in note.subject_ids:
            if ref not in entity_ids:
                warnings.append(
                    f"diagram relation {note.id} references unknown entity {ref}"
                )
        if not note.subject_ids:
            warnings.append(f"diagram relation {note.id} has no subjects")

    for note in document.uncertainties:
        for ref in note.subject_ids:
            if ref not in entity_ids:
                warnings.append(
                    f"uncertainty {note.id} references unknown entity {ref}"
                )

    if (document.image.width_px, document.image.height_px) != (
        prepared_width,
        prepared_height,
    ):
        # Not fatal: the bboxes are still meaningful as a rough region. It does
        # mean the model may be reasoning about a different frame than we think.
        warnings.append(
            f"reported image size {document.image.width_px}x{document.image.height_px} "
            f"does not match the {prepared_width}x{prepared_height} image that was sent"
        )

    if document.entities and not document.uncertainties and any(
        entity.confidence in {"uncertain", "unreadable"} for entity in document.entities
    ):
        warnings.append(
            "low-confidence entities were reported with no matching uncertainty note"
        )

    return warnings[:_MAX_WARNINGS]


def _warn_if_not_squareish(
    bbox: tuple[float, float, float, float],
    kind: str,
    entity_id: str,
    warnings: list[str],
) -> None:
    """Flag a circle whose bounding box is far from square.

    A projection into the prepared frame is uniform, so aspect distortion here
    means either a genuinely non-circular mark or a badly localized box. Either
    way a teacher should know the classification is shaky.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        warnings.append(f"{kind} {entity_id} has a degenerate bounding box")
        return
    longer = max(width, height)
    if longer > 0 and min(width, height) / longer < 0.7:
        warnings.append(
            f"{kind} {entity_id} has a strongly non-square bounding box; "
            "it may not be a true circle"
        )
