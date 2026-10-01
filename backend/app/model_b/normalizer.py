"""Turn a validated Model B document into `ModelBResult`.

This is where the advisory guarantee is enforced. Bounding boxes are projected
into the original-image frame so they can be compared against Model A's
elements, kinds are mapped through the auditable tables, and anything
unmappable is labelled as such rather than being dropped or coerced.
"""

from __future__ import annotations

from ..models.model_b_result import (
    BoundingRegion,
    DiagramRelationNote,
    ModelBResult,
    SuggestedEntity,
    SuggestedRelationship,
    TextNote,
    UncertaintyNote,
    confidence_for,
)
from .preparation import PreparedImage
from .schema import ModelBDocument

# Model B ids are namespaced so they can never collide with a Model A element id
# in a merged view, even though Model A ids are also short strings.
_ID_PREFIX = "b_"


def normalize(
    document: ModelBDocument,
    prepared: PreparedImage,
    *,
    provider: str = "",
    requested_model: str = "",
    resolved_model: str = "",
    upstream_provider: str = "",
    request_id: str = "",
    finish_reason: str = "",
    truncated: bool = False,
    usage: dict | None = None,
    validation_warnings: list[str] | None = None,
) -> ModelBResult:
    """Build the normalized, frontend-ready result."""
    result = ModelBResult(
        schema_version=document.schema_version,
        diagram_kind=document.diagram_summary.kind,
        diagram_description=document.diagram_summary.description,
        image_width=prepared.original_width,
        image_height=prepared.original_height,
        prepared_width=prepared.width,
        prepared_height=prepared.height,
        provider=provider,
        requested_model=requested_model,
        # Never empty: fall back to what was asked for so provenance is always
        # reportable even when the gateway does not echo a model id.
        resolved_model=resolved_model or requested_model,
        upstream_provider=upstream_provider,
        request_id=request_id,
        image_readable=document.image_quality.readable,
        image_quality_issues=list(document.image_quality.issues),
        truncated=truncated,
        finish_reason=finish_reason,
        usage=usage or {},
        validation_warnings=validation_warnings or [],
    )

    geometry_by_id: dict[str, object] = {}

    for entity in document.entities:
        confidence, level = confidence_for(entity.confidence)
        geometry = _map_geometry(entity.kind)
        geometry_by_id[entity.id] = geometry
        result.entities.append(
            SuggestedEntity(
                id=f"{_ID_PREFIX}{entity.id}",
                kind=entity.kind,
                geometry_type=geometry,
                region=_region(entity.bbox, prepared),
                confidence=confidence,
                confidence_level=level,
                needs_review=entity.confidence in {"uncertain", "unreadable"}
                or entity.occluded,
                detection_kind=entity.detection_kind,
                evidence=entity.evidence,
                occluded=entity.occluded,
                label=entity.label,
                mapped=geometry is not None,
            )
        )

    for relationship in document.relationships:
        confidence, level = confidence_for(relationship.confidence)
        target_geometry = geometry_by_id.get(relationship.to_id)
        relationship_type = _map_relationship(relationship.kind, target_geometry)
        result.relationships.append(
            SuggestedRelationship(
                id=f"{_ID_PREFIX}{relationship.id}",
                kind=relationship.kind,
                relationship_type=relationship_type,
                from_id=f"{_ID_PREFIX}{relationship.from_id}",
                to_id=f"{_ID_PREFIX}{relationship.to_id}",
                confidence=confidence,
                confidence_level=level,
                needs_review=relationship.confidence in {"uncertain", "unreadable"},
                detection_kind=relationship.detection_kind,
                evidence=relationship.evidence,
                mapped=relationship_type is not None,
            )
        )

    for note in document.diagram_relations:
        confidence, level = confidence_for(note.confidence)
        result.diagram_relations.append(
            DiagramRelationNote(
                id=f"{_ID_PREFIX}{note.id}",
                kind=note.kind,
                subject_ids=[f"{_ID_PREFIX}{ref}" for ref in note.subject_ids],
                statement=note.statement,
                confidence=confidence,
                confidence_level=level,
                needs_review=note.confidence in {"uncertain", "unreadable"},
                detection_kind=note.detection_kind,
                evidence=note.evidence,
            )
        )

    for item in document.text_items:
        confidence, level = confidence_for(item.confidence)
        result.text_items.append(
            TextNote(
                id=f"{_ID_PREFIX}{item.id}",
                text=item.text,
                role=item.role,
                region=_region(item.bbox, prepared),
                confidence=confidence,
                confidence_level=level,
                # Unreadable text is always a review item; that is the whole
                # point of reporting it rather than dropping it.
                needs_review=item.text is None
                or item.confidence in {"uncertain", "unreadable"},
                evidence=item.evidence,
            )
        )

    for note in document.uncertainties:
        result.uncertainties.append(
            UncertaintyNote(
                id=f"{_ID_PREFIX}{note.id}",
                subject_ids=[f"{_ID_PREFIX}{ref}" for ref in note.subject_ids],
                kind=note.kind,
                note=note.note,
                severity=note.severity,
            )
        )

    return result


def _map_geometry(kind: str):
    # Imported here rather than at module scope to keep the mapping table's
    # home the single obvious place to look when auditing what Model B can emit.
    from ..models.model_b_result import ENTITY_KIND_TO_GEOMETRY

    return ENTITY_KIND_TO_GEOMETRY.get(kind)


def _map_relationship(kind: str, target_geometry):
    from ..models.model_b_result import _relationship_type_for

    return _relationship_type_for(kind, target_geometry)


def _region(
    bbox_norm: tuple[float, float, float, float], prepared: PreparedImage
) -> BoundingRegion:
    x, y, width, height = prepared.project_bbox(bbox_norm)
    return BoundingRegion(norm=tuple(bbox_norm), x=x, y=y, width=width, height=height)
