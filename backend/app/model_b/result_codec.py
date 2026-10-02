"""Lossless JSON encoding of `ModelBResult`, for durable jobs and the cache."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.models.geometry import ConfidenceLevel, GeometryType, RelationshipType
from app.models.model_b_result import (
    BoundingRegion,
    DiagramRelationNote,
    ModelBResult,
    SuggestedEntity,
    SuggestedRelationship,
    TextNote,
    UncertaintyNote,
)


def result_to_dict(result: ModelBResult) -> dict[str, Any]:
    return asdict(result, dict_factory=lambda items: {key: _plain(value) for key, value in items})


def _plain(value: Any) -> Any:
    if isinstance(value, (GeometryType, RelationshipType, ConfidenceLevel)):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def _region(data: dict[str, Any]) -> BoundingRegion:
    return BoundingRegion(
        norm=tuple(data["norm"]), x=data["x"], y=data["y"], width=data["width"], height=data["height"],
    )


def result_from_dict(data: dict[str, Any]) -> ModelBResult:
    entities = [
        SuggestedEntity(
            **{
                **item,
                "geometry_type": GeometryType(item["geometry_type"]) if item["geometry_type"] else None,
                "region": _region(item["region"]),
                "confidence_level": ConfidenceLevel(item["confidence_level"]),
            }
        )
        for item in data.get("entities", [])
    ]
    relationships = [
        SuggestedRelationship(
            **{
                **item,
                "relationship_type": RelationshipType(item["relationship_type"]) if item["relationship_type"] else None,
                "confidence_level": ConfidenceLevel(item["confidence_level"]),
            }
        )
        for item in data.get("relationships", [])
    ]
    diagram_relations = [
        DiagramRelationNote(**{**item, "confidence_level": ConfidenceLevel(item["confidence_level"])})
        for item in data.get("diagram_relations", [])
    ]
    text_items = [
        TextNote(**{**item, "region": _region(item["region"]), "confidence_level": ConfidenceLevel(item["confidence_level"])})
        for item in data.get("text_items", [])
    ]
    uncertainties = [UncertaintyNote(**item) for item in data.get("uncertainties", [])]
    scalars = {
        key: value
        for key, value in data.items()
        if key not in {"entities", "relationships", "diagram_relations", "text_items", "uncertainties"}
    }
    return ModelBResult(
        **scalars,
        entities=entities,
        relationships=relationships,
        diagram_relations=diagram_relations,
        text_items=text_items,
        uncertainties=uncertainties,
    )
