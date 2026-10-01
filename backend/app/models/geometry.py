from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GeometryType(str, Enum):
    POINT = "point"
    LINE_SEGMENT = "line_segment"
    RAY = "ray"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    ARC = "arc"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    ANGLE = "angle"
    AXES = "axes"
    ARROW = "arrow"
    TEXT_LABEL = "text_label"


class RelationshipType(str, Enum):
    POINT_ON_LINE = "point_on_line"
    POINT_ON_CIRCLE = "point_on_circle"
    POINT_INSIDE_SHAPE = "point_inside_shape"
    POINT_OUTSIDE_SHAPE = "point_outside_shape"
    CONNECTED_LINES = "connected_lines"
    INTERSECTS = "intersects"
    PARALLEL_LINES = "parallel_lines"
    PERPENDICULAR_LINES = "perpendicular_lines"
    CIRCLE_CENTER = "circle_center"
    CENTER_OF = "center_of"
    ENDPOINT_OF = "endpoint_of"
    ANGLE_BETWEEN = "angle_between"
    ANGLE_ASSOCIATION = "angle_association"
    LABEL_OBJECT = "label_object"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def classify_confidence(score: float) -> ConfidenceLevel:
    if score >= 0.8:
        return ConfidenceLevel.HIGH
    if score >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


@dataclass
class DetectedElement:
    id: str
    type: GeometryType
    geometry: dict
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    source: str
    bbox: tuple[int, int, int, int] | None = None
    semantic_properties: dict[str, object] = field(default_factory=dict)
    associated_label_id: str | None = None
    provenance: str | None = None


@dataclass
class ElementRelationship:
    id: str
    type: RelationshipType
    element_ids: list[str]
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    explanation: str | None = None
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class TransformationExplanation:
    stage: str
    element_id: str | None
    message: str


@dataclass
class SemanticGeometry:
    elements: list[DetectedElement] = field(default_factory=list)
    relationships: list[ElementRelationship] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    element_count: int = 0
    low_confidence_count: int = 0
    review_flags: list[str] = field(default_factory=list)
    explanations: list[TransformationExplanation] = field(default_factory=list)
