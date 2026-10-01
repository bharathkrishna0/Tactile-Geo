"""Normalized Model B output and the Model A mapping tables.

Two rules govern everything in this module.

1. Model B never contributes authoritative geometry. A normalized bbox is a
   *region of interest*, not a measurement: it says "look here", and the
   normalized-pixel value exists so a human or a fusion heuristic can compare
   regions. It is never a point, radius, centre, or angle.
2. Every kind that Model A cannot express maps to `None` and lands in the
   unmapped bucket. Force-fitting `quadrilateral` into `rectangle`, or
   `symmetric_with` into `perpendicular_to`, would manufacture false precision
   in exactly the cases where a student is most likely to be harmed by it.

The mapping tables below are the complete, auditable list of what Model B can
ever contribute to Model A. Anything absent from them is unmapped by
construction, not by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.geometry import (
    ConfidenceLevel,
    GeometryType,
    RelationshipType,
    classify_confidence,
)

from ..model_b.schema import (
    CONFIDENCE_BAND_VALUES,
    DIAGRAM_RELATION_KIND_VALUES,
    ENTITY_KIND_VALUES,
    RELATIONSHIP_KIND_VALUES,
)

# --- Entity kind -> Model A GeometryType -------------------------------
# `None` means genuinely unmapped. See module docstring.
ENTITY_KIND_TO_GEOMETRY: dict[str, GeometryType | None] = {
    "point": GeometryType.POINT,
    "line_segment": GeometryType.LINE_SEGMENT,
    "ray": GeometryType.RAY,
    "circle": GeometryType.CIRCLE,
    "ellipse": GeometryType.ELLIPSE,
    "arc": GeometryType.ARC,
    "triangle": GeometryType.TRIANGLE,
    "rectangle": GeometryType.RECTANGLE,
    "quadrilateral": None,  # Model A has no general four-sided primitive
    "polygon": GeometryType.POLYGON,
    "angle": GeometryType.ANGLE,
    "axes": GeometryType.AXES,
    "arrow": GeometryType.ARROW,
    "dimension_annotation": None,  # text, not geometry
    "unknown": None,  # reporting "unknown" as a shape type would be a guess
}

# --- Relationship kind -> Model A RelationshipType ---------------------
RELATIONSHIP_KIND_TO_RELATIONSHIP: dict[str, RelationshipType | None] = {
    "connected_to": RelationshipType.CONNECTED_LINES,
    "intersects": RelationshipType.INTERSECTS,
    "parallel_to": RelationshipType.PARALLEL_LINES,
    "perpendicular_to": RelationshipType.PERPENDICULAR_LINES,
    # `point_on` is polymorphic and resolved against the target's type at
    # normalization time. See _relationship_type_for().
    "point_on": None,
    "endpoint_of": RelationshipType.ENDPOINT_OF,
    "center_of": RelationshipType.CENTER_OF,
    "angle_between": RelationshipType.ANGLE_BETWEEN,
    "angle_at": RelationshipType.ANGLE_ASSOCIATION,
    "labels": RelationshipType.LABEL_OBJECT,
    "contains": None,  # would require reversing and reinterpreting the pair
    "adjacent_to": None,
    "symmetric_with": None,
    "dimension_of": None,
}

# Diagram relations are inherently whole-diagram statements. Model A has no
# concept of any of them, so all are unmapped by construction and surfaced as
# advisory context only.
DIAGRAM_RELATION_MAPPED = False

# --- Confidence band -> uncalibrated float ------------------------------
# Midpoints, chosen to sit inside each band rather than at its optimistic
# edge. This value is NOT a probability; the UI labels it as a band.
CONFIDENCE_BAND_TO_FLOAT: dict[str, float] = {
    "certain": 0.90,
    "likely": 0.70,
    "uncertain": 0.40,
    "unreadable": 0.10,
}

REVIEW_BANDS = frozenset({"uncertain", "unreadable"})

# Provenance tag carried on every Model B suggestion. Deliberately
# provider-neutral: the gateway and the model are recorded separately on
# `ModelBResult` (`provider`, `requested_model`, `resolved_model`), so this
# constant does not have to change when the gateway does. What matters here is
# only that the element did not come from Model A.
SOURCE = "model_b"


def _relationship_type_for(
    kind: str, target_geometry: GeometryType | None
) -> RelationshipType | None:
    """Resolve `point_on` against the target's mapped geometry.

    Model A distinguishes a point on a circle from a point on a line because
    the two constrain different reconstruction paths. If the target's geometry
    is unknown, the honest answer is unmapped rather than a coin flip.
    """
    if kind == "point_on":
        if target_geometry is GeometryType.CIRCLE:
            return RelationshipType.POINT_ON_CIRCLE
        if target_geometry in (
            GeometryType.LINE_SEGMENT,
            GeometryType.RAY,
            GeometryType.ARC,
            GeometryType.POLYGON,
        ):
            return RelationshipType.POINT_ON_LINE
        return None
    return RELATIONSHIP_KIND_TO_RELATIONSHIP.get(kind)


@dataclass(frozen=True)
class BoundingRegion:
    """A region of interest, expressed in normalized and pixel coordinates.

    `x`, `y`, `width`, `height` are pixel coordinates in the ORIGINAL uploaded
    image, matching the frame Model A and the existing API expose. `norm` is the
    Model B output in the frame the vision model actually saw, retained so the
    projection stays auditable. The two frames differ whenever image preparation
    resized or padded, and keeping both makes that visible instead of silent.
    """

    norm: tuple[float, float, float, float]
    x: int
    y: int
    width: int
    height: int

    def as_model_a_bbox(self) -> tuple[int, int, int, int]:
        """Region in Model A's (x, y, w, h) convention."""
        return (self.x, self.y, self.width, self.height)

    def iou(self, other: "BoundingRegion") -> float:
        """Overlap with another region in the same frame.

        Used only for advisory agreement hints, never to rewrite geometry.
        """
        ax2, ay2 = self.x + self.width, self.y + self.height
        bx2, by2 = other.x + other.width, other.y + other.height
        overlap_w = min(ax2, bx2) - max(self.x, other.x)
        overlap_h = min(ay2, by2) - max(self.y, other.y)
        if overlap_w <= 0 or overlap_h <= 0:
            return 0.0
        intersection = overlap_w * overlap_h
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union > 0 else 0.0

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


@dataclass
class SuggestedEntity:
    """One advisory entity. `geometry_type` is None when unmapped."""

    id: str
    kind: str
    geometry_type: GeometryType | None
    region: BoundingRegion
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: str
    evidence: str
    occluded: bool
    label: str | None
    mapped: bool

    def to_element_dict(self) -> dict:
        """A Model A-shaped element dict carrying no geometry.

        `geometry` is empty on purpose. Model A's `DetectedElement` expects a
        geometry payload, and this is exactly where a future temptation to
        fabricate one lives, so the absence is explicit and greppable.
        """
        return {
            "id": self.id,
            "type": self.geometry_type.value if self.geometry_type else None,
            "geometry": {},
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "needs_review": self.needs_review,
            "source": SOURCE,
            "bbox": self.region.as_model_a_bbox(),
            "mapped": self.mapped,
        }


@dataclass
class SuggestedRelationship:
    id: str
    kind: str
    relationship_type: RelationshipType | None
    from_id: str
    to_id: str
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: str
    evidence: str
    mapped: bool

    def to_relationship_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.relationship_type.value if self.relationship_type else None,
            "element_ids": [self.from_id, self.to_id],
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "needs_review": self.needs_review,
            "explanation": self.evidence,
            "source": SOURCE,
            "mapped": self.mapped,
        }


@dataclass
class DiagramRelationNote:
    id: str
    kind: str
    subject_ids: list[str]
    statement: str
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: str
    evidence: str


@dataclass
class TextNote:
    id: str
    text: str | None
    role: str
    region: BoundingRegion
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    evidence: str


@dataclass
class UncertaintyNote:
    id: str
    subject_ids: list[str]
    kind: str
    note: str
    severity: str


@dataclass
class ModelBResult:
    """Complete normalized Model B output, ready to serialise.

    `truncated` is set when the provider stopped before finishing the
    document. A partial answer is still useful, but the teacher must be told
    that it is partial, because a partial answer reads exactly like a complete
    one that found nothing.

    Provenance is three fields, not one, because the gateway may be a router:
    `provider` is the gateway that accepted the request, `requested_model` is
    what configuration asked for, and `resolved_model` is what the gateway says
    actually ran. They differ for any router configuration, and reporting only
    the requested model would make a result impossible to trace back to the
    model that produced it.
    """

    schema_version: str
    diagram_kind: str
    diagram_description: str
    image_width: int
    image_height: int
    prepared_width: int
    prepared_height: int
    #: Gateway that served the request, e.g. `openrouter`.
    provider: str = ""
    #: Model identifier from configuration. May be a router.
    requested_model: str = ""
    #: Model that actually served the request. Never empty when requested_model
    #: is set.
    resolved_model: str = ""
    #: Upstream vendor behind the gateway, when reported. Empty otherwise.
    upstream_provider: str = ""
    #: Gateway-assigned id for this call. Not a credential and not a URL; it is
    #: the handle provider support asks for when a live run misbehaves.
    request_id: str = ""
    entities: list[SuggestedEntity] = field(default_factory=list)
    relationships: list[SuggestedRelationship] = field(default_factory=list)
    diagram_relations: list[DiagramRelationNote] = field(default_factory=list)
    text_items: list[TextNote] = field(default_factory=list)
    uncertainties: list[UncertaintyNote] = field(default_factory=list)
    image_readable: bool = True
    image_quality_issues: list[str] = field(default_factory=list)
    truncated: bool = False
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)

    @property
    def mapped_entity_count(self) -> int:
        return sum(1 for entity in self.entities if entity.mapped)

    @property
    def unmapped_entity_count(self) -> int:
        return sum(1 for entity in self.entities if not entity.mapped)

    @property
    def review_entity_count(self) -> int:
        return sum(1 for entity in self.entities if entity.needs_review)


def confidence_for(band: str) -> tuple[float, ConfidenceLevel]:
    value = CONFIDENCE_BAND_TO_FLOAT.get(band, 0.10)
    return value, classify_confidence(value)


def validate_mapping_tables() -> list[str]:
    """Every schema vocabulary value must appear in a mapping table.

    Guards the failure mode where someone adds an entity kind to the schema,
    ships it, and has it silently fall through to unmapped at runtime.
    """
    problems: list[str] = []
    for kind in ENTITY_KIND_VALUES:
        if kind not in ENTITY_KIND_TO_GEOMETRY:
            problems.append(f"entity kind unmapped in table: {kind}")
    for kind in RELATIONSHIP_KIND_VALUES:
        if kind not in RELATIONSHIP_KIND_TO_RELATIONSHIP and kind != "point_on":
            problems.append(f"relationship kind unmapped in table: {kind}")
    for band in CONFIDENCE_BAND_VALUES:
        if band not in CONFIDENCE_BAND_TO_FLOAT:
            problems.append(f"confidence band unmapped: {band}")
    for kind in DIAGRAM_RELATION_KIND_VALUES:
        if kind not in _KNOWN_DIAGRAM_RELATIONS:
            problems.append(f"diagram relation kind unhandled: {kind}")
    return problems


_KNOWN_DIAGRAM_RELATIONS = frozenset(DIAGRAM_RELATION_KIND_VALUES)
