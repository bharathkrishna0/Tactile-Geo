"""Model B structured-output contract.

These Pydantic models are the SINGLE SOURCE OF TRUTH for the Model B response.
`json_schema.py` derives the provider-facing JSON Schema from them, so the wire
contract sent to the model and the contract validated locally can never drift.

Design constraints this file must respect (see DESIGN section 9.2):

* Only the strict structured-output JSON Schema subset is used: string, number, integer,
  boolean, object, array, null; object properties/required/additionalProperties;
  string enum/format; number enum/minimum/maximum; array items/prefixItems/
  minItems/maxItems.
* `pattern`, `maxLength`, `allOf`, `oneOf`, `not`, `if/then` are NOT available,
  so id formats are constrained with bounded `enum` lists and length limits are
  enforced in `validator.py` instead.
* Identifiers are bounded enums rather than free strings, which is what makes
  referential integrity machine-checkable without regex.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "model_b.diag.v1"

# Bounded id spaces. The enum length and the matching array maxItems are kept
# equal so a full response can never run out of ids.
MAX_ENTITIES = 40
MAX_RELATIONSHIPS = 60
MAX_DIAGRAM_RELATIONS = 30
MAX_TEXT_ITEMS = 40
MAX_UNCERTAINTIES = 30

Normalized = Annotated[float, Field(ge=0.0, le=1.0)]
# Fixed-length 4-tuple -> rendered as prefixItems + minItems/maxItems.
BoundingBox = tuple[Normalized, Normalized, Normalized, Normalized]

ConfidenceBand = Literal["certain", "likely", "uncertain", "unreadable"]
DetectionKind = Literal["observed", "inferred"]


def id_space(prefix: str, count: int) -> list[Literal]:
    """Build a bounded id enum, e.g. ['e1', 'e2', ... 'e40']."""
    return [f"{prefix}{index}" for index in range(1, count + 1)]  # type: ignore[misc]


ENTITY_IDS = id_space("e", MAX_ENTITIES)
RELATIONSHIP_IDS = id_space("r", MAX_RELATIONSHIPS)
DIAGRAM_RELATION_IDS = id_space("d", MAX_DIAGRAM_RELATIONS)
TEXT_IDS = id_space("t", MAX_TEXT_ITEMS)
UNCERTAINTY_IDS = id_space("u", MAX_UNCERTAINTIES)

DIAGRAM_KINDS = Literal[
    "single_shape",
    "multiple_shapes",
    "labelled_figure",
    "geometric_construction",
    "coordinate_plane",
    "worksheet_with_questions",
    "table_or_grid",
    "not_a_geometry_diagram",
    "unclear",
]

# Superset of Model A's GeometryType. `quadrilateral`, `dimension_annotation` and
# `unknown` have no Model A counterpart; see DESIGN section 10.2.
ENTITY_KINDS = Literal[
    "point",
    "line_segment",
    "ray",
    "circle",
    "ellipse",
    "arc",
    "triangle",
    "rectangle",
    "quadrilateral",
    "polygon",
    "angle",
    "axes",
    "arrow",
    "dimension_annotation",
    "unknown",
]

# `adjacent_to`, `symmetric_with` and `dimension_of` have NO Model A counterpart
# and are routed to the unmapped bucket rather than force-fitted. DESIGN 10.3.
RELATIONSHIP_KINDS = Literal[
    "connected_to",
    "intersects",
    "parallel_to",
    "perpendicular_to",
    "point_on",
    "endpoint_of",
    "center_of",
    "angle_between",
    "angle_at",
    "labels",
    "contains",
    "adjacent_to",
    "symmetric_with",
    "dimension_of",
]

DIAGRAM_RELATION_KINDS = Literal[
    "right_triangle",
    "isosceles_triangle",
    "equilateral_triangle",
    "tangent",
    "secant",
    "congruent_marking",
    "similar_triangle",
    "cyclic",
    "right_angle_marker_present",
    "parallel_marker_present",
    "scale_drawing",
    "has_dimension_annotations",
    "has_axis_labels",
    "partially_labeled",
    "cropped",
    "blurred",
    "low_contrast",
    "handwriting_detected",
]

TEXT_ROLES = Literal[
    "vertex_label",
    "side_label",
    "angle_label",
    "measurement",
    "axis_label",
    "equation",
    "instruction",
    "caption",
    "legend",
    "unknown",
]

UNCERTAINTY_KINDS = Literal[
    "illegible_text",
    "ambiguous_symbol",
    "occluded_geometry",
    "missing_edge",
    "ambiguous_vertex_count",
    "low_contrast",
    "blur",
    "cropping",
    "possible_second_interpretation",
    "conflicting_notation",
    "partial_diagram",
]

UNCERTAINTY_SEVERITIES = Literal["blocking_review", "worth_review", "informational"]

IMAGE_QUALITY_ISSUES = Literal[
    "blur",
    "low_contrast",
    "cropping",
    "skew",
    "handwriting",
    "print_artifact",
    "jpeg_compression",
    "none",
]

_STRICT = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class _Base(BaseModel):
    model_config = _STRICT


class ImageFrame(_Base):
    """Dimensions of the PREPARED image the vision model actually saw."""

    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)


class DiagramSummary(_Base):
    kind: DIAGRAM_KINDS
    description: str


class Entity(_Base):
    id: Literal[*ENTITY_IDS]  # type: ignore[valid-type]
    kind: ENTITY_KINDS
    detection_kind: DetectionKind
    """Whether the entity is directly visible, or concluded from other evidence."""
    bbox: BoundingBox
    confidence: ConfidenceBand
    evidence: str
    occluded: bool
    label: str | None = None
    """Attached text, only when that exact text is legible. Otherwise null."""


class Relationship(_Base):
    id: Literal[*RELATIONSHIP_IDS]  # type: ignore[valid-type]
    kind: RELATIONSHIP_KINDS
    from_id: str
    to_id: str
    detection_kind: DetectionKind
    confidence: ConfidenceBand
    evidence: str


class DiagramRelation(_Base):
    id: Literal[*DIAGRAM_RELATION_IDS]  # type: ignore[valid-type]
    kind: DIAGRAM_RELATION_KINDS
    subject_ids: list[str] = Field(max_length=12)
    statement: str
    detection_kind: DetectionKind
    confidence: ConfidenceBand
    evidence: str


class TextItem(_Base):
    id: Literal[*TEXT_IDS]  # type: ignore[valid-type]
    text: str | None
    """Null whenever any character is uncertain. Never reconstruct text from context."""
    bbox: BoundingBox
    role: TEXT_ROLES
    confidence: ConfidenceBand
    evidence: str


class Uncertainty(_Base):
    id: Literal[*UNCERTAINTY_IDS]  # type: ignore[valid-type]
    subject_ids: list[str] = Field(max_length=12)
    kind: UNCERTAINTY_KINDS
    note: str
    severity: UNCERTAINTY_SEVERITIES


class ImageQuality(_Base):
    """Advisory only. app/services/image_quality.py remains the sole gate."""

    readable: bool
    issues: list[IMAGE_QUALITY_ISSUES] = Field(max_length=8)

class ModelBDocument(_Base):
    """The complete Model B response document."""

    schema_version: Literal[SCHEMA_VERSION]
    image: ImageFrame
    diagram_summary: DiagramSummary
    entities: list[Entity] = Field(max_length=MAX_ENTITIES)
    relationships: list[Relationship] = Field(max_length=MAX_RELATIONSHIPS)
    diagram_relations: list[DiagramRelation] = Field(max_length=MAX_DIAGRAM_RELATIONS)
    text_items: list[TextItem] = Field(max_length=MAX_TEXT_ITEMS)
    uncertainties: list[Uncertainty] = Field(max_length=MAX_UNCERTAINTIES)
    image_quality: ImageQuality


# Vocabulary lists reused by the prompt so the two never disagree. They are
# derived from the schema rather than hand-written.
ENTITY_KIND_VALUES = list(Entity.model_fields["kind"].annotation.__args__)
RELATIONSHIP_KIND_VALUES = list(Relationship.model_fields["kind"].annotation.__args__)
DIAGRAM_RELATION_KIND_VALUES = list(
    DiagramRelation.model_fields["kind"].annotation.__args__
)
DIAGRAM_KIND_VALUES = list(DiagramSummary.model_fields["kind"].annotation.__args__)
TEXT_ROLE_VALUES = list(TextItem.model_fields["role"].annotation.__args__)
UNCERTAINTY_KIND_VALUES = list(Uncertainty.model_fields["kind"].annotation.__args__)
CONFIDENCE_BAND_VALUES = list(Entity.model_fields["confidence"].annotation.__args__)
