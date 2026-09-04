from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class SessionCreated(BaseModel):
    session_id: str
    filename: str
    created_at: datetime

class ProcessingParams(BaseModel):
    edge_sensitivity: int = Field(default=50, ge=0, le=100)

class DetectedShape(BaseModel):
    type: Literal["line", "contour"]
    points: list[tuple[int, int]]

class DetectedLabel(BaseModel):
    text: str
    braille: str
    bbox: list[tuple[int, int]]
    confidence: float
    anchor_type: Literal["vertex", "edge"]
    anchor: tuple[int, int]
    position: tuple[int, int]
    offset: tuple[int, int]
    collision_adjusted: bool

class ProcessedSession(BaseModel):
    session_id: str
    preview_svg: str
    detected_shapes: list[DetectedShape]
    detected_labels: list[DetectedLabel]
    processing_params: ProcessingParams

class QualityIssueSchema(BaseModel):
    check: str
    severity: str
    message: str
    value: float | None = None
    threshold: float | None = None

class QualityReportSchema(BaseModel):
    passes_gate: bool
    issues: list[QualityIssueSchema]
    image_width: int
    image_height: int

class DetectedElementSchema(BaseModel):
    id: str
    type: str
    geometry: dict
    confidence: float
    confidence_level: str
    needs_review: bool
    source: str
    bbox: tuple[int, int, int, int] | None = None
    semantic_properties: dict = Field(default_factory=dict)
    associated_label_id: str | None = None

class ElementRelationshipSchema(BaseModel):
    id: str
    type: str
    element_ids: list[str]
    confidence: float
    confidence_level: str
    needs_review: bool
    explanation: str | None = None
    properties: dict = Field(default_factory=dict)

class TransformationExplanationSchema(BaseModel):
    stage: str
    element_id: str | None = None
    message: str

class SemanticGeometrySchema(BaseModel):
    elements: list[DetectedElementSchema]
    relationships: list[ElementRelationshipSchema]
    image_width: int
    image_height: int
    element_count: int
    low_confidence_count: int
    review_flags: list[str]
    explanations: list[TransformationExplanationSchema] = Field(default_factory=list)

class SimplificationActionSchema(BaseModel):
    element_id: str
    action: str
    detail: str

class SimplifiedGeometrySchema(BaseModel):
    elements: list[DetectedElementSchema]
    relationships: list[ElementRelationshipSchema]
    actions: list[SimplificationActionSchema]
    removed_count: int
    merged_count: int = 0
    explanations: list[str] = Field(default_factory=list)

class QAIssueSchema(BaseModel):
    check: str
    severity: str
    message: str
    element_id: str | None = None

class QAReportSchema(BaseModel):
    overall_score: float
    passes: bool
    issues: list[QAIssueSchema]
    element_checks: int
    score_0_100: int = 0

class EnhancedProcessedSession(ProcessedSession):
    quality_report: QualityReportSchema | None = None
    semantic_geometry: SemanticGeometrySchema | None = None
    simplified_geometry: SimplifiedGeometrySchema | None = None
    qa_report: QAReportSchema | None = None
    tactile_svg: str | None = None
