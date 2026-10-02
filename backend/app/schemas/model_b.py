"""Model B API response schemas.

These are separate from `app/model_b/schema.py` on purpose. The model contract
and the wire contract evolve for different reasons and different consumers: one
is a prompt-stability concern, the other a frontend-compatibility concern. Merging
them would mean a cosmetic rename in the UI could invalidate every cached prompt
prefix.

Every `b_`-prefixed id and every `bbox` here is advisory. The schemas carry no
field that could be mistaken for a measurement, and no field from which a caller
could reconstruct authoritative geometry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.session import EnhancedProcessedSession

from app.models.geometry import ConfidenceLevel
from app.models.model_b_job import JobStatus


class ModelBAvailability(BaseModel):
    """Whether a teacher can request an analysis right now.

    Reports readiness plus the non-secret half of the configuration. A model
    identifier and a gateway name are not credentials and a teacher needs them:
    an advisory result that cannot be attributed to a model cannot be audited,
    and "Model B is a chatbot" is not an answer a teacher can act on.

    `reason` still distinguishes "off", "no key" and "no model" so setup
    problems are diagnosable without exposing key *state* beyond that, and
    `api_key` has no field here at all.
    """

    available: bool
    enabled: bool
    reason: str | None = None
    #: Gateway serving Model B, e.g. `openrouter`. Never a credential.
    provider: str = ""
    #: Configured model or router. May be a router whose upstream varies.
    model: str = ""


class ModelBBoundingRegion(BaseModel):
    """Advisory region in the original image's pixel frame.

    `norm` is the raw normalized output in the frame the vision model saw. Both
    are exposed so a teacher or a fusion heuristic can audit the projection
    rather than trust it.
    """

    norm: tuple[float, float, float, float]
    x: int
    y: int
    width: int
    height: int
    advisory_only: Literal[True] = True


class ModelBEntity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    kind: str
    geometry_type: str | None
    region: ModelBBoundingRegion
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: Literal["observed", "inferred"]
    evidence: str
    occluded: bool
    label: str | None
    mapped: bool
    advisory_only: Literal[True] = True


class ModelBRelationship(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    kind: str
    relationship_type: str | None
    from_id: str
    to_id: str
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: Literal["observed", "inferred"]
    evidence: str
    mapped: bool
    advisory_only: Literal[True] = True


class ModelBDiagramRelation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    kind: str
    subject_ids: list[str]
    statement: str
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    detection_kind: Literal["observed", "inferred"]
    evidence: str
    advisory_only: Literal[True] = True


class ModelBTextItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    text: str | None
    role: str
    region: ModelBBoundingRegion
    confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    evidence: str
    advisory_only: Literal[True] = True


class ModelBUncertainty(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    subject_ids: list[str]
    kind: str
    note: str
    severity: Literal["blocking_review", "worth_review", "informational"]


class ModelBAnalysis(BaseModel):
    """The normalized advisory analysis."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: str
    diagram_kind: str
    diagram_description: str
    image_width: int
    image_height: int
    prepared_width: int
    prepared_height: int
    entities: list[ModelBEntity]
    relationships: list[ModelBRelationship]
    diagram_relations: list[ModelBDiagramRelation]
    text_items: list[ModelBTextItem]
    uncertainties: list[ModelBUncertainty]
    image_readable: bool
    image_quality_issues: list[str]
    truncated: bool
    finish_reason: str
    mapped_entity_count: int
    unmapped_entity_count: int
    review_entity_count: int
    #: Gateway that served the request, e.g. `openrouter`.
    provider: str = ""
    #: Model identifier from configuration. May be a router.
    requested_model: str = ""
    #: Model that actually served this result. Differs from `requested_model`
    #: whenever a router is configured, which is the common case on the free
    #: tier, so a result is always attributable to a real model.
    resolved_model: str = ""
    #: Upstream vendor behind the gateway, when the gateway reported one.
    upstream_provider: str = ""
    #: Gateway-assigned id for the call that produced this result.
    request_id: str = ""
    #: Token accounting as reported by the provider. Numbers only; absent keys
    #: simply mean the provider did not report them.
    usage: dict[str, float | int] = Field(default_factory=dict)
    validation_warnings: list[str] = Field(default_factory=list)
    advisory_only: Literal[True] = True


class ModelBErrorSchema(BaseModel):
    """A failure, in terms a teacher can act on.

    `code` is a stable machine-readable reason so the UI can pick a state
    without matching on English prose, and `retry_after_s` is present only when
    the provider told us how long to wait.
    """

    code: str
    message: str
    retryable: bool = False
    retry_after_s: float | None = None


class ModelBJobStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    session_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    #: Gateway and model the job was submitted to, known before the call
    #: returns. The model that actually served it appears on the result.
    provider: str = ""
    model: str = ""
    #: True when an identical earlier analysis was reused without a provider call.
    cache_hit: bool = False
    result: ModelBAnalysis | None = None
    error: ModelBErrorSchema | None = None


class ModelBAgreementHint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_b_id: str
    model_a_id: str
    model_b_kind: str
    model_a_type: str
    iou: float
    verdict: Literal["agrees", "weak_overlap", "type_mismatch"]
    advisory_only: Literal[True] = True


class ModelBCandidateAddition(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_b_id: str
    suggested_type: str | None
    region: tuple[int, int, int, int]
    reason: str
    # Fixed true. This field exists so no consumer can accidentally treat an
    # addition as applied; it is not a toggle.
    requires_teacher_approval: Literal[True] = True
    advisory_only: Literal[True] = True


class ModelBDisagreement(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    kind: Literal["type_mismatch", "model_a_only"]
    model_b_id: str
    model_a_id: str | None
    detail: str
    advisory_only: Literal[True] = True


class ModelBEntityReview(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_b_id: str
    model_b_kind: str
    model_a_id: str | None
    correspondence: Literal["strong", "weak", "none"]
    adds_semantics: bool
    contradicts_model_a: bool
    requires_review: bool
    reason: str
    advisory_only: Literal[True] = True


class ModelBDecisionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_b_id: str = Field(min_length=1, max_length=64)
    decision: Literal["accept", "reject", "defer"]
    note: str | None = Field(default=None, max_length=500)


class ModelBDecision(BaseModel):
    """A teacher's recorded decision on one Model B finding.

    Recording ``accept`` changes nothing by itself. Accepted findings reach the
    tactile output only through the explicit apply step, and only where Model A
    geometry backs them.
    """

    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    model_b_id: str
    decision: Literal["accept", "reject", "defer"]
    note: str | None = None
    decided_at: datetime
    applied_to_geometry: Literal[False] = False


class ModelBFusionReport(BaseModel):
    """Read-only reconciliation. Contains no field that can change Model A."""

    model_config = ConfigDict(protected_namespaces=())

    summary: dict[str, int]
    agreements: list[ModelBAgreementHint]
    entity_reviews: list[ModelBEntityReview] = []
    #: Latest teacher decision per Model B id, for this job.
    decisions: dict[str, ModelBDecision] = {}
    candidate_additions: list[ModelBCandidateAddition]
    disagreements: list[ModelBDisagreement]
    diagram_relations: list[dict]
    text_notes: list[dict]
    uncertainties: list[dict]
    advisory_only: Literal[True] = True


class ModelBApplyOutcome(BaseModel):
    """What happened to one accepted finding when building Semantic Geometry v2."""

    model_config = ConfigDict(protected_namespaces=())

    model_b_id: str
    applied: bool
    change: Literal["set_type", "attach_label", "confirm"] | None
    model_a_id: str | None
    detail: str


class ModelBApplyResult(BaseModel):
    """Regenerated session after applying the teacher-accepted findings.

    Coordinates in ``session`` are Model A's; only types, label associations
    and teacher confirmation can differ from the Model A result.
    """

    model_config = ConfigDict(protected_namespaces=())

    session: EnhancedProcessedSession
    outcomes: list[ModelBApplyOutcome]
    reverted: list[str]
