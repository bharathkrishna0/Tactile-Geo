"""Model B API.

Four endpoints, all additive under `/api`:

    GET    /api/model-b/status                      readiness, no config leaked
    POST   /api/sessions/{id}/model-b               enqueue, 202 immediately
    GET    /api/sessions/{id}/model-b/{job_id}      poll
    DELETE /api/sessions/{id}/model-b/{job_id}      cancel if not started

The worker runs as a Starlette background task. Starlette executes sync
background callables in a threadpool, which matters because `analyze_image` is
blocking throughout (OpenCV decode, then a synchronous HTTP call to the Model B
provider). Running it on the event loop would stall every other request for the
duration of that call, turning an optional feature into a denial-of-service
vector for Model A.

Model A's endpoints and payloads are untouched. Nothing here is required for the
product to function.
"""

from __future__ import annotations

import logging
import time
from typing import NoReturn

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import MODEL_B_MAX_CONCURRENT_JOBS, model_b_settings
from app.models.model_b_job import JobStatus, ModelBJob, ModelBJobError
from app.models.model_b_result import ModelBResult
from app.model_b.errors import (
    ModelBApiError,
    ModelBCancelled,
    ModelBDisabled,
    ModelBError,
    ModelBMisconfigured,
    ModelBMalformedResponse,
    ModelBRateLimited,
    ModelBTimeout,
    ModelBValidationError,
)
from app.model_b.fusion import FusionReport
from app.model_b.service import analyze_image, is_available
from app.models.geometry import SemanticGeometry
from app.schemas.model_b import (
    ModelBAvailability,
    ModelBAnalysis,
    ModelBAgreementHint,
    ModelBBoundingRegion,
    ModelBCandidateAddition,
    ModelBDiagramRelation,
    ModelBDisagreement,
    ModelBEntity,
    ModelBErrorSchema,
    ModelBFusionReport,
    ModelBJobStatus,
    ModelBRelationship,
    ModelBTextItem,
    ModelBUncertainty,
)
from app.services.editing import semantic_from_dict
from app.services.model_b_store import model_b_job_store
from app.services.session_store import session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["model-b"])

# Exceptions that are the user's or the deployment's situation rather than a
# bug. Everything else is a genuine 500 and should surface as one.
# Lookup is by exact type, not by isinstance, so `ModelBRateLimited` needs its
# own entry: it is a `ModelBApiError` subclass but must surface as 429, because
# "wait and retry" and "the request failed" call for different teacher
# behaviour.
_TYPED_ERROR_STATUS = {
    ModelBDisabled: 503,
    ModelBMisconfigured: 503,
    ModelBTimeout: 504,
    ModelBRateLimited: 429,
    ModelBApiError: 502,
    ModelBMalformedResponse: 502,
    ModelBValidationError: 502,
    ModelBCancelled: 409,
}


def _error_payload(error: ModelBJobError) -> ModelBErrorSchema:
    """Serialize a stored job failure.

    Takes the job dataclass, not the exception: by the time a failure reaches
    here it has already been copied into durable job state, and the messages on
    it are the ones chosen for a teacher rather than for a log.
    """
    return ModelBErrorSchema(
        code=error.code,
        message=error.message,
        retryable=error.retryable,
        retry_after_s=error.retry_after_s,
    )


def _region_schema(region) -> ModelBBoundingRegion:
    return ModelBBoundingRegion(
        norm=tuple(region.norm), x=region.x, y=region.y, width=region.width, height=region.height
    )


def _analysis_schema(result: ModelBResult) -> ModelBAnalysis:
    """Project the dataclass result onto the wire schema.

    The advisory flag is set by the schema's default, not passed in. That way a
    future refactor cannot forget it on one field and quietly make one part of
    the payload look authoritative.
    """
    return ModelBAnalysis(
        schema_version=result.schema_version,
        diagram_kind=result.diagram_kind,
        diagram_description=result.diagram_description,
        image_width=result.image_width,
        image_height=result.image_height,
        prepared_width=result.prepared_width,
        prepared_height=result.prepared_height,
        entities=[
            ModelBEntity(
                id=entity.id,
                kind=entity.kind,
                geometry_type=entity.geometry_type.value if entity.geometry_type else None,
                region=_region_schema(entity.region),
                confidence=entity.confidence,
                confidence_level=entity.confidence_level,
                needs_review=entity.needs_review,
                detection_kind=entity.detection_kind,
                evidence=entity.evidence,
                occluded=entity.occluded,
                label=entity.label,
                mapped=entity.mapped,
            )
            for entity in result.entities
        ],
        relationships=[
            ModelBRelationship(
                id=relationship.id,
                kind=relationship.kind,
                relationship_type=(
                    relationship.relationship_type.value if relationship.relationship_type else None
                ),
                from_id=relationship.from_id,
                to_id=relationship.to_id,
                confidence=relationship.confidence,
                confidence_level=relationship.confidence_level,
                needs_review=relationship.needs_review,
                detection_kind=relationship.detection_kind,
                evidence=relationship.evidence,
                mapped=relationship.mapped,
            )
            for relationship in result.relationships
        ],
        diagram_relations=[
            ModelBDiagramRelation(
                id=note.id,
                kind=note.kind,
                subject_ids=note.subject_ids,
                statement=note.statement,
                confidence=note.confidence,
                confidence_level=note.confidence_level,
                needs_review=note.needs_review,
                detection_kind=note.detection_kind,
                evidence=note.evidence,
            )
            for note in result.diagram_relations
        ],
        text_items=[
            ModelBTextItem(
                id=item.id,
                text=item.text,
                role=item.role,
                region=_region_schema(item.region),
                confidence=item.confidence,
                confidence_level=item.confidence_level,
                needs_review=item.needs_review,
                evidence=item.evidence,
            )
            for item in result.text_items
        ],
        uncertainties=[
            ModelBUncertainty(
                id=note.id,
                subject_ids=note.subject_ids,
                kind=note.kind,
                note=note.note,
                severity=note.severity,
            )
            for note in result.uncertainties
        ],
        image_readable=result.image_readable,
        image_quality_issues=result.image_quality_issues,
        truncated=result.truncated,
        finish_reason=result.finish_reason,
        provider=result.provider,
        requested_model=result.requested_model,
        resolved_model=result.resolved_model,
        upstream_provider=result.upstream_provider,
        usage=result.usage,
        mapped_entity_count=result.mapped_entity_count,
        unmapped_entity_count=result.unmapped_entity_count,
        review_entity_count=result.review_entity_count,
        validation_warnings=result.validation_warnings,
    )


def _job_schema(job: ModelBJob) -> ModelBJobStatus:
    return ModelBJobStatus(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        provider=job.provider,
        model=job.model,
        result=_analysis_schema(job.result) if isinstance(job.result, ModelBResult) else None,
        error=_error_payload(job.error) if job.error else None,
    )


def _raise_http(error: ModelBError) -> NoReturn:
    status_code = _TYPED_ERROR_STATUS.get(type(error), 502)
    detail = error.detail
    if isinstance(error, ModelBApiError) and not error.retryable:
        # Provider prose can quote the request back. A teacher gets a fixed
        # sentence instead, and the specifics stay in the server log.
        detail = "The Model B request failed. You can try again or continue with the Model A result."
    raise HTTPException(status_code=status_code, detail=detail) from error


def _run_job(job: ModelBJob, image_bytes: bytes) -> None:
    """Background worker. Never raises; every failure becomes job state."""
    store = model_b_job_store
    # Claiming the job is one lock acquisition. A separate status check plus
    # `mark_running()` would let a cancel slip into the gap and be overwritten,
    # resurrecting a job the teacher was told they had stopped.
    current = store.begin(job.job_id)
    if current is None:
        return

    started = time.perf_counter()
    try:
        result = analyze_image(image_bytes, model_b_settings())
    except ModelBError as error:
        logger.warning(
            "Model B job %s failed after %.0fms: %s (%s)",
            job.job_id,
            (time.perf_counter() - started) * 1000,
            error.detail,
            error.reason,
        )
        target = store.get(job.job_id)
        if target is not None and not target.status.is_terminal:
            target.mark_failed(
                ModelBJobError(
                    code=error.reason,
                    message=error.detail,
                    retryable=getattr(error, "retryable", False),
                    retry_after_s=getattr(error, "retry_after_s", None),
                )
            )
        return
    except Exception:  # noqa: BLE001 - a worker must never take the process down
        logger.exception("Unexpected Model B failure in job %s", job.job_id)
        target = store.get(job.job_id)
        if target is not None and not target.status.is_terminal:
            target.mark_failed(
                ModelBJobError(
                    code="internal_error",
                    message="Model B failed unexpectedly.",
                    retryable=True,
                )
            )
        return

    # PROJECT.md section 13 observability. Counts and timings only: no image
    # data, no response body, and nothing that could carry the key.
    logger.info(
        "Model B job %s completed in %.0fms: provider=%s, requested_model=%s, "
        "resolved_model=%s, upstream=%s, %d entities (%d mapped, %d unmapped, "
        "%d to review), %d relations, %d text items, %d uncertainties, "
        "readable=%s, truncated=%s, warnings=%d, usage=%s",
        job.job_id,
        (time.perf_counter() - started) * 1000,
        result.provider,
        result.requested_model,
        result.resolved_model,
        result.upstream_provider or "<not reported>",
        len(result.entities),
        result.mapped_entity_count,
        result.unmapped_entity_count,
        result.review_entity_count,
        len(result.diagram_relations),
        len(result.text_items),
        len(result.uncertainties),
        result.image_readable,
        result.truncated,
        len(result.validation_warnings),
        result.usage or "none",
    )
    for warning in result.validation_warnings:
        logger.debug("Model B job %s validation warning: %s", job.job_id, warning)

    target = store.get(job.job_id)
    if target is not None and not target.status.is_terminal:
        target.mark_completed(result)


@router.get("/model-b/status", response_model=ModelBAvailability)
def model_b_status() -> ModelBAvailability:
    """Report readiness, plus the provider and model that would be used.

    Never reveals the key, whether a key exists in any form other than the
    coarse `reason`, or quota. Provider and model are included because a model
    identifier is not a credential and a teacher cannot reason about an advisory
    result that will not say which model produced it.
    """
    settings = model_b_settings()
    if not settings.enabled:
        reason = "Model B is turned off on this server."
    elif not settings.api_key:
        reason = "Model B is enabled but no OpenRouter API key is configured."
    elif not settings.model:
        reason = "Model B has no model configured."
    else:
        reason = None
    return ModelBAvailability(
        available=is_available(settings),
        enabled=settings.enabled,
        reason=reason,
        provider=settings.provider,
        model=settings.model,
    )


@router.post(
    "/sessions/{session_id}/model-b",
    response_model=ModelBJobStatus,
    status_code=202,
)
def request_model_b(
    session_id: str, background_tasks: BackgroundTasks
) -> ModelBJobStatus:
    """Enqueue an advisory analysis. Returns immediately with a queued job."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload the image again.")

    settings = model_b_settings()
    try:
        settings.require_available()
    except ModelBError as error:
        _raise_http(error)

    if model_b_job_store.has_active_job(session_id):
        raise HTTPException(
            status_code=409,
            detail="A Model B analysis is already running for this session.",
        )
    if model_b_job_store.active_job_count() >= MODEL_B_MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail="Model B is busy. Wait for the current analysis to finish.",
        )

    try:
        image_bytes = session.original_image_path.read_bytes()
    except OSError as error:
        raise HTTPException(status_code=410, detail="The uploaded image is no longer available.") from error
    if not image_bytes:
        raise HTTPException(status_code=410, detail="The uploaded image is no longer available.")

    job = model_b_job_store.create(
        session_id, model=settings.model, provider=settings.provider
    )
    # Left QUEUED on purpose. `_run_job` transitions it to RUNNING, which keeps
    # a real cancellation window open between this response and the worker
    # starting. Marking it RUNNING here would make DELETE permanently 409.
    background_tasks.add_task(_run_job, job, image_bytes)
    return _job_schema(job)


@router.get("/sessions/{session_id}/model-b/{job_id}/fusion", response_model=ModelBFusionReport)
def get_model_b_fusion(session_id: str, job_id: str) -> ModelBFusionReport:
    """Reconcile a completed Model B result against Model A's geometry.

    Read-only and side-effect free. Returns 409 until the job completes, since
    there is nothing to reconcile against a result that does not exist yet.
    """
    job = model_b_job_store.get(job_id)
    if job is None or job.session_id != session_id:
        raise HTTPException(status_code=404, detail="Model B job not found.")
    if not job.status.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"The Model B job is still {job.status.value}.",
        )
    if job.status is not JobStatus.COMPLETED or not isinstance(job.result, ModelBResult):
        raise HTTPException(
            status_code=409, detail="This Model B job did not produce a result to reconcile."
        )

    from app.model_b.fusion import reconcile

    model_a_geometry = _model_a_semantic_geometry(session_id)
    if model_a_geometry is None:
        raise HTTPException(
            status_code=409,
            detail="Process this session with Model A before requesting a comparison.",
        )
    report = reconcile(model_a_geometry, job.result)
    # PROJECT.md section 13: record the fusion result. Counts only, so nothing
    # identifying reaches the log.
    logger.info(
        "Model B fusion for job %s: %d agreements, %d candidate additions, "
        "%d disagreements (%d type mismatches, %d model A only), %d weak overlaps",
        job_id,
        len(report.agreements),
        len(report.candidate_additions),
        len(report.disagreements),
        sum(1 for d in report.disagreements if d.kind == "type_mismatch"),
        sum(1 for d in report.disagreements if d.kind == "model_a_only"),
        sum(1 for a in report.agreements if a.verdict == "weak_overlap"),
    )
    return _fusion_schema(report)


def _model_a_semantic_geometry(session_id: str) -> SemanticGeometry | None:
    """Rebuild Model A's geometry for comparison.

    Read from the session store rather than kept in memory so a fusion request
    after a server restart still reconciles against the geometry that was
    actually stored, not a stale copy.
    """
    session = session_store.get(session_id)
    if session is None or not session.semantic_geometry:
        return None
    try:
        return semantic_from_dict(session.semantic_geometry)
    except (KeyError, TypeError, ValueError):
        return None


def _fusion_schema(report: FusionReport) -> ModelBFusionReport:
    return ModelBFusionReport(
        summary=report.summary(),
        agreements=[
            ModelBAgreementHint(
                model_b_id=hint.model_b_id,
                model_a_id=hint.model_a_id,
                model_b_kind=hint.model_b_kind,
                model_a_type=hint.model_a_type,
                iou=round(hint.iou, 4),
                verdict=hint.verdict,
            )
            for hint in report.agreements
        ],
        candidate_additions=[
            ModelBCandidateAddition(
                model_b_id=addition.model_b_id,
                suggested_type=addition.suggested_type,
                region=addition.region,
                reason=addition.reason,
                requires_teacher_approval=addition.requires_teacher_approval,
            )
            for addition in report.candidate_additions
        ],
        disagreements=[
            ModelBDisagreement(
                kind=item.kind,
                model_b_id=item.model_b_id,
                model_a_id=item.model_a_id,
                detail=item.detail,
            )
            for item in report.disagreements
        ],
        diagram_relations=report.diagram_relations,
        text_notes=report.text_notes,
        uncertainties=report.uncertainties,
    )


@router.get("/sessions/{session_id}/model-b/{job_id}", response_model=ModelBJobStatus)
def get_model_b_job(session_id: str, job_id: str) -> ModelBJobStatus:
    job = model_b_job_store.get(job_id)
    if job is None or job.session_id != session_id:
        raise HTTPException(status_code=404, detail="Model B job not found.")
    return _job_schema(job)


@router.delete("/sessions/{session_id}/model-b/{job_id}", response_model=ModelBJobStatus)
def cancel_model_b_job(session_id: str, job_id: str) -> ModelBJobStatus:
    """Cancel a job that has not begun.

    A running call cannot be interrupted without provider-side support, so
    cancelling one returns 409 and says so, rather than reporting a cancellation
    that did not stop any work.
    """
    job = model_b_job_store.get(job_id)
    if job is None or job.session_id != session_id:
        raise HTTPException(status_code=404, detail="Model B job not found.")
    if job.status.is_terminal:
        raise HTTPException(status_code=409, detail=f"This job already finished ({job.status.value}).")

    cancelled = model_b_job_store.cancel(job_id)
    if cancelled is None:
        raise HTTPException(
            status_code=409,
            detail="This analysis has already started and cannot be cancelled.",
        )
    return _job_schema(cancelled)
