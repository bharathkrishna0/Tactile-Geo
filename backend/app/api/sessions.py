from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import ALLOWED_EXTENSIONS, ALLOWED_MEDIA_TYPES, MAX_UPLOAD_BYTES, UPLOAD_DIRECTORY
from app.models.geometry import DetectedElement, ElementRelationship, SemanticGeometry, TransformationExplanation
from app.models.session import ConversionSession
from app.schemas.editing import BatchEditItem, BatchElementEditRequest, ElementEditRequest
from app.schemas.session import (
    DetectedElementSchema,
    DetectedLabel,
    DetectedShape,
    ElementRelationshipSchema,
    EnhancedProcessedSession,
    ProcessingParams,
    QAIssueSchema,
    QAReportSchema,
    QualityIssueSchema,
    QualityReportSchema,
    SemanticGeometrySchema,
    SessionCreated,
    SimplifiedGeometrySchema,
    SimplificationActionSchema,
    TransformationExplanationSchema,
)
from app.services.braille import LouisBrailleTranslator
from app.services.editing import ElementNotFoundError, InvalidEditError, apply_batch_edits, apply_edit, refresh_semantic_fields, regenerate, semantic_from_dict
from app.services.image_quality import QualityReport
from app.services.pipeline import build_full_analysis
from app.services.rate_limiter import upload_rate_limiter
from app.services.session_store import session_store
from app.services.tactile_qa import QAReport
from app.services.tactile_simplification import SimplifiedGeometry

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_braille_translator() -> LouisBrailleTranslator:
    """Dependency so tests can substitute a fake Liblouis translator."""
    return LouisBrailleTranslator()


def _session_response(session: ConversionSession) -> EnhancedProcessedSession:
    """Build the full processed-session response from the stored plain dicts."""
    params = ProcessingParams(**(session.processing_params or {}))
    return EnhancedProcessedSession(
        session_id=session.session_id,
        preview_svg=session.preview_svg or "",
        detected_shapes=[DetectedShape(**shape) for shape in session.detected_shapes],
        detected_labels=[DetectedLabel(**label) for label in session.detected_labels],
        processing_params=params,
        quality_report=QualityReportSchema(**session.quality_report) if session.quality_report else None,
        semantic_geometry=SemanticGeometrySchema(**session.semantic_geometry) if session.semantic_geometry else None,
        simplified_geometry=SimplifiedGeometrySchema(**session.simplified_geometry) if session.simplified_geometry else None,
        qa_report=QAReportSchema(**session.qa_report) if session.qa_report else None,
        tactile_svg=session.tactile_svg,
    )


def _quality_issue_schema(issue) -> QualityIssueSchema:
    return QualityIssueSchema(
        check=issue.check,
        severity=issue.severity,
        message=issue.message,
        value=issue.value,
        threshold=issue.threshold,
    )


def _quality_report_schema(report: QualityReport) -> QualityReportSchema:
    return QualityReportSchema(
        passes_gate=report.passes_gate,
        issues=[_quality_issue_schema(issue) for issue in report.issues],
        image_width=report.image_width,
        image_height=report.image_height,
    )


def _explanation_schema(explanation: TransformationExplanation) -> TransformationExplanationSchema:
    return TransformationExplanationSchema(
        stage=explanation.stage,
        element_id=explanation.element_id,
        message=explanation.message,
    )


def _element_schema(element: DetectedElement) -> DetectedElementSchema:
    return DetectedElementSchema(
        id=element.id,
        type=element.type.value,
        geometry=element.geometry,
        confidence=element.confidence,
        confidence_level=element.confidence_level.value,
        needs_review=element.needs_review,
        source=element.source,
        bbox=element.bbox,
        semantic_properties=element.semantic_properties,
        associated_label_id=element.associated_label_id,
    )


def _relationship_schema(relationship: ElementRelationship) -> ElementRelationshipSchema:
    return ElementRelationshipSchema(
        id=relationship.id,
        type=relationship.type.value,
        element_ids=relationship.element_ids,
        confidence=relationship.confidence,
        confidence_level=relationship.confidence_level.value,
        needs_review=relationship.needs_review,
        explanation=relationship.explanation,
        properties=relationship.properties,
    )


def _semantic_schema(semantic: SemanticGeometry) -> SemanticGeometrySchema:
    return SemanticGeometrySchema(
        elements=[_element_schema(element) for element in semantic.elements],
        relationships=[_relationship_schema(relationship) for relationship in semantic.relationships],
        image_width=semantic.image_width,
        image_height=semantic.image_height,
        element_count=semantic.element_count,
        low_confidence_count=semantic.low_confidence_count,
        review_flags=semantic.review_flags,
        explanations=[_explanation_schema(explanation) for explanation in semantic.explanations],
    )


def _simplified_schema(simplified: SimplifiedGeometry) -> SimplifiedGeometrySchema:
    return SimplifiedGeometrySchema(
        elements=[_element_schema(element) for element in simplified.elements],
        relationships=[_relationship_schema(relationship) for relationship in simplified.relationships],
        actions=[SimplificationActionSchema(
            element_id=action.element_id,
            action=action.action,
            detail=action.detail,
        ) for action in simplified.actions],
        removed_count=simplified.removed_count,
        merged_count=simplified.merged_count,
        explanations=simplified.explanations,
    )


def _qa_schema(report: QAReport) -> QAReportSchema:
    return QAReportSchema(
        overall_score=report.overall_score,
        passes=report.passes,
        issues=[QAIssueSchema(
            check=issue.check,
            severity=issue.severity,
            message=issue.message,
            element_id=issue.element_id,
        ) for issue in report.issues],
        element_checks=report.element_checks,
        score_0_100=report.score_0_100,
    )


def _element_to_dict(element: DetectedElement) -> dict:
    return {
        "id": element.id,
        "type": element.type.value,
        "geometry": element.geometry,
        "confidence": element.confidence,
        "confidence_level": element.confidence_level.value,
        "needs_review": element.needs_review,
        "source": element.source,
        "bbox": list(element.bbox) if element.bbox else None,
        "semantic_properties": element.semantic_properties,
        "associated_label_id": element.associated_label_id,
    }


def _relationship_to_dict(relationship: ElementRelationship) -> dict:
    return {
        "id": relationship.id,
        "type": relationship.type.value,
        "element_ids": relationship.element_ids,
        "confidence": relationship.confidence,
        "confidence_level": relationship.confidence_level.value,
        "needs_review": relationship.needs_review,
        "explanation": relationship.explanation,
        "properties": relationship.properties,
    }


def _quality_report_to_dict(report: QualityReport) -> dict:
    return {
        "passes_gate": report.passes_gate,
        "issues": [
            {
                "check": issue.check,
                "severity": issue.severity,
                "message": issue.message,
                "value": issue.value,
                "threshold": issue.threshold,
            }
            for issue in report.issues
        ],
        "image_width": report.image_width,
        "image_height": report.image_height,
    }


def _semantic_to_dict(semantic: SemanticGeometry) -> dict:
    return {
        "elements": [_element_to_dict(element) for element in semantic.elements],
        "relationships": [_relationship_to_dict(relationship) for relationship in semantic.relationships],
        "image_width": semantic.image_width,
        "image_height": semantic.image_height,
        "element_count": semantic.element_count,
        "low_confidence_count": semantic.low_confidence_count,
        "review_flags": semantic.review_flags,
        "explanations": [
            {
                "stage": explanation.stage,
                "element_id": explanation.element_id,
                "message": explanation.message,
            }
            for explanation in semantic.explanations
        ],
    }


def _simplified_to_dict(simplified: SimplifiedGeometry) -> dict:
    return {
        "elements": [_element_to_dict(element) for element in simplified.elements],
        "relationships": [_relationship_to_dict(relationship) for relationship in simplified.relationships],
        "actions": [
            {
                "element_id": action.element_id,
                "action": action.action,
                "detail": action.detail,
            }
            for action in simplified.actions
        ],
        "removed_count": simplified.removed_count,
        "merged_count": simplified.merged_count,
        "explanations": simplified.explanations,
    }


def _qa_to_dict(report: QAReport) -> dict:
    return {
        "overall_score": report.overall_score,
        "passes": report.passes,
        "issues": [
            {
                "check": issue.check,
                "severity": issue.severity,
                "message": issue.message,
                "element_id": issue.element_id,
            }
            for issue in report.issues
        ],
        "element_checks": report.element_checks,
        "score_0_100": report.score_0_100,
    }


@router.post("", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
async def create_session(request: Request, image: UploadFile = File(...)) -> SessionCreated:
    client_address = request.client.host if request.client else "unknown"
    if not upload_rate_limiter.allow(client_address):
        raise HTTPException(status_code=429, detail="Too many uploads. Please wait a minute and try again.")
    extension = Path(image.filename or "").suffix.lower()
    if image.content_type not in ALLOWED_MEDIA_TYPES or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, or JPEG image.")
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 10 MB limit.")
    session_id = str(uuid4())
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIRECTORY / f"{session_id}{extension}"
    image_path.write_bytes(contents)
    session = ConversionSession(session_id=session_id, original_filename=image.filename or f"upload{extension}", original_image_path=image_path)
    session_store.add(session)
    return SessionCreated(session_id=session_id, filename=session.original_filename, created_at=session.created_at)


@router.post("/{session_id}/process", response_model=EnhancedProcessedSession)
async def process_session(session_id: str, params: ProcessingParams | None = None) -> EnhancedProcessedSession:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload the image again.")
    processing_params = params or ProcessingParams()
    try:
        result = build_full_analysis(session.original_image_path.read_bytes(), processing_params.edge_sensitivity)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    session.processing_params = processing_params.model_dump()
    session.detected_shapes = result.shapes
    session.detected_labels = result.labels
    session.preview_svg = result.preview_svg
    session.tactile_svg = result.tactile_svg
    session.quality_report = _quality_report_to_dict(result.quality_report)
    session.semantic_geometry = _semantic_to_dict(result.semantic_geometry)
    session.simplified_geometry = _simplified_to_dict(result.simplified_geometry)
    session.qa_report = _qa_to_dict(result.qa_report)
    return _session_response(session)


@router.patch("/{session_id}/elements/{element_id}", response_model=EnhancedProcessedSession)
async def edit_element(
    session_id: str,
    element_id: str,
    payload: ElementEditRequest,
    translator: LouisBrailleTranslator = Depends(get_braille_translator),
) -> EnhancedProcessedSession:
    """Apply one teacher correction and regenerate the tactile output."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload the image again.")
    semantic = _editable_semantic(session)
    try:
        edit = {**payload.model_dump(), "element_id": element_id}
        apply_edit(semantic, element_id, edit, translator=translator.translate)
    except ElementNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidEditError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _persist_edited_session(session, semantic)
    return _session_response(session)


@router.post("/{session_id}/elements/batch", response_model=EnhancedProcessedSession)
async def batch_edit(
    session_id: str,
    payload: BatchElementEditRequest,
    translator: LouisBrailleTranslator = Depends(get_braille_translator),
) -> EnhancedProcessedSession:
    """Apply several corrections atomically, then regenerate once."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload the image again.")
    semantic = _editable_semantic(session)
    edits = [_item_to_edit(item) for item in payload.edits]
    try:
        apply_batch_edits(semantic, edits, translator=translator.translate)
    except ElementNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidEditError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _persist_edited_session(session, semantic)
    return _session_response(session)


def _item_to_edit(item: BatchEditItem) -> dict:
    data = item.model_dump()
    return {"element_id": data.pop("element_id"), **data}


def _editable_semantic(session: ConversionSession) -> SemanticGeometry:
    if not session.semantic_geometry:
        raise HTTPException(status_code=409, detail="Process the image before editing its geometry.")
    return semantic_from_dict(session.semantic_geometry)


def _persist_edited_session(session: ConversionSession, semantic: SemanticGeometry) -> None:
    refresh_semantic_fields(semantic)
    simplified, qa_report, tactile_svg = regenerate(semantic)
    session.semantic_geometry = _semantic_to_dict(semantic)
    session.simplified_geometry = _simplified_to_dict(simplified)
    session.qa_report = _qa_to_dict(qa_report)
    session.tactile_svg = tactile_svg
