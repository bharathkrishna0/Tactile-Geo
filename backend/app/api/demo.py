import logging
import time

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.sessions import (
    _qa_to_dict,
    _quality_report_to_dict,
    _semantic_to_dict,
    _session_response,
    _simplified_to_dict,
)
from app.models.session import ConversionSession
from app.schemas.session import EnhancedProcessedSession, ProcessingParams
from app.services.demo import get_sample, list_samples
from app.services.pipeline import build_full_analysis
from app.services.session_store import session_store


router = APIRouter(prefix="/api/demo", tags=["demo"])

logger = logging.getLogger(__name__)


class DemoSampleOut(BaseModel):
    id: str
    filename: str
    alt: str
    image_data_url: str


class DemoProcessedOut(EnhancedProcessedSession):
    demo_image_data_url: str


@router.get("/samples", response_model=list[DemoSampleOut])
def demo_samples() -> list[DemoSampleOut]:
    return [
        DemoSampleOut(
            id=sample.id,
            filename=sample.filename,
            alt=sample.alt,
            image_data_url=sample.image_data_url,
        )
        for sample in list_samples()
    ]


@router.post("/{sample_id}/process", response_model=DemoProcessedOut, status_code=status.HTTP_200_OK)
def process_demo_sample(sample_id: str) -> DemoProcessedOut:
    sample = get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Unknown demo sample.")
    # PROJECT.md section 13 observability: Model A execution time, same as the
    # real upload path. Timings and counts only.
    started = time.perf_counter()
    try:
        result = build_full_analysis(sample.path.read_bytes(), 50)
    except ValueError as error:
        logger.warning("Model A demo process failed for %s after %.0fms: %s", sample_id, (time.perf_counter() - started) * 1000, error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    logger.info(
        "Model A processed demo %s in %.0fms: %d shapes, %d labels, %d QA issues",
        sample_id,
        (time.perf_counter() - started) * 1000,
        len(result.shapes),
        len(result.labels),
        len(result.qa_report.issues),
    )

    session_id = str(uuid4())
    session = ConversionSession(
        session_id=session_id,
        original_filename=sample.filename,
        original_image_path=sample.path,
    )
    session.processing_params = ProcessingParams().model_dump()
    session.detected_shapes = result.shapes
    session.detected_labels = result.labels
    session.preview_svg = result.preview_svg
    session.tactile_svg = result.tactile_svg
    session.quality_report = _quality_report_to_dict(result.quality_report)
    session.semantic_geometry = _semantic_to_dict(result.semantic_geometry)
    session.simplified_geometry = _simplified_to_dict(result.simplified_geometry)
    session.qa_report = _qa_to_dict(result.qa_report)
    session_store.add(session)
    session_store.record_event(session_id, "model_a_processed", {"qa_passes": result.qa_report.passes, "demo_sample": sample_id})

    return DemoProcessedOut(
        **_session_response(session).model_dump(),
        demo_image_data_url=sample.image_data_url,
    )
