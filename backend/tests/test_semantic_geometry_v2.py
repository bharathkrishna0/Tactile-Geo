"""Semantic Geometry v2: teacher-accepted Model B findings reaching the tactile output."""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import model_b as model_b_api
from app.model_b.semantic_v2 import MARKER_KEY, build_semantic_geometry_v2, geometry_supports, model_a_baseline
from app.model_b.service import ModelBSettings
from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from tests.test_model_b_fusion import doc_with, entity_with_id, make_model_a, make_model_b, snapshot

TRIANGLE_BBOX = (100, 80, 600, 560)
ELLIPSE_BBOX = (550, 240, 300, 240)


def _coordinates(semantic) -> dict:
    return {element.id: (element.geometry, element.bbox) for element in semantic.elements}


def _element(semantic, element_id: str) -> DetectedElement:
    return next(element for element in semantic.elements if element.id == element_id)


def _with_label(text: str):
    model_a = make_model_a()
    model_a.elements.append(
        DetectedElement(
            id="l1",
            type=GeometryType.TEXT_LABEL,
            geometry={"text": text, "braille": "⠁", "position": [400, 700]},
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            needs_review=False,
            source="easyocr",
            bbox=(390, 690, 20, 20),
        )
    )
    return model_a


class TestBuild:
    def test_without_accepted_findings_v2_equals_model_a(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        for decisions in ({}, {"b_e1": "reject"}, {"b_e1": "defer"}):
            v2 = build_semantic_geometry_v2(model_a, result, decisions, "job")
            assert snapshot(v2.semantic) == snapshot(model_a)
            assert v2.outcomes == []

    def test_inputs_are_never_mutated(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        before_a, before_b = snapshot(model_a), copy.deepcopy(result)
        build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job")
        assert snapshot(model_a) == before_a
        assert result == before_b

    def test_accepted_type_correction_keeps_model_a_coordinates(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        v2 = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert outcome.applied and outcome.change == "set_type" and outcome.model_a_id == "m1"
        element = _element(v2.semantic, "m1")
        assert element.type is GeometryType.POLYGON
        assert element.source == "teacher_override" and element.confidence == 1.0
        assert _coordinates(v2.semantic) == _coordinates(model_a)

    def test_type_the_geometry_cannot_support_is_not_applied(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "circle", TRIANGLE_BBOX)]))
        v2 = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert not outcome.applied and "cannot be drawn as a circle" in outcome.detail
        assert snapshot(v2.semantic) == snapshot(model_a)

    def test_accepted_model_b_only_region_adds_no_geometry(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "circle", (900, 700, 60, 60))]))
        v2 = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert not outcome.applied and outcome.model_a_id is None
        assert [e.id for e in v2.semantic.elements] == [e.id for e in model_a.elements]
        assert snapshot(v2.semantic) == snapshot(model_a)

    def test_accepted_agreement_confirms_the_element(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e2", "ellipse", ELLIPSE_BBOX)]))
        v2 = build_semantic_geometry_v2(model_a, result, {"b_e2": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert outcome.applied and outcome.change == "confirm"
        element = _element(v2.semantic, "m2")
        assert element.needs_review is False and element.source == "teacher_override"
        assert element.type is GeometryType.ELLIPSE
        assert _coordinates(v2.semantic) == _coordinates(model_a)

    def test_accepted_label_attaches_the_matching_model_a_label(self) -> None:
        model_a = _with_label("ABC")
        entity = dict(entity_with_id("e1", "triangle", TRIANGLE_BBOX), label="abc")
        v2 = build_semantic_geometry_v2(model_a, make_model_b(doc_with([entity])), {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert outcome.applied and outcome.change == "attach_label"
        assert _element(v2.semantic, "m1").associated_label_id == "l1"
        assert _element(v2.semantic, "l1").associated_label_id == "m1"
        assert len(v2.semantic.elements) == len(model_a.elements)

    def test_label_without_model_a_text_creates_nothing(self) -> None:
        model_a = _with_label("XYZ")
        entity = dict(entity_with_id("e1", "triangle", TRIANGLE_BBOX), label="ABC")
        v2 = build_semantic_geometry_v2(model_a, make_model_b(doc_with([entity])), {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert outcome.change == "confirm"
        assert _element(v2.semantic, "m1").associated_label_id is None
        assert len(v2.semantic.elements) == len(model_a.elements)

    def test_a_label_never_becomes_a_shape(self) -> None:
        """Seen live: a model marked each vertex as a point right on its letter.

        Accepting those must not turn Model A's Braille labels into raised dots.
        """
        model_a = _with_label("A")
        result = make_model_b(doc_with([entity_with_id("e1", "point", (390, 690, 20, 20))]))
        v2 = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job")
        (outcome,) = v2.outcomes
        assert not outcome.applied and outcome.model_a_id == "l1"
        assert "Braille label" in outcome.detail
        assert snapshot(v2.semantic) == snapshot(model_a)

    def test_applying_twice_is_idempotent(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        once = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job").semantic
        twice = build_semantic_geometry_v2(once, result, {"b_e1": "accept"}, "job")
        assert _element(twice.semantic, "m1") == _element(once, "m1")
        assert twice.outcomes[0].applied

    def test_withdrawn_acceptance_restores_model_a(self) -> None:
        model_a = _with_label("ABC")
        entity = dict(entity_with_id("e1", "triangle", TRIANGLE_BBOX), label="ABC")
        result = make_model_b(doc_with([entity]))
        applied = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job").semantic
        withdrawn = build_semantic_geometry_v2(applied, result, {"b_e1": "reject"}, "job")
        assert withdrawn.reverted == ["b_e1"]
        for element in withdrawn.semantic.elements:
            original = _element(model_a, element.id)
            assert MARKER_KEY not in element.semantic_properties
            assert (element.type, element.confidence, element.source, element.associated_label_id) == (
                original.type, original.confidence, original.source, original.associated_label_id,
            )

    def test_a_later_hand_edit_survives_withdrawal(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        applied = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job").semantic
        _element(applied, "m1").type = GeometryType.RECTANGLE
        withdrawn = build_semantic_geometry_v2(applied, result, {}, "job").semantic
        assert _element(withdrawn, "m1").type is GeometryType.RECTANGLE

    def test_baseline_undoes_only_this_job(self) -> None:
        model_a = make_model_a()
        result = make_model_b(doc_with([entity_with_id("e1", "polygon", TRIANGLE_BBOX)]))
        applied = build_semantic_geometry_v2(model_a, result, {"b_e1": "accept"}, "job").semantic
        assert _element(model_a_baseline(applied, "job"), "m1").type is GeometryType.TRIANGLE
        assert _element(model_a_baseline(applied, "other"), "m1").type is GeometryType.POLYGON


@pytest.mark.parametrize(
    ("target", "geometry", "expected"),
    [
        (GeometryType.TRIANGLE, {"points": [[0, 0], [1, 0], [0, 1]]}, True),
        (GeometryType.TRIANGLE, {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}, False),
        (GeometryType.RECTANGLE, {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}, True),
        (GeometryType.CIRCLE, {"center": [0, 0], "radius": 3}, True),
        (GeometryType.CIRCLE, {"center": [0, 0], "semi_axes": [3, 2]}, False),
        (GeometryType.LINE_SEGMENT, {"start": [0, 0], "end": [1, 1]}, True),
        (GeometryType.ANGLE, {"vertex": [0, 0]}, False),
    ],
)
def test_geometry_supports(target, geometry, expected) -> None:
    assert geometry_supports(target, geometry) is expected


def _png(width: int = 1000, height: int = 800) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), 200, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def client(monkeypatch):
    import app.main as main_module
    from app.services.model_b_store import model_b_job_store
    from app.services.rate_limiter import upload_rate_limiter

    model_b_job_store.clear()
    upload_rate_limiter._requests.clear()
    result = make_model_b(doc_with([
        entity_with_id("e1", "polygon", TRIANGLE_BBOX),
        entity_with_id("e2", "circle", (900, 700, 60, 60)),
    ]))
    monkeypatch.setattr(model_b_api, "analyze_image", lambda image_bytes, settings, client=None: copy.deepcopy(result))
    monkeypatch.setattr(model_b_api, "model_b_settings", lambda: ModelBSettings(enabled=True, api_key="k", model="m", timeout_s=5.0))
    with TestClient(main_module.app) as test_client:
        yield test_client


def _session_with_job(client) -> tuple[str, str]:
    from app.services.session_store import session_store

    response = client.post("/api/sessions", files={"image": ("w.png", _png(), "image/png")})
    session_id = response.json()["session_id"]
    session = session_store.get(session_id)
    session.semantic_geometry = {
        "elements": [{
            "id": "m1",
            "type": "triangle",
            "geometry": {"points": [[100, 80], [700, 80], [400, 640]]},
            "confidence": 0.9,
            "confidence_level": "high",
            "needs_review": False,
            "source": "opencv_hough",
            "bbox": list(TRIANGLE_BBOX),
        }],
        "relationships": [],
        "image_width": 1000,
        "image_height": 800,
        "element_count": 1,
    }
    session_store.save(session)
    job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
    return session_id, job_id


def _decide(client, session_id: str, job_id: str, model_b_id: str, decision: str) -> None:
    response = client.post(
        f"/api/sessions/{session_id}/model-b/{job_id}/decisions",
        json={"model_b_id": model_b_id, "decision": decision},
    )
    assert response.status_code == 201


class TestApplyEndpoint:
    def test_accepted_findings_regenerate_the_tactile_output(self, client) -> None:
        from app.services.session_store import session_store

        session_id, job_id = _session_with_job(client)
        _decide(client, session_id, job_id, "b_e1", "accept")
        _decide(client, session_id, job_id, "b_e2", "accept")
        response = client.post(f"/api/sessions/{session_id}/model-b/{job_id}/apply")
        assert response.status_code == 200, response.text
        body = response.json()
        outcomes = {outcome["model_b_id"]: outcome for outcome in body["outcomes"]}
        assert outcomes["b_e1"]["applied"] and outcomes["b_e1"]["change"] == "set_type"
        assert not outcomes["b_e2"]["applied"]
        elements = body["session"]["semantic_geometry"]["elements"]
        assert [(e["id"], e["type"]) for e in elements] == [("m1", "polygon")]
        assert elements[0]["geometry"] == {"points": [[100, 80], [700, 80], [400, 640]]}
        assert body["session"]["qa_report"] is not None and "<polygon" in body["session"]["tactile_svg"]
        stored = session_store.get(session_id)
        assert stored.semantic_geometry["elements"][0]["type"] == "polygon"
        events = [e for e in session_store.events(session_id) if e["event_type"] == "model_b_applied"]
        assert events[-1]["payload"]["applied"] == ["b_e1"]

    def test_fusion_reviews_survive_applying(self, client) -> None:
        session_id, job_id = _session_with_job(client)
        _decide(client, session_id, job_id, "b_e1", "accept")
        client.post(f"/api/sessions/{session_id}/model-b/{job_id}/apply")
        fusion = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion").json()
        review = next(r for r in fusion["entity_reviews"] if r["model_b_id"] == "b_e1")
        assert review["contradicts_model_a"] and review["requires_review"]

    def test_rejecting_and_reapplying_restores_model_a(self, client) -> None:
        session_id, job_id = _session_with_job(client)
        _decide(client, session_id, job_id, "b_e1", "accept")
        client.post(f"/api/sessions/{session_id}/model-b/{job_id}/apply")
        _decide(client, session_id, job_id, "b_e1", "reject")
        body = client.post(f"/api/sessions/{session_id}/model-b/{job_id}/apply").json()
        assert body["reverted"] == ["b_e1"]
        element = body["session"]["semantic_geometry"]["elements"][0]
        assert element["type"] == "triangle" and element["source"] == "opencv_hough"

    def test_unprocessed_session_cannot_apply(self, client) -> None:
        response = client.post("/api/sessions", files={"image": ("w.png", _png(), "image/png")})
        session_id = response.json()["session_id"]
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        assert client.post(f"/api/sessions/{session_id}/model-b/{job_id}/apply").status_code == 409

    def test_unknown_job_is_404(self, client) -> None:
        session_id, _ = _session_with_job(client)
        assert client.post(f"/api/sessions/{session_id}/model-b/nope/apply").status_code == 404
