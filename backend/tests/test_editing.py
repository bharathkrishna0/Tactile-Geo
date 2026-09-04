from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.sessions import _semantic_to_dict, _simplified_to_dict, get_braille_translator
from app.main import app
from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType, SemanticGeometry
from app.models.session import ConversionSession
from app.services import editing
from app.services.braille import LouisBrailleTranslator
from app.services.session_store import session_store

client = TestClient(app)


class FakeLouis:
    def translateString(self, tables, text):
        return {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉"}.get(text, text)


def _fake_translate(text):
    return f"[{text}]"


def make_semantic() -> SemanticGeometry:
    line = DetectedElement(
        id="el_0", type=GeometryType.LINE_SEGMENT,
        geometry={"start": [10, 10], "end": [80, 10]},
        confidence=0.9, confidence_level=ConfidenceLevel.HIGH,
        needs_review=False, source="hough",
    )
    label = DetectedElement(
        id="el_1", type=GeometryType.TEXT_LABEL,
        geometry={"text": "A", "braille": "⠠⠁", "position": [40, 30]},
        confidence=0.9, confidence_level=ConfidenceLevel.HIGH,
        needs_review=False, source="ocr",
    )
    point = DetectedElement(
        id="el_2", type=GeometryType.POINT,
        geometry={"position": [80, 10]},
        confidence=0.85, confidence_level=ConfidenceLevel.HIGH,
        needs_review=False, source="heuristic",
    )
    return SemanticGeometry(elements=[line, label, point], image_width=100, image_height=100)


def _register_session(session_id: str = "edit_session") -> ConversionSession:
    session = ConversionSession(
        session_id=session_id,
        original_filename="diagram.png",
        original_image_path=Path("/tmp/diagram.png"),
    )
    session.semantic_geometry = _semantic_to_dict(make_semantic())
    session_store.add(session)
    return session


# ---------------------------------------------------------------- unit tests

def test_apply_set_type_marks_override():
    semantic = make_semantic()
    editing.apply_edit(semantic, "el_0", {"action": "set_type", "type": "ray"})
    element = semantic.elements[0]
    assert element.type is GeometryType.RAY
    assert element.confidence == 1.0
    assert element.confidence_level is ConfidenceLevel.HIGH
    assert element.source == "teacher_override"


def test_apply_set_type_rejects_label_element():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_1", {"action": "set_type", "type": "point"})


def test_apply_set_type_unknown_type_raises():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_0", {"action": "set_type", "type": "blob"})


def test_apply_set_label_translates_braille():
    semantic = make_semantic()
    editing.apply_edit(semantic, "el_1", {"action": "set_label", "text": "B"}, translator=_fake_translate)
    element = semantic.elements[1]
    assert element.geometry["text"] == "B"
    assert element.geometry["braille"] == "[B]"


def test_apply_delete_clears_references():
    semantic = make_semantic()
    semantic.elements[0].associated_label_id = "el_1"
    semantic.elements[1].associated_label_id = "el_0"
    editing.apply_edit(semantic, "el_1", {"action": "delete"})
    assert all(e.id != "el_1" for e in semantic.elements)
    assert semantic.elements[0].associated_label_id is None


def test_apply_set_confidence_marks_uncertain():
    semantic = make_semantic()
    editing.apply_edit(semantic, "el_0", {"action": "set_confidence", "confidence": 0.2})
    element = semantic.elements[0]
    assert element.needs_review is True
    assert element.confidence_level is ConfidenceLevel.LOW


def test_apply_set_confidence_out_of_range_raises():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_0", {"action": "set_confidence", "confidence": 5.0})


def test_apply_set_association_rebinds_label():
    semantic = make_semantic()
    # el_1 (label) had no association; bind it to el_0 via the label's target.
    editing.apply_edit(semantic, "el_1", {"action": "set_association", "association_target_id": "el_0"})
    label = semantic.elements[1]
    target = semantic.elements[0]
    assert label.associated_label_id == "el_0"
    assert target.associated_label_id == "el_1"


def test_apply_set_association_geometry_side():
    semantic = make_semantic()
    editing.apply_edit(semantic, "el_0", {"action": "set_association", "association_label_id": "el_1"})
    assert semantic.elements[0].associated_label_id == "el_1"
    assert semantic.elements[1].associated_label_id == "el_0"


def test_apply_set_association_missing_target_raises():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_1", {"action": "set_association", "association_target_id": "el_99"})


def test_apply_nudge_moves_coordinates():
    semantic = make_semantic()
    editing.apply_edit(semantic, "el_0", {"action": "nudge", "offset": [5, -2]})
    element = semantic.elements[0]
    assert element.geometry["start"] == [15, 8]
    assert element.geometry["end"] == [85, 8]


def test_apply_nudge_out_of_bounds_raises():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_0", {"action": "nudge", "offset": [-100, 0]})


def test_apply_unknown_element_raises_not_found():
    semantic = make_semantic()
    with pytest.raises(editing.ElementNotFoundError):
        editing.apply_edit(semantic, "el_99", {"action": "delete"})


def test_apply_unsupported_action_raises():
    semantic = make_semantic()
    with pytest.raises(editing.InvalidEditError):
        editing.apply_edit(semantic, "el_0", {"action": "teleport"})


def test_batch_applies_in_order():
    semantic = make_semantic()
    edits = [
        {"element_id": "el_2", "action": "nudge", "offset": [0, 5]},
        {"element_id": "el_2", "action": "nudge", "offset": [0, 5]},
    ]
    editing.apply_batch_edits(semantic, edits)
    assert semantic.elements[2].geometry["position"] == [80, 20]


def test_regenerate_recomputes_simplified_qa_svg():
    semantic = make_semantic()
    simplified, qa, svg = editing.regenerate(semantic)
    assert simplified.elements
    assert 0 <= qa.score_0_100 <= 100
    assert svg.startswith("<svg")
    assert 'data-element-id="el_0"' in svg


def test_refresh_recomputes_relationships():
    semantic = make_semantic()
    semantic.relationships = []
    editing.refresh_semantic_fields(semantic)
    # el_2 is a point at the shared endpoint of the line -> endpoint_of rel.
    assert any(rel.type.value == "endpoint_of" for rel in semantic.relationships)


# ------------------------------------------------------------------ API tests

def test_patch_set_type_returns_updated_session():
    _register_session()
    response = client.patch(
        "/api/sessions/edit_session/elements/el_0",
        json={"action": "set_type", "type": "line_segment"},
    )
    assert response.status_code == 200
    payload = response.json()
    semantic = payload["semantic_geometry"]
    el = next(e for e in semantic["elements"] if e["id"] == "el_0")
    assert el["type"] == "line_segment"
    assert payload["qa_report"]["score_0_100"] is not None
    assert payload["tactile_svg"].startswith("<svg")


def test_patch_unknown_session_404():
    response = client.patch(
        "/api/sessions/nope/elements/el_0",
        json={"action": "set_type", "type": "point"},
    )
    assert response.status_code == 404


def test_patch_unknown_element_404():
    _register_session()
    response = client.patch(
        "/api/sessions/edit_session/elements/el_99",
        json={"action": "delete"},
    )
    assert response.status_code == 404


def test_patch_invalid_type_422():
    _register_session()
    response = client.patch(
        "/api/sessions/edit_session/elements/el_0",
        json={"action": "set_type", "type": "blob"},
    )
    assert response.status_code == 422


def test_patch_nudge_out_of_bounds_422():
    _register_session()
    response = client.patch(
        "/api/sessions/edit_session/elements/el_0",
        json={"action": "nudge", "offset": [-1000, 0]},
    )
    assert response.status_code == 422


def test_patch_delete_reduces_element_count():
    _register_session()
    response = client.patch(
        "/api/sessions/edit_session/elements/el_2",
        json={"action": "delete"},
    )
    assert response.status_code == 200
    count = response.json()["semantic_geometry"]["element_count"]
    assert count == 2


def test_batch_edits_apply_atomically():
    _register_session()
    original = dict(app.dependency_overrides)
    app.dependency_overrides[get_braille_translator] = lambda: LouisBrailleTranslator(bindings=FakeLouis())
    try:
        response = client.post(
            "/api/sessions/edit_session/elements/batch",
            json={"edits": [
                {"element_id": "el_2", "action": "nudge", "offset": [0, 10]},
                {"element_id": "el_1", "action": "set_label", "text": "C"},
            ]},
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)
    assert response.status_code == 200
    payload = response.json()
    label = next(e for e in payload["semantic_geometry"]["elements"] if e["id"] == "el_1")
    point = next(e for e in payload["semantic_geometry"]["elements"] if e["id"] == "el_2")
    assert label["type"] == "text_label"
    assert point["geometry"]["position"] == [80, 20]


def test_edit_requires_processing_first():
    Session = ConversionSession(
        session_id="unprocessed", original_filename="x.png", original_image_path=Path("/tmp/x.png")
    )
    session_store.add(Session)
    response = client.patch(
        "/api/sessions/unprocessed/elements/el_0",
        json={"action": "delete"},
    )
    assert response.status_code == 409


def test_patch_set_label_uses_translator_dependency():
    _register_session()
    original = dict(app.dependency_overrides)
    app.dependency_overrides[get_braille_translator] = lambda: LouisBrailleTranslator(bindings=FakeLouis())
    try:
        response = client.patch(
            "/api/sessions/edit_session/elements/el_1",
            json={"action": "set_label", "text": "B"},
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)
    assert response.status_code == 200
    label = next(e for e in response.json()["semantic_geometry"]["elements"] if e["id"] == "el_1")
    assert label["geometry"]["braille"] == "⠠⠃"


@pytest.fixture(autouse=True)
def _clean_store():
    yield
    session_store._sessions = {}