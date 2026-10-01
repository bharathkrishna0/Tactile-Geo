"""Tests for Milestone 7: Competition Demo Mode."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_demo_samples_lists_available_samples():
    response = client.get("/api/demo/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    ids = {s["id"] for s in samples}
    assert "triangle" in ids
    assert "circle" in ids
    first = samples[0]
    assert first["image_data_url"].startswith("data:image/")


def test_demo_sample_process_runs_real_pipeline():
    response = client.post("/api/demo/triangle/process")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert data["demo_image_data_url"].startswith("data:image/")
    assert data["semantic_geometry"] is not None
    assert data["qa_report"] is not None
    assert data["tactile_svg"]
    assert data["semantic_geometry"]["element_count"] > 0


def test_demo_circle_process():
    response = client.post("/api/demo/circle/process")
    assert response.status_code == 200
    data = response.json()
    assert data["semantic_geometry"]["element_count"] > 0
    assert data["tactile_svg"]


def test_demo_unknown_sample_404():
    response = client.post("/api/demo/does-not-exist/process")
    assert response.status_code == 404


def test_demo_result_survives_edit_regeneration():
    response = client.post("/api/demo/labelled_triangle/process").json()
    session_id = response["session_id"]
    elements = response["semantic_geometry"]["elements"]
    target = next((e for e in elements if e["type"] == "text_label"), elements[0])
    edit_payload = {"action": "set_label", "text": "Z"}
    patch = client.patch(f"/api/sessions/{session_id}/elements/{target['id']}", json=edit_payload)
    assert patch.status_code == 200
    updated = patch.json()
    assert updated["tactile_svg"]
    assert updated["qa_report"] is not None
