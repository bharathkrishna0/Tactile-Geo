"""Object storage backends and their use by the session API."""
import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import object_storage as storage_module
from app.services.object_storage import (
    LocalObjectStorage,
    ObjectStorageError,
    SupabaseObjectStorage,
    source_image_key,
)


def test_local_storage_round_trip_and_delete(tmp_path):
    storage = LocalObjectStorage(tmp_path)
    key = source_image_key("abc", ".png")
    storage.put(key, b"png-bytes", "image/png")
    assert storage.get(key) == b"png-bytes"
    storage.delete([key])
    with pytest.raises(ObjectStorageError):
        storage.get(key)


def test_local_storage_rejects_keys_outside_its_root(tmp_path):
    storage = LocalObjectStorage(tmp_path / "root")
    with pytest.raises(ObjectStorageError):
        storage.put("../escape.png", b"x", "image/png")


class _Recorder:
    def __init__(self, responses=None, error=None):
        self.requests = []
        self._responses = list(responses or [])
        self._error = error

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        body = self._responses.pop(0) if self._responses else b"{}"
        return io.BytesIO(body)


def _storage():
    return SupabaseObjectStorage("https://project.supabase.co/", "service-key", "sources")


def test_supabase_upload_targets_the_private_bucket_with_server_side_auth(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(storage_module.urllib.request, "urlopen", recorder)
    _storage().put("sessions/a/source.png", b"img", "image/png")
    request = recorder.requests[0]
    assert request.get_method() == "POST"
    assert request.full_url == "https://project.supabase.co/storage/v1/object/sources/sessions/a/source.png"
    assert request.get_header("Authorization") == "Bearer service-key"
    assert request.get_header("X-upsert") == "true"
    assert request.data == b"img"


def test_supabase_signed_url_is_absolute_and_expiring(monkeypatch):
    recorder = _Recorder([json.dumps({"signedURL": "/object/sign/sources/k.png?token=t"}).encode()])
    monkeypatch.setattr(storage_module.urllib.request, "urlopen", recorder)
    url = _storage().signed_url("k.png", 300)
    assert url == "https://project.supabase.co/storage/v1/object/sign/sources/k.png?token=t"
    assert json.loads(recorder.requests[0].data) == {"expiresIn": 300}


def test_supabase_delete_sends_prefixes(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(storage_module.urllib.request, "urlopen", recorder)
    _storage().delete(["a.png", "b.png"])
    assert recorder.requests[0].get_method() == "DELETE"
    assert json.loads(recorder.requests[0].data) == {"prefixes": ["a.png", "b.png"]}


def test_supabase_errors_never_leak_the_response_body_or_key(monkeypatch):
    error = urllib.error.HTTPError("https://x", 403, "Forbidden", {}, io.BytesIO(b"Bearer service-key"))
    monkeypatch.setattr(storage_module.urllib.request, "urlopen", _Recorder(error=error))
    with pytest.raises(ObjectStorageError) as raised:
        _storage().get("k.png")
    assert "403" in str(raised.value)
    assert "service-key" not in str(raised.value)


client = TestClient(app)


def test_upload_is_stored_by_key_and_audited(fixture_directory):
    from app.services.session_store import object_storage, session_store

    with open(fixture_directory / "triangle_worksheet.png", "rb") as handle:
        response = client.post("/api/sessions", files={"image": ("t.png", handle, "image/png")})
    assert response.status_code == 201
    session = session_store.get(response.json()["session_id"])
    assert session.source_storage_key == source_image_key(session.session_id, ".png")
    assert object_storage.get(session.source_storage_key)[:4] == b"\x89PNG"
    assert [e["event_type"] for e in session_store.events(session.session_id)] == ["session_created"]


def test_processing_and_edits_are_saved_and_audited(fixture_directory):
    from app.services.session_store import session_store

    with open(fixture_directory / "labelled_triangle_worksheet.png", "rb") as handle:
        session_id = client.post("/api/sessions", files={"image": ("t.png", handle, "image/png")}).json()["session_id"]
    processed = client.post(f"/api/sessions/{session_id}/process").json()
    element_id = processed["semantic_geometry"]["elements"][0]["id"]
    edited = client.patch(f"/api/sessions/{session_id}/elements/{element_id}", json={"action": "set_confidence", "confidence": 0.95})
    assert edited.status_code == 200
    stored = session_store.get(session_id)
    assert stored.qa_report == edited.json()["qa_report"]
    assert [e["event_type"] for e in session_store.events(session_id)] == ["session_created", "model_a_processed", "teacher_edit"]


def test_missing_source_image_is_reported_as_gone(fixture_directory):
    from app.services.session_store import object_storage

    with open(fixture_directory / "triangle_worksheet.png", "rb") as handle:
        session_id = client.post("/api/sessions", files={"image": ("t.png", handle, "image/png")}).json()["session_id"]
    object_storage.delete([source_image_key(session_id, ".png")])
    assert client.post(f"/api/sessions/{session_id}/process").status_code == 410
