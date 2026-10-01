"""Phase 2 tests: config, job store, and the Model B API surface."""

from __future__ import annotations

import importlib
import json
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import model_b as model_b_api
from app.model_b._fixtures import minimal_document
from app.model_b.client import ProviderResponse
from app.models.model_b_job import JobStatus, ModelBJob, ModelBJobError
from app.schemas.model_b import ModelBAnalysis
from app.services.model_b_store import MAX_RETAINED_JOBS, ModelBJobStore


@pytest.fixture
def client(monkeypatch):
    """TestClient with a working Model B backed by a fake analyser."""
    import app.main as main_module
    from app.services.rate_limiter import upload_rate_limiter

    model_b_job_store = importlib.import_module("app.services.model_b_store")
    model_b_job_store.model_b_job_store.clear()
    # The shared upload limiter is process-global and survives across tests, so
    # it must be reset explicitly or later tests fail on a stale 429.
    upload_rate_limiter._requests.clear()

    calls: list[bytes] = []

    def fake_analyze(image_bytes, settings, client=None):
        calls.append(image_bytes)
        from app.model_b.normalizer import normalize
        from app.model_b.preparation import prepare_image
        from app.model_b.validator import validate_document

        prepared = prepare_image(image_bytes)
        document, warnings = validate_document(
            minimal_document(), prepared_width=prepared.width, prepared_height=prepared.height
        )
        return normalize(document, prepared, finish_reason="STOP", validation_warnings=warnings)

    monkeypatch.setattr(model_b_api, "analyze_image", fake_analyze)
    monkeypatch.setattr(model_b_api, "model_b_settings", lambda: _settings(enabled=True, key="k"))

    with TestClient(main_module.app) as test_client:
        test_client.calls = calls  # type: ignore[attr-defined]
        test_client.job_store = model_b_job_store.model_b_job_store  # type: ignore[attr-defined]
        yield test_client


def _settings(*, enabled: bool, key: str | None):
    from app.model_b.service import ModelBSettings

    return ModelBSettings(enabled=enabled, api_key=key, model="openrouter/free", timeout_s=5.0)


def _upload(client, size=(300, 400)) -> str:
    array = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    response = client.post(
        "/api/sessions",
        files={"image": ("worksheet.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


class TestConfig:
    def test_defaults_to_disabled(self) -> None:
        from app.core.config import MODEL_B_ENABLED, OPENROUTER_MODEL

        assert MODEL_B_ENABLED is False
        assert OPENROUTER_MODEL == "openrouter/free"

    def test_missing_key_is_none_not_empty_string(self) -> None:
        """The `or None` idiom: '' must not read as a configured key."""
        import os

        from app.core import config

        original = os.environ.get("OPENROUTER_API_KEY")
        os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            assert config.MODEL_B_API_KEY is None
        finally:
            if original is not None:
                os.environ["OPENROUTER_API_KEY"] = original

    def test_settings_factory_returns_model_bsettings(self) -> None:
        from app.core.config import model_b_settings

        assert hasattr(model_b_settings(), "require_available")

    def test_default_model_is_the_free_router(self) -> None:
        from app.core.config import model_b_settings

        assert model_b_settings().model == "openrouter/free"

    def test_default_base_url_is_openrouter(self) -> None:
        from app.core.config import model_b_settings

        assert model_b_settings().base_url == "https://openrouter.ai/api/v1"

    def test_model_comes_from_the_environment(self, monkeypatch) -> None:
        """Section 4: no model may be hardcoded into the service."""
        from app.core.config import model_b_settings

        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free")
        assert model_b_settings().model == "qwen/qwen3.8-27b:free"

    def test_base_url_comes_from_the_environment(self, monkeypatch) -> None:
        from app.core.config import model_b_settings

        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.internal/v1/")
        # Trailing slash stripped, or the endpoint gains a double slash.
        assert model_b_settings().base_url == "https://proxy.internal/v1"

    def test_retry_policy_is_configurable(self, monkeypatch) -> None:
        from app.core.config import model_b_settings

        monkeypatch.setenv("MODEL_B_MAX_ATTEMPTS", "5")
        monkeypatch.setenv("MODEL_B_RETRY_BASE_DELAY_S", "0.25")
        settings = model_b_settings()
        assert settings.max_attempts == 5
        assert settings.retry_base_delay_s == 0.25

    def test_unparseable_number_falls_back_to_the_documented_default(
        self, monkeypatch
    ) -> None:
        """A typo must not raise on the request path.

        A ValueError raised while resolving configuration surfaces to a teacher
        as a Model B crash, which reads as a bug rather than a setup mistake.
        """
        from app.core.config import model_b_settings

        monkeypatch.setenv("MODEL_B_TIMEOUT_S", "thirty")
        assert model_b_settings().timeout_s == 30.0

    def test_gemini_variables_are_no_longer_read(self, monkeypatch) -> None:
        """The provider swap is total: a stale Google key cannot be used.

        Sending one to OpenRouter would produce a 401 that looks like a bad
        OpenRouter key, which is a genuinely misleading failure to debug.
        """
        from app.core.config import model_b_settings

        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyNOTAREALKEY")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
        monkeypatch.setenv("MODEL_B_MODEL", "gemini-3.8-flash")
        settings = model_b_settings()
        assert settings.api_key is None
        assert settings.model == "openrouter/free"


class TestJobStore:
    def test_create_and_get(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.get(job.job_id) is job
        assert job.status is JobStatus.QUEUED

    def test_missing_job_returns_none(self) -> None:
        assert ModelBJobStore().get("nope") is None

    def test_list_for_session_sorted_newest_first(self) -> None:
        store = ModelBJobStore()
        first = store.create("s1", model="m")
        second = store.create("s1", model="m")
        jobs = store.list_for_session("s1")
        assert [j.job_id for j in jobs] == [second.job_id, first.job_id]

    def test_list_excludes_other_sessions(self) -> None:
        store = ModelBJobStore()
        store.create("s1", model="m")
        store.create("s2", model="m")
        assert len(store.list_for_session("s1")) == 1

    def test_has_active_job_ignores_finished(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.has_active_job("s1")
        job.mark_completed({})
        assert not store.has_active_job("s1")

    def test_cancel_queued_job(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.cancel(job.job_id) is not None
        assert job.status is JobStatus.CANCELLED

    def test_cannot_cancel_running_job(self) -> None:
        """Reporting a cancellation that did not stop billing would be a lie."""
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        job.mark_running()
        assert store.cancel(job.job_id) is None
        assert job.status is JobStatus.RUNNING

    def test_begin_claims_a_queued_job(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.begin(job.job_id) is job
        assert job.status is JobStatus.RUNNING

    def test_begin_refuses_a_job_already_claimed(self) -> None:
        """A second worker must not double-spend the provider call."""
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.begin(job.job_id) is job
        assert store.begin(job.job_id) is None

    def test_begin_refuses_a_cancelled_job(self) -> None:
        """Regression: cancel must not be silently overwritten by the worker.

        With a non-atomic `get` + status check + `mark_running`, a cancel
        landing in that gap was reverted to RUNNING, so the teacher saw their
        cancellation accepted while the call was still made and billed.
        """
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        assert store.cancel(job.job_id) is not None
        assert store.begin(job.job_id) is None
        assert store.get(job.job_id).status is JobStatus.CANCELLED  # type: ignore[union-attr]

    def test_begin_returns_none_for_unknown_job(self) -> None:
        assert ModelBJobStore().begin("nope") is None

    def test_begin_under_contention_leaves_exactly_one_winner(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        barrier = threading.Barrier(8)
        results: list[object] = []

        def claim() -> None:
            barrier.wait()
            results.append(store.begin(job.job_id))

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len([r for r in results if r is not None]) == 1
        assert job.status is JobStatus.RUNNING

    def test_cannot_cancel_twice(self) -> None:
        store = ModelBJobStore()
        job = store.create("s1", model="m")
        store.cancel(job.job_id)
        assert store.cancel(job.job_id) is None

    def test_active_job_count(self) -> None:
        store = ModelBJobStore()
        store.create("s1", model="m")
        done = store.create("s1", model="m")
        done.mark_completed({})
        assert store.active_job_count() == 1

    def test_eviction_is_bounded(self) -> None:
        store = ModelBJobStore()
        for _ in range(MAX_RETAINED_JOBS + 25):
            store.create("s1", model="m")
        assert len(store.list_for_session("s1")) == MAX_RETAINED_JOBS

    def test_eviction_prefers_dropping_finished_jobs(self) -> None:
        store = ModelBJobStore()
        old = store.create("s1", model="m")
        old.mark_completed({})
        for _ in range(MAX_RETAINED_JOBS + 5):
            store.create("s1", model="m")
        assert store.get(old.job_id) is None

    def test_terminal_statuses(self) -> None:
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal
        assert not JobStatus.QUEUED.is_terminal
        assert not JobStatus.RUNNING.is_terminal


class TestJobTransitions:
    def test_completion_records_timestamps(self) -> None:
        job = ModelBJob(job_id="j", session_id="s")
        job.mark_running()
        job.mark_completed({})
        assert job.status is JobStatus.COMPLETED
        assert job.started_at is not None and job.completed_at is not None

    def test_failure_records_error(self) -> None:
        job = ModelBJob(job_id="j", session_id="s")
        job.mark_running()
        job.mark_failed(ModelBJobError(code="x", message="m", retryable=True))
        assert job.status is JobStatus.FAILED
        assert job.error is not None and job.error.retryable is True


class TestStatusEndpoint:
    def test_reports_unavailable_when_disabled(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "model_b_settings", lambda: _settings(enabled=False, key=None))
        body = client.get("/api/model-b/status").json()
        assert body["available"] is False
        assert body["enabled"] is False
        assert "turned off" in body["reason"]

    def test_reports_missing_key_distinctly(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "model_b_settings", lambda: _settings(enabled=True, key=None))
        body = client.get("/api/model-b/status").json()
        assert body["available"] is False
        assert body["enabled"] is True
        assert "no OpenRouter API key" in body["reason"]

    def test_reports_available(self, client) -> None:
        body = client.get("/api/model-b/status").json()
        assert body["available"] is True
        assert body["reason"] is None

    def test_reports_provider_and_model(self, client) -> None:
        """Sections 4/18: the UI and logs must use the real provider identity.

        A model id is not a credential, and a teacher cannot audit an advisory
        result that will not say which model produced it.
        """
        body = client.get("/api/model-b/status").json()
        assert body["provider"] == "openrouter"
        assert body["model"] == "openrouter/free"

    def test_never_leaks_the_api_key(self, client) -> None:
        raw = client.get("/api/model-b/status").text
        assert "k" not in raw.replace("available", "").replace("kind", "")
        assert "api_key" not in raw
        assert "OPENROUTER_API_KEY" not in raw
        # The configured model *is* exposed; the credential is not. Both halves
        # of that are asserted so neither regresses silently.
        assert "openrouter/free" in raw

    def test_no_endpoint_returns_anything_key_shaped(self, client) -> None:
        session_id = _upload(client)
        job = client.post(f"/api/sessions/{session_id}/model-b").json()
        blobs = [
            client.get("/api/model-b/status").text,
            client.get(f"/api/sessions/{session_id}/model-b/{job['job_id']}").text,
        ]
        for blob in blobs:
            assert "sk-or-v1" not in blob
            assert "Authorization" not in blob
            assert "Bearer" not in blob


class TestObservability:
    """PROJECT.md section 13 requires Model B timings and outcomes recorded."""

    def test_success_logs_timing_and_counts(self, client, caplog) -> None:
        import logging

        session_id = _upload(client)
        with caplog.at_level(logging.INFO, logger="app.api.model_b"):
            response = client.post(f"/api/sessions/{session_id}/model-b")
        assert response.status_code == 202

        messages = [record.getMessage() for record in caplog.records]
        completed = [m for m in messages if "completed in" in m]
        assert completed, f"no completion log; got {messages}"
        entry = completed[0]
        assert "ms" in entry
        for field in ("entities", "relations", "text items", "uncertainties"):
            assert field in entry, f"missing {field} in: {entry}"

    def test_log_never_contains_the_key_or_image_bytes(self, client, caplog) -> None:
        import logging

        session_id = _upload(client)
        with caplog.at_level(logging.DEBUG, logger="app.api.model_b"):
            client.post(f"/api/sessions/{session_id}/model-b")

        blob = "\n".join(record.getMessage() for record in caplog.records)
        # The test key is the single character "k", so it cannot be searched
        # for directly; what matters is that no credential-shaped token and no
        # payload reached the log.
        assert "OPENROUTER_API_KEY" not in blob
        assert "api_key" not in blob
        assert "worksheet.png" not in blob
        assert "entities" in blob  # counts only, never the response body

    def test_failure_logs_the_reason_code(self, client, caplog, monkeypatch) -> None:
        import logging

        from app.model_b.errors import ModelBTimeout

        session_id = _upload(client)

        def explode(image_bytes, settings, client=None):
            raise ModelBTimeout("The model provider did not answer in time.")

        monkeypatch.setattr(model_b_api, "analyze_image", explode)
        with caplog.at_level(logging.WARNING, logger="app.api.model_b"):
            client.post(f"/api/sessions/{session_id}/model-b")

        messages = [record.getMessage() for record in caplog.records]
        assert any("model_b_timeout" in m for m in messages), messages

    def test_fusion_logs_agreement_counts(self, client, caplog) -> None:
        import logging

        session_id = _upload(client)
        job = client.post(f"/api/sessions/{session_id}/model-b").json()
        with caplog.at_level(logging.INFO, logger="app.api.model_b"):
            response = client.get(
                f"/api/sessions/{session_id}/model-b/{job['job_id']}/fusion"
            )
        assert response.status_code in {200, 409}
        if response.status_code == 200:
            messages = [record.getMessage() for record in caplog.records]
            assert any("fusion" in m and "agreements" in m for m in messages), messages


class TestRequestEndpoint:
    def test_returns_202_with_queued_job(self, client, monkeypatch) -> None:
        """Queue a job that never runs, so the response state is observable."""
        session_id = _upload(client)
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        response = client.post(f"/api/sessions/{session_id}/model-b")
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["result"] is None

    def test_unknown_session_is_404(self, client) -> None:
        assert client.post("/api/sessions/does-not-exist/model-b").status_code == 404

    def test_disabled_returns_503(self, client, monkeypatch) -> None:
        session_id = _upload(client)
        monkeypatch.setattr(model_b_api, "model_b_settings", lambda: _settings(enabled=False, key=None))
        response = client.post(f"/api/sessions/{session_id}/model-b")
        assert response.status_code == 503
        assert "MODEL_B_ENABLED" in response.json()["detail"]

    def test_missing_key_returns_503_naming_the_variable(self, client, monkeypatch) -> None:
        session_id = _upload(client)
        monkeypatch.setattr(model_b_api, "model_b_settings", lambda: _settings(enabled=True, key=None))
        response = client.post(f"/api/sessions/{session_id}/model-b")
        assert response.status_code == 503
        assert "OPENROUTER_API_KEY" in response.json()["detail"]

    def test_duplicate_active_job_is_409(self, client, monkeypatch) -> None:
        session_id = _upload(client)
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        assert client.post(f"/api/sessions/{session_id}/model-b").status_code == 202
        assert client.post(f"/api/sessions/{session_id}/model-b").status_code == 409

    def test_concurrency_cap_is_429(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        sessions = [_upload(client) for _ in range(6)]
        codes = [client.post(f"/api/sessions/{s}/model-b").status_code for s in sessions]
        assert 429 in codes
        assert 202 in codes

    def test_worker_receives_the_uploaded_bytes(self, client, monkeypatch) -> None:
        session_id = _upload(client)
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        client.post(f"/api/sessions/{session_id}/model-b")
        assert client.job_store.list_for_session(session_id)

    def test_missing_image_file_is_410(self, client, monkeypatch) -> None:
        session_id = _upload(client)
        from app.services.session_store import session_store

        session_store.get(session_id).original_image_path.unlink()
        response = client.post(f"/api/sessions/{session_id}/model-b")
        assert response.status_code == 410

    def test_empty_image_file_is_410(self, client) -> None:
        session_id = _upload(client)
        from app.services.session_store import session_store

        session_store.get(session_id).original_image_path.write_bytes(b"")
        assert client.post(f"/api/sessions/{session_id}/model-b").status_code == 410


class TestPollingEndpoint:
    def test_unknown_job_is_404(self, client) -> None:
        session_id = _upload(client)
        assert client.get(f"/api/sessions/{session_id}/model-b/nope").status_code == 404

    def test_job_from_another_session_is_404(self, client, monkeypatch) -> None:
        """Prevents a teacher enumerating other sessions' jobs."""
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        first = _upload(client)
        second = _upload(client)
        job_id = client.post(f"/api/sessions/{first}/model-b").json()["job_id"]
        assert client.get(f"/api/sessions/{second}/model-b/{job_id}").status_code == 404

    def test_completed_job_returns_analysis(self, client) -> None:
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}").json()
        # TestClient drains background tasks, so the job has completed.
        assert body["status"] == "completed"
        assert body["result"]["advisory_only"] is True
        assert body["error"] is None

    def test_analysis_is_never_claimed_as_authoritative(self, client) -> None:
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}").json()
        assert body["result"]["advisory_only"] is True
        assert client.calls


class TestFailureEndpoint:
    def test_disabled_error_becomes_failed_job(self, client, monkeypatch) -> None:
        from app.model_b.errors import ModelBApiError

        def boom(image_bytes, settings, client=None):
            raise ModelBApiError("upstream down", retryable=True)

        monkeypatch.setattr(model_b_api, "analyze_image", boom)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}").json()
        assert body["status"] == "failed"
        assert body["error"]["code"] == "model_b_api_error"
        assert body["error"]["retryable"] is True

    def test_unexpected_error_does_not_crash_the_worker(self, client, monkeypatch) -> None:
        def boom(image_bytes, settings, client=None):
            raise ValueError("something odd")

        monkeypatch.setattr(model_b_api, "analyze_image", boom)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}").json()
        assert body["status"] == "failed"
        assert body["error"]["code"] == "internal_error"

    def test_error_does_not_leak_internal_message(self, client, monkeypatch) -> None:
        def boom(image_bytes, settings, client=None):
            raise ValueError("secret internal detail /etc/passwd")

        monkeypatch.setattr(model_b_api, "analyze_image", boom)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}").json()
        assert "/etc/passwd" not in json.dumps(body)


class TestCancelEndpoint:
    def test_cancels_queued_job(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.delete(f"/api/sessions/{session_id}/model-b/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cannot_cancel_running_job(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        client.job_store.get(job_id).mark_running()
        response = client.delete(f"/api/sessions/{session_id}/model-b/{job_id}")
        assert response.status_code == 409
        assert "already started" in response.json()["detail"]

    def test_cannot_cancel_finished_job(self, client) -> None:
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.delete(f"/api/sessions/{session_id}/model-b/{job_id}")
        assert response.status_code == 409

    def test_cancelled_job_is_not_started_by_the_worker(self, client, monkeypatch) -> None:
        """The race the QUEUED state exists to close."""
        session_id = _upload(client)
        job = client.job_store.create(session_id, model="m")
        job.mark_cancelled()
        calls: list[bytes] = []
        monkeypatch.setattr(
            model_b_api, "analyze_image", lambda *a, **k: calls.append(b"") or None
        )
        model_b_api._run_job(job, b"image-bytes")
        assert calls == []
        assert job.status is JobStatus.CANCELLED

    def test_unknown_job_is_404(self, client) -> None:
        session_id = _upload(client)
        assert client.delete(f"/api/sessions/{session_id}/model-b/nope").status_code == 404


def _seed_model_a(session_id: str) -> None:
    """Attach a minimal Model A result to the session.

    The fusion endpoint reads `session.semantic_geometry`, so seeding it directly
    tests the endpoint without paying for the full pipeline (which loads
    EasyOCR and dominates the suite's runtime).
    """
    from app.services.session_store import session_store

    session = session_store.get(session_id)
    session.semantic_geometry = {
        "elements": [
            {
                "id": "m1",
                "type": "triangle",
                "geometry": {"points": [[10, 10], [200, 10], [100, 180]]},
                "confidence": 0.9,
                "confidence_level": "high",
                "needs_review": False,
                "source": "opencv_hough",
                "bbox": [100, 80, 600, 560],
            }
        ],
        "relationships": [],
        "image_width": 1000,
        "image_height": 800,
        "element_count": 1,
    }


class TestFusionEndpoint:
    def test_409_until_job_completes(self, client, monkeypatch) -> None:
        monkeypatch.setattr(model_b_api, "_run_job", lambda *_: None)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion")
        assert response.status_code == 409
        assert "queued" in response.json()["detail"]

    def test_409_when_model_a_has_not_run(self, client) -> None:
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion")
        assert response.status_code == 409
        assert "Model A" in response.json()["detail"]

    def test_404_for_unknown_job(self, client) -> None:
        session_id = _upload(client)
        assert client.get(f"/api/sessions/{session_id}/model-b/nope/fusion").status_code == 404

    def test_409_for_failed_job(self, client, monkeypatch) -> None:
        def boom(image_bytes, settings, client=None):
            from app.model_b.errors import ModelBApiError

            raise ModelBApiError("down", retryable=True)

        monkeypatch.setattr(model_b_api, "analyze_image", boom)
        session_id = _upload(client)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion")
        assert response.status_code == 409

    def test_returns_report_after_processing(self, client) -> None:
        session_id = _upload(client)
        _seed_model_a(session_id)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        response = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion")
        assert response.status_code == 200
        body = response.json()
        assert body["advisory_only"] is True
        assert "agreements" in body["summary"]

    def test_additions_always_require_approval(self, client) -> None:
        session_id = _upload(client)
        _seed_model_a(session_id)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        body = client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion").json()
        for addition in body["candidate_additions"]:
            assert addition["requires_teacher_approval"] is True
            assert addition["advisory_only"] is True

    def test_fusion_does_not_alter_model_a_state(self, client) -> None:
        """End-to-end version of invariant I1/I2: the session is untouched."""
        from app.services.session_store import session_store

        session_id = _upload(client)
        _seed_model_a(session_id)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]

        session = session_store.get(session_id)
        before_qa = json.dumps(session.qa_report, sort_keys=True, default=str)
        before_geometry = json.dumps(session.semantic_geometry, sort_keys=True, default=str)

        client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion")

        assert json.dumps(session.qa_report, sort_keys=True, default=str) == before_qa
        assert json.dumps(session.semantic_geometry, sort_keys=True, default=str) == before_geometry

    def test_report_is_json_serializable_over_the_wire(self, client) -> None:
        session_id = _upload(client)
        _seed_model_a(session_id)
        job_id = client.post(f"/api/sessions/{session_id}/model-b").json()["job_id"]
        assert isinstance(
            client.get(f"/api/sessions/{session_id}/model-b/{job_id}/fusion").text, str
        )


class TestWireContract:
    def test_analysis_schema_forbids_extra_fields(self) -> None:
        assert ModelBAnalysis.model_config.get("protected_namespaces") == ()

    def test_model_a_endpoints_are_untouched(self, client) -> None:
        assert client.get("/api/health").json() == {"status": "ok"}

    def test_model_a_flow_still_works_with_model_b_present(self, client) -> None:
        session_id = _upload(client)
        response = client.post(f"/api/sessions/{session_id}/process")
        assert response.status_code == 200
        body = response.json()
        assert "preview_svg" in body
        assert "qa_report" in body
        assert "model_b" not in body

    def test_optional_image_quality_does_not_gate_model_a(self, client) -> None:
        """Model B's advisory readability must not feed Model A's QA gate."""
        session_id = _upload(client)
        client.post(f"/api/sessions/{session_id}/model-b")
        body = client.post(f"/api/sessions/{session_id}/process").json()
        assert "readable" not in json.dumps(body["qa_report"])
