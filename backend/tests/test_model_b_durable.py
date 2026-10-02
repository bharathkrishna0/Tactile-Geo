"""Durable Model B jobs: result encoding, cache, recovery and atomic claims."""
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import model_b as model_b_api
from app.model_b._fixtures import minimal_document, valid_document
from app.model_b.cache import model_b_cache_key
from app.model_b.normalizer import normalize
from app.model_b.preparation import prepare_image
from app.model_b.result_codec import result_from_dict, result_to_dict
from app.model_b.service import ModelBSettings
from app.model_b.validator import validate_document
from app.models.model_b_job import JobStatus, ModelBJobError
from app.services.model_b_store import ModelBJobStore


def _image(shade: int = 200) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((400, 300, 3), shade, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _result(image_bytes: bytes, document: dict | None = None):
    prepared = prepare_image(image_bytes)
    document, warnings = validate_document(
        document or minimal_document(), prepared_width=prepared.width, prepared_height=prepared.height
    )
    return normalize(document, prepared, finish_reason="STOP", validation_warnings=warnings)


def test_result_codec_round_trips_losslessly():
    result = _result(_image(), valid_document())
    assert result.entities and result.relationships and result.text_items, "fixture must exercise nested records"
    assert result_from_dict(result_to_dict(result)) == result


def test_cache_key_depends_on_image_and_model():
    settings = ModelBSettings(enabled=True, api_key="k", model="a")
    key = model_b_cache_key(_image(200), settings)
    assert key == model_b_cache_key(_image(200), settings)
    assert key != model_b_cache_key(_image(100), settings)
    assert key != model_b_cache_key(_image(200), ModelBSettings(enabled=True, api_key="k", model="b"))


@pytest.fixture
def client(monkeypatch):
    import app.main as main_module
    from app.services.model_b_store import model_b_job_store
    from app.services.rate_limiter import upload_rate_limiter

    model_b_job_store.clear()
    upload_rate_limiter._requests.clear()
    calls: list[bytes] = []

    def fake_analyze(image_bytes, settings, client=None):
        calls.append(image_bytes)
        return _result(image_bytes)

    monkeypatch.setattr(model_b_api, "analyze_image", fake_analyze)
    monkeypatch.setattr(
        model_b_api, "model_b_settings", lambda: ModelBSettings(enabled=True, api_key="k", model="m", timeout_s=5.0)
    )
    with TestClient(main_module.app) as test_client:
        test_client.calls = calls  # type: ignore[attr-defined]
        yield test_client


def _upload(client, image: bytes) -> str:
    response = client.post("/api/sessions", files={"image": ("w.png", image, "image/png")})
    assert response.status_code == 201
    return response.json()["session_id"]


def test_identical_image_reuses_the_cached_analysis(client):
    first = client.post(f"/api/sessions/{_upload(client, _image())}/model-b").json()
    assert first["cache_hit"] is False
    second_session = _upload(client, _image())
    second = client.post(f"/api/sessions/{second_session}/model-b").json()
    assert second["status"] == "completed"
    assert second["cache_hit"] is True
    assert second["result"]["entities"] == client.get(
        f"/api/sessions/{second_session}/model-b/{second['job_id']}"
    ).json()["result"]["entities"]
    assert len(client.calls) == 1


def test_a_different_image_is_not_served_from_the_cache(client):
    client.post(f"/api/sessions/{_upload(client, _image(200))}/model-b")
    other = client.post(f"/api/sessions/{_upload(client, _image(90))}/model-b").json()
    assert other["cache_hit"] is False
    assert len(client.calls) == 2


def test_failed_analyses_are_not_cached(client, monkeypatch):
    from app.model_b.errors import ModelBTimeout

    def timeout(image_bytes, settings, client=None):
        raise ModelBTimeout("slow")

    monkeypatch.setattr(model_b_api, "analyze_image", timeout)
    failed = client.post(f"/api/sessions/{_upload(client, _image())}/model-b").json()
    assert client.get(f"/api/sessions/{failed['session_id']}/model-b/{failed['job_id']}").json()["status"] == "failed"
    from app.services.model_b_store import model_b_job_store

    assert model_b_job_store.cache_get(model_b_cache_key(_image(), model_b_api.model_b_settings())) is None


def test_stale_jobs_are_failed_as_interrupted_and_retryable():
    store = ModelBJobStore()
    job = store.create("s", model="m")
    job.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    assert store.recover_interrupted(stale_after_s=60) == 1
    recovered = store.get(job.job_id)
    assert recovered.status is JobStatus.FAILED
    assert recovered.error.code == "interrupted" and recovered.error.retryable is True
    assert store.recover_interrupted(stale_after_s=60) == 0


def test_complete_does_not_resurrect_a_cancelled_job():
    store = ModelBJobStore()
    job = store.create("s", model="m")
    store.cancel(job.job_id)
    assert store.complete(job.job_id, _result(_image())) is None
    assert store.get(job.job_id).status is JobStatus.CANCELLED


# --- Postgres -----------------------------------------------------------------

DSN = os.getenv("TEST_DATABASE_URL", "").strip()
postgres = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def pg_store():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from migrate import apply_migrations

    from app.model_b.postgres_job_store import PostgresModelBJobStore
    from app.services.postgres_store import PostgresSessionStore
    from app.models.session import ConversionSession

    apply_migrations(DSN)
    sessions = PostgresSessionStore(DSN, ttl_seconds=3600)
    session = ConversionSession(session_id=str(uuid4()), original_filename="w.png", original_image_path=Path("/tmp/w.png"))
    sessions.add(session)
    store = PostgresModelBJobStore(DSN)
    return store, session.session_id


@postgres
def test_pg_job_lifecycle_survives_a_new_store_instance(pg_store):
    from app.model_b.postgres_job_store import PostgresModelBJobStore

    store, session_id = pg_store
    job = store.create(session_id, model="m", provider="openrouter", cache_key="key")
    assert store.has_active_job(session_id)
    assert store.begin(job.job_id).status is JobStatus.RUNNING
    result = _result(_image())
    store.complete(job.job_id, result)

    restarted = PostgresModelBJobStore(DSN)
    loaded = restarted.get(job.job_id)
    assert loaded.status is JobStatus.COMPLETED
    assert loaded.result == result
    assert not restarted.has_active_job(session_id)
    assert [j.job_id for j in restarted.list_for_session(session_id)] == [job.job_id]


@postgres
def test_pg_claim_is_atomic_across_concurrent_workers(pg_store):
    store, session_id = pg_store
    job = store.create(session_id, model="m")
    winners: list[bool] = []
    barrier = threading.Barrier(8)

    def claim():
        barrier.wait()
        winners.append(store.begin(job.job_id) is not None)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert winners.count(True) == 1


@postgres
def test_pg_cancel_wins_over_a_late_claim_and_blocks_completion(pg_store):
    store, session_id = pg_store
    job = store.create(session_id, model="m")
    assert store.cancel(job.job_id).status is JobStatus.CANCELLED
    assert store.begin(job.job_id) is None
    assert store.complete(job.job_id, _result(_image())) is None
    assert store.fail(job.job_id, ModelBJobError(code="x", message="m")) is None
    assert store.get(job.job_id).status is JobStatus.CANCELLED


@postgres
def test_pg_interrupted_jobs_are_recovered(pg_store):
    import psycopg

    store, session_id = pg_store
    job = store.create(session_id, model="m")
    store.begin(job.job_id)
    with psycopg.connect(DSN) as connection:
        connection.execute(
            "update model_b_jobs set started_at = now() - interval '1 hour' where id = %s", (job.job_id,)
        )
    assert store.recover_interrupted(stale_after_s=60) >= 1
    recovered = store.get(job.job_id)
    assert recovered.status is JobStatus.FAILED and recovered.error.code == "interrupted"


@postgres
def test_pg_cache_round_trip(pg_store):
    store, _ = pg_store
    key = f"test:{uuid4()}"
    assert store.cache_get(key) is None
    result = _result(_image())
    store.cache_put(key, result)
    assert store.cache_get(key) == result
