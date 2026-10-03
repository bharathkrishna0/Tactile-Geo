"""Hermetic tests for the Model B benchmark runner: no key, no network."""

import json

import cv2
import numpy as np
import pytest

from app.model_b import errors as mb_errors
from app.model_b._fixtures import valid_document
from app.model_b.client import ProviderResponse
from app.model_b.service import ModelBSettings
from evaluation.model_b_eval import ModelBRunner, failure_category

ON = ModelBSettings(enabled=True, api_key="test-key", model="vendor/model:free", timeout_s=5.0,
                    max_attempts=1, retry_base_delay_s=0.0, retry_max_delay_s=0.0)


class FakeClient:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text, self.error, self.calls = text, error, 0

    def analyze(self, *, image_png: bytes, prepared_width: int, prepared_height: int):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ProviderResponse(text=self.text, finish_reason="stop", requested_model=ON.model, model=ON.model)


def png(seed: int = 0) -> bytes:
    img = np.full((800, 1000, 3), 255, np.uint8)
    cv2.line(img, (100 + seed, 100), (700, 640), (0, 0, 0), 3)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.mark.parametrize("error,category", [
    (mb_errors.ModelBRateLimited("429"), "rate_limited"),
    (mb_errors.ModelBTimeout("slow"), "timeout"),
    (mb_errors.ModelBMalformedResponse("not json"), "invalid_response"),
    (mb_errors.ModelBValidationError("bad", ["x"]), "invalid_response"),
    (mb_errors.ModelBApiError("500"), "api_error"),
    (mb_errors.ModelBMisconfigured("no key"), "misconfigured"),
    (mb_errors.ModelBCancelled("stop"), "other_error"),
])
def test_failure_categories(error, category):
    assert failure_category(error) == category


def test_runner_refuses_without_key(tmp_path):
    with pytest.raises(mb_errors.ModelBMisconfigured):
        ModelBRunner(cache_dir=tmp_path, settings=ModelBSettings(enabled=True))


def test_unknown_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        ModelBRunner(policy="guess", cache_dir=tmp_path, settings=ON, client=FakeClient())


def test_success_is_cached_and_reused(tmp_path):
    client = FakeClient(text=json.dumps(valid_document()))
    runner = ModelBRunner(policy="none", cache_dir=tmp_path, settings=ON, client=client)
    first = runner.call(png())
    assert first["status"] == "ok" and first["result"]["entities"]
    assert runner.call(png())["result"] == first["result"]
    assert client.calls == 1
    runner.call(png(seed=5))
    assert client.calls == 2


def test_provider_failure_is_recorded_not_replaced(tmp_path):
    runner = ModelBRunner(cache_dir=tmp_path, settings=ON, client=FakeClient(error=mb_errors.ModelBRateLimited("429")))
    record = runner.call(png())
    assert record["status"] == "rate_limited" and "result" not in record
    captured = {"timings_ms": {"total": 10.0}, "semantic": {"elements": [{"id": "a"}]}}
    out = runner.augment("x", png(), {}, captured, None, None)
    assert out["model_b"]["status"] == "rate_limited"
    assert out["semantic"] == {"elements": [{"id": "a"}]}
    assert out["timings_ms"]["model_b"] == record["latency_ms"]


def test_model_a_failure_skips_model_b(tmp_path):
    client = FakeClient(text=json.dumps(valid_document()))
    runner = ModelBRunner(cache_dir=tmp_path, settings=ON, client=client)
    out = runner.augment("x", png(), {}, {"error": "boom"}, None, None)
    assert out["model_b"]["status"] == "skipped_model_a_failed" and client.calls == 0


def test_summary_rates_count_failures(tmp_path):
    runner = ModelBRunner(cache_dir=tmp_path, settings=ON, client=FakeClient())
    rows = [{"captured": {"model_b": m}} for m in (
        {"status": "ok", "latency_ms": 70000.0, "truncated": False, "resolved_model": "m"},
        {"status": "rate_limited", "latency_ms": 500.0},
        {"status": "invalid_response", "latency_ms": 60000.0},
        {"status": "skipped_model_a_failed"},
    )]
    summary = runner.summary(rows)
    assert summary["images_called"] == 3
    assert summary["success_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["rate_limited_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["cost_usd"] == 0.0 and summary["free_models_only"] is True
