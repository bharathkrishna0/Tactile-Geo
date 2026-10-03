"""Model A + Model B (+ fusion, + simulated teacher) for the benchmark runner.

Model B is called once per image and its outcome (result or failure) is cached
on disk by image hash and model, so every teacher policy is scored against the
same provider responses. Failures are recorded by category and are never
replaced by Model A output silently: the captured result says what happened.

Teacher policies (``MODEL_B_TEACHER``):

* ``none``        fusion only; the tactile output stays Model A's.
* ``accept_all``  accept every finding fusion pairs with a Model A element.
* ``oracle``      accept a finding only when ground truth says it is right
                  (upper bound on what a careful teacher could get).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from app.core.config import model_b_settings
from app.model_b import errors as mb_errors
from app.model_b.fusion import reconcile
from app.model_b.result_codec import result_from_dict, result_to_dict
from app.model_b.semantic_v2 import build_semantic_geometry_v2
from app.model_b.client import ModelBClientProtocol
from app.model_b.service import ModelBSettings, analyze_image
from app.services.editing import refresh_semantic_fields, regenerate

from . import metrics
from .capture import capture_result

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("none", "accept_all", "oracle")

FAILURE_CATEGORY = [
    (mb_errors.ModelBRateLimited, "rate_limited"),
    (mb_errors.ModelBTimeout, "timeout"),
    (mb_errors.ModelBMalformedResponse, "invalid_response"),
    (mb_errors.ModelBValidationError, "invalid_response"),
    (mb_errors.ModelBApiError, "api_error"),
    (mb_errors.ModelBMisconfigured, "misconfigured"),
    (mb_errors.ModelBDisabled, "disabled"),
]


def failure_category(error: Exception) -> str:
    for cls, name in FAILURE_CATEGORY:
        if isinstance(error, cls):
            return name
    return "other_error"


class ModelBRunner:
    def __init__(self, policy: str | None = None, cache_dir: Path | None = None,
                 settings: ModelBSettings | None = None, client: ModelBClientProtocol | None = None) -> None:
        self.settings = settings or model_b_settings()
        self.settings.require_available()
        self.client = client
        self.policy = policy or os.getenv("MODEL_B_TEACHER", "accept_all")
        if self.policy not in POLICIES:
            raise ValueError(f"MODEL_B_TEACHER must be one of {POLICIES}")
        self.cache_dir = cache_dir or ROOT / "evaluation" / "results" / "model_b_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {"provider": self.settings.provider, "requested_model": self.settings.model,
                "timeout_s": self.settings.timeout_s, "max_attempts": self.settings.max_attempts,
                "teacher_policy": self.policy, "free_models_only": self.settings.model.endswith(":free")
                or self.settings.model == "openrouter/free"}

    def _cache_path(self, data: bytes) -> Path:
        model = re.sub(r"[^A-Za-z0-9]+", "_", self.settings.model)
        return self.cache_dir / f"{hashlib.sha256(data).hexdigest()[:16]}_{model}.json"

    def call(self, data: bytes) -> dict:
        path = self._cache_path(data)
        if path.exists():
            return {**json.loads(path.read_text()), "from_cache": True}
        start = time.perf_counter()
        try:
            result = analyze_image(data, self.settings, self.client)
            record = {"status": "ok", "result": result_to_dict(result), "resolved_model": result.resolved_model,
                      "truncated": result.truncated, "usage": result.usage}
        except mb_errors.ModelBError as error:
            record = {"status": failure_category(error), "error": f"{type(error).__name__}: {str(error)[:300]}"}
        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n")
        return record

    def _decisions(self, report, gt: dict, captured: dict) -> dict[str, str]:
        paired = [r for r in report.entity_reviews if r.model_a_id is not None]
        if self.policy == "none":
            return {}
        if self.policy == "accept_all":
            return {r.model_b_id: "accept" for r in paired}
        preds = [e for e in captured["semantic"]["elements"] if e["type"] in metrics.GEOMETRY_PRED]
        m = metrics.match_objects(gt["objects"], preds, gt["width"], gt["height"])
        truth = {preds[j]["id"]: m["gts"][i] for i, j in m["pairs"]}
        decisions = {}
        for r in paired:
            obj = truth.get(r.model_a_id)
            if obj is None or obj["semantic_importance"] == "omittable":
                decisions[r.model_b_id] = "reject"
                continue
            expected = metrics.EXPECTED_TYPE.get(obj["type"])
            if r.contradicts_model_a:
                decisions[r.model_b_id] = "accept" if expected and r.model_b_kind == expected else "reject"
            else:
                decisions[r.model_b_id] = "accept"
        return decisions

    def augment(self, image_id: str, data: bytes, gt: dict, captured: dict, result, translator) -> dict:
        if captured.get("error"):
            captured["model_b"] = {"status": "skipped_model_a_failed"}
            return captured
        record = self.call(data)
        info = {"status": record["status"], "latency_ms": record["latency_ms"], "from_cache": record.get("from_cache", False),
                "resolved_model": record.get("resolved_model"), "truncated": record.get("truncated"),
                "error": record.get("error"), "policy": self.policy}
        timings = dict(captured["timings_ms"])
        timings["model_b"] = record["latency_ms"]
        if record["status"] != "ok":
            timings["total"] = round(timings["total"] + record["latency_ms"], 1)
            captured["timings_ms"] = timings
            captured["model_b"] = info
            return captured
        mb = result_from_dict(record["result"])
        start = time.perf_counter()
        report = reconcile(result.semantic_geometry, mb)
        timings["fusion"] = round((time.perf_counter() - start) * 1000, 1)
        info.update(entities=len(mb.entities), relationships=len(mb.relationships),
                    diagram_relations=len(mb.diagram_relations), fusion=report.summary(),
                    model_b_kind_accuracy=self._kind_accuracy(report, gt, captured))
        decisions = self._decisions(report, gt, captured)
        start = time.perf_counter()
        v2 = build_semantic_geometry_v2(result.semantic_geometry, mb, decisions, f"bench-{image_id}")
        semantic = refresh_semantic_fields(v2.semantic)
        simplified, qa_report, tactile_svg = regenerate(semantic)
        timings["apply_and_compile"] = round((time.perf_counter() - start) * 1000, 1)
        info.update(accepted=sum(1 for d in decisions.values() if d == "accept"),
                    applied=Counter(o.change for o in v2.applied), not_applied=len(v2.not_applied))
        applied_result = copy.copy(result)
        applied_result.semantic_geometry = semantic
        applied_result.simplified_geometry = simplified
        applied_result.qa_report = qa_report
        applied_result.tactile_svg = tactile_svg
        timings["total"] = round(timings["total"] + record["latency_ms"] + timings["fusion"] + timings["apply_and_compile"], 1)
        new = capture_result(applied_result, {}, 0.0)
        new["timings_ms"] = timings
        new["model_b"] = info
        return new

    @staticmethod
    def _kind_accuracy(report, gt: dict, captured: dict) -> dict:
        preds = [e for e in captured["semantic"]["elements"] if e["type"] in metrics.GEOMETRY_PRED]
        m = metrics.match_objects(gt["objects"], preds, gt["width"], gt["height"])
        truth = {preds[j]["id"]: m["gts"][i] for i, j in m["pairs"]}
        total = correct = model_a_correct = 0
        for r in report.entity_reviews:
            obj = truth.get(r.model_a_id) if r.model_a_id else None
            expected = metrics.EXPECTED_TYPE.get(obj["type"]) if obj else None
            if expected is None or obj["semantic_importance"] == "omittable":
                continue
            total += 1
            correct += r.model_b_kind == expected
            model_a_correct += r.model_a_id is not None and next(
                (p["type"] for p in preds if p["id"] == r.model_a_id), None) == expected
        return {"paired_with_truth": total, "model_b_kind_correct": correct, "model_a_type_correct": model_a_correct}

    def summary(self, rows: list[dict]) -> dict:
        infos = [r["captured"].get("model_b", {}) for r in rows]
        statuses = Counter(i.get("status") for i in infos)
        called = [i for i in infos if i.get("status") != "skipped_model_a_failed"]
        n = len(called)
        kind = Counter()
        for i in infos:
            for k, v in (i.get("model_b_kind_accuracy") or {}).items():
                kind[k] += v
        return {
            **self.describe(),
            "images_called": n,
            "status_counts": dict(statuses),
            "success_rate": metrics.ratio(statuses.get("ok", 0), n),
            "invalid_response_rate": metrics.ratio(statuses.get("invalid_response", 0), n),
            "timeout_rate": metrics.ratio(statuses.get("timeout", 0), n),
            "rate_limited_rate": metrics.ratio(statuses.get("rate_limited", 0), n),
            "truncated_ok_responses": sum(1 for i in infos if i.get("status") == "ok" and i.get("truncated")),
            "resolved_models": dict(Counter(i.get("resolved_model") for i in infos if i.get("status") == "ok")),
            "latency_ms": metrics.percentiles([i["latency_ms"] for i in called if "latency_ms" in i]),
            "accepted_findings": sum(i.get("accepted", 0) for i in infos),
            "applied_changes": dict(sum((Counter(i.get("applied", {})) for i in infos), Counter())),
            "kind_accuracy_on_paired_truth": {
                **kind, "model_b": metrics.ratio(kind["model_b_kind_correct"], kind["paired_with_truth"]),
                "model_a": metrics.ratio(kind["model_a_type_correct"], kind["paired_with_truth"])},
            "cost_usd": 0.0 if self.describe()["free_models_only"] else "NOT MEASURED",
        }
