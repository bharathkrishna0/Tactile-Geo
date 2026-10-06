"""Run Model A on one image and capture every intermediate as plain JSON.

Stage timings are taken by wrapping the functions ``app.services.pipeline``
calls, so the measured code path is exactly ``build_full_analysis`` with no
copy of the pipeline here.
"""

from __future__ import annotations

import contextlib
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from enum import Enum

STAGES = {
    "decode_image": "preprocessing",
    "assess_image_quality": "preprocessing",
    "enhance_copy": "preprocessing",
    "preprocess_image": "preprocessing",
    "find_diagram_regions": "preprocessing",
    "mask_to_regions": "preprocessing",
    "extract_shapes": "vectorization",
    "ink_contrast_copy": "ocr",
    "postprocess_detections": "ocr",
    "restore_radicals": "ocr",
    "drop_shapes_inside_text_regions": "vectorization",
    "detect_right_angle_markers": "vectorization",
    "drop_marker_strokes": "vectorization",
    "map_label_to_geometry": "labels_braille",
    "place_braille_markers": "labels_braille",
    "shapes_to_svg": "preview_svg",
    "analyze_diagram": "semantic_analysis",
    "simplify_geometry": "simplification",
    "run_tactile_qa": "tactile_qa",
    "render_tactile_svg": "tactile_svg",
}


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = defaultdict(float)

    def wrap(self, stage: str, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.ms[stage] += (time.perf_counter() - start) * 1000
        return timed


class _TimedProvider:
    def __init__(self, inner, timer: _Timer, stage: str, method: str) -> None:
        self._inner, self._timer, self._stage, self._method = inner, timer, stage, method

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        return self._timer.wrap(self._stage, attr) if name == self._method else attr


@contextlib.contextmanager
def timed_pipeline(timer: _Timer, overrides: dict | None = None):
    """Patch pipeline module globals with timed wrappers (and optional overrides)."""
    from app.services import pipeline

    saved = {}
    for name, stage in STAGES.items():
        saved[name] = getattr(pipeline, name)
        fn = (overrides or {}).get(name, saved[name])
        setattr(pipeline, name, timer.wrap(stage, fn))
    try:
        yield pipeline
    finally:
        for name, fn in saved.items():
            setattr(pipeline, name, fn)


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 3)
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    return value


def element_json(el, reading_order: dict[str, int] | None = None) -> dict:
    out = {
        "id": el.id, "type": _plain(el.type), "geometry": _plain(el.geometry), "bbox": _plain(el.bbox),
        "confidence": round(float(el.confidence), 3), "source": el.source,
        "associated_label_id": el.associated_label_id,
    }
    if out["type"] == "text_label":
        out["text"] = el.geometry.get("text", "")
        out["braille"] = el.geometry.get("braille", "")
        out["associated_element_id"] = el.associated_label_id
        out["association"] = _plain((el.semantic_properties or {}).get("association"))
    role = (el.semantic_properties or {}).get("role")
    if role:
        out["role"] = role
        if reading_order is not None:
            out["reading_order"] = reading_order.get(el.id)
    return out


def relationship_json(rel) -> dict:
    return {"id": rel.id, "type": _plain(rel.type), "element_ids": list(rel.element_ids),
            "confidence": round(float(rel.confidence), 3)}


def capture_result(result, timer_ms: dict, total_ms: float) -> dict:
    sem, simp = result.semantic_geometry, result.simplified_geometry
    # Text-label elements are created in the same order as the placed labels.
    sem_labels = [e for e in sem.elements if _plain(e.type) == "text_label"]
    order = {e.id: lab.get("reading_order") for e, lab in zip(sem_labels, result.labels)}
    return {
        "error": None,
        "image_quality": _plain(asdict(result.quality_report)) if hasattr(result.quality_report, "__dataclass_fields__") else None,
        "raw_shapes": len(result.shapes),
        "ocr_labels": [{"text": lab.get("text"), "bbox": _plain(lab.get("bbox")), "confidence": _plain(lab.get("confidence")),
                        "braille": lab.get("braille"), "reading_order": lab.get("reading_order")} for lab in result.labels],
        "semantic": {"elements": [element_json(e, order) for e in sem.elements],
                     "relationships": [relationship_json(r) for r in sem.relationships],
                     "review_flags": list(sem.review_flags)},
        "simplified": {"elements": [element_json(e, order) for e in simp.elements],
                       "relationships": [relationship_json(r) for r in simp.relationships],
                       "actions": [{"element_id": a.element_id, "action": a.action, "detail": a.detail} for a in simp.actions],
                       "removed_count": simp.removed_count, "merged_count": simp.merged_count},
        "qa": {"passes": bool(result.qa_report.passes), "score_0_100": result.qa_report.score_0_100,
               "issues": [{"check": i.check, "severity": i.severity, "message": i.message, "element_id": i.element_id}
                          for i in result.qa_report.issues]},
        "tactile_svg": result.tactile_svg,
        "timings_ms": {**{k: round(v, 1) for k, v in timer_ms.items()}, "total": round(total_ms, 1)},
    }


def run_model_a(image_bytes: bytes, ocr_provider, translator, overrides: dict | None = None) -> tuple[dict, object | None]:
    """Return (captured JSON, PipelineResult or None on failure)."""
    timer = _Timer()
    provider = _TimedProvider(ocr_provider, timer, "ocr", "detect")
    tr = _TimedProvider(translator, timer, "labels_braille", "translate")
    start = time.perf_counter()
    with timed_pipeline(timer, overrides) as pipeline:
        try:
            result = pipeline.build_full_analysis(image_bytes, ocr_provider=provider, braille_translator=tr)
        except Exception as error:  # a failed analysis is a scored outcome
            total = (time.perf_counter() - start) * 1000
            return ({"error": f"{type(error).__name__}: {error}",
                     "traceback_tail": traceback.format_exc().splitlines()[-4:],
                     "timings_ms": {**{k: round(v, 1) for k, v in timer.ms.items()}, "total": round(total, 1)}}, None)
    total = (time.perf_counter() - start) * 1000
    return capture_result(result, timer.ms, total), result
