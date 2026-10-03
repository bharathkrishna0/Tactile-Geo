"""Aggregate per-image scores into split-level metrics (micro-averaged)."""

from __future__ import annotations

from collections import Counter

from .metrics import normalize_text, percentiles, prf, ratio

GEOMETRY_KEYS = ("chamfer_px", "endpoint_px", "angle_deg", "length_rel", "polygon_iou", "vertex_count_match",
                 "circle_center_px", "circle_radius_rel")


def _sum(rows, *path):
    total = 0
    for r in rows:
        v = r
        for p in path:
            v = v.get(p, 0) if isinstance(v, dict) else 0
        total += v or 0
    return total


def summarize(rows: list[dict]) -> dict:
    """``rows`` are per-image dicts holding ``score`` (from metrics.score_image)."""
    scored = [r["score"] for r in rows if not r["score"].get("failed")]
    failed = [r for r in rows if r["score"].get("failed")]
    n = len(rows)
    out: dict = {"images": n, "failed": len(failed), "failure_rate": ratio(len(failed), n)}
    if not n:
        return out
    # Objects: failed images count every counted object as missed.
    fail_gt = sum(r["score"].get("gt_essential", 0) for r in failed)
    for key, label in (("objects", "objects_semantic"), ("objects_final", "objects_final")):
        tp = _sum(scored, key, "strict", "tp")
        fp = _sum(scored, key, "strict", "fp")
        fn = _sum(scored, key, "strict", "fn")
        tps = _sum(scored, key, "structural", "tp")
        fns = _sum(scored, key, "structural", "fn")
        ess = _sum(scored, key, "gt_essential") + fail_gt
        out[label] = {
            "strict": prf(tp, fp, fn),
            "structural": prf(tps, fp, fns),
            "essential_recall_structural": ratio(_sum(scored, key, "essential_found_structural"), ess),
            "essential_recall_strict": ratio(_sum(scored, key, "essential_found_strict"), ess),
            "found_as_parts": _sum(scored, key, "found_as_parts"),
            "fragmented_linear": _sum(scored, key, "fragmented_linear"),
            "closed_as_sides": _sum(scored, key, "closed_as_sides"),
            "fp_reasons": dict(sum((Counter(s[key]["fp_reasons"]) for s in scored), Counter())),
            "on_omittable": _sum(scored, key, "on_omittable"),
            "note": "failed images are excluded from precision/recall but included in essential recall denominators",
        }
    obj = [s["objects"] for s in scored]
    per_type: dict[str, dict] = {}
    for o in obj:
        for t, d in o["per_gt_type"].items():
            agg = per_type.setdefault(t, {"gt": 0, "found_strict": 0, "found_structural": 0})
            for k in agg:
                agg[k] += d[k]
    for d in per_type.values():
        d["recall_strict"] = ratio(d["found_strict"], d["gt"])
        d["recall_structural"] = ratio(d["found_structural"], d["gt"])
    pred_type: dict[str, dict] = {}
    for o in obj:
        for t, d in o["per_pred_type"].items():
            agg = pred_type.setdefault(t, {"predicted": 0, "true_positive": 0})
            for k in agg:
                agg[k] += d[k]
    for d in pred_type.values():
        d["precision"] = ratio(d["true_positive"], d["predicted"])
    out["per_gt_type"] = dict(sorted(per_type.items()))
    out["per_pred_type"] = dict(sorted(pred_type.items()))
    out["classification"] = {
        "object_type_accuracy": ratio(_sum(obj, "type_correct"), _sum(obj, "type_total")),
        "matched_with_supported_type": _sum(obj, "type_total"),
        "confusion": dict(sum((Counter(o["type_confusion"]) for o in obj), Counter()).most_common()),
        "diagram_type_accuracy": "NOT MEASURED (Model A produces no diagram-level type)",
    }
    out["right_angle_markers"] = {k: _sum(obj, "right_angle_markers", k) for k in ("gt", "found", "predicted")}
    out["right_angle_markers"]["recall"] = ratio(out["right_angle_markers"]["found"], out["right_angle_markers"]["gt"])
    geo = {k: percentiles([v for o in obj for v in o["geometry_errors"][k]]) for k in GEOMETRY_KEYS}
    pts = [s["points"] for s in scored]
    geo["point_localization_px"] = percentiles([v for p in pts for v in p["error_px"]])
    geo["point_recall"] = ratio(_sum(pts, "found"), _sum(pts, "gt"))
    out["geometry"] = geo

    lab = [s["labels"] for s in scored]
    out["ocr"] = {
        "label_detection_recall": ratio(_sum(lab, "detected"), _sum(lab, "gt_essential")),
        "label_detection_precision": ratio(sum((l["detection_precision"] or 0) * l["predicted"] for l in lab), _sum(lab, "predicted")),
        "cer": ratio(_sum(lab, "char_errors"), _sum(lab, "chars")),
        "cer_detected_only": ratio(sum(pl["char_errors"] for l in lab for pl in l["per_label"] if pl["detected"]),
                                   sum(len(normalize_text(pl["gt"])) for l in lab for pl in l["per_label"] if pl["detected"])),
        "wer": ratio(_sum(lab, "word_errors"), _sum(lab, "words")),
        "exact_match_rate_detected": ratio(_sum(lab, "exact_match"), _sum(lab, "detected")),
        "exact_match_rate_normalized_detected": ratio(_sum(lab, "exact_match_normalized"), _sum(lab, "detected")),
        "label_association_accuracy": ratio(_sum(lab, "association_correct"), _sum(lab, "association_total")),
        "association_evaluated": _sum(lab, "association_total"),
    }
    out["braille"] = {
        "evaluated_labels": _sum(lab, "braille_total"),
        "end_to_end_exact": ratio(_sum(lab, "braille_exact"), _sum(lab, "braille_total")),
        "translation_consistency": ratio(_sum(lab, "braille_consistent"), _sum(lab, "braille_total")),
        "cell_error_rate": ratio(_sum(lab, "braille_cell_errors"), _sum(lab, "braille_cells")),
        "reading_order_concordance": ratio(_sum(lab, "reading_order_concordant"), _sum(lab, "reading_order_pairs")),
        "collision_violations": _sum(scored, "braille", "collisions"),
        "on_line_violations": _sum(scored, "braille", "on_line"),
    }
    for key, label in (("relationships", "relationships_semantic"), ("relationships_final", "relationships_final")):
        rel = [s[key] for s in scored]
        by_type: dict[str, dict] = {}
        for r in rel:
            for t, d in r["by_type"].items():
                agg = by_type.setdefault(t, {"gt": 0, "found": 0, "predicted_mappable": 0})
                for k in agg:
                    agg[k] += d[k]
        for d in by_type.values():
            d["recall"] = ratio(d["found"], d["gt"])
        rec = ratio(_sum(rel, "found"), _sum(rel, "gt_supported"))
        prec = ratio(_sum(rel, "predicted_mappable_correct"), _sum(rel, "predicted_mappable"))
        out[label] = {
            "recall": rec, "precision_lower_bound": prec,
            "f1_lower_bound": round(2 * rec * prec / (rec + prec), 4) if rec and prec else None,
            "gt_supported": _sum(rel, "gt_supported"),
            "gt_unsupported_by_type": dict(sum((Counter(r["gt_unsupported"]) for r in rel), Counter())),
            "by_type": by_type,
        }
    simp = [s["simplification"] for s in scored]
    out["simplification"] = {
        "actions": dict(sum((Counter(s["actions"]) for s in simp), Counter())),
        "removed": _sum(simp, "removed"), "merged": _sum(simp, "merged"), "modified": _sum(simp, "modified"),
        "newly_inferred": _sum(simp, "newly_inferred"),
        "essential_found_before": _sum(simp, "gt_found_before"),
        "essential_found_after": _sum(simp, "gt_found_after"),
        "essential_removed_by_simplification": sum(len(s["essential_lost_ids"]) for s in simp),
        "semantic_preservation_rate": ratio(_sum(simp, "gt_found_after"), _sum(simp, "gt_found_before")),
    }
    lab_ok = _sum(lab, "detected")
    out["final_preservation"] = {
        "essential_elements": out["objects_final"]["essential_recall_structural"],
        "relationships": out["relationships_final"]["recall"],
        "labels": ratio(lab_ok, _sum(lab, "gt_essential")),
        "quantitative_labels_exact": ratio(_sum(lab, "quantitative_preserved"), _sum(lab, "quantitative_labels")),
    }
    qa = [s["qa"] for s in scored]
    svg = [s["svg"] for s in scored]
    out["tactile_qa"] = {
        "pass_rate_all_images": ratio(sum(q["passes"] for q in qa), n),
        "issues": dict(sum((Counter(q["issues"]) for q in qa), Counter()).most_common()),
        "blocking_images": dict(sum((Counter(q["blocking"]) for q in qa), Counter()).most_common()),
    }
    out["svg"] = {
        "svg_rate_all_images": ratio(sum(1 for s in svg if s.get("present")), n),
        "valid_rate_of_present": ratio(sum(1 for s in svg if s.get("valid")), sum(1 for s in svg if s.get("present"))),
        "out_of_bounds": _sum(svg, "out_of_bounds"),
        "duplicates": _sum(svg, "duplicates"),
        "self_intersecting_polygons": _sum(svg, "self_intersecting_polygons"),
        "open_paths": _sum(svg, "open_paths"),
        "near_duplicate_geometry": _sum(svg, "near_duplicate_geometry"),
        "images_over_40_features": sum(1 for s in svg if s.get("over_target")),
        "images_over_60_features": sum(1 for s in svg if s.get("over_limit")),
        "embossed_features": percentiles([s["embossed_features"] for s in svg]),
    }
    stages = sorted({k for r in rows for k in r["score"].get("timings_ms", r.get("timings_ms", {}))})
    timing_rows = [r.get("timings_ms") or r["score"].get("timings_ms", {}) for r in rows]
    out["latency_ms"] = {k: percentiles([t[k] for t in timing_rows if k in t]) for k in stages}
    return out


def group(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        for k in key(r):
            groups.setdefault(k, []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}
