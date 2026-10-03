"""Metrics engine: scores one captured TactileGeo result against ground truth.

Pure functions over plain dicts (the per-image result JSON and the annotation
JSON), so the arithmetic is unit-tested without running the pipeline.

Matching
--------
Every object is reduced to a dense sample of its outline. A prediction matches
a ground-truth object when their families are compatible and the symmetric
mean outline distance (chamfer) is at most ``tau = max(4 px, 1.5 % of the
image diagonal)``; one-to-one pairs are chosen by optimal assignment on that
distance. A ground-truth object that no single prediction explains, but whose
outline is >= 90 % covered by otherwise unmatched predictions lying on it, is
counted as ``found_as_parts`` (e.g. a triangle recovered as three segments, or
a line broken in two). Strict recall excludes those; structural recall
includes them.

Omittable ground truth (grid lines, titles, question text, name lines) is not
in any recall denominator, and predictions on it are neither true nor false
positives; they are reported as ``on_omittable``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

LINEAR_GT = {"line_segment", "ray", "arrow", "axis", "dimension_line", "tick", "grid_line", "table_line", "number_line"}
CLOSED_GT = {"triangle", "rectangle", "polygon", "bar"}
ROUND_GT = {"circle", "ellipse"}
ANY_GT = {"curve", "arc", "angle_marker"}
MARKER_GT = {"right_angle_marker"}
DISTRACTOR_GT = {"handwriting", "decoration"}

LINEAR_PRED = {"line_segment", "ray", "arrow", "axes"}
CLOSED_PRED = {"triangle", "rectangle", "polygon"}
ROUND_PRED = {"circle", "ellipse", "arc"}
GEOMETRY_PRED = LINEAR_PRED | CLOSED_PRED | ROUND_PRED

# The semantically correct Model A type for each ground-truth type; None = the
# Model A vocabulary has no type for it, so it is excluded from classification.
EXPECTED_TYPE = {
    "line_segment": "line_segment", "tick": "line_segment", "grid_line": "line_segment", "table_line": "line_segment",
    "ray": "ray", "arrow": "arrow", "dimension_line": "arrow", "number_line": "arrow", "axis": "axes",
    "triangle": "triangle", "rectangle": "rectangle", "bar": "rectangle", "polygon": "polygon",
    "circle": "circle", "ellipse": "ellipse", "arc": "arc", "angle_marker": "arc",
    "right_angle_marker": "angle", "curve": None,
}

# Ground-truth relation -> Model A relationship types that express it.
RELATION_MAP = {
    "PARALLEL": {"parallel_lines"},
    "PERPENDICULAR": {"perpendicular_lines"},
    "INTERSECTS": {"intersects", "connected_lines"},
    "ENDPOINT_OF": {"endpoint_of"},
    "ON": {"point_on_line", "point_on_circle"},
    "CENTER_OF": {"center_of", "circle_center"},
}
UNSUPPORTED_RELATIONS = {"VERTEX_OF", "INSIDE", "REFLECTION_OF", "TRANSLATION_OF", "DIMENSION_OF", "ANGLE_AT", "LABELS"}
PRED_TO_GT_RELATION = {p: g for g, ps in RELATION_MAP.items() for p in ps}

TACTILE_TARGET, TACTILE_LIMIT = 40, 60


# ------------------------------------------------------------------ helpers
def diag(width: float, height: float) -> float:
    return math.hypot(width, height)


def tolerance(width: float, height: float) -> float:
    return max(4.0, 0.015 * diag(width, height))


def _sample_polyline(points, closed: bool, step: float = 3.0) -> np.ndarray:
    pts = [tuple(map(float, p)) for p in points]
    if closed and len(pts) > 2:
        pts = pts + [pts[0]]
    if len(pts) == 1:
        return np.array(pts)
    out = []
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(pts[-1])
    return np.array(out)


def _sample_ellipse(center, a, b, angle_deg=0.0) -> np.ndarray:
    n = max(24, int(2 * math.pi * max(a, b) / 3))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    x, y = a * np.cos(t), b * np.sin(t)
    return np.stack([center[0] + x * ca - y * sa, center[1] + x * sa + y * ca], axis=1)


def gt_samples(obj: dict) -> np.ndarray:
    g = obj["geometry"]
    if obj.get("outline"):
        return _sample_polyline(obj["outline"], closed=g["kind"] in ("circle", "ellipse"))
    if g["kind"] in ("segment", "polyline"):
        return _sample_polyline(g["points"], closed=False)
    if g["kind"] == "polygon":
        return _sample_polyline(g["points"], closed=True)
    if g["kind"] == "circle":
        return _sample_ellipse(g["center"], g["radius"], g["radius"])
    if g["kind"] == "ellipse":
        return _sample_ellipse(g["center"], g["axes"][0], g["axes"][1], g.get("angle", 0.0))
    raise ValueError(f"unknown ground-truth geometry kind {g['kind']!r}")


def pred_samples(el: dict) -> np.ndarray | None:
    g = el.get("geometry") or {}
    if "start" in g and "end" in g:
        return _sample_polyline([g["start"], g["end"]], closed=False)
    if el["type"] == "circle" and "center" in g and "radius" in g:
        return _sample_ellipse(g["center"], float(g["radius"]), float(g["radius"]))
    if el["type"] == "ellipse" and "center" in g and "semi_axes" in g:
        return _sample_ellipse(g["center"], float(g["semi_axes"][0]), float(g["semi_axes"][1]), float(g.get("angle", 0.0)))
    if "points" in g and len(g["points"]) >= 2:
        return _sample_polyline(g["points"], closed=el["type"] in CLOSED_PRED)
    return None


def vertex_candidates(el: dict) -> list[tuple[float, float]]:
    g = el.get("geometry") or {}
    out = []
    for key in ("start", "end", "center", "position", "vertex"):
        if key in g and g[key] is not None:
            out.append((float(g[key][0]), float(g[key][1])))
    if el["type"] in CLOSED_PRED:
        out.extend((float(p[0]), float(p[1])) for p in g.get("points", []))
    return out


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float((da.mean() + db.mean()) / 2)


def compatible(gt_type: str, pred_type: str) -> bool:
    if gt_type in LINEAR_GT:
        return pred_type in LINEAR_PRED
    if gt_type in CLOSED_GT | ROUND_GT:
        return pred_type in CLOSED_PRED | ROUND_PRED
    if gt_type in ANY_GT:
        return pred_type in GEOMETRY_PRED
    return False


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), **{f"p{q}": round(float(np.percentile(arr, q)), 3) for q in (25, 50, 75, 95)},
            "mean": round(float(arr.mean()), 3)}


def ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def prf(tp: int, fp: int, fn: int) -> dict:
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("′", "'").replace("’", "'").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _point_in_box(p, box, pad=0.0) -> bool:
    x, y, w, h = box
    return x - pad <= p[0] <= x + w + pad and y - pad <= p[1] <= y + h + pad


def _center(box):
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


# ----------------------------------------------------------- object matching
def match_objects(gt_objects: list[dict], preds: list[dict], width: int, height: int) -> dict:
    """Optimal one-to-one matching plus the found-as-parts fallback."""
    tau = tolerance(width, height)
    gts = [o for o in gt_objects if o["type"] not in MARKER_GT | DISTRACTOR_GT]
    gs = [gt_samples(o) for o in gts]
    ps = [pred_samples(p) for p in preds]
    cost = np.full((len(gts), len(preds)), 1e9)
    dist = {}
    for i, (o, a) in enumerate(zip(gts, gs)):
        for j, (p, b) in enumerate(zip(preds, ps)):
            if b is None or not compatible(o["type"], p["type"]):
                continue
            d = chamfer(a, b)
            dist[(i, j)] = d
            if d <= tau:
                cost[i, j] = d
    pairs = []
    if gts and preds:
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= tau]
    used_g = {i for i, _ in pairs}
    used_p = {j for _, j in pairs}

    parts: dict[int, list[int]] = {}
    for i, o in enumerate(gts):
        if i in used_g or o["semantic_importance"] == "omittable":
            continue
        a = gs[i]
        tree_a = cKDTree(a)
        on_gt = []
        for j, b in enumerate(ps):
            if j in used_p or b is None or preds[j]["type"] not in GEOMETRY_PRED:
                continue
            d, _ = tree_a.query(b)
            if (d <= tau * 0.6).mean() >= 0.8:
                on_gt.append(j)
        if not on_gt:
            continue
        cover = cKDTree(np.vstack([ps[j] for j in on_gt]))
        d, _ = cover.query(a)
        if (d <= tau * 0.6).mean() >= 0.9:
            parts[i] = on_gt
            used_p.update(on_gt)
            used_g.add(i)
    return {"gts": gts, "pairs": pairs, "parts": parts, "dist": dist, "tau": tau,
            "unmatched_pred": [j for j in range(len(preds)) if j not in used_p]}


def _fp_reason(pred: dict, samples, gt: dict, tau: float) -> str:
    if samples is None:
        return "other"
    c = samples.mean(axis=0)
    for lab in gt["labels"]:
        if _point_in_box(c, lab["bbox"], pad=tau * 0.5):
            return "on_text"
    for d in (o for o in gt["objects"] if o["type"] in DISTRACTOR_GT):
        pts = np.asarray(d["geometry"]["points"], dtype=float)
        if len(pts) and cKDTree(pts).query(samples)[0].mean() <= tau:
            return "on_distractor"
    return "other"


def _linear_errors(gt_obj: dict, pred: dict) -> dict | None:
    g, p = gt_obj["geometry"], pred["geometry"]
    if g["kind"] != "segment" or "start" not in p:
        return None
    a, b = g["points"]
    s, e = p["start"], p["end"]
    endpoint = min((math.dist(a, s) + math.dist(b, e)) / 2, (math.dist(a, e) + math.dist(b, s)) / 2)
    ang_g = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
    ang_p = math.degrees(math.atan2(e[1] - s[1], e[0] - s[0])) % 180
    da = abs(ang_g - ang_p)
    lg = math.dist(a, b)
    return {"endpoint_px": endpoint, "angle_deg": min(da, 180 - da),
            "length_rel": abs(math.dist(s, e) - lg) / lg if lg else None}


def _polygon_errors(gt_obj: dict, pred: dict) -> dict | None:
    from shapely.geometry import Polygon

    g, p = gt_obj["geometry"], pred["geometry"]
    if g["kind"] != "polygon" or "points" not in p or len(p["points"]) < 3:
        return None
    pg, pp = Polygon(g["points"]).buffer(0), Polygon(p["points"]).buffer(0)
    union = pg.union(pp).area
    return {"iou": pg.intersection(pp).area / union if union else 0.0,
            "vertex_count_match": len(p["points"]) == len(g["points"])}


def _round_errors(gt_obj: dict, pred: dict) -> dict | None:
    g, p = gt_obj["geometry"], pred["geometry"]
    if g["kind"] != "circle" or "center" not in p:
        return None
    r = p.get("radius") or (sum(p["semi_axes"]) / 2 if p.get("semi_axes") else None)
    return {"center_px": math.dist(g["center"], p["center"]),
            "radius_rel": abs(float(r) - g["radius"]) / g["radius"] if r else None}


def score_objects(gt: dict, preds: list[dict]) -> dict:
    w, h = gt["width"], gt["height"]
    all_preds = preds
    preds = [p for p in all_preds if p["type"] in GEOMETRY_PRED]
    m = match_objects(gt["objects"], preds, w, h)
    gts, tau = m["gts"], m["tau"]
    counted = [i for i, o in enumerate(gts) if o["semantic_importance"] != "omittable"]
    strict = {i for i, _ in m["pairs"] if i in counted}
    parts = set(m["parts"])
    on_omittable = [j for i, j in m["pairs"] if gts[i]["semantic_importance"] == "omittable"]
    tp_strict = len(strict)
    fp_idx = m["unmatched_pred"]
    samples = {j: pred_samples(preds[j]) for j in fp_idx}
    reasons = Counter(_fp_reason(preds[j], samples[j], gt, tau) for j in fp_idx)

    per_type: dict[str, dict] = {}
    for i in counted:
        t = gts[i]["type"]
        d = per_type.setdefault(t, {"gt": 0, "found_strict": 0, "found_structural": 0})
        d["gt"] += 1
        d["found_strict"] += i in strict
        d["found_structural"] += i in strict or i in parts
    pred_types: dict[str, dict] = {}
    matched_pred = {j: i for i, j in m["pairs"]}
    for j, p in enumerate(preds):
        d = pred_types.setdefault(p["type"], {"predicted": 0, "true_positive": 0})
        d["predicted"] += 1
        d["true_positive"] += j in matched_pred and matched_pred[j] in counted

    type_correct = type_total = 0
    confusion: Counter = Counter()
    geo = {"chamfer_px": [], "endpoint_px": [], "angle_deg": [], "length_rel": [], "polygon_iou": [],
           "vertex_count_match": [], "circle_center_px": [], "circle_radius_rel": []}
    for i, j in m["pairs"]:
        if i not in counted:
            continue
        o, p = gts[i], preds[j]
        expected = EXPECTED_TYPE.get(o["type"])
        if expected is not None:
            type_total += 1
            type_correct += p["type"] == expected
            confusion[f"{expected}->{p['type']}"] += 1
        geo["chamfer_px"].append(m["dist"][(i, j)])
        if (le := _linear_errors(o, p)):
            geo["endpoint_px"].append(le["endpoint_px"])
            geo["angle_deg"].append(le["angle_deg"])
            if le["length_rel"] is not None:
                geo["length_rel"].append(le["length_rel"])
        if (pe := _polygon_errors(o, p)):
            geo["polygon_iou"].append(pe["iou"])
            geo["vertex_count_match"].append(1.0 if pe["vertex_count_match"] else 0.0)
        if (ce := _round_errors(o, p)):
            geo["circle_center_px"].append(ce["center_px"])
            if ce["radius_rel"] is not None:
                geo["circle_radius_rel"].append(ce["radius_rel"])

    # Right-angle markers: Model A reports them as ANGLE elements with a flag.
    markers = [o for o in gt["objects"] if o["type"] in MARKER_GT]
    pred_markers = [p for p in all_preds if p["type"] == "angle" and (p.get("geometry") or {}).get("right_angle_marker")]
    found_markers = 0
    free = list(range(len(pred_markers)))
    for o in markers:
        v = o["geometry"]["vertex"]
        best = min(free, key=lambda k: math.dist(v, pred_markers[k]["geometry"]["vertex"]), default=None)
        if best is not None and math.dist(v, pred_markers[best]["geometry"]["vertex"]) <= tau * 1.5:
            found_markers += 1
            free.remove(best)

    essential = [i for i in counted if gts[i]["semantic_importance"] == "essential"]
    gt_id_to_preds: dict[str, list[int]] = {}
    for i, j in m["pairs"]:
        gt_id_to_preds.setdefault(gts[i]["id"], []).append(j)
    for i, js in m["parts"].items():
        gt_id_to_preds.setdefault(gts[i]["id"], []).extend(js)
    return {
        "tau_px": round(tau, 2),
        "gt_counted": len(counted),
        "gt_essential": len(essential),
        "predicted": len(preds),
        "strict": prf(tp_strict, len(fp_idx), len(counted) - tp_strict),
        "structural": prf(tp_strict + len(parts & set(counted)), len(fp_idx), len(counted) - tp_strict - len(parts & set(counted))),
        "essential_found_strict": sum(i in strict for i in essential),
        "essential_found_structural": sum(i in strict or i in parts for i in essential),
        "found_as_parts": len(parts),
        "fragmented_linear": sum(1 for i in parts if gts[i]["type"] in LINEAR_GT),
        "closed_as_sides": sum(1 for i in parts if gts[i]["type"] in CLOSED_GT),
        "on_omittable": len(on_omittable),
        "fp_reasons": dict(reasons),
        "per_gt_type": per_type,
        "per_pred_type": pred_types,
        "type_correct": type_correct,
        "type_total": type_total,
        "type_confusion": dict(confusion),
        "geometry_errors": geo,
        "right_angle_markers": {"gt": len(markers), "found": found_markers, "predicted": len(pred_markers)},
        "missed_essential_ids": [gts[i]["id"] for i in essential if i not in strict and i not in parts],
        "_gt_id_to_preds": {k: [preds[j]["id"] for j in v] for k, v in gt_id_to_preds.items()},
    }


# ------------------------------------------------------------------- points
def score_points(gt: dict, preds: list[dict]) -> dict:
    tau = tolerance(gt["width"], gt["height"])
    cands = [c for p in preds for c in vertex_candidates(p)]
    pts = [p for p in gt["points"] if p["semantic_importance"] == "essential"]
    errors, found = [], 0
    if cands:
        tree = cKDTree(np.asarray(cands))
        for p in pts:
            d, _ = tree.query((p["x"], p["y"]))
            if d <= tau:
                found += 1
                errors.append(float(d))
    return {"gt": len(pts), "found": found, "recall": ratio(found, len(pts)), "error_px": errors}


def _point_to_pred_ids(gt: dict, preds: list[dict], tau: float) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in gt["points"]:
        for el in preds:
            if any(math.dist((p["x"], p["y"]), c) <= tau for c in vertex_candidates(el)):
                out.setdefault(p["id"], set()).add(el["id"])
    return out


# ------------------------------------------------------------------- labels
def assign_labels(gt: dict, pred_labels: list[dict]) -> dict[str, list[dict]]:
    """Assign every OCR label to the nearest ground-truth label it sits on."""
    out: dict[str, list[dict]] = {lab["id"]: [] for lab in gt["labels"]}
    for pl in pred_labels:
        if not pl.get("bbox"):
            continue
        c = _center(pl["bbox"])
        best, best_d = None, float("inf")
        for lab in gt["labels"]:
            pad = max(6.0, 0.6 * lab["bbox"][3])
            if _point_in_box(c, lab["bbox"], pad):
                d = math.dist(c, _center(lab["bbox"]))
                if d < best_d:
                    best, best_d = lab["id"], d
        if best:
            out[best].append(pl)
    for v in out.values():
        v.sort(key=lambda pl: (round(pl["bbox"][1] / 20), pl["bbox"][0]))
    return out


def score_labels(gt: dict, pred_labels: list[dict], gt_braille: dict[str, str] | None = None,
                 translate=None, preds: list[dict] | None = None, gt_to_preds: dict | None = None) -> dict:
    assigned = assign_labels(gt, pred_labels)
    essential = [lab for lab in gt["labels"] if lab["semantic_importance"] == "essential"]
    tau = tolerance(gt["width"], gt["height"])
    char_err = char_tot = word_err = word_tot = 0
    char_err_det = char_tot_det = 0
    exact = exact_norm = detected = 0
    quantitative = quantitative_ok = 0
    assoc_total = assoc_ok = 0
    braille_total = braille_exact = braille_consistent = 0
    cell_err = cell_tot = 0
    order_pairs = order_ok = 0
    per_label = []
    point_map = _point_to_pred_ids(gt, preds or [], tau) if preds is not None else {}
    gt_to_preds = gt_to_preds or {}
    matched_for_order = []
    for lab in essential:
        got = assigned[lab["id"]]
        pred_text = " ".join(pl["text"] for pl in got)
        g, p = normalize_text(lab["text"]), normalize_text(pred_text)
        e = levenshtein(p, g)
        char_err += e
        char_tot += len(g)
        word_err += levenshtein(p.split(), g.split())
        word_tot += len(g.split())
        is_quant = any(ch.isdigit() for ch in lab["text"])
        quantitative += is_quant
        if got:
            detected += 1
            char_err_det += e
            char_tot_det += len(g)
            exact += pred_text.strip() == lab["text"].strip()
            exact_norm += p == g
            quantitative_ok += is_quant and p == g
            matched_for_order.append((lab, got[0]))
            target = lab.get("associated_object")
            if target and preds is not None:
                assoc_total += 1
                linked = {pl.get("associated_element_id") for pl in got} - {None}
                ok_ids = set(gt_to_preds.get(target, [])) | point_map.get(target, set())
                assoc_ok += bool(linked & ok_ids)
            if gt_braille is not None and len(got) == 1:
                braille_total += 1
                pb = got[0].get("braille") or ""
                gb = gt_braille.get(lab["id"], "")
                braille_exact += pb == gb
                if translate is not None:
                    braille_consistent += translate(got[0]["text"]) == pb
                cell_err += levenshtein(list(pb), list(gb))
                cell_tot += len(gb)
        per_label.append({"id": lab["id"], "gt": lab["text"], "pred": pred_text, "detected": bool(got), "char_errors": e})
    # Reading order: pairwise agreement between ground-truth order and Model A order.
    gt_rank = sorted(matched_for_order, key=lambda t: (round(t[0]["bbox"][1] / 15), t[0]["bbox"][0]))
    for a in range(len(gt_rank)):
        for b in range(a + 1, len(gt_rank)):
            ra, rb = gt_rank[a][1].get("reading_order"), gt_rank[b][1].get("reading_order")
            if ra is None or rb is None:
                continue
            order_pairs += 1
            order_ok += ra < rb
    all_assigned = {id(pl) for v in assigned.values() for pl in v}
    return {
        "gt_essential": len(essential),
        "predicted": len(pred_labels),
        "detected": detected,
        "detection_recall": ratio(detected, len(essential)),
        "detection_precision": ratio(sum(1 for pl in pred_labels if id(pl) in all_assigned), len(pred_labels)),
        "char_errors": char_err, "chars": char_tot,
        "cer": ratio(char_err, char_tot),
        "cer_detected_only": ratio(char_err_det, char_tot_det),
        "word_errors": word_err, "words": word_tot,
        "wer": ratio(word_err, word_tot),
        "exact_match": exact, "exact_match_normalized": exact_norm,
        "quantitative_labels": quantitative, "quantitative_preserved": quantitative_ok,
        "association_total": assoc_total, "association_correct": assoc_ok,
        "braille_total": braille_total, "braille_exact": braille_exact, "braille_consistent": braille_consistent,
        "braille_cell_errors": cell_err, "braille_cells": cell_tot,
        "reading_order_pairs": order_pairs, "reading_order_concordant": order_ok,
        "per_label": per_label,
    }


# ------------------------------------------------------------ relationships
def score_relationships(gt: dict, preds: list[dict], relationships: list[dict], gt_to_preds: dict) -> dict:
    tau = tolerance(gt["width"], gt["height"])
    point_map = _point_to_pred_ids(gt, preds, tau)
    pred_to_gt: dict[str, set[str]] = {}
    for gid, pids in gt_to_preds.items():
        for pid in pids:
            pred_to_gt.setdefault(pid, set()).add(gid)
    for gid, pids in point_map.items():
        for pid in pids:
            if any(p["id"] == pid and p["type"] == "point" for p in preds):
                pred_to_gt.setdefault(pid, set()).add(gid)
    supported = [r for r in gt["relationships"] if r["type"] in RELATION_MAP]
    unsupported = Counter(r["type"] for r in gt["relationships"] if r["type"] not in RELATION_MAP)
    pred_sets = []
    for r in relationships:
        gtype = PRED_TO_GT_RELATION.get(r["type"])
        if not gtype:
            continue
        ids = r["element_ids"][:2]
        if len(ids) < 2:
            continue
        a, b = pred_to_gt.get(ids[0], set()), pred_to_gt.get(ids[1], set())
        pred_sets.append((gtype, a, b))
    by_type: dict[str, dict] = {}
    tp = 0
    for r in supported:
        hit = any(t == r["type"] and ((r["source"] in a and r["target"] in b) or (r["source"] in b and r["target"] in a))
                  for t, a, b in pred_sets)
        tp += hit
        d = by_type.setdefault(r["type"], {"gt": 0, "found": 0, "predicted_mappable": 0})
        d["gt"] += 1
        d["found"] += hit
    gt_keys = {(r["type"], frozenset((r["source"], r["target"]))) for r in supported}
    mappable = [(t, a, b) for t, a, b in pred_sets if a and b]
    pred_tp = sum(1 for t, a, b in mappable if any((t, frozenset((x, y))) in gt_keys for x in a for y in b))
    for t, _, _ in mappable:
        by_type.setdefault(t, {"gt": 0, "found": 0, "predicted_mappable": 0})["predicted_mappable"] += 1
    return {
        "gt_supported": len(supported),
        "gt_unsupported": dict(unsupported),
        "found": tp,
        "recall": ratio(tp, len(supported)),
        "predicted_supported_type": len(pred_sets),
        "predicted_mappable": len(mappable),
        "predicted_mappable_correct": pred_tp,
        "precision_lower_bound": ratio(pred_tp, len(mappable)),
        "by_type": by_type,
    }


# ------------------------------------------------------------ SVG structure
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def score_svg(svg: str | None) -> dict:
    if not svg:
        return {"present": False, "valid": False}
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        return {"present": True, "valid": False, "error": str(error)}
    view = [float(v) for v in (root.get("viewBox") or "0 0 0 0").split()]
    vx, vy, vw, vh = view if len(view) == 4 else (0, 0, 0, 0)
    tags: Counter = Counter()
    seen: Counter = Counter()
    out_of_bounds = self_intersecting = open_paths = 0
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        tags[tag] += 1
        if tag not in ("line", "polyline", "polygon", "circle", "ellipse", "path"):
            continue
        attrs = tuple(sorted((k, v) for k, v in el.attrib.items() if k in ("points", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "d")))
        seen[(tag, attrs)] += 1
        xs, ys = [], []
        if tag == "line":
            xs = [float(el.get("x1", 0)), float(el.get("x2", 0))]
            ys = [float(el.get("y1", 0)), float(el.get("y2", 0))]
        elif tag in ("polyline", "polygon"):
            nums = [float(n) for n in _NUM.findall(el.get("points", ""))]
            xs, ys = nums[0::2], nums[1::2]
            if tag == "polygon" and len(xs) >= 4:
                if not Polygon(list(zip(xs, ys))).is_simple:
                    self_intersecting += 1
        elif tag in ("circle", "ellipse"):
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            rx = float(el.get("r", el.get("rx", 0)))
            ry = float(el.get("r", el.get("ry", 0)))
            xs, ys = [cx - rx, cx + rx], [cy - ry, cy + ry]
        elif tag == "path":
            d = el.get("d", "")
            open_paths += "z" not in d.lower()
        if vw and xs and (min(xs) < vx - 0.5 or max(xs) > vx + vw + 0.5 or min(ys) < vy - 0.5 or max(ys) > vy + vh + 0.5):
            out_of_bounds += 1
    return {
        "present": True, "valid": True,
        "element_counts": dict(tags),
        "out_of_bounds": out_of_bounds,
        "duplicates": sum(c - 1 for c in seen.values() if c > 1),
        "self_intersecting_polygons": self_intersecting,
        "open_paths": open_paths,
    }


def near_duplicate_geometry(elements: list[dict], px: float = 3.0) -> int:
    lines = [pred_samples(e) for e in elements if e["type"] in GEOMETRY_PRED]
    lines = [s for s in lines if s is not None and len(s) > 1]
    count = 0
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            if chamfer(lines[a], lines[b]) <= px:
                count += 1
    return count


# --------------------------------------------------------------- per image
def geometry_elements(elements: list[dict]) -> list[dict]:
    return [e for e in elements if e["type"] != "text_label"]


def label_elements(elements: list[dict]) -> list[dict]:
    return [e for e in elements if e["type"] == "text_label"]


def score_image(gt: dict, result: dict, gt_braille: dict | None = None, translate=None) -> dict:
    """Score a captured result (see ``evaluation.capture``) against an annotation."""
    if result.get("error"):
        essential = [o for o in gt["objects"] if o["semantic_importance"] == "essential"]
        return {"failed": True, "error": result["error"], "gt_essential": len(essential)}
    sem = result["semantic"]["elements"]
    fin = result["simplified"]["elements"]
    pre = score_objects(gt, geometry_elements(sem))
    post = score_objects(gt, geometry_elements(fin))
    labels = score_labels(gt, label_elements(fin), gt_braille, translate, geometry_elements(fin), post["_gt_id_to_preds"])
    rel_pre = score_relationships(gt, geometry_elements(sem), result["semantic"]["relationships"], pre["_gt_id_to_preds"])
    rel_post = score_relationships(gt, geometry_elements(fin), result["simplified"]["relationships"], post["_gt_id_to_preds"])
    actions = Counter(a["action"] for a in result["simplified"]["actions"])
    sem_ids = {e["id"] for e in sem}
    lost = sorted(set(pre["_gt_id_to_preds"]) - set(post["_gt_id_to_preds"]))
    essential_ids = {o["id"] for o in gt["objects"] if o["semantic_importance"] == "essential"}
    qa = result["qa"]
    issues = Counter(i["check"] for i in qa["issues"])
    embossed = [e for e in fin if e["type"] not in ("text_label", "arc")
                and (e["type"] != "angle" or (e.get("geometry") or {}).get("right_angle_marker"))]
    for part in (pre, post):
        part.pop("_gt_id_to_preds")
    return {
        "failed": False,
        "objects": pre,
        "objects_final": post,
        "points": score_points(gt, geometry_elements(sem)),
        "labels": labels,
        "relationships": rel_pre,
        "relationships_final": rel_post,
        "simplification": {
            "actions": dict(actions),
            "removed": sum(v for k, v in actions.items() if k.startswith("removed") or k == "dropped_unexplained"),
            "merged": actions.get("merged_collinear", 0),
            "modified": actions.get("simplified_contour", 0),
            "newly_inferred": sum(1 for e in fin if e["id"] not in sem_ids),
            "gt_found_before": pre["essential_found_structural"],
            "gt_found_after": post["essential_found_structural"],
            "essential_lost_ids": sorted(set(lost) & essential_ids),
        },
        "braille": {"collisions": issues.get("braille_collision", 0), "on_line": issues.get("braille_on_line", 0)},
        "qa": {"passes": qa["passes"], "score": qa.get("score_0_100"), "issues": dict(issues),
               "blocking": sorted({i["check"] for i in qa["issues"] if i["severity"] == "error"})},
        "svg": {**score_svg(result.get("tactile_svg")), "near_duplicate_geometry": near_duplicate_geometry(fin),
                "embossed_features": len(embossed), "over_target": len(embossed) > TACTILE_TARGET,
                "over_limit": len(embossed) > TACTILE_LIMIT},
        "timings_ms": result.get("timings_ms", {}),
    }
