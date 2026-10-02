"""Score Model A on the STEM benchmark and the reference images.

    python benchmarks/stem/generate.py          # once; images are not committed
    python scripts/benchmark.py --out benchmarks/stem/results/after.json
    python scripts/benchmark.py --backend-root ../baseline/backend --out .../before.json
    python scripts/benchmark.py --compare before.json after.json

``--backend-root`` imports the pipeline from another checkout, so the same
scorer measures an older commit and the current one identically.

Scoring. Predictions are Model A semantic elements, excluding derived
annotations (points, angles, text labels). A prediction matches a ground-truth
shape when both are in the same family (closed straight-sided, round, or
linear) and their bounding boxes, each padded by 2% of the image diagonal,
overlap with IoU >= 0.5. Padding keeps thin line boxes comparable. A
straight-sided shape whose every edge is matched by a separate line segment
also counts as found (reported as ``found_as_sides``) but not as type-correct.
Type accuracy is exact type agreement among found shapes.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BENCH = HERE.parents[1] / "benchmarks" / "stem"
FAMILY = {
    "triangle": "closed", "rectangle": "closed", "polygon": "closed",
    "circle": "round", "ellipse": "round", "arc": "round",
    "line_segment": "linear", "ray": "linear", "arrow": "linear", "axes": "linear",
}
IOU_THRESHOLD = 0.5


def _iou(a, b, pad: float) -> float:
    ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[0] + a[2] + pad, a[1] + a[3] + pad
    bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[0] + b[2] + pad, b[1] + b[3] + pad
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def _type(element) -> str:
    raw = element.type
    return str(getattr(raw, "value", raw))


def _match_outlines(shapes: list[dict], predictions: list[dict], used_gt: set, used_pred: set, pad: float) -> int:
    """Credit a straight-sided shape recovered as its individual sides.

    Model A represents a figure as a segment graph when it needs vertex angles
    (e.g. a right triangle with a marker). Every edge must be matched by its
    own linear prediction; the shape then counts as found but not type-correct.
    """
    found = 0
    for gi, shape in enumerate(shapes):
        if gi in used_gt or "points" not in shape:
            continue
        points = shape["points"]
        claimed: list[int] = []
        for a, b in zip(points, points[1:] + points[:1]):
            edge = [min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1])]
            best = max(
                (
                    (_iou(edge, pred["bbox"], pad), pi)
                    for pi, pred in enumerate(predictions)
                    if pi not in used_pred and pi not in claimed and FAMILY[pred["type"]] == "linear"
                ),
                default=(0.0, None),
            )
            if best[0] < IOU_THRESHOLD:
                break
            claimed.append(best[1])
        else:
            used_gt.add(gi)
            used_pred.update(claimed)
            found += 1
    return found


def score_case(case: dict, result) -> dict:
    semantic = result.semantic_geometry
    width, height = semantic.image_width, semantic.image_height
    pad = 0.02 * (width ** 2 + height ** 2) ** 0.5
    predictions = [
        {"type": _type(e), "bbox": list(e.bbox)}
        for e in semantic.elements
        if e.bbox and _type(e) in FAMILY
    ]
    pairs = sorted(
        (
            (_iou(gt["bbox"], pred["bbox"], pad), gi, pi)
            for gi, gt in enumerate(case["shapes"])
            for pi, pred in enumerate(predictions)
            if FAMILY[gt["type"]] == FAMILY[pred["type"]]
        ),
        reverse=True,
    )
    used_gt, used_pred, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < IOU_THRESHOLD or gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi); used_pred.add(pi); matches.append((gi, pi))
    outline_matches = _match_outlines(case["shapes"], predictions, used_gt, used_pred, pad)
    texts = {str(label.get("text", "")).strip() for label in result.labels}
    qa = result.qa_report
    tp = len(matches) + outline_matches
    return {
        "tp": tp,
        "fp": len(predictions) - len(used_pred),
        "fn": len(case["shapes"]) - tp,
        "outline_matches": outline_matches,
        "type_correct": sum(case["shapes"][g]["type"] == predictions[p]["type"] for g, p in matches),
        "labels_expected": len(case["labels"]),
        "labels_found": sum(1 for text in case["labels"] if text in texts),
        "qa_passes": bool(qa.passes),
        "blocking": sorted({i.check for i in qa.issues if i.severity == "error"}),
        "tactile_elements": len(result.simplified_geometry.elements),
        "has_svg": bool(result.tactile_svg),
        "predictions": predictions,
        "label_texts": sorted(texts),
    }


def failed_case(case: dict, failure: str) -> dict:
    return {
        "tp": 0, "fp": 0, "fn": len(case["shapes"]), "outline_matches": 0, "type_correct": 0,
        "labels_expected": len(case["labels"]), "labels_found": 0,
        "qa_passes": False, "blocking": [], "tactile_elements": 0, "has_svg": False,
        "error": failure,
    }


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def aggregate(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else 0.0
    times = sorted(r["ms"] for r in rows)
    return {
        "cases": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "type_accuracy": _ratio(sum(r["type_correct"] for r in rows), tp),
        "found_as_sides": sum(r["outline_matches"] for r in rows),
        "label_recall": _ratio(sum(r["labels_found"] for r in rows), sum(r["labels_expected"] for r in rows)),
        "qa_pass_rate": _ratio(sum(r["qa_passes"] for r in rows), len(rows)),
        "svg_rate": _ratio(sum(r["has_svg"] for r in rows), len(rows)),
        "error_rate": _ratio(sum("error" in r for r in rows), len(rows)),
        "median_ms": round(statistics.median(times)),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))]),
    }


def run(backend_root: Path, references: Path | None) -> dict:
    sys.path.insert(0, str(backend_root.resolve()))
    from app.services.pipeline import build_full_analysis

    truth = json.loads((BENCH / "ground_truth.json").read_text())
    cases = truth["cases"]
    if not (BENCH / cases[0]["file"]).exists():
        raise SystemExit("Benchmark images missing; run benchmarks/stem/generate.py first.")

    build_full_analysis((BENCH / cases[0]["file"]).read_bytes())  # warm OCR model; not timed

    rows = []
    for case in cases:
        data = (BENCH / case["file"]).read_bytes()
        start = time.perf_counter()
        try:
            result = build_full_analysis(data)
        except Exception as error:  # a failed analysis is a scored outcome, not a crash
            result, failure = None, f"{type(error).__name__}: {error}"
        elapsed = (time.perf_counter() - start) * 1000
        row = {"file": case["file"], "category": case["category"], "variant": case["variant"], "ms": round(elapsed)}
        if result is None:
            row.update(failed_case(case, failure))
        else:
            row.update(score_case(case, result))
        rows.append(row)
        print(f"{case['file']:<46} tp={row['tp']} fp={row['fp']} fn={row['fn']} qa={row['qa_passes']} {row['ms']}ms", file=sys.stderr)

    refs = []
    for path in sorted(references.iterdir()) if references else []:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        start = time.perf_counter()
        try:
            result = build_full_analysis(path.read_bytes())
        except Exception as error:
            refs.append({"file": path.name, "ms": round((time.perf_counter() - start) * 1000), "semantic_elements": 0,
                         "tactile_elements": 0, "qa_passes": False, "blocking": [f"error: {type(error).__name__}"]})
            continue
        qa = result.qa_report
        refs.append({
            "file": path.name,
            "ms": round((time.perf_counter() - start) * 1000),
            "semantic_elements": len(result.semantic_geometry.elements),
            "tactile_elements": len(result.simplified_geometry.elements),
            "qa_passes": bool(qa.passes),
            "blocking": sorted({i.check for i in qa.issues if i.severity == "error"}),
        })

    by = lambda key: {v: aggregate([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}  # noqa: E731
    return {
        "backend_root": str(backend_root.resolve()),
        "overall": aggregate(rows),
        "by_category": by("category"),
        "by_variant": by("variant"),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "references": refs,
        "cases": rows,
    }


def compare(before: dict, after: dict) -> str:
    keys = ["precision", "recall", "f1", "type_accuracy", "found_as_sides", "label_recall", "qa_pass_rate", "error_rate", "median_ms", "p95_ms"]
    lines = ["| Slice | Metric | Before | After |", "|---|---|---|---|"]
    slices = [("overall", before["overall"], after["overall"])]
    slices += [(f"variant: {k}", before["by_variant"][k], after["by_variant"][k]) for k in after["by_variant"]]
    slices += [(f"category: {k}", before["by_category"][k], after["by_category"][k]) for k in after["by_category"]]
    for name, b, a in slices:
        for key in keys if name == "overall" else ["f1", "qa_pass_rate", "error_rate"]:
            lines.append(f"| {name} | {key} | {b.get(key)} | {a.get(key)} |")
    lines.append(f"| overall | peak_rss_mb | {before['peak_rss_mb']} | {after['peak_rss_mb']} |")
    lines += ["", "| Reference | Before: tactile / QA | After: tactile / QA | After: blocking |", "|---|---|---|---|"]
    prior = {r["file"]: r for r in before["references"]}
    for ref in after["references"]:
        b = prior.get(ref["file"], {})
        lines.append(
            f"| {ref['file']} | {b.get('tactile_elements')} / {'pass' if b.get('qa_passes') else 'block'} "
            f"| {ref['tactile_elements']} / {'pass' if ref['qa_passes'] else 'block'} | {', '.join(ref['blocking']) or '-'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--references", type=Path, help="directory of real worksheet images (no ground truth)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        print(compare(*(json.loads(p.read_text()) for p in args.compare)))
        return 0
    report = run(args.backend_root, args.references)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps({"overall": report["overall"], "peak_rss_mb": report["peak_rss_mb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
