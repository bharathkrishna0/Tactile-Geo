"""Rank the worst images of a run and attribute each to a root cause.

    python -m evaluation.failure_analysis model_a_v2-all [--model-b-run RUN] [--unet-run RUN]

Writes ``evaluation/reports/failure_analysis.json`` and prints a Markdown
table. Root causes are assigned by fixed rules over the captured stage outputs
(listed in ``RULES``), not by hand, so the same run always yields the same
analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .report import DATASET, RESULTS, ROOT

SPLITS = {s: set((DATASET / "splits" / f"{s}.txt").read_text().split()) for s in ("train", "validation", "test")}
IMAGE_NOISE = {"photographed", "scanned", "low_contrast", "shadow", "blur", "faded", "jpeg_heavy", "low_resolution",
               "colored_ink", "skew", "strong_rotation"}
NO_MODEL_A_TYPE = {"tick", "angle_marker", "bar", "arrow", "ray", "axis", "axes", "grid", "dimension_line",
                   "number_line", "point"}

FIX = {
    "timeout_noise_explosion": "denoise before binarising; cap contour count per region; process large photos at reduced scale",
    "binarisation_noise": "grain-aware enhancement; adaptive threshold tuned for photos; learned foreground mask",
    "small_features_dropped": "keep short strokes that touch a number line or axis; tick detector",
    "text_vectorised_as_geometry": "mask OCR text boxes before vectorisation",
    "closed_shape_split": "join sides into triangles/polygons when endpoints coincide",
    "ocr_misread_or_missed": "upscale label crops before OCR; restrict charset for single-letter labels",
    "label_association": "associate labels to nearest vertex/segment with a distance budget",
    "density_block": "split or crop worksheet into per-figure sheets",
    "type_vocabulary": "add ray/arrow/tick/axis types to Model A's vocabulary",
    "right_angle_marker_missed": "detect markers at T-junctions (foot of an altitude), not only where two edges end",
}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2}


def badness(score: dict) -> float:
    if score.get("failed"):
        return 10.0
    o = score["objects_final"]
    ess = o["essential_found_structural"] / o["gt_essential"] if o["gt_essential"] else 1.0
    lab = score["labels"]
    lab_r = lab["detected"] / lab["gt_essential"] if lab["gt_essential"] else 1.0
    fp = o["structural"]["fp"] / max(o["predicted"], 1)
    return round((1 - ess) * 3 + (1 - lab_r) + fp + (0 if score["qa"]["passes"] else 1), 3)


def causes(row: dict, gt: dict) -> list[str]:
    s = row["score"]
    if s.get("failed"):
        return ["timeout_noise_explosion" if "Timeout" in s.get("error", "") else "binarisation_noise"]
    out = []
    o = s["objects_final"]
    by_id = {obj["id"]: obj for obj in gt["objects"]}
    missed = Counter(by_id[i]["type"] for i in o["missed_essential_ids"] if i in by_id)
    if missed.get("tick", 0) or missed.get("angle_marker", 0):
        out.append("small_features_dropped")
    if o["fp_reasons"].get("on_text", 0) >= 2:
        out.append("text_vectorised_as_geometry")
    if o["closed_as_sides"] or o["found_as_parts"]:
        out.append("closed_shape_split")
    noisy = IMAGE_NOISE & set(row["conditions"])
    if noisy and (o["fp_reasons"].get("other", 0) >= 5 or o["structural"]["recall"] in (None, 0)):
        out.append("binarisation_noise")
    lab = s["labels"]
    if lab["gt_essential"] and ((lab["detection_recall"] or 0) < 0.6 or (lab["cer"] or 0) > 0.3):
        out.append("ocr_misread_or_missed")
    if lab["association_total"] and lab["association_correct"] / lab["association_total"] < 0.5:
        out.append("label_association")
    ra = o.get("right_angle_markers") or {}
    if ra.get("gt", 0) > ra.get("found", 0):
        out.append("right_angle_marker_missed")
    if not s["qa"]["passes"] and any("density" in b for b in s["qa"]["blocking"]):
        out.append("density_block")
    if any(obj["type"] in NO_MODEL_A_TYPE for obj in gt["objects"] if obj["semantic_importance"] == "essential") \
            and s["objects_final"]["type_total"] > s["objects_final"]["type_correct"]:
        out.append("type_vocabulary")
    return out or ["other"]


def severity(row: dict) -> str:
    s = row["score"]
    if s.get("failed") or not s["qa"]["passes"]:
        return "critical"
    o = s["objects_final"]
    ess = o["essential_found_structural"] / o["gt_essential"] if o["gt_essential"] else 1.0
    return "high" if ess < 0.5 else "medium"


def summarize_expected(gt: dict) -> str:
    ess = Counter(o["type"] for o in gt["objects"] if o["semantic_importance"] == "essential")
    labels = [lab["text"] for lab in gt["labels"] if lab["semantic_importance"] == "essential"]
    return f"{dict(ess)}; labels {labels[:8]}{'...' if len(labels) > 8 else ''}"


def summarize_actual(row: dict) -> str:
    s, cap = row["score"], row["captured"]
    if s.get("failed"):
        return s.get("error", "failed")
    types = Counter(e["type"] for e in cap["semantic"]["elements"] if e["type"] != "text_label")
    o = s["objects_final"]
    return (f"{dict(types)}; essential found {o['essential_found_structural']}/{o['gt_essential']}; "
            f"labels {s['labels']['detected']}/{s['labels']['gt_essential']} (CER {s['labels']['cer']}); "
            f"QA {'pass' if s['qa']['passes'] else 'blocked: ' + ', '.join(s['qa']['blocking'])}")


def compare(run: str | None, image_id: str, base: float) -> str:
    if run is None:
        return "NOT MEASURED"
    path = RESULTS / run / "per_image" / f"{image_id}.json"
    if not path.exists():
        return "NOT MEASURED (image not in that run)"
    row = json.loads(path.read_text())
    mb = row["captured"].get("model_b", {})
    if mb and mb.get("status") not in (None, "ok"):
        return f"no ({mb['status']})"
    new = badness(row["score"])
    return "yes" if new < base - 0.05 else ("worse" if new > base + 0.05 else "no change")


def unet_potential(row: dict, root: list[str]) -> str:
    if {"binarisation_noise", "timeout_noise_explosion", "text_vectorised_as_geometry"} & set(root):
        return "possibly: a foreground mask removes paper grain/text before vectorisation"
    return "unlikely: the error is downstream of segmentation"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--model-b-run")
    p.add_argument("--unet-run")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "reports" / "failure_analysis.json")
    args = p.parse_args(argv)
    rows = [json.loads(f.read_text()) for f in sorted((RESULTS / args.run / "per_image").glob("*.json"))]
    ranked = sorted(rows, key=lambda r: -badness(r["score"]))[: args.top]
    out = []
    for r in ranked:
        gt = json.loads((DATASET / "annotations" / f"{r['image_id']}.json").read_text())
        root = causes(r, gt)
        b = badness(r["score"])
        out.append({
            "image_id": r["image_id"], "source_image": f"evaluation/dataset/{gt['file']}",
            "split": next(s for s, ids in SPLITS.items() if r["image_id"] in ids),
            "difficulty": r["difficulty"], "category": r["category"],
            "conditions": r["conditions"] + r["content_stressors"], "badness": b,
            "expected": summarize_expected(gt), "actual": summarize_actual(r),
            "root_causes": root, "severity": severity(r), "possible_fix": [FIX.get(c, "investigate") for c in root],
            "model_b_helped": compare(args.model_b_run, r["image_id"], b),
            "unet_could_help": unet_potential(r, root),
            "unet_measured": compare(args.unet_run, r["image_id"], b),
        })
    out.sort(key=lambda x: (SEVERITY_ORDER[x["severity"]], -x["badness"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"run": args.run, "model_b_run": args.model_b_run, "unet_run": args.unet_run,
                                    "ranking": "badness = 3*(1-essential recall) + (1-label recall) + FP share "
                                               "+ 1 if QA blocks; failed images = 10",
                                    "root_cause_counts_all_images": dict(Counter(
                                        c for r in rows for c in causes(r, json.loads(
                                            (DATASET / "annotations" / f"{r['image_id']}.json").read_text())))),
                                    "failures": out}, indent=1) + "\n")
    print("| # | Image | Split | Difficulty | Category | Severity | Root causes | Model B helped | U-Net |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, x in enumerate(out, 1):
        print(f"| {k} | {x['image_id']} | {x['split']} | {x['difficulty']} | {x['category']} | {x['severity']} | "
              f"{', '.join(x['root_causes'])} | {x['model_b_helped']} | {x['unet_measured']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
