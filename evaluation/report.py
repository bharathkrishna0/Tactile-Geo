"""Side-by-side Markdown tables from benchmark runs, recomputed on any split.

    python -m evaluation.report --split test model_a-all model_a_v2-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from . import metrics  # noqa: E402
from .aggregate import group, summarize  # noqa: E402
RESULTS = ROOT / "evaluation" / "results"
DATASET = ROOT / "evaluation" / "dataset"

KEY_METRICS = [
    ("Images failed (error/timeout)", ("failed",)),
    ("Object precision (strict)", ("objects_semantic", "strict", "precision")),
    ("Object recall (strict)", ("objects_semantic", "strict", "recall")),
    ("Object F1 (strict)", ("objects_semantic", "strict", "f1")),
    ("Object F1 (structural: parts allowed)", ("objects_semantic", "structural", "f1")),
    ("Essential object recall, final tactile", ("final_preservation", "essential_elements")),
    ("Object type accuracy (matched)", ("classification", "object_type_accuracy")),
    ("Line endpoint error px, P50", ("geometry", "endpoint_px", "p50")),
    ("Line endpoint error px, P95", ("geometry", "endpoint_px", "p95")),
    ("Angle error deg, P50", ("geometry", "angle_deg", "p50")),
    ("Polygon IoU, P50", ("geometry", "polygon_iou", "p50")),
    ("Circle centre error px, P50", ("geometry", "circle_center_px", "p50")),
    ("Point recall", ("geometry", "point_recall")),
    ("Right-angle marker recall", ("right_angle_markers", "recall")),
    ("Angle-arc marker recall", ("angle_markers", "recall")),
    ("Label detection recall", ("ocr", "label_detection_recall")),
    ("OCR CER (all essential labels; missed label = all chars wrong)", ("ocr", "cer")),
    ("OCR CER (detected labels only)", ("ocr", "cer_detected_only")),
    ("OCR WER (all essential labels)", ("ocr", "wer")),
    ("Label association accuracy", ("ocr", "label_association_accuracy")),
    ("Braille end-to-end exact", ("braille", "end_to_end_exact")),
    ("Braille cell error rate", ("braille", "cell_error_rate")),
    ("Braille reading-order concordance", ("braille", "reading_order_concordance")),
    ("Relationship recall (supported, final)", ("relationships_final", "recall")),
    ("Relationship precision (mappable predictions)", ("relationships_final", "precision_lower_bound")),
    ("Relationship F1", ("relationships_final", "f1_lower_bound")),
    ("Edges represented once (correct grouping)", ("line_grouping", "correct_grouping_rate")),
    ("Edge over-segmentation rate", ("line_grouping", "over_segmentation_rate")),
    ("Edge duplicate rate", ("line_grouping", "duplicate_rate")),
    ("Edges missed", ("line_grouping", "missed_rate")),
    ("Semantic Preservation Rate (end to end)", ("semantic_preservation", "rate")),
    ("Simplification semantic preservation", ("simplification", "semantic_preservation_rate")),
    ("Essential elements removed by simplification", ("simplification", "essential_removed_by_simplification")),
    ("Quantitative labels exact", ("final_preservation", "quantitative_labels_exact")),
    ("Tactile QA pass rate", ("tactile_qa", "pass_rate_all_images")),
    ("Critical problems exported (not blocked)", ("qa_gate", "critical_unblocked_images")),
    ("Images over 60 features", ("svg", "images_over_60_features")),
    ("Total latency ms, P50", ("latency_ms", "total", "p50")),
    ("Total latency ms, P95", ("latency_ms", "total", "p95")),
]

BY_DIFFICULTY = [
    ("Obj F1 strict", ("objects_semantic", "strict", "f1")),
    ("Obj F1 struct", ("objects_semantic", "structural", "f1")),
    ("Essential final", ("final_preservation", "essential_elements")),
    ("Label recall", ("ocr", "label_detection_recall")),
    ("CER", ("ocr", "cer")),
    ("Rel recall", ("relationships_final", "recall")),
    ("QA pass", ("tactile_qa", "pass_rate_all_images")),
    ("Failed", ("failed",)),
]


def get(summary: dict, path: tuple) -> object:
    v: object = summary
    for p in path:
        if not isinstance(v, dict) or p not in v:
            return None
        v = v[p]
    return v


def fmt(v: object) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


ITERATIONS = ROOT / "evaluation" / "iterations"


def split_ids(split: str) -> set[str] | None:
    if split == "all":
        return None
    path = DATASET / "splits" / f"{split}.txt"
    if not path.exists():
        path = ITERATIONS / f"{split}.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


_translator = None


def rescore(row: dict) -> dict:
    """Recompute a row's score from its captured output with the current metrics engine."""
    global _translator
    if _translator is None:
        from app.services.braille import LouisBrailleTranslator

        _translator = LouisBrailleTranslator()
    gt = json.loads((DATASET / "annotations" / f"{row['image_id']}.json").read_text())
    gt_braille = {lab["id"]: _translator.translate(lab["text"]) for lab in gt["labels"]
                  if lab["semantic_importance"] == "essential"}
    return {**row, "score": metrics.score_image(gt, row["captured"], gt_braille, _translator.translate)}


def load_rows(run: str, split: str, fresh: bool = False) -> list[dict]:
    ids = split_ids(split)
    rows = [json.loads(p.read_text()) for p in sorted((RESULTS / run / "per_image").glob("*.json"))]
    rows = [r for r in rows if ids is None or r["image_id"] in ids]
    return [rescore(r) for r in rows] if fresh else rows


def comparison_table(runs: list[str], split: str, metrics=KEY_METRICS, fresh: bool = False) -> str:
    sums = {run: summarize(load_rows(run, split, fresh)) for run in runs}
    lines = [f"| Metric ({split}, n={sums[runs[0]]['images']}) | " + " | ".join(runs) + " |",
             "|---|" + "---|" * len(runs)]
    for label, path in metrics:
        lines.append(f"| {label} | " + " | ".join(fmt(get(sums[r], path)) for r in runs) + " |")
    return "\n".join(lines)


def difficulty_table(run: str, split: str, fresh: bool = False) -> str:
    groups = group(load_rows(run, split, fresh), lambda r: [r["difficulty"]])
    order = [d for d in ("easy", "moderate", "hard", "very_hard", "adversarial") if d in groups]
    lines = ["| Difficulty | n | " + " | ".join(lbl for lbl, _ in BY_DIFFICULTY) + " |",
             "|---|---|" + "---|" * len(BY_DIFFICULTY)]
    for d in order:
        s = groups[d]
        lines.append(f"| {d} | {s['images']} | " + " | ".join(fmt(get(s, p)) for _, p in BY_DIFFICULTY) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--by-difficulty", action="store_true")
    parser.add_argument("--rescore", action="store_true",
                        help="recompute scores from captured outputs with the current metrics engine")
    args = parser.parse_args(argv)
    print(comparison_table(args.runs, args.split, fresh=args.rescore))
    if args.by_difficulty:
        for run in args.runs:
            print(f"\n**{run}**\n")
            print(difficulty_table(run, args.split, args.rescore))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
