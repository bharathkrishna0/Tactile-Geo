"""Score Model A on multi-figure worksheet pages, whole page vs one figure at a time.

    OMP_NUM_THREADS=2 PYTHONPATH=backend:. backend/.venv/bin/python -m evaluation.run_pages \\
        --split dev --mode figures --run-id pages-figures-dev

``--mode page`` runs each page as one drawing (the behaviour before figure
splitting) and scores it against every figure's ground truth at once.
``--mode figures`` splits the page with ``find_figures``, matches each found
figure to a ground-truth figure by box overlap, runs Model A on each matched
figure, and scores it against that figure's ground truth only. A ground-truth
figure with no matching found figure is scored as all of its geometry missed.
Both modes count the same ground-truth objects, labels and relationships, so
the pooled scores compare directly. Page furniture is annotated as omittable,
so tracing it lowers precision.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "backend", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from evaluation import aggregate, metrics  # noqa: E402
from evaluation.capture import run_model_a  # noqa: E402
from evaluation.run_benchmark import git_commit, load_annotation, load_split  # noqa: E402

DATASET = ROOT / "evaluation" / "dataset_pages"
MATCH_IOU = 0.3


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_figures(gt_figures: list[dict], found: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by box IoU; returns (gt index, found index, IoU)."""
    gt_boxes = [(f["bbox"][0], f["bbox"][1], f["bbox"][0] + f["bbox"][2], f["bbox"][1] + f["bbox"][3]) for f in gt_figures]
    found_boxes = [(f["x"], f["y"], f["x"] + f["width"], f["y"] + f["height"]) for f in found]
    pairs = sorted(((_iou(g, p), i, j) for i, g in enumerate(gt_boxes) for j, p in enumerate(found_boxes)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < MATCH_IOU or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, round(iou, 3)))
    return sorted(out)


def _shift(value, dx: float, dy: float):
    return [value[0] - dx, value[1] - dy]


def figure_annotation(gt: dict, figure: dict, dx: float = 0.0, dy: float = 0.0) -> dict:
    """The page annotation restricted to one figure, moved by (-dx, -dy).

    Width and height stay the page's, so the matching tolerance is the same as in page mode.
    """
    keep = set(figure["object_ids"]) | set(figure["point_ids"]) | set(figure["label_ids"])
    sub = copy.deepcopy({k: v for k, v in gt.items() if k not in ("figures",)})
    sub["objects"] = [o for o in sub["objects"] if o["id"] in keep]
    sub["points"] = [p for p in sub["points"] if p["id"] in keep]
    sub["labels"] = [lab for lab in sub["labels"] if lab["id"] in keep]
    sub["relationships"] = [r for r in sub["relationships"] if r["source"] in keep and r["target"] in keep]
    sub["dimensions"] = [d for d in sub["dimensions"] if d.get("object") in keep]
    sub["omittable_ids"] = [i for i in sub.get("omittable_ids", []) if i in keep]
    for obj in sub["objects"]:
        obj["bbox"][:2] = _shift(obj["bbox"], dx, dy)
        geom = obj["geometry"]
        if "points" in geom:
            geom["points"] = [_shift(p, dx, dy) for p in geom["points"]]
        for key in ("center", "vertex"):
            if geom.get(key) is not None:
                geom[key] = _shift(geom[key], dx, dy)
    for point in sub["points"]:
        point["x"], point["y"] = point["x"] - dx, point["y"] - dy
    for label in sub["labels"]:
        label["bbox"][:2] = _shift(label["bbox"], dx, dy)
    return sub


EMPTY_CAPTURE = {
    "semantic": {"elements": [], "relationships": []},
    "simplified": {"elements": [], "relationships": [], "actions": []},
    "ocr_labels": [], "qa": {"passes": False, "issues": []}, "tactile_svg": None, "timings_ms": {},
}


def _score(gt: dict, captured: dict, translator) -> dict:
    gt_braille = {lab["id"]: translator.translate(lab["text"]) for lab in gt["labels"]
                  if lab["semantic_importance"] == "essential"}
    return metrics.score_image(gt, captured, gt_braille, translator.translate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mode", choices=("page", "figures"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from app.services.braille import LouisBrailleTranslator
    from app.services.figure_split import find_figures
    from app.services.ocr import EasyOcrProvider

    ids = load_split(args.dataset, args.split)[: args.limit]
    out = args.output_dir / args.run_id
    (out / "per_image").mkdir(parents=True, exist_ok=True)
    ocr, translator = EasyOcrProvider(), LouisBrailleTranslator()
    rows, pages = [], []
    for k, image_id in enumerate(ids, 1):
        gt = load_annotation(args.dataset, image_id)
        data = (args.dataset / gt["file"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        found = [f.as_dict(n) for n, f in enumerate(find_figures(image), 1)]
        pairs = match_figures(gt["figures"], found)
        page = {"image_id": image_id, "gt_figures": len(gt["figures"]), "found_figures": len(found),
                "matched": len(pairs), "pairs": pairs, "figure_qa": []}
        start = time.perf_counter()
        if args.mode == "page":
            captured, _ = run_model_a(data, ocr, translator)
            score = _score(gt, captured, translator)
            rows.append({"image_id": image_id, "score": score, "captured": captured})
            page["qa_passes"] = score.get("qa", {}).get("passes")
        else:
            by_gt = {i: j for i, j, _ in pairs}
            for i, fig in enumerate(gt["figures"]):
                if i in by_gt:
                    box = found[by_gt[i]]
                    captured, _ = run_model_a(data, ocr, translator, figure=by_gt[i])
                    sub = figure_annotation(gt, fig, box["x"], box["y"])
                else:
                    captured, sub = EMPTY_CAPTURE, figure_annotation(gt, fig)
                score = _score(sub, captured, translator)
                rows.append({"image_id": f"{image_id}/{fig['id']}", "score": score, "captured": captured,
                             "figure_missed": i not in by_gt})
                page["figure_qa"].append(bool(i in by_gt and score.get("qa", {}).get("passes")))
            page["qa_passes"] = all(page["figure_qa"])
        page["seconds"] = round(time.perf_counter() - start, 1)
        pages.append(page)
        (out / "per_image" / f"{image_id}.json").write_text(json.dumps(
            {"page": page, "rows": [r for r in rows if r["image_id"].split("/")[0] == image_id]},
            indent=1, default=str) + "\n")
        print(f"[{k}/{len(ids)}] {image_id} found={len(found)}/{len(gt['figures'])} matched={len(pairs)} "
              f"qa={page['qa_passes']} {page['seconds']}s", file=sys.stderr)

    gt_total = sum(p["gt_figures"] for p in pages)
    found_total = sum(p["found_figures"] for p in pages)
    matched = sum(p["matched"] for p in pages)
    figure_qa = [q for p in pages for q in p["figure_qa"]]
    summary = {
        "manifest": {"run_id": args.run_id, "mode": args.mode, "split": args.split, "pages": len(ids),
                     "git_commit": git_commit(), "match_iou": MATCH_IOU},
        "segmentation": {"gt_figures": gt_total, "found_figures": found_total, "matched": matched,
                         "precision": aggregate.ratio(matched, found_total),
                         "recall": aggregate.ratio(matched, gt_total),
                         "mean_iou": round(float(np.mean([iou for p in pages for *_, iou in p["pairs"]] or [0])), 3)},
        "qa": {"pages_all_exportable": aggregate.ratio(sum(1 for p in pages if p["qa_passes"]), len(pages)),
               "figures_exportable": aggregate.ratio(sum(figure_qa), len(figure_qa)) if figure_qa else None},
        "overall": aggregate.summarize(rows),
        "pages": pages,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
