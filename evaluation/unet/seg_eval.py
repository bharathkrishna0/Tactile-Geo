"""Pixel-level comparison on one split: U-Net vs Model A's own binarisation.

    python -m evaluation.unet.seg_eval --split test

Both are scored against the exact geometry mask (text, grid lines and
handwriting are background). ``tolerant`` scores allow a 2 px boundary slack,
because Model A's binary image is deliberately eroded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.services.image_enhancement import enhance_copy  # noqa: E402
from app.services.image_preprocessing import preprocess_image  # noqa: E402
from evaluation.unet.data import load, split_ids  # noqa: E402
from evaluation.unet.infer import THRESHOLD, load_model, predict  # noqa: E402

K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    gt_d = cv2.dilate(gt.astype(np.uint8), K) > 0
    pr_d = cv2.dilate(pred.astype(np.uint8), K) > 0
    return {"tp": tp, "fp": fp, "fn": fn,
            "tol_prec_hits": int((pred & gt_d).sum()), "pred": int(pred.sum()),
            "tol_rec_hits": int((gt & pr_d).sum()), "gt": int(gt.sum())}


def scores(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    tp_ = c["tol_prec_hits"] / c["pred"] if c["pred"] else 0.0
    tr_ = c["tol_rec_hits"] / c["gt"] if c["gt"] else 0.0
    return {"dice": round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0, "iou": round(tp / (tp + fp + fn), 4) if tp else 0.0,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "tolerant_precision": round(tp_, 4), "tolerant_recall": round(tr_, 4),
            "tolerant_f1": round(2 * tp_ * tr_ / (tp_ + tr_), 4) if tp_ + tr_ else 0.0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--weights", type=Path, default=Path(__file__).parent / "weights" / "unet_small.pt")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results" / "unet_segmentation.json")
    args = p.parse_args(argv)
    model, meta = load_model(args.weights)
    totals = {"unet": {}, "model_a_binary": {}}
    per_image, ms = [], {"unet": [], "model_a_binary": []}
    for image_id in split_ids(args.split):
        bgr, gray, mask = load(image_id)
        gt = mask > 0
        t = time.perf_counter()
        u = predict(model, gray) >= THRESHOLD
        ms["unet"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        a = preprocess_image(enhance_copy(bgr)) > 0
        ms["model_a_binary"].append((time.perf_counter() - t) * 1000)
        row = {"image_id": image_id}
        for name, pred in (("unet", u), ("model_a_binary", a)):
            c = counts(pred, gt)
            for k, v in c.items():
                totals[name][k] = totals[name].get(k, 0) + v
            row[name] = scores(c)
        per_image.append(row)
    out = {"split": args.split, "threshold": THRESHOLD, "unet": meta,
           "pooled": {k: scores(v) for k, v in totals.items()},
           "latency_ms_median": {k: round(float(np.median(v)), 1) for k, v in ms.items()},
           "per_image": per_image}
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[args.split] = out
    args.out.write_text(json.dumps(existing, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("split", "pooled", "latency_ms_median")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
