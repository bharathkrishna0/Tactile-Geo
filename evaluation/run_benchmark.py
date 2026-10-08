"""Run TactileGeo on the math benchmark and score it.

    python -m evaluation.run_benchmark --split test --model-a
    python -m evaluation.run_benchmark --split test --model-a --model-b
    python -m evaluation.run_benchmark --split test --unet
    python -m evaluation.run_benchmark --split test --model-a --unet

Run from the repository root with the backend virtualenv's Python. Writes
``<output-dir>/<run-id>/`` with a manifest, one JSON per image (captured
pipeline output + scores) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import aggregate, metrics  # noqa: E402
from evaluation.capture import run_model_a  # noqa: E402

DATASET = ROOT / "evaluation" / "dataset"


class ImageTimeout(BaseException):
    """Raised by SIGALRM; a BaseException so pipeline ``except Exception`` blocks cannot swallow it."""


def _alarm(_signum, _frame):
    raise ImageTimeout()


def load_split(dataset: Path, split: str) -> list[str]:
    if split == "all":
        return sorted(sum((load_split(dataset, s) for s in ("train", "validation", "test")), []))
    return [line.strip() for line in (dataset / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]


def load_annotation(dataset: Path, image_id: str) -> dict:
    return json.loads((dataset / "annotations" / f"{image_id}.json").read_text())


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "backend/app"], cwd=ROOT) != 0
        return sha + ("+dirty-backend" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def variant_name(args) -> str:
    if args.unet and not args.model_a:
        return "unet"
    if args.unet:
        return "model_a+unet"
    if args.model_b:
        return "model_a+model_b"
    return "model_a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--model-a", action="store_true", help="Model A (deterministic CV) baseline")
    parser.add_argument("--model-b", action="store_true", help="add Model B + fusion + simulated teacher (needs key)")
    parser.add_argument("--unet", action="store_true", help="U-Net foreground mask (alone: replaces CV binarisation; "
                                                            "with --model-a: intersected with it)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="evaluate the image ids listed in this file instead of a split (e.g. a tuning subset)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--image-timeout", type=int, default=180,
                        help="seconds per image before it is scored as a timeout failure (0 = none)")
    parser.add_argument("--resume", action="store_true", help="reuse per-image results already in the run directory")
    parser.add_argument("--teacher", default=None, choices=["none", "accept_all", "oracle"],
                        help="simulated teacher for --model-b (default accept_all)")
    parser.add_argument("--unet-weights", type=Path, default=ROOT / "evaluation" / "unet" / "weights" / "unet_small.pt")
    args = parser.parse_args(argv)
    if not (args.model_a or args.unet):
        parser.error("choose --model-a and/or --unet")
    if args.model_b and not args.model_a:
        parser.error("--model-b is an addition to --model-a")

    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    from app.services.braille import LouisBrailleTranslator
    from app.services.ocr import EasyOcrProvider

    variant = variant_name(args)
    if args.ids_file:
        ids = [line.strip() for line in args.ids_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
        test_ids = set(load_split(args.dataset, "test"))
        if test_ids & set(ids):
            parser.error("--ids-file must not contain held-out test images")
        args.split = args.ids_file.stem
    else:
        ids = load_split(args.dataset, args.split)
    ids = ids[: args.limit]
    meta = json.loads((args.dataset / "metadata" / "dataset.json").read_text())
    run_id = args.run_id or f"{variant}-{args.split}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = args.output_dir / run_id
    (out / "per_image").mkdir(parents=True, exist_ok=True)

    ocr, translator = EasyOcrProvider(), LouisBrailleTranslator()
    overrides = None
    unet_info = None
    if args.unet:
        from evaluation.unet.infer import mask_override

        overrides, unet_info = mask_override(args.unet_weights, combine="intersect" if args.model_a else "replace")
    model_b_runner = None
    if args.model_b:
        from evaluation.model_b_eval import ModelBRunner

        model_b_runner = ModelBRunner(policy=args.teacher)

    manifest = {
        "run_id": run_id, "variant": variant, "split": args.split, "images": len(ids), "seed": args.seed,
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "dataset_version": meta["dataset_version"], "dataset_seed": meta["seed"],
        "model_a": {"name": "TactileGeo Model A", "entry": "app.services.pipeline.build_full_analysis",
                    "edge_sensitivity": 50, "ocr": "EasyOCR (en)", "braille": translator.table},
        "model_b": model_b_runner.describe() if model_b_runner else None,
        "unet": unet_info, "image_timeout_s": args.image_timeout,
        "matching_tolerance": "max(4 px, 1.5% of image diagonal), symmetric mean outline distance",
        "python": platform.python_version(), "platform": platform.platform(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # Warm OCR models once so the first image's latency is not a model download.
    warm = (args.dataset / load_annotation(args.dataset, ids[0])["file"]).read_bytes()
    run_model_a(warm, ocr, translator, overrides)

    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for k, image_id in enumerate(ids, 1):
        cached = out / "per_image" / f"{image_id}.json"
        if args.resume and cached.exists():
            rows.append(json.loads(cached.read_text()))
            continue
        gt = load_annotation(args.dataset, image_id)
        data = (args.dataset / gt["file"]).read_bytes()
        start = time.perf_counter()
        signal.alarm(args.image_timeout)
        try:
            captured, result = run_model_a(data, ocr, translator, overrides)
        except ImageTimeout:
            captured, result = {"error": f"ImageTimeout: Model A exceeded {args.image_timeout} s",
                                "timings_ms": {"total": round((time.perf_counter() - start) * 1000, 1)}}, None
        finally:
            signal.alarm(0)
        if model_b_runner is not None:
            captured = model_b_runner.augment(image_id, data, gt, captured, result, translator)
        gt_braille = {lab["id"]: translator.translate(lab["text"]) for lab in gt["labels"]
                      if lab["semantic_importance"] == "essential"}
        score = metrics.score_image(gt, captured, gt_braille, translator.translate)
        row = {"image_id": image_id, "difficulty": gt["difficulty"], "category": gt["category"],
               "layout": gt["layout"], "conditions": gt["conditions"], "content_stressors": gt["content_stressors"],
               "timings_ms": captured.get("timings_ms", {}), "score": score, "captured": captured}
        (out / "per_image" / f"{image_id}.json").write_text(json.dumps(row, indent=1, default=str) + "\n")
        rows.append(row)
        o = score.get("objects", {}).get("structural", {})
        print(f"[{k}/{len(ids)}] {image_id} {gt['difficulty']:<11} f1={o.get('f1')} "
              f"qa={score.get('qa', {}).get('passes')} {captured.get('timings_ms', {}).get('total')}ms"
              + (f" ERROR {score['error']}" if score.get("failed") else ""), file=sys.stderr)

    summary = {
        "manifest": manifest,
        "overall": aggregate.summarize(rows),
        "by_difficulty": aggregate.group(rows, lambda r: [r["difficulty"]]),
        "by_category": aggregate.group(rows, lambda r: [r["category"]]),
        "by_layout": aggregate.group(rows, lambda r: [r["layout"]]),
        "by_condition": aggregate.group(rows, lambda r: r["conditions"] + r["content_stressors"]),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    if model_b_runner is not None:
        summary["model_b"] = model_b_runner.summary(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
