"""Run Model A on a real worksheet slice that has no exact ground truth.

    python -m evaluation.run_real --slice ncert

Only measurements that need no ground truth are reported: failures, latency,
QA outcome and blocking reasons, embossed feature density, simplification
drops, OCR labels found, and SVG structure. Accuracy against ground truth is
``NOT MEASURED`` until a teacher verifies annotations for the slice. A side-by-
side original/tactile sheet is written for visual inspection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from evaluation import metrics  # noqa: E402
from evaluation.capture import run_model_a  # noqa: E402
from evaluation.run_benchmark import ImageTimeout, _alarm, git_commit  # noqa: E402

REAL = ROOT / "evaluation" / "dataset_real"
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def render_svg(svg: str, width: int = 700) -> np.ndarray:
    """Rasterise the tactile SVG's primitives (line/polyline/polygon/circle) for inspection."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)
    vx, vy, vw, vh = (float(v) for v in root.get("viewBox").split())
    s = width / vw
    img = np.full((int(vh * s), width, 3), 255, np.uint8)

    def pt(x, y):
        return int((float(x) - vx) * s), int((float(y) - vy) * s)

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        braille = "braille" in (el.get("class") or "")
        if tag == "line":
            cv2.line(img, pt(el.get("x1"), el.get("y1")), pt(el.get("x2"), el.get("y2")), (0, 0, 0), 2, cv2.LINE_AA)
        elif tag in ("polyline", "polygon"):
            nums = [float(n) for n in _NUM.findall(el.get("points", ""))]
            pts = np.array([pt(x, y) for x, y in zip(nums[0::2], nums[1::2])], np.int32)
            cv2.polylines(img, [pts], tag == "polygon", (0, 0, 0), 2, cv2.LINE_AA)
        elif tag == "circle":
            r = float(el.get("r", 0)) * s
            filled = braille or r < 3
            cv2.circle(img, pt(el.get("cx"), el.get("cy")), max(1, int(r)), (180, 0, 0) if filled else (0, 0, 0),
                       -1 if filled else 2, cv2.LINE_AA)
    return img


def side_by_side(original: np.ndarray, tactile: np.ndarray | None, title: str) -> np.ndarray:
    h = 520
    o = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    t = (cv2.resize(tactile, (int(tactile.shape[1] * h / tactile.shape[0]), h)) if tactile is not None
         else np.full((h, int(h * 1.41), 3), 230, np.uint8))
    sheet = np.hstack([o, np.full((h, 10, 3), 255, np.uint8), t])
    bar = np.full((34, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    return np.vstack([bar, sheet])


def describe(captured: dict) -> dict:
    if captured.get("error"):
        return {"failed": True, "error": captured["error"]}
    qa = captured["qa"]
    issues = Counter(i["check"] for i in qa["issues"])
    simp = captured["simplified"]
    svg = metrics.score_svg(captured.get("tactile_svg"))
    embossed = sum(1 for e in simp["elements"] if e["type"] != "text_label")
    labels = captured["ocr_labels"]
    return {
        "failed": False,
        "qa_passes": qa["passes"], "qa_score": qa["score_0_100"],
        "qa_blocking": sorted({i["check"] for i in qa["issues"] if i["severity"] == "error"}),
        "qa_issue_counts": dict(issues),
        "semantic_elements": len(captured["semantic"]["elements"]),
        "embossed_features": embossed, "over_target_40": embossed > 40, "over_limit_60": embossed > 60,
        "simplification_removed": simp["removed_count"], "simplification_merged": simp["merged_count"],
        "labels_found": len(labels),
        "labels": [lab["text"] for lab in labels][:40],
        "svg_valid": svg.get("valid", False), "svg_duplicates": svg.get("duplicates"),
        "svg_out_of_bounds": svg.get("out_of_bounds"),
        "accuracy_vs_ground_truth": "NOT MEASURED (no verified annotation)",
    }


def summarize(slice_name: str, rows: list[dict]) -> dict:
    ok = [r["measures"] for r in rows if not r["measures"]["failed"]]
    totals = [r["timings_ms"].get("total") for r in rows if r["timings_ms"].get("total") is not None]
    blocking = Counter(b for m in ok for b in m["qa_blocking"])
    return {
        "slice": slice_name, "images": len(rows), "git_commit": git_commit(),
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "failed": len(rows) - len(ok),
        "qa_pass_rate": round(sum(m["qa_passes"] for m in ok) / len(rows), 3) if rows else None,
        "qa_blocking_reasons": dict(blocking.most_common()),
        "embossed_features": metrics.percentiles([m["embossed_features"] for m in ok]),
        "over_target_40": sum(m["over_target_40"] for m in ok), "over_limit_60": sum(m["over_limit_60"] for m in ok),
        "labels_found": metrics.percentiles([m["labels_found"] for m in ok]),
        "svg_valid": sum(m["svg_valid"] for m in ok),
        "latency_ms": metrics.percentiles(totals),
        "accuracy_vs_ground_truth": "NOT MEASURED (no verified annotations)",
        "per_image": [{"image_id": r["image_id"], **{k: v for k, v in r["measures"].items() if k != "labels"},
                       "total_ms": r["timings_ms"].get("total")} for r in rows],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slice", default="ncert")
    p.add_argument("--image-timeout", type=int, default=180)
    p.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    p.add_argument("--summarize-only", action="store_true",
                   help="rebuild summary.json from the per-image results already on disk")
    args = p.parse_args(argv)
    out = args.output_dir / f"real_{args.slice}"
    if args.summarize_only:
        rows = [json.loads(f.read_text()) for f in sorted((out / "per_image").glob("*.json"))]
        (out / "summary.json").write_text(json.dumps(summarize(args.slice, rows), indent=1) + "\n")
        print(out)
        return 0
    from app.services.braille import LouisBrailleTranslator
    from app.services.ocr import EasyOcrProvider

    src = REAL / args.slice
    files = sorted((src / "images").glob("*.png")) + sorted((src / "images").glob("*.jpg"))
    (out / "per_image").mkdir(parents=True, exist_ok=True)
    examples = ROOT / "evaluation" / "reports" / "examples" / f"real_{args.slice}"
    examples.mkdir(parents=True, exist_ok=True)
    ocr, translator = EasyOcrProvider(), LouisBrailleTranslator()
    run_model_a(files[0].read_bytes(), ocr, translator)
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for k, f in enumerate(files, 1):
        data = f.read_bytes()
        start = time.perf_counter()
        signal.alarm(args.image_timeout)
        try:
            captured, _ = run_model_a(data, ocr, translator)
        except ImageTimeout:
            captured = {"error": f"ImageTimeout: Model A exceeded {args.image_timeout} s",
                        "timings_ms": {"total": round((time.perf_counter() - start) * 1000, 1)}}
        finally:
            signal.alarm(0)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        row = {"image_id": f.stem, "sha256": hashlib.sha256(data).hexdigest(), "size": [bgr.shape[1], bgr.shape[0]],
               "timings_ms": captured.get("timings_ms", {}), "measures": describe(captured), "captured": captured}
        (out / "per_image" / f"{f.stem}.json").write_text(json.dumps(row, indent=1, default=str) + "\n")
        rows.append(row)
        m = row["measures"]
        tactile = render_svg(captured["tactile_svg"]) if captured.get("tactile_svg") else None
        verdict = "FAILED" if m["failed"] else f"QA {'pass' if m['qa_passes'] else 'BLOCKED'} | {m['embossed_features']} features"
        cv2.imwrite(str(examples / f"{f.stem}.jpg"), side_by_side(bgr, tactile, f"{f.stem}: {verdict}"),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"[{k}/{len(files)}] {f.stem} {verdict} {captured['timings_ms'].get('total')}ms", file=sys.stderr)

    (out / "summary.json").write_text(json.dumps(summarize(args.slice, rows), indent=1) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
