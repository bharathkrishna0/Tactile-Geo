"""Build multi-figure worksheet pages with a per-figure answer key.

    python -m evaluation.generator.pages [--out evaluation/dataset_pages] [--seed 20261101]

Each page carries several numbered figures plus the page furniture of a real
worksheet: a title, a solid logo, Name/Date boxes, a dotted border, a pale
watermark and an empty answer box beside every figure. The annotation uses the
math benchmark schema and adds ``figures``: for each figure its box and the
ids of the objects, points and labels drawn for it, so a pipeline that splits
the page can be scored figure by figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import degrade
from .build import SCHEMA_VERSION
from .figures import TEMPLATES
from .scene import Scene, font

DATASET_VERSION = "pages-v1"
DEFAULT_SEED = 20261101
ROOT = Path(__file__).resolve().parents[1]
PAGE_SIZE = (1100, 1556)
PAGE_COUNT = 30
CATEGORIES = ["G_angles", "A_geometry_figure", "F_shapes_polygons", "H_circles", "E_measurement", "G_angles"]
CONDITIONS = [["clean_digital"], ["scanned"], ["photographed"]]
FIGURE_COUNTS = [4, 6, 8, 10]
MAX_FIGURE_ATTEMPTS = 6


def _dotted_border(s: Scene, inset: int, step: int = 14, radius: int = 2) -> None:
    w, h = s.width, s.height
    corners = [(inset, inset), (w - inset, inset), (w - inset, h - inset), (inset, h - inset)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        n = int(math.hypot(bx - ax, by - ay) // step)
        for k in range(n):
            x, y = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            s.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=40)
    s.distractor("decoration", corners + [corners[0]], "dotted page border")


def _header(s: Scene, rng, number: int) -> None:
    w = s.width
    s.text((70, 52), f"Lines and Angles - Worksheet {number}", size=34, role="title", face="sans_bold")
    lx0, ly0, lx1, ly1 = w - 250, 40, w - 70, 100
    s.draw.rounded_rectangle([lx0, ly0, lx1, ly1], radius=14, fill=50)
    s.draw.text(((lx0 + lx1) / 2, (ly0 + ly1) / 2), "MathsKit", font=font("sans_bold", 28), fill=255, anchor="mm")
    s.distractor("decoration", [(lx0, ly0), (lx1, ly0), (lx1, ly1), (lx0, ly1), (lx0, ly0)], "logo")
    for x0, caption in ((70, "Name:"), (620, "Date:")):
        s.text((x0, 125), caption, size=22, role="name_line")
        box = [(x0 + 80, 118), (x0 + (420 if x0 == 70 else 330), 118),
               (x0 + (420 if x0 == 70 else 330), 158), (x0 + 80, 158)]
        s.draw.polygon(box, outline=0, width=2)
        s.distractor("decoration", box + [box[0]], "name/date box")


def _watermark(s: Scene, rng) -> None:
    text = str(rng.choice(["MathsKit", "SAMPLE", "Practice"]))
    size = int(rng.integers(130, 170))
    layer = np.full((s.height, s.width), 255, np.uint8)
    from PIL import Image, ImageDraw
    img = Image.fromarray(layer)
    ImageDraw.Draw(img).text((s.width / 2, s.height / 2), text, font=font("sans_bold", size), fill=0, anchor="mm")
    angle = float(rng.uniform(20, 40))
    img = img.rotate(angle, fillcolor=255, resample=Image.BICUBIC)
    shade = int(rng.integers(212, 232))
    ink = np.asarray(img) < 128
    page = np.array(s.page)
    page[ink] = np.minimum(page[ink], shade)
    s.page.paste(Image.fromarray(page))
    ys, xs = np.nonzero(ink)
    s.distractor("decoration", [(xs.min(), ys.min()), (xs.max(), ys.min()), (xs.max(), ys.max()), (xs.min(), ys.max())],
                 "pale watermark")


def _draw_inside(s: Scene, rng, level: int, fs: int, cell: tuple, allowed: tuple) -> str:
    """Draw one template figure that stays inside its cell, retrying when it overflows."""
    x0, y0, cell_w, cell_h = cell
    page, mask = s.page.copy(), s.mask.copy()
    sizes = {name: len(getattr(s, name)) for name in ("objects", "points", "labels", "relationships", "dimensions")}
    for attempt in range(MAX_FIGURE_ATTEMPTS):
        cat = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        scale = 1.0 - 0.15 * attempt
        fig_w = min(cell_w - 150, (cell_h - 72) * 1.25) * scale
        fig_h = (cell_h - 72) * scale
        box = (x0 + 40, y0 + 36, x0 + 40 + fig_w, y0 + 36 + fig_h)
        TEMPLATES[cat](s, box, rng, level, fs)
        ys, xs = np.nonzero(np.asarray(s.mask) != np.asarray(mask))
        if xs.size == 0 or (xs.min() >= allowed[0] and ys.min() >= allowed[1]
                            and xs.max() <= allowed[2] and ys.max() <= allowed[3]):
            return cat
        if attempt == MAX_FIGURE_ATTEMPTS - 1:
            return cat
        s.page.paste(page)
        s.mask.paste(mask)
        for name, size in sizes.items():
            del getattr(s, name)[size:]
    return cat


def render_page(rng, number: int, n_figures: int, level: int) -> tuple[Scene, list[dict]]:
    w, h = PAGE_SIZE
    s = Scene(w, h, stroke=3)
    _watermark(s, rng)
    _dotted_border(s, 24)
    _header(s, rng, number)
    cols = 2
    rows = math.ceil(n_figures / cols)
    top, bottom, side = 185, 60, 60
    cell_w, cell_h = (w - 2 * side) / cols, (h - top - bottom) / rows
    fs = 20 if rows >= 4 else 24
    figures = []
    for i in range(n_figures):
        col, row = i % cols, i // cols
        x0, y0 = side + col * cell_w, top + row * cell_h
        s.text((x0 + 8, y0 + 6), f"{i + 1}.", size=fs, role="question_text", face="serif")
        answer = [(x0 + cell_w - 120, y0 + cell_h - 46), (x0 + cell_w - 20, y0 + cell_h - 46),
                  (x0 + cell_w - 20, y0 + cell_h - 12), (x0 + cell_w - 120, y0 + cell_h - 12)]
        s.draw.polygon(answer, outline=0, width=2)
        s.distractor("decoration", answer + [answer[0]], "answer box")
        start = (len(s.objects), len(s.points), len(s.labels))
        allowed = (x0 + 4, y0 + 28, x0 + cell_w - 130, y0 + cell_h - 8)
        cat = _draw_inside(s, rng, level, fs, (x0, y0, cell_w, cell_h), allowed)
        figures.append({"id": f"figure_{i + 1}", "category": cat, "number": i + 1,
                        "object_ids": [o["id"] for o in s.objects[start[0]:]],
                        "point_ids": [p["id"] for p in s.points[start[1]:]],
                        "label_ids": [lab["id"] for lab in s.labels[start[2]:]]})
    return s, figures


def _figure_boxes(gt: dict, figures: list[dict]) -> None:
    objects = {o["id"]: o for o in gt["objects"]}
    labels = {lab["id"]: lab for lab in gt["labels"]}
    points = {p["id"]: p for p in gt["points"]}
    for fig in figures:
        boxes = [objects[i]["bbox"] for i in fig["object_ids"]] + [labels[i]["bbox"] for i in fig["label_ids"]]
        boxes += [[points[i]["x"], points[i]["y"], 0, 0] for i in fig["point_ids"]]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        fig["bbox"] = [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def build(out: Path, seed: int) -> dict:
    for sub in ("images", "annotations", "masks", "splits", "metadata"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(PAGE_COUNT):
        image_id = f"page_{i + 1:03d}"
        rng = np.random.default_rng(seed + i)
        conditions = CONDITIONS[i % len(CONDITIONS)]
        n_figures = FIGURE_COUNTS[(i // len(CONDITIONS)) % len(FIGURE_COUNTS)]
        level = 1 + (i % 2)
        scene, figures = render_page(rng, i + 1, n_figures, level)
        bgr, fg, H, ow, oh, params = degrade.apply(np.array(scene.page), np.array(scene.mask), conditions, rng)
        lossy = conditions != ["clean_digital"]
        ext = ".jpg" if lossy else ".png"
        image_path = out / "images" / f"{image_id}{ext}"
        cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92] if lossy else [])
        mask_path = out / "masks" / f"{image_id}.png"
        cv2.imwrite(str(mask_path), fg)
        gt = scene.export(H, ow, oh)
        _figure_boxes(gt, figures)
        annotation = {
            "schema_version": SCHEMA_VERSION, "image_id": image_id,
            "file": f"images/{image_id}{ext}", "mask_file": f"masks/{image_id}.png",
            "width": ow, "height": oh, "difficulty": "worksheet_page", "category": "M_multi_figure_page",
            "diagram_type": "multi_figure_worksheet", "layout": "worksheet", "conditions": conditions,
            "content_stressors": ["page_border", "decorations", "watermark", "answer_boxes"],
            "adversarial_focus": None, "degradation_parameters": params,
            "source": {"kind": "synthetic_rendered", "generator": "evaluation.generator.pages",
                       "dataset_version": DATASET_VERSION, "seed": seed + i},
            "annotation": {"method": "render_time_exact", "machine_assisted": False, "verified": "exact_by_construction"},
            **{k: gt[k] for k in ("objects", "points", "labels", "relationships", "dimensions")},
            "figures": figures,
        }
        annotation["omittable_ids"] = sorted(
            [o["id"] for o in gt["objects"] if o["semantic_importance"] == "omittable"]
            + [lab["id"] for lab in gt["labels"] if lab["semantic_importance"] == "omittable"]
        )
        (out / "annotations" / f"{image_id}.json").write_text(json.dumps(annotation, indent=1) + "\n")
        index.append({"image_id": image_id, "file": annotation["file"], "conditions": conditions,
                      "figures": n_figures, "sha256_image": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    order = np.random.default_rng(seed).permutation(PAGE_COUNT)
    ids = [index[j]["image_id"] for j in order]
    splits = {"dev": sorted(ids[:12]), "validation": sorted(ids[12:21]), "test": sorted(ids[21:])}
    for name, members in splits.items():
        (out / "splits" / f"{name}.txt").write_text("\n".join(members) + "\n")
    meta = {"dataset_version": DATASET_VERSION, "schema_version": SCHEMA_VERSION, "seed": seed,
            "count": len(index), "split_sizes": {k: len(v) for k, v in splits.items()}, "images": index}
    (out / "metadata" / "dataset.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dataset_pages")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    meta = build(args.out, args.seed)
    print(json.dumps({k: meta[k] for k in ("count", "split_sizes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
