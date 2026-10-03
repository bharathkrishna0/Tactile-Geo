"""Build the 100-image math benchmark with render-time ground truth.

    python -m evaluation.generator.build [--out evaluation/dataset] [--seed 20261002]

Deterministic for a given seed and library versions; every file's SHA-256 is
recorded in ``metadata/dataset.json`` so a regenerated copy can be checked
against the frozen one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import degrade
from .figures import DIAGRAM_TYPES, TEMPLATES, regular_polygon
from .scene import Scene

SCHEMA_VERSION = "tactilegeo.math_gt.v1"
DATASET_VERSION = "math100-v1"
DEFAULT_SEED = 20261002
ROOT = Path(__file__).resolve().parents[1]
FIGURE_SIZE = (1000, 800)
PAGE_SIZE = (1100, 1556)

GEOMETRY_QUESTIONS = {
    "A_geometry_figure": "In the figure, find the measure of the unknown angle.",
    "C_graph": "Read the value of y when x = 3 from the graph.",
    "D_coordinate_geometry": "Write the coordinates of each plotted point.",
    "E_measurement": "Find the area and perimeter of the figure.",
    "F_shapes_polygons": "Name each polygon and count its sides.",
    "G_angles": "Find the value of x. Give reasons.",
    "H_circles": "O is the centre of the circle. Find the length of the diameter.",
    "I_transformations": "Draw the image of the shape after the transformation.",
    "J_fractions_number_lines": "Mark the given numbers on the number line.",
    "K_tables_diagrams": "Use the diagram to answer the questions below.",
}


@dataclass
class Spec:
    tier: str
    category: str
    level: int
    conditions: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)  # scene-level stressors
    layout: str = "figure"  # figure | worksheet
    figures: list[str] = field(default_factory=list)
    focus: str = ""


def _specs() -> list[Spec]:
    specs: list[Spec] = []
    easy = "A A F F G G H H J J D C E E K I A F G H".split()
    for i, cat in enumerate(easy):
        specs.append(Spec("easy", cat, 1, ["clean_digital"] if i % 5 else ["scanned"]))

    moderate = "A C D E F G H I J K B B A C D E G H I K".split()
    cond_cycle = [["clean_digital"], ["scanned"], ["photographed"], ["clean_digital"], ["scanned"]]
    for i, cat in enumerate(moderate):
        if cat == "B":
            specs.append(Spec("moderate", "B_worksheet_geometry", 2, cond_cycle[i % 5], layout="worksheet",
                              figures=["A_geometry_figure"] if i % 2 == 0 else ["G_angles"]))
        else:
            specs.append(Spec("moderate", cat, 2, cond_cycle[i % 5]))

    hard = "B B B L L C D E G H I J K A F B L D C K".split()
    hard_conds = [["scanned"], ["photographed"], ["scanned", "low_contrast"], ["photographed"], ["scanned"]]
    geo = ["A_geometry_figure", "G_angles", "H_circles", "E_measurement", "F_shapes_polygons"]
    mixed = ["C_graph", "D_coordinate_geometry", "J_fractions_number_lines", "K_tables_diagrams", "I_transformations", "G_angles"]
    for i, cat in enumerate(hard):
        conds = hard_conds[i % 5]
        if cat == "B":
            specs.append(Spec("hard", "B_worksheet_geometry", 2, conds, layout="worksheet",
                              figures=[geo[i % 5], geo[(i + 2) % 5]]))
        elif cat == "L":
            specs.append(Spec("hard", "L_mixed_worksheet", 2, conds, layout="worksheet",
                              figures=[mixed[i % 6], geo[i % 5], mixed[(i + 3) % 6]]))
        else:
            specs.append(Spec("hard", cat, 3, conds, content=["small_text"]))

    very = "L L L B B B C D H G E I K A F L B J D H".split()
    very_conds = [["photographed", "skew"], ["scanned", "blur"], ["photographed", "low_contrast"],
                  ["scanned", "skew", "low_contrast"], ["photographed", "blur"]]
    for i, cat in enumerate(very):
        conds = very_conds[i % 5]
        if cat == "B":
            specs.append(Spec("very_hard", "B_worksheet_geometry", 3, conds, layout="worksheet",
                              figures=[geo[i % 5], geo[(i + 1) % 5], geo[(i + 3) % 5], geo[(i + 4) % 5]]))
        elif cat == "L":
            specs.append(Spec("very_hard", "L_mixed_worksheet", 3, conds, layout="worksheet",
                              figures=[mixed[i % 6], geo[i % 5], mixed[(i + 2) % 6], geo[(i + 2) % 5]]))
        else:
            specs.append(Spec("very_hard", cat, 3, conds, content=["small_text"]))

    adversarial = [
        ("A", 2, ["clean_digital"], ["handwriting"], "handwritten working drawn over the figure"),
        ("G", 2, ["scanned"], ["handwriting"], "handwriting on a scanned angle figure"),
        ("H", 2, ["clean_digital"], ["decorations"], "decorative stars and clip-art around the figure"),
        ("K", 2, ["scanned"], ["decorations", "page_border"], "page border and decorations"),
        ("F", 2, ["clean_digital"], ["dashed"], "every line dashed"),
        ("I", 2, ["scanned"], ["broken_lines"], "lines broken by print gaps"),
        ("C", 3, ["low_resolution"], [], "dense grid at low resolution"),
        ("D", 3, ["clean_digital"], ["tiny_labels"], "labels too small to read reliably"),
        ("E", 2, ["clean_digital"], ["text_crossing"], "question text running across the figure"),
        ("A", 2, ["clean_digital"], ["near_touching"], "two lines drawn about 2 mm apart"),
        ("H", 2, ["jpeg_heavy", "blur"], [], "heavy JPEG compression and blur"),
        ("G", 2, ["dark_page"], [], "light chalk-style drawing on a dark page"),
        ("J", 2, ["photographed", "shadow"], [], "phone photo with a hard shadow"),
        ("F", 2, ["faded"], ["thin_strokes"], "faded print with hairline strokes"),
        ("B", 2, ["photographed", "colored_ink"], [], "coloured ink worksheet photo"),
        ("L", 2, ["scanned"], ["handwriting", "decorations"], "worksheet with handwriting and decorations"),
        ("C", 2, ["strong_rotation", "low_contrast"], [], "page rotated about 10 degrees, low contrast"),
        ("D", 2, ["clean_digital"], ["cropped"], "figure cut off at the image border"),
        ("K", 3, ["clean_digital"], ["handwriting", "overlapping"], "overlapping shapes with handwriting"),
        ("L", 2, ["low_resolution", "jpeg_heavy"], [], "low-resolution, heavily compressed worksheet"),
    ]
    for cat, level, conds, content, focus in adversarial:
        if cat in ("B", "L"):
            figs = ["A_geometry_figure", "H_circles"] if cat == "B" else ["C_graph", "G_angles"]
            name = "B_worksheet_geometry" if cat == "B" else "L_mixed_worksheet"
            specs.append(Spec("adversarial", name, level, conds, content, layout="worksheet", figures=figs, focus=focus))
        else:
            specs.append(Spec("adversarial", cat, level, conds, content, focus=focus))

    full = {k[0]: k for k in TEMPLATES}
    for spec in specs:
        if len(spec.category) == 1:
            spec.category = full[spec.category]
    return specs


# ----------------------------------------------------------------- content
def _wobbly(rng, start, n, step):
    pts = [start]
    angle = float(rng.uniform(0, 2 * math.pi))
    for _ in range(n):
        angle += float(rng.normal(0, 0.6))
        pts.append((pts[-1][0] + step * math.cos(angle), pts[-1][1] + step * math.sin(angle)))
    return pts


def add_handwriting(s: Scene, box, rng, count=3) -> None:
    x0, y0, x1, y1 = box
    for _ in range(count):
        start = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        pts = _wobbly(rng, start, int(rng.integers(14, 30)), float(rng.uniform(5, 9)))
        ink = int(rng.integers(60, 110))
        s.draw.line(pts, fill=ink, width=int(rng.integers(2, 4)), joint="curve")
        s.distractor("handwriting", pts, "pencil working / scribble")
    # a hand-drawn loop around part of the figure
    cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
    rx, ry = float(rng.uniform(30, 70)), float(rng.uniform(20, 45))
    loop = [(cx + rx * math.cos(t) * (1 + 0.08 * math.sin(3 * t)), cy + ry * math.sin(t)) for t in np.linspace(0, 2.2 * math.pi, 40)]
    s.draw.line(loop, fill=80, width=3, joint="curve")
    s.distractor("handwriting", loop, "hand-drawn loop")


def add_decorations(s: Scene, rng, border=False) -> None:
    w, h = s.width, s.height
    for _ in range(3):
        c = (float(rng.uniform(40, w - 40)), float(rng.choice([rng.uniform(30, 80), rng.uniform(h - 80, h - 30)])))
        r = float(rng.uniform(14, 26))
        outer = regular_polygon(c, r, 5)
        inner = regular_polygon(c, r * 0.45, 5, phase=-math.pi / 2 + math.pi / 5)
        star = [p for pair in zip(outer, inner) for p in pair]
        s.draw.polygon(star, outline=0, fill=None if rng.integers(0, 2) else 90, width=2)
        s.distractor("decoration", star + [star[0]], "star")
    # pencil clip-art
    px, py = float(rng.uniform(60, w - 200)), float(h - 50)
    body = [(px, py - 10), (px + 110, py - 10), (px + 110, py + 10), (px, py + 10)]
    tip = [(px + 110, py - 10), (px + 135, py), (px + 110, py + 10)]
    s.draw.polygon(body, outline=0, width=2)
    s.draw.polygon(tip, outline=0, fill=60)
    s.distractor("decoration", body + tip, "pencil clip-art")
    if border:
        for inset in (12, 20):
            frame = [(inset, inset), (w - inset, inset), (w - inset, h - inset), (inset, h - inset)]
            s.draw.line(frame + [frame[0]], fill=0, width=3)
            s.distractor("decoration", frame + [frame[0]], "page border")


def add_near_touching(s: Scene, box) -> None:
    x0, y0, x1, y1 = box
    y = y0 + (y1 - y0) * 0.92
    a = s.segment((x0 + 40, y), (x1 - 40, y), importance="essential")
    b = s.segment((x0 + 40, y + 9), (x1 - 40, y + 9), importance="essential")
    s.relate("PARALLEL", a, b)


# ----------------------------------------------------------------- layout
def render_figure(spec: Spec, rng) -> Scene:
    w, h = FIGURE_SIZE
    small = "small_text" in spec.content
    tiny = "tiny_labels" in spec.content
    s = Scene(w, h, stroke=1 if "thin_strokes" in spec.content else 4)
    if "dashed" in spec.content:
        s.dash_pattern = (16, 10)
    if "broken_lines" in spec.content:
        s.dash_pattern = (70, 6)
    fs = 13 if tiny else (22 if small else 30)
    box = (0, 0, w, h)
    if "cropped" in spec.content:
        box = (w * 0.22, h * 0.1, w * 1.22, h * 1.0)
    TEMPLATES[spec.category](s, box, rng, spec.level, fs)
    if "near_touching" in spec.content:
        add_near_touching(s, (0, 0, w, h))
    if "text_crossing" in spec.content:
        s.text((w * 0.05, h * 0.48), "Find the area of the shaded part and write your answer here.", size=24, role="question_text")
    if "overlapping" in spec.content:
        a = s.polygon([(w * 0.15, h * 0.2), (w * 0.55, h * 0.2), (w * 0.55, h * 0.6), (w * 0.15, h * 0.6)], kind="rectangle")
        b = s.polygon([(w * 0.4, h * 0.4), (w * 0.8, h * 0.4), (w * 0.6, h * 0.85)], kind="triangle")
        s.relate("INTERSECTS", a, b)
    if "handwriting" in spec.content:
        add_handwriting(s, (w * 0.1, h * 0.1, w * 0.9, h * 0.9), rng)
    if "decorations" in spec.content:
        add_decorations(s, rng, border="page_border" in spec.content)
    return s


def render_worksheet(spec: Spec, rng, number: int) -> Scene:
    w, h = PAGE_SIZE
    s = Scene(w, h, stroke=3)
    fs = 22 if spec.level < 3 else 19
    s.text((60, 50), f"Class {['VI', 'VII', 'VIII', 'IX'][number % 4]} Mathematics - Worksheet {number % 9 + 1}",
           size=34, role="title", face="sans_bold")
    s.text((60, 105), "Name: ____________________   Date: ___________", size=22, role="name_line")
    n = len(spec.figures)
    two_col = n >= 4
    top = 165
    rows = math.ceil(n / 2) if two_col else n
    row_h = (h - top - 60) / rows
    col_w = (w - 120) / (2 if two_col else 1)
    for i, cat in enumerate(spec.figures):
        col = i % 2 if two_col else 0
        row = i // 2 if two_col else i
        x0 = 60 + col * col_w
        y0 = top + row * row_h
        s.text((x0, y0), f"{i + 1}. {GEOMETRY_QUESTIONS[cat]}" if not two_col else f"{i + 1}.",
               size=fs, role="question_text", face="serif")
        if two_col:
            s.text((x0 + 30, y0), GEOMETRY_QUESTIONS[cat][:38] + ("..." if len(GEOMETRY_QUESTIONS[cat]) > 38 else ""),
                   size=int(fs * 0.85), role="question_text", face="serif")
        fig_w = min(col_w * 0.85, row_h * 1.25)
        box = (x0 + (col_w - fig_w) / 2, y0 + fs * 2.0, x0 + (col_w + fig_w) / 2, y0 + row_h - fs * 0.5)
        TEMPLATES[cat](s, box, rng, spec.level, fs)
    if "handwriting" in spec.content:
        add_handwriting(s, (80, top, w - 80, h - 80), rng, count=5)
    if "decorations" in spec.content:
        add_decorations(s, rng, border=True)
    return s


# ------------------------------------------------------------------- build
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _splits(specs: list[Spec], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out = {"train": [], "validation": [], "test": []}
    for tier in ("easy", "moderate", "hard", "very_hard", "adversarial"):
        ids = [f"math_{i + 1:03d}" for i, s in enumerate(specs) if s.tier == tier]
        order = rng.permutation(len(ids))
        shuffled = [ids[j] for j in order]
        out["test"] += shuffled[:3]
        out["validation"] += shuffled[3:6]
        out["train"] += shuffled[6:]
    return {k: sorted(v) for k, v in out.items()}


def build(out: Path, seed: int) -> dict:
    specs = _specs()
    assert len(specs) == 100, len(specs)
    for sub in ("images", "annotations", "masks", "splits", "metadata"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    index = []
    for i, spec in enumerate(specs):
        image_id = f"math_{i + 1:03d}"
        rng = np.random.default_rng(seed + i)
        scene = render_worksheet(spec, rng, i) if spec.layout == "worksheet" else render_figure(spec, rng)
        page = np.array(scene.page)
        mask = np.array(scene.mask)
        bgr, fg, H, ow, oh, params = degrade.apply(page, mask, spec.conditions, rng)
        lossy = any(c in spec.conditions for c in ("scanned", "photographed", "jpeg_heavy"))
        ext = ".jpg" if lossy else ".png"
        image_path = out / "images" / f"{image_id}{ext}"
        if lossy:
            cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            cv2.imwrite(str(image_path), bgr)
        mask_path = out / "masks" / f"{image_id}.png"
        cv2.imwrite(str(mask_path), fg)
        gt = scene.export(H, ow, oh)
        annotation = {
            "schema_version": SCHEMA_VERSION,
            "image_id": image_id,
            "file": f"images/{image_id}{ext}",
            "mask_file": f"masks/{image_id}.png",
            "width": ow,
            "height": oh,
            "difficulty": spec.tier,
            "category": spec.category,
            "diagram_type": DIAGRAM_TYPES[spec.category],
            "layout": spec.layout,
            "conditions": spec.conditions,
            "content_stressors": spec.content,
            "adversarial_focus": spec.focus or None,
            "degradation_parameters": params,
            "source": {"kind": "synthetic_rendered", "generator": "evaluation.generator.build",
                       "dataset_version": DATASET_VERSION, "seed": seed + i},
            "annotation": {"method": "render_time_exact", "machine_assisted": False,
                           "verified": "exact_by_construction",
                           "notes": "Written from the drawing coordinates, then mapped through the same homography as the image."},
            "mask_definition": "255 = mathematical geometry ink (lines, shapes, arrows, ticks, axes, point dots); "
                               "0 = background, text, handwriting, decorations, grid lines, shading.",
            **{k: gt[k] for k in ("objects", "points", "labels", "relationships", "dimensions")},
        }
        annotation["omittable_ids"] = sorted(
            [o["id"] for o in gt["objects"] if o["semantic_importance"] == "omittable"]
            + [lab["id"] for lab in gt["labels"] if lab["semantic_importance"] == "omittable"]
        )
        (out / "annotations" / f"{image_id}.json").write_text(json.dumps(annotation, indent=1) + "\n")
        index.append({
            "image_id": image_id, "file": annotation["file"], "difficulty": spec.tier, "category": spec.category,
            "layout": spec.layout, "conditions": spec.conditions, "content_stressors": spec.content,
            "objects": len(gt["objects"]), "labels": len(gt["labels"]), "relationships": len(gt["relationships"]),
            "sha256_image": _sha256(image_path), "sha256_mask": _sha256(mask_path),
        })
    splits = _splits(specs, seed)
    for name, ids in splits.items():
        (out / "splits" / f"{name}.txt").write_text("\n".join(ids) + "\n")
    meta = {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_seed": seed,
        "count": len(index),
        "by_difficulty": _count(index, "difficulty"),
        "by_category": _count(index, "category"),
        "by_layout": _count(index, "layout"),
        "by_condition": _count_multi(index, "conditions"),
        "by_stressor": _count_multi(index, "content_stressors"),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "images": index,
    }
    (out / "metadata" / "dataset.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _count(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))


def _count_multi(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        for v in r[key]:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    meta = build(args.out, args.seed)
    print(json.dumps({k: meta[k] for k in ("count", "by_difficulty", "by_category", "split_sizes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
