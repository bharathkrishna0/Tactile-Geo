#!/usr/bin/env python
"""Build the static assets for the frontend Template demo (a pie chart).

    python scripts/build_template_demo.py

Draws the pie chart, runs the real Model A pipeline on it (EasyOCR + Liblouis),
reconciles it with a recorded Model B response, applies the findings a teacher
would accept through Semantic Geometry v2, and writes everything the frontend
needs into ``frontend/public/demo/tactile-template/``.

Model B is never called here. Its response is read from
``scripts/data/template_pie_model_b.json``,
which was recorded from one live OpenRouter call (the model id is stored in the
file). The frontend Template mode only reads the files this script writes; it
never calls the backend.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.model_b.fusion import reconcile  # noqa: E402
from app.model_b.result_codec import result_from_dict  # noqa: E402
from app.model_b.semantic_v2 import build_semantic_geometry_v2  # noqa: E402
from app.models.geometry import DetectedElement, GeometryType  # noqa: E402
from app.services.braille import UEB_GRADE_2_TABLE  # noqa: E402
from app.services.editing import refresh_semantic_fields, regenerate  # noqa: E402
from app.services.pipeline import build_full_analysis  # noqa: E402
from app.services.tactile_rules import TACTILE_RULES  # noqa: E402
from app.services.tactile_svg import page_layout, render_tactile_svg  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo" / "tactile-template"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
MODEL_B_RECORDING = Path(__file__).resolve().parent / "data" / "template_pie_model_b.json"
JOB_ID = "template-demo"
WIDTH, HEIGHT = 1200, 900
SECTORS = (("Apple", 0, 180, None), ("Mango", 180, 288, "hatch"), ("Banana", 288, 360, "dots"))

TYPE_COLOURS = {
    "circle": "#2383e2",
    "line_segment": "#0f7b6c",
    "rectangle": "#9065b0",
    "polygon": "#9065b0",
    "triangle": "#9065b0",
    "point": "#d9730d",
    "angle": "#d44c47",
    "text_label": "#cb912f",
}


def draw_pie_chart() -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    bold = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 44)
    regular = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 40)
    draw.text((WIDTH // 2, 60), "Favourite Fruits", fill="black", font=bold, anchor="mm")
    cx, cy, r = 400, 500, 280
    box = [cx - r, cy - r, cx + r, cy + r]

    hatch = Image.new("L", (WIDTH, HEIGHT), 0)
    hatch_draw = ImageDraw.Draw(hatch)
    for k in range(-HEIGHT, WIDTH, 18):
        hatch_draw.line([(k, 0), (k + HEIGHT, HEIGHT)], fill=255, width=3)
    blank = Image.new("L", (WIDTH, HEIGHT), 0)

    def sector_mask(start: int, end: int) -> Image.Image:
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(mask).pieslice(box, start - 90, end - 90, fill=255)
        return mask

    for _, start, end, fill in SECTORS:
        mask = sector_mask(start, end)
        if fill == "hatch":
            image.paste((0, 0, 0), mask=Image.composite(hatch, blank, mask))
        elif fill == "dots":
            for x in range(cx - r, cx + r, 24):
                for y in range(cy - r, cy + r, 24):
                    if all(mask.getpixel((x + dx, y + dy)) for dx in (-7, 7) for dy in (-7, 7)):
                        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="black")

    draw.ellipse(box, outline="black", width=6)
    for _, start, _, _ in SECTORS:
        angle = math.radians(start - 90)
        draw.line([(cx, cy), (cx + r * math.cos(angle), cy + r * math.sin(angle))], fill="black", width=6)

    lx, ly = 780, 330
    for index, (name, _, _, fill) in enumerate(SECTORS):
        y = ly + index * 110
        swatch = [lx, y, lx + 80, y + 60]
        if fill == "hatch":
            mask = Image.new("L", (WIDTH, HEIGHT), 0)
            ImageDraw.Draw(mask).rectangle(swatch, fill=255)
            image.paste((0, 0, 0), mask=Image.composite(hatch, blank, mask))
        elif fill == "dots":
            for x in range(lx + 12, lx + 80, 20):
                for yy in range(y + 12, y + 60, 20):
                    draw.ellipse([x - 4, yy - 4, x + 4, yy + 4], fill="black")
        draw.rectangle(swatch, outline="black", width=5)
        draw.text((lx + 110, y + 30), name, fill="black", font=regular, anchor="lm")
    draw.rectangle([lx - 30, ly - 40, lx + 300, ly + 2 * 110 + 100], outline="black", width=4)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _shape_markup(element: DetectedElement, colour: str, width: float, extra: str = "") -> str:
    geo = element.geometry
    attrs = f'data-element-id="{escape(element.id)}" stroke="{colour}" stroke-width="{width}" fill="none"{extra}'
    kind = element.type
    if kind in (GeometryType.LINE_SEGMENT, GeometryType.RAY, GeometryType.AXES, GeometryType.ARROW):
        points = geo.get("points") or [geo.get("start"), geo.get("end")]
        (x1, y1), (x2, y2) = points[0], points[-1]
        return f'<line {attrs} x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke-linecap="round"/>'
    if kind in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in geo["points"])
        return f'<polygon {attrs} points="{points}"/>'
    if kind is GeometryType.CIRCLE:
        (x, y), radius = geo["center"], geo["radius"]
        return f'<circle {attrs} cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}"/>'
    if kind is GeometryType.POINT:
        x, y = geo["position"]
        return f'<circle data-element-id="{escape(element.id)}" fill="{colour}" cx="{x:.1f}" cy="{y:.1f}" r="7"/>'
    if kind is GeometryType.ANGLE:
        x, y = geo["vertex"]
        return f'<circle {attrs} cx="{x:.1f}" cy="{y:.1f}" r="16"/>'
    if kind is GeometryType.TEXT_LABEL and element.bbox:
        x, y, w, h = element.bbox
        return f'<rect {attrs} x="{x}" y="{y}" width="{w}" height="{h}" rx="6" stroke-dasharray="10 6"/>'
    return ""


def _overlay(body: list[str], label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">'
        + "".join(body) + "</svg>\n"
    )


def model_a_overlay(elements: list[DetectedElement]) -> str:
    body = [_shape_markup(e, TYPE_COLOURS.get(e.type.value, "#37352f"), 5) for e in elements]
    return _overlay(body, f"Model A geometry: {len(elements)} detected elements drawn over the original image.")


def model_b_overlay(entities: list[dict]) -> str:
    body = []
    for entity in entities:
        region = entity["region"]
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#9065b0" fill-opacity="0.08" stroke="#9065b0" '
            f'stroke-width="4" stroke-dasharray="14 8" rx="8"/>'
            f'<text x="{x + 8}" y="{y + 26}" font-family="sans-serif" font-size="22" font-weight="700" fill="#6b3fa0">'
            f'{escape(entity["id"])}</text>'
        )
    return _overlay(body, f"Model B regions: {len(entities)} advisory findings drawn over the original image.")


def fusion_overlay(elements: list[DetectedElement], matched: dict[str, str]) -> str:
    body = []
    for element in elements:
        if element.id in matched:
            body.append(_shape_markup(element, "#0f7b6c", 8))
        else:
            body.append(_shape_markup(element, "#9b9a97", 3, ' stroke-opacity="0.8"'))
    return _overlay(body, f"Fused geometry: {len(matched)} Model A elements confirmed by Model B; coordinates unchanged.")


def build(out_dir: Path, model_b_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = draw_pie_chart()
    (out_dir / "original.png").write_bytes(png)

    model_a = build_full_analysis(png)
    semantic = model_a.semantic_geometry
    recorded = json.loads(model_b_path.read_text(encoding="utf-8"))
    model_b = result_from_dict(recorded["result"])
    report = reconcile(semantic, model_b)

    # The demo teacher accepts every finding that strongly matches a Model A
    # element without contradicting it, and leaves the rest undecided.
    decisions = {
        review.model_b_id: "accept"
        for review in report.entity_reviews
        if review.model_a_id and review.correspondence == "strong" and not review.contradicts_model_a
    }
    v2 = build_semantic_geometry_v2(semantic, model_b, decisions, JOB_ID)
    fused = refresh_semantic_fields(v2.semantic)
    simplified, qa_report, final_svg = regenerate(fused)

    elements_by_id = {e.id: e for e in semantic.elements}
    matched = {r.model_a_id: r.model_b_id for r in report.entity_reviews if r.model_a_id}
    geometry_only = [e for e in simplified.elements if e.type is not GeometryType.TEXT_LABEL]
    labels = sorted(
        (e for e in simplified.elements if e.type is GeometryType.TEXT_LABEL),
        key=lambda e: (round(e.geometry["position"][1] / 40), e.geometry["position"][0]),
    )
    layout = page_layout(semantic.image_width, semantic.image_height)

    (out_dir / "model-a.svg").write_text(model_a_overlay(semantic.elements), encoding="utf-8")
    (out_dir / "model-b.svg").write_text(model_b_overlay(recorded["result"]["entities"]), encoding="utf-8")
    (out_dir / "fusion.svg").write_text(fusion_overlay(fused.elements, matched), encoding="utf-8")
    (out_dir / "tactile.svg").write_text(render_tactile_svg(geometry_only, semantic.image_width, semantic.image_height) + "\n", encoding="utf-8")
    (out_dir / "final.svg").write_text(final_svg + "\n", encoding="utf-8")

    outcomes = {o.model_b_id: o for o in v2.outcomes}
    entities = recorded["result"]["entities"]
    metadata = {
        "version": 1,
        "title": "Favourite Fruits pie chart",
        "description": "Successful TactileGeo demonstration",
        "is_precomputed_demo": True,
        "provenance": {
            "generator": "backend/scripts/build_template_demo.py",
            "model_a": "Real Model A pipeline run (OpenCV + EasyOCR + Liblouis) at build time.",
            "model_b": {
                "model": recorded["result"]["resolved_model"] or recorded["model"],
                "provider": recorded["result"]["provider"],
                "elapsed_s": recorded["elapsed_s"],
                "note": "Recorded from one live OpenRouter call and replayed; Template mode makes no call.",
            },
            "teacher": "Demo decisions: accept findings that strongly match a Model A element without contradicting it.",
        },
        "image": {"width": semantic.image_width, "height": semantic.image_height},
        "model_a": {
            "element_count": len(semantic.elements),
            "by_type": dict(Counter(e.type.value for e in semantic.elements).most_common()),
            "relationship_count": len(semantic.relationships),
            "relationships_by_type": dict(Counter(r.type.value for r in semantic.relationships).most_common()),
            "labels": [
                {"id": e.id, "text": e.geometry.get("text", "")}
                for e in semantic.elements if e.type is GeometryType.TEXT_LABEL
            ],
            "review_flags": len(semantic.review_flags),
        },
        "model_b": {
            "diagram_kind": model_b.diagram_kind,
            "description": model_b.diagram_description,
            "entities": [
                {"id": e["id"], "kind": e["kind"], "confidence_level": e["confidence_level"], "evidence": e["evidence"]}
                for e in entities
            ],
            "text_items": [{"text": t["text"], "role": t["role"]} for t in recorded["result"]["text_items"]],
            "relationships": [
                {"kind": r["kind"], "from_id": r["from_id"], "to_id": r["to_id"], "evidence": r["evidence"]}
                for r in recorded["result"]["relationships"]
            ],
            "uncertainties": [
                {"kind": u["kind"], "severity": u["severity"], "note": u["note"]}
                for u in recorded["result"]["uncertainties"]
            ],
        },
        "fusion": {
            "summary": report.summary(),
            "reviews": [
                {
                    "model_b_id": r.model_b_id,
                    "model_b_kind": r.model_b_kind,
                    "model_a_id": r.model_a_id,
                    "model_a_type": elements_by_id[r.model_a_id].type.value if r.model_a_id else None,
                    "correspondence": r.correspondence,
                    "decision": decisions.get(r.model_b_id, "undecided"),
                    "applied": bool(outcomes.get(r.model_b_id) and outcomes[r.model_b_id].applied),
                    "detail": outcomes[r.model_b_id].detail if r.model_b_id in outcomes else r.reason,
                }
                for r in report.entity_reviews
            ],
            "applied": len(v2.applied),
            "not_applied": len(v2.not_applied),
            "coordinates_changed": 0,
        },
        "simplification": {
            "before": len(fused.elements),
            "after": len(simplified.elements),
            "removed": simplified.removed_count,
            "merged": simplified.merged_count,
            "actions": dict(Counter(a.action for a in simplified.actions).most_common()),
            "explanations": simplified.explanations,
        },
        "braille": {
            "table": UEB_GRADE_2_TABLE,
            "labels": [
                {
                    "id": e.id,
                    "text": e.geometry.get("text", ""),
                    "braille": e.geometry.get("braille", ""),
                    "position_mm": [
                        round(e.geometry["position"][0] * layout.mm_per_px, 1),
                        round(e.geometry["position"][1] * layout.mm_per_px, 1),
                    ],
                }
                for e in labels
            ],
        },
        "qa": {
            "passes": qa_report.passes,
            "score": qa_report.score_0_100,
            "errors": [i.message for i in qa_report.issues if i.severity == "error"],
            "issues_by_check": dict(Counter(i.check for i in qa_report.issues).most_common()),
        },
        "final": {
            "page": f"A4 {'landscape' if layout.page_width_mm > layout.page_height_mm else 'portrait'}",
            "width_mm": layout.page_width_mm,
            "height_mm": layout.page_height_mm,
            "stroke_width_pt": TACTILE_RULES.stroke_width_pt,
            "feature_count": len(geometry_only),
            "label_count": len(labels),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--model-b-result", type=Path, default=MODEL_B_RECORDING)
    args = parser.parse_args()
    metadata = build(args.out, args.model_b_result)
    print(json.dumps({k: metadata[k] for k in ("model_a", "fusion", "simplification", "qa", "final")}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
