"""Tactile SVG generation.

Renders the simplified semantic geometry as a print-ready tactile representation:
thicker strokes at the configured width, labelled Braille markers, and an
accessibility summary so teachers/experts can review before printing.
"""
from __future__ import annotations

import re

from app.models.geometry import DetectedElement, GeometryType
from app.services.tactile_rules import TACTILE_RULES

# Unicode braille needs a font that actually contains the U+2800 block. Without
# this declaration SVG renderers fall back to a font that shows hex boxes.
BRAILLE_FONT_STACK = "'DejaVu Sans','Noto Sans Symbols 2','Segoe UI Symbol','Arial Unicode MS',sans-serif"
BRAILLE_FONT_SIZE_PX = int(TACTILE_RULES.stroke_width_pt * 8)
# Advance width of one braille cell in the font stack above (DejaVu Sans: 0.73em).
BRAILLE_CELL_ADVANCE_EM = 0.75


def braille_text_extent(braille: str) -> tuple[float, float]:
    """Width and height in px of a braille string as rendered by the tactile SVG."""
    return len(braille) * BRAILLE_CELL_ADVANCE_EM * BRAILLE_FONT_SIZE_PX, float(BRAILLE_FONT_SIZE_PX)

_SCRIPT_TAG = re.compile(r"<\s*script", re.IGNORECASE)
_FOREIGN_OBJECT = re.compile(r"<\s*foreignObject", re.IGNORECASE)
_EVENT_HANDLER = re.compile(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_JS_URL = re.compile(r"(href|xlink:href)\s*=\s*(\"|')\s*javascript:[^\"']*(\"|')", re.IGNORECASE)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def sanitize_svg(svg: str) -> str:
    """Strip active content from generated SVG before it reaches the browser.

    The SVG is rendered through ``dangerouslySetInnerHTML`` and downloaded by the
    teacher, so executable content must never survive generation.
    """
    cleaned = _SCRIPT_TAG.sub("<blocked-script", svg)
    cleaned = _FOREIGN_OBJECT.sub("<blocked-foreignobject", cleaned)
    cleaned = _EVENT_HANDLER.sub("", cleaned)
    cleaned = _JS_URL.sub(r'\1="#"', cleaned)
    return cleaned



def _element_points(element: DetectedElement) -> list[tuple[float, float]]:
    geo = element.geometry
    points: list[tuple[float, float]] = []
    if "points" in geo:
        points.extend((float(p[0]), float(p[1])) for p in geo["points"])
    if "start" in geo:
        points.append((float(geo["start"][0]), float(geo["start"][1])))
    if "end" in geo:
        points.append((float(geo["end"][0]), float(geo["end"][1])))
    if "center" in geo:
        points.append((float(geo["center"][0]), float(geo["center"][1])))
    if "position" in geo:
        points.append((float(geo["position"][0]), float(geo["position"][1])))
    if "vertex" in geo:
        points.append((float(geo["vertex"][0]), float(geo["vertex"][1])))
    if "arms" in geo:
        for arm in geo["arms"]:
            points.append((float(arm[0]), float(arm[1])))
    return points


def render_tactile_svg(elements: list[DetectedElement], width: int, height: int) -> str:
    stroke = TACTILE_RULES.stroke_width_pt
    glyphs: list[str] = []
    for element in elements:
        if element.type is GeometryType.TEXT_LABEL:
            continue
        geo = element.geometry
        attr = f'data-element-id="{_escape(element.id)}"'
        if element.type in (GeometryType.LINE_SEGMENT, GeometryType.RAY, GeometryType.AXES, GeometryType.ARROW):
            points = _element_points(element)
            if len(points) >= 2:
                glyphs.append(f'<line {attr} x1="{points[0][0]:.1f}" y1="{points[0][1]:.1f}" x2="{points[-1][0]:.1f}" y2="{points[-1][1]:.1f}"/>')
        elif element.type in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON):
            points = _element_points(element)
            if len(points) >= 3:
                glyphs.append(f'<polygon {attr} points="' + ' '.join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
        elif element.type is GeometryType.CIRCLE:
            glyphs.append(f'<circle {attr} cx="{geo["center"][0]:.1f}" cy="{geo["center"][1]:.1f}" r="{geo["radius"]:.1f}"/>')
        elif element.type is GeometryType.ELLIPSE:
            cx, cy = geo["center"]
            semi_a, semi_b = geo["semi_axes"]
            angle = geo.get("angle", 0)
            glyphs.append(f'<ellipse {attr} cx="{cx:.1f}" cy="{cy:.1f}" rx="{semi_a:.1f}" ry="{semi_b:.1f}" transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"/>')
        elif element.type is GeometryType.POINT:
            pos = geo.get("position")
            if pos:
                glyphs.append(f'<circle {attr} cx="{pos[0]:.1f}" cy="{pos[1]:.1f}" r="{max(2.0, stroke)}"/>')

    # Braille markers. The tactile output must carry the braille translation, not
    # the plain OCR string: the printed sheet is read by touch, so emitting "A"
    # instead of the braille cell meant the export was not actually tactile.
    braille_glyphs: list[str] = []
    for element in elements:
        if element.type is not GeometryType.TEXT_LABEL:
            continue
        pos = element.geometry.get("position")
        if not pos:
            continue
        braille = str(element.geometry.get("braille") or "").strip()
        label = braille or str(element.geometry.get("text") or "")
        if not label:
            continue
        braille_glyphs.append(
            f'<text data-element-id="{_escape(element.id)}" x="{pos[0]:.1f}" y="{pos[1]:.1f}" '
            f'class="braille" aria-label="{_escape(str(element.geometry.get("text") or ""))}">'
            f'{_escape(label)}</text>'
        )

    aria = f"Tactile representation of {sum(1 for e in elements if e.type is not GeometryType.TEXT_LABEL)} geometry features and {sum(1 for e in elements if e.type is GeometryType.TEXT_LABEL)} labels."
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{_escape(aria)}">'
        '<defs><style>'
        f'.stroke {{ stroke:#000; stroke-width:{stroke:.1f}; stroke-linecap:round; stroke-linejoin:round; fill:none; }} '
        f'.braille {{ font-family:{BRAILLE_FONT_STACK}; font-size:{BRAILLE_FONT_SIZE_PX}px; fill:#000; text-anchor:middle; dominant-baseline:middle; }}'
        '</style></defs>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<g class="stroke">{"" .join(glyphs)}</g>'
        f'<g class="tactile-braille">{"".join(braille_glyphs)}</g></svg>'
    )
    return sanitize_svg(svg)
