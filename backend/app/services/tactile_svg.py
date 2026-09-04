"""Tactile SVG generation.

Renders the simplified semantic geometry as a print-ready tactile representation:
thicker strokes at the configured width, labelled Braille markers, and an
accessibility summary so teachers/experts can review before printing.
"""
from __future__ import annotations

from app.models.geometry import DetectedElement, GeometryType
from app.services.tactile_rules import TACTILE_RULES


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


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
    stroke = TACTILE_RULES.stroke_width_max_pt
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
        elif element.type is GeometryType.POINT:
            pos = geo.get("position")
            if pos:
                glyphs.append(f'<circle {attr} cx="{pos[0]:.1f}" cy="{pos[1]:.1f}" r="{max(2.0, stroke)}"/>')

    # Braille markers as text placeholders placed for the teacher.
    braille_text = ""
    for element in elements:
        if element.type is not GeometryType.TEXT_LABEL:
            continue
        pos = element.geometry.get("position")
        text = element.geometry.get("text", "")
        if pos:
            braille_text += f'<text data-element-id="{_escape(element.id)}" x="{pos[0]:.1f}" y="{pos[1]:.1f}" class="braille">{_escape(str(text))}</text>'

    aria = f"Tactile representation of {sum(1 for e in elements if e.type is not GeometryType.TEXT_LABEL)} geometry features and {sum(1 for e in elements if e.type is GeometryType.TEXT_LABEL)} labels."
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{_escape(aria)}">'
        '<defs><style>'
        f'.stroke {{ stroke:#000; stroke-width:{stroke:.1f}; stroke-linecap:round; stroke-linejoin:round; fill:none; }} '
        f'.braille {{ font-size:{TACTILE_RULES.stroke_width_max_pt * 8:.0f}px; fill:#000; text-anchor:middle; }}'
        '</style></defs>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<g class="stroke">{"" .join(glyphs)}</g>'
        f'<g class="tactile-braille">{braille_text}</g></svg>'
    )