"""Tactile SVG generation.

Renders the simplified semantic geometry as a print-ready tactile representation:
thicker strokes at the configured width, labelled Braille markers, and an
accessibility summary so teachers/experts can review before printing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.models.geometry import DetectedElement, GeometryType
from app.services.tactile_rules import TACTILE_RULES

# Physical page. The tactile SVG keeps image pixels as its user units, so the
# geometry coordinates stay traceable to the source, and maps them onto an A4
# sheet (portrait or landscape, whichever fits the image better).
A4_MM = (210.0, 297.0)
PAGE_MARGIN_MM = 12.7
MM_PER_PT = 25.4 / 72.0

# Standard braille cell (BANA/Library of Congress specification 800):
# 2.5 mm dot pitch within a cell, 6.0 mm cell-to-cell pitch, 1.5 mm dot base.
BRAILLE_DOT_PITCH_MM = 2.5
BRAILLE_CELL_PITCH_MM = 6.0
BRAILLE_DOT_DIAMETER_MM = 1.5
# Right-angle symbol: an open square of fixed physical size in the corner,
# never more than this share of the shorter edge so it stays inside the angle.
RIGHT_ANGLE_SYMBOL_MM = 6.0
RIGHT_ANGLE_MAX_ARM_FRACTION = 0.4
# Unicode braille dot bit -> (column, row) inside the cell.
_BRAILLE_DOT_POSITIONS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3))
UNICODE_BRAILLE_BASE = 0x2800


@dataclass(frozen=True)
class PageLayout:
    page_width_mm: float
    page_height_mm: float
    mm_per_px: float

    @property
    def view_box(self) -> tuple[float, float, float, float]:
        """Page rectangle in image pixels, with the image centred on the sheet."""
        width_px = self.page_width_mm / self.mm_per_px
        height_px = self.page_height_mm / self.mm_per_px
        return (0.0, 0.0, width_px, height_px)


def page_layout(width: int, height: int) -> PageLayout:
    """Fit a ``width`` x ``height`` px image into the printable area of an A4 sheet."""
    short, long = A4_MM
    page_width, page_height = (long, short) if width > height else (short, long)
    mm_per_px = min(
        (page_width - 2 * PAGE_MARGIN_MM) / max(width, 1),
        (page_height - 2 * PAGE_MARGIN_MM) / max(height, 1),
    )
    return PageLayout(page_width, page_height, mm_per_px)


def _braille_cells(braille: str) -> list[int]:
    return [
        ord(char) - UNICODE_BRAILLE_BASE if UNICODE_BRAILLE_BASE <= ord(char) <= UNICODE_BRAILLE_BASE + 0xFF else 0
        for char in braille
    ]


def _braille_rows(braille: str) -> int:
    return 4 if any(cell & 0xC0 for cell in _braille_cells(braille)) else 3


def braille_text_extent(braille: str, width: int, height: int) -> tuple[float, float]:
    """Width and height in image px of ``braille`` embossed at standard cell size."""
    cells = max(len(braille), 1)
    width_mm = (cells - 1) * BRAILLE_CELL_PITCH_MM + BRAILLE_DOT_PITCH_MM + BRAILLE_DOT_DIAMETER_MM
    height_mm = (_braille_rows(braille) - 1) * BRAILLE_DOT_PITCH_MM + BRAILLE_DOT_DIAMETER_MM
    mm_per_px = page_layout(width, height).mm_per_px
    return width_mm / mm_per_px, height_mm / mm_per_px


def _braille_dots(braille: str, centre: tuple[float, float], mm_per_px: float) -> list[tuple[float, float]]:
    """Dot centres, in image px, for ``braille`` centred on ``centre``."""
    pitch = BRAILLE_DOT_PITCH_MM / mm_per_px
    cell_pitch = BRAILLE_CELL_PITCH_MM / mm_per_px
    cells = _braille_cells(braille)
    left = centre[0] - ((len(cells) - 1) * cell_pitch + pitch) / 2
    top = centre[1] - (_braille_rows(braille) - 1) * pitch / 2
    dots: list[tuple[float, float]] = []
    for index, cell in enumerate(cells):
        for bit, (column, row) in enumerate(_BRAILLE_DOT_POSITIONS):
            if cell & (1 << bit):
                dots.append((left + index * cell_pitch + column * pitch, top + row * pitch))
    return dots


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


def _right_angle_symbol(geo: dict, side: float) -> str | None:
    vx, vy = (float(v) for v in geo["vertex"])
    units = []
    lengths = []
    for arm in geo.get("arms") or []:
        dx, dy = float(arm[0]) - vx, float(arm[1]) - vy
        length = math.hypot(dx, dy)
        if length == 0:
            return None
        units.append((dx / length, dy / length))
        lengths.append(length)
    if len(units) != 2:
        return None
    side = min(side, RIGHT_ANGLE_MAX_ARM_FRACTION * min(lengths))
    (ax, ay), (bx, by) = units
    p1 = (vx + ax * side, vy + ay * side)
    p2 = (vx + bx * side, vy + by * side)
    corner = (p1[0] + bx * side, p1[1] + by * side)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in (p1, corner, p2))


def is_embossed_geometry(element: DetectedElement) -> bool:
    """True for elements the compiler embosses as a tactile line, shape or dot."""
    if element.type is GeometryType.TEXT_LABEL:
        return False
    if element.type is GeometryType.ANGLE:
        return bool(element.geometry.get("right_angle_marker"))
    return element.type is not GeometryType.ARC


def render_tactile_svg(elements: list[DetectedElement], width: int, height: int) -> str:
    layout = page_layout(width, height)
    stroke = TACTILE_RULES.stroke_width_pt
    # The configured stroke is in points on paper; convert to image-px user units.
    stroke_user = stroke * MM_PER_PT / layout.mm_per_px
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
        elif element.type is GeometryType.ANGLE and geo.get("right_angle_marker"):
            symbol = _right_angle_symbol(geo, RIGHT_ANGLE_SYMBOL_MM / layout.mm_per_px)
            if symbol:
                glyphs.append(f'<polyline {attr} class="right-angle" points="{symbol}"/>')
        elif element.type is GeometryType.POINT:
            pos = geo.get("position")
            if pos:
                glyphs.append(f'<circle {attr} cx="{pos[0]:.1f}" cy="{pos[1]:.1f}" r="{max(2.0, stroke_user):.1f}"/>')

    # Braille markers are embossed as dots at standard cell size, so the printed
    # sheet does not depend on a braille font being installed or scaled right.
    braille_glyphs: list[str] = []
    dot_radius = BRAILLE_DOT_DIAMETER_MM / 2 / layout.mm_per_px
    for element in elements:
        if element.type is not GeometryType.TEXT_LABEL:
            continue
        pos = element.geometry.get("position")
        braille = str(element.geometry.get("braille") or "").strip()
        if not pos or not braille:
            continue
        text = str(element.geometry.get("text") or "")
        dots = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_radius:.2f}"/>'
            for x, y in _braille_dots(braille, (float(pos[0]), float(pos[1])), layout.mm_per_px)
        )
        braille_glyphs.append(
            f'<g data-element-id="{_escape(element.id)}" class="braille" role="img" '
            f'aria-label="{_escape(text)}" data-braille="{_escape(braille)}"><title>{_escape(text)}</title>{dots}</g>'
        )

    aria = f"Tactile representation of {sum(1 for e in elements if e.type is not GeometryType.TEXT_LABEL)} geometry features and {sum(1 for e in elements if e.type is GeometryType.TEXT_LABEL)} labels."
    vx, vy, vw, vh = layout.view_box
    offset_x = (vw - width) / 2
    offset_y = (vh - height) / 2
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx - offset_x:.2f} {vy - offset_y:.2f} {vw:.2f} {vh:.2f}" '
        f'width="{layout.page_width_mm:g}mm" height="{layout.page_height_mm:g}mm" '
        f'data-mm-per-px="{layout.mm_per_px:.5f}" role="img" aria-label="{_escape(aria)}">'
        '<defs><style>'
        f'.stroke {{ stroke:#000; stroke-width:{stroke_user:.3f}; stroke-linecap:round; stroke-linejoin:round; fill:none; }} '
        '.braille { fill:#000; stroke:none; }'
        '</style></defs>'
        f'<rect x="{vx - offset_x:.2f}" y="{vy - offset_y:.2f}" width="{vw:.2f}" height="{vh:.2f}" fill="white"/>'
        f'<g class="stroke">{"" .join(glyphs)}</g>'
        f'<g class="tactile-braille">{"".join(braille_glyphs)}</g></svg>'
    )
    return sanitize_svg(svg)
