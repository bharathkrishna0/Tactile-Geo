import math

import cv2
import numpy as np

from app.services.segment_geometry import normalize_segments
from app.services.tactile_rules import TACTILE_RULES

# Half of the simplification tolerance: raw Hough fragments of one drawn edge sit
# within a few pixels of each other, so the merge distance is deliberately tighter
# than the semantic-level merge distance.
COLLINEAR_DISTANCE_TOLERANCE_PX = TACTILE_RULES.collinear_merge_tolerance_px / 2.0
COLLINEAR_ANGLE_TOLERANCE_DEG = 3.0


def _line_segments(binary_image: np.ndarray, edge_sensitivity: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, max(20, 110 - edge_sensitivity), minLineLength=max(20, 90 - edge_sensitivity), maxLineGap=12)
    if lines is None:
        return []
    raw = [((int(x1), int(y1)), (int(x2), int(y2))) for x1, y1, x2, y2 in lines[:, 0]]
    return [
        ((int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))))
        for a, b in normalize_segments(raw, COLLINEAR_ANGLE_TOLERANCE_DEG, COLLINEAR_DISTANCE_TOLERANCE_PX)
    ]


# A drawn closed outline has an inner hole; a hole smaller than this fraction of
# the outer area is a filled blob or a glyph counter, not an outline.
MIN_HOLE_AREA_FRACTION = 0.2
# Mean normalised radial error below which a contour is accepted as an ellipse.
# A regular hexagon scores ~0.06 against its best-fit ellipse, so polygons stay
# polygons instead of gaining an ellipse that is not in the source.
ELLIPSE_FIT_MAX_ERROR = 0.035
ELLIPSE_MAX_ASPECT = 0.85
ELLIPSE_MIN_AREA = 200
MIN_CONTOUR_AREA = 80
# Distance from a stroke boundary within which a sample still counts as "on" it.
STROKE_BAND_TOLERANCE_PX = 4.0
LINE_SAMPLES = 12
COVERED_FRACTION = 0.8


def _ellipse_from_fit(fit) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return centre, (semi-major, semi-minor) and the major-axis angle in degrees.

    ``cv2.fitEllipse`` reports ``angle`` for its *first* axis, which is not
    necessarily the major one, so the angle turns by 90 degrees when the axes are
    reordered.
    """
    (cx, cy), (first, second), angle = fit
    if first >= second:
        return (cx, cy), (first / 2, second / 2), angle % 180
    return (cx, cy), (second / 2, first / 2), (angle + 90) % 180


def _ellipse_fit_error(contour: np.ndarray, center, semi_axes, angle: float) -> float:
    a, b = semi_axes
    if a <= 0 or b <= 0:
        return float("inf")
    points = contour.reshape(-1, 2).astype(float) - np.array(center, dtype=float)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = points[:, 0] * cos_t + points[:, 1] * sin_t
    v = -points[:, 0] * sin_t + points[:, 1] * cos_t
    radial = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return float(np.mean(np.abs(radial - 1.0)))


def _approx_points(contour: np.ndarray) -> list[tuple[int, int]]:
    approximation = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    return [(int(point[0][0]), int(point[0][1])) for point in approximation]


def _centerline_points(outer: list[tuple[int, int]], inner: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Average matching outer/inner outline vertices to sit on the stroke centre."""
    if not inner or len(inner) != len(outer):
        return outer
    centred = []
    for x, y in outer:
        ix, iy = min(inner, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        centred.append((int(round((x + ix) / 2)), int(round((y + iy) / 2))))
    return centred


def _stroke_half_width(binary_image: np.ndarray) -> float:
    """Median half-width of the drawn strokes, from the distance-transform ridge."""
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    values = distance[ridge]
    return float(np.median(values)) if values.size else 1.0


def _line_covered_by_outline(segment, outer: np.ndarray, band_px: float) -> bool:
    """True when the segment runs along the outline stroke rather than across it."""
    (x1, y1), (x2, y2) = segment
    samples = [
        (x1 + (x2 - x1) * t / (LINE_SAMPLES - 1), y1 + (y2 - y1) * t / (LINE_SAMPLES - 1))
        for t in range(LINE_SAMPLES)
    ]
    on_outline = sum(
        abs(cv2.pointPolygonTest(outer, (float(x), float(y)), True)) <= band_px for x, y in samples
    )
    return on_outline / LINE_SAMPLES >= COVERED_FRACTION


def _point_segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def _contour_explained_by_lines(contour: np.ndarray, segments, stroke_half_width: float) -> bool:
    """True when a hole-less blob is just the stroke of already-detected lines."""
    points = contour.reshape(-1, 2)
    if not segments or len(points) == 0:
        return False
    tolerance = stroke_half_width + STROKE_BAND_TOLERANCE_PX
    near = sum(
        1 for point in points
        if min(_point_segment_distance(point, a, b) for a, b in segments) <= tolerance
    )
    return near / len(points) >= COVERED_FRACTION


def extract_shapes(binary_image: np.ndarray, edge_sensitivity: int = 50) -> list[dict]:
    """Vectorise a binary stroke mask into lines, closed outlines and ellipses.

    Each drawn feature is emitted once: a closed outline (a contour with an inner
    hole) is authoritative for the Hough lines that run along its stroke, and a
    stroke without a hole is left to the Hough lines that explain it.
    """
    segments = _line_segments(binary_image, edge_sensitivity)
    stroke_half = _stroke_half_width(binary_image)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    closed_shapes: list[dict] = []
    outlines: list[np.ndarray] = []
    open_contours: list[dict] = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        holes = []
        child = hierarchy[index][2]
        while child != -1:
            holes.append(contours[child])
            child = hierarchy[child][0]
        hole_area = sum(cv2.contourArea(hole) for hole in holes)

        outer_points = _approx_points(contour)
        if len(outer_points) < 3:
            continue
        if hole_area < area * MIN_HOLE_AREA_FRACTION:
            if not _contour_explained_by_lines(contour, segments, stroke_half):
                open_contours.append({"type": "contour", "points": outer_points})
            continue

        outlines.append(contour)
        inner_points = _approx_points(holes[0]) if len(holes) == 1 else None
        points = _centerline_points(outer_points, inner_points)
        ellipse = None
        if len(outer_points) >= 5 and len(contour) >= 5:
            center, (a, b), angle = _ellipse_from_fit(cv2.fitEllipse(contour))
            if _ellipse_fit_error(contour, center, (a, b), angle) <= ELLIPSE_FIT_MAX_ERROR:
                # The outer boundary sits half a stroke outside the drawn centreline.
                a, b = max(a - stroke_half, 1.0), max(b - stroke_half, 1.0)
                if b / a < ELLIPSE_MAX_ASPECT and math.pi * a * b > ELLIPSE_MIN_AREA:
                    ellipse = {
                        "type": "ellipse",
                        "center": (round(center[0]), round(center[1])),
                        "semi_axes": (round(a), round(b)),
                        "angle": round(angle, 1),
                        "area": round(math.pi * a * b, 2),
                        "contour_points": points,
                    }
        closed_shapes.append(ellipse or {"type": "contour", "points": points})

    band_px = 2 * stroke_half + STROKE_BAND_TOLERANCE_PX
    shapes: list[dict] = [
        {"type": "line", "points": [start, end]}
        for start, end in segments
        if not any(_line_covered_by_outline((start, end), outer, band_px) for outer in outlines)
    ]
    shapes.extend(closed_shapes)
    shapes.extend(open_contours)
    return shapes


def drop_shapes_inside_text_regions(shapes: list[dict], text_boxes: list[list[tuple[int, int]]], padding: int = 2) -> list[dict]:
    """Remove contours that fall inside an OCR text region.

    Letter glyphs have enough closed area to survive the contour-area filter, so
    without this a worksheet label is vectorized twice: once as braille text and
    once as a tiny polygon. The polygon then trips the minimum-size blocking QA
    check, blocking export on a correctly processed worksheet.
    """
    if not text_boxes:
        return shapes
    regions: list[tuple[int, int, int, int]] = []
    for box in text_boxes:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        regions.append((min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding))

    kept: list[dict] = []
    for shape in shapes:
        if shape["type"] == "line":
            kept.append(shape)
            continue
        points = shape.get("points") or ([shape["center"]] if shape.get("center") else [])
        if not points:
            kept.append(shape)
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            xs = [cx - rx, cx + rx]
            ys = [cy - ry, cy + ry]
        box = (min(xs), min(ys), max(xs), max(ys))
        if any(_mostly_inside(box, region) for region in regions):
            continue
        kept.append(shape)
    return kept


TEXT_CONTAINMENT_FRACTION = 0.6


def _mostly_inside(box: tuple[float, float, float, float], region: tuple[int, int, int, int]) -> bool:
    """True when most of ``box`` lies inside ``region``.

    A plain overlap test also removed real geometry, such as a triangle whose
    vertex label sits just inside its bounding box.
    """
    width = max(box[2] - box[0], 1.0)
    height = max(box[3] - box[1], 1.0)
    overlap_w = min(box[2], region[2]) - max(box[0], region[0])
    overlap_h = min(box[3], region[3]) - max(box[1], region[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return (overlap_w * overlap_h) / (width * height) >= TEXT_CONTAINMENT_FRACTION


def shapes_to_svg(shapes: list[dict], width: int, height: int) -> str:
    elements: list[str] = []
    for shape in shapes:
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            angle = shape.get("angle", 0)
            elements.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({angle} {cx} {cy})" />')
            continue
        points = shape.get("points")
        if not points:
            continue
        if shape["type"] == "line":
            (x1, y1), (x2, y2) = points
            elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        else:
            elements.append('<polygon points="' + ' '.join(f"{x},{y}" for x, y in points) + '" />')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Raw geometry vector preview">'
            '<rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + ''.join(elements) + '</g></svg>')
