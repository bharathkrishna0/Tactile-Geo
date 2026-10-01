"""Right-angle marker detection (Model A).

Textbook figures mark a right angle with a small open square drawn in the
corner between two perpendicular edges. The square is too small to survive as
geometry, yet it carries meaning a learner needs, so it is detected here and
re-emitted as a tactile symbol of standard size instead of being dropped.
"""

from __future__ import annotations

import math

import numpy as np

PERPENDICULAR_TOLERANCE_DEG = 10.0
VERTEX_TOLERANCE_PX = 15.0
MIN_MARKER_PX = 8
MAX_MARKER_PX = 40
MAX_MARKER_ARM_FRACTION = 0.4
# Both edges must be real figure edges, not the short strokes of a letter.
MIN_ARM_PX = 30
SIDE_SAMPLES = 10
SIDE_INK_FRACTION = 0.8
# The marker side must stop at the square's corner; a longer parallel line is
# another edge of the figure, not a marker.
OVERRUN_PX = 6


def _edges(shapes: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    edges = []
    for shape in shapes:
        points = shape.get("points") or []
        if shape["type"] == "line" and len(points) == 2:
            edges.append((tuple(map(float, points[0])), tuple(map(float, points[1]))))
        elif shape["type"] == "contour" and len(points) >= 3:
            for index, start in enumerate(points):
                end = points[(index + 1) % len(points)]
                edges.append((tuple(map(float, start)), tuple(map(float, end))))
    return edges


def _corner(a, b):
    """Shared corner of two edges and their far ends, allowing small overshoot."""
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    vertex = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    near_a = min(a, key=lambda point: math.dist(point, vertex))
    near_b = min(b, key=lambda point: math.dist(point, vertex))
    if math.dist(near_a, vertex) > VERTEX_TOLERANCE_PX or math.dist(near_b, vertex) > VERTEX_TOLERANCE_PX:
        return None
    far_a = a[1] if near_a == a[0] else a[0]
    far_b = b[1] if near_b == b[0] else b[0]
    return vertex, far_a, far_b


def _ink(binary: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = round(point[0]), round(point[1])
    height, width = binary.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return False
    return bool(binary[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any())


def _side_inked(binary: np.ndarray, start, end) -> bool:
    hits = sum(
        _ink(binary, (start[0] + (end[0] - start[0]) * t / SIDE_SAMPLES, start[1] + (end[1] - start[1]) * t / SIDE_SAMPLES))
        for t in range(1, SIDE_SAMPLES + 1)
    )
    return hits >= SIDE_INK_FRACTION * SIDE_SAMPLES


def _marker_size(binary: np.ndarray, vertex, u1, u2, max_size: float, min_size: int) -> int | None:
    for size in range(min_size, int(max_size) + 1):
        p1 = (vertex[0] + u1[0] * size, vertex[1] + u1[1] * size)
        p2 = (vertex[0] + u2[0] * size, vertex[1] + u2[1] * size)
        corner = (p1[0] + u2[0] * size, p1[1] + u2[1] * size)
        if not (_side_inked(binary, p1, corner) and _side_inked(binary, p2, corner)):
            continue
        # A drawn marker is an open square; ink inside it means a dot or a blob.
        if _ink(binary, ((vertex[0] + corner[0]) / 2, (vertex[1] + corner[1]) / 2)):
            continue
        beyond1 = (corner[0] + u2[0] * OVERRUN_PX, corner[1] + u2[1] * OVERRUN_PX)
        beyond2 = (corner[0] + u1[0] * OVERRUN_PX, corner[1] + u1[1] * OVERRUN_PX)
        if _ink(binary, beyond1) or _ink(binary, beyond2):
            continue
        return size
    return None


def detect_right_angle_markers(binary: np.ndarray, shapes: list[dict], stroke_half_width: float = 1.0) -> list[dict]:
    """Corners between perpendicular edges that carry a drawn right-angle square."""
    edges = _edges(shapes)
    min_size = max(MIN_MARKER_PX, math.ceil(2 * stroke_half_width) + 2)
    markers: list[dict] = []
    for i, a in enumerate(edges):
        for b in edges[i + 1:]:
            corner = _corner(a, b)
            if corner is None:
                continue
            vertex, far_a, far_b = corner
            if any(math.dist(vertex, m["vertex"]) <= VERTEX_TOLERANCE_PX for m in markers):
                continue
            len_a, len_b = math.dist(vertex, far_a), math.dist(vertex, far_b)
            if min(len_a, len_b) < max(MIN_ARM_PX, 2 * min_size):
                continue
            u1 = ((far_a[0] - vertex[0]) / len_a, (far_a[1] - vertex[1]) / len_a)
            u2 = ((far_b[0] - vertex[0]) / len_b, (far_b[1] - vertex[1]) / len_b)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))))
            if abs(angle - 90.0) > PERPENDICULAR_TOLERANCE_DEG:
                continue
            max_size = min(MAX_MARKER_PX, MAX_MARKER_ARM_FRACTION * min(len_a, len_b))
            size = _marker_size(binary, vertex, u1, u2, max_size, min_size)
            if size is None:
                continue
            markers.append({
                "vertex": (round(vertex[0]), round(vertex[1])),
                "arms": [(round(far_a[0]), round(far_a[1])), (round(far_b[0]), round(far_b[1]))],
                "size": size,
                "degrees": round(angle, 1),
            })
    return markers


def drop_marker_strokes(shapes: list[dict], markers: list[dict]) -> list[dict]:
    """Remove small contours that are the drawn marker squares themselves."""
    def inside_marker(shape: dict) -> bool:
        points = shape.get("points") or []
        if shape["type"] != "contour" or not points:
            return False
        for marker in markers:
            reach = marker["size"] * math.sqrt(2) + VERTEX_TOLERANCE_PX
            if all(math.dist(point, marker["vertex"]) <= reach for point in points):
                return True
        return False

    return [shape for shape in shapes if not inside_marker(shape)]
