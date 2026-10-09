"""Right-angle marker detection (Model A).

Textbook figures mark a right angle with a small open square drawn in the
corner between two perpendicular edges. The square is too small to survive as
geometry, yet it carries meaning a learner needs, so it is detected here and
re-emitted as a tactile symbol of standard size instead of being dropped.
"""

from __future__ import annotations

import math

import cv2
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


def _t_corners(a, b):
    """Corners at a T-junction: ``b`` ends on the interior of ``a``.

    The square can sit on either side of the through edge, so both arms of
    ``a`` are returned as candidates.
    """
    for through, stem in ((a, b), (b, a)):
        (x1, y1), (x2, y2) = through
        (x3, y3), (x4, y4) = stem
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-9:
            continue
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        vertex = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        near_stem = min(stem, key=lambda point: math.dist(point, vertex))
        if math.dist(near_stem, vertex) > VERTEX_TOLERANCE_PX:
            continue
        if min(math.dist(vertex, through[0]), math.dist(vertex, through[1])) <= VERTEX_TOLERANCE_PX:
            continue
        if not 0.0 < t < 1.0:
            continue
        far_stem = stem[1] if near_stem == stem[0] else stem[0]
        return [(vertex, through[0], far_stem), (vertex, through[1], far_stem)]
    return []


# A marker whose corner edges were not vectorised (a dashed altitude, a broken
# scan) is still visible as an L of leftover ink once the long edges are erased.
# Its missing corner must sit on a measured edge parallel to one arm, with ink
# continuing past the square along the other.
L_PERPENDICULAR_TOLERANCE_DEG = 15.0
MIN_L_ARM_RATIO = 0.6
MIN_L_ON_ARMS = 0.85
L_VERTEX_EDGE_PX = 6.0
MIN_CONTINUATION_INK = 0.3


def _segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _parallel(u, edge) -> bool:
    length = math.dist(*edge)
    cos = abs(u[0] * (edge[1][0] - edge[0][0]) + u[1] * (edge[1][1] - edge[0][1])) / length
    return cos >= math.cos(math.radians(PERPENDICULAR_TOLERANCE_DEG))


def _ink_near(binary: np.ndarray, point, radius: int) -> bool:
    x, y = round(point[0]), round(point[1])
    height, width = binary.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return False
    return bool(binary[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].any())


def _in_boxes(point, boxes) -> bool:
    for box in boxes:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        if min(xs) - 2 <= point[0] <= max(xs) + 2 and min(ys) - 2 <= point[1] <= max(ys) + 2:
            return True
    return False


def _l_marker(binary, pts: np.ndarray, edges, stroke_half_width: float, min_size: int) -> dict | None:
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        return None
    end_a, end_b = max(((p, q) for i, p in enumerate(hull) for q in hull[i + 1:]), key=lambda pq: math.dist(*pq))
    chord = math.dist(end_a, end_b)
    normal = ((end_b[1] - end_a[1]) / chord, -(end_b[0] - end_a[0]) / chord)
    elbow = max(hull, key=lambda p: abs((p[0] - end_a[0]) * normal[0] + (p[1] - end_a[1]) * normal[1]))
    len_a, len_b = math.dist(elbow, end_a), math.dist(elbow, end_b)
    if min(len_a, len_b) < 0.6 * min_size or max(len_a, len_b) > MAX_MARKER_PX:
        return None
    if min(len_a, len_b) < MIN_L_ARM_RATIO * max(len_a, len_b):
        return None
    cos = ((end_a[0] - elbow[0]) * (end_b[0] - elbow[0]) + (end_a[1] - elbow[1]) * (end_b[1] - elbow[1])) / (len_a * len_b)
    if abs(math.degrees(math.acos(max(-1.0, min(1.0, cos)))) - 90.0) > L_PERPENDICULAR_TOLERANCE_DEG:
        return None
    band = stroke_half_width + 1.5
    on_arms = sum(1 for p in pts if min(_segment_distance(p, elbow, end_a), _segment_distance(p, elbow, end_b)) <= band)
    if on_arms < MIN_L_ON_ARMS * len(pts):
        return None
    vertex = (end_a[0] + end_b[0] - elbow[0], end_a[1] + end_b[1] - elbow[1])
    u_a = ((end_a[0] - vertex[0]) / len_b, (end_a[1] - vertex[1]) / len_b)
    u_b = ((end_b[0] - vertex[0]) / len_a, (end_b[1] - vertex[1]) / len_a)
    size = int(round((len_a + len_b) / 2))
    near = [edge for edge in edges if _segment_distance(vertex, *edge) <= L_VERTEX_EDGE_PX]
    for along, across in ((u_a, u_b), (u_b, u_a)):
        if not any(_parallel(along, edge) for edge in near):
            continue
        samples = [(vertex[0] + across[0] * t, vertex[1] + across[1] * t) for t in np.linspace(1.3 * size, 3 * size, SIDE_SAMPLES)]
        continued = any(_parallel(across, edge) for edge in near)
        reach_px = max(2, int(round(stroke_half_width + 3)))
        inked = sum(_ink_near(binary, p, reach_px) for p in samples)
        if continued or inked >= MIN_CONTINUATION_INK * len(samples):
            reach = max(MIN_ARM_PX, 3 * size)
            return {
                "vertex": (round(vertex[0]), round(vertex[1])),
                "arms": [(round(vertex[0] + u[0] * reach), round(vertex[1] + u[1] * reach)) for u in (along, across)],
                "size": size,
                "degrees": 90.0,
            }
    return None


def _residual_markers(binary: np.ndarray, shapes: list[dict], stroke_half_width: float, min_size: int,
                      markers: list[dict], text_boxes) -> list[dict]:
    edges = [edge for edge in _edges(shapes) if math.dist(*edge) >= MIN_ARM_PX]
    if not edges:
        return []
    thickness = max(3, int(round(2 * stroke_half_width + 3)))
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    for start, end in edges:
        cv2.line(mask, tuple(map(int, start)), tuple(map(int, end)), 255, thickness)
    residual = cv2.bitwise_and((binary > 0).astype(np.uint8) * 255, cv2.bitwise_not(mask))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    found: list[dict] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if max(w, h) < 0.6 * min_size or max(w, h) > MAX_MARKER_PX * 1.2 or area < min_size:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == label)
        pts = np.stack([xs + x, ys + y], axis=1).astype(np.int32)
        marker = _l_marker(binary, pts, edges, stroke_half_width, min_size)
        if marker is None or _in_boxes(marker["vertex"], text_boxes):
            continue
        if any(math.dist(marker["vertex"], m["vertex"]) <= VERTEX_TOLERANCE_PX for m in markers + found):
            continue
        found.append(marker)
    return found


def detect_right_angle_markers(binary: np.ndarray, shapes: list[dict], stroke_half_width: float = 1.0,
                               text_boxes=()) -> list[dict]:
    """Corners and T-junctions between perpendicular edges that carry a drawn right-angle square."""
    edges = _edges(shapes)
    min_size = max(MIN_MARKER_PX, math.ceil(2 * stroke_half_width) + 2)
    markers: list[dict] = []
    for i, a in enumerate(edges):
        for b in edges[i + 1:]:
            corner = _corner(a, b)
            candidates = [corner] if corner is not None else _t_corners(a, b)
            for vertex, far_a, far_b in candidates:
                marker = _marker_at(binary, vertex, far_a, far_b, min_size, markers)
                if marker:
                    markers.append(marker)
                    break
    return markers + _residual_markers(binary, shapes, stroke_half_width, min_size, markers, text_boxes)


def _marker_at(binary: np.ndarray, vertex, far_a, far_b, min_size: int, markers: list[dict]) -> dict | None:
    if any(math.dist(vertex, m["vertex"]) <= VERTEX_TOLERANCE_PX for m in markers):
        return None
    len_a, len_b = math.dist(vertex, far_a), math.dist(vertex, far_b)
    if min(len_a, len_b) < max(MIN_ARM_PX, 2 * min_size):
        return None
    u1 = ((far_a[0] - vertex[0]) / len_a, (far_a[1] - vertex[1]) / len_a)
    u2 = ((far_b[0] - vertex[0]) / len_b, (far_b[1] - vertex[1]) / len_b)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))))
    if abs(angle - 90.0) > PERPENDICULAR_TOLERANCE_DEG:
        return None
    max_size = min(MAX_MARKER_PX, MAX_MARKER_ARM_FRACTION * min(len_a, len_b))
    size = _marker_size(binary, vertex, u1, u2, max_size, min_size)
    if size is None:
        return None
    return {
        "vertex": (round(vertex[0]), round(vertex[1])),
        "arms": [(round(far_a[0]), round(far_a[1])), (round(far_b[0]), round(far_b[1]))],
        "size": size,
        "degrees": round(angle, 1),
    }


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
