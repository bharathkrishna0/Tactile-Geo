"""Drawn convention marks that carry relationships (Model A).

Textbook figures state facts with small marks rather than words: an arc at a
vertex marks an angle, and matching chevrons on two lines mark them parallel.
The marks are too small to survive as geometry, so they are read from the ink
left over once the measured edges are erased, and each one is only accepted
when it agrees with those measured edges.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MIN_EDGE_PX = 30
VERTEX_MERGE_PX = 6.0
VERTEX_EDGE_PX = 8.0

MIN_ARC_RADIUS_PX = 12
MAX_ARC_RADIUS_PX = 110
ARC_BIN_DEG = 3
ARC_MAX_GAP_BINS = 3
MIN_ARC_SPAN_DEG = 15.0
MAX_ARC_SPAN_DEG = 300.0
MIN_ARC_COVERAGE = 0.75
ARC_ARM_DEG = 14.0

MIN_CHEVRON_PX = 5
MAX_CHEVRON_PX = 30
MIN_CHEVRON_DEG = 20.0
MAX_CHEVRON_DEG = 70.0
CHEVRON_MIRROR_DEG = 15.0
CHEVRON_END_MARGIN = 0.15
CHEVRON_GROUP_PX = 2.5


Edge = tuple[tuple[float, float], tuple[float, float]]


def long_edges(shapes: list[dict]) -> list[Edge]:
    edges: list[Edge] = []
    for shape in shapes:
        points = shape.get("points") or []
        if shape.get("role") == "tick":
            continue
        if shape["type"] == "line" and len(points) == 2:
            pairs = [(points[0], points[1])]
        elif shape["type"] == "contour" and len(points) >= 3:
            pairs = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        else:
            continue
        for start, end in pairs:
            edge = (tuple(map(float, start)), tuple(map(float, end)))
            if math.dist(*edge) >= MIN_EDGE_PX:
                edges.append(edge)
    return edges


def residual_ink(binary: np.ndarray, edges: list[Edge], stroke_half: float, text_boxes=()) -> np.ndarray:
    """Ink not explained by a measured edge or a recognised label."""
    thickness = max(3, int(round(2 * stroke_half + 3)))
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    for start, end in edges:
        cv2.line(mask, tuple(map(int, start)), tuple(map(int, end)), 255, thickness)
    for box in text_boxes:
        xs, ys = [int(p[0]) for p in box], [int(p[1]) for p in box]
        cv2.rectangle(mask, (min(xs) - 2, min(ys) - 2), (max(xs) + 2, max(ys) + 2), 255, -1)
    return cv2.bitwise_and((binary > 0).astype(np.uint8) * 255, cv2.bitwise_not(mask))


def _segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _intersection(a: Edge, b: Edge):
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    point = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    if _segment_distance(point, *a) > VERTEX_EDGE_PX or _segment_distance(point, *b) > VERTEX_EDGE_PX:
        return None
    return point


def _vertices(edges: list[Edge]) -> list[tuple[float, float]]:
    candidates = [p for edge in edges for p in edge]
    for i, a in enumerate(edges):
        for b in edges[i + 1:]:
            point = _intersection(a, b)
            if point is not None:
                candidates.append(point)
    out: list[tuple[float, float]] = []
    for point in candidates:
        if all(math.dist(point, q) > VERTEX_MERGE_PX for q in out):
            out.append(point)
    return out


def _directions(vertex, edges: list[Edge]) -> list[float]:
    """Angles (deg, image coordinates) of the measured edges leaving ``vertex``."""
    out: list[float] = []
    for start, end in edges:
        if _segment_distance(vertex, start, end) > VERTEX_EDGE_PX:
            continue
        for far in (start, end):
            if math.dist(far, vertex) >= MIN_EDGE_PX:
                out.append(math.degrees(math.atan2(far[1] - vertex[1], far[0] - vertex[0])) % 360)
    return out


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _longest_run(occupied: np.ndarray) -> tuple[int, int, int] | None:
    """Longest circular run of occupied bins allowing short gaps: (start, length, filled)."""
    n = len(occupied)
    if occupied.all():
        return 0, n, n
    best = None
    start_bins = [i for i in range(n) if occupied[i] and not occupied[i - 1]]
    for start in start_bins:
        length = filled = gap = 0
        last_end = 0
        for step in range(n):
            if occupied[(start + step) % n]:
                filled += 1
                gap = 0
                last_end = step + 1
            else:
                gap += 1
                if gap > ARC_MAX_GAP_BINS:
                    break
        length = last_end
        if best is None or length > best[1]:
            best = (start, length, filled)
    return best


def detect_angle_arcs(binary: np.ndarray, shapes: list[dict], stroke_half: float, text_boxes=(),
                      exclude_vertices=()) -> list[dict]:
    """Arcs drawn around a measured vertex whose ends meet two arms of that vertex."""
    edges = long_edges(shapes)
    if not edges:
        return []
    residual = residual_ink(binary, edges, stroke_half, text_boxes)
    ys, xs = np.nonzero(residual)
    if not len(xs):
        return []
    band = stroke_half + 1.5
    bins = 360 // ARC_BIN_DEG
    arcs: list[dict] = []
    for vertex in _vertices(edges):
        if any(math.dist(vertex, v) <= 12 for v in exclude_vertices):
            continue
        directions = _directions(vertex, edges)
        if not directions:
            continue
        near = (np.abs(xs - vertex[0]) <= MAX_ARC_RADIUS_PX + band) & (np.abs(ys - vertex[1]) <= MAX_ARC_RADIUS_PX + band)
        if not near.any():
            continue
        dx, dy = xs[near] - vertex[0], ys[near] - vertex[1]
        radii = np.hypot(dx, dy)
        thetas = (np.degrees(np.arctan2(dy, dx)) % 360)
        best = None
        for radius in range(MIN_ARC_RADIUS_PX, MAX_ARC_RADIUS_PX + 1, 2):
            on = np.abs(radii - radius) <= band
            if on.sum() < 6:
                continue
            occupied = np.zeros(bins, dtype=bool)
            occupied[(thetas[on] // ARC_BIN_DEG).astype(int) % bins] = True
            run = _longest_run(occupied)
            if run is None:
                continue
            start, length, filled = run
            span = length * ARC_BIN_DEG
            if not MIN_ARC_SPAN_DEG <= span <= MAX_ARC_SPAN_DEG or filled < MIN_ARC_COVERAGE * length:
                continue
            # The arc's own pixels must mostly lie on this circle, not cross it.
            if best is None or span * filled / length > best[0]:
                best = (span * filled / length, radius, start * ARC_BIN_DEG, (start + length) * ARC_BIN_DEG % 360, span)
        if best is None:
            continue
        _, radius, a0, a1, span = best
        arm0 = min(directions, key=lambda d: _angle_gap(d, a0))
        arm1 = min(directions, key=lambda d: _angle_gap(d, a1))
        # Both ends of the arc must stop on two different measured arms, and the
        # arc must sweep the angle between them, not more and not less.
        if _angle_gap(a0, arm0) > ARC_ARM_DEG or _angle_gap(a1, arm1) > ARC_ARM_DEG:
            continue
        degrees = _angle_gap(arm0, arm1)
        if span > 180:
            degrees = 360 - degrees
        if degrees < MIN_ARC_SPAN_DEG or abs(degrees - span) > ARC_ARM_DEG:
            continue
        ends = [arm0, arm1]
        arm_len = max(MIN_EDGE_PX, 3 * radius)
        arcs.append({
            "vertex": (round(vertex[0]), round(vertex[1])),
            "arms": [(round(vertex[0] + math.cos(math.radians(d)) * arm_len),
                      round(vertex[1] + math.sin(math.radians(d)) * arm_len)) for d in ends],
            "radius": radius,
            "degrees": round(degrees, 1),
        })
    return arcs


def _piece_angle(pts: np.ndarray) -> float:
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    return math.degrees(math.atan2(vt[0][1], vt[0][0]))


def detect_parallel_chevrons(binary: np.ndarray, shapes: list[dict], stroke_half: float, text_boxes=()) -> list[dict]:
    """Chevrons (">", ">>") drawn on the interior of a line.

    Erasing the line splits each chevron into two short oblique strokes, one on
    each side, that touch the line at the same place and mirror each other.
    """
    hosts = [s for s in shapes if s["type"] == "line" and s.get("role") != "tick"
             and math.dist(*s["points"]) >= 2 * MIN_EDGE_PX]
    if not hosts:
        return []
    residual = residual_ink(binary, long_edges(shapes), stroke_half, text_boxes)
    thickness = max(3, int(round(2 * stroke_half + 3)))
    found: list[dict] = []
    for index, host in enumerate(hosts):
        (x1, y1), (x2, y2) = map(lambda p: tuple(map(float, p)), host["points"])
        length = math.dist((x1, y1), (x2, y2))
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        band = np.zeros_like(residual)
        cv2.line(band, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2 * MAX_CHEVRON_PX)
        local = cv2.bitwise_and(residual, band)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
        pieces = []
        for label in range(1, count):
            if stats[label][cv2.CC_STAT_AREA] < 4:
                continue
            ys, xs = np.nonzero(labels == label)
            pts = np.column_stack([xs, ys]).astype(float)
            t = (pts[:, 0] - x1) * ux + (pts[:, 1] - y1) * uy
            n = (pts[:, 0] - x1) * -uy + (pts[:, 1] - y1) * ux
            if t.min() < CHEVRON_END_MARGIN * length or t.max() > (1 - CHEVRON_END_MARGIN) * length:
                continue
            near = np.abs(n).argmin()
            if abs(n[near]) > thickness / 2 + 2:
                continue
            extent = math.hypot(t.max() - t.min(), n.max() - n.min())
            if not MIN_CHEVRON_PX <= extent <= MAX_CHEVRON_PX:
                continue
            if len(pts) > extent * (2 * stroke_half + 3):
                continue
            far = np.abs(n).argmax()
            offset = (_piece_angle(pts) - math.degrees(math.atan2(uy, ux))) % 180
            offset = min(offset, 180 - offset)
            if not MIN_CHEVRON_DEG <= offset <= MAX_CHEVRON_DEG:
                continue
            pieces.append((float(t[near]), math.copysign(1, n[far]), math.copysign(1, t[far] - t[near]), offset))
        pairs = sorted(
            (abs(a[0] - b[0]), i, j)
            for i, a in enumerate(pieces) for j, b in enumerate(pieces) if i < j
            and a[1] != b[1] and a[2] == b[2] and abs(a[0] - b[0]) <= thickness + 2
            and abs(a[3] - b[3]) <= CHEVRON_MIRROR_DEG
        )
        used: set[int] = set()
        chevrons: list[tuple[float, float]] = []
        for _, i, j in pairs:
            if i in used or j in used:
                continue
            used.update((i, j))
            chevrons.append(((pieces[i][0] + pieces[j][0]) / 2, -pieces[i][2]))
        if not chevrons:
            continue
        chevrons.sort()
        groups: list[list[tuple[float, float]]] = []
        for chevron in chevrons:
            if groups and chevron[0] - groups[-1][-1][0] <= CHEVRON_GROUP_PX * MAX_CHEVRON_PX / 2 and chevron[1] == groups[-1][-1][1]:
                groups[-1].append(chevron)
            else:
                groups.append([chevron])
        group = max(groups, key=len)
        t_mid = sum(c[0] for c in group) / len(group)
        found.append({
            "host": [tuple(map(int, host["points"][0])), tuple(map(int, host["points"][1]))],
            "count": len(group),
            "pointing": (ux * group[0][1], uy * group[0][1]),
            "position": (round(x1 + ux * t_mid), round(y1 + uy * t_mid)),
        })
    return found
