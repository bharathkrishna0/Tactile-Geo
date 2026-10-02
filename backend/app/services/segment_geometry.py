"""Shared pure-geometry helpers for line-segment reasoning.

Extracted so that vectorization (raw Hough output) and tactile simplification
(semantic elements) apply exactly the same collinearity test. Keeping a single
implementation is what makes "normalize/deduplicate segments before angle
generation" and "merge collinear segments" agree with each other.

All helpers work on plain ``(x, y)`` tuples so they can be unit tested without
any CV or model dependency.
"""
from __future__ import annotations

import math

Point = tuple[float, float]
Segment = tuple[Point, Point]

# A segment is collinear with another when it lies within this perpendicular
# distance of the same supporting line.
DEFAULT_DISTANCE_TOLERANCE_PX = 8.0
# Orientation tolerance for "same supporting line".
DEFAULT_ANGLE_TOLERANCE_DEG = 3.0


def line_angle_deg(start: Point, end: Point) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def segment_length(segment: Segment) -> float:
    (x1, y1), (x2, y2) = segment
    return math.hypot(x2 - x1, y2 - y1)


def point_to_line_distance(a0: Point, a1: Point, point: Point) -> float:
    """Perpendicular distance from ``point`` to the infinite line through a0-a1."""
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    denom = math.hypot(dx, dy)
    if denom == 0:
        return math.hypot(point[0] - a0[0], point[1] - a0[1])
    return abs(dy * (point[0] - a0[0]) - dx * (point[1] - a0[1])) / denom


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def segments_overlap_or_touch(a0: Point, a1: Point, b0: Point, b1: Point, tolerance: float = 1.0) -> bool:
    """True when the two segments share at least one point, or are within ``tolerance``."""

    def project(point: Point) -> float:
        return (point[0] - a0[0]) * (a1[0] - a0[0]) + (point[1] - a0[1]) * (a1[1] - a0[1])

    pa0, pa1 = 0.0, project(a1)
    pb0, pb1 = project(b0), project(b1)
    if pa1 < pa0:
        pa0, pa1 = pa1, pa0
    lo = max(min(pb0, pb1), pa0)
    hi = min(max(pb0, pb1), pa1)
    return hi >= lo - tolerance


def on_same_supporting_line(base: Segment, candidate: Segment, angle_tolerance_deg: float, distance_tolerance_px: float) -> bool:
    """True when ``candidate`` lies on the same line as ``base`` and touches it.

    Both endpoints of ``candidate`` must sit within ``distance_tolerance_px`` of
    the infinite line through ``base``. Requiring contact prevents two parallel
    but visually distinct lines from being merged into one.
    """
    b0, b1 = base
    c0, c1 = candidate
    if abs(angle_diff(line_angle_deg(b0, b1), line_angle_deg(c0, c1))) > angle_tolerance_deg:
        return False
    if point_to_line_distance(b0, b1, c0) > distance_tolerance_px:
        return False
    if point_to_line_distance(b0, b1, c1) > distance_tolerance_px:
        return False
    return segments_overlap_or_touch(b0, b1, c0, c1, tolerance=distance_tolerance_px)


def spanning_endpoints(points: list[Point]) -> tuple[Point, Point]:
    """The two points furthest apart, used as the endpoints of a merged segment."""
    if len(points) < 2:
        return tuple(points[0]), tuple(points[0])
    anchor = points[0]
    far = max(points, key=lambda p: math.hypot(p[0] - anchor[0], p[1] - anchor[1]))
    farthest = max(points, key=lambda p: math.hypot(p[0] - far[0], p[1] - far[1]))
    return tuple(far), tuple(farthest)


def cluster_collinear(
    segments: list[Segment],
    angle_tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    distance_tolerance_px: float = DEFAULT_DISTANCE_TOLERANCE_PX,
) -> list[list[int]]:
    """Group indices of ``segments`` that describe the same physical line.

    Deterministic: the first unassigned segment of a group becomes the reference
    for that group's collinearity test, so repeated runs over the same input
    always produce the same grouping.
    """
    groups: list[list[int]] = []
    assigned = [False] * len(segments)
    for i, base in enumerate(segments):
        if assigned[i]:
            continue
        assigned[i] = True
        group = [i]
        for j in range(i + 1, len(segments)):
            if assigned[j]:
                continue
            if on_same_supporting_line(base, segments[j], angle_tolerance_deg, distance_tolerance_px):
                assigned[j] = True
                group.append(j)
        groups.append(group)
    return groups


def merge_collinear(
    segments: list[Segment],
    angle_tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    distance_tolerance_px: float = DEFAULT_DISTANCE_TOLERANCE_PX,
) -> list[Segment]:
    """Collapse collinear/duplicate segments to one spanning segment per line."""
    merged: list[Segment] = []
    for group in cluster_collinear(segments, angle_tolerance_deg, distance_tolerance_px):
        points: list[Point] = []
        for index in group:
            points.extend(segments[index])
        start, end = spanning_endpoints(points)
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 1.0:
            # Degenerate cluster (a single sub-pixel fragment): keep it as-is so
            # normalization never deletes input.
            merged.append(segments[group[0]])
            continue
        merged.append((start, end))
    return merged


def deduplicate_segments(
    segments: list[Segment],
    endpoint_tolerance_px: float = 2.0,
) -> list[Segment]:
    """Drop segments that repeat an already-seen segment (same direction and span)."""
    kept: list[Segment] = []
    for segment in segments:
        if segment_length(segment) < 1.0:
            continue
        duplicate = False
        for existing in kept:
            if segment_length(existing) < 1.0:
                continue
            same = (
                point_to_segment_distance(segment[0], existing[0], existing[1]) <= endpoint_tolerance_px
                and point_to_segment_distance(segment[1], existing[0], existing[1]) <= endpoint_tolerance_px
            )
            flipped = (
                point_to_segment_distance(segment[0], existing[1], existing[0]) <= endpoint_tolerance_px
                and point_to_segment_distance(segment[1], existing[1], existing[0]) <= endpoint_tolerance_px
            )
            if same or flipped:
                duplicate = True
                break
        if not duplicate:
            kept.append(segment)
    return kept


def normalize_segments(
    segments: list[Segment],
    angle_tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    distance_tolerance_px: float = DEFAULT_DISTANCE_TOLERANCE_PX,
) -> list[Segment]:
    """Deduplicate then merge collinear fragments.

    This is the normalization step that must run before angle generation: without
    it, N overlapping fragments of the same physical edge yield O(N^2) candidate
    angles and the semantic graph explodes.
    """
    return merge_collinear(deduplicate_segments(segments), angle_tolerance_deg, distance_tolerance_px)


# --- rectangle distances ----------------------------------------------------
# Braille is embossed as a rectangular cell block, so clearance is measured from
# that block rather than from the label's centre point.
Rect = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


def rect_around(centre: Point, width: float, height: float) -> Rect:
    return (centre[0] - width / 2, centre[1] - height / 2, centre[0] + width / 2, centre[1] + height / 2)


def point_to_rect_distance(point: Point, rect: Rect) -> float:
    dx = max(rect[0] - point[0], 0.0, point[0] - rect[2])
    dy = max(rect[1] - point[1], 0.0, point[1] - rect[3])
    return math.hypot(dx, dy)


def _segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(b0, b1, a0), orient(b0, b1, a1)
    d3, d4 = orient(a0, a1, b0), orient(a0, a1, b1)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def segment_to_rect_distance(start: Point, end: Point, rect: Rect) -> float:
    if point_to_rect_distance(start, rect) == 0 or point_to_rect_distance(end, rect) == 0:
        return 0.0
    corners = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    if any(_segments_intersect(start, end, c0, c1) for c0, c1 in edges):
        return 0.0
    return min(
        point_to_rect_distance(start, rect),
        point_to_rect_distance(end, rect),
        *(point_to_segment_distance(corner, start, end) for corner in corners),
    )


def rect_gap(a: Rect, b: Rect) -> float:
    dx = max(b[0] - a[2], 0.0, a[0] - b[2])
    dy = max(b[1] - a[3], 0.0, a[1] - b[3])
    return math.hypot(dx, dy)
