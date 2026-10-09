"""Relationship inference over semantic geometry elements.

Operates on :class:`app.models.geometry.DetectedElement` instances (points,
lines, segments, circles, polygons) and produces
:class:`app.models.geometry.ElementRelationship` instances with confidence.

Relationships are only emitted when they can be reasonably inferred from the
available coordinates. When evidence is weak the confidence reflects that and
``needs_review`` is set instead of guessing.
"""
from __future__ import annotations

import math

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
)


def _point_distance(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def _line_angle_deg(start: tuple, end: tuple) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _point_to_segment_distance(point: tuple, start: tuple, end: tuple) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return _point_distance(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    px = start[0] + ratio * dx
    py = start[1] + ratio * dy
    return _point_distance(point, (px, py))


STRAIGHT_TYPES = (GeometryType.LINE_SEGMENT, GeometryType.RAY, GeometryType.AXES, GeometryType.ARROW)


def _is_line_like(element: DetectedElement) -> bool:
    return element.type in STRAIGHT_TYPES and "start" in element.geometry and "end" in element.geometry


def _angles(a: DetectedElement, b: DetectedElement) -> tuple[float, float] | None:
    if not (_is_line_like(a) and _is_line_like(b)):
        return None
    return _line_angle_deg(a.geometry["start"], a.geometry["end"]), _line_angle_deg(b.geometry["start"], b.geometry["end"])


def infer_relationships(elements: list[DetectedElement]) -> list[ElementRelationship]:
    relationships: list[ElementRelationship] = []
    added: set[tuple] = set()

    def _add(rel: ElementRelationship) -> None:
        key = (rel.type.value, tuple(sorted(rel.element_ids)))
        if key in added:
            return
        added.add(key)
        relationships.append(rel)

    points = [e for e in elements if e.type is GeometryType.POINT]
    all_lines = [e for e in elements if _is_line_like(e)]
    ticks = [e for e in all_lines if e.semantic_properties.get("role") == "tick"]
    lines = [e for e in all_lines if e.semantic_properties.get("role") != "tick"]
    circles = [e for e in elements if e.type is GeometryType.CIRCLE]
    polygons = [e for e in elements if e.type in POLYGON_TYPES and len(e.geometry.get("points") or []) >= 3]

    _infer_line_pairs(_add, lines, ticks)
    _infer_line_polygon(_add, lines, polygons)
    _infer_point_line(_add, points, lines)
    _infer_point_circle(_add, points, circles)
    _infer_circle_center(_add, circles, points)
    _infer_point_vertex(_add, points, polygons)
    _infer_ticks(_add, ticks, lines)
    _infer_equal_lengths(_add, ticks, lines)

    return relationships


POLYGON_TYPES = (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON)
ANGLE_TOLERANCE_DEG = 2.0
TOUCH_PX = 10.0
# Parallel lines worth stating face each other: their projections overlap and
# the gap between them is not much larger than the shorter line.
PARALLEL_MIN_OVERLAP = 0.3
PARALLEL_MAX_GAP_RATIO = 1.5
VERTEX_PX = 8.0


def _length(start, end) -> float:
    return _point_distance(start, end)


# Three or more facing lines of one orientation and similar length are a grid,
# table or hatching: their pairwise parallels and crossings are background.
GRID_MIN_FAMILY = 3
GRID_LENGTH_RATIO = 0.8


def grid_like_ids(lines: list[DetectedElement]) -> set[str]:
    straight = [line for line in lines if _is_line_like(line)]
    out: set[str] = set()
    for a in straight:
        a0, a1 = _line_endpoints(a)
        length_a = _length(a0, a1)
        if length_a == 0:
            continue
        family = 0
        for b in straight:
            if b is a:
                continue
            b0, b1 = _line_endpoints(b)
            length_b = _length(b0, b1)
            diff = abs((_line_angle_deg(a0, a1) - _line_angle_deg(b0, b1) + 90) % 180 - 90)
            if (diff < ANGLE_TOLERANCE_DEG and min(length_a, length_b) >= GRID_LENGTH_RATIO * max(length_a, length_b)
                    and _parallel_pair_faces(a, b)):
                family += 1
        if family + 1 >= GRID_MIN_FAMILY:
            out.add(a.id)
    return out


# A line met at right angles by several others is a table or grid rule, unless
# tick marks show it is a scale (an axis or number line).
LATTICE_MIN_CROSSINGS = 3
SCALE_MIN_TICKS = 2


def _ticks_on(line: DetectedElement, ticks: list[DetectedElement]) -> int:
    start, end = _line_endpoints(line)
    count = 0
    for tick in ticks:
        t0, t1 = _line_endpoints(tick)
        mid = ((t0[0] + t1[0]) / 2, (t0[1] + t1[1]) / 2)
        count += _point_to_segment_distance(mid, start, end) <= TOUCH_PX
    return count


def background_line_ids(lines: list[DetectedElement], ticks: list[DetectedElement] = ()) -> set[str]:
    straight = [line for line in lines if _is_line_like(line)]
    out = set(grid_like_ids(straight))
    for a in straight:
        a0, a1 = _line_endpoints(a)
        crossings = 0
        for b in straight:
            if b is a:
                continue
            b0, b1 = _line_endpoints(b)
            diff = abs((_line_angle_deg(a0, a1) - _line_angle_deg(b0, b1) + 90) % 180 - 90)
            if abs(diff - 90) < ANGLE_TOLERANCE_DEG and (_segments_cross(a, b) or _segments_intersect_or_connect(a, b)):
                crossings += 1
        if crossings >= LATTICE_MIN_CROSSINGS:
            out.add(a.id)
    return {line_id for line_id in out
            if _ticks_on(next(line for line in straight if line.id == line_id), list(ticks)) < SCALE_MIN_TICKS}


def _parallel_pair_faces(a: DetectedElement, b: DetectedElement) -> bool:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)
    length_a = _length(a0, a1)
    if length_a == 0:
        return False
    ux, uy = (a1[0] - a0[0]) / length_a, (a1[1] - a0[1]) / length_a
    tb = sorted(((p[0] - a0[0]) * ux + (p[1] - a0[1]) * uy) for p in (b0, b1))
    overlap = min(length_a, tb[1]) - max(0.0, tb[0])
    shorter = min(length_a, _length(b0, b1))
    if shorter == 0 or overlap < PARALLEL_MIN_OVERLAP * shorter:
        return False
    mid = ((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2)
    gap = abs((mid[0] - a0[0]) * -uy + (mid[1] - a0[1]) * ux)
    return gap <= PARALLEL_MAX_GAP_RATIO * shorter


def _sides(polygon: DetectedElement) -> list[tuple[tuple, tuple]]:
    points = [tuple(p) for p in polygon.geometry["points"]]
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _infer_line_polygon(add, lines: list[DetectedElement], polygons: list[DetectedElement]) -> None:
    """A line that meets a polygon side at a right angle (an altitude, a perpendicular)."""
    for line in lines:
        start, end = _line_endpoints(line)
        line_angle = _line_angle_deg(start, end)
        for polygon in polygons:
            for side in _sides(polygon):
                diff = abs((line_angle - _line_angle_deg(*side) + 90) % 180 - 90)
                if abs(diff - 90) >= ANGLE_TOLERANCE_DEG:
                    continue
                touches = any(_point_to_segment_distance(p, *side) < TOUCH_PX for p in (start, end))
                if touches and min(_length(*side), _length(start, end)) > TOUCH_PX:
                    add(ElementRelationship(
                        id=f"rel_{line.id}_{polygon.id}_perp",
                        type=RelationshipType.PERPENDICULAR_LINES,
                        element_ids=[line.id, polygon.id],
                        confidence=0.8,
                        confidence_level=ConfidenceLevel.HIGH,
                        needs_review=False,
                        explanation=f"Line {line.id} meets a side of {polygon.id} at a right angle.",
                    ))
                    break


def _infer_point_vertex(add, points: list[DetectedElement], polygons: list[DetectedElement]) -> None:
    for point in points:
        pos = tuple(point.geometry["position"])
        for polygon in polygons:
            if any(_point_distance(pos, tuple(v)) <= VERTEX_PX for v in polygon.geometry["points"]):
                add(ElementRelationship(
                    id=f"rel_{point.id}_{polygon.id}_vertex",
                    type=RelationshipType.VERTEX_OF,
                    element_ids=[point.id, polygon.id],
                    confidence=0.9,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Point {point.id} is a vertex of {polygon.id}.",
                ))


def _infer_ticks(add, ticks: list[DetectedElement], lines: list[DetectedElement]) -> None:
    """Each recovered tick lies on the line it was drawn across."""
    for tick in ticks:
        t0, t1 = _line_endpoints(tick)
        mid = ((t0[0] + t1[0]) / 2, (t0[1] + t1[1]) / 2)
        host = min(lines, key=lambda line: _point_to_segment_distance(mid, *_line_endpoints(line)), default=None)
        if host is None or _point_to_segment_distance(mid, *_line_endpoints(host)) > TOUCH_PX:
            continue
        add(ElementRelationship(
            id=f"rel_{tick.id}_{host.id}_lies_on",
            type=RelationshipType.LIES_ON,
            element_ids=[tick.id, host.id],
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            needs_review=False,
            explanation=f"Tick {tick.id} is drawn across line {host.id}.",
        ))


# Equal-length hash marks: one to three ticks bunched around the middle of a
# segment. Axis ticks are spread along the whole line, so they never qualify.
MAX_HASH_TICKS = 3
HASH_MIDDLE_FRACTION = 0.15
MAX_HASH_SPREAD_PX = 24.0
EQUAL_LENGTH_RATIO = 0.88
# Drawn marks state the relation; the measurement only has to agree with it.
MARKED_PARALLEL_TOLERANCE_DEG = 5.0


def _hash_count(line: DetectedElement, ticks: list[DetectedElement]) -> int:
    start, end = _line_endpoints(line)
    length = _length(start, end)
    if length == 0:
        return 0
    on_line = []
    for tick in ticks:
        t0, t1 = _line_endpoints(tick)
        mid = ((t0[0] + t1[0]) / 2, (t0[1] + t1[1]) / 2)
        if _point_to_segment_distance(mid, start, end) > TOUCH_PX:
            continue
        on_line.append(((mid[0] - start[0]) * (end[0] - start[0]) + (mid[1] - start[1]) * (end[1] - start[1])) / length)
    if not 1 <= len(on_line) <= MAX_HASH_TICKS:
        return 0
    if max(on_line) - min(on_line) > MAX_HASH_SPREAD_PX:
        return 0
    if any(abs(t - length / 2) > HASH_MIDDLE_FRACTION * length for t in on_line):
        return 0
    return len(on_line)


def _infer_equal_lengths(add, ticks: list[DetectedElement], lines: list[DetectedElement]) -> None:
    """Segments carrying the same number of hash marks, when their measured lengths agree."""
    marked = [(line, count) for line in lines if (count := _hash_count(line, ticks))]
    for i, (a, count_a) in enumerate(marked):
        for b, count_b in marked[i + 1:]:
            if count_a != count_b:
                continue
            length_a, length_b = _length(*_line_endpoints(a)), _length(*_line_endpoints(b))
            if min(length_a, length_b) < EQUAL_LENGTH_RATIO * max(length_a, length_b):
                continue
            add(ElementRelationship(
                id=f"rel_{a.id}_{b.id}_equal",
                type=RelationshipType.EQUAL_LENGTH,
                element_ids=[a.id, b.id],
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                needs_review=False,
                explanation=(f"Segments {a.id} and {b.id} both carry {count_a} hash mark(s); "
                             f"measured lengths {length_a:.0f} and {length_b:.0f} px."),
            ))


def _infer_line_pairs(add, lines: list[DetectedElement], ticks: list[DetectedElement] = ()) -> None:
    grid = background_line_ids(lines, ticks)
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            angles = _angles(a, b)
            if angles is None:
                continue
            a_angle, b_angle = angles
            diff = abs((a_angle - b_angle + 90) % 180 - 90)
            background = a.id in grid and b.id in grid
            on_background = a.id in grid or b.id in grid
            marks = a.semantic_properties.get("parallel_marks")
            if marks and marks == b.semantic_properties.get("parallel_marks") and diff <= MARKED_PARALLEL_TOLERANCE_DEG:
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_parallel",
                    type=RelationshipType.PARALLEL_LINES,
                    element_ids=[a.id, b.id],
                    confidence=0.95,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=(f"Lines {a.id} and {b.id} carry matching parallel marks ({marks}) "
                                 f"and measure {diff:.1f} degrees apart."),
                ))

            if diff < ANGLE_TOLERANCE_DEG and not on_background and _parallel_pair_faces(a, b):
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_parallel",
                    type=RelationshipType.PARALLEL_LINES,
                    element_ids=[a.id, b.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Lines {a.id} and {b.id} share the same orientation.",
                ))
            elif (abs(diff - 90) < ANGLE_TOLERANCE_DEG
                  and (_segments_share_endpoint(a, b) if background
                       else _segments_intersect_or_connect(a, b) or _segments_cross(a, b))):
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_perp",
                    type=RelationshipType.PERPENDICULAR_LINES,
                    element_ids=[a.id, b.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Lines {a.id} and {b.id} meet at a right angle.",
                ))

            if _segments_cross(a, b) and not on_background and not _segments_intersect_or_connect(a, b):
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_intersect",
                    type=RelationshipType.INTERSECTS,
                    element_ids=[a.id, b.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Segments {a.id} and {b.id} cross.",
                ))
            elif _segments_intersect_or_connect(a, b) and not on_background:
                if _segments_share_endpoint(a, b):
                    add(ElementRelationship(
                        id=f"rel_{a.id}_{b.id}_conn",
                        type=RelationshipType.CONNECTED_LINES,
                        element_ids=[a.id, b.id],
                        confidence=0.9,
                        confidence_level=ConfidenceLevel.HIGH,
                        needs_review=False,
                        explanation=f"Lines {a.id} and {b.id} share an endpoint.",
                    ))


def _line_endpoints(a: DetectedElement) -> tuple[tuple, tuple]:
    return a.geometry["start"], a.geometry["end"]


def _endpoint_near(a: tuple, b: tuple, tolerance: float = 10.0) -> bool:
    return _point_distance(a, b) <= tolerance


def _segments_share_endpoint(a: DetectedElement, b: DetectedElement) -> bool:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)
    return (_endpoint_near(a0, b0) or _endpoint_near(a0, b1)
            or _endpoint_near(a1, b0) or _endpoint_near(a1, b1))


def _segments_cross(a: DetectedElement, b: DetectedElement) -> bool:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a0, a1, b0) * orient(a0, a1, b1) < 0 and orient(b0, b1, a0) * orient(b0, b1, a1) < 0


def _segments_intersect_or_connect(a: DetectedElement, b: DetectedElement) -> bool:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)
    for p in (a0, a1):
        if _point_to_segment_distance(p, b0, b1) < 10.0:
            return True
    for p in (b0, b1):
        if _point_to_segment_distance(p, a0, a1) < 10.0:
            return True
    return False


def _infer_point_line(add, points: list[DetectedElement], lines: list[DetectedElement]) -> None:
    for point in points:
        for line in lines:
            start, end = line.geometry["start"], line.geometry["end"]
            pos = tuple(point.geometry["position"])
            distance = _point_to_segment_distance(pos, start, end)
            if distance <= 8.0:
                add(ElementRelationship(
                    id=f"rel_{point.id}_{line.id}_on_line",
                    type=RelationshipType.POINT_ON_LINE,
                    element_ids=[point.id, line.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Point {point.id} lies on line {line.id}.",
                ))
            # endpoint_of for segments whose endpoint coincides with a point
            for ep in (start, end):
                if _endpoint_near(tuple(ep), pos, tolerance=8.0):
                    add(ElementRelationship(
                        id=f"rel_{point.id}_{line.id}_endpoint",
                        type=RelationshipType.ENDPOINT_OF,
                        element_ids=[point.id, line.id],
                        confidence=0.9,
                        confidence_level=ConfidenceLevel.HIGH,
                        needs_review=False,
                        explanation=f"Point {point.id} is an endpoint of segment {line.id}.",
                    ))


def _infer_point_circle(add, points: list[DetectedElement], circles: list[DetectedElement]) -> None:
    for point in points:
        for circle in circles:
            center = tuple(circle.geometry["center"])
            radius = float(circle.geometry["radius"])
            pos = tuple(point.geometry["position"])
            distance = _point_distance(pos, center)
            if abs(distance - radius) <= 8.0:
                add(ElementRelationship(
                    id=f"rel_{point.id}_{circle.id}_on_circle",
                    type=RelationshipType.POINT_ON_CIRCLE,
                    element_ids=[point.id, circle.id],
                    confidence=0.75,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    needs_review=True,
                    explanation=f"Point {point.id} lies near the circumference of {circle.id}.",
                ))
            if distance <= radius:
                add(ElementRelationship(
                    id=f"rel_{point.id}_{circle.id}_inside",
                    type=RelationshipType.POINT_INSIDE_SHAPE,
                    element_ids=[point.id, circle.id],
                    confidence=0.8,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Point {point.id} lies inside {circle.id}.",
                ))


def _infer_circle_center(add, circles: list[DetectedElement], points: list[DetectedElement]) -> None:
    for circle in circles:
        center = tuple(circle.geometry["center"])
        for point in points:
            pos = tuple(point.geometry["position"])
            if _endpoint_near(pos, center, tolerance=8.0):
                add(ElementRelationship(
                    id=f"rel_{circle.id}_{point.id}_center",
                    type=RelationshipType.CENTER_OF,
                    element_ids=[point.id, circle.id],
                    confidence=0.9,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Point {point.id} is the center of circle {circle.id}.",
                ))