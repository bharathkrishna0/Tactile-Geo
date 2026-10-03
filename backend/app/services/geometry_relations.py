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
    classify_confidence,
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
    lines = [e for e in elements if _is_line_like(e)]
    circles = [e for e in elements if e.type is GeometryType.CIRCLE]

    _infer_line_pairs(_add, lines)
    _infer_point_line(_add, points, lines)
    _infer_point_circle(_add, points, circles)
    _infer_circle_center(_add, circles, points)

    return relationships


def _infer_line_pairs(add, lines: list[DetectedElement]) -> None:
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            angles = _angles(a, b)
            if angles is None:
                continue
            a_angle, b_angle = angles
            diff = abs((a_angle - b_angle + 90) % 180 - 90)

            if diff < 2.0:
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_parallel",
                    type=RelationshipType.PARALLEL_LINES,
                    element_ids=[a.id, b.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Lines {a.id} and {b.id} share the same orientation.",
                ))
            elif abs(diff - 90) < 2.0:
                add(ElementRelationship(
                    id=f"rel_{a.id}_{b.id}_perp",
                    type=RelationshipType.PERPENDICULAR_LINES,
                    element_ids=[a.id, b.id],
                    confidence=0.85,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    explanation=f"Lines {a.id} and {b.id} meet at a right angle.",
                ))

            if _segments_intersect_or_connect(a, b):
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
                else:
                    add(ElementRelationship(
                        id=f"rel_{a.id}_{b.id}_intersect",
                        type=RelationshipType.INTERSECTS,
                        element_ids=[a.id, b.id],
                        confidence=0.6,
                        confidence_level=ConfidenceLevel.MEDIUM,
                        needs_review=True,
                        explanation=f"Segments {a.id} and {b.id} appear to cross.",
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