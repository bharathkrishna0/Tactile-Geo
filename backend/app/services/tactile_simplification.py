from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.models.geometry import ConfidenceLevel, DetectedElement, ElementRelationship, GeometryType, SemanticGeometry
from app.services.tactile_rules import TACTILE_RULES

NOISE_CONFIDENCE_THRESHOLD = 0.3
TEXT_LABEL_ALWAYS_KEEP = True


def _best_confidence_level(levels: list) -> ConfidenceLevel:
    rank = {ConfidenceLevel.HIGH: 2, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 0}
    return max(levels, key=lambda cl: rank[cl])


@dataclass
class SimplificationAction:
    element_id: str
    action: str
    detail: str


@dataclass
class SimplifiedGeometry:
    elements: list[DetectedElement] = field(default_factory=list)
    relationships: list[ElementRelationship] = field(default_factory=list)
    actions: list[SimplificationAction] = field(default_factory=list)
    removed_count: int = 0
    merged_count: int = 0
    explanations: list[str] = field(default_factory=list)


def simplify_geometry(semantic: SemanticGeometry) -> SimplifiedGeometry:
    # Pass 1: remove low-confidence noise (labels always kept).
    kept: list[DetectedElement] = []
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    removed_count = 0

    for element in semantic.elements:
        if element.type is GeometryType.TEXT_LABEL and TEXT_LABEL_ALWAYS_KEEP:
            kept.append(element)
            continue
        if element.confidence < NOISE_CONFIDENCE_THRESHOLD:
            removed_count += 1
            actions.append(SimplificationAction(
                element_id=element.id,
                action="removed_noise",
                detail=f"Element confident at {element.confidence:.2f} treated as background noise.",
            ))
            explanations.append(f"Removed low-confidence element {element.id} as noise.")
            continue
        if element.confidence < 0.5:
            actions.append(SimplificationAction(
                element_id=element.id,
                action="uncertain_kept",
                detail="Low-confidence element preserved pending teacher review.",
            ))
        kept.append(element)

    # Pass 2: merge duplicate collinear line segments into a single segment.
    kept, merge_actions, merge_explanations, merged_count = _merge_collinear_segments(kept)
    actions.extend(merge_actions)
    explanations.extend(merge_explanations)

    # Pass 3: simplify contour polygons by dropping redundant collinear vertices.
    kept, simplify_actions, simplify_explanations = _simplify_contours(kept)
    actions.extend(simplify_actions)
    explanations.extend(simplify_explanations)

    kept_ids = {element.id for element in kept}
    kept_relationships = [
        rel for rel in semantic.relationships
        if set(rel.element_ids).issubset(kept_ids)
    ]

    return SimplifiedGeometry(
        elements=kept,
        relationships=kept_relationships,
        actions=actions,
        removed_count=removed_count,
        merged_count=merged_count,
        explanations=explanations,
    )


def _merge_collinear_segments(elements: list[DetectedElement]) -> tuple[list[DetectedElement], list[SimplificationAction], list[str], int]:
    segments = [e for e in elements if e.type is GeometryType.LINE_SEGMENT]
    others = [e for e in elements if e.type is not GeometryType.LINE_SEGMENT]
    kept: list[DetectedElement] = []
    used: set[int] = set()
    merged_count = 0
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    tolerance = TACTILE_RULES.collinear_merge_tolerance_px
    angle_tolerance = 3.0  # degrees

    for i, a in enumerate(segments):
        if i in used:
            continue
        used.add(i)
        a0, a1 = a.geometry["start"], a.geometry["end"]
        a_angle = _line_angle_deg(a0, a1)
        merged_into_a = [a]
        for j, b in enumerate(segments):
            if j == i or j in used:
                continue
            b0, b1 = b.geometry["start"], b.geometry["end"]
            b_angle = _line_angle_deg(b0, b1)
            if abs(_angle_diff(a_angle, b_angle)) > angle_tolerance:
                continue
            # Collinear: projected distance from b endpoints to line a is tiny.
            if not (_on_extended_line(a0, a1, b0, tolerance) and _on_extended_line(a0, a1, b1, tolerance)):
                continue
            if not _overlap_or_touch(a0, a1, b0, b1):
                continue
            # Keep exact duplicates untouched; only merge genuinely distinct
            # collinear fragments (their endpooints differ).
            if _exact_duplicate_segment((a0, a1), (b0, b1)):
                continue
            merged_into_a.append(b)
            used.add(j)
        if len(merged_into_a) > 1:
            endpoints = []
            for seg in merged_into_a:
                endpoints.extend([seg.geometry["start"], seg.geometry["end"]])
            start, end = _spanning_endpoints(endpoints)
            if start != a0 or end != a1:
                merged = DetectedElement(
                    id=a.id,
                    type=GeometryType.LINE_SEGMENT,
                    geometry={"start": start, "end": end, "length": round(math.hypot(end[0] - start[0], end[1] - start[1]), 2)},
                    confidence=min(0.99, max(seg.confidence for seg in merged_into_a)),
                    confidence_level=_best_confidence_level([seg.confidence_level for seg in merged_into_a]),
                    needs_review=any(seg.needs_review for seg in merged_into_a),
                    source=a.source,
                    bbox=(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]), abs(end[1] - start[1])),
                    semantic_properties={"merged_from": [seg.id for seg in merged_into_a], "length": round(math.hypot(end[0] - start[0], end[1] - start[1]), 2)},
                )
                merged_count += len(merged_into_a) - 1
                actions.append(SimplificationAction(
                    element_id=a.id,
                    action="merged_collinear",
                    detail=f"Merged {len(merged_into_a)} collinear segments into one.",
                ))
                explanations.append(f"Merged {len(merged_into_a)} collinear segments into {a.id}.")
                kept.append(merged)
                continue
        kept.append(a)

    return kept + others, actions, explanations, merged_count


def _simplify_contours(elements: list[DetectedElement]) -> tuple[list[DetectedElement], list[SimplificationAction], list[str]]:
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    kept: list[DetectedElement] = []
    tolerance = TACTILE_RULES.collinear_merge_tolerance_px

    for element in elements:
        if element.type not in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON):
            kept.append(element)
            continue
        points = element.geometry.get("points", [])
        if len(points) < 4:
            kept.append(element)
            continue
        simplified = _drop_redundant_vertices(points, tolerance)
        if len(simplified) < len(points):
            new_area = _polygon_area(simplified)
            new_element = DetectedElement(
                id=element.id,
                type=element.type,
                geometry={"points": simplified, "area": round(new_area, 2)},
                confidence=element.confidence,
                confidence_level=element.confidence_level,
                needs_review=element.needs_review,
                source=element.source,
                bbox=element.bbox,
                semantic_properties={**element.semantic_properties},
                associated_label_id=element.associated_label_id,
            )
            kept.append(new_element)
            actions.append(SimplificationAction(
                element_id=element.id,
                action="simplified_contour",
                detail=f"Removed {len(points) - len(simplified)} redundant collinear vertices.",
            ))
            explanations.append(f"Simplified contour {element.id} removed redundant collinear vertices.")
        else:
            kept.append(element)
    return kept, actions, explanations


# --- geometry helpers -------------------------------------------------------

def _line_angle_deg(start: tuple, end: tuple) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _point_to_segment_distance(point: tuple, start: tuple, end: tuple) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def _on_extended_line(a0: tuple, a1: tuple, point: tuple, tolerance: float) -> bool:
    # Distance from point to the (extended) line through a0-a1.
    dx, dy = a1[0] - a0[0], a1[1] - a0[1]
    denom = math.hypot(dx, dy)
    if denom == 0:
        return math.hypot(point[0] - a0[0], point[1] - a0[1]) <= tolerance
    distance = abs(dy * (point[0] - a0[0]) - dx * (point[1] - a0[1])) / denom
    return distance <= tolerance


def _segments_overlap(a0: tuple, a1: tuple, b0: tuple, b1: tuple, tolerance: float = 0.0) -> bool:
    def project(t: tuple) -> float:
        dx, dy = a1[0] - a0[0], a1[1] - a0[1]
        return (t[0] - a0[0]) * dx + (t[1] - a0[1]) * dy

    pa0, pa1 = 0.0, project(a1)
    pb0, pb1 = project(b0), project(b1)
    if pa1 < pa0:
        pa0, pa1 = pa1, pa0
    lo = max(min(pb0, pb1), pa0)
    hi = min(max(pb0, pb1), pa1)
    # Parametric gap must be within tolerance-ish; use raw overlap for touch.
    return hi >= lo - tolerance


def _overlap_or_touch(a0: tuple, a1: tuple, b0: tuple, b1: tuple) -> bool:
    return _segments_overlap(a0, a1, b0, b1, tolerance=1.0)


def _exact_duplicate_segment(a: tuple, b: tuple) -> bool:
    return (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0])


def _spanning_endpoints(endpoints: list[tuple]) -> tuple[tuple, tuple]:
    anchor = endpoints[0]
    far = max(endpoints, key=lambda p: math.hypot(p[0] - anchor[0], p[1] - anchor[1]))
    farthest_from_far = max(endpoints, key=lambda p: math.hypot(p[0] - far[0], p[1] - far[1]))
    return tuple(far), tuple(farthest_from_far)


def _drop_redundant_vertices(points: list[tuple], tolerance: float) -> list[tuple]:
    n = len(points)
    kept = [points[0]]
    for i in range(1, n):
        prev = kept[-1]
        nxt = points[(i + 1) % n]
        distance = _point_to_segment_distance(points[i], prev, nxt)
        if distance <= tolerance and n - len(kept) > 0:
            # Avoid over-collapsing: always keep at least 3 vertices.
            if len(kept) + (n - i - 1) >= 3:
                continue
        kept.append(points[i])
    if len(kept) < 3:
        kept = points[:3]
    return kept


def _polygon_area(points: list[tuple]) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))))