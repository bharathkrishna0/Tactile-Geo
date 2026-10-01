from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.models.geometry import ConfidenceLevel, DetectedElement, ElementRelationship, GeometryType, SemanticGeometry
from app.services.segment_geometry import (
    angle_diff as sg_angle_diff,
    line_angle_deg as sg_line_angle_deg,
    point_to_line_distance as sg_point_to_line_distance,
    point_to_segment_distance as sg_point_to_segment_distance,
    segments_overlap_or_touch as sg_segments_overlap_or_touch,
    spanning_endpoints as sg_spanning_endpoints,
)
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
    removed_ids: set[str] = set()
    consumed_ids: set[str] = set()

    for element in semantic.elements:
        if element.type is GeometryType.TEXT_LABEL and TEXT_LABEL_ALWAYS_KEEP:
            kept.append(element)
            continue
        if element.confidence < NOISE_CONFIDENCE_THRESHOLD:
            removed_count += 1
            removed_ids.add(element.id)
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
    kept, merge_actions, merge_explanations, merged_count, merge_consumed = _merge_collinear_segments(kept)
    actions.extend(merge_actions)
    explanations.extend(merge_explanations)
    consumed_ids |= merge_consumed

    # Pass 3: simplify contour polygons by dropping redundant collinear vertices.
    kept, simplify_actions, simplify_explanations = _simplify_contours(kept)
    actions.extend(simplify_actions)
    explanations.extend(simplify_explanations)

    kept_ids = {element.id for element in kept}
    kept_relationships = [
        rel for rel in semantic.relationships
        if set(rel.element_ids).issubset(kept_ids)
    ]

    # Every input element must be either kept, removed as noise, or explicitly
    # consumed by a merge. A silent drop means the teacher is told the diagram was
    # simplified without being able to see what disappeared, so this is a hard
    # invariant rather than a best-effort log.
    unaccounted = [
        element.id for element in semantic.elements
        if element.id not in kept_ids and element.id not in removed_ids and element.id not in consumed_ids
    ]
    for element_id in unaccounted:
        actions.append(SimplificationAction(
            element_id=element_id,
            action="dropped_unexplained",
            detail="Element was not present after simplification and no action explains why.",
        ))
        explanations.append(f"Element {element_id} was dropped without a recorded reason; flagged for teacher review.")
    removed_count += len(unaccounted)

    return SimplifiedGeometry(
        elements=kept,
        relationships=kept_relationships,
        actions=actions,
        removed_count=removed_count,
        merged_count=merged_count,
        explanations=explanations,
    )


def _merge_collinear_segments(elements: list[DetectedElement]) -> tuple[list[DetectedElement], list[SimplificationAction], list[str], int, set[str]]:
    segments = [e for e in elements if e.type is GeometryType.LINE_SEGMENT]
    others = [e for e in elements if e.type is not GeometryType.LINE_SEGMENT]
    kept: list[DetectedElement] = []
    used: set[int] = set()
    merged_count = 0
    consumed: set[str] = set()
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
                    provenance=f"Merged {len(merged_into_a)} collinear segments ({', '.join(seg.id for seg in merged_into_a)}) into one.",
                )
                merged_count += len(merged_into_a) - 1
                consumed.update(seg.id for seg in merged_into_a if seg is not a)
                absorbed_ids = [seg.id for seg in merged_into_a if seg is not a]
                actions.append(SimplificationAction(
                    element_id=a.id,
                    action="merged_collinear",
                    detail=f"Merged {len(merged_into_a)} collinear segments ({', '.join(seg.id for seg in merged_into_a)}) into one.",
                ))
                explanations.append(
                    f"Merged collinear segments {', '.join(absorbed_ids)} into {a.id}."
                )
                kept.append(merged)
                continue
            # The cluster's spanning endpoints are identical to `a`, i.e. the other
            # segments were coincident duplicates. Previously this fell through and
            # `b` was marked used but never kept or reported, so elements vanished
            # with no SimplificationAction. Record the removal explicitly instead.
            absorbed = [seg for seg in merged_into_a if seg is not a]
            for seg in absorbed:
                actions.append(SimplificationAction(
                    element_id=seg.id,
                    action="removed_duplicate",
                    detail=f"Duplicate collinear segment coincident with {a.id}; removed without changing geometry.",
                ))
                explanations.append(f"Removed duplicate segment {seg.id} coincident with {a.id}.")
            consumed.update(seg.id for seg in absorbed)
            merged_count += len(absorbed)
        kept.append(a)

    return kept + others, actions, explanations, merged_count, consumed


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
                provenance=(element.provenance or "") + f" Simplified by removing {len(points) - len(simplified)} redundant collinear vertices.",
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
# Collinearity logic is shared with vectorization via segment_geometry so the
# "normalize before angle generation" and "merge collinear segments" rules cannot
# drift apart. The thin wrappers below keep the existing private call sites.

def _line_angle_deg(start: tuple, end: tuple) -> float:
    return sg_line_angle_deg(start, end)


def _angle_diff(a: float, b: float) -> float:
    return sg_angle_diff(a, b)


def _point_to_segment_distance(point: tuple, start: tuple, end: tuple) -> float:
    return sg_point_to_segment_distance(point, start, end)


def _on_extended_line(a0: tuple, a1: tuple, point: tuple, tolerance: float) -> bool:
    return sg_point_to_line_distance(a0, a1, point) <= tolerance


def _segments_overlap(a0: tuple, a1: tuple, b0: tuple, b1: tuple, tolerance: float = 0.0) -> bool:
    return sg_segments_overlap_or_touch(a0, a1, b0, b1, tolerance)


def _overlap_or_touch(a0: tuple, a1: tuple, b0: tuple, b1: tuple) -> bool:
    return _segments_overlap(a0, a1, b0, b1, tolerance=1.0)


def _spanning_endpoints(endpoints: list[tuple]) -> tuple[tuple, tuple]:
    return sg_spanning_endpoints(endpoints)


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