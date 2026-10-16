from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services.braille_layout import clear_braille_position
from app.services.segment_geometry import (
    angle_diff as sg_angle_diff,
    line_angle_deg as sg_line_angle_deg,
    point_to_line_distance as sg_point_to_line_distance,
    point_to_segment_distance as sg_point_to_segment_distance,
    rect_gap,
    segment_to_rect_distance,
    segments_overlap_or_touch as sg_segments_overlap_or_touch,
    spanning_endpoints as sg_spanning_endpoints,
)
from app.services.tactile_qa import braille_clearance_mm, braille_rect, element_extent, outline_segments
from app.services.tactile_rules import TACTILE_RULES
from app.services.tactile_svg import angle_arc_sweep, braille_text_extent, is_embossed_geometry, page_layout

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

    # Pass 4: lengthen ticks to the touch minimum instead of losing them.
    kept, tick_actions, tick_explanations = _enlarge_ticks(kept, semantic.image_width, semantic.image_height)
    actions.extend(tick_actions)
    explanations.extend(tick_explanations)
    kept, reseat_actions, reseat_explanations = _reseat_braille_beside_ticks(
        kept, {action.element_id for action in tick_actions}, semantic.image_width, semantic.image_height,
    )
    actions.extend(reseat_actions)
    explanations.extend(reseat_explanations)

    # Pass 5: fit the sheet to what can be explored by touch on one A4 page.
    kept, budget_actions, budget_explanations, budget_removed = _fit_tactile_budget(
        kept, semantic.image_width, semantic.image_height, semantic.relationships,
    )
    actions.extend(budget_actions)
    explanations.extend(budget_explanations)
    removed_ids |= budget_removed
    removed_count += len(budget_removed)

    # Pass 6: seat each angle's braille inside its arc, and move braille off arcs.
    kept, arc_actions, arc_explanations = _seat_braille_at_angle_arcs(
        kept, semantic.image_width, semantic.image_height,
    )
    actions.extend(arc_actions)
    explanations.extend(arc_explanations)

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


DENSITY_ACTIONS = frozenset({"removed_density_budget", "removed_density_essential"})


def density_removed_count(simplified: SimplifiedGeometry) -> int:
    return sum(1 for action in simplified.actions if action.action in DENSITY_ACTIONS)


def essential_density_removed_count(simplified: SimplifiedGeometry) -> int:
    return sum(1 for action in simplified.actions if action.action == "removed_density_essential")


# Relationships that make a feature part of the figure's structure: losing it
# changes the shape, an angle, or a stated equality.
STRUCTURAL_RELATIONSHIPS = frozenset({
    RelationshipType.ENDPOINT_OF,
    RelationshipType.VERTEX_OF,
    RelationshipType.CONNECTED_LINES,
    RelationshipType.ANGLE_BETWEEN,
    RelationshipType.ANGLE_ASSOCIATION,
    RelationshipType.PERPENDICULAR_LINES,
    RelationshipType.PARALLEL_LINES,
    RelationshipType.EQUAL_LENGTH,
    RelationshipType.CENTER_OF,
    RelationshipType.CIRCLE_CENTER,
})
MARKER_PROPERTIES = ("angle_marker", "right_angle", "parallel_marks")


CLOSED_FIGURES = frozenset({
    GeometryType.CIRCLE,
    GeometryType.ELLIPSE,
    GeometryType.TRIANGLE,
    GeometryType.RECTANGLE,
    GeometryType.POLYGON,
})


def essential_ids(
    elements: list[DetectedElement],
    relationships: list[ElementRelationship],
    mm_per_px: float | None = None,
) -> set[str]:
    """Non-point features that carry mathematical meaning: markers, ticks, and figure structure.

    With ``mm_per_px``, a closed figure large enough to be felt is a figure in its own
    right, even when nothing is related to it.
    """
    ids = {e.id for e in elements}
    structural: set[str] = set()
    for rel in relationships:
        if rel.type in STRUCTURAL_RELATIONSHIPS and len(set(rel.element_ids) & ids) >= 2:
            structural.update(rel.element_ids)
    essential: set[str] = set()
    for element in elements:
        if element.type in (GeometryType.POINT, GeometryType.TEXT_LABEL):
            continue
        props = element.semantic_properties
        if (
            element.id in structural
            or props.get("role") == "tick"
            or any(props.get(key) for key in MARKER_PROPERTIES)
            or element.geometry.get("right_angle_marker")
            or (
                mm_per_px is not None
                and element.type in CLOSED_FIGURES
                and element_extent(element) * mm_per_px >= TACTILE_RULES.minimum_feature_size_mm
            )
        ):
            essential.add(element.id)
    return essential


def is_protected(element: DetectedElement) -> bool:
    """Elements simplification must never drop to meet the tactile budget.

    Labelled features carry meaning, a drawn right-angle marker is a deliberate
    symbol, and a feature the teacher accepted is the teacher's decision.
    """
    if element.associated_label_id:
        return True
    if element.geometry.get("right_angle_marker"):
        return True
    return element.source == "teacher_override" and element.confidence >= TACTILE_RULES.high_confidence_threshold


TICK_MARGIN_MM = 0.5


def _enlarge_ticks(
    elements: list[DetectedElement], image_width: int, image_height: int,
) -> tuple[list[DetectedElement], list[SimplificationAction], list[str]]:
    """Extend each short tick about its midpoint, across its line, to the touch minimum."""
    if image_width <= 0 or image_height <= 0:
        return elements, [], []
    mm_per_px = page_layout(image_width, image_height).mm_per_px
    target_px = (TACTILE_RULES.minimum_feature_size_mm + TICK_MARGIN_MM) / mm_per_px
    result: list[DetectedElement] = []
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    for element in elements:
        geo = element.geometry
        if (
            element.type is not GeometryType.LINE_SEGMENT
            or element.semantic_properties.get("role") != "tick"
            or "start" not in geo
        ):
            result.append(element)
            continue
        (x0, y0), (x1, y1) = geo["start"], geo["end"]
        length = math.dist((x0, y0), (x1, y1))
        if length == 0 or length >= target_px:
            result.append(element)
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = target_px / 2
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        start = [round(cx - ux * half), round(cy - uy * half)]
        end = [round(cx + ux * half), round(cy + uy * half)]
        xs, ys = (start[0], end[0]), (start[1], end[1])
        result.append(replace(element, **{
            "geometry": {**geo, "start": start, "end": end, "length": round(math.dist(start, end), 2)},
            "bbox": (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
        }))
        actions.append(SimplificationAction(
            element_id=element.id,
            action="enlarged_tick",
            detail=f"Tick lengthened from {length * mm_per_px:.1f}mm to {target_px * mm_per_px:.1f}mm so it can be felt.",
        ))
        explanations.append(f"Lengthened tick {element.id} to the touch minimum.")
    return result, actions, explanations


def _reseat_braille_beside_ticks(
    elements: list[DetectedElement], enlarged: set[str], image_width: int, image_height: int,
) -> tuple[list[DetectedElement], list[SimplificationAction], list[str]]:
    """Move braille that a lengthened tick now runs into; it was placed against the drawn tick."""
    if not enlarged or image_width <= 0 or image_height <= 0:
        return elements, [], []
    bounds = (image_width, image_height)
    mm_per_px = page_layout(*bounds).mm_per_px
    ticks = [e for e in elements if e.id in enlarged]
    labels = [e for e in elements if e.type is GeometryType.TEXT_LABEL]
    segments = [seg for e in elements if e.type is not GeometryType.TEXT_LABEL for seg in outline_segments(e)]
    rects = {e.id: braille_rect(e, *bounds) for e in labels}
    moved: dict[str, DetectedElement] = {}
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    for label in labels:
        rect = rects[label.id]
        if rect is None or all(
            braille_clearance_mm(rect, tick, mm_per_px) >= TACTILE_RULES.braille_to_line_clearance_mm for tick in ticks
        ):
            continue
        position = tuple(label.geometry["position"])
        others = [r for key, r in rects.items() if key != label.id and r is not None]
        braille = str(label.geometry.get("braille") or label.geometry.get("text") or "")
        new_position = clear_braille_position(position, braille, bounds, segments, others)
        if new_position == position:
            continue
        moved[label.id] = replace(label, geometry={**label.geometry, "position": list(new_position)})
        rects[label.id] = braille_rect(moved[label.id], *bounds)
        actions.append(SimplificationAction(
            element_id=label.id,
            action="moved_braille_from_tick",
            detail="Braille moved clear of a tick that was lengthened to the touch minimum.",
        ))
        explanations.append(f"Moved braille {label.id} clear of a lengthened tick.")
    return [moved.get(e.id, e) for e in elements], actions, explanations


ANGLE_BRAILLE_STEP_FRACTION = 0.25
ANGLE_BRAILLE_SWEEP_FRACTIONS = (0.5, 0.4, 0.6, 0.3, 0.7)


def _seat_braille_at_angle_arcs(
    elements: list[DetectedElement], image_width: int, image_height: int,
) -> tuple[list[DetectedElement], list[SimplificationAction], list[str]]:
    """Put an angle's braille inside the angle, past its arc; clear other braille off arcs.

    A measure printed inside an angle belongs inside it on the sheet too, and the
    embossed arc needs the same clearance from braille as any line. Every angle
    first tries for a clear spot between its own arms; only those with no room get
    the nearest clear spot to their arc. A label is only moved to a spot that
    clears every line and every other braille block.
    """
    arcs = {e.id: (e, sweep) for e in elements if e.type is GeometryType.ANGLE and (sweep := angle_arc_sweep(e.geometry))}
    if not arcs or image_width <= 0 or image_height <= 0:
        return elements, [], []
    bounds = (image_width, image_height)
    mm_per_px = page_layout(*bounds).mm_per_px
    clearance_px = TACTILE_RULES.braille_to_line_clearance_mm / mm_per_px
    spacing_px = TACTILE_RULES.braille_to_braille_spacing_mm / mm_per_px
    angle_of = {e.associated_label_id: key for key, (e, _) in arcs.items() if e.associated_label_id}
    labels = [e for e in elements if e.type is GeometryType.TEXT_LABEL and braille_rect(e, *bounds) is not None]
    segments = [seg for e in elements if is_embossed_geometry(e) for seg in outline_segments(e)]
    rects = {e.id: braille_rect(e, *bounds) for e in labels}
    moved: dict[str, DetectedElement] = {}
    actions: list[SimplificationAction] = []
    explanations: list[str] = []

    def braille_of(label: DetectedElement) -> str:
        return str(label.geometry.get("braille") or label.geometry.get("text") or "")

    def clear_at(label: DetectedElement, position: tuple[int, int]):
        candidate = replace(label, geometry={**label.geometry, "position": list(position)})
        rect = braille_rect(candidate, *bounds)
        if rect is None or not (rect[0] >= 0 and rect[1] >= 0 and rect[2] <= image_width and rect[3] <= image_height):
            return None
        if all(segment_to_rect_distance(a, b, rect) >= clearance_px for a, b in segments) and all(
            rect_gap(rect, other) >= spacing_px for key, other in rects.items() if key != label.id
        ):
            return candidate
        return None

    def seat(label: DetectedElement, candidate: DetectedElement, reason: str) -> None:
        moved[label.id] = candidate
        rects[label.id] = braille_rect(candidate, *bounds)
        actions.append(SimplificationAction(
            element_id=label.id, action="seated_braille_at_angle_arc", detail=f"Braille placed {reason}.",
        ))
        explanations.append(f"Moved braille {label.id} {reason}.")

    def inside_angle(label: DetectedElement, angle: DetectedElement, sweep) -> DetectedElement | None:
        (vx, vy), radius, start, span = sweep
        width, height = braille_text_extent(braille_of(label), *bounds)
        vertex = angle.geometry["vertex"]
        reach_limit = min(math.dist(vertex, arm) for arm in angle.geometry["arms"])
        step = max(1.0, ANGLE_BRAILLE_STEP_FRACTION * min(width, height))
        reach = radius + clearance_px + min(width, height) / 2
        while reach <= reach_limit:
            for fraction in ANGLE_BRAILLE_SWEEP_FRACTIONS:
                direction = start + span * fraction
                position = (round(vx + reach * math.cos(direction)), round(vy + reach * math.sin(direction)))
                if (candidate := clear_at(label, position)) is not None:
                    return candidate
            reach += step
        return None

    angle_labels = [label for label in labels if label.id in angle_of]
    crowded: list[DetectedElement] = []
    for label in sorted(angle_labels, key=lambda e: -arcs[angle_of[e.id]][1][3]):
        angle, sweep = arcs[angle_of[label.id]]
        candidate = inside_angle(label, angle, sweep)
        if candidate is None:
            crowded.append(label)
        elif list(candidate.geometry["position"]) != list(label.geometry["position"]):
            seat(label, candidate, f"inside angle {angle.id}, past its arc")
    arc_elements = [e for e, _ in arcs.values()]
    for label in crowded + [label for label in labels if label.id not in angle_of]:
        current = moved.get(label.id, label)
        if label.id not in angle_of and all(
            braille_clearance_mm(rects[label.id], arc, mm_per_px) >= TACTILE_RULES.braille_to_line_clearance_mm
            for arc in arc_elements
        ):
            continue
        if label.id in angle_of:
            (vx, vy), radius, start, span = arcs[angle_of[label.id]][1]
            reach = radius + clearance_px
            desired = (round(vx + reach * math.cos(start + span / 2)), round(vy + reach * math.sin(start + span / 2)))
        else:
            desired = tuple(int(round(v)) for v in current.geometry["position"])
        position = clear_braille_position(
            desired, braille_of(label), bounds, segments, [r for key, r in rects.items() if key != label.id],
        )
        candidate = clear_at(label, position)
        if candidate is None or list(position) == list(current.geometry["position"]):
            continue
        seat(label, candidate, "beside its angle's arc" if label.id in angle_of else "clear of an angle arc")
    return [moved.get(e.id, e) for e in elements], actions, explanations


GRID_AXIS_TOLERANCE_DEG = 3.0
GRID_MIN_LENGTH_PX = 80.0
GRID_MIN_FAMILY = 5
GRID_DISTINCT_OFFSET_PX = 10.0


def _grid_lines(elements: list[DetectedElement]) -> set[str]:
    """Long axis-aligned segments in a family of five or more distinct parallel rows or columns.

    A regular family like this is background grid or ruling, so it gives way
    to the figure when the sheet is over the feature budget.
    """
    families: dict[str, list[tuple[float, str]]] = {"horizontal": [], "vertical": []}
    for element in elements:
        if element.type is not GeometryType.LINE_SEGMENT:
            continue
        start, end = element.geometry.get("start"), element.geometry.get("end")
        if not start or not end or math.dist(start, end) < GRID_MIN_LENGTH_PX:
            continue
        angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180
        if min(angle, 180 - angle) <= GRID_AXIS_TOLERANCE_DEG:
            families["horizontal"].append(((start[1] + end[1]) / 2, element.id))
        elif abs(angle - 90) <= GRID_AXIS_TOLERANCE_DEG:
            families["vertical"].append(((start[0] + end[0]) / 2, element.id))
    grid: set[str] = set()
    for members in families.values():
        offsets = sorted(offset for offset, _ in members)
        distinct = sum(1 for i, offset in enumerate(offsets) if i == 0 or offset - offsets[i - 1] > GRID_DISTINCT_OFFSET_PX)
        if distinct >= GRID_MIN_FAMILY:
            grid.update(element_id for _, element_id in members)
    return grid


def _budget_priority(element: DetectedElement, mm_per_px: float, grid: set[str]) -> tuple[int, int, float, float]:
    # Lowest first: unlabelled junction dots, then grid lines, then
    # low-confidence, then small features.
    return (
        0 if element.type is GeometryType.POINT else 1,
        0 if element.id in grid else 1,
        element.confidence,
        element_extent(element) * mm_per_px,
    )


def _isolated_speck(
    element: DetectedElement, elements: list[DetectedElement], essential: set[str], spacing_px: float,
) -> bool:
    """A labelled sub-minimum outline touching nothing: a text fragment or blot, not a figure."""
    if element.type not in (GeometryType.POLYGON, GeometryType.TRIANGLE, GeometryType.RECTANGLE) or element.id in essential:
        return False
    if element.source == "teacher_override" or element.geometry.get("right_angle_marker"):
        return False
    return all(
        _outline_gap(element, other) > spacing_px
        for other in elements
        if other.id != element.id and is_embossed_geometry(other)
    )


def _outline_gap(a: DetectedElement, b: DetectedElement) -> float:
    """Closest approach between two embossed outlines, measured to their sides, not only their vertices."""
    a_segments, b_segments = outline_segments(a), outline_segments(b)
    if not a_segments or not b_segments:
        return float("inf")
    return min(
        min(sg_point_to_segment_distance(p, *seg) for p in (s0, s1) for seg in others)
        for own, others in ((a_segments, b_segments), (b_segments, a_segments))
        for s0, s1 in own
    )


def _fit_tactile_budget(
    elements: list[DetectedElement],
    image_width: int,
    image_height: int,
    relationships: list[ElementRelationship] | None = None,
) -> tuple[list[DetectedElement], list[SimplificationAction], list[str], set[str]]:
    if image_width <= 0 or image_height <= 0:
        return elements, [], [], set()
    mm_per_px = page_layout(image_width, image_height).mm_per_px
    minimum_mm = TACTILE_RULES.minimum_feature_size_mm
    spacing_px = TACTILE_RULES.minimum_feature_spacing_mm / mm_per_px
    essential = essential_ids(elements, relationships or [], mm_per_px)
    actions: list[SimplificationAction] = []
    explanations: list[str] = []
    removed: set[str] = set()

    for element in elements:
        if not is_embossed_geometry(element):
            continue
        extent_mm = element_extent(element) * mm_per_px
        if not 0 < extent_mm < minimum_mm:
            continue
        if not is_protected(element):
            removed.add(element.id)
            actions.append(SimplificationAction(
                element_id=element.id,
                action="removed_below_tactile_size",
                detail=f"Unlabelled feature is {extent_mm:.1f}mm on the printed sheet, below the {minimum_mm:g}mm touch minimum.",
            ))
            explanations.append(f"Removed {element.id}: {extent_mm:.1f}mm is too small to feel.")
        elif _isolated_speck(element, elements, essential, spacing_px):
            removed.add(element.id)
            actions.append(SimplificationAction(
                element_id=element.id,
                action="removed_isolated_speck",
                detail=(
                    f"Isolated {extent_mm:.1f}mm outline touching no other feature, linked to label "
                    f"{element.associated_label_id}; treated as noise. The label stays on the sheet."
                ),
            ))
            explanations.append(f"Removed {element.id}: an isolated {extent_mm:.1f}mm speck, not a figure.")

    target = TACTILE_RULES.tactile_feature_target
    limit = TACTILE_RULES.tactile_feature_limit
    embossed = [e for e in elements if is_embossed_geometry(e) and e.id not in removed]
    excess = len(embossed) - target
    if excess > 0:
        grid = _grid_lines(embossed)
        candidates = sorted(
            (e for e in embossed if not is_protected(e)),
            key=lambda e: _budget_priority(e, mm_per_px, grid),
        )
        optional = [e for e in candidates if e.id not in essential]
        for element in optional[:excess]:
            removed.add(element.id)
            actions.append(SimplificationAction(
                element_id=element.id,
                action="removed_density_budget",
                detail=(
                    f"Lowest-priority unlabelled feature removed so the sheet stays within {target} embossed "
                    "features; accept it in the editor to keep it."
                ),
            ))
            explanations.append(f"Removed {element.id} to keep the sheet within the {target}-feature tactile budget.")
        # Figure structure, ticks and markers give way only above the hard limit,
        # and QA then blocks export: the sheet no longer shows the whole figure.
        over_limit = len(embossed) - len(optional[:excess]) - limit
        for element in [e for e in candidates if e.id in essential][:max(0, over_limit)]:
            removed.add(element.id)
            actions.append(SimplificationAction(
                element_id=element.id,
                action="removed_density_essential",
                detail=(
                    f"Part of the figure's structure, removed because the sheet has more than {limit} embossed "
                    "features. Crop or split the image before exporting."
                ),
            ))
            explanations.append(f"Removed essential feature {element.id} to stay under the {limit}-feature limit.")

    return [e for e in elements if e.id not in removed], actions, explanations, removed


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