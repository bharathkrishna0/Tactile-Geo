from __future__ import annotations

import math

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
    TransformationExplanation,
    classify_confidence,
)
from app.services.geometry_relations import grid_like_ids, infer_relationships
from app.services.tactile_rules import TACTILE_RULES
from app.services.vectorization import MIN_ROUND_VERTICES

# Hard ceiling on heuristic angles. With one angle per vertex the count is already
# bounded by the vertex count; this is a second guard against pathological input.
MAX_ANGLE_ELEMENTS = TACTILE_RULES.complexity_threshold


def analyze_diagram(shapes: list[dict], width: int, height: int, labels: list[dict], right_angles: list[dict] | None = None,
                    angle_arcs: list[dict] | None = None, parallel_marks: list[dict] | None = None) -> SemanticGeometry:
    elements: list[DetectedElement] = []
    relationships: list[ElementRelationship] = []
    explanations: list[TransformationExplanation] = []
    id_counter = 0

    for shape in shapes:
        if shape["type"] == "line":
            element, rels = _analyze_line(shape, id_counter)
            id_counter += 1
            elements.extend(element)
            relationships.extend(rels)
        elif shape["type"] == "contour":
            element, rels = _analyze_contour(shape, id_counter, width, height)
            id_counter += 1
            elements.extend(element)
            relationships.extend(rels)
        elif shape["type"] == "ellipse":
            element = _analyze_ellipse(shape, id_counter)
            id_counter += 1
            elements.append(element)
        elif shape["type"] == "dot":
            elements.append(_analyze_dot(shape, id_counter))
            id_counter += 1

    for mark in parallel_marks or []:
        _attach_parallel_marks(elements, mark)

    # Detect significant points (shared line endpoints, shape vertices).
    point_elements, point_explanations = _detect_points(elements, id_counter, labels)
    id_counter += len(point_elements)
    elements.extend(point_elements)
    explanations.extend(point_explanations)

    # Right angles marked in the source keep their marker as a tactile symbol.
    for marker in right_angles or []:
        elements.append(_right_angle_element(marker, id_counter))
        explanations.append(TransformationExplanation(
            stage="diagram_analysis",
            element_id=f"el_{id_counter}",
            message=f"Detected a drawn right-angle marker at {tuple(marker['vertex'])}.",
        ))
        id_counter += 1

    for arc in angle_arcs or []:
        element = _angle_arc_element(arc, id_counter)
        elements.append(element)
        relationships.extend(_angle_at_point(element, elements))
        explanations.append(TransformationExplanation(
            stage="diagram_analysis",
            element_id=element.id,
            message=f"Detected a drawn angle arc at {tuple(arc['vertex'])}.",
        ))
        id_counter += 1

    # Detect angles where two connected segments meet.
    angle_elements, angle_explanations, extra_relationships = _detect_angles(elements, id_counter, relationships)
    id_counter += len(angle_elements)
    elements.extend(angle_elements)
    relationships.extend(extra_relationships)
    explanations.extend(angle_explanations)

    for label in labels:
        element = DetectedElement(
            id=f"el_{id_counter}",
            type=GeometryType.TEXT_LABEL,
            geometry={
                "text": label.get("text", ""),
                "braille": label.get("braille", ""),
                "position": label.get("position") or label.get("desired_position") or [0, 0],
            },
            confidence=float(label.get("confidence", 0.9)),
            confidence_level=classify_confidence(float(label.get("confidence", 0.9))),
            needs_review=float(label.get("confidence", 0.9)) < 0.5,
            source="ocr",
            bbox=_labels_bbox(label),
            provenance=f"OCR detected text {label.get('text', '')!r} with confidence {float(label.get('confidence', 0.9)):.2f}.",
        )
        id_counter += 1
        elements.append(element)

    # Relationship inference from shared geometry (parallel/perpendicular/connected/
    # point_on_line/circle/center/etc.), with confidence.
    relationships.extend(infer_relationships(elements))
    relationships = _dedupe_relationships(relationships)

    # Associate labels with nearby geometry elements (candidate + confidence).
    elements, label_explanations = _associate_labels(elements, relationships)
    explanations.extend(label_explanations)

    low_confidence_count = sum(1 for element in elements if element.needs_review or element.confidence_level is ConfidenceLevel.LOW)
    review_flags: list[str] = []
    for element in elements:
        if element.needs_review:
            review_flags.append(f"Low-confidence {element.type.value} detected — may be noise. Review before export.")

    return SemanticGeometry(
        elements=elements,
        relationships=relationships,
        image_width=width,
        image_height=height,
        element_count=len(elements),
        low_confidence_count=low_confidence_count,
        review_flags=review_flags,
        explanations=explanations,
    )


def _analyze_line(shape: dict, id: int) -> tuple[list[DetectedElement], list[ElementRelationship]]:
    points = shape["points"]
    start, end = points[0], points[1]
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    confidence = 0.95 if length > 20 else 0.45
    element = DetectedElement(
        id=f"el_{id}",
        type=GeometryType.LINE_SEGMENT,
        geometry={"start": start, "end": end, "length": round(length, 2)},
        confidence=confidence,
        confidence_level=classify_confidence(confidence),
        needs_review=confidence < 0.5,
        source="hough",
        bbox=(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]), abs(end[1] - start[1])),
        semantic_properties={"length": round(length, 2), **({"role": shape["role"]} if shape.get("role") else {})},
        provenance=_line_provenance(length, start, end, confidence)
        + (" Recovered as a tick drawn across a longer line." if shape.get("role") == "tick" else ""),
    )
    return [element], []


def _line_provenance(length: float, start, end, confidence: float) -> str:
    reason = "long enough to be a confident segment" if length > 20 else "short; may be a fragment or noise"
    return f"Detected line segment from ({start[0]},{start[1]}) to ({end[0]},{end[1]}) of length {length:.0f}px ({reason}); confidence {confidence:.2f}."


def _analyze_contour(shape: dict, id: int, width: int, height: int) -> tuple[list[DetectedElement], list[ElementRelationship]]:
    points = shape["points"]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    bbox = (min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
    area = _polygon_area(points)
    perimeter = _polygon_perimeter(points)
    circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    gtype, confidence_boost = _classify_contour(points, area, perimeter, circularity)
    confidence = _contour_confidence(points, area, perimeter, width, height, confidence_boost)
    geometry = _contour_geometry(gtype, points, area)
    element = DetectedElement(
        id=f"el_{id}",
        type=gtype,
        geometry=geometry,
        confidence=confidence,
        confidence_level=classify_confidence(confidence),
        needs_review=confidence < 0.5,
        source="contour",
        bbox=bbox,
        semantic_properties=_contour_properties(gtype, points, area, perimeter),
        provenance=f"Detected as {gtype.value} from contour of {len(points)} vertices with area {area:.0f}px and confidence {confidence:.2f}."
        + (" Outline joined from line segments that close one loop." if shape.get("source") == "joined_sides" else ""),
    )
    relationships: list[ElementRelationship] = []
    if gtype is GeometryType.CIRCLE and "center" in geometry:
        relationships.append(ElementRelationship(
            id=f"rel_{id}_center",
            type=RelationshipType.CIRCLE_CENTER,
            element_ids=[element.id],
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            needs_review=False,
        ))
    return [element], relationships


DOT_CONFIDENCE = 0.9


def _analyze_dot(shape: dict, id: int) -> DetectedElement:
    x, y = shape["points"][0]
    radius = float(shape.get("radius", 3.0))
    return DetectedElement(
        id=f"el_{id}",
        type=GeometryType.POINT,
        geometry={"position": [x, y]},
        confidence=DOT_CONFIDENCE,
        confidence_level=classify_confidence(DOT_CONFIDENCE),
        needs_review=False,
        source="contour",
        bbox=(x - radius, y - radius, 2 * radius, 2 * radius),
        semantic_properties={"drawn_dot": True, "radius": radius},
        provenance=f"Detected a drawn dot (radius {radius:.1f} px) at ({x},{y}).",
    )


def _analyze_ellipse(shape: dict, id: int) -> DetectedElement:
    center = shape["center"]
    semi_a, semi_b = shape["semi_axes"]
    angle = shape.get("angle", 0.0)
    area = shape.get("area", math.pi * semi_a * semi_b)
    confidence = 0.88 if area > 500 else 0.65
    return DetectedElement(
        id=f"el_{id}",
        type=GeometryType.ELLIPSE,
        geometry={"center": center, "semi_axes": [semi_a, semi_b], "angle": angle, "area": round(area, 2)},
        confidence=confidence,
        confidence_level=classify_confidence(confidence),
        needs_review=confidence < 0.5,
        source="contour",
        bbox=(center[0] - semi_a, center[1] - semi_b, semi_a * 2, semi_b * 2),
        semantic_properties={"area": round(area, 2), "aspect_ratio": round(min(semi_a, semi_b) / max(semi_a, semi_b), 3) if semi_a > 0 else 0, "rotation_deg": angle},
        provenance=f"Detected as ellipse (semi-axes {semi_a} x {semi_b}, rotated {angle} deg) with confidence {confidence:.2f}.",
    )


def _classify_contour(points: list[tuple[int, int]], area: float, perimeter: float, circularity: float) -> tuple[GeometryType, float]:
    n = len(points)
    if n == 3:
        return GeometryType.TRIANGLE, 0.15
    if n == 4:
        if _is_rectangle(points):
            return GeometryType.RECTANGLE, 0.15
        return GeometryType.POLYGON, 0.08
    if n >= MIN_ROUND_VERTICES and circularity > 0.85:
        return GeometryType.CIRCLE, 0.15
    if n >= 5 and circularity > 0.6:
        return GeometryType.POLYGON, 0.05
    return GeometryType.POLYGON, 0.02


def _is_rectangle(points: list[tuple[int, int]]) -> bool:
    if len(points) != 4:
        return False
    angles = []
    for i in range(4):
        a = points[i]
        b = points[(i + 1) % 4]
        c = points[(i + 2) % 4]
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ab[0] * bc[0] + ab[1] * bc[1]
        len_ab = math.hypot(*ab)
        len_bc = math.hypot(*bc)
        if len_ab == 0 or len_bc == 0:
            return False
        cos_angle = dot / (len_ab * len_bc)
        angle = abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))))
        angles.append(abs(angle - 90))
    return max(angles) < 15


def _contour_confidence(points: list[tuple[int, int]], area: float, perimeter: float, width: int, height: int, boost: float) -> float:
    if area < 80:
        return 0.35
    if area < 300:
        return 0.5
    if area < 1000:
        return 0.6
    image_area = max(1, width * height)
    area_fraction = area / image_area
    if area_fraction > 0.05:
        return min(0.99, 0.7 + boost)
    return min(0.95, 0.62 + boost)


def _contour_properties(gtype: GeometryType, points: list[tuple[int, int]], area: float, perimeter: float) -> dict:
    properties: dict = {"area": round(area, 2), "perimeter": round(perimeter, 2), "vertex_count": len(points)}
    if gtype in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON) and len(points) >= 3:
        properties["interior_angles_deg"] = [
            round(_angle_between_vectors(points[i - 1], points[i], points[(i + 1) % len(points)]), 1)
            for i in range(len(points))
        ]
    return properties


def _contour_geometry(gtype: GeometryType, points: list[tuple[int, int]], area: float) -> dict:
    if gtype is GeometryType.CIRCLE:
        center = _polygon_centroid(points)
        radius = math.sqrt(area / math.pi)
        return {"center": center, "radius": round(radius, 2)}
    return {"points": points, "area": round(area, 2)}


def _polygon_centroid(points: list[tuple[int, int]]) -> list[int]:
    if not points:
        return [0, 0]
    return [round(sum(p[0] for p in points) / len(points)), round(sum(p[1] for p in points) / len(points))]


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))))


def _polygon_perimeter(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(points[(i + 1) % len(points)][0] - points[i][0], points[(i + 1) % len(points)][1] - points[i][1]) for i in range(len(points)))


def _labels_bbox(label: dict) -> tuple[int, int, int, int] | None:
    bbox = label.get("bbox")
    if not bbox:
        return None
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _line_angle(start: tuple, end: tuple) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _endpoint_near(a: tuple, b: tuple, tolerance: float = 10.0) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance


def _dedupe_relationships(relationships: list[ElementRelationship]) -> list[ElementRelationship]:
    seen: set[tuple] = set()
    result: list[ElementRelationship] = []
    for rel in relationships:
        key = (rel.type.value, tuple(sorted(rel.element_ids)))
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


LABELLED_VERTEX_PX = 45.0
CIRCLE_CENTRE_PX = 8.0


def _label_anchor(label: dict) -> tuple[float, float] | None:
    """Where the text was printed; the braille position may have been moved away."""
    box = label.get("bbox")
    if box:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    position = label.get("position") or label.get("desired_position")
    if position:
        return float(position[0]), float(position[1])
    return None


# A line ending on another outline (a radius on its circle, an altitude's foot
# on the base) makes a point there, as does a labelled crossing of two lines.
JUNCTION_PX = 6.0
JUNCTION_END_MARGIN_PX = 10.0
POINT_MERGE_PX = 4.0
# A stroke ending at a drawn dot stops at the dot's rim, not its centre.
DOT_MERGE_PX = 12.0


def _round_outline(element: DetectedElement) -> tuple[tuple[float, float], float] | None:
    geo = element.geometry
    if element.type is GeometryType.CIRCLE and "center" in geo and "radius" in geo:
        return (float(geo["center"][0]), float(geo["center"][1])), float(geo["radius"])
    if element.type is GeometryType.ELLIPSE and "center" in geo and geo.get("semi_axes"):
        a, b = (float(v) for v in geo["semi_axes"])
        if max(a, b) > 0 and min(a, b) / max(a, b) >= 0.9:
            return (float(geo["center"][0]), float(geo["center"][1])), (a + b) / 2
    return None


def _segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _touches_interior(point, start, end) -> bool:
    return (_segment_distance(point, start, end) <= JUNCTION_PX
            and math.dist(point, start) > JUNCTION_END_MARGIN_PX
            and math.dist(point, end) > JUNCTION_END_MARGIN_PX)


def _crossing(a0, a1, b0, b1) -> tuple[int, int] | None:
    """Where two segments properly cross, away from either one's ends."""
    rx, ry = a1[0] - a0[0], a1[1] - a0[1]
    sx, sy = b1[0] - b0[0], b1[1] - b0[1]
    denom = rx * sy - ry * sx
    if denom == 0:
        return None
    t = ((b0[0] - a0[0]) * sy - (b0[1] - a0[1]) * sx) / denom
    u = ((b0[0] - a0[0]) * ry - (b0[1] - a0[1]) * rx) / denom
    if not (0 < t < 1 and 0 < u < 1):
        return None
    point = (a0[0] + t * rx, a0[1] + t * ry)
    if min(math.dist(point, p) for p in (a0, a1, b0, b1)) <= JUNCTION_END_MARGIN_PX:
        return None
    return int(round(point[0])), int(round(point[1]))


def _detect_points(
    elements: list[DetectedElement], start_id: int, labels: list[dict] | None = None
) -> tuple[list[DetectedElement], list[TransformationExplanation]]:
    """Create point elements at shared endpoints, labelled polygon vertices and marked circle centres."""
    significant: dict[tuple[int, int], int] = {}
    explanations: list[TransformationExplanation] = []
    anchors = [a for a in (_label_anchor(label) for label in labels or []) if a]
    plain_lines = [
        element for element in elements
        if element.type is GeometryType.LINE_SEGMENT and not element.semantic_properties.get("role")
    ]
    line_ends = [tuple(element.geometry[key]) for element in plain_lines for key in ("start", "end")]
    grid = grid_like_ids(plain_lines)
    rounds = [r for r in (_round_outline(element) for element in elements) if r]
    outline_edges = [
        (tuple(element.geometry["start"]), tuple(element.geometry["end"]), element.id)
        for element in plain_lines if element.id not in grid
    ] + [
        (tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), element.id)
        for element in elements if element.type in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON)
        for pts in [element.geometry.get("points", [])]
        for i in range(len(pts))
    ]
    dots = [tuple(element.geometry["position"]) for element in elements
            if element.type is GeometryType.POINT and element.geometry.get("position")]

    def labelled(coord) -> bool:
        return any(math.dist(coord, anchor) <= LABELLED_VERTEX_PX for anchor in anchors)

    for element in elements:
        if element.type is GeometryType.LINE_SEGMENT:
            if element.semantic_properties.get("role") == "tick":
                continue
            for coord in (element.geometry["start"], element.geometry["end"]):
                significant[coord] = significant.get(coord, 0) + 1
                if element.semantic_properties.get("role") or element.id in grid:
                    continue
                on_round = any(abs(math.dist(coord, centre) - radius) <= JUNCTION_PX for centre, radius in rounds)
                on_outline = any(owner != element.id and _touches_interior(coord, start, end)
                                 for start, end, owner in outline_edges)
                if on_round or on_outline:
                    significant[coord] += 1
        elif element.type in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON):
            points = element.geometry.get("points", [])
            for coord in points:
                # A vertex named by a nearby label is a point in its own right.
                significant[coord] = significant.get(coord, 0) + (2 if labelled(coord) else 1)
        elif element.type is GeometryType.CIRCLE and "center" in element.geometry:
            centre = tuple(int(round(v)) for v in element.geometry["center"])
            # A radius or diameter drawn to the centre makes the centre a point.
            if any(math.dist(centre, end) <= CIRCLE_CENTRE_PX for end in line_ends):
                significant[centre] = significant.get(centre, 0) + 2

    crossable = [element for element in plain_lines if element.id not in grid]
    for i, a in enumerate(crossable):
        for b in crossable[i + 1:]:
            coord = _crossing(a.geometry["start"], a.geometry["end"], b.geometry["start"], b.geometry["end"])
            if coord and labelled(coord):
                significant[coord] = significant.get(coord, 0) + 2

    # A coordinate shared by two or more features is a real point.
    point_elements: list[DetectedElement] = []
    pid = start_id
    emitted: list[tuple[tuple[int, int], float]] = [(dot, DOT_MERGE_PX) for dot in dots]
    for coord, count in significant.items():
        if count >= 2:
            if any(math.dist(coord, other) <= px for other, px in emitted):
                continue
            emitted.append((coord, POINT_MERGE_PX))
            element = DetectedElement(
                id=f"el_{pid}",
                type=GeometryType.POINT,
                geometry={"position": [coord[0], coord[1]]},
                confidence=0.85,
                confidence_level=ConfidenceLevel.HIGH,
                needs_review=False,
                source="heuristic",
                bbox=(coord[0] - 3, coord[1] - 3, 6, 6),
                semantic_properties={"shared_by": count},
                provenance=f"Heuristically detected point at ({coord[0]},{coord[1]}) shared by {count} features.",
            )
            pid += 1
            point_elements.append(element)
            explanations.append(TransformationExplanation(
                stage="diagram_analysis",
                element_id=element.id,
                message=f"Detected point at ({coord[0]},{coord[1]}) shared by {count} features.",
            ))
    return point_elements, explanations


MIN_ARM_LENGTH_PX = 12.0
MIN_ANGLE_DEGREES = 5.0
MAX_ANGLE_DEGREES = 175.0
VERTEX_MERGE_TOLERANCE_PX = 8.0
ANGLE_RELATIONSHIP_CONFIDENCE = 0.6


def _vertex_key(coord: tuple[int, int]) -> tuple[int, int]:
    return (int(coord[0]), int(coord[1]))


def _far_endpoint_from(element: DetectedElement, common: tuple) -> tuple:
    """The endpoint of ``element`` that is furthest from ``common``.

    Picking by equality with the shared vertex is wrong whenever two segments meet
    start-to-start or end-to-end: the "other" endpoint would then be the one that
    is nearly coincident with the vertex, giving a zero-length arm.
    """
    e0, e1 = element.geometry["start"], element.geometry["end"]
    d0 = math.hypot(e0[0] - common[0], e0[1] - common[1])
    d1 = math.hypot(e1[0] - common[0], e1[1] - common[1])
    return e0 if d0 >= d1 else e1


def _angle_between_vectors(a: tuple, vertex: tuple, b: tuple) -> float:
    va = (a[0] - vertex[0], a[1] - vertex[1])
    vb = (b[0] - vertex[0], b[1] - vertex[1])
    dot = va[0] * vb[0] + va[1] * vb[1]
    la = math.hypot(*va)
    lb = math.hypot(*vb)
    if la == 0 or lb == 0:
        return 0.0
    cos_angle = dot / (la * lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def _detect_angles(elements: list[DetectedElement], start_id: int, existing: list[ElementRelationship]) -> tuple[list[DetectedElement], list[TransformationExplanation], list[ElementRelationship]]:
    """Detect angle elements where two connected line segments meet.

    Gate rationale: the pairwise loop is O(n^2) over line segments, so unfiltered
    it turns every duplicated or fragmented edge into many candidate angles. Two
    rules keep it meaningful and bounded:

    1. both arms must be real edges (long enough to be geometry, not Hough noise);
    2. at most one angle per vertex, choosing the pair with the longest arms.

    Rule 2 is what removes the combinatorial explosion — the number of angles can
    never exceed the number of distinct vertices.
    """
    segments = [e for e in elements if e.type is GeometryType.LINE_SEGMENT]
    candidates: list[tuple[float, tuple, DetectedElement, DetectedElement, tuple, tuple]] = []

    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            a0, a1 = a.geometry["start"], a.geometry["end"]
            b0, b1 = b.geometry["start"], b.geometry["end"]
            common = None
            for pa in (a0, a1):
                for pb in (b0, b1):
                    if _endpoint_near(tuple(pa), tuple(pb), tolerance=VERTEX_MERGE_TOLERANCE_PX):
                        common = tuple(pa)
                        break
                if common:
                    break
            if common is None:
                continue
            other_a = _far_endpoint_from(a, common)
            other_b = _far_endpoint_from(b, common)
            arm_a = math.hypot(other_a[0] - common[0], other_a[1] - common[1])
            arm_b = math.hypot(other_b[0] - common[0], other_b[1] - common[1])
            # Rule 1: reject stub arms and coincident segments.
            if arm_a < MIN_ARM_LENGTH_PX or arm_b < MIN_ARM_LENGTH_PX:
                continue
            angle = _angle_between_vectors(other_a, common, other_b)
            if angle < MIN_ANGLE_DEGREES or angle > MAX_ANGLE_DEGREES:
                continue
            candidates.append((arm_a * arm_b, common, a, b, other_a, other_b))

    # Rule 2: strongest arms win, and only the first claim on a vertex counts.
    candidates.sort(key=lambda item: -item[0])
    angle_elements: list[DetectedElement] = []
    explanations: list[TransformationExplanation] = []
    extra_relationships: list[ElementRelationship] = []
    claimed_vertices: list[tuple[int, int]] = [
        _vertex_key(tuple(e.geometry["vertex"])) for e in elements if e.type is GeometryType.ANGLE
    ]
    aid = start_id

    for _, common, a, b, other_a, other_b in candidates:
        if len(angle_elements) >= MAX_ANGLE_ELEMENTS:
            break
        vertex = _vertex_key(common)
        if any(math.hypot(vertex[0] - taken[0], vertex[1] - taken[1]) <= VERTEX_MERGE_TOLERANCE_PX for taken in claimed_vertices):
            continue
        claimed_vertices.append(vertex)
        angle = _angle_between_vectors(other_a, common, other_b)
        element = DetectedElement(
            id=f"el_{aid}",
            type=GeometryType.ANGLE,
            geometry={"vertex": list(common), "arms": [list(other_a), list(other_b)], "degrees": round(angle, 1)},
            confidence=ANGLE_RELATIONSHIP_CONFIDENCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            needs_review=True,
            source="heuristic",
            bbox=(common[0] - 5, common[1] - 5, 10, 10),
            semantic_properties={"degrees": round(angle, 1)},
            provenance=f"Heuristically detected angle of about {angle:.0f} degrees at {common} where segments {a.id} and {b.id} meet.",
        )
        aid += 1
        angle_elements.append(element)
        extra_relationships.append(ElementRelationship(
            id=f"rel_{a.id}_{b.id}_angle",
            type=RelationshipType.ANGLE_BETWEEN,
            element_ids=[a.id, b.id],
            confidence=ANGLE_RELATIONSHIP_CONFIDENCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            needs_review=True,
            explanation=f"Segments {a.id} and {b.id} form an angle of about {angle:.0f} degrees.",
        ))
        explanations.append(TransformationExplanation(
            stage="diagram_analysis",
            element_id=element.id,
            message=f"Detected angle of ~{angle:.0f} degrees at {common}.",
        ))
    return angle_elements, explanations, extra_relationships



RIGHT_ANGLE_MARKER_CONFIDENCE = 0.85


def _right_angle_element(marker: dict, index: int) -> DetectedElement:
    vertex = [int(v) for v in marker["vertex"]]
    size = int(marker["size"])
    return DetectedElement(
        id=f"el_{index}",
        type=GeometryType.ANGLE,
        geometry={
            "vertex": vertex,
            "arms": [[int(v) for v in arm] for arm in marker["arms"]],
            "degrees": 90.0,
            "right_angle_marker": True,
            "marker_size": size,
        },
        confidence=RIGHT_ANGLE_MARKER_CONFIDENCE,
        confidence_level=classify_confidence(RIGHT_ANGLE_MARKER_CONFIDENCE),
        needs_review=False,
        source="right_angle_marker",
        bbox=(vertex[0] - size, vertex[1] - size, 2 * size, 2 * size),
        semantic_properties={"degrees": 90.0, "right_angle": True},
        provenance=f"Right-angle marker drawn at {tuple(vertex)}; measured corner {marker['degrees']} degrees.",
    )


ANGLE_ARC_CONFIDENCE = 0.85
MARK_HOST_PX = 6.0
ANGLE_POINT_PX = 10.0


def _attach_parallel_marks(elements: list[DetectedElement], mark: dict) -> None:
    host = [tuple(map(float, p)) for p in mark["host"]]
    for element in elements:
        geo = element.geometry
        if "start" not in geo or "end" not in geo:
            continue
        ends = [tuple(map(float, geo["start"])), tuple(map(float, geo["end"]))]
        if (math.dist(ends[0], host[0]) <= MARK_HOST_PX and math.dist(ends[1], host[1]) <= MARK_HOST_PX) or (
                math.dist(ends[0], host[1]) <= MARK_HOST_PX and math.dist(ends[1], host[0]) <= MARK_HOST_PX):
            element.semantic_properties["parallel_marks"] = int(mark["count"])
            element.provenance = (element.provenance or "") + f" Carries {mark['count']} parallel mark(s)."
            return


def _angle_arc_element(arc: dict, index: int) -> DetectedElement:
    vertex = [int(v) for v in arc["vertex"]]
    radius = int(arc["radius"])
    degrees = float(arc["degrees"])
    return DetectedElement(
        id=f"el_{index}",
        type=GeometryType.ANGLE,
        geometry={
            "vertex": vertex,
            "arms": [[int(v) for v in arm] for arm in arc["arms"]],
            "degrees": degrees,
            "angle_marker": True,
            "arc_radius": radius,
        },
        confidence=ANGLE_ARC_CONFIDENCE,
        confidence_level=classify_confidence(ANGLE_ARC_CONFIDENCE),
        needs_review=False,
        source="angle_arc",
        bbox=(vertex[0] - radius, vertex[1] - radius, 2 * radius, 2 * radius),
        semantic_properties={"degrees": degrees, "angle_marker": True},
        provenance=f"Angle arc (radius {radius} px) drawn at {tuple(vertex)} between two measured arms; measured {degrees:.1f} degrees.",
    )


def _angle_at_point(angle: DetectedElement, elements: list[DetectedElement]) -> list[ElementRelationship]:
    vertex = tuple(angle.geometry["vertex"])
    points = [e for e in elements if e.type is GeometryType.POINT]
    point = min(points, key=lambda p: math.dist(vertex, tuple(p.geometry["position"])), default=None)
    if point is None or math.dist(vertex, tuple(point.geometry["position"])) > ANGLE_POINT_PX:
        return []
    return [ElementRelationship(
        id=f"rel_{angle.id}_{point.id}_angle_at",
        type=RelationshipType.ANGLE_ASSOCIATION,
        element_ids=[angle.id, point.id],
        confidence=ANGLE_ARC_CONFIDENCE,
        confidence_level=classify_confidence(ANGLE_ARC_CONFIDENCE),
        needs_review=False,
        explanation=f"Angle arc {angle.id} is drawn at point {point.id}.",
    )]


def _associate_labels(elements: list[DetectedElement], relationships: list[ElementRelationship]) -> tuple[list[DetectedElement], list[TransformationExplanation]]:
    """Associate each label element with the best nearby geometry element."""
    from app.services.label_association import associate_label

    explanations: list[TransformationExplanation] = []
    label_elements = [e for e in elements if e.type is GeometryType.TEXT_LABEL]
    geometry_elements = [e for e in elements if e.type is not GeometryType.TEXT_LABEL]

    for label in label_elements:
        result = associate_label(label, geometry_elements)
        target_id = result.get("target_id")
        label.semantic_properties["association"] = {
            key: result.get(key)
            for key in ("target_id", "target_type", "reason", "confidence", "needs_review")
        }
        if target_id:
            label.associated_label_id = target_id
            for element in geometry_elements:
                if element.id == target_id and not element.associated_label_id:
                    element.associated_label_id = label.id
                    break
            explanations.append(TransformationExplanation(
                stage="label_association",
                element_id=label.id,
                message=result["explanation"],
            ))
        else:
            explanations.append(TransformationExplanation(
                stage="label_association",
                element_id=label.id,
                message="Retained text — no confident geometry association; teacher review recommended.",
            ))

    return elements, explanations