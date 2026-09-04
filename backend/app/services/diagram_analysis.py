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
from app.services.geometry_relations import infer_relationships


def analyze_diagram(shapes: list[dict], width: int, height: int, labels: list[dict]) -> SemanticGeometry:
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

    # Detect significant points (shared line endpoints, shape vertices).
    point_elements, point_explanations = _detect_points(elements, id_counter)
    id_counter += len(point_elements)
    elements.extend(point_elements)
    explanations.extend(point_explanations)

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
        semantic_properties={"length": round(length, 2)},
    )
    return [element], []


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
        semantic_properties={"area": round(area, 2), "perimeter": round(perimeter, 2), "vertex_count": len(points)},
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


def _classify_contour(points: list[tuple[int, int]], area: float, perimeter: float, circularity: float) -> tuple[GeometryType, float]:
    n = len(points)
    if n == 3:
        return GeometryType.TRIANGLE, 0.15
    if n == 4:
        if _is_rectangle(points):
            return GeometryType.RECTANGLE, 0.15
        return GeometryType.POLYGON, 0.08
    if n >= 5 and circularity > 0.7:
        return GeometryType.CIRCLE, 0.15
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


def _detect_points(elements: list[DetectedElement], start_id: int) -> tuple[list[DetectedElement], list[TransformationExplanation]]:
    """Create point elements at shared endpoints / polygon vertices."""
    significant: dict[tuple[int, int], int] = {}
    explanations: list[TransformationExplanation] = []

    for element in elements:
        if element.type is GeometryType.LINE_SEGMENT:
            for coord in (element.geometry["start"], element.geometry["end"]):
                significant[coord] = significant.get(coord, 0) + 1
        elif element.type in (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON):
            points = element.geometry.get("points", [])
            for coord in points:
                significant[coord] = significant.get(coord, 0) + 1

    # A coordinate shared by two or more features, or a polygon vertex, is a real point.
    point_elements: list[DetectedElement] = []
    pid = start_id
    for coord, count in significant.items():
        if count >= 2:
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
            )
            pid += 1
            point_elements.append(element)
            explanations.append(TransformationExplanation(
                stage="diagram_analysis",
                element_id=element.id,
                message=f"Detected point at ({coord[0]},{coord[1]}) shared by {count} features.",
            ))
    return point_elements, explanations


def _detect_angles(elements: list[DetectedElement], start_id: int, existing: list[ElementRelationship]) -> tuple[list[DetectedElement], list[TransformationExplanation], list[ElementRelationship]]:
    """Detect angle elements where two connected line segments meet."""
    segments = [e for e in elements if e.type is GeometryType.LINE_SEGMENT]
    angle_elements: list[DetectedElement] = []
    explanations: list[TransformationExplanation] = []
    extra_relationships: list[ElementRelationship] = []
    aid = start_id
    used: set[tuple] = set()

    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            a0, a1 = a.geometry["start"], a.geometry["end"]
            b0, b1 = b.geometry["start"], b.geometry["end"]
            # Find the common vertex.
            common = None
            for pa in (a0, a1):
                for pb in (b0, b1):
                    if _endpoint_near(tuple(pa), tuple(pb), tolerance=8.0):
                        common = tuple(pa)
                        break
                if common:
                    break
            if common is None:
                continue
            other_a = a1 if tuple(a0) == common else a0
            other_b = b1 if tuple(b0) == common else b0
            angle = _angle_between_vectors(other_a, common, other_b)
            if angle < 5.0 or angle > 175.0:
                continue
            key = tuple(sorted([a.id, b.id]))
            if key in used:
                continue
            used.add(key)
            element = DetectedElement(
                id=f"el_{aid}",
                type=GeometryType.ANGLE,
                geometry={"vertex": list(common), "arms": [list(other_a), list(other_b)], "degrees": round(angle, 1)},
                confidence=0.6,
                confidence_level=ConfidenceLevel.MEDIUM,
                needs_review=True,
                source="heuristic",
                bbox=(common[0] - 5, common[1] - 5, 10, 10),
                semantic_properties={"degrees": round(angle, 1)},
            )
            aid += 1
            angle_elements.append(element)
            extra_relationships.append(ElementRelationship(
                id=f"rel_{a.id}_{b.id}_angle",
                type=RelationshipType.ANGLE_BETWEEN,
                element_ids=[a.id, b.id],
                confidence=0.6,
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


def _associate_labels(elements: list[DetectedElement], relationships: list[ElementRelationship]) -> tuple[list[DetectedElement], list[TransformationExplanation]]:
    """Associate each label element with the best nearby geometry element."""
    from app.services.label_association import associate_label

    explanations: list[TransformationExplanation] = []
    label_elements = [e for e in elements if e.type is GeometryType.TEXT_LABEL]
    geometry_elements = [e for e in elements if e.type is not GeometryType.TEXT_LABEL]

    for label in label_elements:
        result = associate_label(label, geometry_elements)
        target_id = result.get("target_id")
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