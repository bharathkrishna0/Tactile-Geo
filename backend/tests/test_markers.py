"""Convention marks read against measured geometry: angle arcs, parallel chevrons,
equal-length hash marks and right-angle squares."""
import math

import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType, RelationshipType
from app.services.diagram_analysis import analyze_diagram
from app.services.geometry_relations import infer_relationships
from app.services.markers import detect_angle_arcs, detect_parallel_chevrons
from app.services.right_angles import detect_right_angle_markers


def _line_shape(a, b):
    return {"type": "line", "points": [a, b]}


def _draw(size, lines, thickness=3):
    binary = np.zeros(size, dtype=np.uint8)
    for a, b in lines:
        cv2.line(binary, a, b, 255, thickness)
    return binary


def _element(eid, gtype, geometry):
    return DetectedElement(
        id=eid, type=gtype, geometry=geometry, confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough",
    )


def _line(eid, start, end):
    return _element(eid, GeometryType.LINE_SEGMENT, {"start": tuple(start), "end": tuple(end)})


def _tick(eid, centre, horizontal_host=True, half=6):
    x, y = centre
    start, end = ((x, y - half), (x, y + half)) if horizontal_host else ((x - half, y), (x + half, y))
    element = _line(eid, start, end)
    element.semantic_properties["role"] = "tick"
    return element


def _types(rels, rtype):
    return [r for r in rels if r.type is rtype]


# --- angle arcs ------------------------------------------------------------------

VERTEX = (100, 300)
ARM_A = (400, 300)
ARM_B = (100 + round(300 * math.cos(math.radians(-60))), 300 + round(300 * math.sin(math.radians(-60))))


def _angle_figure(arc_from=-60, arc_to=0, radius=40):
    binary = _draw((400, 500), [(VERTEX, ARM_A), (VERTEX, ARM_B)])
    cv2.ellipse(binary, VERTEX, (radius, radius), 0, arc_from, arc_to, 255, 2)
    return binary, [_line_shape(VERTEX, ARM_A), _line_shape(VERTEX, ARM_B)]


def test_arc_between_two_measured_arms_is_an_angle_marker():
    binary, shapes = _angle_figure()
    arcs = detect_angle_arcs(binary, shapes, 1.5)
    assert len(arcs) == 1
    assert math.dist(arcs[0]["vertex"], VERTEX) <= 3
    assert abs(arcs[0]["degrees"] - 60) <= 3
    assert abs(arcs[0]["radius"] - 40) <= 3


def test_arc_that_overshoots_the_arms_is_rejected():
    binary, shapes = _angle_figure(arc_from=-110, arc_to=0)
    assert detect_angle_arcs(binary, shapes, 1.5) == []


def test_arms_without_an_arc_give_no_marker():
    binary = _draw((400, 500), [(VERTEX, ARM_A), (VERTEX, ARM_B)])
    shapes = [_line_shape(VERTEX, ARM_A), _line_shape(VERTEX, ARM_B)]
    assert detect_angle_arcs(binary, shapes, 1.5) == []


def test_arc_inside_a_label_box_is_ignored():
    binary, shapes = _angle_figure()
    box = [(130, 250), (150, 250), (150, 310), (130, 310)]
    big = [(VERTEX[0] + 10, 240), (VERTEX[0] + 60, 240), (VERTEX[0] + 60, 298), (VERTEX[0] + 10, 298)]
    assert detect_angle_arcs(binary, shapes, 1.5, [box, big]) == []


def test_angle_arc_is_associated_with_the_vertex_point():
    binary, shapes = _angle_figure()
    arcs = detect_angle_arcs(binary, shapes, 1.5)
    semantic = analyze_diagram(shapes, 500, 400, [], angle_arcs=arcs)
    angles = [e for e in semantic.elements if e.type is GeometryType.ANGLE and e.geometry.get("angle_marker")]
    assert len(angles) == 1
    links = _types(semantic.relationships, RelationshipType.ANGLE_ASSOCIATION)
    assert [r.element_ids[0] for r in links] == [angles[0].id]
    point = next(e for e in semantic.elements if e.id == links[0].element_ids[1])
    assert math.dist(point.geometry["position"], VERTEX) <= 6


def test_right_angle_vertex_is_not_also_an_angle_arc():
    binary, shapes = _angle_figure()
    assert detect_angle_arcs(binary, shapes, 1.5, exclude_vertices=[VERTEX]) == []


# --- parallel chevrons ----------------------------------------------------------

def _chevron(binary, apex, size=10, count=1, gap=8):
    for k in range(count):
        x = apex[0] + k * gap
        cv2.line(binary, (x, apex[1]), (x - size, apex[1] - size), 255, 2)
        cv2.line(binary, (x, apex[1]), (x - size, apex[1] + size), 255, 2)


def _parallel_figure(counts=(1, 1)):
    lines = [((60, 100), (440, 100)), ((60, 260), (440, 260))]
    binary = _draw((360, 500), lines)
    _chevron(binary, (250, 100), count=counts[0])
    _chevron(binary, (250, 260), count=counts[1])
    return binary, [_line_shape(*line) for line in lines]


def test_matching_chevrons_mark_two_lines_parallel():
    binary, shapes = _parallel_figure()
    marks = detect_parallel_chevrons(binary, shapes, 1.5)
    assert sorted(m["count"] for m in marks) == [1, 1]
    semantic = analyze_diagram(shapes, 500, 360, [], parallel_marks=marks)
    parallels = _types(semantic.relationships, RelationshipType.PARALLEL_LINES)
    assert len(parallels) == 1
    assert parallels[0].confidence == 0.95 and "parallel marks" in parallels[0].explanation


def test_double_chevrons_are_counted():
    binary, shapes = _parallel_figure(counts=(2, 2))
    assert sorted(m["count"] for m in detect_parallel_chevrons(binary, shapes, 1.5)) == [2, 2]


def test_plain_lines_and_end_arrowheads_are_not_chevrons():
    lines = [((60, 100), (440, 100)), ((60, 260), (440, 260))]
    binary = _draw((360, 500), lines)
    cv2.fillPoly(binary, [np.array([(450, 100), (430, 92), (430, 108)])], 255)
    assert detect_parallel_chevrons(binary, [_line_shape(*line) for line in lines], 1.5) == []


def test_matching_marks_on_lines_that_measure_non_parallel_are_not_parallel():
    a, b = _line("a", (60, 100), (440, 100)), _line("b", (60, 300), (440, 200))
    a.semantic_properties["parallel_marks"] = 1
    b.semantic_properties["parallel_marks"] = 1
    rels = infer_relationships([a, b])
    assert _types(rels, RelationshipType.PARALLEL_LINES) == []


# --- equal-length hash marks -----------------------------------------------------

def test_segments_with_matching_hash_marks_are_equal_length():
    a, b = _line("a", (100, 100), (300, 100)), _line("b", (100, 300), (300, 300))
    rels = infer_relationships([a, b, _tick("t1", (200, 100)), _tick("t2", (200, 300))])
    equal = _types(rels, RelationshipType.EQUAL_LENGTH)
    assert [sorted(r.element_ids) for r in equal] == [["a", "b"]]


def test_different_hash_counts_are_not_equal_length():
    a, b = _line("a", (100, 100), (300, 100)), _line("b", (100, 300), (300, 300))
    ticks = [_tick("t1", (200, 100)), _tick("t2", (195, 300)), _tick("t3", (205, 300))]
    assert _types(infer_relationships([a, b, *ticks]), RelationshipType.EQUAL_LENGTH) == []


def test_hash_marks_on_measurably_unequal_segments_are_rejected():
    a, b = _line("a", (100, 100), (300, 100)), _line("b", (100, 300), (240, 300))
    rels = infer_relationships([a, b, _tick("t1", (200, 100)), _tick("t2", (170, 300))])
    assert _types(rels, RelationshipType.EQUAL_LENGTH) == []


def test_axis_ticks_are_not_hash_marks():
    a, b = _line("a", (100, 100), (500, 100)), _line("b", (100, 300), (500, 300))
    ticks = [_tick(f"a{x}", (x, 100)) for x in range(140, 500, 40)]
    ticks += [_tick(f"b{x}", (x, 300)) for x in range(140, 500, 40)]
    assert _types(infer_relationships([a, b, *ticks]), RelationshipType.EQUAL_LENGTH) == []


# --- right-angle squares ---------------------------------------------------------

def _altitude_figure():
    base = ((100, 300), (500, 300))
    binary = _draw((400, 600), [base])
    for y in range(110, 270, 16):
        cv2.line(binary, (300, y), (300, min(y + 8, 274)), 255, 3)
    cv2.line(binary, (300, 282), (318, 282), 255, 2)
    cv2.line(binary, (318, 282), (318, 298), 255, 2)
    return binary, [_line_shape(*base)]


def test_square_at_a_dashed_altitude_foot_is_a_right_angle_marker():
    binary, shapes = _altitude_figure()
    markers = detect_right_angle_markers(binary, shapes, 1.5)
    assert len(markers) == 1
    assert math.dist(markers[0]["vertex"], (300, 300)) <= 6


def test_square_inside_a_text_box_is_not_a_marker():
    binary, shapes = _altitude_figure()
    box = [(290, 270), (330, 270), (330, 299), (290, 299)]
    assert detect_right_angle_markers(binary, shapes, 1.5, [box]) == []


def test_free_floating_l_shape_is_not_a_marker():
    base = ((100, 300), (500, 300))
    binary = _draw((400, 600), [base])
    cv2.line(binary, (300, 120), (318, 120), 255, 2)
    cv2.line(binary, (318, 120), (318, 138), 255, 2)
    assert detect_right_angle_markers(binary, [_line_shape(*base)], 1.5) == []
