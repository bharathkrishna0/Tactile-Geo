"""Drawn dots, junction points and background filtering for relationships."""
import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType, RelationshipType
from app.services.diagram_analysis import _detect_points, analyze_diagram
from app.services.geometry_relations import background_line_ids, grid_like_ids, infer_relationships
from app.services.vectorization import detect_dots, extract_shapes


def _element(eid, gtype, geometry):
    return DetectedElement(
        id=eid, type=gtype, geometry=geometry, confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough",
    )


def _line(eid, start, end):
    return _element(eid, GeometryType.LINE_SEGMENT, {"start": tuple(start), "end": tuple(end)})


def _positions(points):
    return sorted(tuple(p.geometry["position"]) for p in points)


# --- drawn dots ------------------------------------------------------------------

def test_filled_disc_is_a_dot_and_strokes_are_not():
    binary = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(binary, (100, 150), 6, 255, -1)
    cv2.line(binary, (150, 50), (350, 250), 255, 2)
    cv2.rectangle(binary, (250, 20), (380, 60), 255, -1)
    dots = detect_dots(binary, 1.0)
    assert [d["points"][0] for d in dots] == [(100, 150)]


def test_filled_arrowhead_is_not_a_dot():
    binary = np.zeros((200, 300), dtype=np.uint8)
    cv2.line(binary, (20, 100), (250, 100), 255, 2)
    cv2.fillPoly(binary, [np.array([(270, 100), (248, 93), (248, 107)])], 255)
    assert detect_dots(binary, 1.0) == []


def test_dot_on_a_circle_centre_becomes_a_point_element():
    binary = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(binary, (200, 200), 120, 255, 2)
    cv2.circle(binary, (200, 200), 9, 255, -1)
    shapes = extract_shapes(binary)
    assert [s["points"][0] for s in shapes if s["type"] == "dot"] == [(200, 200)]
    semantic = analyze_diagram(shapes, 400, 400, [])
    points = [e for e in semantic.elements if e.type is GeometryType.POINT]
    assert len(points) == 1 and points[0].semantic_properties["drawn_dot"]


# --- junction points -------------------------------------------------------------

def test_radius_end_on_the_circle_is_a_point():
    circle = _element("c", GeometryType.CIRCLE, {"center": (200, 200), "radius": 100})
    radius = _line("r", (200, 200), (300, 200))
    points, _ = _detect_points([circle, radius], 10)
    assert (300, 200) in _positions(points)


def test_altitude_foot_on_the_base_is_a_point():
    base = _line("base", (100, 300), (400, 300))
    altitude = _line("alt", (250, 100), (250, 302))
    points, _ = _detect_points([base, altitude], 10)
    assert _positions(points) == [(250, 302)]


def test_labelled_crossing_is_a_point_and_unlabelled_is_not():
    a = _line("a", (100, 100), (300, 300))
    b = _line("b", (100, 300), (300, 100))
    assert _detect_points([a, b], 10)[0] == []
    label = {"text": "O", "bbox": [[205, 185], [215, 185], [215, 197], [205, 197]], "position": [400, 400]}
    points, _ = _detect_points([a, b], 10, [label])
    assert _positions(points) == [(200, 200)]


def test_vertex_label_is_judged_where_it_was_printed_not_where_braille_moved():
    triangle = _element("t", GeometryType.TRIANGLE, {"points": [(100, 300), (300, 300), (200, 120)]})
    label = {"text": "A", "bbox": [[85, 305], [97, 305], [97, 320], [85, 320]], "position": [40, 380]}
    points, _ = _detect_points([triangle], 10, [label])
    assert _positions(points) == [(100, 300)]


# --- background filtering ----------------------------------------------------------

def test_grid_family_is_background_for_parallels_but_a_lone_pair_is_kept():
    grid = [_line(f"g{i}", (100, 100 + 40 * i), (500, 100 + 40 * i)) for i in range(5)]
    assert grid_like_ids(grid) == {g.id for g in grid}
    assert not [r for r in infer_relationships(grid) if r.type is RelationshipType.PARALLEL_LINES]
    pair = [_line("p", (100, 100), (300, 100)), _line("q", (110, 160), (320, 160))]
    assert grid_like_ids(pair) == set()
    assert [r for r in infer_relationships(pair) if r.type is RelationshipType.PARALLEL_LINES]


def test_crossing_segments_intersect():
    a = _line("a", (100, 100), (300, 300))
    b = _line("b", (100, 300), (300, 100))
    rels = infer_relationships([a, b])
    assert [sorted(r.element_ids) for r in rels if r.type is RelationshipType.INTERSECTS] == [["a", "b"]]


# --- tables and grids --------------------------------------------------------------

def _table(columns, rows):
    horizontal = [_line(f"h{i}", (100, 100 + 40 * i), (100 + 60 * (columns - 1), 100 + 40 * i)) for i in range(rows)]
    vertical = [_line(f"v{i}", (100 + 60 * i, 100), (100 + 60 * i, 100 + 40 * (rows - 1))) for i in range(columns)]
    return horizontal + vertical


def test_table_rules_state_no_relationships():
    lines = _table(4, 4)
    pair_types = {RelationshipType.PARALLEL_LINES, RelationshipType.PERPENDICULAR_LINES,
                  RelationshipType.INTERSECTS, RelationshipType.CONNECTED_LINES}
    corner = {frozenset(("h0", "v0")), frozenset(("h0", "v3")), frozenset(("h3", "v0")), frozenset(("h3", "v3"))}
    rels = [r for r in infer_relationships(lines) if r.type in pair_types]
    assert {frozenset(r.element_ids) for r in rels} <= corner


def test_line_drawn_across_a_grid_does_not_intersect_its_rules():
    lines = _table(5, 5) + [_line("plot", (110, 330), (330, 110))]
    rels = infer_relationships(lines)
    assert not [r for r in rels if "plot" in r.element_ids]


def test_ticked_axes_on_a_grid_stay_perpendicular_scales():
    lines = _table(5, 5)
    ticks = [_element(f"t{i}", GeometryType.LINE_SEGMENT, {"start": (130 + 60 * i, 256), "end": (130 + 60 * i, 264)})
             for i in range(3)]
    for tick in ticks:
        tick.semantic_properties["role"] = "tick"
    assert "h4" not in background_line_ids(lines, ticks)
    assert {"h1", "v1"} <= background_line_ids(lines, ticks)


def test_crossing_segments_without_a_grid_still_intersect_and_touching_ones_do_not():
    a = _line("a", (100, 100), (300, 300))
    b = _line("b", (100, 300), (300, 100))
    c = _line("c", (400, 100), (400, 300))
    d = _line("d", (330, 200), (395, 260))
    rels = infer_relationships([a, b, c, d])
    pairs = {frozenset(r.element_ids) for r in rels if r.type is RelationshipType.INTERSECTS}
    assert frozenset(("a", "b")) in pairs and frozenset(("c", "d")) not in pairs
