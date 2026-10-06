"""Geometry, relationship and OCR changes measured on the math benchmark."""
import cv2
import numpy as np

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services.diagram_analysis import _associate_labels
from app.services.geometry_relations import infer_relationships
from app.services.ocr import OcrDetection, _drop_dash_runs, glyph_candidates
from app.services.right_angles import detect_right_angle_markers
from app.services.tactile_simplification import simplify_geometry
from app.services.image_preprocessing import preprocess_image
from app.services.tactile_svg import page_layout
from app.services.tactile_rules import TACTILE_RULES
from app.services.vectorization import reconstruct_closed_shapes, recover_ticks


def _element(eid, gtype, geometry, role=None):
    return DetectedElement(
        id=eid, type=gtype, geometry=geometry, confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough",
        semantic_properties={"role": role} if role else {},
    )


def _line(eid, start, end, role=None):
    return _element(eid, GeometryType.LINE_SEGMENT, {"start": list(start), "end": list(end)}, role)


def _shape_line(a, b):
    return {"type": "line", "points": [a, b]}


def _types(relationships, kind):
    return [r for r in relationships if r.type is kind]


# --- closed-shape reconstruction ---------------------------------------------

def test_triangle_with_a_tail_becomes_one_outline_and_the_tail_stays_a_line():
    a, b, c, tail = (100, 300), (300, 300), (200, 120), (400, 300)
    shapes = [_shape_line(a, b), _shape_line(b, c), _shape_line(c, a), _shape_line(b, tail)]
    result = reconstruct_closed_shapes(shapes)
    contours = [s for s in result if s["type"] == "contour"]
    lines = [s for s in result if s["type"] == "line"]
    assert len(contours) == 1 and len(contours[0]["points"]) == 3
    assert contours[0]["source"] == "joined_sides"
    assert lines == [_shape_line(b, tail)]


def test_rectangle_with_a_diagonal_is_ambiguous_and_left_as_lines():
    p = [(100, 100), (300, 100), (300, 250), (100, 250)]
    shapes = [_shape_line(p[i], p[(i + 1) % 4]) for i in range(4)] + [_shape_line(p[0], p[2])]
    assert reconstruct_closed_shapes(shapes) == shapes


# --- tick recovery -------------------------------------------------------------

def test_ticks_must_cross_their_line():
    binary = np.zeros((200, 500), dtype=np.uint8)
    cv2.line(binary, (50, 100), (450, 100), 255, 2)
    cv2.line(binary, (200, 90), (200, 110), 255, 2)  # crosses the line: a tick
    cv2.line(binary, (320, 100), (320, 80), 255, 2)  # one side only: not a tick
    ticks = recover_ticks(binary, [_shape_line((50, 100), (450, 100))], 1.0)
    assert len(ticks) == 1
    (x0, _), (x1, _) = ticks[0]["points"]
    assert abs(x0 - 200) <= 2 and abs(x1 - 200) <= 2
    assert ticks[0]["role"] == "tick"


def test_short_ticks_are_lengthened_to_the_touch_minimum():
    width, height = 800, 600
    tick = _line("tick", (400, 295), (400, 305), role="tick")
    host = _line("host", (100, 300), (700, 300))
    simplified = simplify_geometry(SemanticGeometry(image_width=width, image_height=height, elements=[host, tick]))
    out = next(e for e in simplified.elements if e.id == "tick")
    length_mm = abs(out.geometry["end"][1] - out.geometry["start"][1]) * page_layout(width, height).mm_per_px
    assert length_mm >= TACTILE_RULES.minimum_feature_size_mm
    assert out.geometry["start"][0] == out.geometry["end"][0] == 400
    assert any(a.action == "enlarged_tick" and a.element_id == "tick" for a in simplified.actions)


# --- relationships -------------------------------------------------------------

def test_point_on_a_polygon_corner_is_a_vertex_of_it():
    triangle = _element("tri", GeometryType.TRIANGLE, {"points": [[100, 300], [300, 300], [200, 120]]})
    point = _element("p", GeometryType.POINT, {"position": [201, 122]})
    rels = infer_relationships([triangle, point])
    assert [r.element_ids for r in _types(rels, RelationshipType.VERTEX_OF)] == [["p", "tri"]]


def test_parallel_needs_lines_that_face_each_other():
    a = _line("a", (100, 100), (300, 100))
    facing = _line("b", (120, 160), (320, 160))
    far_off = _line("c", (900, 600), (1100, 600))
    rels = infer_relationships([a, facing, far_off])
    pairs = [sorted(r.element_ids) for r in _types(rels, RelationshipType.PARALLEL_LINES)]
    assert pairs == [["a", "b"]]


def test_altitude_is_perpendicular_to_a_polygon_side():
    triangle = _element("tri", GeometryType.TRIANGLE, {"points": [[100, 300], [300, 300], [200, 120]]})
    altitude = _line("alt", (200, 120), (200, 300))
    rels = infer_relationships([triangle, altitude])
    assert [r.element_ids for r in _types(rels, RelationshipType.PERPENDICULAR_LINES)] == [["alt", "tri"]]


def test_tick_lies_on_its_host_line_and_is_not_paired_with_it():
    host = _line("host", (100, 300), (700, 300))
    tick = _line("tick", (400, 290), (400, 310), role="tick")
    rels = infer_relationships([host, tick])
    assert [r.element_ids for r in _types(rels, RelationshipType.LIES_ON)] == [["tick", "host"]]
    assert not _types(rels, RelationshipType.PERPENDICULAR_LINES)


# --- right-angle markers -------------------------------------------------------

def test_right_angle_marker_at_a_t_junction():
    binary = np.zeros((400, 500), dtype=np.uint8)
    cv2.line(binary, (50, 300), (450, 300), 255, 2)
    cv2.line(binary, (250, 300), (250, 80), 255, 2)
    cv2.line(binary, (250, 285), (265, 285), 255, 2)
    cv2.line(binary, (265, 285), (265, 300), 255, 2)
    shapes = [_shape_line((50, 300), (450, 300)), _shape_line((250, 300), (250, 80))]
    markers = detect_right_angle_markers(binary, shapes, 1.0)
    assert len(markers) == 1
    assert markers[0]["vertex"] == (250, 300)


# --- OCR -----------------------------------------------------------------------

def test_glyph_candidates_find_an_isolated_letter_but_not_the_drawing():
    gray = np.full((300, 500), 255, dtype=np.uint8)
    cv2.line(gray, (40, 250), (460, 250), 0, 2)
    cv2.putText(gray, "B", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    boxes = glyph_candidates(gray, [])
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 195 <= x <= 205 and 125 <= y <= 135 and h <= 30


def test_glyph_candidates_skip_text_already_read():
    gray = np.full((300, 500), 255, dtype=np.uint8)
    cv2.putText(gray, "B", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    read = OcrDetection(text="B", bbox=[(196, 128), (220, 128), (220, 152), (196, 152)], confidence=0.9)
    assert glyph_candidates(gray, [read]) == []


def test_dash_runs_are_not_labels():
    def det(text, x, y):
        return OcrDetection(text=text, bbox=[(x, y), (x + 10, y), (x + 10, y + 12), (x, y + 12)], confidence=0.9)
    dashes = [det("-", 100 + 20 * i, 200) for i in range(4)]
    label = det("A", 300, 50)
    assert _drop_dash_runs(dashes + [label]) == [label]


def test_thin_marker_joined_to_bold_sides_survives_preprocessing():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.line(image, (40, 160), (180, 160), (0, 0, 0), 4)
    cv2.line(image, (60, 20), (60, 160), (0, 0, 0), 4)
    cv2.line(image, (62, 139), (82, 139), (0, 0, 0), 1)
    cv2.line(image, (82, 139), (82, 158), (0, 0, 0), 1)
    binary = preprocess_image(image)
    assert binary[138:141, 66:80].any()
    assert binary[142:156, 81:84].any()
    # A bold line's anti-aliased fringe is not restored as extra strokes.
    assert not binary[150:156, 100:170].any()


def test_label_association_records_its_evidence():
    point = DetectedElement(id="p", type=GeometryType.POINT, geometry={"position": (100, 100)}, confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="t")
    line = DetectedElement(id="l", type=GeometryType.LINE_SEGMENT, geometry={"start": (0, 300), "end": (400, 300)}, confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="t")
    label = DetectedElement(id="t", type=GeometryType.TEXT_LABEL, geometry={"text": "A", "position": (112, 92)}, confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="t", bbox=(106, 84, 12, 16))
    _associate_labels([point, line, label], [])
    assert label.associated_label_id == "p"
    evidence = label.semantic_properties["association"]
    assert evidence["target_id"] == "p"
    assert evidence["target_type"] == "point"
    assert evidence["reason"] == "point"
    assert 0 < evidence["confidence"] <= 1
