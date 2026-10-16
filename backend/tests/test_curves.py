"""Circles and arcs that share ink with other strokes."""
import math

import cv2
import numpy as np
import pytest

from app.api.sessions import _semantic_to_dict
from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.schemas.session import DetectedShape, SemanticGeometrySchema
from app.services.curves import recover_curves
from app.services.diagram_analysis import analyze_diagram
from app.services.tactile_qa import outline_segments
from app.services.tactile_simplification import essential_ids
from app.services.tactile_svg import is_embossed_geometry, render_tactile_svg
from app.services.vectorization import extract_shapes, stroke_half_width


def _square_with_curves(thickness=2):
    """Square ABCD, a semicircle on AB and a circle touching DC and BC."""
    binary = np.zeros((600, 600), dtype=np.uint8)
    cv2.rectangle(binary, (100, 100), (500, 500), 255, thickness)
    cv2.ellipse(binary, (300, 500), (200, 200), 0, 180, 360, 255, thickness)
    cv2.circle(binary, (400, 200), 100, 255, thickness)
    cv2.line(binary, (400, 200), (400, 500), 255, thickness)
    return binary


def _curves(binary):
    shapes = extract_shapes(binary, 50)
    return recover_curves(binary, shapes, stroke_half_width(binary))


def test_circle_and_semicircle_sharing_ink_with_a_square_are_recovered():
    shapes = _curves(_square_with_curves())
    circles = [s for s in shapes if s["type"] == "circle"]
    arcs = [s for s in shapes if s["type"] == "arc"]
    assert len(circles) == 1 and len(arcs) == 1
    assert math.dist(circles[0]["center"], (400, 200)) <= 3 and abs(circles[0]["radius"] - 100) <= 3
    arc = arcs[0]
    assert math.dist(arc["center"], (300, 500)) <= 3 and abs(arc["radius"] - 200) <= 3
    assert abs(arc["sweep_deg"] - 180) <= 8
    # The arc spans the upper half: image y grows downward, so 180..360 degrees.
    mid = math.radians(arc["start_deg"] + arc["sweep_deg"] / 2)
    assert math.sin(mid) < -0.9


def test_straight_lines_only_are_not_curves():
    binary = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(binary, (50, 50), (350, 350), 255, 2)
    cv2.line(binary, (50, 50), (350, 350), 255, 2)
    cv2.line(binary, (50, 350), (200, 50), 255, 2)
    assert not [s for s in _curves(binary) if s["type"] in ("circle", "arc")]


def test_small_rounds_such_as_letters_are_not_curves():
    binary = np.zeros((300, 300), dtype=np.uint8)
    cv2.line(binary, (20, 150), (280, 150), 255, 2)
    cv2.circle(binary, (150, 140), 8, 255, 2)
    assert not [s for s in _curves(binary) if s["type"] in ("circle", "arc")]


def test_hough_chords_along_a_recovered_curve_are_dropped():
    binary = _square_with_curves()
    shapes = extract_shapes(binary, 50) + [{"type": "line", "points": [(147, 371), (177, 342)]}]
    kept = recover_curves(binary, shapes, stroke_half_width(binary))
    assert not any(s["type"] == "line" and s["points"] == [(147, 371), (177, 342)] for s in kept)


def test_recovered_curves_reach_the_tactile_svg_and_qa():
    binary = _square_with_curves()
    shapes = _curves(binary)
    semantic = analyze_diagram(shapes, 600, 600, [])
    arc = next(e for e in semantic.elements if e.type is GeometryType.ARC)
    circle = next(e for e in semantic.elements if e.type is GeometryType.CIRCLE)
    assert is_embossed_geometry(arc) and is_embossed_geometry(circle)
    assert len(outline_segments(arc)) > 10
    svg = render_tactile_svg(semantic.elements, 600, 600)
    assert 'class="arc"' in svg and "<circle" in svg


def test_arc_without_measured_rim_is_not_embossed():
    element = DetectedElement(
        id="el_0", type=GeometryType.ARC, geometry={}, confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="model_b",
    )
    assert not is_embossed_geometry(element)


def test_a_circle_traced_as_a_polygon_becomes_one_circle():
    binary = np.zeros((600, 600), dtype=np.uint8)
    cv2.circle(binary, (300, 300), 200, 255, 2)
    polygon = [
        (round(300 + 200 * math.cos(2 * math.pi * i / 15)), round(300 + 200 * math.sin(2 * math.pi * i / 15)))
        for i in range(15)
    ]
    shapes = recover_curves(binary, [{"type": "contour", "points": polygon}], stroke_half_width(binary))
    assert [s["type"] for s in shapes] == ["circle"]
    assert math.dist(shapes[0]["center"], (300, 300)) <= 2 and abs(shapes[0]["radius"] - 200) <= 2


@pytest.mark.parametrize("radius, thickness", [(200, 2), (43, 3)])
def test_a_regular_polygon_stays_a_polygon(radius, thickness):
    binary = np.zeros((600, 600), dtype=np.uint8)
    octagon = [
        (round(300 + radius * math.cos(2 * math.pi * i / 8)), round(300 + radius * math.sin(2 * math.pi * i / 8)))
        for i in range(8)
    ]
    cv2.polylines(binary, [np.array(octagon, np.int32)], True, 255, thickness)
    shapes = recover_curves(binary, [{"type": "contour", "points": octagon}], stroke_half_width(binary))
    assert [s["type"] for s in shapes] == ["contour"]


def test_a_graph_curve_is_not_mistaken_for_an_arc():
    binary = np.zeros((600, 800), dtype=np.uint8)
    cv2.line(binary, (40, 300), (760, 300), 255, 2)
    cv2.line(binary, (400, 40), (400, 560), 255, 2)
    xs = np.arange(40, 761)
    curve = np.c_[xs, 300 - 0.0012 * (xs - 400) ** 2 + 120].astype(np.int32)
    cv2.polylines(binary, [curve], False, 255, 2)
    assert not [s for s in _curves(binary) if s["type"] in ("circle", "arc")]


def test_a_large_recovered_arc_is_essential():
    semantic = analyze_diagram(_curves(_square_with_curves()), 600, 600, [])
    arc = next(e for e in semantic.elements if e.type is GeometryType.ARC)
    assert arc.id in essential_ids(semantic.elements, semantic.relationships, 0.5)


def test_recovered_curves_fit_the_session_schemas():
    shapes = _curves(_square_with_curves())
    for shape in shapes:
        DetectedShape(**{k: v for k, v in shape.items() if k in DetectedShape.model_fields})
    SemanticGeometrySchema(**_semantic_to_dict(analyze_diagram(shapes, 600, 600, [])))


def test_a_circle_traced_with_its_tangent_becomes_a_circle():
    binary = np.zeros((600, 600), dtype=np.uint8)
    cv2.circle(binary, (300, 300), 150, 255, 2)
    cv2.line(binary, (450, 150), (450, 450), 255, 2)
    corners = [
        (round(300 + 150 * math.cos(2 * math.pi * i / 12)), round(300 + 150 * math.sin(2 * math.pi * i / 12)))
        for i in range(1, 12)
    ]
    outline = [(450, 150), (450, 450)] + corners[1:11]
    shapes = recover_curves(
        binary,
        [{"type": "contour", "points": outline}, {"type": "line", "points": [(450, 150), (450, 450)]}],
        stroke_half_width(binary),
    )
    circles = [s for s in shapes if s["type"] == "circle"]
    assert len(circles) == 1 and math.dist(circles[0]["center"], (300, 300)) <= 3
    assert abs(circles[0]["radius"] - 150) <= 3
    assert not any(s["type"] == "contour" for s in shapes)
    assert any(s["type"] == "line" for s in shapes)
