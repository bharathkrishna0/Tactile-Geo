"""Circles and arcs that share ink with other strokes."""
import math

import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.curves import recover_curves
from app.services.diagram_analysis import analyze_diagram
from app.services.tactile_qa import outline_segments
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
