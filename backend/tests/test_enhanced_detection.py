"""Tests for Milestone 1: Enhanced geometry detection.

Exercises ellipse detection, improved circle classification, and the new
ELLIPSE geometry type through the real vectorization and diagram_analysis
pipeline.
"""
import cv2
import numpy as np

from app.models.geometry import GeometryType
from app.services.diagram_analysis import analyze_diagram
from app.services.image_preprocessing import preprocess_image
from app.services.vectorization import extract_shapes, shapes_to_svg


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[190, 8], [210, 8], [210, 28], [190, 28]], "E", 0.95),
        ]


class FakeLouis:
    def translateString(self, tables, text):
        return {"E": "⠠⠑"}.get(text, text)


# --- ellipse detection in vectorization ---

def test_vectorization_detects_ellipse_shape(fixture_directory):
    image = cv2.imread(str(fixture_directory / "ellipse_worksheet.png"))
    filtered = preprocess_image(image, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    ellipses = [s for s in shapes if s["type"] == "ellipse"]
    assert len(ellipses) >= 1
    ellipse = ellipses[0]
    assert "center" in ellipse
    assert "semi_axes" in ellipse
    assert "angle" in ellipse
    assert ellipse["semi_axes"][0] > ellipse["semi_axes"][1]


def test_vectorization_ellipse_has_reasonable_geometry(fixture_directory):
    image = cv2.imread(str(fixture_directory / "ellipse_worksheet.png"))
    filtered = preprocess_image(image, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    ellipses = [s for s in shapes if s["type"] == "ellipse"]
    assert len(ellipses) >= 1
    ellipse = ellipses[0]
    cx, cy = ellipse["center"]
    assert 100 < cx < 300
    assert 80 < cy < 220
    rx, ry = ellipse["semi_axes"]
    assert rx > 30
    assert ry > 15


def test_vectorization_does_not_detect_ellipse_for_circle(fixture_directory):
    image = cv2.imread(str(fixture_directory / "circle_worksheet.png"))
    filtered = preprocess_image(image, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    ellipses = [s for s in shapes if s["type"] == "ellipse"]
    assert len(ellipses) == 0


def test_vectorization_ellipse_skips_tiny_contours():
    tiny = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.ellipse(tiny, (50, 50), (15, 8), 30, 0, 360, (0, 0, 0), thickness=2)
    filtered = preprocess_image(tiny, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    ellipses = [s for s in shapes if s["type"] == "ellipse"]
    assert len(ellipses) == 0


# --- shapes_to_svg handles ellipses ---

def test_shapes_to_svg_renders_ellipse():
    shapes = [{"type": "ellipse", "center": (200, 150), "semi_axes": (100, 50), "angle": 0, "area": 15707.96}]
    svg = shapes_to_svg(shapes, 400, 300)

    assert "<ellipse" in svg
    assert 'cx="200"' in svg
    assert 'cy="150"' in svg
    assert 'rx="100"' in svg
    assert 'ry="50"' in svg


def test_shapes_to_svg_mixed_types():
    shapes = [
        {"type": "line", "points": [(10, 10), (100, 10)]},
        {"type": "contour", "points": [(50, 50), (100, 50), (75, 100)]},
        {"type": "ellipse", "center": (200, 150), "semi_axes": (80, 40), "angle": 0, "area": 10053.1},
    ]
    svg = shapes_to_svg(shapes, 400, 300)

    assert "<line" in svg
    assert "<polygon" in svg
    assert "<ellipse" in svg


# --- diagram_analysis handles ellipse shapes ---

def test_diagram_analysis_classifies_ellipse_element():
    shapes = [
        {"type": "ellipse", "center": (200, 150), "semi_axes": (100, 50), "angle": 0, "area": 15707.96,
         "contour_points": [(100, 150), (200, 100), (300, 150), (200, 200)]},
    ]
    semantic = analyze_diagram(shapes, 400, 300, [])

    ellipses = [e for e in semantic.elements if e.type is GeometryType.ELLIPSE]
    assert len(ellipses) == 1
    assert ellipses[0].confidence >= 0.6
    assert ellipses[0].source == "contour"
    assert "center" in ellipses[0].geometry
    assert "semi_axes" in ellipses[0].geometry


def test_diagram_analysis_ellipse_has_semantic_properties():
    shapes = [
        {"type": "ellipse", "center": (200, 150), "semi_axes": (100, 50), "angle": 45.0, "area": 15707.96,
         "contour_points": [(100, 150), (200, 100), (300, 150), (200, 200)]},
    ]
    semantic = analyze_diagram(shapes, 400, 300, [])

    ellipse = [e for e in semantic.elements if e.type is GeometryType.ELLIPSE][0]
    assert "area" in ellipse.semantic_properties
    assert "aspect_ratio" in ellipse.semantic_properties
    assert "rotation_deg" in ellipse.semantic_properties
    assert ellipse.semantic_properties["aspect_ratio"] == 0.5
    assert ellipse.semantic_properties["rotation_deg"] == 45.0


def test_diagram_analysis_ellipse_with_label():
    shapes = [
        {"type": "ellipse", "center": (200, 150), "semi_axes": (100, 50), "angle": 0, "area": 15707.96,
         "contour_points": [(100, 150), (200, 100), (300, 150), (200, 200)]},
    ]
    labels = [{"text": "E", "confidence": 0.95, "position": [210, 145]}]
    semantic = analyze_diagram(shapes, 400, 300, labels)

    text_labels = [e for e in semantic.elements if e.type is GeometryType.TEXT_LABEL]
    ellipses = [e for e in semantic.elements if e.type is GeometryType.ELLIPSE]
    assert len(text_labels) == 1
    assert len(ellipses) == 1
    assert text_labels[0].associated_label_id == ellipses[0].id


# --- improved circle classification ---

def test_circle_requires_higher_circularity():
    shapes = [
        {"type": "contour", "points": [(10, 10), (60, 10), (60, 60), (10, 60), (35, 5)]},
    ]
    semantic = analyze_diagram(shapes, 200, 200, [])

    circles = [e for e in semantic.elements if e.type is GeometryType.CIRCLE]
    polygons = [e for e in semantic.elements if e.type is GeometryType.POLYGON]
    assert len(circles) == 0 or len(polygons) >= 1


def test_clean_circle_still_detected(fixture_directory):
    image = cv2.imread(str(fixture_directory / "circle_worksheet.png"))
    filtered = preprocess_image(image, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)
    height, width = image.shape[:2]
    semantic = analyze_diagram(shapes, width, height, [])

    assert any(e.type is GeometryType.CIRCLE for e in semantic.elements)


# --- ellipse type in inspector ---

def test_ellipse_type_in_enum():
    assert GeometryType("ellipse") is GeometryType.ELLIPSE


def test_ellipse_tactile_svg_renders():
    from app.services.tactile_svg import render_tactile_svg
    from app.models.geometry import DetectedElement, ConfidenceLevel

    ellipse = DetectedElement(
        id="el_0",
        type=GeometryType.ELLIPSE,
        geometry={"center": [200, 150], "semi_axes": [100, 50], "angle": 0, "area": 15707.96},
        confidence=0.88,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
        source="contour",
    )
    svg = render_tactile_svg([ellipse], 400, 300)
    assert "<svg" in svg
    assert "<ellipse" in svg
    assert 'data-element-id="el_0"' in svg
    assert 'cx="200.0"' in svg
    assert 'rx="100.0"' in svg
