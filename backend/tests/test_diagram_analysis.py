import cv2
import numpy as np

from app.models.geometry import GeometryType
from app.services.diagram_analysis import analyze_diagram
from app.services.image_preprocessing import preprocess_image
from app.services.vectorization import extract_shapes


def _analyze(fixture_directory, name, edge_sensitivity=100):
    image = cv2.imread(str(fixture_directory / name))
    filtered = preprocess_image(image, edge_sensitivity)
    shapes = extract_shapes(filtered, edge_sensitivity)
    height, width = image.shape[:2]
    return analyze_diagram(shapes, width, height, [])


def test_analysis_classifies_triangle_contour(fixture_directory):
    semantic = _analyze(fixture_directory, "triangle_worksheet.png")

    assert any(element.type is GeometryType.TRIANGLE for element in semantic.elements)
    assert semantic.element_count > 0


def test_analysis_classifies_circle_contour(fixture_directory):
    semantic = _analyze(fixture_directory, "circle_worksheet.png")

    assert any(element.type is GeometryType.CIRCLE for element in semantic.elements)


def test_analysis_detects_parallel_lines():
    shapes = [
        {"type": "line", "points": [(10, 10), (110, 10)]},
        {"type": "line", "points": [(10, 60), (110, 60)]},
    ]
    semantic = analyze_diagram(shapes, 200, 200, [])

    from app.models.geometry import RelationshipType
    assert any(rel.type is RelationshipType.PARALLEL_LINES for rel in semantic.relationships)


def test_analysis_assigns_high_confidence_to_clean_shapes(fixture_directory):
    semantic = _analyze(fixture_directory, "triangle_worksheet.png")

    assert all(element.confidence >= 0.5 for element in semantic.elements if element.type is not GeometryType.LINE_SEGMENT)


def test_analysis_flags_small_noisy_contours():
    shapes = [
        {"type": "contour", "points": [(10, 10), (12, 10), (12, 12), (10, 12)]},
    ]
    semantic = analyze_diagram(shapes, 200, 200, [])

    assert semantic.low_confidence_count == 1
    assert semantic.review_flags
    assert semantic.elements[0].needs_review


def test_analysis_includes_text_labels(fixture_directory):
    image = cv2.imread(str(fixture_directory / "labelled_triangle_worksheet.png"))
    filtered = preprocess_image(image, 100)
    shapes = extract_shapes(filtered, 100)
    height, width = image.shape[:2]
    labels = [
        {"text": "A", "confidence": 0.98, "position": [120, 35]},
        {"text": "B", "confidence": 0.5, "position": [35, 195]},
    ]
    semantic = analyze_diagram(shapes, width, height, labels)

    text_labels = [e for e in semantic.elements if e.type is GeometryType.TEXT_LABEL]
    assert len(text_labels) == 2
    assert text_labels[0].source == "ocr"


def test_analysis_does_not_guess_uncertain_geometry():
    shapes = [
        {"type": "line", "points": [(0, 0), (180, 180)]},
        {"type": "line", "points": [(180, 180), (280, 40)]},
    ]
    semantic = analyze_diagram(shapes, 300, 200, [])

    assert not any(element.type is GeometryType.TRIANGLE for element in semantic.elements if element.source != "contour")
