"""Tests for Milestone 6: Explainability / provenance."""
from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.diagram_analysis import analyze_diagram


def _analyze(shapes, width=300, height=300, labels=None):
    return analyze_diagram(shapes, width, height, labels or [])


def test_line_element_has_provenance():
    semantic = _analyze([{"type": "line", "points": [(10, 10), (110, 10)]}])
    line = [e for e in semantic.elements if e.type is GeometryType.LINE_SEGMENT][0]
    assert line.provenance is not None
    assert "line segment" in line.provenance.lower()
    assert "confidence" in line.provenance.lower()


def test_short_line_provenance_notes_review():
    semantic = _analyze([{"type": "line", "points": [(10, 10), (20, 10)]}])
    line = [e for e in semantic.elements if e.type is GeometryType.LINE_SEGMENT][0]
    assert line.provenance is not None
    assert "noise" in line.provenance.lower() or "fragment" in line.provenance.lower()


def test_contour_element_has_provenance():
    semantic = _analyze([{"type": "contour", "points": [(50, 50), (150, 50), (100, 150)]}])
    tri = [e for e in semantic.elements if e.type is GeometryType.TRIANGLE][0]
    assert tri.provenance is not None
    assert "triangle" in tri.provenance.lower()


def test_ellipse_element_has_provenance():
    semantic = _analyze([
        {"type": "ellipse", "center": (100, 100), "semi_axes": (60, 30), "angle": 0, "area": 5654.87,
         "contour_points": [(40, 100), (100, 70), (160, 100), (100, 130)]},
    ], width=300, height=300)
    ellipse = [e for e in semantic.elements if e.type is GeometryType.ELLIPSE][0]
    assert ellipse.provenance is not None
    assert "ellipse" in ellipse.provenance.lower()


def test_label_element_has_provenance():
    semantic = _analyze(
        [{"type": "line", "points": [(10, 10), (110, 10)]}],
        labels=[{"text": "A", "confidence": 0.98, "position": [50, 5]}],
    )
    label = [e for e in semantic.elements if e.type is GeometryType.TEXT_LABEL][0]
    assert label.provenance is not None
    assert "OCR" in label.provenance
    assert "A" in label.provenance


def test_point_element_has_provenance():
    semantic = _analyze([
        {"type": "line", "points": [(10, 10), (110, 10)]},
        {"type": "line", "points": [(110, 10), (110, 110)]},
    ])
    pts = [e for e in semantic.elements if e.type is GeometryType.POINT]
    assert any(p.provenance is not None for p in pts)


def test_angle_element_has_provenance():
    semantic = _analyze([
        {"type": "line", "points": [(0, 0), (100, 0)]},
        {"type": "line", "points": [(100, 0), (100, 100)]},
    ])
    angles = [e for e in semantic.elements if e.type is GeometryType.ANGLE]
    assert any(a.provenance is not None for a in angles)


# --- provenance survives serialization / editing ---

def test_provenance_roundtrips_through_dict():
    from app.api.sessions import _element_to_dict, _semantic_to_dict
    from app.services.editing import semantic_from_dict

    semantic = _analyze([{"type": "line", "points": [(10, 10), (110, 10)]}])
    semantic.image_width = 300
    semantic.image_height = 300
    as_dict = _semantic_to_dict(semantic)
    rebuilt = semantic_from_dict(as_dict)

    original_line = [e for e in semantic.elements if e.type is GeometryType.LINE_SEGMENT][0]
    rebuilt_line = [e for e in rebuilt.elements if e.type is GeometryType.LINE_SEGMENT][0]
    assert rebuilt_line.provenance == original_line.provenance


def test_merge_preserves_provenance():
    from app.services.tactile_simplification import simplify_geometry
    from app.models.geometry import SemanticGeometry

    a = DetectedElement(id="el_0", type=GeometryType.LINE_SEGMENT, geometry={"start": [10, 10], "end": [60, 10]},
                        confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough")
    b = DetectedElement(id="el_1", type=GeometryType.LINE_SEGMENT, geometry={"start": [60, 10], "end": [110, 10]},
                        confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough")
    semantic = SemanticGeometry(elements=[a, b], image_width=300, image_height=300)
    simplified = simplify_geometry(semantic)
    merged = [e for e in simplified.elements if e.id == "el_0"][0]
    assert merged.provenance is not None
    assert "Merged" in merged.provenance
