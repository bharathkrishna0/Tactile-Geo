"""Deterministic unit tests for the Phase 2 semantic pipeline additions.

These tests exercise the new/inferred behaviour with hand-built diagrams so they
do not depend on OCR or the Liblouis native library.
"""
import math

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services import geometry_relations, tactile_svg
from app.services.diagram_analysis import analyze_diagram
from app.services.label_association import associate_label
from app.services.tactile_qa import run_tactile_qa
from app.services.tactile_rules import TACTILE_RULES


def _element(eid, gtype, geometry, confidence=0.9, source="contour", needs_review=False):
    level = ConfidenceLevel.HIGH if confidence >= 0.8 else (ConfidenceLevel.MEDIUM if confidence >= 0.5 else ConfidenceLevel.LOW)
    return DetectedElement(
        id=eid, type=gtype, geometry=geometry, confidence=confidence,
        confidence_level=level, needs_review=needs_review, source=source,
    )


def _line(eid, start, end, confidence=0.9):
    return _element(eid, GeometryType.LINE_SEGMENT, {"start": start, "end": end}, confidence=confidence, source="hough")


def _point(eid, pos):
    return _element(eid, GeometryType.POINT, {"position": pos}, source="heuristic")


# --- geometry_relations -----------------------------------------------------

def test_infers_parallel_perpendicular_and_connected():
    a = _line("l1", (0, 0), (100, 0))
    b = _line("l2", (0, 50), (100, 50))
    c = _line("l3", (0, 0), (0, 100))
    rels = {r.type for r in geometry_relations.infer_relationships([a, b, c])}
    assert RelationshipType.PARALLEL_LINES in rels
    assert RelationshipType.PERPENDICULAR_LINES in rels
    assert RelationshipType.CONNECTED_LINES in rels


def test_infers_point_on_line():
    a = _line("l1", (0, 0), (100, 0))
    p = _point("p1", (50, 2))
    rels = geometry_relations.infer_relationships([a, p])
    assert any(r.type is RelationshipType.POINT_ON_LINE for r in rels)


def test_infers_point_on_circle_and_center():
    circle = _element("c1", GeometryType.CIRCLE, {"center": (50, 50), "radius": 30})
    on_circle = _point("p1", (80, 50))
    center = _point("p2", (50, 50))
    rels = geometry_relations.infer_relationships([circle, on_circle, center])
    types = {r.type for r in rels}
    assert RelationshipType.POINT_ON_CIRCLE in types
    assert RelationshipType.CENTER_OF in types


def test_relationships_carried_with_explanation():
    a = _line("l1", (0, 0), (100, 0))
    b = _line("l2", (0, 50), (100, 50))
    rels = geometry_relations.infer_relationships([a, b])
    parallel = [r for r in rels if r.type is RelationshipType.PARALLEL_LINES][0]
    assert parallel.explanation
    assert parallel.element_ids == ["l1", "l2"]


# --- label association ------------------------------------------------------

def test_label_association_picks_nearest_element():
    triangle = _element("t1", GeometryType.TRIANGLE, {"points": [(120, 35), (35, 195), (205, 195)], "area": 1000})
    label = _element("lb", GeometryType.TEXT_LABEL, {"text": "A", "position": [125, 40]})
    result = associate_label(label, [triangle])
    assert result["target_id"] == "t1"
    assert result["confidence"] > 0.7
    assert result["needs_review"] is False


def test_label_association_uncertain_when_far():
    triangle = _element("t1", GeometryType.TRIANGLE, {"points": [(120, 35), (35, 195), (205, 195)], "area": 1000})
    label = _element("lb", GeometryType.TEXT_LABEL, {"text": "X", "position": [50, 260]})
    result = associate_label(label, [triangle])
    assert result["needs_review"] is True


# --- diagram_analysis wiring ------------------------------------------------

def test_diagram_analysis_publishes_explanations():
    # A bounded polygon (or connected segments) yields point/angle explanations.
    shapes = [
        {"type": "line", "points": [(0, 0), (100, 0)]},
        {"type": "line", "points": [(100, 0), (100, 100)]},
    ]
    semantic = analyze_diagram(shapes, 200, 200, [])
    assert semantic.explanations


def test_diagram_analysis_associates_label_to_element():
    shapes = [{"type": "line", "points": [(10, 10), (110, 10)]}]
    labels = [{"text": "A", "confidence": 0.98, "position": [120, 15]}]
    semantic = analyze_diagram(shapes, 200, 200, labels)
    label = [e for e in semantic.elements if e.type is GeometryType.TEXT_LABEL][0]
    assert label.associated_label_id is not None
    # The line element should also point back to the label.
    line = [e for e in semantic.elements if e.type is GeometryType.LINE_SEGMENT][0]
    assert line.associated_label_id == label.id


def test_diagram_analysis_detects_angle_between_segments():
    shapes = [
        {"type": "line", "points": [(0, 0), (100, 0)]},
        {"type": "line", "points": [(100, 0), (100, 100)]},
    ]
    semantic = analyze_diagram(shapes, 200, 200, [])
    assert any(e.type is GeometryType.ANGLE for e in semantic.elements)
    assert any(r.type is RelationshipType.ANGLE_BETWEEN for r in semantic.relationships)


# --- tactile QA additions ---------------------------------------------------

def test_qa_reports_score_out_of_100():
    clean = _element("t1", GeometryType.TRIANGLE, {"points": [(50, 50), (150, 50), (100, 150)], "area": 1000})
    clean.bbox = (50, 50, 100, 100)
    report = run_tactile_qa([clean], 300, 300)
    assert 0 <= report.score_0_100 <= 100
    assert isinstance(report.score_0_100, int)


def test_qa_flags_label_collision_with_geometry():
    line = _line("l1", (100, 100), (200, 100))
    label = _element("lb", GeometryType.TEXT_LABEL, {"text": "A", "position": [105, 100]})
    report = run_tactile_qa([line, label], 300, 300)
    assert any(issue.check == "braille_collision" for issue in report.issues)


# --- tactile SVG ------------------------------------------------------------

def test_tactile_svg_renders_geometry_and_aria():
    triangle = _element("t1", GeometryType.TRIANGLE, {"points": [(50, 50), (150, 50), (100, 150)], "area": 3000})
    svg = tactile_svg.render_tactile_svg([triangle], 200, 200)
    assert svg.startswith("<svg")
    assert "polygon" in svg
    assert 'aria-label=' in svg


def test_tactile_svg_uses_configured_stroke_width():
    triangle = _element("t1", GeometryType.TRIANGLE, {"points": [(50, 50), (150, 50), (100, 150)], "area": 3000})
    svg = tactile_svg.render_tactile_svg([triangle], 200, 200)
    assert f"stroke-width:{TACTILE_RULES.stroke_width_max_pt:.1f}" in svg