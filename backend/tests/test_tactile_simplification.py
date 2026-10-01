from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services.tactile_simplification import simplify_geometry


def _element(eid: str, confidence: float, gtype=GeometryType.LINE_SEGMENT, start=(0, 0), end=(100, 100)) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=gtype,
        geometry={"start": start, "end": end},
        confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH if confidence >= 0.8 else (ConfidenceLevel.MEDIUM if confidence >= 0.5 else ConfidenceLevel.LOW),
        needs_review=confidence < 0.5,
        source="hough",
        bbox=(0, 0, 100, 100),
    )


def _label_element(eid: str, confidence: float = 0.2) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.TEXT_LABEL,
        geometry={"text": "A", "position": [50, 50]},
        confidence=confidence,
        confidence_level=ConfidenceLevel.LOW,
        needs_review=True,
        source="ocr",
        bbox=(40, 40, 20, 20),
    )


def test_simplification_removes_low_confidence():
    semantic = SemanticGeometry(elements=[_element("el_0", 0.2), _element("el_1", 0.9)])

    result = simplify_geometry(semantic)

    assert result.removed_count == 1
    assert [e.id for e in result.elements] == ["el_1"]


def test_simplification_preserves_high_confidence():
    # Two non-collinear segments: this test is about confidence, so the segments
    # must not be candidates for the collinear merge.
    semantic = SemanticGeometry(elements=[
        _element("el_0", 0.9, end=(100, 100)),
        _element("el_1", 0.85, start=(0, 100), end=(100, 0)),
    ])

    result = simplify_geometry(semantic)

    assert result.removed_count == 0
    assert len(result.elements) == 2


def test_simplification_merges_exact_duplicate_segments_and_reports_it():
    semantic = SemanticGeometry(elements=[
        _element("el_0", 0.9, start=(10, 10), end=(120, 120)),
        _element("el_1", 0.85, start=(10, 10), end=(120, 120)),
    ])

    result = simplify_geometry(semantic)

    assert [e.id for e in result.elements] == ["el_0"]
    assert result.merged_count == 1
    # The absorbed segment is recorded on the surviving element and in the action log.
    assert "el_1" in result.elements[0].semantic_properties["merged_from"]
    assert any(a.element_id == "el_0" and a.action == "merged_collinear" for a in result.actions)
    assert any("el_1" in message for message in result.explanations)


def test_simplification_never_drops_an_element_without_an_action():
    semantic = SemanticGeometry(elements=[
        _element("el_0", 0.9, start=(10, 10), end=(120, 120)),
        _element("el_1", 0.85, start=(10, 11), end=(121, 119)),
        _element("el_2", 0.7, start=(0, 0), end=(100, 0)),
        _element("el_3", 0.6, start=(0, 0), end=(0, 100)),
    ])

    result = simplify_geometry(semantic)

    kept_ids = {e.id for e in result.elements}
    merged_from = {
        element_id
        for e in result.elements
        for element_id in (e.semantic_properties or {}).get("merged_from", [])
    }
    reported = {a.element_id for a in result.actions}
    for element in semantic.elements:
        assert (
            element.id in kept_ids
            or element.id in merged_from
            or element.id in reported
        ), f"{element.id} vanished without an explanation"
    assert not any(a.action == "dropped_unexplained" for a in result.actions)


def test_simplification_preserves_labels_even_when_low_confidence():
    semantic = SemanticGeometry(elements=[_label_element("el_0", confidence=0.2)])

    result = simplify_geometry(semantic)

    assert len(result.elements) == 1
    assert result.elements[0].type is GeometryType.TEXT_LABEL


def test_simplification_records_actions_and_keeps_relationships():
    rel = ElementRelationship(
        id="rel_0",
        type=RelationshipType.PARALLEL_LINES,
        element_ids=["el_0", "el_1"],
        confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
    )
    semantic = SemanticGeometry(
        elements=[_element("el_0", 0.9), _element("el_1", 0.9), _element("el_2", 0.2)],
        relationships=[rel],
    )

    result = simplify_geometry(semantic)

    assert result.removed_count == 1
    assert any(action.action == "removed_noise" for action in result.actions)
