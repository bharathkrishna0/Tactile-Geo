from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services.tactile_simplification import simplify_geometry


def _element(eid: str, confidence: float, gtype=GeometryType.LINE_SEGMENT) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=gtype,
        geometry={"start": (0, 0), "end": (100, 100)},
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
    semantic = SemanticGeometry(elements=[_element("el_0", 0.9), _element("el_1", 0.85)])

    result = simplify_geometry(semantic)

    assert result.removed_count == 0
    assert len(result.elements) == 2


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
