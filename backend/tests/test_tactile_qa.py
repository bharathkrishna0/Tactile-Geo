from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType, SemanticGeometry
from app.services.tactile_qa import MIN_FEATURE_SPACING, run_tactile_qa


def _line_element(eid: str, start, end) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.LINE_SEGMENT,
        geometry={"start": start, "end": end},
        confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
        source="hough",
        bbox=(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]), abs(end[1] - start[1])),
    )


def _triangle_element(eid: str, points) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.TRIANGLE,
        geometry={"points": points, "area": 1000},
        confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
        source="contour",
        bbox=(0, 0, 200, 200),
    )


def test_qa_passes_for_clean_geometry():
    elements = [_triangle_element("el_0", [(50, 50), (150, 50), (100, 150)])]

    report = run_tactile_qa(elements, 300, 300)

    assert report.passes is True
    assert not any(issue.severity == "error" for issue in report.issues)


def test_qa_flags_small_elements():
    elements = [DetectedElement(
        id="el_0",
        type=GeometryType.LINE_SEGMENT,
        geometry={"start": (100, 100), "end": (105, 100)},
        confidence=0.5,
        confidence_level=ConfidenceLevel.MEDIUM,
        needs_review=False,
        source="hough",
        bbox=(100, 100, 5, 0),
    )]

    report = run_tactile_qa(elements, 200, 200)

    assert any(issue.check == "feature_below_minimum_size" for issue in report.issues)


def test_qa_flags_crowded_geometry():
    a = _line_element("el_0", (100, 100), (200, 100))
    b = _line_element("el_1", (100, 108), (200, 108))

    report = run_tactile_qa([a, b], 300, 300)

    assert any(issue.check == "spacing" for issue in report.issues)


def test_qa_flags_excessive_complexity():
    elements = [_line_element(f"el_{i}", (i * 5, 0), (i * 5, 20)) for i in range(60)]

    report = run_tactile_qa(elements, 400, 200)

    assert any(issue.check == "complexity" for issue in report.issues)


def test_qa_report_has_structured_issues():
    a = _line_element("el_0", (100, 100), (200, 100))
    b = _line_element("el_1", (100, 108), (200, 108))

    report = run_tactile_qa([a, b], 300, 300)

    assert report.element_checks > 0
    for issue in report.issues:
        assert issue.check
        assert issue.severity in {"info", "warning", "error"}
        assert issue.message
