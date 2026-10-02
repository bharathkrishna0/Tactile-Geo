"""Tests for Milestone 4: Extended tactile QA checks."""
from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.tactile_qa import run_tactile_qa


def _line_element(eid: str, start, end, confidence=0.9) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.LINE_SEGMENT,
        geometry={"start": start, "end": end},
        confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH if confidence >= 0.8 else (ConfidenceLevel.MEDIUM if confidence >= 0.5 else ConfidenceLevel.LOW),
        needs_review=confidence < 0.5,
        source="hough",
        bbox=(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]), abs(end[1] - start[1])),
    )


def _label_element(eid: str, position, text="A", associated=None) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.TEXT_LABEL,
        geometry={"text": text, "position": position},
        confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
        source="ocr",
        bbox=(position[0] - 5, position[1] - 5, 10, 10),
        associated_label_id=associated,
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


def test_qa_flags_duplicate_geometry():
    a = _line_element("el_0", (50, 50), (200, 50))
    b = _line_element("el_1", (51, 50), (200, 50))
    report = run_tactile_qa([a, b], 300, 300)
    assert any(issue.check == "duplicate_geometry" for issue in report.issues)


def test_qa_does_not_flag_distinct_geometry():
    a = _line_element("el_0", (50, 50), (200, 50))
    b = _line_element("el_1", (50, 150), (200, 150))
    report = run_tactile_qa([a, b], 300, 300)
    assert not any(issue.check == "duplicate_geometry" for issue in report.issues)


def test_qa_flags_dangling_label():
    geo = _line_element("el_0", (50, 50), (200, 50))
    label = _label_element("el_1", (400, 400))
    report = run_tactile_qa([geo, label], 500, 500)
    assert any(issue.check == "dangling_label" for issue in report.issues)


def test_qa_does_not_flag_associated_label():
    geo = _line_element("el_0", (50, 50), (200, 50))
    label = _label_element("el_1", (120, 30), associated="el_0")
    report = run_tactile_qa([geo, label], 300, 300)
    assert not any(issue.check == "dangling_label" for issue in report.issues)


def test_qa_does_not_flag_label_near_geometry():
    geo = _line_element("el_0", (50, 50), (200, 50))
    label = _label_element("el_1", (120, 40))
    report = run_tactile_qa([geo, label], 300, 300)
    assert not any(issue.check == "dangling_label" for issue in report.issues)


def test_qa_sparse_diagram_single_shape():
    a = _line_element("el_0", (50, 50), (200, 50))
    report = run_tactile_qa([a], 300, 300)
    assert any(issue.check == "sparse_diagram" for issue in report.issues)


def test_qa_no_sparse_warning_with_multiple_shapes():
    a = _line_element("el_0", (50, 50), (200, 50))
    b = _line_element("el_1", (50, 150), (200, 150))
    c = _line_element("el_2", (50, 250), (200, 250))
    report = run_tactile_qa([a, b, c], 300, 300)
    assert not any(issue.check == "sparse_diagram" for issue in report.issues)


def test_qa_new_checks_do_not_break_existing():
    triangle = _triangle_element("el_0", [(50, 50), (150, 50), (100, 150)])
    line = _line_element("el_1", (200, 100), (250, 100))
    # Braille cells clear of the line by more than the physical clearance.
    label = _label_element("el_2", (225, 80), associated="el_1")
    report = run_tactile_qa([triangle, line, label], 400, 400)
    assert 0 <= report.score_0_100 <= 100
    assert report.passes is True
