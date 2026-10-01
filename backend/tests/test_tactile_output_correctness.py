"""Regression tests for tactile output that the teacher would actually export.

These run the real vectoriser and QA on rendered images, because the defects
they guard against (invented ellipses, double-struck edges, rotated ellipses,
clipped braille, empty sheets marked ready) were invisible to synthetic tests.
"""
import math
import re
from itertools import pairwise

import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.braille import LouisBrailleTranslator
from app.services.braille_layout import place_braille_markers
from app.services.diagram_analysis import analyze_diagram
from app.services.image_preprocessing import preprocess_image
from app.services.label_association import associate_label
from app.services.label_mapping import map_label_to_geometry
from app.services.ocr import EasyOcrProvider, OcrDetection
from app.services.pipeline import build_full_analysis
from app.services.tactile_qa import run_tactile_qa
from app.services.tactile_rules import TACTILE_RULES
from app.services.tactile_svg import (
    MM_PER_PT,
    _braille_dots,
    braille_text_extent,
    page_layout,
    render_tactile_svg,
)
from app.services.vectorization import extract_shapes


def _shapes(image: np.ndarray) -> list[dict]:
    return extract_shapes(preprocess_image(image, edge_sensitivity=50), edge_sensitivity=50)


def _blank(width: int = 300, height: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_triangle_does_not_gain_an_ellipse(fixture_directory):
    shapes = _shapes(cv2.imread(str(fixture_directory / "triangle_worksheet.png")))

    assert not any(shape["type"] == "ellipse" for shape in shapes)


def test_triangle_records_interior_angles(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))
    semantic = analyze_diagram(_shapes(image), image.shape[1], image.shape[0], [])

    triangle = next(e for e in semantic.elements if e.type is GeometryType.TRIANGLE)
    angles = triangle.semantic_properties["interior_angles_deg"]
    assert len(angles) == 3
    assert abs(sum(angles) - 180) < 2


def test_hexagon_stays_a_polygon():
    image = _blank()
    centre, radius = (150, 150), 100
    points = np.array([
        (round(centre[0] + radius * math.cos(math.radians(60 * k))), round(centre[1] + radius * math.sin(math.radians(60 * k))))
        for k in range(6)
    ], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=4)

    shapes = _shapes(image)

    assert [shape["type"] for shape in shapes] == ["contour"]
    assert len(shapes[0]["points"]) == 6


def test_horizontal_ellipse_keeps_its_orientation(fixture_directory):
    shapes = _shapes(cv2.imread(str(fixture_directory / "ellipse_worksheet.png")))

    ellipses = [shape for shape in shapes if shape["type"] == "ellipse"]
    assert len(ellipses) == 1
    semi_major, semi_minor = ellipses[0]["semi_axes"]
    assert abs(semi_major - 120) <= 3 and abs(semi_minor - 60) <= 3
    # Major axis horizontal: angle ~0 or ~180 degrees.
    assert min(ellipses[0]["angle"] % 180, 180 - ellipses[0]["angle"] % 180) < 3


def test_rotated_ellipse_reports_major_axis_angle():
    image = _blank(400, 400)
    cv2.ellipse(image, (200, 200), (140, 60), 30, 0, 360, (0, 0, 0), thickness=3)

    ellipse = next(shape for shape in _shapes(image) if shape["type"] == "ellipse")

    assert abs(ellipse["angle"] - 30) < 3


def test_circle_diameter_is_kept_and_rim_is_not_duplicated(fixture_directory):
    shapes = _shapes(cv2.imread(str(fixture_directory / "circle_worksheet.png")))

    lines = [shape for shape in shapes if shape["type"] == "line"]
    assert len(lines) == 1
    (x1, y1), (x2, y2) = lines[0]["points"]
    assert abs(x2 - x1) > 120 and abs(y2 - y1) < 10
    assert sum(shape["type"] == "contour" for shape in shapes) == 1


def test_open_angle_strokes_stay_lines():
    image = _blank()
    cv2.line(image, (50, 250), (250, 250), (0, 0, 0), 4)
    cv2.line(image, (50, 250), (200, 80), (0, 0, 0), 4)

    shapes = _shapes(image)

    lines = [shape["points"] for shape in shapes if shape["type"] == "line"]
    assert len(lines) == 2
    assert not any(shape["type"] in {"contour", "ellipse"} for shape in shapes)
    # Both arms meet at exactly one shared vertex near (50, 250).
    shared = set(lines[0]) & set(lines[1])
    assert len(shared) == 1
    vertex = shared.pop()
    assert math.dist(vertex, (50, 250)) <= 6


def test_label_inside_triangle_bbox_does_not_delete_triangle(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))
    cv2.putText(image, "A", (112, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    class Reader:
        def readtext(self, image, detail=1, paragraph=False):
            return [([(110, 72), (128, 72), (128, 92), (110, 92)], "A", 0.99)]

    class Louis:
        def translateString(self, tables, text):
            return ",a"

    _, encoded = cv2.imencode(".png", image)
    result = build_full_analysis(
        encoded.tobytes(),
        ocr_provider=EasyOcrProvider(reader=Reader()),
        braille_translator=LouisBrailleTranslator(bindings=Louis()),
    )

    assert any(e.type is GeometryType.TRIANGLE for e in result.simplified_geometry.elements)


def test_label_maps_to_ellipse_only_diagram():
    shapes = [{
        "type": "ellipse", "center": (200, 150), "semi_axes": (120, 60), "angle": 0.0,
        "area": 22619.5, "contour_points": [(80, 150), (200, 90), (320, 150), (200, 210)],
    }]
    detection = OcrDetection(text="E", bbox=[(190, 15), (205, 15), (205, 35), (190, 35)], confidence=0.9)

    mapped = map_label_to_geometry(detection, shapes)

    assert mapped["anchor_type"] in {"vertex", "edge"}


def test_braille_markers_stay_on_the_page():
    labels = [{"text": "A", "braille": "\u2820\u2801", "desired_position": (120, 2)}]
    shapes = [{"type": "contour", "points": [(119, 37), (36, 194), (203, 194)]}]

    placed = place_braille_markers(labels, shapes, bounds=(240, 240))

    x, y = placed[0]["position"]
    width, height = braille_text_extent("\u2820\u2801", 240, 240)
    assert x - width / 2 >= 0 and y - height / 2 >= 0
    assert x + width / 2 <= 240 and y + height / 2 <= 240


def _label(position, braille="\u2820\u2801") -> DetectedElement:
    return DetectedElement(
        id="lbl", type=GeometryType.TEXT_LABEL,
        geometry={"text": "A", "braille": braille, "position": list(position)},
        confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="ocr",
    )


def _segment() -> DetectedElement:
    return DetectedElement(
        id="seg", type=GeometryType.LINE_SEGMENT,
        geometry={"start": [40, 150], "end": [200, 150], "length": 160},
        confidence=0.95, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="hough",
        bbox=(40, 150, 160, 0),
    )


def test_qa_blocks_braille_clipped_at_page_edge():
    report = run_tactile_qa([_segment(), _label((120, 0))], 240, 240)

    assert not report.passes
    assert any(i.check == "element_outside_printable_area" and i.element_id == "lbl" for i in report.issues)


def test_qa_blocks_sheet_with_no_geometry(fixture_directory):
    report = run_tactile_qa([], 240, 240)

    assert not report.passes
    assert [i.check for i in report.issues if i.severity == "error"] == ["no_tactile_geometry"]


def test_dark_fixture_is_not_export_ready(fixture_directory):
    class Reader:
        def readtext(self, image, detail=1, paragraph=False):
            return []

    result = build_full_analysis(
        (fixture_directory / "dark_worksheet.png").read_bytes(),
        ocr_provider=EasyOcrProvider(reader=Reader()),
    )

    if not any(e.type is not GeometryType.TEXT_LABEL for e in result.simplified_geometry.elements):
        assert not result.qa_report.passes


def test_tactile_svg_is_a4_with_physical_stroke_and_braille_dots():
    label = _label((120, 60))
    svg = render_tactile_svg([_segment(), label], 240, 240)

    layout = page_layout(240, 240)
    assert 'width="210mm" height="297mm"' in svg
    stroke = float(re.search(r"stroke-width:([0-9.]+)", svg).group(1))
    assert abs(stroke * layout.mm_per_px - TACTILE_RULES.stroke_width_pt * MM_PER_PT) < 0.01
    # "A" is ⠠⠁: dot 6 then dot 1, i.e. two embossed dots and no font glyphs.
    braille_group = re.search(r'<g data-element-id="lbl" class="braille".*?</g>', svg).group(0)
    assert braille_group.count("<circle") == 2
    assert "<text" not in svg


def test_braille_dots_use_standard_cell_spacing():
    layout = page_layout(240, 240)
    dots = _braille_dots("\u283f\u283f", (120, 120), layout.mm_per_px)

    xs = sorted({round(x * layout.mm_per_px, 2) for x, _ in dots})
    ys = sorted({round(y * layout.mm_per_px, 2) for _, y in dots})
    assert [round(b - a, 2) for a, b in pairwise(xs)] == [2.5, 3.5, 2.5]
    assert [round(b - a, 2) for a, b in pairwise(ys)] == [2.5, 2.5]


def test_label_beside_circle_rim_is_associated_not_dangling():
    circle = DetectedElement(
        id="c", type=GeometryType.CIRCLE, geometry={"center": [150, 150], "radius": 100},
        confidence=0.95, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="contour",
        bbox=(50, 50, 200, 200),
    )
    label = _label((150, 30))

    assert associate_label(label, [circle])["target_id"] == "c"
    report = run_tactile_qa([circle, _label((150, 25))], 300, 300)
    assert not any(issue.check == "dangling_label" for issue in report.issues)
