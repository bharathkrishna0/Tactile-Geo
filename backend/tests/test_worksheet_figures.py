"""Textbook worksheet figures: hairline colour strokes, radicals, right-angle markers, page isolation."""

import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.braille import LouisBrailleTranslator
from app.services.diagram_analysis import analyze_diagram
from app.services.diagram_region import find_diagram_regions, inside_regions, mask_to_regions
from app.services.image_enhancement import ink_contrast_copy
from app.services.image_preprocessing import preprocess_image
from app.services.ocr import EasyOcrProvider, OcrDetection
from app.services.ocr_postprocessing import restore_radicals
from app.services.pipeline import build_full_analysis
from app.services.right_angles import detect_right_angle_markers
from app.services.tactile_svg import render_tactile_svg
from app.services.vectorization import extract_shapes, stroke_half_width

LIGHT_BLUE = (235, 190, 120)


def _blank(width=400, height=400, value=255):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _right_triangle(marker: bool) -> np.ndarray:
    image = _blank()
    cv2.line(image, (80, 320), (320, 320), (0, 0, 0), 3)
    cv2.line(image, (80, 320), (80, 80), (0, 0, 0), 3)
    cv2.line(image, (80, 80), (320, 320), (0, 0, 0), 3)
    if marker:
        cv2.line(image, (80, 295), (105, 295), (0, 0, 0), 2)
        cv2.line(image, (105, 295), (105, 320), (0, 0, 0), 2)
    return image


def _markers(image: np.ndarray) -> list[dict]:
    binary = preprocess_image(image)
    shapes = extract_shapes(binary)
    return detect_right_angle_markers(binary, shapes, stroke_half_width(binary))


def test_hairline_coloured_strokes_survive_preprocessing():
    image = _blank()
    cv2.line(image, (50, 200), (350, 200), LIGHT_BLUE, 1)
    cv2.line(image, (200, 50), (200, 350), LIGHT_BLUE, 1)

    binary = preprocess_image(image)

    assert np.count_nonzero(binary[195:206, 50:350].any(axis=0)) > 280
    assert np.count_nonzero(binary[50:350, 195:206].any(axis=1)) > 280


def test_hairline_recovery_ignores_isolated_specks():
    image = _blank()
    rng = np.random.default_rng(0)
    for x, y in rng.integers(20, 380, size=(60, 2)):
        image[y, x] = LIGHT_BLUE

    assert np.count_nonzero(preprocess_image(image)) == 0


def test_hairline_recovery_is_off_for_dark_photos():
    image = _blank(value=40)
    cv2.line(image, (50, 200), (350, 200), (60, 60, 60), 1)

    assert np.count_nonzero(preprocess_image(image)) == 0


def test_ink_contrast_copy_darkens_coloured_ink_and_keeps_black():
    image = _blank(4, 1)
    image[0, 0] = LIGHT_BLUE
    image[0, 1] = (0, 0, 0)

    copy = ink_contrast_copy(image)

    assert copy[0, 0].tolist() == [120, 120, 120]
    assert copy[0, 1].tolist() == [0, 0, 0]
    assert copy[0, 2].tolist() == [255, 255, 255]
    assert ink_contrast_copy(np.zeros((2, 2), dtype=np.uint8)).shape == (2, 2, 3)


def _label_image(with_vinculum: bool) -> tuple[np.ndarray, list[tuple[int, int]]]:
    gray = np.full((60, 120), 255, dtype=np.uint8)
    cv2.putText(gray, "V4", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    if with_vinculum:
        cv2.line(gray, (30, 12), (100, 12), 0, 2)
    return gray, [(18, 10), (102, 10), (102, 55), (18, 55)]


def test_radical_is_restored_when_a_vinculum_is_drawn():
    gray, bbox = _label_image(with_vinculum=True)
    detections = [OcrDetection("V4", bbox, 0.9), OcrDetection("Vi7", bbox, 0.9)]

    assert [d.text for d in restore_radicals(detections, gray)] == ["√4", "√17"]


def test_plain_v_labels_are_left_alone():
    gray, bbox = _label_image(with_vinculum=False)
    detections = [OcrDetection("V4", bbox, 0.9), OcrDetection("V", bbox, 0.9), OcrDetection("VA", bbox, 0.9)]

    assert [d.text for d in restore_radicals(detections, gray)] == ["V4", "V", "VA"]


def test_radical_translates_to_ueb_braille():
    translator = LouisBrailleTranslator()

    assert translator.translate("√4") == "⠐⠩⠼⠙"
    assert translator.translate("√17") == "⠐⠩⠼⠁⠛"


def test_drawn_right_angle_marker_is_detected():
    markers = _markers(_right_triangle(marker=True))

    assert len(markers) == 1
    vx, vy = markers[0]["vertex"]
    assert abs(vx - 80) <= 6 and abs(vy - 320) <= 6
    assert abs(markers[0]["degrees"] - 90) <= 10


def test_unmarked_right_angle_and_plain_rectangle_have_no_marker():
    rectangle = _blank()
    cv2.rectangle(rectangle, (80, 100), (320, 300), (0, 0, 0), 3)

    assert _markers(_right_triangle(marker=False)) == []
    assert _markers(rectangle) == []


def test_right_angle_marker_becomes_a_tactile_symbol():
    image = _right_triangle(marker=True)
    binary = preprocess_image(image)
    shapes = extract_shapes(binary)
    markers = detect_right_angle_markers(binary, shapes, stroke_half_width(binary))

    semantic = analyze_diagram(shapes, 400, 400, [], right_angles=markers)
    marker_elements = [e for e in semantic.elements if e.geometry.get("right_angle_marker")]
    svg = render_tactile_svg(semantic.elements, 400, 400)

    assert len(marker_elements) == 1
    assert marker_elements[0].type is GeometryType.ANGLE
    assert marker_elements[0].semantic_properties["right_angle"] is True
    assert svg.count('class="right-angle"') == 1


def test_right_angle_symbol_stays_inside_short_edges():
    element = DetectedElement(
        id="ra", type=GeometryType.ANGLE,
        geometry={"vertex": [100, 100], "arms": [[110, 100], [100, 110]], "degrees": 90.0, "right_angle_marker": True},
        confidence=0.85, confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="right_angle_marker",
    )

    svg = render_tactile_svg([element], 400, 400)

    assert 'points="104.0,100.0 104.0,104.0 100.0,104.0"' in svg


def _worksheet_page() -> np.ndarray:
    page = _blank(1200, 1600)
    cv2.rectangle(page, (5, 5), (1194, 1594), (0, 0, 0), 4)
    for row in range(14):
        y = 120 + row * 40
        cv2.putText(page, "Represent the square root of 2 on the number line", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(page, (400, 1450), (800, 1520), (140, 40, 110), -1)
    cv2.putText(page, "byjus.com", (480, 1500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.line(page, (300, 1300), (900, 1300), (0, 0, 0), 3)
    cv2.line(page, (300, 1300), (300, 800), (0, 0, 0), 3)
    cv2.line(page, (300, 800), (900, 1300), (0, 0, 0), 3)
    return page


def test_full_page_isolates_the_figure_from_text_and_page_furniture():
    page = _worksheet_page()
    binary = preprocess_image(page)

    regions = find_diagram_regions(binary, page)

    assert regions is not None and len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert x0 <= 300 and y0 <= 800 and x1 >= 900 and y1 >= 1300
    assert y0 > 700 and y1 < 1450
    masked = mask_to_regions(binary, regions)
    assert np.count_nonzero(masked[:700]) == 0
    assert inside_regions([(600, 1310), (620, 1310), (620, 1330), (600, 1330)], regions)
    assert not inside_regions([(60, 120), (90, 120), (90, 140), (60, 140)], regions)


def test_cropped_diagram_with_labels_is_not_masked(fixture_directory):
    image = cv2.imread(str(fixture_directory / "labelled_triangle_worksheet.png"))

    assert find_diagram_regions(preprocess_image(image), image) is None


class _PageReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[60, 95], [700, 95], [700, 130], [60, 130]], "Represent the square root", 0.95),
            ([[280, 1310], [310, 1310], [310, 1340], [280, 1340]], "O", 0.95),
        ]


class _PageLouis:
    def translateString(self, tables, text):
        return "⠠⠕" if text == "O" else "⠠⠗"


def test_full_page_pipeline_keeps_only_figure_geometry_and_labels():
    ok, encoded = cv2.imencode(".png", _worksheet_page())
    assert ok

    result = build_full_analysis(
        encoded.tobytes(),
        ocr_provider=EasyOcrProvider(reader=_PageReader()),
        braille_translator=LouisBrailleTranslator(bindings=_PageLouis()),
    )

    assert [label["text"] for label in result.labels] == ["O"]
    geometry = [e for e in result.semantic_geometry.elements if e.type is not GeometryType.TEXT_LABEL]
    assert [e.type for e in geometry] == [GeometryType.TRIANGLE]
    x, y, w, h = geometry[0].bbox
    assert 290 <= x and 790 <= y and x + w <= 910 and y + h <= 1310
