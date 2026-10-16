"""Small clean line art: kept unsharpened, thin strokes not shattered, degree
signs read from the ink, every arc at a vertex, and short crossing ticks."""
import math

import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType
from app.services.image_enhancement import enhance_copy
from app.services.image_preprocessing import is_clean_line_art, preprocess_image
from app.services.label_association import associate_label, names_a_point
from app.services.markers import detect_angle_arcs, drop_ticks_on_arcs
from app.services.ocr import OcrDetection
from app.services.ocr_postprocessing import postprocess_detections, restore_degrees
from app.services.vectorization import recover_ticks
from app.models.geometry import SemanticGeometry
from app.services.tactile_qa import outline_segments, run_tactile_qa
from app.services.tactile_simplification import simplify_geometry
from app.services.tactile_svg import angle_arc_points, is_embossed_geometry, render_tactile_svg


def _jpeg(image: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _small_drawing() -> np.ndarray:
    image = np.full((360, 480, 3), 255, dtype=np.uint8)
    for a, b in [((95, 292), (390, 292)), ((95, 292), (157, 120)), ((157, 120), (304, 133)),
                 ((304, 133), (390, 292)), ((157, 120), (390, 292))]:
        cv2.line(image, a, b, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "x = ?", (360, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return _jpeg(image)


def _noisy_photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(170, 25, (360, 480)), 0, 255).astype(np.uint8)
    cv2.line(image, (40, 300), (440, 300), 30, 3)
    return _jpeg(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))


def test_clean_line_art_is_recognised_and_paper_noise_is_not():
    assert is_clean_line_art(cv2.cvtColor(_small_drawing(), cv2.COLOR_BGR2GRAY))
    assert not is_clean_line_art(cv2.cvtColor(_noisy_photo(), cv2.COLOR_BGR2GRAY))


def test_enhancement_leaves_clean_full_contrast_line_art_unsharpened():
    drawing = _small_drawing()
    assert np.array_equal(enhance_copy(drawing), drawing)


def test_thin_jpeg_strokes_survive_preprocessing_unbroken():
    mask = preprocess_image(enhance_copy(_small_drawing()), 50)
    # Sample along the diagonal: every point of the stroke must still be inked.
    samples = [(round(157 + (390 - 157) * k / 40), round(120 + (292 - 120) * k / 40)) for k in range(2, 39)]
    hits = [mask[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].any() for x, y in samples]
    assert all(hits)


# --- degree signs --------------------------------------------------------------

def _degree_label(text: str, sign: bool = True) -> tuple[np.ndarray, list[tuple[int, int]]]:
    gray = np.full((60, 120), 255, dtype=np.uint8)
    cv2.putText(gray, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    right = 20 + w
    if sign:
        cv2.circle(gray, (right + 6, 40 - h + 3), 3, 0, 1, cv2.LINE_AA)
        right += 12
    return gray, [(16, 40 - h - 3), (right + 3, 40 - h - 3), (right + 3, 45), (16, 45)]


def _det(text, bbox):
    return OcrDetection(text=text, bbox=bbox, confidence=0.4)


def test_degree_sign_read_as_a_character_or_dropped_is_restored():
    gray, bbox = _degree_label("70")
    for read in ("70?", "708", "70o", "70"):
        assert restore_degrees([_det(read, bbox)], gray)[0].text == "70\u00b0"


def test_full_height_last_digit_is_not_a_degree_sign():
    gray, bbox = _degree_label("358", sign=False)
    assert restore_degrees([_det("358", bbox)], gray)[0].text == "358"


def test_speck_after_a_number_is_not_a_degree_sign():
    gray, bbox = _degree_label("13", sign=False)
    right = bbox[1][0] - 3
    gray[30, right - 1] = 0
    assert restore_degrees([_det("13", bbox)], gray)[0].text == "13"


def test_degree_labels_pass_text_validation():
    gray, bbox = _degree_label("25")
    restored = restore_degrees([_det('25"', bbox)], gray)
    assert [d.text for d in postprocess_detections(restored)] == ["25\u00b0"]


def test_glyph_read_inside_a_restored_degree_label_is_dropped():
    gray, bbox = _degree_label("70")
    inner = [(20, 18), (34, 18), (34, 40), (20, 40)]
    restored = restore_degrees([_det("70?", bbox), _det("7", inner)], gray)
    assert [d.text for d in restored] == ["70\u00b0"]


# --- arcs at a vertex with an inner arm ----------------------------------------

VERTEX = (400, 300)


def _arm(degrees, length=300):
    return (round(VERTEX[0] + length * math.cos(math.radians(degrees))),
            round(VERTEX[1] + length * math.sin(math.radians(degrees))))


def _fan(arc_from, arc_to, radius=40, thickness=3):
    arms = [_arm(180), _arm(215), _arm(240)]
    binary = np.zeros((420, 520), dtype=np.uint8)
    for end in arms:
        cv2.line(binary, VERTEX, end, 255, thickness)
    cv2.ellipse(binary, VERTEX, (radius, radius), 0, arc_from, arc_to, 255, 2)
    return binary, [{"type": "line", "points": [VERTEX, end]} for end in arms]


def test_arc_crossing_an_inner_arm_marks_both_angles():
    binary, shapes = _fan(180, 240)
    arcs = sorted(detect_angle_arcs(binary, shapes, 1.5), key=lambda a: a["degrees"])
    assert [round(a["degrees"]) for a in arcs] == [25, 35]


def test_arc_between_adjacent_arms_marks_only_that_angle():
    binary, shapes = _fan(215, 240)
    arcs = detect_angle_arcs(binary, shapes, 1.5)
    assert [round(a["degrees"]) for a in arcs] == [25]


def test_small_arc_on_a_thick_stroke_is_found_despite_clipped_ends():
    binary, shapes = _fan(180, 215, radius=24, thickness=4)
    arcs = detect_angle_arcs(binary, shapes, 2.0)
    assert [round(a["degrees"]) for a in arcs] == [35]


def test_arc_crossing_an_arm_is_not_a_tick():
    binary, shapes = _fan(180, 240)
    arcs = detect_angle_arcs(binary, shapes, 1.5)
    crossing = (round(VERTEX[0] + 40 * math.cos(math.radians(215))), round(VERTEX[1] + 40 * math.sin(math.radians(215))))
    tick = {"type": "line", "points": [(crossing[0] - 5, crossing[1] + 4), (crossing[0] + 5, crossing[1] - 4)], "role": "tick"}
    assert drop_ticks_on_arcs(shapes + [tick], arcs, 1.5) == shapes


# --- short crossing ticks -------------------------------------------------------

def _side_with_tick(tick_half=5, arrow=False):
    binary = np.zeros((360, 480), dtype=np.uint8)
    start, end = (95, 292), (157, 120)
    cv2.line(binary, start, end, 255, 3)
    length = math.dist(start, end)
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    mx, my = start[0] + ux * length / 2, start[1] + uy * length / 2
    cv2.line(binary, (round(mx + uy * tick_half), round(my - ux * tick_half)),
             (round(mx - uy * tick_half), round(my + ux * tick_half)), 255, 2)
    if arrow:
        tip = np.array(end, dtype=float)
        base = tip - 14 * np.array([ux, uy])
        wing = 6 * np.array([-uy, ux])
        cv2.fillPoly(binary, [np.array([tip, base + wing, base - wing], dtype=np.int32)], 255)
    return binary, [{"type": "line", "points": [start, end]}], (mx, my)


def test_short_tick_barely_clearing_the_stroke_is_found():
    binary, shapes, centre = _side_with_tick()
    ticks = recover_ticks(binary, shapes, 1.5)
    assert len(ticks) == 1
    (ax, ay), (bx, by) = ticks[0]["points"]
    assert math.dist(((ax + bx) / 2, (ay + by) / 2), centre) <= 3


def test_arrowhead_widening_is_not_a_tick():
    binary, shapes, _ = _side_with_tick(tick_half=0, arrow=True)
    assert recover_ticks(binary, shapes, 1.5) == []


# --- angle labels ---------------------------------------------------------------

def _element(id_, type_, geometry, bbox=None):
    return DetectedElement(id=id_, type=type_, geometry=geometry, confidence=0.9,
                           confidence_level=ConfidenceLevel.HIGH, needs_review=False, source="test", bbox=bbox)


def _corner_with_arcs():
    base = _element("base", GeometryType.LINE_SEGMENT, {"start": [390, 292], "end": [94, 292]})
    diagonal = _element("diag", GeometryType.LINE_SEGMENT, {"start": [390, 292], "end": [157, 120]})
    side = _element("side", GeometryType.LINE_SEGMENT, {"start": [390, 292], "end": [304, 133]})
    corner = _element("corner", GeometryType.POINT, {"position": [390, 292]})
    lower = _element("lower", GeometryType.ANGLE, {"vertex": [390, 292], "arms": [[270, 292], [293, 221]],
                                                   "degrees": 36.4, "angle_marker": True, "arc_radius": 40})
    upper = _element("upper", GeometryType.ANGLE, {"vertex": [390, 292], "arms": [[293, 221], [333, 186]],
                                                   "degrees": 25.2, "angle_marker": True, "arc_radius": 40})
    return [base, diagonal, side, corner, lower, upper]


def _text(text, bbox, position=None):
    x, y, w, h = bbox
    return _element("label", GeometryType.TEXT_LABEL,
                    {"text": text, "position": position or [x + w / 2, y + h / 2]}, bbox=bbox)


def test_angle_measure_between_arc_arms_names_that_angle():
    elements = _corner_with_arcs()
    # Braille for the label was moved below the base line; the text itself sits between the arms.
    assert associate_label(_text("35\u00b0", (320, 266, 40, 20), position=[364, 306]), elements)["target_id"] == "lower"
    assert associate_label(_text("25\u00b0", (335, 235, 30, 20)), elements)["target_id"] == "upper"


def test_angle_measure_does_not_name_the_vertex_point():
    assert not names_a_point("70\u00b0")
    assert names_a_point("A")


def test_statement_is_not_read_as_a_tick_value():
    side = _element("side", GeometryType.LINE_SEGMENT, {"start": [390, 292], "end": [304, 133]})
    tick = _element("tick", GeometryType.LINE_SEGMENT, {"start": [340, 215], "end": [351, 209]})
    tick.semantic_properties["role"] = "tick"
    result = associate_label(_text("X=?", (359, 167, 50, 22)), [side, tick])
    assert result["target_id"] == "side"


def _quadrilateral_with_arc_labels():
    elements = _corner_with_arcs() + [
        _element("left", GeometryType.LINE_SEGMENT, {"start": [94, 292], "end": [157, 120]}),
        _element("top", GeometryType.LINE_SEGMENT, {"start": [157, 120], "end": [304, 133]}),
    ]
    lower = next(e for e in elements if e.id == "lower")
    lower.associated_label_id = "deg35"
    label = _element("deg35", GeometryType.TEXT_LABEL, {"text": "35\u00b0", "braille": "\u283c\u2809\u2811\u2818\u281a",
                                                        "position": [364, 306]}, bbox=(320, 266, 40, 20))
    label.associated_label_id = "lower"
    return elements + [label]


def test_drawn_angle_arc_is_embossed_along_its_measured_sweep():
    lower = next(e for e in _corner_with_arcs() if e.id == "lower")
    assert is_embossed_geometry(lower)
    points = angle_arc_points(lower.geometry)
    assert all(abs(math.dist(p, (390, 292)) - 40) < 1e-6 for p in points)
    turn = math.degrees(math.atan2(points[-1][1] - 292, points[-1][0] - 390) - math.atan2(points[0][1] - 292, points[0][0] - 390))
    assert abs(abs((turn + 180) % 360 - 180) - 36.4) < 2.0
    assert len(outline_segments(lower)) == len(points) - 1
    svg = render_tactile_svg(_corner_with_arcs(), 480, 360)
    assert svg.count('class="angle-arc"') == 2


def test_inferred_angle_without_a_drawn_arc_is_not_embossed():
    angle = _element("inferred", GeometryType.ANGLE, {"vertex": [304, 133], "arms": [[157, 120], [390, 292]],
                                                      "degrees": 123.5})
    assert not is_embossed_geometry(angle)
    assert 'class="angle-arc"' not in render_tactile_svg([angle], 480, 360)


def test_angle_braille_is_seated_inside_its_angle_and_clear_of_the_arc():
    semantic = SemanticGeometry(elements=_quadrilateral_with_arc_labels(), image_width=480, image_height=360)
    simplified = simplify_geometry(semantic)
    label = next(e for e in simplified.elements if e.id == "deg35")
    x, y = label.geometry["position"]
    direction = math.degrees(math.atan2(y - 292, x - 390)) % 360
    # Between the base (180 degrees) and the diagonal (about 216 degrees), past the 40 px arc.
    assert 180 < direction < 217
    assert math.dist((x, y), (390, 292)) > 40
    qa = run_tactile_qa(simplified.elements, 480, 360)
    assert "braille_on_line" not in {i.check for i in qa.issues}
