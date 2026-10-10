"""Text glyphs are erased before tracing; strokes near or entering labels survive."""
import cv2
import numpy as np

from app.services.vectorization import extract_shapes, mask_text_glyphs


def _box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _canvas():
    return np.zeros((300, 400), dtype=np.uint8)


def _text(img, text, org, scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2)


def _has_line(shapes, start, end, tol=8):
    for s in shapes:
        if s["type"] != "line":
            continue
        a, b = s["points"]
        for p, q in ((a, b), (b, a)):
            if np.hypot(p[0] - start[0], p[1] - start[1]) <= tol and np.hypot(q[0] - end[0], q[1] - end[1]) <= tol:
                return True
    return False


def test_isolated_label_glyphs_are_erased():
    img = _canvas()
    _text(img, "AB", (100, 100))
    masked = mask_text_glyphs(img, [_box(95, 75, 145, 105)])
    assert masked.sum() == 0
    assert img.sum() > 0


def test_label_beside_a_line_keeps_the_line():
    img = _canvas()
    cv2.line(img, (50, 150), (350, 150), 255, 3)
    _text(img, "C", (190, 135))
    masked = mask_text_glyphs(img, [_box(185, 112, 212, 140)])
    assert masked[150, 50:350].all()
    assert masked[112:140, 185:212].sum() == 0


def test_label_touching_a_line_is_not_cut_out_of_the_line():
    img = _canvas()
    cv2.line(img, (50, 150), (350, 150), 255, 3)
    _text(img, "D", (190, 150))
    masked = mask_text_glyphs(img, [_box(185, 125, 212, 152)])
    assert masked[150, 50:350].all()
    assert _has_line(extract_shapes(masked), (50, 150), (350, 150))


def test_leader_line_entering_a_label_box_survives():
    img = _canvas()
    _text(img, "P", (100, 100))
    cv2.line(img, (110, 95), (300, 250), 255, 2)
    masked = mask_text_glyphs(img, [_box(95, 75, 125, 105)])
    assert masked[250, 295:305].any()
    assert masked[95, 108:113].any()


def test_dimension_text_between_extension_lines_keeps_the_dimension():
    img = _canvas()
    cv2.line(img, (60, 80), (60, 220), 255, 2)
    cv2.line(img, (340, 80), (340, 220), 255, 2)
    cv2.line(img, (60, 200), (160, 200), 255, 2)
    cv2.line(img, (240, 200), (340, 200), 255, 2)
    _text(img, "5 cm", (165, 210))
    masked = mask_text_glyphs(img, [_box(160, 185, 238, 215)])
    assert masked[80:220, 60].all() and masked[80:220, 340].all()
    assert masked[200, 60:160].all() and masked[200, 240:340].all()
    assert masked[188:212, 165:236].sum() == 0
    shapes = extract_shapes(masked)
    assert _has_line(shapes, (60, 80), (60, 220)) and _has_line(shapes, (340, 80), (340, 220))


def test_angle_label_inside_an_angle_keeps_both_arms():
    img = _canvas()
    cv2.line(img, (50, 250), (350, 250), 255, 3)
    cv2.line(img, (50, 250), (300, 60), 255, 3)
    _text(img, "40", (130, 235), 0.6)
    masked = mask_text_glyphs(img, [_box(126, 218, 160, 238)])
    assert masked[218:238, 126:160].sum() == 0
    shapes = extract_shapes(masked)
    assert _has_line(shapes, (50, 250), (350, 250)) and _has_line(shapes, (50, 250), (300, 60))


def test_no_text_returns_the_same_mask():
    img = _canvas()
    cv2.line(img, (50, 150), (350, 150), 255, 3)
    assert mask_text_glyphs(img, []) is img


def test_dash_read_as_a_digit_is_kept():
    img = _canvas()
    for y in range(40, 260, 24):
        cv2.line(img, (200, y), (200, y + 12), 255, 3)
    masked = mask_text_glyphs(img, [_box(197, 112, 203, 126)])
    assert np.array_equal(masked, img)
