"""Line ends that Hough runs on across a gap into other ink are cut back."""
import cv2
import numpy as np

from app.services.vectorization import extract_shapes


def _lines(binary):
    return [s["points"] for s in extract_shapes(binary, 50) if s["type"] == "line"]


def test_line_is_not_carried_across_a_gap_onto_a_dimension_line():
    binary = np.zeros((400, 600), dtype=np.uint8)
    cv2.line(binary, (300, 50), (300, 300), 255, 2)
    cv2.line(binary, (100, 315), (500, 315), 255, 2)
    cv2.line(binary, (300, 308), (300, 314), 255, 2)
    vertical = [p for p in _lines(binary) if abs(p[0][0] - p[1][0]) <= 2]
    assert vertical, "vertical line not found"
    assert max(y for _, y in vertical[0]) <= 303


def test_dashed_line_keeps_its_end_dashes():
    binary = np.zeros((300, 600), dtype=np.uint8)
    for x in range(100, 500, 14):
        cv2.line(binary, (x, 150), (x + 9, 150), 255, 2)
    lines = _lines(binary)
    assert lines
    xs = [x for points in lines for x, _ in points]
    assert min(xs) <= 103 and max(xs) >= 495
