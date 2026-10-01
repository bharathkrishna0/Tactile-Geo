import math

import cv2
import numpy as np

from app.services.segment_geometry import normalize_segments
from app.services.tactile_rules import TACTILE_RULES

# Half of the simplification tolerance: raw Hough fragments of one drawn edge sit
# within a few pixels of each other, so the merge distance is deliberately tighter
# than the semantic-level merge distance.
COLLINEAR_DISTANCE_TOLERANCE_PX = TACTILE_RULES.collinear_merge_tolerance_px / 2.0
COLLINEAR_ANGLE_TOLERANCE_DEG = 3.0


def _line_segments(binary_image: np.ndarray, edge_sensitivity: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, max(20, 110 - edge_sensitivity), minLineLength=max(20, 90 - edge_sensitivity), maxLineGap=12)
    if lines is None:
        return []
    raw = [((int(x1), int(y1)), (int(x2), int(y2))) for x1, y1, x2, y2 in lines[:, 0]]
    return [
        ((int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))))
        for a, b in normalize_segments(raw, COLLINEAR_ANGLE_TOLERANCE_DEG, COLLINEAR_DISTANCE_TOLERANCE_PX)
    ]


def extract_shapes(binary_image: np.ndarray, edge_sensitivity: int = 50) -> list[dict]:
    shapes: list[dict] = []
    for start, end in _line_segments(binary_image, edge_sensitivity):
        shapes.append({"type": "line", "points": [start, end]})
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        approximation = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        points = [(int(point[0][0]), int(point[0][1])) for point in approximation]
        if len(points) >= 3:
            shapes.append({"type": "contour", "points": points})
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            a, b = max(major, minor) / 2, min(major, minor) / 2
            if a > 0:
                aspect = b / a
                if aspect < 0.85 and cv2.contourArea(contour) > 200:
                    shapes.append({
                        "type": "ellipse",
                        "center": (round(cx), round(cy)),
                        "semi_axes": (round(a), round(b)),
                        "angle": round(angle, 1),
                        "area": round(math.pi * a * b, 2),
                        "contour_points": points,
                    })
    return shapes

def drop_shapes_inside_text_regions(shapes: list[dict], text_boxes: list[list[tuple[int, int]]], padding: int = 2) -> list[dict]:
    """Remove contours that fall inside an OCR text region.

    Letter glyphs have enough closed area to survive the contour-area filter, so
    without this a worksheet label is vectorized twice: once as braille text and
    once as a tiny polygon. The polygon then trips the minimum-size blocking QA
    check, blocking export on a correctly processed worksheet.
    """
    if not text_boxes:
        return shapes
    regions: list[tuple[int, int, int, int]] = []
    for box in text_boxes:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        regions.append((min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding))

    kept: list[dict] = []
    for shape in shapes:
        if shape["type"] == "line":
            kept.append(shape)
            continue
        points = shape.get("points") or ([shape["center"]] if shape.get("center") else [])
        if not points:
            kept.append(shape)
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            xs = [cx - rx, cx + rx]
            ys = [cy - ry, cy + ry]
        box = (min(xs), min(ys), max(xs), max(ys))
        if any(_boxes_overlap(box, region) for region in regions):
            continue
        kept.append(shape)
    return kept


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def shapes_to_svg(shapes: list[dict], width: int, height: int) -> str:
    elements: list[str] = []
    for shape in shapes:
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            angle = shape.get("angle", 0)
            elements.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({angle} {cx} {cy})" />')
            continue
        points = shape.get("points")
        if not points:
            continue
        if shape["type"] == "line":
            (x1, y1), (x2, y2) = points
            elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        else:
            elements.append('<polygon points="' + ' '.join(f"{x},{y}" for x, y in points) + '" />')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Raw geometry vector preview">'
            '<rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + ''.join(elements) + '</g></svg>')
