"""V2a geometry reconstruction: axis ticks and open strokes traced around both sides."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.diagram_region import find_diagram_regions, mask_to_regions
from app.services.image_enhancement import enhance_copy
from app.services.image_preprocessing import decode_image, preprocess_image, tick_ink_mask
from app.services.vectorization import MIN_ROUND_VERTICES, extract_shapes, recover_ticks


def _shape_line(a, b):
    return {"type": "line", "points": [a, b]}


def _ticks(shapes):
    return [s for s in shapes if s.get("role") == "tick"]


def _tick_x(tick):
    (x0, _), (x1, _) = tick["points"]
    return (x0 + x1) / 2


# --- ticks ---------------------------------------------------------------------

def _axis_page(tick_ys, xs=(175, 275, 375, 475, 575)):
    page = np.full((400, 700, 3), 255, dtype=np.uint8)
    for x in range(100, 650, 50):
        cv2.line(page, (x, 60), (x, 340), (215, 215, 215), 1)  # light grid
    cv2.line(page, (80, 200), (640, 200), (0, 0, 0), 4)
    for x in xs:
        cv2.line(page, (x, tick_ys[0]), (x, tick_ys[1]), (0, 0, 0), 2)
    return page


BENCHMARK = Path(__file__).resolve().parents[2] / "evaluation" / "dataset"


def _benchmark_ticks(image_id, with_evidence):
    image = decode_image((BENCHMARK / "images" / f"{image_id}.jpg").read_bytes())
    processing = enhance_copy(image)
    stroke_mask = preprocess_image(processing, 50)
    evidence = tick_ink_mask(processing, 50) if with_evidence else None
    regions = find_diagram_regions(stroke_mask, processing)
    if regions:
        stroke_mask = mask_to_regions(stroke_mask, regions)
        evidence = mask_to_regions(evidence, regions) if evidence is not None else None
    return _ticks(extract_shapes(stroke_mask, 50, evidence))


def _midpoint(points):
    (x0, y0), (x1, y1) = points
    return ((x0 + x1) / 2, (y0 + y1) / 2)


@pytest.mark.parametrize("image_id", ["math_046", "math_096"])
def test_thin_axis_ticks_cut_by_the_stroke_mask_are_recovered_from_unthinned_ink(image_id):
    annotation = json.loads((BENCHMARK / "annotations" / f"{image_id}.json").read_text())
    truth = [_midpoint(o["geometry"]["points"]) for o in annotation["objects"] if o["type"] == "tick"]
    without = _benchmark_ticks(image_id, with_evidence=False)
    found = _benchmark_ticks(image_id, with_evidence=True)
    assert len(found) >= len(without) + 3
    for tick in found:
        assert min(math.dist(_midpoint(tick["points"]), t) for t in truth) <= 8


def test_tick_evidence_ignores_light_grid_lines_and_dark_paper():
    page = _axis_page((193, 207))
    evidence = tick_ink_mask(page, 50)
    assert evidence[120, 300] == 0  # a light grid line is not tick ink
    assert evidence[200, 300] == 255  # the bold axis is
    assert tick_ink_mask(np.full((100, 100, 3), 60, dtype=np.uint8), 50) is None


def test_evenly_spaced_one_sided_ticks_form_a_series():
    binary = np.zeros((200, 600), dtype=np.uint8)
    cv2.line(binary, (40, 100), (560, 100), 255, 2)
    for x in (100, 200, 300, 400, 500):
        cv2.line(binary, (x, 100), (x, 112), 255, 2)
    ticks = recover_ticks(binary, [_shape_line((40, 100), (560, 100))], 1.0)
    assert sorted(round(_tick_x(t)) for t in ticks) == [100, 200, 300, 400, 500]
    assert all(t["tick_evidence"] == "series" for t in ticks)
    for tick in ticks:
        (_, y0), (_, y1) = tick["points"]
        assert min(y0, y1) <= 101 and max(y0, y1) >= 110  # from the axis outward


def test_irregular_one_sided_strokes_are_not_ticks():
    binary = np.zeros((200, 600), dtype=np.uint8)
    cv2.line(binary, (40, 100), (560, 100), 255, 2)
    for x in (100, 170, 390):  # uneven spacing: label strokes or arc ends
        cv2.line(binary, (x, 100), (x, 112), 255, 2)
    assert recover_ticks(binary, [_shape_line((40, 100), (560, 100))], 1.0) == []


# --- open strokes traced around both sides -------------------------------------

def _hexagon(center=(300, 300), radius=120):
    return [
        (int(center[0] + radius * math.cos(math.radians(60 * k))), int(center[1] + radius * math.sin(math.radians(60 * k))))
        for k in range(6)
    ]


def _covering_lines(shapes, a, b, band=5.0):
    def dist(p, s, e):
        dx, dy = e[0] - s[0], e[1] - s[1]
        t = max(0.0, min(1.0, ((p[0] - s[0]) * dx + (p[1] - s[1]) * dy) / (dx * dx + dy * dy)))
        return math.dist(p, (s[0] + t * dx, s[1] + t * dy))
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return [s for s in shapes if s["type"] == "line" and dist(mid, *s["points"]) <= band]


def test_broken_polygon_ring_is_one_open_path_not_a_doubled_outline():
    binary = np.zeros((600, 600), dtype=np.uint8)
    corners = _hexagon()
    cv2.polylines(binary, [np.array(corners, np.int32)], False, 255, 3)  # five sides, ring left open
    shapes = extract_shapes(binary, 50)
    assert not [s for s in shapes if s["type"] == "contour"]
    for a, b in zip(corners, corners[1:]):
        assert len(_covering_lines(shapes, a, b)) == 1


def test_closed_thin_polygon_stays_one_closed_outline():
    binary = np.zeros((600, 600), dtype=np.uint8)
    cv2.polylines(binary, [np.array(_hexagon(), np.int32)], True, 255, 3)
    shapes = extract_shapes(binary, 50)
    assert len([s for s in shapes if s["type"] == "contour"]) == 1
    assert not [s for s in shapes if s.get("origin") == "folded_outline"]


def test_filled_blob_is_not_treated_as_an_open_stroke():
    binary = np.zeros((400, 400), dtype=np.uint8)
    cv2.fillPoly(binary, [np.array([(100, 100), (300, 120), (200, 300)], np.int32)], 255)
    shapes = extract_shapes(binary, 50)
    assert not [s for s in shapes if s.get("origin") == "folded_outline"]



def _benchmark_shapes(image_id):
    path = next(BENCHMARK.glob(f"images/{image_id}.*"))
    processing = enhance_copy(decode_image(path.read_bytes()))
    stroke_mask = preprocess_image(processing, 50)
    regions = find_diagram_regions(stroke_mask, processing)
    if regions:
        stroke_mask = mask_to_regions(stroke_mask, regions)
    return extract_shapes(stroke_mask, 50)


@pytest.mark.parametrize(("image_id", "corners"), [("math_055", None), ("math_063", 8)])
def test_ring_that_broke_in_thresholding_is_one_outline_traced_once(image_id, corners):
    """math_055 is a circle, math_063 a regular octagon: the break must not add a corner."""
    folded = [s for s in _benchmark_shapes(image_id) if s.get("origin") == "folded_outline"]
    assert len(folded) == 1 and folded[0]["type"] == "contour"
    points = folded[0]["points"]
    if corners is None:
        assert len(points) >= MIN_ROUND_VERTICES
    else:
        assert len(points) == corners
    # One side of the stroke only: no vertex is doubled across the stroke width.
    assert all(math.dist(p, q) > 8 for i, p in enumerate(points) for q in points[i + 1:])
