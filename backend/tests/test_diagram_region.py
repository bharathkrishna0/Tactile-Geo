import cv2
import numpy as np

from app.services.diagram_region import find_diagram_regions


def _page_with_clutter() -> np.ndarray:
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    for row in range(5):
        for col in range(12):
            x, y = 30 + col * 45, 30 + row * 25
            cv2.rectangle(page, (x, y), (x + 6, y + 10), (0, 0, 0), 1)
    cv2.polylines(page, [np.array([[100, 400], [300, 400], [200, 250]])], True, (0, 0, 0), 2)
    return page


def _binary(page: np.ndarray) -> np.ndarray:
    return np.where(page.min(axis=2) < 240, 255, 0).astype(np.uint8)


def _covers(regions, x: int, y: int) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in regions)


def test_small_dark_figure_away_from_main_figure_is_kept():
    page = _page_with_clutter()
    cv2.rectangle(page, (450, 650), (490, 690), (0, 0, 0), 2)
    regions = find_diagram_regions(_binary(page), page)
    assert regions is not None
    assert _covers(regions, 200, 350)
    assert _covers(regions, 470, 670)


def test_pale_watermark_letter_is_not_a_figure():
    page = _page_with_clutter()
    cv2.rectangle(page, (450, 650), (490, 690), (210, 210, 210), 2)
    regions = find_diagram_regions(_binary(page), page)
    assert regions is not None
    assert not _covers(regions, 470, 670)


def test_prose_line_is_not_a_figure():
    page = _page_with_clutter()
    cv2.rectangle(page, (400, 650), (560, 662), (0, 0, 0), -1)
    cv2.rectangle(page, (400, 700), (440, 712), (0, 0, 0), 1)
    regions = find_diagram_regions(_binary(page), page)
    assert regions is not None
    assert not _covers(regions, 420, 706)
