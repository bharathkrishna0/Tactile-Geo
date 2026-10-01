"""Diagram region isolation for full worksheet pages (Model A).

A scanned or photographed worksheet carries headings, paragraphs, logos and
page furniture around the figure. Left in, their glyph strokes and borders turn
into false geometry and their text into paragraphs of Braille. This keeps only
the regions built around figure strokes, plus a margin for their labels.
"""

from __future__ import annotations

import cv2
import numpy as np

# A figure stroke spans a meaningful share of the page; letters do not.
LONG_COMPONENT_FRACTION = 0.08
# Strokes touching the image edge are page borders, header bars or scan edges.
BORDER_FRACTION = 0.01
# Strokes inside a large, mostly solid dark blob are the edges and lettering of
# a filled banner, button or logo, not a line drawing.
SOLID_FILL_FRACTION = 0.5
BANNER_MIN_THICKNESS_FRACTION = 0.015
REGION_MARGIN_FRACTION = 0.035
# A page needs isolating only when many separate marks (words, logos, rules)
# lie outside the figures; a labelled diagram has just a few label glyphs.
MIN_CLUTTER_COMPONENTS = 40

Box = tuple[int, int, int, int]


def _banner_boxes(image: np.ndarray, short_side: int, pad: int) -> list[Box]:
    """Boxes of large, mostly solid dark blobs: banners, buttons and logos."""
    gray = image if image.ndim == 2 else image.min(axis=2)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    boxes: list[Box] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if min(w, h) >= BANNER_MIN_THICKNESS_FRACTION * short_side and area > SOLID_FILL_FRACTION * w * h:
            boxes.append((x - pad, y - pad, x + w + pad, y + h + pad))
    return boxes


def _merge_boxes(boxes: list[Box]) -> list[Box]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def find_diagram_regions(binary: np.ndarray, image: np.ndarray) -> list[Box] | None:
    """Boxes (x0, y0, x1, y1) around the figures on a page, or None to keep everything."""
    height, width = binary.shape[:2]
    short_side = min(height, width)
    border = max(1, round(short_side * BORDER_FRACTION))
    margin = round(short_side * REGION_MARGIN_FRACTION)
    banners = _banner_boxes(image, short_side, border)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[Box] = []
    for index in range(1, count):
        x, y, w, h = (int(v) for v in stats[index][:4])
        if max(w, h) < LONG_COMPONENT_FRACTION * short_side:
            continue
        if x <= border or y <= border or x + w >= width - border or y + h >= height - border:
            continue
        if any(bx0 <= x and by0 <= y and x + w <= bx1 and y + h <= by1 for bx0, by0, bx1, by1 in banners):
            continue
        boxes.append((max(0, x - margin), max(0, y - margin), min(width, x + w + margin), min(height, y + h + margin)))
    if not boxes:
        return None
    regions = _merge_boxes(boxes)
    outside = 0
    for index in range(1, count):
        x, y, w, h = (int(v) for v in stats[index][:4])
        cx, cy = x + w / 2, y + h / 2
        if not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions):
            outside += 1
    return regions if outside >= MIN_CLUTTER_COMPONENTS else None


def mask_to_regions(binary: np.ndarray, regions: list[Box]) -> np.ndarray:
    masked = np.zeros_like(binary)
    for x0, y0, x1, y1 in regions:
        masked[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
    return masked


def inside_regions(bbox: list[tuple[int, int]], regions: list[Box]) -> bool:
    cx = sum(point[0] for point in bbox) / len(bbox)
    cy = sum(point[1] for point in bbox) / len(bbox)
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions)
