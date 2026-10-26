"""Split a worksheet page into its separate figures (Model A, deterministic).

A page with several numbered figures is one drawing to the rest of the
pipeline, so every figure competes for one tactile sheet's feature budget.
This module finds each figure's strokes and nearby labels so each can be
processed and embossed on its own sheet. It measures; it never draws.

Ink is read from the darkest colour channel, so coloured strokes count and
pale watermarks do not. Strokes that are close together form one figure;
small marks (letters, degree signs, dashes, arc pieces) join the figure they
sit next to. Page furniture is dropped: border dots and rules along the page
edge, empty form boxes (answer and Name/Date boxes) and solid blobs (logos).
Text that is not near any figure, such as titles and questions, is left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

STROKE_EXTENT_FRACTION = 0.05
STROKE_GAP_FRACTION = 0.045
LABEL_GAP_FRACTION = 0.035
EDGE_BAND_FRACTION = 0.035
PAGE_RULE_FRACTION = 0.6
#: A page rule or a writing line (``Name: ____``) is at most this thick.
RULE_THICKNESS_FRACTION = 0.01
WRITING_LINE_MIN_ASPECT = 8
#: A word blurred into one blob (title text on a photo) is short, wide and dense, unlike a thin stroke.
TEXT_BLOB_MAX_HEIGHT_FRACTION = 0.04
TEXT_BLOB_MIN_ASPECT = 1.5
TEXT_BLOB_MIN_FILL = 0.45
SOLID_FILL_SHARE = 0.45
FORM_BOX_MIN_ASPECT = 2.0
FORM_BOX_MAX_HEIGHT_FRACTION = 0.06
FORM_BOX_INTERIOR_INK = 0.02
FORM_BOX_EDGE_COVER = 0.6
FORM_PROMPT_GAP_FRACTION = 0.08
FIGURE_MARGIN_FRACTION = 0.015
MIN_SPECK_PX = 4
KEEP_DILATE_PX = 3
MIN_FIGURES = 2
CROP_PAD_FRACTION = 0.15
#: Small neighbouring shapes (a triangle beside a square, each with a caption) are one figure
#: when they are this much closer than the usual stroke gap allows.
SMALL_GROUP_FRACTION = 0.15
SMALL_GROUP_GAP_FACTOR = 2.0


@dataclass(frozen=True)
class Figure:
    #: The figure's ink plus a small margin.
    box: Box
    #: ``box`` grown with paper so Braille, which is larger than the printed text, still fits.
    crop: Box
    #: Which pixels of ``crop`` are this figure's ink.
    keep: np.ndarray
    page_size: tuple[int, int]

    def as_dict(self, number: int) -> dict:
        x0, y0, x1, y1 = self.crop
        return {"number": number, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                "page_width": self.page_size[0], "page_height": self.page_size[1], "box": list(self.box)}


def _ink(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    darkest = image.min(axis=2) if image.ndim == 3 else image
    paper = float(np.percentile(darkest, 75))
    ink_level = float(np.percentile(darkest, 0.5))
    if paper - ink_level < 40:
        return np.zeros(darkest.shape, np.uint8), darkest
    return (darkest < (paper + ink_level) / 2).astype(np.uint8), darkest


def _gap(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float(max(dx, dy))


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _is_form_box(ink: np.ndarray, box: Box, page_height: int) -> bool:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if h <= 0 or w / h < FORM_BOX_MIN_ASPECT or h > FORM_BOX_MAX_HEIGHT_FRACTION * page_height:
        return False
    mx, my = max(2, w // 6), max(2, h // 4)
    interior = ink[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if interior.size == 0 or interior.mean() > FORM_BOX_INTERIOR_INK:
        return False
    band = max(3, h // 3)
    edges = [ink[y0:y0 + band, x0:x1], ink[y1 - band:y1, x0:x1], ink[y0:y1, x0:x0 + band], ink[y0:y1, x1 - band:x1]]
    covered = sum(edge.any(axis=0 if k < 2 else 1).mean() > FORM_BOX_EDGE_COVER for k, edge in enumerate(edges))
    return covered >= 3


def _is_writing_line(labels: np.ndarray, index: int, box: Box, short: int) -> bool:
    """A long, evenly thin stroke with no ticks or arrowheads, like the line after ``Name:``."""
    x0, y0, x1, y1 = box
    if x1 - x0 < WRITING_LINE_MIN_ASPECT * (y1 - y0):
        return False
    columns = (labels[y0:y1, x0:x1] == index).sum(axis=0)
    columns = columns[columns > 0]
    typical = float(np.median(columns))
    return typical <= RULE_THICKNESS_FRACTION * short and columns.max() <= 2 * typical + 1


def _is_text_blob(labels: np.ndarray, index: int, box: Box, short: int) -> bool:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if h > TEXT_BLOB_MAX_HEIGHT_FRACTION * short or w < TEXT_BLOB_MIN_ASPECT * h:
        return False
    columns = (labels[y0:y1, x0:x1] == index).sum(axis=0)
    return float(np.median(columns[columns > 0])) >= TEXT_BLOB_MIN_FILL * h


def _is_form_prompt(box: Box, form_boxes: list[Box], short: int) -> bool:
    """A small mark on the same line as an answer box and just left of it, like ``x =``."""
    height = box[3] - box[1]
    for fx0, fy0, fx1, fy1 in form_boxes:
        overlap = min(box[3], fy1) - max(box[1], fy0)
        if overlap > 0.5 * height and 0 <= fx0 - box[2] <= FORM_PROMPT_GAP_FRACTION * short:
            return True
    return False


def find_figures(image: np.ndarray) -> list[Figure]:
    """Return the page's figures in reading order, or [] for a single-figure image."""
    ink, _ = _ink(image)
    height, width = ink.shape
    short = min(width, height)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    edge = EDGE_BAND_FRACTION * short
    strokes: list[tuple[int, Box]] = []
    marks: list[tuple[int, Box]] = []
    form_boxes: list[Box] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < MIN_SPECK_PX:
            continue
        box = (x, y, x + w, y + h)
        thin = RULE_THICKNESS_FRACTION * short
        if (w >= PAGE_RULE_FRACTION * width and h <= thin) or (h >= PAGE_RULE_FRACTION * height and w <= thin):
            continue
        if x < edge or y < edge or x + w > width - edge or y + h > height - edge:
            continue
        if max(w, h) >= STROKE_EXTENT_FRACTION * short and not _is_text_blob(labels, index, box, short):
            if _is_form_box(ink, box, height):
                form_boxes.append(box)
                continue
            strokes.append((index, box))
        else:
            marks.append((index, box))
    marks = [(index, box) for index, box in marks if not _is_form_prompt(box, form_boxes, short)]
    writing_lines = {index for index, box in strokes if _is_writing_line(labels, index, box, short)}
    groups = _merge([[i] for i, _ in strokes], [b for _, b in strokes], STROKE_GAP_FRACTION * short)
    groups = [(members, box) for members, box in groups
              if stats[members, cv2.CC_STAT_AREA].sum() < SOLID_FILL_SHARE * (box[2] - box[0]) * (box[3] - box[1])
              and not all(m in writing_lines for m in members)]
    groups = _merge_small(groups, STROKE_GAP_FRACTION * short, SMALL_GROUP_FRACTION * short)
    if len(groups) < MIN_FIGURES:
        return []
    groups = _attach(groups, marks, LABEL_GAP_FRACTION * short)
    margin = int(round(FIGURE_MARGIN_FRACTION * short))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * KEEP_DILATE_PX + 1, 2 * KEEP_DILATE_PX + 1))
    figures = []
    for members, (x0, y0, x1, y1) in _reading_order(groups):
        box = (max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin))
        pad = int(round(CROP_PAD_FRACTION * max(x1 - x0, y1 - y0)))
        crop = (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))
        window = labels[crop[1]:crop[3], crop[0]:crop[2]]
        keep = cv2.dilate(np.isin(window, members).astype(np.uint8), kernel) > 0
        figures.append(Figure(box, crop, keep, (width, height)))
    return figures


def _merge(members: list[list[int]], boxes: list[Box], gap: float) -> list[tuple[list[int], Box]]:
    groups = list(zip(members, boxes))
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if _gap(groups[a][1], groups[b][1]) <= gap:
                    groups[a] = (groups[a][0] + groups[b][0], _union(groups[a][1], groups[b][1]))
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    return groups


def _merge_small(groups, gap: float, small: float):
    def size(box: Box) -> int:
        return max(box[2] - box[0], box[3] - box[1])

    groups = list(groups)
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                box_a, box_b = groups[a][1], groups[b][1]
                if size(box_a) < small and size(box_b) < small and _gap(box_a, box_b) <= SMALL_GROUP_GAP_FACTOR * gap:
                    groups[a] = (groups[a][0] + groups[b][0], _union(box_a, box_b))
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    return groups


def _attach(groups, marks, gap: float):
    groups = [(list(m), b) for m, b in groups]
    pending = list(marks)
    for _ in range(3):
        left = []
        for index, box in pending:
            distances = [_gap(box, group_box) for _, group_box in groups]
            best = int(np.argmin(distances))
            if distances[best] <= gap:
                groups[best][0].append(index)
            else:
                left.append((index, box))
        for k, (members, group_box) in enumerate(groups):
            grown = group_box
            for index, box in pending:
                if index in members:
                    grown = _union(grown, box)
            groups[k] = (members, grown)
        if len(left) == len(pending):
            break
        pending = left
    return groups


def _reading_order(groups):
    ordered = sorted(groups, key=lambda g: (g[1][1] + g[1][3]) / 2)
    rows: list[list] = []
    for group in ordered:
        y0, y1 = group[1][1], group[1][3]
        for row in rows:
            ry0, ry1 = row[0][1][1], row[0][1][3]
            if min(y1, ry1) - max(y0, ry0) > 0.3 * min(y1 - y0, ry1 - ry0):
                row.append(group)
                break
        else:
            rows.append([group])
    return [group for row in rows for group in sorted(row, key=lambda g: g[1][0])]


def figure_image(image: np.ndarray, figure: Figure) -> np.ndarray:
    """Crop one figure and paint everything that is not its ink as paper."""
    x0, y0, x1, y1 = figure.crop
    crop = image[y0:y1, x0:x1].copy()
    darkest = image.min(axis=2) if image.ndim == 3 else image
    paper = np.percentile(image.reshape(-1, image.shape[2])[darkest.reshape(-1) >= np.percentile(darkest, 75)], 50, axis=0) \
        if image.ndim == 3 else np.percentile(image, 75)
    crop[~figure.keep] = paper
    return crop
