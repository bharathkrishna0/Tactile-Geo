import cv2
import numpy as np

def decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image

# A 3x3 opening erases hairline strokes such as thin coloured textbook figures.
# A thresholded component that loses more than this share of its ink to the
# opening is a hairline, and is kept whole instead of being eroded away.
THIN_STROKE_KEPT_FRACTION = 0.4
MIN_THIN_COMPONENT_AREA = 25
# A hairline component spans far more than its ink; noise specks do not.
MIN_THIN_COMPONENT_SPAN_PX = 15
# Hairline recovery is for light paper; on a dark or underexposed photo the thin
# threshold response is sensor noise and edge halos, not drawn lines.
MIN_PAPER_LEVEL = 128
# A hairline joined to a bold stroke (a thin right-angle marker on a thick side)
# survives as one component, so the opening still erases it. Such a piece runs
# away from the bold ink; the fringe of a bold stroke runs along it.
MAX_ATTACHED_HAIRLINE_WIDTH_PX = 3
MAX_ATTACHED_HAIRLINE_CONTACT = 0.3
# ...and is drawn in the same ink. Light grid lines are left to the opening.
MIN_ATTACHED_HAIRLINE_DARKNESS = 0.6
# A tick drawn across a bold axis leaves two short thin halves: kept when they
# are slender and as dark as the bold ink.
MIN_ATTACHED_TICK_SPAN_PX = 5
MIN_ATTACHED_TICK_ELONGATION = 2.5
MIN_ATTACHED_TICK_DARKNESS = 0.6
# Per-pixel darkness, relative to the bold ink, for the unthinned tick evidence.
MIN_TICK_EVIDENCE_DARKNESS = 0.8
# Clean line art (a rendered or digital drawing) thresholds into strokes and
# glyphs with almost no speck-sized components; paper texture and sensor noise
# threshold into thousands of them.
SPECK_AREA_PX = 6
MAX_CLEAN_SPECK_SHARE = 0.1
# A drawing with only a few strokes and glyphs may still keep a JPEG speck or two.
CLEAN_SPECK_ALLOWANCE = 2
# On clean line art the opening only thins strokes. A stroke two or three pixels
# wide (a small or downscaled drawing) is not thinned but cut into a row of dots;
# such a component is kept unthinned.
MIN_SHATTERED_PIECES = 6


def _hairline_components(thresholded: np.ndarray, opened: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    kept = np.bincount(labels[opened > 0], minlength=count)
    hairline = (
        (kept < area * THIN_STROKE_KEPT_FRACTION)
        & (area >= MIN_THIN_COMPONENT_AREA)
        & (span >= MIN_THIN_COMPONENT_SPAN_PX)
    )
    hairline[0] = False
    return np.where(hairline[labels], 255, 0).astype(np.uint8)


def _attached_hairlines(grayscale: np.ndarray, thresholded: np.ndarray, opened: np.ndarray) -> np.ndarray:
    if not opened.any():
        return np.zeros_like(opened)
    residual = cv2.bitwise_and(thresholded, cv2.bitwise_not(opened))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    darkness = 255.0 - grayscale.astype(np.float64)
    bold_darkness = float(np.median(darkness[opened > 0]))
    ink = np.bincount(labels.ravel(), weights=darkness.ravel(), minlength=count) / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
    area = stats[:, cv2.CC_STAT_AREA]
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    touching = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
    contact = np.bincount(labels[touching & (residual > 0)], minlength=count)
    narrow = np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    tick_half = (
        (span >= MIN_ATTACHED_TICK_SPAN_PX)
        & (span >= MIN_ATTACHED_TICK_ELONGATION * narrow)
        & (contact > 0)
        & (ink >= bold_darkness * MIN_ATTACHED_TICK_DARKNESS)
    )
    keep = (
        ((span >= MIN_THIN_COMPONENT_SPAN_PX) | tick_half)
        & (area <= (stats[:, cv2.CC_STAT_WIDTH] + stats[:, cv2.CC_STAT_HEIGHT]) * MAX_ATTACHED_HAIRLINE_WIDTH_PX)
        & (contact < area * MAX_ATTACHED_HAIRLINE_CONTACT)
        & (ink >= bold_darkness * MIN_ATTACHED_HAIRLINE_DARKNESS)
    )
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _shattered_strokes(thresholded: np.ndarray, opened: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    pieces_count, pieces = cv2.connectedComponents(opened, connectivity=8)
    inked = opened > 0
    pairs = np.unique(labels[inked].astype(np.int64) * pieces_count + pieces[inked])
    fragments = np.bincount(pairs // pieces_count, minlength=count)
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    shattered = (fragments >= MIN_SHATTERED_PIECES) & (span >= MIN_THIN_COMPONENT_SPAN_PX)
    shattered[0] = False
    return np.where(shattered[labels], 255, 0).astype(np.uint8)


def is_clean_line_art(grayscale: np.ndarray, edge_sensitivity: int = 50) -> bool:
    """True when thresholding leaves strokes and glyphs but almost no specks."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(_threshold(grayscale, edge_sensitivity), connectivity=8)
    if count <= 1:
        return False
    specks = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < SPECK_AREA_PX))
    return specks <= max(CLEAN_SPECK_ALLOWANCE, MAX_CLEAN_SPECK_SHARE * (count - 1))


def _threshold(grayscale: np.ndarray, edge_sensitivity: int) -> np.ndarray:
    threshold_offset = max(2, min(16, 16 - round(edge_sensitivity * 0.14)))
    return cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, threshold_offset)


def tick_ink_mask(image: np.ndarray, edge_sensitivity: int = 50) -> np.ndarray | None:
    """Unthinned ink as dark as the bold strokes: evidence for thin axis ticks.

    The stroke mask's opening and erosion cut a thin tick into short pieces
    detached from its axis. Light grid lines and paper texture are excluded by
    darkness. None on dark paper, where the thin threshold response is noise.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.median(grayscale) < MIN_PAPER_LEVEL:
        return None
    thresholded = _threshold(grayscale, edge_sensitivity)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    if not opened.any():
        return None
    darkness = 255.0 - grayscale.astype(np.float64)
    bold_darkness = float(np.median(darkness[opened > 0]))
    dark = darkness >= bold_darkness * MIN_TICK_EVIDENCE_DARKNESS
    return np.where((thresholded > 0) & dark, 255, 0).astype(np.uint8)


def preprocess_image(image: np.ndarray, edge_sensitivity: int = 50) -> np.ndarray:
    """Section 6.1: adaptive thresholding then 3x3 opening and erosion.

    Hairline strokes that the opening would erase, and on clean line art
    strokes it would cut into dots, are kept unthinned.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = _threshold(grayscale, edge_sensitivity)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(opened, kernel, iterations=1)
    if np.median(grayscale) < MIN_PAPER_LEVEL:
        return eroded
    hairlines = cv2.bitwise_or(_hairline_components(thresholded, opened), _attached_hairlines(grayscale, thresholded, opened))
    if is_clean_line_art(grayscale, edge_sensitivity):
        hairlines = cv2.bitwise_or(hairlines, _shattered_strokes(thresholded, opened))
    return cv2.bitwise_or(eroded, hairlines)
