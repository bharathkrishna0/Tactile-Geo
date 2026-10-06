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
    keep = (
        (span >= MIN_THIN_COMPONENT_SPAN_PX)
        & (area <= (stats[:, cv2.CC_STAT_WIDTH] + stats[:, cv2.CC_STAT_HEIGHT]) * MAX_ATTACHED_HAIRLINE_WIDTH_PX)
        & (contact < area * MAX_ATTACHED_HAIRLINE_CONTACT)
        & (ink >= bold_darkness * MIN_ATTACHED_HAIRLINE_DARKNESS)
    )
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def preprocess_image(image: np.ndarray, edge_sensitivity: int = 50) -> np.ndarray:
    """Section 6.1: adaptive thresholding then 3x3 opening and erosion.

    Hairline strokes that the opening would erase are kept unthinned.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_offset = max(2, min(16, 16 - round(edge_sensitivity * 0.14)))
    thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, threshold_offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(opened, kernel, iterations=1)
    if np.median(grayscale) < MIN_PAPER_LEVEL:
        return eroded
    hairlines = cv2.bitwise_or(_hairline_components(thresholded, opened), _attached_hairlines(grayscale, thresholded, opened))
    return cv2.bitwise_or(eroded, hairlines)
