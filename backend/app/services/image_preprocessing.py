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
    return cv2.bitwise_or(eroded, _hairline_components(thresholded, opened))
