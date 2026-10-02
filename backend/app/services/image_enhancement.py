"""Image enhancement for processing.

Creates an enhanced *processing copy* of the uploaded image so OpenCV/EasyOCR
see better input. The user's original image is NEVER modified; enhancement
only affects the internal processing copy used by the CV pipeline. The UI
continues to show the original.
"""
from __future__ import annotations

import cv2
import numpy as np

# CLAHE (adaptive) contrast limit and tile grid size.
_CLIP_LIMIT = 2.0
_TILE_GRID_SIZE = (8, 8)

# Sharpen kernel (unsharp-mask style).
_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

# Gaussian noise threshold (from image_quality._estimate_noise) beyond which we sharpen.
_NOISE_BLUR = 3

# Below this median luminance the page itself is dark, so the ink may be the
# lighter tone (chalkboard, inverted scan, underexposed light-on-dark print).
_DARK_PAGE_MEDIAN = 128


def enhance_copy(image: np.ndarray) -> np.ndarray:
    """Produce an improved copy of ``image`` for CV processing.

    Steps (each applied to a working copy, original untouched):
      1. Adaptive contrast (CLAHE) on the luminance channel.
      2. Conditional mild sharpening to restore edge definition.
      3. Polarity normalisation so ink is always darker than the page.
    Returns a BGR image; the caller may convert/duplicate as needed.
    """
    working = image.copy()
    if len(working.shape) == 2:
        working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLIP_LIMIT, tileGridSize=_TILE_GRID_SIZE)
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    noise = _estimate_noise(gray)
    if noise >= _NOISE_BLUR:
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    return normalize_polarity(enhanced)


def has_light_ink_on_dark_page(gray: np.ndarray) -> bool:
    """True when the drawing is lighter than its background.

    The thresholding stage assumes dark ink on light paper; on a dark page the
    ink sits in the bright tail of the histogram rather than the dark one.
    """
    if gray.size == 0:
        return False
    low, median, high = np.percentile(gray, [1, 50, 99])
    return bool(median < _DARK_PAGE_MEDIAN and (high - median) > (median - low))


def normalize_polarity(image: np.ndarray) -> np.ndarray:
    """Invert light-on-dark drawings so downstream stages see dark ink on light paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if has_light_ink_on_dark_page(gray):
        return cv2.bitwise_not(image)
    return image


def _estimate_noise(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    cropped = gray[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float64)
    laplacian = cv2.Laplacian(cropped, cv2.CV_64F)
    return float(np.std(laplacian))


def ink_contrast_copy(image: np.ndarray) -> np.ndarray:
    """Copy where coloured ink is as dark as black ink, for text recognition.

    Taking the darkest channel turns light-blue or red textbook print into dark
    strokes on white while leaving black-on-white input unchanged.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image.min(axis=2), cv2.COLOR_GRAY2BGR)
