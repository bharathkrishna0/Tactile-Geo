from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityIssue:
    check: str
    severity: str
    message: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class QualityReport:
    passes_gate: bool
    issues: list[QualityIssue] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0


_MIN_SHORT_SIDE = 300
_BLUR_LOW_THRESHOLD = 50
_BLUR_ERROR_THRESHOLD = 20
_BRIGHTNESS_LOW = 30
_BRIGHTNESS_HIGH = 225
_CONTRAST_MIN_STDDEV = 25
_NOISE_ESTIMATE_THRESHOLD = 30.0
_MAX_LONG_SIDE = 4000
_SKEW_THRESHOLD_DEGREES = 5.0


def assess_image_quality(image: np.ndarray) -> QualityReport:
    height, width = image.shape[:2]
    issues: list[QualityIssue] = []

    short_side = min(height, width)
    if short_side < _MIN_SHORT_SIDE:
        issues.append(QualityIssue(
            check="resolution",
            severity="warning",
            message=f"Image is small ({width}x{height}). Larger images produce better results.",
            value=float(short_side),
            threshold=float(_MIN_SHORT_SIDE),
        ))

    long_side = max(height, width)
    if long_side > _MAX_LONG_SIDE:
        issues.append(QualityIssue(
            check="resolution",
            severity="info",
            message=f"Image is very large ({width}x{height}). Processing may be slower.",
            value=float(long_side),
            threshold=float(_MAX_LONG_SIDE),
        ))

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_score = cv2.Laplacian(grayscale, cv2.CV_64F).var()
    if blur_score < _BLUR_ERROR_THRESHOLD:
        issues.append(QualityIssue(
            check="blur",
            severity="error",
            message="Image is extremely blurry. Try a clearer photo.",
            value=round(blur_score, 2),
            threshold=float(_BLUR_ERROR_THRESHOLD),
        ))
    elif blur_score < _BLUR_LOW_THRESHOLD:
        issues.append(QualityIssue(
            check="blur",
            severity="warning",
            message="Image appears somewhat blurry. Results may be less accurate.",
            value=round(blur_score, 2),
            threshold=float(_BLUR_LOW_THRESHOLD),
        ))

    mean_brightness = float(np.mean(grayscale))
    if mean_brightness < _BRIGHTNESS_LOW:
        issues.append(QualityIssue(
            check="brightness",
            severity="error",
            message="Image is too dark to process reliably.",
            value=round(mean_brightness, 1),
            threshold=float(_BRIGHTNESS_LOW),
        ))
    elif mean_brightness > _BRIGHTNESS_HIGH:
        issues.append(QualityIssue(
            check="brightness",
            severity="warning",
            message="Image is very bright. Some details may be washed out.",
            value=round(mean_brightness, 1),
            threshold=float(_BRIGHTNESS_HIGH),
        ))

    contrast_stddev = float(np.std(grayscale))
    if contrast_stddev < _CONTRAST_MIN_STDDEV:
        issues.append(QualityIssue(
            check="contrast",
            severity="warning",
            message="Image has low contrast. Diagram lines may be hard to distinguish.",
            value=round(contrast_stddev, 1),
            threshold=float(_CONTRAST_MIN_STDDEV),
        ))

    noise_estimate = _estimate_noise(grayscale)
    if noise_estimate > _NOISE_ESTIMATE_THRESHOLD:
        issues.append(QualityIssue(
            check="noise",
            severity="warning",
            message="Image appears noisy. Results may contain artifacts.",
            value=round(noise_estimate, 2),
            threshold=_NOISE_ESTIMATE_THRESHOLD,
        ))

    skew_angle = _estimate_skew(grayscale)
    if abs(skew_angle) > _SKEW_THRESHOLD_DEGREES:
        issues.append(QualityIssue(
            check="skew",
            severity="warning",
            message=f"Image is rotated approximately {skew_angle:.1f} degrees.",
            value=round(skew_angle, 1),
            threshold=_SKEW_THRESHOLD_DEGREES,
        ))

    passes_gate = not any(issue.severity == "error" for issue in issues)
    return QualityReport(
        passes_gate=passes_gate,
        issues=issues,
        image_width=width,
        image_height=height,
    )


def _estimate_noise(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    cropped = gray[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float64)
    laplacian = cv2.Laplacian(cropped, cv2.CV_64F)
    return float(np.std(laplacian))


def _estimate_skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 3:
        return 0.0
    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    median_angle = sorted(angles)[len(angles) // 2]
    return median_angle
