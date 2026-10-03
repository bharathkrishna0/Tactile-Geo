"""Acquisition degradations: scan, phone photo, skew, blur, contrast, resolution.

Geometric effects are collected into one homography so the page, the
foreground mask and every ground-truth coordinate move identically.
Photometric effects only touch the page, never the mask or the annotation.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

GEOMETRIC = {"scanned", "photographed", "skew", "low_resolution", "strong_rotation"}


def _rotation(w, h, degrees):
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return np.vstack([m, [0, 0, 1]])


def _perspective(w, h, rng, amount):
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-amount, amount, (4, 2)) * [w, h]
    dst = np.float32(src + jitter)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def apply(page: np.ndarray, mask: np.ndarray, conditions: list[str], rng: np.random.Generator):
    """Return (bgr image, mask, homography, out_w, out_h, applied parameters)."""
    h, w = page.shape
    H = np.eye(3)
    params: dict[str, float | str] = {}
    out_w, out_h = w, h
    if "scanned" in conditions:
        deg = float(rng.uniform(-2.0, 2.0))
        H = _rotation(w, h, deg) @ H
        params["scan_rotation_deg"] = round(deg, 2)
    if "skew" in conditions:
        deg = float(rng.choice([-1, 1]) * rng.uniform(3.5, 6.5))
        H = _rotation(w, h, deg) @ H
        params["skew_deg"] = round(deg, 2)
    if "strong_rotation" in conditions:
        deg = float(rng.choice([-1, 1]) * rng.uniform(8, 11))
        H = _rotation(w, h, deg) @ H
        params["rotation_deg"] = round(deg, 2)
    if "photographed" in conditions:
        H = _perspective(w, h, rng, 0.035) @ H
        params["perspective_jitter"] = 0.035
    if "low_resolution" in conditions:
        factor = float(rng.uniform(0.42, 0.55))
        out_w, out_h = int(w * factor), int(h * factor)
        H = np.diag([out_w / w, out_h / h, 1.0]) @ H
        params["scale"] = round(factor, 3)

    img = cv2.warpPerspective(page, H, (out_w, out_h), flags=cv2.INTER_AREA if out_w < w else cv2.INTER_LINEAR,
                              borderValue=255)
    m = cv2.warpPerspective(mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
    img = img.astype(np.float32)

    if "faded" in conditions:
        img = 255 - (255 - img) * 0.45
        params["ink_strength"] = 0.45
    if "low_contrast" in conditions:
        paper, ink = float(rng.uniform(195, 215)), float(rng.uniform(120, 150))
        img = ink + (img / 255.0) * (paper - ink)
        params["contrast_range"] = f"{ink:.0f}-{paper:.0f}"
    if "photographed" in conditions or "shadow" in conditions:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        angle = float(rng.uniform(0, 2 * math.pi))
        grad = (xx * math.cos(angle) + yy * math.sin(angle))
        grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
        light = 1.0 - 0.22 * grad
        if "shadow" in conditions:
            cx, cy = float(rng.uniform(0.2, 0.8)) * out_w, float(rng.uniform(0.2, 0.8)) * out_h
            normal = (math.cos(angle + 1.2), math.sin(angle + 1.2))
            side = ((xx - cx) * normal[0] + (yy - cy) * normal[1]) > 0
            light = light * np.where(side, 0.55, 1.0)
            light = cv2.GaussianBlur(light, (0, 0), 15)
            params["shadow"] = "hard-edged diagonal shadow"
        img = img * light
    if "blur" in conditions:
        sigma = float(rng.uniform(1.4, 2.0))
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        params["blur_sigma"] = round(sigma, 2)
    if "scanned" in conditions or "photographed" in conditions:
        sigma = 0.7 if "scanned" in conditions else 1.0
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        img = img + rng.normal(0, 4.0 if "scanned" in conditions else 6.0, img.shape)
    if "dark_page" in conditions:
        img = 255 - img
        img = 35 + img * (200 / 255.0)
        params["dark_page"] = "chalkboard (light ink on dark)"
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if "colored_ink" in conditions:
        ink = (255 - img).astype(np.float32) / 255.0
        color = np.array([200, 80, 30], np.float32) if rng.integers(0, 2) else np.array([40, 40, 210], np.float32)
        bgr = (255 - ink[..., None] * (255 - color)).astype(np.uint8)
        params["ink_color_bgr"] = str(color.astype(int).tolist())
    if "photographed" in conditions:
        tint = np.array([0.90, 0.96, 1.0])
        bgr = np.clip(bgr * tint, 0, 255).astype(np.uint8)
    quality = None
    if "jpeg_heavy" in conditions:
        quality = int(rng.integers(12, 22))
    elif "photographed" in conditions:
        quality = 70
    elif "scanned" in conditions:
        quality = 80
    if quality:
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        params["jpeg_quality"] = quality
    return bgr, m, H, out_w, out_h, params
