"""Prepare an uploaded image for the Model B vision model.

Model A already owns decoding, decode errors, and EXIF behaviour, so this
module takes raw bytes and reuses the same conventions rather than introducing
a second opinion.

Why letterbox instead of a plain resize: resizing a tall worksheet to a fixed
width distorts the aspect ratio, and a circle then genuinely looks like an
ellipse. A geometry model asked to classify shapes should never be handed
geometrically distorted input. Letterboxing preserves the ratio exactly and
pads the remainder.

The cost of padding is that normalized coordinates are relative to the padded
frame, so mapping them back to the original image is a subtract-then-scale, not
a scale. `PreparedImage.project_bbox` owns that inverse transform, and keeping
it in one place is what stops the off-by-a-padding-strip bug.

EXIF orientation is intentionally NOT applied. Model A does not apply it either,
so both models see the same decode and stay mutually consistent. Fixing it in
one place only would silently desynchronise Model A from Model B; it belongs
in the shared decode path for both, as a separate change.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Vision models downsample large images aggressively, and a long edge at or below
# this is not downscaled internally, so `media_resolution` and our own
# preparation agree instead of fighting each other.
LONG_EDGE_TARGET = 1568
# Below this, small worksheet text becomes unreliable. Upscaling a small image
# adds no information but does let the model's own resampler see the glyphs at a
# scale it can resolve.
SHORT_EDGE_UPSCALE = 800
MAX_PIXELS = 4_000_000
MEDIA_RESOLUTION = "high"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PreparedImage:
    """A letterboxed, PNG-encoded image plus the geometry to invert it.

    `scale_x`/`scale_y` multiply PREPARED content pixels to ORIGINAL pixels. They
    are computed from the actual rounded content size rather than the requested
    scale factor, so the inverse transform is exact instead of off by a pixel
    per edge.
    """

    png_bytes: bytes
    width: int
    height: int
    original_width: int
    original_height: int
    content_width: int
    content_height: int
    pad_x: int
    pad_y: int
    scale_x: float
    scale_y: float
    media_resolution: str = MEDIA_RESOLUTION

    @property
    def was_resized(self) -> bool:
        return self.content_width != self.original_width or self.pad_x or self.pad_y

    def project_bbox(
        self, bbox_norm: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        """Map a normalized Model B bbox onto the original image.

        Args:
            bbox_norm: (x_min, y_min, x_max, y_max) in [0, 1] relative to the
                PREPARED frame, including padding.

        Returns:
            (x, y, width, height) in original-image pixels, clamped to the
            image. A bbox falling entirely inside the padding collapses to a
            zero-area rect at the nearest edge rather than a negative one.
        """
        x_min, y_min, x_max, y_max = bbox_norm
        left = (x_min * self.width - self.pad_x) * self.scale_x
        top = (y_min * self.height - self.pad_y) * self.scale_y
        right = (x_max * self.width - self.pad_x) * self.scale_x
        bottom = (y_max * self.height - self.pad_y) * self.scale_y

        left_c = _clamp(left, 0.0, float(self.original_width))
        right_c = _clamp(right, 0.0, float(self.original_width))
        top_c = _clamp(top, 0.0, float(self.original_height))
        bottom_c = _clamp(bottom, 0.0, float(self.original_height))

        return (
            int(round(left_c)),
            int(round(top_c)),
            int(round(max(0.0, right_c - left_c))),
            int(round(max(0.0, bottom_c - top_c))),
        )


def _target_scale(width: int, height: int) -> float:
    """Scale factor to apply to the image content, before padding."""
    long_edge = max(width, height)
    short_edge = min(width, height)
    scale = 1.0

    if long_edge > LONG_EDGE_TARGET:
        scale = LONG_EDGE_TARGET / long_edge
    elif short_edge < SHORT_EDGE_UPSCALE:
        # Upscale small input, but never past the long-edge target: upscaling
        # beyond it just spends tokens on interpolation.
        scale = min(SHORT_EDGE_UPSCALE / short_edge, LONG_EDGE_TARGET / long_edge)

    if width * height * scale * scale > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5

    return scale


def prepare_image(image_bytes: bytes) -> PreparedImage:
    """Decode, letterbox to a square canvas, and re-encode as PNG.

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Model B could not decode the uploaded image.")

    original_height, original_width = decoded.shape[:2]
    if original_width < 1 or original_height < 1:
        raise ValueError("Uploaded image has zero extent.")

    scale = _target_scale(original_width, original_height)
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))

    if (content_width, content_height) != (original_width, original_height):
        # INTER_AREA for downscale is a proper box filter; INTER_CUBIC for
        # upscale avoids the blockiness of nearest-neighbour on thin strokes.
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        content = cv2.resize(
            decoded, (content_width, content_height), interpolation=interpolation
        )
    else:
        content = decoded

    canvas_size = max(content_width, content_height)
    channel = content.shape[2] if content.ndim == 3 else 1
    canvas = np.full((canvas_size, canvas_size, channel), 255, dtype=np.uint8)
    pad_x = (canvas_size - content_width) // 2
    pad_y = (canvas_size - content_height) // 2
    canvas[pad_y : pad_y + content_height, pad_x : pad_x + content_width] = content

    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Model B could not encode the prepared image.")

    return PreparedImage(
        png_bytes=encoded.tobytes(),
        width=canvas_size,
        height=canvas_size,
        original_width=original_width,
        original_height=original_height,
        content_width=content_width,
        content_height=content_height,
        pad_x=pad_x,
        pad_y=pad_y,
        scale_x=original_width / content_width,
        scale_y=original_height / content_height,
    )


def encode_jpeg_preview(image: np.ndarray, max_edge: int = 640) -> bytes:
    """Small JPEG of an image, for logging and the UI thumbnail."""
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise ValueError("Could not encode preview.")
    return encoded.tobytes()
