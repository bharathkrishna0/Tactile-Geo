"""U-Net inference and the Model A preprocessing override used by the runner."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import SmallUNet

THRESHOLD = 0.5


def load_model(weights: Path) -> tuple[SmallUNet, dict]:
    if not weights.exists():
        raise FileNotFoundError(f"U-Net weights not found at {weights}; run python -m evaluation.unet.train")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model = SmallUNet(base=state["base"])
    model.load_state_dict(state["model"])
    model.eval()
    return model, {k: v for k, v in state.items() if k != "model"}


@torch.no_grad()
def predict(model: SmallUNet, gray01: np.ndarray) -> np.ndarray:
    """Probability map, same size as the input. Pads to a multiple of 8."""
    h, w = gray01.shape
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(gray01, ((0, ph), (0, pw)), mode="edge")
    x = torch.from_numpy(padded)[None, None]
    prob = torch.sigmoid(model(x))[0, 0].numpy()
    return prob[:h, :w]


def mask_override(weights: Path, combine: str = "replace"):
    """Return ``(overrides, info)`` for ``evaluation.capture.run_model_a``.

    ``replace``: the U-Net mask is the binary image Model A vectorizes.
    ``intersect``: Model A's own binary, kept only where the (dilated) U-Net
    mask says geometry ink, i.e. the U-Net acts as a text/clutter filter.
    """
    from app.services.image_preprocessing import preprocess_image

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, meta = load_model(weights)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def unet_preprocess(image: np.ndarray, edge_sensitivity: int = 50) -> np.ndarray:
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = (predict(model, gray) >= THRESHOLD).astype(np.uint8) * 255
        unet_preprocess.last_ms = (time.perf_counter() - start) * 1000
        if combine == "replace":
            return mask
        return cv2.bitwise_and(preprocess_image(image, edge_sensitivity), cv2.dilate(mask, kernel))

    info = {"weights": str(weights.name), "combine": combine, "threshold": THRESHOLD, **meta}
    return {"preprocess_image": unet_preprocess}, info
