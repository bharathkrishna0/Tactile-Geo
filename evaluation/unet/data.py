"""Benchmark images and exact masks as model inputs (same enhancement as Model A)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.services.image_enhancement import enhance_copy

DATASET = Path(__file__).resolve().parents[1] / "dataset"


def split_ids(split: str, dataset: Path = DATASET) -> list[str]:
    return (dataset / "splits" / f"{split}.txt").read_text().split()


def to_input(bgr: np.ndarray) -> np.ndarray:
    """Model A's enhanced copy, grayscale, scaled to [0, 1]."""
    gray = cv2.cvtColor(enhance_copy(bgr), cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def load(image_id: str, dataset: Path = DATASET) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = json.loads((dataset / "annotations" / f"{image_id}.json").read_text())
    bgr = cv2.imread(str(dataset / a["file"]))
    mask = (cv2.imread(str(dataset / a["mask_file"]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
    return bgr, to_input(bgr), mask
