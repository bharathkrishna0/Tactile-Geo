"""Competition demo mode: process bundled sample worksheets end-to-end.

This lets a reviewer see the full upload -> analysis -> tactile workflow
without needing a physical worksheet on hand. Each sample is a real image
processed by the actual pipeline (no fabricated results).

Samples live in app/demo_samples/ and are copied from the test fixtures so
the demo output is reproducible.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

DEMO_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "demo_samples"

# id -> human-facing description shown in the demo picker.
DEMO_SAMPLES: dict[str, dict] = {
    "triangle": {
        "filename": "triangle_worksheet.png",
        "alt": "A clear triangle worksheet with labeled vertices.",
    },
    "circle": {
        "filename": "circle_worksheet.png",
        "alt": "A clear circle worksheet.",
    },
    "labelled_triangle": {
        "filename": "labelled_triangle_worksheet.png",
        "alt": "A labelled triangle worksheet (geometry diagram).",
    },
}


@dataclass
class DemoSample:
    id: str
    filename: str
    alt: str

    @property
    def path(self) -> Path:
        return DEMO_SAMPLE_DIR / self.filename

    @property
    def image_data_url(self) -> str:
        mime = "image/png" if self.path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(self.path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def list_samples() -> list[DemoSample]:
    return [DemoSample(sample_id, **meta) for sample_id, meta in DEMO_SAMPLES.items()]


def get_sample(sample_id: str) -> DemoSample | None:
    meta = DEMO_SAMPLES.get(sample_id)
    if meta is None:
        return None
    return DemoSample(sample_id, **meta)
