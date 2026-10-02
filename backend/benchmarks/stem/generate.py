"""Generate the synthetic STEM benchmark set with exact ground truth.

    python benchmarks/stem/generate.py

Every figure is drawn from primitives whose bounding boxes are known exactly,
so detection can be scored without hand annotation. Each base figure is also
rendered as degraded variants (half resolution, phone-photo style, dark page)
because those are the conditions classroom uploads actually arrive in.

Synthetic figures are clean by construction. They measure whether the
pipeline recovers known structure; they do not replace evaluation on real
worksheets, which is why the benchmark also reports the reference images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
W, H = 1000, 800
INK = (0, 0, 0)
STROKE = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Figure:
    def __init__(self, name: str, category: str, description: str) -> None:
        self.name = name
        self.category = category
        self.description = description
        self.image = np.full((H, W, 3), 255, dtype=np.uint8)
        self.shapes: list[dict] = []
        self.labels: list[str] = []

    def _shape(self, kind: str, xs: list[float], ys: list[float], points: list[tuple[int, int]] | None = None) -> None:
        x0, y0 = int(min(xs)), int(min(ys))
        shape: dict = {"type": kind, "bbox": [x0, y0, int(max(xs)) - x0, int(max(ys)) - y0]}
        if points:
            shape["points"] = [list(p) for p in points]
        self.shapes.append(shape)

    def polygon(self, kind: str, points: list[tuple[int, int]]) -> None:
        cv2.polylines(self.image, [np.array(points, np.int32)], True, INK, STROKE, cv2.LINE_AA)
        self._shape(kind, [p[0] for p in points], [p[1] for p in points], points)

    def circle(self, center: tuple[int, int], radius: int) -> None:
        cv2.circle(self.image, center, radius, INK, STROKE, cv2.LINE_AA)
        self._shape("circle", [center[0] - radius, center[0] + radius], [center[1] - radius, center[1] + radius])

    def ellipse(self, center: tuple[int, int], axes: tuple[int, int]) -> None:
        cv2.ellipse(self.image, center, axes, 0, 0, 360, INK, STROKE, cv2.LINE_AA)
        self._shape("ellipse", [center[0] - axes[0], center[0] + axes[0]], [center[1] - axes[1], center[1] + axes[1]])

    def segment(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        cv2.line(self.image, a, b, INK, STROKE, cv2.LINE_AA)
        self._shape("line_segment", [a[0], b[0]], [a[1], b[1]])

    def right_angle_marker(self, corner: tuple[int, int], size: int = 28) -> None:
        x, y = corner
        cv2.polylines(self.image, [np.array([(x, y - size), (x + size, y - size), (x + size, y)], np.int32)], False, INK, 2)

    def label(self, text: str, origin: tuple[int, int]) -> None:
        cv2.putText(self.image, text, origin, FONT, 1.3, INK, 3, cv2.LINE_AA)
        self.labels.append(text)


def regular_polygon(center: tuple[int, int], radius: int, sides: int, phase: float = -math.pi / 2) -> list[tuple[int, int]]:
    return [
        (int(center[0] + radius * math.cos(phase + 2 * math.pi * i / sides)),
         int(center[1] + radius * math.sin(phase + 2 * math.pi * i / sides)))
        for i in range(sides)
    ]


def base_figures() -> list[Figure]:
    figures = []

    f = Figure("triangle_labelled", "geometry", "Scalene triangle with labelled vertices")
    f.polygon("triangle", [(500, 150), (220, 620), (800, 620)])
    f.label("A", (485, 125)); f.label("B", (170, 660)); f.label("C", (815, 660))
    figures.append(f)

    f = Figure("right_triangle", "geometry", "Right triangle with a right-angle marker")
    f.polygon("triangle", [(250, 150), (250, 620), (780, 620)])
    f.right_angle_marker((250, 620))
    f.label("P", (200, 140)); f.label("Q", (200, 670)); f.label("R", (795, 670))
    figures.append(f)

    f = Figure("circle_centre", "geometry", "Circle with a marked radius")
    f.circle((500, 400), 250)
    f.segment((500, 400), (750, 400))
    f.label("O", (460, 390))
    figures.append(f)

    f = Figure("rectangle_abcd", "geometry", "Rectangle with labelled corners")
    f.polygon("rectangle", [(200, 200), (800, 200), (800, 600), (200, 600)])
    f.label("A", (160, 190)); f.label("B", (815, 190)); f.label("C", (815, 650)); f.label("D", (160, 650))
    figures.append(f)

    f = Figure("pentagon", "geometry", "Regular pentagon")
    f.polygon("polygon", regular_polygon((500, 420), 260, 5))
    figures.append(f)

    f = Figure("two_circles", "geometry", "Two separate circles")
    f.circle((300, 400), 170)
    f.circle((720, 400), 140)
    f.label("X", (285, 640)); f.label("Y", (705, 640))
    figures.append(f)

    f = Figure("inclined_plane", "physics", "Inclined plane with a block")
    f.polygon("triangle", [(150, 650), (850, 650), (850, 250)])
    f.polygon("rectangle", [(470, 400), (590, 400), (590, 500), (470, 500)])
    f.label("m", (510, 380))
    figures.append(f)

    f = Figure("lever", "physics", "Lever on a triangular fulcrum")
    f.segment((120, 400), (880, 400))
    f.polygon("triangle", [(500, 410), (420, 600), (580, 600)])
    f.label("F", (130, 370))
    figures.append(f)

    f = Figure("axes_line_graph", "graphs", "Coordinate axes with a straight-line graph")
    f.segment((150, 650), (880, 650))
    f.segment((150, 650), (150, 120))
    f.segment((180, 600), (820, 200))
    f.label("x", (895, 665)); f.label("y", (135, 100))
    figures.append(f)

    f = Figure("bar_chart", "graphs", "Three-bar chart")
    for x0, top in ((220, 400), (440, 250), (660, 500)):
        f.polygon("rectangle", [(x0, top), (x0 + 140, top), (x0 + 140, 650), (x0, 650)])
    figures.append(f)

    f = Figure("benzene_ring", "chemistry", "Hexagonal benzene ring with inner circle")
    f.polygon("polygon", regular_polygon((500, 400), 260, 6, phase=0))
    f.circle((500, 400), 150)
    figures.append(f)

    f = Figure("cell", "biology", "Cell outline with nucleus")
    f.ellipse((500, 400), (360, 240))
    f.circle((560, 380), 90)
    f.label("N", (545, 395))
    figures.append(f)

    return figures


def half_resolution(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (W // 2, H // 2), interpolation=cv2.INTER_AREA)


def phone_photo(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gradient = np.linspace(225, 185, W, dtype=np.float32)[None, :, None]
    paper = np.where(image < 128, 40.0, gradient)
    noisy = paper + rng.normal(0, 8, paper.shape)
    return cv2.GaussianBlur(np.clip(noisy, 0, 255).astype(np.uint8), (5, 5), 0)


def dark_page(image: np.ndarray) -> np.ndarray:
    return np.where(image < 128, 235, 30).astype(np.uint8)


def scaled_shapes(shapes: list[dict], factor: float) -> list[dict]:
    scaled = []
    for shape in shapes:
        copy = {"type": shape["type"], "bbox": [int(v * factor) for v in shape["bbox"]]}
        if "points" in shape:
            copy["points"] = [[int(v * factor) for v in p] for p in shape["points"]]
        scaled.append(copy)
    return scaled


def main() -> None:
    images = OUT / "images"
    images.mkdir(exist_ok=True)
    cases = []
    for index, figure in enumerate(base_figures()):
        variants = [
            ("clean", figure.image, figure.shapes),
            ("halfres", half_resolution(figure.image), scaled_shapes(figure.shapes, 0.5)),
            ("photo", phone_photo(figure.image, index), figure.shapes),
            ("dark", dark_page(figure.image), figure.shapes),
        ]
        for variant, image, shapes in variants:
            filename = f"{figure.name}__{variant}.png"
            cv2.imwrite(str(images / filename), image)
            cases.append(
                {
                    "file": f"images/{filename}",
                    "figure": figure.name,
                    "category": figure.category,
                    "variant": variant,
                    "description": figure.description,
                    "shapes": shapes,
                    "labels": figure.labels,
                }
            )
    (OUT / "ground_truth.json").write_text(json.dumps({"version": 1, "cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
