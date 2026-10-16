"""Circles and circular arcs that touch or cross other strokes.

A circle drawn on its own is a closed contour and is fitted in ``extract_shapes``.
A circle inscribed in a square, or a semicircle standing on a line, shares ink
with those lines, so it has no outline of its own. Here the ink that no traced
shape explains is split into pieces, each piece is fitted with a circle, pieces
on the same circle are joined, and the full stroke mask decides how much of the
rim is drawn: nearly all of it makes a circle, less makes an arc.
"""

import math

import cv2
import numpy as np

# A piece of rim needs this many ink pixels to be fitted on its own.
MIN_PIECE_PIXELS = 40
# Smaller rounds are angle arcs, right-angle squares and letters such as "O".
MIN_CURVE_RADIUS_PX = 15.0
# A fitted piece is straight, not curved, when its radius dwarfs its extent.
MAX_RADIUS_TO_EXTENT = 3.0
# Pieces whose circles agree within this fraction of the radius are one curve.
SAME_CIRCLE_FRACTION = 0.06
SAME_CIRCLE_PX = 3.0
# A piece must turn through this much of its circle to be a curve.
MIN_PIECE_SWEEP_DEG = 20.0
# Drawn rim of a whole curve, and of an arc, in degrees.
MIN_ARC_SWEEP_DEG = 45.0
FULL_CIRCLE_DEG = 330.0
# Uncovered runs of rim up to this long are where other strokes cross or touch it.
GAP_BRIDGE_DEG = 12.0
BIN_DEG = 2
LINE_SAMPLES = 16
COVERED_FRACTION = 0.8


def _fit_error_limit(stroke_half: float) -> float:
    # Pixels spread evenly across a stroke sit, on average, a quarter of its
    # width from the centreline.
    return 0.5 * stroke_half + 0.75


def fit_circle(xy: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle through ``xy``: ``(cx, cy, r, mean radial error)``."""
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    a = np.c_[2 * x, 2 * y, np.ones_like(x)]
    solution, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = float(solution[0]), float(solution[1])
    r = math.sqrt(max(float(solution[2]) + cx * cx + cy * cy, 0.0))
    error = float(np.abs(np.hypot(x - cx, y - cy) - r).mean()) if r > 0 else float("inf")
    return cx, cy, r, error


def _sweep_deg(xy: np.ndarray, cx: float, cy: float) -> float:
    angles = np.sort(np.degrees(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)) % 360)
    gaps = np.diff(np.r_[angles, angles[0] + 360])
    return float(360 - gaps.max())


def explained_ink(shape_mask_shape, shapes: list[dict], stroke_half: float, text_boxes=(), angle_arcs=()) -> np.ndarray:
    """Pixels drawn by traced shapes, recognised labels and angle markers."""
    thickness = max(3, int(round(2 * stroke_half + 4)))
    mask = np.zeros(shape_mask_shape[:2], dtype=np.uint8)
    for shape in shapes:
        if shape["type"] == "ellipse":
            cv2.ellipse(mask, (tuple(map(int, shape["center"])), tuple(int(2 * v) for v in shape["semi_axes"]), float(shape.get("angle", 0.0))), 255, thickness)
            continue
        if shape["type"] in ("circle", "arc"):
            points = np.array(shape["points"], np.int32)
            cv2.polylines(mask, [points], shape["type"] == "circle", 255, thickness)
            continue
        points = shape.get("points") or []
        if shape["type"] == "dot" and points:
            cv2.circle(mask, tuple(map(int, points[0])), int(round(float(shape.get("radius", 3.0)) + thickness)), 255, -1)
        elif len(points) >= 2:
            cv2.polylines(mask, [np.array(points, np.int32)], shape["type"] == "contour", 255, thickness)
    for box in text_boxes:
        xs, ys = [int(p[0]) for p in box], [int(p[1]) for p in box]
        cv2.rectangle(mask, (min(xs) - 2, min(ys) - 2), (max(xs) + 2, max(ys) + 2), 255, -1)
    for arc in angle_arcs:
        vertex = tuple(map(int, arc["vertex"]))
        reach = float(arc.get("arc_radius") or 1.5 * float(arc.get("size") or 0))
        cv2.circle(mask, vertex, int(round(reach + thickness)), 255, -1)
    return mask


def _rim_points(cx: float, cy: float, r: float, start_deg: float, sweep_deg: float) -> list[tuple[int, int]]:
    steps = max(8, int(math.ceil(sweep_deg / 4)))
    return [
        (round(cx + r * math.cos(math.radians(start_deg + sweep_deg * i / steps))),
         round(cy + r * math.sin(math.radians(start_deg + sweep_deg * i / steps))))
        for i in range(steps + 1)
    ]


def _drawn_rim(ink: np.ndarray, cx: float, cy: float, r: float, band: float) -> tuple[float, float] | None:
    """``(start, sweep)`` in degrees of the drawn part of a rim, or ``None`` when it is whole."""
    bins = 360 // BIN_DEG
    covered = np.zeros(bins, dtype=bool)
    for index in range(bins):
        theta = math.radians((index + 0.5) * BIN_DEG)
        for offset in np.linspace(-band, band, 5):
            x, y = int(round(cx + (r + offset) * math.cos(theta))), int(round(cy + (r + offset) * math.sin(theta)))
            if 0 <= y < ink.shape[0] and 0 <= x < ink.shape[1] and ink[y, x]:
                covered[index] = True
                break
    if not covered.any():
        return 0.0, 0.0
    # Bridge short gaps: crossing and tangent strokes were cut out of the residual.
    bridge = int(GAP_BRIDGE_DEG // BIN_DEG)
    filled = covered.copy()
    start = int(np.argmax(covered))
    run = 0
    for step in range(1, bins + 1):
        index = (start + step) % bins
        if covered[index]:
            if 0 < run <= bridge:
                for back in range(1, run + 1):
                    filled[(index - back) % bins] = True
            run = 0
        else:
            run += 1
    if filled.sum() * BIN_DEG >= FULL_CIRCLE_DEG:
        return None
    # The arc is the rim outside the longest undrawn run.
    longest, longest_end, run = 0, 0, 0
    for step in range(2 * bins):
        index = step % bins
        run = 0 if filled[index] else run + 1
        if run > longest:
            longest, longest_end = min(run, bins), index
    arc_start = (longest_end + 1) % bins
    return arc_start * BIN_DEG, (bins - longest) * BIN_DEG


def _covered_by_curve(points, curve: dict, band: float) -> bool:
    (ax, ay), (bx, by) = points
    cx, cy = curve["center"]
    r = curve["radius"]
    near = 0
    for i in range(LINE_SAMPLES + 1):
        t = i / LINE_SAMPLES
        x, y = ax + t * (bx - ax), ay + t * (by - ay)
        if abs(math.hypot(x - cx, y - cy) - r) > band:
            continue
        if curve["type"] == "arc":
            angle = (math.degrees(math.atan2(y - cy, x - cx)) - curve["start_deg"]) % 360
            if angle > curve["sweep_deg"] + 2 * BIN_DEG:
                continue
        near += 1
    return near >= COVERED_FRACTION * (LINE_SAMPLES + 1)


# Each round grows the seed along its rim by this much on either side, in degrees.
GROW_STEPS_DEG = (10.0, 25.0, 50.0, 90.0, 180.0)
# Of a seed piece's pixels, this share must stay on the refined circle.
SEED_KEPT_FRACTION = 0.8


def _angular_span(xy: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """``(start, sweep)`` in degrees of the smallest arc holding every point."""
    angles = np.sort(np.degrees(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)) % 360)
    gaps = np.diff(np.r_[angles, angles[0] + 360])
    widest = int(np.argmax(gaps))
    start = float(angles[(widest + 1) % len(angles)])
    return start, float(360 - gaps[widest])


def _refine(ink_xy: np.ndarray, seed, band: float, limit: float):
    """Grow a seed piece along its rim through all ink, refitting the circle each step.

    A short piece of a large circle fits its radius poorly; the rest of the rim,
    including ink that straight-line tracing claimed, pins it down. Growing a
    little at a time keeps crossing strokes and other curves from pulling the fit.
    """
    cx, cy, r, seed_xy = seed
    start, sweep = _angular_span(seed_xy, cx, cy)
    for grow in GROW_STEPS_DEG:
        width = max(band, 0.04 * r)
        distance = np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r)
        angle = (np.degrees(np.arctan2(ink_xy[:, 1] - cy, ink_xy[:, 0] - cx)) - (start - grow)) % 360
        near = ink_xy[(distance <= width) & (angle <= min(360.0, sweep + 2 * grow))]
        if len(near) < MIN_PIECE_PIXELS:
            return None
        cx, cy, r, _ = fit_circle(near)
        if r < MIN_CURVE_RADIUS_PX:
            return None
        start, sweep = _angular_span(seed_xy, cx, cy)
    distance = np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r)
    on_rim = distance <= band
    error = float(np.median(distance[on_rim])) if on_rim.any() else float("inf")
    seed_distance = np.abs(np.hypot(seed_xy[:, 0] - cx, seed_xy[:, 1] - cy) - r)
    if error > limit or (seed_distance <= band).mean() < SEED_KEPT_FRACTION:
        return None
    return cx, cy, r, error


def recover_curves(binary: np.ndarray, shapes: list[dict], stroke_half: float, text_boxes=(), angle_arcs=()) -> list[dict]:
    """``shapes`` plus the circles and arcs among the ink they leave unexplained.

    Straight lines that Hough fitted along a recovered curve are dropped: the
    curve explains them.
    """
    ink = binary > 0
    residual = ink & (explained_ink(binary.shape, shapes, stroke_half, text_boxes, angle_arcs) == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
    limit = _fit_error_limit(stroke_half)
    pieces: list[tuple[float, float, float, np.ndarray]] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_PIECE_PIXELS:
            continue
        ys, xs = np.nonzero(labels == index)
        xy = np.c_[xs, ys]
        cx, cy, r, error = fit_circle(xy)
        extent = max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT])
        if (
            error <= limit
            and r >= MIN_CURVE_RADIUS_PX
            and r <= MAX_RADIUS_TO_EXTENT * extent
            and _sweep_deg(xy, cx, cy) >= MIN_PIECE_SWEEP_DEG
        ):
            pieces.append((cx, cy, r, xy))

    band = stroke_half + 1.5
    ink_xy = np.c_[np.nonzero(ink)[::-1]]
    candidates: list[tuple[float, float, float, float]] = []
    for seed in sorted(pieces, key=lambda p: -len(p[3])):
        seed_xy = seed[3]
        if any(
            (np.abs(np.hypot(seed_xy[:, 0] - c[0], seed_xy[:, 1] - c[1]) - c[2]) <= band).mean() >= SEED_KEPT_FRACTION
            for c in candidates
        ):
            continue
        refined = _refine(ink_xy, seed, band, limit)
        if refined is None:
            continue
        cx, cy, r, _ = refined
        tolerance = SAME_CIRCLE_FRACTION * r + SAME_CIRCLE_PX
        if any(math.dist((cx, cy), c[:2]) <= tolerance and abs(r - c[2]) <= tolerance for c in candidates):
            continue
        candidates.append(refined)

    curves: list[dict] = []
    for cx, cy, r, error in candidates:
        rim = _drawn_rim(ink, cx, cy, r, band)
        center = (round(cx), round(cy))
        if rim is None:
            curves.append({
                "type": "circle", "center": center, "radius": round(r, 1),
                "points": _rim_points(cx, cy, r, 0.0, 360.0)[:-1],
                "fit_error_px": round(error, 2),
            })
            continue
        start, sweep = rim
        if sweep < MIN_ARC_SWEEP_DEG:
            continue
        curves.append({
            "type": "arc", "center": center, "radius": round(r, 1),
            "start_deg": round(start, 1), "sweep_deg": round(sweep, 1),
            "points": _rim_points(cx, cy, r, start, sweep),
            "fit_error_px": round(error, 2),
        })
    if not curves:
        return shapes
    kept = [
        shape for shape in shapes
        if not (shape["type"] == "line" and shape.get("role") != "tick"
                and any(_covered_by_curve(shape["points"], curve, band + 1.0) for curve in curves))
    ]
    return kept + curves
