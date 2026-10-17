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
# Of a curve's drawn rim, this share must carry ink; handwriting and graph curves that
# a circle happens to touch now and then leave most of it bare.
MIN_ARC_INKED_FRACTION = 0.7
# A traced outline whose vertices lie this many bands from a curve's rim is that curve.
TRACED_OUTLINE_BANDS = 2.0
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


def _drawn_rim(ink: np.ndarray, cx: float, cy: float, r: float, band: float) -> tuple[float, float, float]:
    """``(start, sweep, inked fraction)`` of the drawn part of a rim; a whole rim sweeps 360.

    Start and sweep are in degrees; the inked fraction is the share of that
    sweep with ink on the rim before short gaps are bridged.
    """
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
        return 0.0, 0.0, 0.0
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
        return 0.0, 360.0, float(covered.mean())
    # The arc is the rim outside the longest undrawn run.
    longest, longest_end, run = 0, 0, 0
    for step in range(2 * bins):
        index = step % bins
        run = 0 if filled[index] else run + 1
        if run > longest:
            longest, longest_end = min(run, bins), index
    arc_start = (longest_end + 1) % bins
    span = bins - longest
    inked = sum(bool(covered[(arc_start + step) % bins]) for step in range(span))
    return arc_start * BIN_DEG, span * BIN_DEG, inked / max(span, 1)


def _traced_outline(shapes: list[dict], cx: float, cy: float, r: float, band: float) -> dict | None:
    """The closed outline or ellipse that ``extract_shapes`` traced along this circle, if any."""
    tolerance = TRACED_OUTLINE_BANDS * band
    for shape in shapes:
        if shape["type"] == "ellipse":
            a, b = shape["semi_axes"]
            if math.dist(shape["center"], (cx, cy)) <= tolerance and max(abs(a - r), abs(b - r)) <= tolerance:
                return shape
        elif shape["type"] == "contour" and len(shape.get("points") or []) >= 3:
            points = np.array(shape["points"], float)
            if (np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r) <= tolerance).mean() >= COVERED_FRACTION:
                if _sweep_deg(points, cx, cy) >= FULL_CIRCLE_DEG - 60:
                    return shape
    return None


def _outline_error(xy: np.ndarray, points) -> float:
    """Mean distance from ``xy`` to the closed outline through ``points``, each edge refitted.

    Every pixel goes to its nearest edge and every edge is refitted as a line
    through its pixels, so a traced outline offset from the stroke centre is
    judged on its shape alone, as the circle it competes with is.
    """
    a = np.array(points, float)
    ab = np.roll(a, -1, axis=0) - a
    t = np.clip(((xy[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0.0, 1.0)
    gap = np.linalg.norm(xy[:, None, :] - (a + t[..., None] * ab), axis=-1)
    edge = gap.argmin(axis=1)
    distance = gap[np.arange(len(xy)), edge]
    for index in np.unique(edge):
        mine = edge == index
        if mine.sum() < 3:
            continue
        centred = xy[mine] - xy[mine].mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        distance[mine] = np.abs(centred @ normal)
    return float(distance.mean()) if len(distance) else float("inf")


# A traced outline needs this many vertices before it can be a polygonised circle.
MIN_ROUND_OUTLINE_VERTICES = 8


def _rounded_outlines(ink: np.ndarray, shapes: list[dict], band: float, limit: float) -> list[dict]:
    """``shapes`` with polygonised circles replaced by the circle their ink draws.

    A large circle traced as a closed contour keeps only its corners; ink between
    them bows away from the chords and would be left over. A regular polygon has
    its corners on a circle too, but its ink runs along the chords, so the
    outline fits it at least as well as any circle and it stays a polygon.
    """
    ink_xy = np.c_[np.nonzero(ink)[::-1]]
    out = []
    for shape in shapes:
        points = shape.get("points") or []
        if shape["type"] != "contour" or len(points) < MIN_ROUND_OUTLINE_VERTICES:
            out.append(shape)
            continue
        cx, cy, r, _ = fit_circle(np.array(points, float))
        width = max(band, 0.04 * r)
        near = ink_xy[np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r) <= width]
        if r < MIN_CURVE_RADIUS_PX or len(near) < MIN_PIECE_PIXELS:
            out.append(shape)
            continue
        cx, cy, r, error = fit_circle(near)
        _, sweep, inked = _drawn_rim(ink, cx, cy, r, band)
        if (
            error > limit or sweep < 360.0 or inked < MIN_ARC_INKED_FRACTION
            or _outline_error(near, points) <= error
        ):
            out.append(shape)
            continue
        out.append({
            "type": "circle", "center": (round(cx), round(cy)), "radius": round(r, 1),
            "points": _rim_points(cx, cy, r, 0.0, 360.0)[:-1],
            "fit_error_px": round(error, 2),
        })
    return out


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
# Refits on the stroke band alone, so a stroke that touches the rim (a tangent,
# a chord's end) stops pulling the circle once the rim is found.
TIGHTEN_STEPS = 3
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
    for _ in range(TIGHTEN_STEPS):
        distance = np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r)
        near = ink_xy[distance <= band]
        if len(near) < MIN_PIECE_PIXELS:
            return None
        cx, cy, r, _ = fit_circle(near)
    distance = np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r)
    on_rim = distance <= band
    error = float(np.median(distance[on_rim])) if on_rim.any() else float("inf")
    seed_distance = np.abs(np.hypot(seed_xy[:, 0] - cx, seed_xy[:, 1] - cy) - r)
    if error > limit or (seed_distance <= band).mean() < SEED_KEPT_FRACTION:
        return None
    return cx, cy, r, error


def reconcile_angle_arcs(angle_arcs: list[dict], shapes: list[dict], stroke_half: float) -> tuple[list[dict], list[dict]]:
    """Keep one reading of each arc that is both an angle mark and a recovered curve on its vertex.

    A curve sweeping no further than the angle (plus ``MIN_ARC_SWEEP_DEG``) is the
    angle mark itself, so the curve is dropped. A curve that runs on well past the
    angle, such as a circle centred on the vertex, is real geometry and the
    angle mark is the part of it between the arms, so the angle mark is dropped.
    """
    band = 2 * stroke_half + 3
    kept_arcs: list[dict] = []
    curves_on_marks: list[dict] = []
    for arc in angle_arcs:
        curve = next(
            (
                shape for shape in shapes
                if shape["type"] in ("circle", "arc")
                and math.dist(arc["vertex"], shape["center"]) <= band
                and abs(arc["radius"] - shape["radius"]) <= band
            ),
            None,
        )
        sweep = None if curve is None else (360.0 if curve["type"] == "circle" else curve["sweep_deg"])
        if sweep is not None and sweep > arc["degrees"] + MIN_ARC_SWEEP_DEG:
            continue
        if curve is not None:
            curves_on_marks.append(curve)
        kept_arcs.append(arc)
    return kept_arcs, [shape for shape in shapes if not any(shape is curve for curve in curves_on_marks)]


def recover_curves(binary: np.ndarray, shapes: list[dict], stroke_half: float, text_boxes=(), angle_arcs=()) -> list[dict]:
    """``shapes`` plus the circles and arcs among the ink they leave unexplained.

    Straight lines that Hough fitted along a recovered curve are dropped: the
    curve explains them.
    """
    ink = binary > 0
    limit = _fit_error_limit(stroke_half)
    band = stroke_half + 1.5
    shapes = _rounded_outlines(ink, shapes, band, limit)
    residual = ink & (explained_ink(binary.shape, shapes, stroke_half, text_boxes, angle_arcs) == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), 8)
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
    replaced: list[dict] = []
    for cx, cy, r, error in candidates:
        traced = _traced_outline(shapes, cx, cy, r, band)
        if traced is not None and traced["type"] == "ellipse":
            continue
        start, sweep, inked = _drawn_rim(ink, cx, cy, r, band)
        if inked < MIN_ARC_INKED_FRACTION:
            continue
        if traced is not None:
            rim = ink_xy[np.abs(np.hypot(ink_xy[:, 0] - cx, ink_xy[:, 1] - cy) - r) <= band]
            if sweep < 360.0 or _outline_error(rim, traced["points"]) <= error:
                continue
            replaced.append(traced)
        center = (round(cx), round(cy))
        if sweep >= 360.0:
            curves.append({
                "type": "circle", "center": center, "radius": round(r, 1),
                "points": _rim_points(cx, cy, r, 0.0, 360.0)[:-1],
                "fit_error_px": round(error, 2),
            })
            continue
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
        if not any(shape is outline for outline in replaced)
        and not (shape["type"] == "line"
                and any(_covered_by_curve(shape["points"], curve, band + 1.0) for curve in curves))
    ]
    return kept + curves
