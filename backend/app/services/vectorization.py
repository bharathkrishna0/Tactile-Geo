import math

import cv2
import numpy as np

from app.services.segment_geometry import normalize_segments, point_to_segment_distance
from app.services.tactile_rules import TACTILE_RULES

# Half of the simplification tolerance: raw Hough fragments of one drawn edge sit
# within a few pixels of each other, so the merge distance is deliberately tighter
# than the semantic-level merge distance.
COLLINEAR_DISTANCE_TOLERANCE_PX = TACTILE_RULES.collinear_merge_tolerance_px / 2.0
COLLINEAR_ANGLE_TOLERANCE_DEG = 3.0


def _line_segments(binary_image: np.ndarray, edge_sensitivity: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, max(20, 110 - edge_sensitivity), minLineLength=max(20, 90 - edge_sensitivity), maxLineGap=12)
    if lines is None:
        return []
    raw = [((int(x1), int(y1)), (int(x2), int(y2))) for x1, y1, x2, y2 in lines[:, 0]]
    return [
        ((int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))))
        for a, b in normalize_segments(raw, COLLINEAR_ANGLE_TOLERANCE_DEG, COLLINEAR_DISTANCE_TOLERANCE_PX)
    ]


# A drawn closed outline has an inner hole; a hole smaller than this fraction of
# the outer area is a filled blob or a glyph counter, not an outline.
MIN_HOLE_AREA_FRACTION = 0.2
# Mean normalised radial error below which a contour is accepted as an ellipse.
# A regular hexagon scores ~0.06 against its best-fit ellipse, so polygons stay
# polygons instead of gaining an ellipse that is not in the source.
ELLIPSE_FIT_MAX_ERROR = 0.035
# approxPolyDP at 1 % of the perimeter keeps 5-8 corners of a drawn regular
# polygon but needs 9 or more vertices for a drawn circle (an octagon fits an
# ellipse within ELLIPSE_FIT_MAX_ERROR, so vertex count separates the two).
MIN_ROUND_VERTICES = 9
ELLIPSE_MAX_ASPECT = 0.85
ELLIPSE_MIN_AREA = 200
MIN_CONTOUR_AREA = 80
# Distance from a stroke boundary within which a sample still counts as "on" it.
STROKE_BAND_TOLERANCE_PX = 4.0
LINE_SAMPLES = 12
COVERED_FRACTION = 0.8


def _ellipse_from_fit(fit) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return centre, (semi-major, semi-minor) and the major-axis angle in degrees.

    ``cv2.fitEllipse`` reports ``angle`` for its *first* axis, which is not
    necessarily the major one, so the angle turns by 90 degrees when the axes are
    reordered.
    """
    (cx, cy), (first, second), angle = fit
    if first >= second:
        return (cx, cy), (first / 2, second / 2), angle % 180
    return (cx, cy), (second / 2, first / 2), (angle + 90) % 180


def _ellipse_fit_error(contour: np.ndarray, center, semi_axes, angle: float) -> float:
    a, b = semi_axes
    if a <= 0 or b <= 0:
        return float("inf")
    points = contour.reshape(-1, 2).astype(float) - np.array(center, dtype=float)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = points[:, 0] * cos_t + points[:, 1] * sin_t
    v = -points[:, 0] * sin_t + points[:, 1] * cos_t
    radial = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return float(np.mean(np.abs(radial - 1.0)))


def _approx_points(contour: np.ndarray) -> list[tuple[int, int]]:
    approximation = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    return [(int(point[0][0]), int(point[0][1])) for point in approximation]


def _centerline_points(outer: list[tuple[int, int]], inner: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Average matching outer/inner outline vertices to sit on the stroke centre."""
    if not inner or len(inner) != len(outer):
        return outer
    centred = []
    for x, y in outer:
        ix, iy = min(inner, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
        centred.append((int(round((x + ix) / 2)), int(round((y + iy) / 2))))
    return centred


def stroke_half_width(binary_image: np.ndarray) -> float:
    """Median half-width of the drawn strokes, from the distance-transform ridge."""
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    values = distance[ridge]
    return float(np.median(values)) if values.size else 1.0


def _collapse_stroke_edges(segments, band_px: float):
    """Merge Hough lines that trace the two edges of one wide (e.g. blurred) stroke.

    Two segments whose endpoints each lie within one stroke width of the other
    segment are the same drawn line; they are replaced by their mid-line.
    """
    kept: list = []
    for segment in segments:
        for index, existing in enumerate(kept):
            near = all(point_to_segment_distance(p, *existing) <= band_px for p in segment) and all(
                point_to_segment_distance(p, *segment) <= band_px for p in existing
            )
            if not near:
                continue
            a, b = segment
            if math.dist(a, existing[0]) > math.dist(a, existing[1]):
                a, b = b, a
            kept[index] = (
                (round((a[0] + existing[0][0]) / 2), round((a[1] + existing[0][1]) / 2)),
                (round((b[0] + existing[1][0]) / 2), round((b[1] + existing[1][1]) / 2)),
            )
            break
        else:
            kept.append(segment)
    return kept


def _snap_shared_endpoints(segments, tolerance_px: float):
    """Join line ends that meet at one vertex so corners neither gap nor overshoot."""
    endpoints = [point for segment in segments for point in segment]
    clusters: list[list[tuple[int, int]]] = []
    for point in endpoints:
        for cluster in clusters:
            if math.dist(point, cluster[0]) <= tolerance_px:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    snapped = {}
    for cluster in clusters:
        centre = (round(sum(p[0] for p in cluster) / len(cluster)), round(sum(p[1] for p in cluster) / len(cluster)))
        for point in cluster:
            snapped[point] = centre
    return [(snapped[a], snapped[b]) for a, b in segments if snapped[a] != snapped[b]]


def _line_covered_by_outline(segment, outer: np.ndarray, band_px: float) -> bool:
    """True when the segment runs along the outline stroke rather than across it."""
    (x1, y1), (x2, y2) = segment
    samples = [
        (x1 + (x2 - x1) * t / (LINE_SAMPLES - 1), y1 + (y2 - y1) * t / (LINE_SAMPLES - 1))
        for t in range(LINE_SAMPLES)
    ]
    on_outline = sum(
        abs(cv2.pointPolygonTest(outer, (float(x), float(y)), True)) <= band_px for x, y in samples
    )
    return on_outline / LINE_SAMPLES >= COVERED_FRACTION


def _point_segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


# A hole-less outline whose every point has a partner across one stroke width,
# further along the outline, is the two sides of one open stroke (an open
# polyline, or a polygon whose ring is broken), not a filled shape.
FOLDED_PARTNER_FRACTION = 0.85
FOLDED_PARTNER_SEPARATION = 4.0
# The stroke's two ends are where the outline turns back on itself: points a
# short way either side along the outline lie across one stroke width.
FOLDED_TIP_REACH = 2.0
# Ends this close (in stroke bands) are a closed figure whose ring has a break.
FOLDED_CLOSED_GAP = 4.0
# More sides than this is a curve or scribble, left as an outline.
MAX_FOLDED_SIDES = 5


def _folded_tips(points: np.ndarray, band: float) -> list[int]:
    count = len(points)
    reach = max(2, int(FOLDED_TIP_REACH * band))
    index = np.arange(count)
    across = np.hypot(*(points[(index - reach) % count] - points[(index + reach) % count]).T)
    turning = across <= band
    if turning.all() or not turning.any():
        return []
    first = int(np.argmin(turning))
    tips, run = [], []
    for offset in range(1, count + 1):
        i = (first + offset) % count
        if turning[i]:
            run.append(i)
        elif run:
            tips.append(run[len(run) // 2])
            run = []
    return tips


def _folded_stroke_path(
    contour: np.ndarray, stroke_half: float,
) -> tuple[str, list[tuple[int, int]]] | None:
    """The drawn path of a stroke traced around both of its sides, traced once.

    ``("closed", vertices)`` when the two ends almost meet (a polygon whose ring
    broke in thresholding), ``("open", vertices)`` for an open polyline, and
    None for a filled shape, a junction of several strokes, or a curve.
    """
    points = contour.reshape(-1, 2).astype(np.float64)
    count = len(points)
    band = 2 * stroke_half + STROKE_BAND_TOLERANCE_PX
    min_separation = int(FOLDED_PARTNER_SEPARATION * band)
    if count < 4 * min_separation:
        return None
    step = max(1, count // 400)
    sample = np.arange(0, count, step)
    index = np.arange(count)
    partnered = 0
    for i in sample:
        along = np.abs(index - i)
        along = np.minimum(along, count - along)
        far = along > min_separation
        if np.hypot(*(points[far] - points[i]).T).min() <= band:
            partnered += 1
    if partnered < FOLDED_PARTNER_FRACTION * len(sample):
        return None
    tips = _folded_tips(points, band)
    if len(tips) != 2:
        return None
    a, b = tips
    side = contour[a:b + 1] if a < b else np.concatenate([contour[a:], contour[:b + 1]])
    epsilon = 0.01 * cv2.arcLength(side, True)
    path = [(int(p[0][0]), int(p[0][1])) for p in cv2.approxPolyDP(side, epsilon, False)]
    if math.dist(points[a], points[b]) <= FOLDED_CLOSED_GAP * band:
        if math.dist(path[0], path[-1]) <= band:
            path = path[:-1]
        # The tips fall wherever the ring broke, usually part-way along a side.
        path = _drop_collinear_vertices(path)
        return ("closed", path) if len(path) >= 3 else None
    if len(path) - 1 > MAX_FOLDED_SIDES or any(math.dist(p, q) < 2 * band for p, q in zip(path, path[1:])):
        return None
    return "open", path


def _segment_within(a, b, other, band: float) -> bool:
    return _point_segment_distance(a, *other) <= band and _point_segment_distance(b, *other) <= band


def _contour_explained_by_lines(contour: np.ndarray, segments, stroke_half_width: float) -> bool:
    """True when a hole-less blob is just the stroke of already-detected lines."""
    points = contour.reshape(-1, 2)
    if not segments or len(points) == 0:
        return False
    tolerance = stroke_half_width + STROKE_BAND_TOLERANCE_PX
    near = int(np.count_nonzero(_min_segment_distances(points, segments) <= tolerance))
    return near / len(points) >= COVERED_FRACTION


def _min_segment_distances(points: np.ndarray, segments, chunk: int = 4096) -> np.ndarray:
    """Distance from each point to its nearest segment (same formula as `_point_segment_distance`)."""
    seg = np.asarray([(a[0], a[1], b[0], b[1]) for a, b in segments], dtype=np.float64)
    sx, sy = seg[:, 0], seg[:, 1]
    dx, dy = seg[:, 2] - sx, seg[:, 3] - sy
    length_sq = dx * dx + dy * dy
    safe = np.where(length_sq == 0, 1.0, length_sq)
    pts = points.astype(np.float64)
    out = np.empty(len(pts))
    for start in range(0, len(pts), chunk):
        px = pts[start:start + chunk, 0:1]
        py = pts[start:start + chunk, 1:2]
        t = np.clip(((px - sx) * dx + (py - sy) * dy) / safe, 0.0, 1.0)
        t = np.where(length_sq == 0, 0.0, t)
        out[start:start + chunk] = np.hypot(px - (sx + t * dx), py - (sy + t * dy)).min(axis=1)
    return out


def extract_shapes(binary_image: np.ndarray, edge_sensitivity: int = 50, tick_evidence: np.ndarray | None = None) -> list[dict]:
    """Vectorise a binary stroke mask into lines, closed outlines and ellipses.

    Each drawn feature is emitted once: a closed outline (a contour with an inner
    hole) is authoritative for the Hough lines that run along its stroke, and a
    stroke without a hole is left to the Hough lines that explain it.
    """
    stroke_half = stroke_half_width(binary_image)
    segments = _snap_shared_endpoints(
        _collapse_stroke_edges(_line_segments(binary_image, edge_sensitivity), 2 * stroke_half + STROKE_BAND_TOLERANCE_PX),
        2 * stroke_half + STROKE_BAND_TOLERANCE_PX,
    )
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    closed_shapes: list[dict] = []
    outlines: list[np.ndarray] = []
    open_contours: list[dict] = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        holes = []
        child = hierarchy[index][2]
        while child != -1:
            holes.append(contours[child])
            child = hierarchy[child][0]
        hole_area = sum(cv2.contourArea(hole) for hole in holes)

        outer_points = _approx_points(contour)
        if len(outer_points) < 3:
            continue
        if hole_area < area * MIN_HOLE_AREA_FRACTION:
            if not _contour_explained_by_lines(contour, segments, stroke_half):
                folded = _folded_stroke_path(contour, stroke_half)
                if folded is None:
                    open_contours.append({"type": "contour", "points": outer_points})
                elif folded[0] == "closed":
                    open_contours.append({"type": "contour", "points": folded[1], "origin": "folded_outline"})
                else:
                    band = 2 * stroke_half + STROKE_BAND_TOLERANCE_PX
                    open_contours.extend(
                        {"type": "line", "points": [a, b], "origin": "folded_outline"}
                        for a, b in zip(folded[1], folded[1][1:])
                        if not any(_segment_within(a, b, seg, band) for seg in segments)
                    )
            continue

        outlines.append(contour)
        inner_points = _approx_points(holes[0]) if len(holes) == 1 else None
        points = _centerline_points(outer_points, inner_points)
        ellipse = None
        round_outline = False
        if len(outer_points) >= MIN_ROUND_VERTICES and len(contour) >= 5:
            center, (a, b), angle = _ellipse_from_fit(cv2.fitEllipse(contour))
            round_outline = _ellipse_fit_error(contour, center, (a, b), angle) <= ELLIPSE_FIT_MAX_ERROR
            if round_outline:
                # The outer boundary sits half a stroke outside the drawn centreline.
                a, b = max(a - stroke_half, 1.0), max(b - stroke_half, 1.0)
                if b / a < ELLIPSE_MAX_ASPECT and math.pi * a * b > ELLIPSE_MIN_AREA:
                    ellipse = {
                        "type": "ellipse",
                        "center": (round(center[0]), round(center[1])),
                        "semi_axes": (round(a), round(b)),
                        "angle": round(angle, 1),
                        "area": round(math.pi * a * b, 2),
                        "contour_points": points,
                    }
        if not round_outline and len(holes) > 1 and _contour_explained_by_lines(contour, segments, stroke_half):
            # Several enclosed regions that the detected lines already trace form a
            # line network (e.g. a figure built from triangles); keep the lines.
            outlines.pop()
            continue
        closed_shapes.append(ellipse or {"type": "contour", "points": points})

    band_px = 2 * stroke_half + STROKE_BAND_TOLERANCE_PX
    shapes: list[dict] = [
        {"type": "line", "points": [start, end]}
        for start, end in segments
        if not any(_line_covered_by_outline((start, end), outer, band_px) for outer in outlines)
    ]
    shapes = reconstruct_closed_shapes(shapes)
    shapes.extend(recover_ticks(binary_image, shapes, stroke_half, tick_evidence))
    shapes.extend(closed_shapes)
    shapes.extend(open_contours)
    dots = detect_dots(binary_image, stroke_half)
    # The disc's own outline and any short Hough stroke across it are the dot.
    margin = 2 * stroke_half + STROKE_BAND_TOLERANCE_PX
    shapes = [s for s in shapes if s["type"] == "ellipse" or not _inside_dot(s["points"], dots, margin)]
    return shapes + dots


MAX_CYCLE_SIDES = 8
COLLINEAR_VERTEX_DEG = 12.0
MIN_CYCLE_AREA = 400.0


def _drop_collinear_vertices(cycle: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove vertices where two consecutive sides continue in the same direction."""
    changed = True
    while changed and len(cycle) > 3:
        changed = False
        for index, point in enumerate(cycle):
            prev, nxt = cycle[index - 1], cycle[(index + 1) % len(cycle)]
            a = math.atan2(point[1] - prev[1], point[0] - prev[0])
            b = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
            turn = abs((math.degrees(b - a) + 180) % 360 - 180)
            if turn < COLLINEAR_VERTEX_DEG:
                cycle = cycle[:index] + cycle[index + 1:]
                changed = True
                break
    return cycle


def _simple_polygon(points: list[tuple[int, int]]) -> bool:
    edges = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    for i, (a, b) in enumerate(edges):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            c, d = edges[j]
            if _segments_cross(a, b, c, d):
                return False
    return True


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def reconstruct_closed_shapes(shapes: list[dict]) -> list[dict]:
    """Join line segments that close exactly one loop into one closed outline.

    A closed figure crossed by another line (an altitude, a transversal) has
    several enclosed regions, so ``extract_shapes`` keeps it as separate lines.
    Within each connected group of lines that share (snapped) endpoints, a group
    containing exactly one loop - edges = vertices - number of loops + 1 - has an
    unambiguous outline: its loop becomes a contour and its other lines stay
    lines. Groups with two or more loops (a rectangle with a diagonal) are left
    as lines because the intended outline is ambiguous.
    """
    lines = [s for s in shapes if s["type"] == "line"]
    others = [s for s in shapes if s["type"] != "line"]
    adjacency: dict[tuple, list[int]] = {}
    for index, line in enumerate(lines):
        a, b = map(tuple, line["points"])
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)

    used: set[int] = set()
    seen: set[int] = set()
    contours: list[dict] = []
    for start in range(len(lines)):
        if start in seen:
            continue
        component, stack, vertices = set(), [start], set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            for end in map(tuple, lines[index]["points"]):
                vertices.add(end)
                stack.extend(adjacency[end])
        seen |= component
        if len(component) - len(vertices) + 1 != 1:
            continue
        cycle = _single_cycle(component, lines, adjacency)
        if cycle is None:
            continue
        vertices_in_order, cycle_lines = cycle
        polygon = _drop_collinear_vertices(vertices_in_order)
        if not 3 <= len(polygon) <= MAX_CYCLE_SIDES or not _simple_polygon(polygon):
            continue
        area = abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(polygon, polygon[1:] + polygon[:1]))) / 2
        if area < MIN_CYCLE_AREA:
            continue
        used |= cycle_lines
        contours.append({"type": "contour", "points": [tuple(p) for p in polygon], "source": "joined_sides"})
    kept = [line for index, line in enumerate(lines) if index not in used]
    return kept + contours + others


def _single_cycle(component: set[int], lines: list[dict], adjacency: dict) -> tuple[list, set[int]] | None:
    """The one loop of a connected line group with cyclomatic number 1."""
    degree = {v: len([i for i in idx if i in component]) for v, idx in adjacency.items()}
    remaining = set(component)
    # Peel pendant lines until only the loop remains.
    changed = True
    while changed:
        changed = False
        for index in list(remaining):
            ends = list(map(tuple, lines[index]["points"]))
            if any(degree[e] == 1 for e in ends):
                remaining.discard(index)
                for e in ends:
                    degree[e] -= 1
                changed = True
    if len(remaining) < 3:
        return None
    first = next(iter(remaining))
    a, b = map(tuple, lines[first]["points"])
    order, current, previous_line = [a], b, first
    while current != a:
        order.append(current)
        nxt = [i for i in adjacency[current] if i in remaining and i != previous_line]
        if len(nxt) != 1:
            return None
        previous_line = nxt[0]
        p, q = map(tuple, lines[previous_line]["points"])
        current = q if p == current else p
        if len(order) > len(remaining):
            return None
    return order, remaining


MIN_TICK_HOST_PX = 120
TICK_BAND_PX = 26
MIN_TICK_PX = 6
MAX_TICK_PX = 40
TICK_PERPENDICULAR_DEG = 20.0
TICK_MIN_ELONGATION = 2.5
TICK_PIECE_ELONGATION = 1.5
TICK_PAIR_PX = 5
MIN_TICK_PIECE_PIXELS = 3
# Pixel extents measured along a slightly tilted line fall just short of whole pixels.
PIXEL_SAMPLING_TOLERANCE = 0.5
# Ticks drawn on one side of an axis form a regular series: at least this many
# strokes of similar length (within the tolerance) at multiples of one pitch.
MIN_TICK_SERIES = 3
TICK_SERIES_LENGTH_TOLERANCE = 0.4
TICK_SERIES_PITCH_TOLERANCE = 0.12
TICK_SERIES_MIN_SLACK_PX = 3.0
TICK_SERIES_MAX_SKIP = 2
# Blur or JPEG can open a small gap between a tick half and its line.
TICK_GAP_PX = 2
# A short crossing tick on a small drawing barely clears the line's stroke mask.
# It is found instead as a local widening of the stroke's cross-section that
# sticks out on both sides by this much beyond the usual stroke edges.
MIN_TICK_OVERHANG_PX = 2.0
TICK_PROFILE_STEP_PX = 0.5
TICK_END_MARGIN_PX = 12


# A drawn dot is a filled disc several strokes wide: its distance-transform core
# is round and compact, unlike the ridge along a stroke, the wedge of a filled
# arrowhead or a filled bar.
DOT_CORE_STROKES = 3.0
MIN_DOT_CORE_PX = 3.0
MIN_DOT_CORE_AREA = 8
MAX_DOT_CORE_AREA = 400
MAX_DOT_CORE_ASPECT = 1.3
MIN_DOT_CORE_FILL = 0.5


def detect_dots(binary: np.ndarray, stroke_half: float) -> list[dict]:
    """Filled discs (marked points, circle centres, plotted points)."""
    distance = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    core = (distance >= max(MIN_DOT_CORE_PX, DOT_CORE_STROKES * stroke_half)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(core, connectivity=8)
    dots: list[dict] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not MIN_DOT_CORE_AREA <= area <= MAX_DOT_CORE_AREA:
            continue
        if max(w, h) > MAX_DOT_CORE_ASPECT * min(w, h) or area < MIN_DOT_CORE_FILL * w * h:
            continue
        cx, cy = centroids[label]
        radius = float(distance[y:y + h, x:x + w].max())
        dots.append({"type": "dot", "points": [(int(round(cx)), int(round(cy)))], "radius": round(radius, 1)})
    return dots


def _inside_dot(points, dots: list[dict], margin: float) -> bool:
    return any(all(math.dist(p, dot["points"][0]) <= dot["radius"] + margin for p in points) for dot in dots)


def recover_ticks(binary: np.ndarray, shapes: list[dict], stroke_half: float, evidence: np.ndarray | None = None) -> list[dict]:
    """Short strokes drawn across, or outward from, a long line (axis / number-line ticks).

    Ticks are shorter than the Hough minimum line length, so they are found as
    ink left over next to a long line once that line's stroke is removed: a
    small elongated component, perpendicular to the line, that touches it. The
    pieces on either side of the line at the same position form one tick. A
    one-sided stroke is a tick only as part of an evenly spaced series of
    similar strokes on the same side, which an angle arc or a label is not.

    ``evidence`` is an unthinned ink mask. Thin tick halves that the stroke
    mask's opening and erosion cut away are measured there too; a tick found in
    either mask is kept once.
    """
    ticks = _ticks_in(binary, shapes, stroke_half)
    candidates = _profile_ticks(binary, shapes, stroke_half)
    if evidence is not None:
        candidates = _ticks_in(evidence, shapes, max(stroke_half, stroke_half_width(evidence))) + candidates
    for tick in candidates:
        if not any(_tick_distance(tick, kept) <= TICK_PAIR_PX for kept in ticks):
            ticks.append(tick)
    return ticks


def _ink_run_across(ink: np.ndarray, cx: float, cy: float, nx: float, ny: float, offsets: np.ndarray):
    """Extent (lo, hi) along the normal of the ink run through the stroke at (cx, cy)."""
    height, width = ink.shape[:2]
    xs = np.clip(np.rint(cx + nx * offsets).astype(int), 0, width - 1)
    ys = np.clip(np.rint(cy + ny * offsets).astype(int), 0, height - 1)
    inked = ink[ys, xs] > 0
    centre = len(offsets) // 2
    near = np.flatnonzero(inked[max(0, centre - 3):centre + 4])
    if not len(near):
        return None
    start = end = max(0, centre - 3) + int(near[0])
    while start > 0 and inked[start - 1]:
        start -= 1
    while end + 1 < len(inked) and inked[end + 1]:
        end += 1
    return float(offsets[start]), float(offsets[end])


def _plain_stroke(run: tuple[float, float] | None, lo_edge: float, hi_edge: float, overhang: float) -> bool:
    """A cross-section at the line's usual width: a tick is a short widening, an arrowhead is not."""
    return run is not None and run[0] > lo_edge - overhang / 2 and run[1] < hi_edge + overhang / 2


def _profile_ticks(ink: np.ndarray, shapes: list[dict], ink_half: float) -> list[dict]:
    """Short ticks drawn across a long line, measured as a local widening of its stroke."""
    hosts = [s for s in shapes if s["type"] == "line" and math.dist(*s["points"]) >= MIN_TICK_HOST_PX]
    if not hosts:
        return []
    thickness = max(3, int(round(2 * ink_half + 3)))
    overhang = max(MIN_TICK_OVERHANG_PX, ink_half)
    offsets = np.arange(-MAX_TICK_PX / 2, MAX_TICK_PX / 2 + TICK_PROFILE_STEP_PX, TICK_PROFILE_STEP_PX)
    ticks: list[dict] = []
    for host in hosts:
        others = np.zeros_like(ink)
        for shape in shapes:
            points = shape.get("points") or []
            if shape is host or len(points) < 2 or shape["type"] not in ("line", "contour"):
                continue
            closed = shape["type"] == "contour"
            cv2.polylines(others, [np.array(points, dtype=np.int32)], closed, 255, thickness)
        (x1, y1), (x2, y2) = host["points"]
        length = math.dist((x1, y1), (x2, y2))
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        samples = []
        for t in np.arange(TICK_END_MARGIN_PX, length - TICK_END_MARGIN_PX, 1.0):
            cx, cy = x1 + ux * t, y1 + uy * t
            run = _ink_run_across(ink, cx, cy, -uy, ux, offsets)
            if run is not None:
                samples.append((float(t), *run))
        if len(samples) < 2 * TICK_END_MARGIN_PX:
            continue
        profile = {t: (lo, hi) for t, lo, hi in samples}
        lo_edge = float(np.median([lo for _, lo, _ in samples]))
        hi_edge = float(np.median([hi for _, _, hi in samples]))
        wide = [(t, lo, hi) for t, lo, hi in samples if lo <= lo_edge - overhang and hi >= hi_edge + overhang]
        groups: list[list[tuple[float, float, float]]] = []
        for sample in wide:
            if groups and sample[0] - groups[-1][-1][0] <= 1.5:
                groups[-1].append(sample)
            else:
                groups.append([sample])
        for group in groups:
            along = group[-1][0] - group[0][0] + 1
            lo, hi = min(g[1] for g in group), max(g[2] for g in group)
            span = hi - lo
            if along > 2 * thickness or not MIN_TICK_PX <= span <= MAX_TICK_PX or span < TICK_MIN_ELONGATION * along:
                continue
            if lo <= offsets[0] or hi >= offsets[-1]:
                continue
            if not all(_plain_stroke(profile.get(t), lo_edge, hi_edge, overhang)
                       for t in (group[0][0] - thickness, group[-1][0] + thickness)):
                continue
            t_mid = float(np.median([g[0] for g in group]))
            cx, cy = x1 + ux * t_mid, y1 + uy * t_mid
            if others[int(round(cy)), int(round(cx))]:
                continue
            a = (int(round(cx - uy * lo)), int(round(cy + ux * lo)))
            b = (int(round(cx - uy * hi)), int(round(cy + ux * hi)))
            ticks.append({"type": "line", "points": [a, b], "role": "tick", "tick_evidence": "crossing"})
    return ticks


def _tick_distance(a: dict, b: dict) -> float:
    (a0, a1), (b0, b1) = a["points"], b["points"]
    return math.dist(((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2), ((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2))


def _ticks_in(ink: np.ndarray, shapes: list[dict], ink_half: float) -> list[dict]:
    height, width = ink.shape[:2]
    hosts = [s for s in shapes if s["type"] == "line" and math.dist(*s["points"]) >= MIN_TICK_HOST_PX]
    if not hosts:
        return []
    line_mask = np.zeros_like(ink)
    thickness = max(3, int(round(2 * ink_half + 3)))
    for shape in shapes:
        if shape["type"] == "line":
            cv2.line(line_mask, tuple(map(int, shape["points"][0])), tuple(map(int, shape["points"][1])), 255, thickness)
    residual = cv2.bitwise_and(ink, cv2.bitwise_not(line_mask))
    ticks: list[dict] = []
    for host in hosts:
        (x1, y1), (x2, y2) = host["points"]
        length = math.dist((x1, y1), (x2, y2))
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        band = np.zeros_like(ink)
        cv2.line(band, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2 * TICK_BAND_PX)
        local = cv2.bitwise_and(residual, band)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
        pieces = []
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            if len(xs) < MIN_TICK_PIECE_PIXELS:
                continue
            pts = np.column_stack([xs, ys]).astype(float)
            t = (pts[:, 0] - x1) * ux + (pts[:, 1] - y1) * uy
            n = (pts[:, 0] - x1) * -uy + (pts[:, 1] - y1) * ux
            if t.min() < 4 or t.max() > length - 4:
                continue
            along, across = t.max() - t.min() + 1, n.max() - n.min() + 1
            near = np.abs(n).min()
            if near > thickness / 2 + TICK_GAP_PX or across < MIN_TICK_PX / 2 - PIXEL_SAMPLING_TOLERANCE or across > MAX_TICK_PX:
                continue
            if across < TICK_PIECE_ELONGATION * along:
                continue
            mean = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
            angle = math.degrees(math.atan2(vt[0][1], vt[0][0]))
            host_angle = math.degrees(math.atan2(uy, ux))
            off = abs((angle - host_angle) % 180 - 90)
            if off > TICK_PERPENDICULAR_DEG:
                continue
            pieces.append((float(np.median(t)), float(n.min()), float(n.max()), float(along)))
        pieces.sort()
        merged: list[list[float]] = []
        for t_mid, n_lo, n_hi, along in pieces:
            if merged and abs(merged[-1][0] - t_mid) <= TICK_PAIR_PX:
                merged[-1][1] = min(merged[-1][1], n_lo)
                merged[-1][2] = max(merged[-1][2], n_hi)
                merged[-1][3] = max(merged[-1][3], along)
            else:
                merged.append([t_mid, n_lo, n_hi, along])
        found: list[tuple[float, float, float, str]] = []
        one_sided: dict[int, list[tuple[float, float, float]]] = {-1: [], 1: []}
        for t_mid, n_lo, n_hi, along in merged:
            crosses = n_lo <= -thickness / 2 and n_hi >= thickness / 2
            if crosses:
                lo, hi = n_lo, n_hi
            elif n_lo > -thickness / 2:
                lo, hi = 0.0, n_hi
            else:
                lo, hi = n_lo, 0.0
            span = hi - lo
            if span < MIN_TICK_PX - PIXEL_SAMPLING_TOLERANCE or span > MAX_TICK_PX or span < TICK_MIN_ELONGATION * along:
                continue
            if crosses:
                found.append((t_mid, lo, hi, "crossing"))
            else:
                one_sided[1 if hi > 0 else -1].append((t_mid, lo, hi))
        for series in one_sided.values():
            found.extend((t_mid, lo, hi, "series") for t_mid, lo, hi in _regular_tick_series(series))
        for t_mid, n_lo, n_hi, kind in found:
            cx, cy = x1 + ux * t_mid, y1 + uy * t_mid
            a = (int(round(cx - uy * n_lo)), int(round(cy + ux * n_lo)))
            b = (int(round(cx - uy * n_hi)), int(round(cy + ux * n_hi)))
            if all(0 <= p[0] < width and 0 <= p[1] < height for p in (a, b)):
                ticks.append({"type": "line", "points": [a, b], "role": "tick", "tick_evidence": kind})
    return ticks


def _regular_tick_series(candidates: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """One-sided strokes of similar length at evenly spaced positions along their line."""
    if len(candidates) < MIN_TICK_SERIES:
        return []
    spans = sorted(hi - lo for _, lo, hi in candidates)
    typical = spans[len(spans) // 2]
    similar = sorted(
        (c for c in candidates if abs((c[2] - c[1]) - typical) <= TICK_SERIES_LENGTH_TOLERANCE * typical),
        key=lambda c: c[0],
    )
    if len(similar) < MIN_TICK_SERIES:
        return []
    gaps = sorted(b[0] - a[0] for a, b in zip(similar, similar[1:]))
    pitch = gaps[len(gaps) // 2]
    if pitch <= 0:
        return []
    slack = max(TICK_SERIES_MIN_SLACK_PX, TICK_SERIES_PITCH_TOLERANCE * pitch)

    def on_pitch(gap: float) -> bool:
        steps = round(gap / pitch)
        return 1 <= steps <= TICK_SERIES_MAX_SKIP and abs(gap - steps * pitch) <= slack

    kept = [
        c for i, c in enumerate(similar)
        if any(on_pitch(abs(c[0] - other[0])) for j, other in enumerate(similar) if j != i)
    ]
    return kept if len(kept) >= MIN_TICK_SERIES else []


GLYPH_MASK_MAX_ASPECT = 2.5


def mask_text_glyphs(binary_image: np.ndarray, text_boxes: list[list[tuple[int, int]]], padding: int = 2) -> np.ndarray:
    """Copy of a stroke mask with the glyphs of read text removed, before tracing.

    Only ink components lying wholly inside a (padded) text box are erased. A
    stroke that enters or crosses the box, such as a leader line, a dimension
    line or a side the label touches, extends beyond it and is kept whole.
    Stroke-shaped components (a "1", "l" or "-", or one dash of a dashed line
    that OCR read as one) are ambiguous and are kept for the tracer to judge.
    """
    if not text_boxes:
        return binary_image
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    height, width = binary_image.shape[:2]
    glyph = np.zeros(count, dtype=bool)
    for bbox in text_boxes:
        xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
        x0, y0 = max(0, min(xs) - padding), max(0, min(ys) - padding)
        x1, y1 = min(width, max(xs) + padding + 1), min(height, max(ys) + padding + 1)
        for label in np.unique(labels[y0:y1, x0:x1]):
            if label == 0:
                continue
            x, y, w, h = (int(v) for v in stats[label, :4])
            stroke_shaped = max(w, h) >= GLYPH_MASK_MAX_ASPECT * max(1, min(w, h))
            if x >= x0 and y >= y0 and x + w <= x1 and y + h <= y1 and not stroke_shaped:
                glyph[label] = True
    masked = binary_image.copy()
    masked[glyph[labels]] = 0
    return masked


def drop_shapes_inside_text_regions(shapes: list[dict], text_boxes: list[list[tuple[int, int]]], padding: int = 2) -> list[dict]:
    """Remove contours that fall inside an OCR text region.

    Letter glyphs have enough closed area to survive the contour-area filter, so
    without this a worksheet label is vectorized twice: once as braille text and
    once as a tiny polygon. The polygon then trips the minimum-size blocking QA
    check, blocking export on a correctly processed worksheet.
    """
    if not text_boxes:
        return shapes
    regions: list[tuple[int, int, int, int]] = []
    for box in text_boxes:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        regions.append((min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding))

    kept: list[dict] = []
    for shape in shapes:
        if shape["type"] == "line":
            # A stroke that starts and ends inside one text box is part of a glyph.
            if not any(all(r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3] for p in shape["points"]) for r in regions):
                kept.append(shape)
            continue
        points = shape.get("points") or ([shape["center"]] if shape.get("center") else [])
        if not points:
            kept.append(shape)
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            xs = [cx - rx, cx + rx]
            ys = [cy - ry, cy + ry]
        box = (min(xs), min(ys), max(xs), max(ys))
        if any(_mostly_inside(box, region) for region in regions):
            continue
        kept.append(shape)
    return kept


TEXT_CONTAINMENT_FRACTION = 0.6


def _mostly_inside(box: tuple[float, float, float, float], region: tuple[int, int, int, int]) -> bool:
    """True when most of ``box`` lies inside ``region``.

    A plain overlap test also removed real geometry, such as a triangle whose
    vertex label sits just inside its bounding box.
    """
    width = max(box[2] - box[0], 1.0)
    height = max(box[3] - box[1], 1.0)
    overlap_w = min(box[2], region[2]) - max(box[0], region[0])
    overlap_h = min(box[3], region[3]) - max(box[1], region[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return (overlap_w * overlap_h) / (width * height) >= TEXT_CONTAINMENT_FRACTION


def shapes_to_svg(shapes: list[dict], width: int, height: int) -> str:
    elements: list[str] = []
    for shape in shapes:
        if shape["type"] == "ellipse":
            cx, cy = shape["center"]
            rx, ry = shape["semi_axes"]
            angle = shape.get("angle", 0)
            elements.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({angle} {cx} {cy})" />')
            continue
        points = shape.get("points")
        if not points:
            continue
        if shape["type"] == "line":
            (x1, y1), (x2, y2) = points
            elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        else:
            elements.append('<polygon points="' + ' '.join(f"{x},{y}" for x, y in points) + '" />')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Raw geometry vector preview">'
            '<rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + ''.join(elements) + '</g></svg>')
