"""Math figure templates, one family per benchmark category (A-L).

Each template draws into a box on a :class:`Scene` and records ground truth
through the scene. ``level`` (1-3) raises the number of elements and
relationships; the random generator varies proportions, labels and layout so
that no two images are the same drawing.
"""

from __future__ import annotations

import math
import string

import numpy as np

from .scene import Scene

Box = tuple[float, float, float, float]


def _frame(box: Box, margin: float = 0.12):
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return x0 + mx, y0 + my, x1 - mx, y1 - my


def _letters(rng: np.random.Generator, n: int) -> list[str]:
    start = int(rng.integers(0, 26 - n))
    return list(string.ascii_uppercase[start:start + n])


def _outward(p, centroid, dist):
    dx, dy = p[0] - centroid[0], p[1] - centroid[1]
    n = math.hypot(dx, dy) or 1.0
    return (dx / n * dist, dy / n * dist)


def _labelled_polygon(s: Scene, pts, kind, names, fs, rng, dashed=False):
    poly = s.polygon(pts, kind=kind, dashed=dashed)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pids = []
    for p, name in zip(pts, names):
        pid = s.point(p, role="vertex")
        s.relate("VERTEX_OF", pid, poly)
        s.label_near(p, name, _outward(p, (cx, cy), fs * 0.9), size=fs, associated=pid, face="serif_italic")
        pids.append(pid)
    return poly, pids


def geometry_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """A: triangles and quadrilaterals with constructions."""
    x0, y0, x1, y1 = _frame(box)
    w, h = x1 - x0, y1 - y0
    variant = int(rng.integers(0, 3)) if level > 1 else 0
    if variant in (0, 1):
        apex = (x0 + w * rng.uniform(0.3, 0.7), y0 + h * rng.uniform(0.0, 0.1))
        left = (x0 + w * rng.uniform(0.0, 0.1), y1)
        right = (x1 - w * rng.uniform(0.0, 0.1), y1)
        names = _letters(rng, 4)
        tri, (pa, pb, pc) = _labelled_polygon(s, [apex, left, right], "triangle", names[:3], fs, rng)
        if variant == 1 or level >= 2:
            foot = (apex[0], y1)
            alt = s.segment(apex, foot, dashed=bool(rng.integers(0, 2)))
            pd = s.point(foot, role="foot")
            s.relate("ENDPOINT_OF", pa, alt)
            s.relate("ENDPOINT_OF", pd, alt)
            s.relate("INSIDE", alt, tri)
            s.label_near(foot, names[3], (0, fs * 0.9), size=fs, associated=pd, face="serif_italic")
            s.right_angle_marker(foot, (1, 0), (0, -1), size=fs * 0.7)
            s.relate("PERPENDICULAR", alt, tri)
        if level >= 3:
            mid = ((left[0] + right[0]) / 2 + w * 0.15, y1)
            med = s.segment(apex, ((apex[0] + right[0]) / 2, (apex[1] + right[1]) / 2))
            s.relate("INSIDE", med, tri)
            s.label_near(((apex[0] + left[0]) / 2, (apex[1] + left[1]) / 2),
                         f"{int(rng.integers(4, 13))} cm", (-fs * 1.6, 0), size=int(fs * 0.8), role="side_label", associated=tri)
            _ = mid
        return tri
    # quadrilateral with a diagonal
    pts = [(x0 + w * rng.uniform(0.05, 0.25), y0), (x1 - w * rng.uniform(0.0, 0.15), y0 + h * rng.uniform(0.0, 0.2)),
           (x1, y1), (x0, y1 - h * rng.uniform(0.0, 0.15))]
    names = _letters(rng, 4)
    quad, pids = _labelled_polygon(s, pts, "polygon", names, fs, rng)
    diag = s.segment(pts[0], pts[2])
    s.relate("ENDPOINT_OF", pids[0], diag)
    s.relate("ENDPOINT_OF", pids[2], diag)
    s.relate("INSIDE", diag, quad)
    if level >= 3:
        diag2 = s.segment(pts[1], pts[3])
        s.relate("INTERSECTS", diag, diag2)
        s.relate("INSIDE", diag2, quad)
    return quad


def graph_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """C: axes with ticks, a plotted line or curve, optional grid and legend."""
    x0, y0, x1, y1 = _frame(box, 0.1)
    ox, oy = x0 + (x1 - x0) * 0.12, y1 - (y1 - y0) * 0.12
    n = 5 + level
    sx, sy = (x1 - ox) / (n + 0.6), (oy - y0) / (n + 0.6)
    if level >= 2:
        for i in range(1, n + 1):
            s.segment((ox + i * sx, y0), (ox + i * sx, oy), kind="grid_line", importance="omittable", width=1, ink=170, mask=False)
            s.segment((ox, oy - i * sy), (x1, oy - i * sy), kind="grid_line", importance="omittable", width=1, ink=170, mask=False)
    xa = s.arrow((ox, oy), (x1, oy), kind="axis")
    ya = s.arrow((ox, oy), (ox, y0), kind="axis")
    s.relate("PERPENDICULAR", xa, ya)
    origin = s.point((ox, oy), role="origin")
    s.relate("ENDPOINT_OF", origin, xa)
    s.relate("ENDPOINT_OF", origin, ya)
    s.label_near((x1, oy), "x", (0, fs * 0.9), size=fs, role="axis_label", associated=xa, face="serif_italic")
    s.label_near((ox, y0), "y", (-fs * 0.9, 0), size=fs, role="axis_label", associated=ya, face="serif_italic")
    tick_fs = max(14, int(fs * 0.7))
    for i in range(1, n + 1, 1 if level == 1 else 2):
        t = s.segment((ox + i * sx, oy - 7), (ox + i * sx, oy + 7), kind="tick", importance="supporting", width=2)
        s.relate("ON", t, xa)
        s.label_near((ox + i * sx, oy), str(i), (0, tick_fs * 1.2), size=tick_fs, role="tick_label", associated=t)
        t2 = s.segment((ox - 7, oy - i * sy), (ox + 7, oy - i * sy), kind="tick", importance="supporting", width=2)
        s.relate("ON", t2, ya)
        s.label_near((ox, oy - i * sy), str(i), (-tick_fs * 1.1, 0), size=tick_fs, role="tick_label", associated=t2)
    kind = int(rng.integers(0, 2))
    if kind == 0:
        m, c = rng.uniform(0.4, 1.4), rng.uniform(0.3, 1.5)
        xs = np.linspace(0.2, n - 0.2, 2)
        pts = [(ox + x * sx, oy - min(n, m * x + c) * sy) for x in xs]
        curve = s.segment(pts[0], pts[1], kind="line_segment")
        eq = f"y = {m:.1f}x + {c:.1f}"
    else:
        a = rng.uniform(0.15, 0.3)
        xs = np.linspace(0.3, n - 0.3, 40)
        pts = [(ox + x * sx, oy - min(n, a * (x - n / 2) ** 2 + 0.5) * sy) for x in xs]
        curve = s.polyline(pts, kind="curve")
        eq = "y = f(x)"
    if level >= 2:
        s.text((x1 - (x1 - x0) * 0.42, y0 + fs * 0.2), eq, size=int(fs * 0.8), role="legend", associated=curve, face="serif_italic")
    if level >= 3:
        k = int(rng.integers(2, n - 1))
        px = ox + k * sx
        py = oy - min(n, a * (k - n / 2) ** 2 + 0.5) * sy if kind else pts[0][1] + (pts[1][1] - pts[0][1]) * (k - 0.2) / (n - 0.4)
        p = s.point((px, py), role="marked_point", dot=True)
        s.relate("ON", p, curve)
        s.label_near((px, py), "P", (fs * 0.8, -fs * 0.8), size=fs, associated=p, face="serif_italic")
    return curve


def coordinate_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """D: plotted points on a grid, joined into a segment or polygon."""
    x0, y0, x1, y1 = _frame(box, 0.08)
    n = 6 if level < 3 else 8
    size = min(x1 - x0, y1 - y0)
    step = size / (n + 1)
    ox, oy = x0 + step * 0.8, y0 + step * (n + 0.2)
    for i in range(1, n + 1):
        s.segment((ox + i * step, oy - n * step), (ox + i * step, oy), kind="grid_line", importance="omittable", width=1, ink=160, mask=False)
        s.segment((ox, oy - i * step), (ox + n * step, oy - i * step), kind="grid_line", importance="omittable", width=1, ink=160, mask=False)
    xa = s.arrow((ox, oy), (ox + (n + 0.6) * step, oy), kind="axis")
    ya = s.arrow((ox, oy), (ox, oy - (n + 0.6) * step), kind="axis")
    s.relate("PERPENDICULAR", xa, ya)
    s.label_near((ox, oy), "O", (-fs * 0.7, fs * 0.7), size=fs, role="axis_label", associated=None, face="serif_italic")
    count = 2 if level == 1 else 3 + (level == 3)
    used = set()
    coords = []
    while len(coords) < count:
        c = (int(rng.integers(1, n)), int(rng.integers(1, n)))
        if c not in used:
            used.add(c)
            coords.append(c)
    names = _letters(rng, count)
    pts, pids = [], []
    for (cx, cy), name in zip(coords, names):
        p = (ox + cx * step, oy - cy * step)
        pid = s.point(p, role="plotted_point", dot=True)
        text = f"{name}({cx}, {cy})" if level >= 2 else name
        s.label_near(p, text, (fs * (1.6 if level >= 2 else 0.8), -fs * 0.8), size=int(fs * 0.85), associated=pid, face="serif_italic")
        pts.append(p)
        pids.append(pid)
    if count == 2:
        seg = s.segment(pts[0], pts[1])
        for pid in pids:
            s.relate("ENDPOINT_OF", pid, seg)
        return seg
    kind = "triangle" if count == 3 else "polygon"
    poly = s.polygon(pts, kind=kind)
    for pid in pids:
        s.relate("VERTEX_OF", pid, poly)
    return poly


def measurement_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """E: shapes with dimension lines and measured lengths."""
    x0, y0, x1, y1 = _frame(box, 0.18)
    w = x1 - x0
    unit = str(rng.choice(["cm", "m", "mm"]))
    a, b = int(rng.integers(3, 15)), int(rng.integers(2, 10))
    if level >= 2 and rng.integers(0, 2):
        pts = [(x0, y1), (x1, y1), (x0 + w * rng.uniform(0.2, 0.8), y0)]
        shape = s.polygon(pts, kind="triangle")
    else:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        shape = s.polygon(pts, kind="rectangle")
    gap = fs * 1.1
    d1 = s.arrow((x0, y1 + gap), (x1, y1 + gap), kind="dimension_line", both=True, importance="essential", width=2)
    l1 = s.label_near(((x0 + x1) / 2, y1 + gap), f"{a} {unit}", (0, fs * 0.9), size=fs, role="measurement", associated=d1)
    s.dimension(str(a), unit, d1, l1)
    s.relate("DIMENSION_OF", d1, shape)
    if pts and len(pts) == 4:
        d2 = s.arrow((x1 + gap, y0), (x1 + gap, y1), kind="dimension_line", both=True, importance="essential", width=2)
        l2 = s.label_near((x1 + gap, (y0 + y1) / 2), f"{b} {unit}", (fs * 1.9, 0), size=fs, role="measurement", associated=d2)
        s.dimension(str(b), unit, d2, l2)
        s.relate("DIMENSION_OF", d2, shape)
    else:
        hgt = s.segment(pts[2], (pts[2][0], y1), dashed=True, kind="line_segment", importance="essential")
        s.right_angle_marker((pts[2][0], y1), (1, 0), (0, -1), size=fs * 0.6)
        s.relate("PERPENDICULAR", hgt, shape)
        lh = s.label_near((pts[2][0], (pts[2][1] + y1) / 2), f"h = {b} {unit}", (fs * 2.4, 0), size=int(fs * 0.85), role="measurement", associated=hgt)
        s.dimension(str(b), unit, hgt, lh)
    if level >= 3:
        s.text((x0, y0 - fs * 2.2), "Not drawn to scale", size=int(fs * 0.65), role="instruction", face="serif_italic")
    return shape


def regular_polygon(center, radius, sides, phase=-math.pi / 2):
    return [(center[0] + radius * math.cos(phase + 2 * math.pi * i / sides),
             center[1] + radius * math.sin(phase + 2 * math.pi * i / sides)) for i in range(sides)]


def shapes_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """F: regular and irregular polygons, optionally several or nested."""
    x0, y0, x1, y1 = _frame(box, 0.1)
    count = 1 if level == 1 else 2 + (level == 3)
    cell = (x1 - x0) / count
    first = None
    for i in range(count):
        cx = x0 + cell * (i + 0.5)
        cy = (y0 + y1) / 2
        r = min(cell, y1 - y0) * 0.38
        sides = int(rng.choice([3, 4, 5, 6, 8]))
        pts = regular_polygon((cx, cy), r, sides, phase=-math.pi / 2 + (math.pi / 4 if sides == 4 else 0))
        kind = {3: "triangle", 4: "rectangle"}.get(sides, "polygon")
        if level == 1 or rng.integers(0, 2):
            names = _letters(rng, sides) if sides <= 6 else [str(j + 1) for j in range(sides)]
            poly, _ = _labelled_polygon(s, pts, kind, names, int(fs * 0.85), rng)
        else:
            poly = s.polygon(pts, kind=kind)
            s.label_near((cx, cy + r + fs * 1.2), {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 8: "octagon"}[sides],
                         (0, 0), size=int(fs * 0.75), role="caption", associated=poly)
        if level == 3 and i == 0:
            inner = s.circle((cx, cy), r * math.cos(math.pi / sides) * 0.93)
            s.relate("INSIDE", inner, poly)
        first = first or poly
    return first


def angles_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """G: angles at a vertex, linear pairs, parallel lines with a transversal."""
    x0, y0, x1, y1 = _frame(box, 0.12)
    w, h = x1 - x0, y1 - y0
    if level == 1 or rng.integers(0, 3) == 0:
        v = (x0 + w * 0.2, y1 - h * 0.1)
        deg = int(rng.integers(25, 140))
        length = w * 0.7
        end1 = (v[0] + length, v[1])
        end2 = (v[0] + length * math.cos(math.radians(deg)), v[1] - length * math.sin(math.radians(deg)))
        names = _letters(rng, 3)
        r1 = s.arrow(v, end1, kind="ray")
        r2 = s.arrow(v, end2, kind="ray")
        pv = s.point(v, role="vertex")
        s.relate("ENDPOINT_OF", pv, r1)
        s.relate("ENDPOINT_OF", pv, r2)
        s.label_near(v, names[1], (-fs * 0.8, fs * 0.5), size=fs, associated=pv, face="serif_italic")
        s.label_near(end1, names[2], (fs * 0.4, fs * 0.9), size=fs, associated=r1, face="serif_italic")
        s.label_near(end2, names[0], (-fs * 0.9, -fs * 0.3), size=fs, associated=r2, face="serif_italic")
        arc = s.arc(v, fs * 2.2, -deg, 0, kind="angle_marker", importance="essential")
        s.relate("ANGLE_AT", arc, pv)
        mid = math.radians(deg / 2)
        s.label_near(v, f"{deg}°", (fs * 3.6 * math.cos(mid), -fs * 3.6 * math.sin(mid)), size=fs, role="angle_label", associated=arc)
        return arc
    # parallel lines cut by a transversal
    ya_, yb_ = y0 + h * 0.3, y0 + h * 0.75
    l1 = s.arrow((x0, ya_), (x1, ya_), kind="line_segment", both=True)
    l2 = s.arrow((x0, yb_), (x1, yb_), kind="line_segment", both=True)
    s.relate("PARALLEL", l1, l2)
    slope = rng.uniform(0.6, 1.6)
    cx = x0 + w * rng.uniform(0.4, 0.6)
    t0 = (cx - (ya_ - y0) / slope, y0)
    t1 = (cx + (y1 - ya_) / slope, y1)
    tr = s.segment(t0, t1, kind="line_segment")
    s.relate("INTERSECTS", tr, l1)
    s.relate("INTERSECTS", tr, l2)
    s.label_near((x1, ya_), "l", (fs * 0.6, -fs * 0.6), size=fs, associated=l1, face="serif_italic")
    s.label_near((x1, yb_), "m", (fs * 0.6, -fs * 0.6), size=fs, associated=l2, face="serif_italic")
    s.label_near(t1, "t", (fs * 0.6, 0), size=fs, associated=tr, face="serif_italic")
    deg = int(round(math.degrees(math.atan(slope))))
    xi1 = cx
    xi2 = cx + (yb_ - ya_) / slope
    p1 = s.point((xi1, ya_), role="intersection")
    p2 = s.point((xi2, yb_), role="intersection")
    s.relate("ON", p1, l1)
    s.relate("ON", p1, tr)
    s.relate("ON", p2, l2)
    s.relate("ON", p2, tr)
    s.label_near((xi1, ya_), f"{deg}°", (fs * 1.6, -fs * 0.7), size=int(fs * 0.85), role="angle_label", associated=p1)
    if level >= 2:
        s.label_near((xi2, yb_), "x", (-fs * 1.3, fs * 0.8), size=fs, role="angle_label", associated=p2, face="serif_italic")
    if level >= 3:
        s.label_near((xi2, yb_), "y", (fs * 1.2, fs * 0.8), size=fs, role="angle_label", associated=p2, face="serif_italic")
    return l1


def circles_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """H: circle with centre, radius, chord, diameter, tangent."""
    x0, y0, x1, y1 = _frame(box, 0.14)
    r = min(x1 - x0, y1 - y0) * 0.42
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    circ = s.circle(c, r)
    o = s.point(c, role="centre", dot=True)
    s.relate("CENTER_OF", o, circ)
    s.label_near(c, "O", (-fs * 0.8, fs * 0.6), size=fs, associated=o, face="serif_italic")
    a1 = math.radians(float(rng.uniform(-60, 30)))
    a = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
    rad = s.segment(c, a)
    pa = s.point(a, role="on_circle")
    s.relate("ON", pa, circ)
    s.relate("ENDPOINT_OF", o, rad)
    s.relate("ENDPOINT_OF", pa, rad)
    s.label_near(a, "A", _outward(a, c, fs * 0.9), size=fs, associated=pa, face="serif_italic")
    s.label_near(((c[0] + a[0]) / 2, (c[1] + a[1]) / 2), f"{int(rng.integers(3, 9))} cm", (0, -fs * 0.9), size=int(fs * 0.8),
                 role="measurement", associated=rad)
    if level >= 2:
        b1, b2 = math.radians(float(rng.uniform(110, 160))), math.radians(float(rng.uniform(200, 250)))
        pb_ = (c[0] + r * math.cos(b1), c[1] + r * math.sin(b1))
        pc_ = (c[0] + r * math.cos(b2), c[1] + r * math.sin(b2))
        chord = s.segment(pb_, pc_)
        s.relate("INSIDE", chord, circ)
        for p, name in ((pb_, "B"), (pc_, "C")):
            pid = s.point(p, role="on_circle")
            s.relate("ON", pid, circ)
            s.relate("ENDPOINT_OF", pid, chord)
            s.label_near(p, name, _outward(p, c, fs * 0.9), size=fs, associated=pid, face="serif_italic")
    if level >= 3:
        t = (-math.sin(a1), math.cos(a1))
        length = r * 0.9
        tangent = s.segment((a[0] - t[0] * length, a[1] - t[1] * length), (a[0] + t[0] * length, a[1] + t[1] * length))
        s.relate("PERPENDICULAR", tangent, rad)
        s.relate("ON", pa, tangent)
        s.right_angle_marker(a, (-math.cos(a1), -math.sin(a1)), t, size=fs * 0.6)
    return circ


def transformation_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """I: reflection across a mirror line or a translation, on a grid."""
    x0, y0, x1, y1 = _frame(box, 0.08)
    n = 10
    step = min((x1 - x0), (y1 - y0)) / n
    gx, gy = x0, y0
    if level >= 2:
        for i in range(n + 1):
            s.segment((gx + i * step, gy), (gx + i * step, gy + n * step), kind="grid_line", importance="omittable", width=1, ink=170, mask=False)
            s.segment((gx, gy + i * step), (gx + n * step, gy + i * step), kind="grid_line", importance="omittable", width=1, ink=170, mask=False)
    tri = [(1, 2), (4, 2), (2, 5)] if rng.integers(0, 2) else [(1, 1), (4, 1), (4, 3), (1, 4)]
    kind = "triangle" if len(tri) == 3 else "polygon"
    pts = [(gx + x * step, gy + y * step) for x, y in tri]
    names = _letters(rng, len(tri))
    shape, _ = _labelled_polygon(s, pts, kind, names, int(fs * 0.8), rng)
    if rng.integers(0, 2) or level == 1:
        mirror = s.segment((gx + 5 * step, gy - step * 0.3), (gx + 5 * step, gy + n * step + step * 0.3), dashed=True, kind="line_segment")
        s.label_near((gx + 5 * step, gy - step * 0.3), "m", (fs * 0.6, -fs * 0.3), size=fs, associated=mirror, face="serif_italic")
        img = [(gx + (10 - x) * step, gy + y * step) for x, y in tri]
        image, _ = _labelled_polygon(s, img, kind, [n_ + "′" for n_ in names], int(fs * 0.8), rng)
        s.relate("REFLECTION_OF", image, shape)
        return image
    dx, dy = 5, 4
    img = [(gx + (x + dx) * step, gy + (y + dy) * step) for x, y in tri]
    image, _ = _labelled_polygon(s, img, kind, [n_ + "′" for n_ in names], int(fs * 0.8), rng)
    s.arrow(pts[0], img[0], kind="arrow", width=2)
    s.relate("TRANSLATION_OF", image, shape)
    return image


def number_line_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """J: number lines with marked points, and fraction bars."""
    x0, y0, x1, y1 = _frame(box, 0.06)
    w, h = x1 - x0, y1 - y0
    if level == 1 or rng.integers(0, 2):
        y = y0 + h * (0.5 if level < 3 else 0.3)
        lo = int(rng.integers(-5, 1))
        n = 10
        line = s.arrow((x0, y), (x1, y), kind="number_line", both=True)
        step = w * 0.9 / n
        sx = x0 + w * 0.05
        for i in range(n + 1):
            t = s.segment((sx + i * step, y - 10), (sx + i * step, y + 10), kind="tick", importance="supporting", width=2)
            s.relate("ON", t, line)
            s.label_near((sx + i * step, y), str(lo + i), (0, fs * 1.1), size=int(fs * 0.75), role="tick_label", associated=t)
        for k in range(1 if level == 1 else 2):
            v = int(rng.integers(1, n))
            p = s.point((sx + v * step, y), role="marked_point", dot=True, radius=8)
            s.relate("ON", p, line)
            s.label_near((sx + v * step, y), "PQ"[k], (0, -fs * 1.0), size=fs, associated=p, face="serif_italic")
        if level < 3:
            return line
        y0 = y + fs * 2.5
    # fraction bar
    parts = int(rng.integers(3, 9))
    shaded = int(rng.integers(1, parts))
    by0, by1 = y0 + (y1 - y0) * 0.3, y0 + (y1 - y0) * 0.3 + min(h * 0.3, fs * 3)
    bar = s.polygon([(x0, by0), (x1, by0), (x1, by1), (x0, by1)], kind="bar")
    pw = (x1 - x0) / parts
    for i in range(parts):
        if i < shaded:
            s.draw.rectangle([x0 + i * pw + 3, by0 + 3, x0 + (i + 1) * pw - 3, by1 - 3], fill=185)
        if i:
            d = s.segment((x0 + i * pw, by0), (x0 + i * pw, by1), kind="line_segment", importance="essential")
            s.relate("INSIDE", d, bar)
    s.text((x0, by1 + fs * 0.6), f"{shaded}/{parts} shaded", size=int(fs * 0.85), role="caption", associated=bar)
    return bar


def table_figure(s: Scene, box: Box, rng, level: int, fs: int) -> str:
    """K: tables, Venn diagrams and bar charts."""
    x0, y0, x1, y1 = _frame(box, 0.1)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        rows, cols = 3 + level, 3
        cw, rh = (x1 - x0) / cols, min((y1 - y0) / rows, fs * 2.2)
        frame = s.polygon([(x0, y0), (x1, y0), (x1, y0 + rows * rh), (x0, y0 + rows * rh)], kind="rectangle", importance="essential")
        for r in range(1, rows):
            s.segment((x0, y0 + r * rh), (x1, y0 + r * rh), kind="table_line", importance="supporting", width=2)
        for c in range(1, cols):
            s.segment((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * rh), kind="table_line", importance="supporting", width=2)
        heads = ["x", "y", "x + y"]
        for c in range(cols):
            s.label_near((x0 + (c + 0.5) * cw, y0 + 0.5 * rh), heads[c], (0, 0), size=int(fs * 0.8), role="table_cell", associated=frame, face="serif_italic")
        for r in range(1, rows):
            a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
            for c, val in enumerate((a, b, a + b)):
                s.label_near((x0 + (c + 0.5) * cw, y0 + (r + 0.5) * rh), str(val), (0, 0), size=int(fs * 0.8), role="table_cell", associated=frame)
        return frame
    if kind == 1:
        r = min(x1 - x0, y1 - y0) * 0.3
        cy = (y0 + y1) / 2
        ca = s.circle(((x0 + x1) / 2 - r * 0.55, cy), r)
        cb = s.circle(((x0 + x1) / 2 + r * 0.55, cy), r)
        s.relate("INTERSECTS", ca, cb)
        box_ = s.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], kind="rectangle")
        s.relate("INSIDE", ca, box_)
        s.relate("INSIDE", cb, box_)
        s.label_near(((x0 + x1) / 2 - r * 1.1, cy - r * 1.05), "A", (0, 0), size=fs, associated=ca, face="serif_italic")
        s.label_near(((x0 + x1) / 2 + r * 1.1, cy - r * 1.05), "B", (0, 0), size=fs, associated=cb, face="serif_italic")
        s.label_near((x0, y0), "U", (fs * 0.8, fs * 0.8), size=fs, associated=box_, face="serif_italic")
        if level >= 2:
            for dx, val in ((-r * 1.0, rng.integers(1, 20)), (0, rng.integers(1, 20)), (r * 1.0, rng.integers(1, 20))):
                s.label_near(((x0 + x1) / 2 + dx, cy), str(val), (0, 0), size=int(fs * 0.85), role="region_value", associated=box_)
        return ca
    # bar chart
    ox, oy = x0 + (x1 - x0) * 0.12, y1 - fs * 1.5
    xa = s.segment((ox, oy), (x1, oy), kind="axis")
    ya = s.segment((ox, oy), (ox, y0), kind="axis")
    s.relate("PERPENDICULAR", xa, ya)
    bars = 3 + level
    bw = (x1 - ox) / (bars * 1.6)
    first = None
    for i in range(bars):
        top = y0 + (oy - y0) * rng.uniform(0.1, 0.8)
        bx = ox + bw * (0.6 + 1.6 * i)
        b = s.polygon([(bx, top), (bx + bw, top), (bx + bw, oy), (bx, oy)], kind="bar")
        s.relate("ON", b, xa)
        s.label_near((bx + bw / 2, oy), "ABCDEFG"[i], (0, fs * 0.8), size=int(fs * 0.8), role="tick_label", associated=b)
        first = first or b
    return first


TEMPLATES = {
    "A_geometry_figure": geometry_figure,
    "C_graph": graph_figure,
    "D_coordinate_geometry": coordinate_figure,
    "E_measurement": measurement_figure,
    "F_shapes_polygons": shapes_figure,
    "G_angles": angles_figure,
    "H_circles": circles_figure,
    "I_transformations": transformation_figure,
    "J_fractions_number_lines": number_line_figure,
    "K_tables_diagrams": table_figure,
}

DIAGRAM_TYPES = {
    "A_geometry_figure": "triangle_quadrilateral_geometry",
    "B_worksheet_geometry": "worksheet_with_geometry_questions",
    "C_graph": "function_graph",
    "D_coordinate_geometry": "coordinate_plane",
    "E_measurement": "measurement_diagram",
    "F_shapes_polygons": "polygons",
    "G_angles": "angles",
    "H_circles": "circle_geometry",
    "I_transformations": "transformation_on_grid",
    "J_fractions_number_lines": "number_line_or_fraction",
    "K_tables_diagrams": "table_or_chart",
    "L_mixed_worksheet": "mixed_worksheet",
}
