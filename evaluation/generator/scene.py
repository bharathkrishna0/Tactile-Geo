"""A drawing surface that records exact ground truth for everything it draws.

Every primitive is drawn onto the visible page and, when it is mathematical
geometry, onto a parallel foreground mask. The annotation is written at draw
time from the same coordinates, so it is exact by construction rather than
inferred afterwards. Geometric degradations (skew, perspective, rescale) are a
single homography applied to the page, the mask and every recorded coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype")
FONTS = {
    "sans": FONT_DIR / "dejavu" / "DejaVuSans.ttf",
    "sans_bold": FONT_DIR / "dejavu" / "DejaVuSans-Bold.ttf",
    "serif": FONT_DIR / "liberation" / "LiberationSerif-Regular.ttf",
    "serif_italic": FONT_DIR / "liberation" / "LiberationSerif-Italic.ttf",
    "serif_bold": FONT_DIR / "liberation" / "LiberationSerif-Bold.ttf",
}

GEOMETRY_TYPES = {
    "line_segment", "ray", "arrow", "axis", "dimension_line", "tick", "grid_line",
    "circle", "ellipse", "arc", "angle_marker", "right_angle_marker",
    "triangle", "rectangle", "polygon", "curve", "bar", "table_line", "number_line",
}
DISTRACTOR_TYPES = {"handwriting", "decoration"}
OMITTABLE_TEXT_ROLES = {"title", "question_text", "instruction", "name_line", "decoration_text"}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS[name]), size)


def circle_outline(center, radius, start=0.0, end=360.0, steps=72):
    count = max(8, int(steps * abs(end - start) / 360))
    return [
        [center[0] + radius * math.cos(math.radians(start + (end - start) * i / count)),
         center[1] + radius * math.sin(math.radians(start + (end - start) * i / count))]
        for i in range(count + 1)
    ]


def ellipse_outline(center, axes, angle=0.0, steps=72):
    a, b = axes
    t = math.radians(angle)
    out = []
    for i in range(steps + 1):
        u = 2 * math.pi * i / steps
        x, y = a * math.cos(u), b * math.sin(u)
        out.append([center[0] + x * math.cos(t) - y * math.sin(t), center[1] + x * math.sin(t) + y * math.cos(t)])
    return out


@dataclass
class Scene:
    width: int
    height: int
    stroke: int = 4
    ink: int = 0
    objects: list[dict] = field(default_factory=list)
    points: list[dict] = field(default_factory=list)
    labels: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    dimensions: list[dict] = field(default_factory=list)
    dash_pattern: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.page = Image.new("L", (self.width, self.height), 255)
        self.mask = Image.new("L", (self.width, self.height), 0)
        self.draw = ImageDraw.Draw(self.page)
        self.mdraw = ImageDraw.Draw(self.mask)
        self._counters: dict[str, int] = {}

    def new_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]}"

    # ------------------------------------------------------------- recording
    def _object(self, kind: str, geometry: dict, outline: list, importance: str, **extra) -> str:
        oid = self.new_id(kind)
        self.objects.append({
            "id": oid,
            "type": kind,
            "geometry": geometry,
            "_outline": [list(map(float, p)) for p in outline],
            "semantic_importance": importance,
            "preserve_in_tactile": importance != "omittable",
            **extra,
        })
        return oid

    def relate(self, kind: str, source: str, target: str) -> None:
        self.relationships.append({"type": kind, "source": source, "target": target})

    # ------------------------------------------------------------ primitives
    def _line(self, a, b, width, ink, dashed, mask):
        if dashed or self.dash_pattern:
            length = math.dist(a, b) or 1.0
            dash, gap = self.dash_pattern or (max(10.0, width * 3.5), max(10.0, width * 3.5))
            period = dash + gap
            n = max(1, int(length // period))
            for i in range(n + 1):
                t0, t1 = (i * period) / length, min(1.0, (i * period + dash) / length)
                if t0 >= 1:
                    break
                p = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                q = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                self.draw.line([p, q], fill=ink, width=width)
                if mask:
                    self.mdraw.line([p, q], fill=255, width=width)
            return
        self.draw.line([tuple(a), tuple(b)], fill=ink, width=width)
        if mask:
            self.mdraw.line([tuple(a), tuple(b)], fill=255, width=width)

    def segment(self, a, b, kind="line_segment", importance="essential", dashed=False,
                width=None, ink=None, mask=True, **extra) -> str:
        width = width or self.stroke
        ink = self.ink if ink is None else ink
        self._line(a, b, width, ink, dashed, mask)
        geometry = {"kind": "segment", "points": [list(a), list(b)]}
        if dashed:
            extra["dashed"] = True
        return self._object(kind, geometry, [a, b], importance, **extra)

    def arrowhead(self, tip, tail, size=None, ink=None, mask=True) -> None:
        size = size or self.stroke * 4
        ink = self.ink if ink is None else ink
        angle = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
        left = (tip[0] - size * math.cos(angle - 0.4), tip[1] - size * math.sin(angle - 0.4))
        right = (tip[0] - size * math.cos(angle + 0.4), tip[1] - size * math.sin(angle + 0.4))
        self.draw.polygon([tuple(tip), left, right], fill=ink)
        if mask:
            self.mdraw.polygon([tuple(tip), left, right], fill=255)

    def arrow(self, a, b, kind="arrow", both=False, importance="essential", width=None, **extra) -> str:
        oid = self.segment(a, b, kind=kind, importance=importance, width=width, **extra)
        self.arrowhead(b, a)
        if both:
            self.arrowhead(a, b)
        self.objects[-1]["geometry"]["arrowheads"] = 2 if both else 1
        return oid

    def polygon(self, points, kind="polygon", importance="essential", width=None, ink=None,
                fill=None, dashed=False, mask=True, **extra) -> str:
        width = width or self.stroke
        ink = self.ink if ink is None else ink
        pts = [tuple(map(float, p)) for p in points]
        if fill is not None:
            self.draw.polygon(pts, fill=fill)
        for i in range(len(pts)):
            self._line(pts[i], pts[(i + 1) % len(pts)], width, ink, dashed, mask)
        geometry = {"kind": "polygon", "points": [list(p) for p in pts]}
        return self._object(kind, geometry, pts + [pts[0]], importance, **extra)

    def polyline(self, points, kind="curve", importance="essential", width=None, ink=None, mask=True, **extra) -> str:
        width = width or self.stroke
        ink = self.ink if ink is None else ink
        pts = [tuple(map(float, p)) for p in points]
        self.draw.line(pts, fill=ink, width=width, joint="curve")
        if mask:
            self.mdraw.line(pts, fill=255, width=width, joint="curve")
        return self._object(kind, {"kind": "polyline", "points": [list(p) for p in pts]}, pts, importance, **extra)

    def circle(self, center, radius, kind="circle", importance="essential", width=None, ink=None, mask=True, dashed=False, **extra) -> str:
        width = width or self.stroke
        ink = self.ink if ink is None else ink
        box = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        if dashed:
            outline = circle_outline(center, radius, steps=48)
            for i in range(0, len(outline) - 1, 2):
                self.draw.line([tuple(outline[i]), tuple(outline[i + 1])], fill=ink, width=width)
                if mask:
                    self.mdraw.line([tuple(outline[i]), tuple(outline[i + 1])], fill=255, width=width)
        else:
            self.draw.ellipse(box, outline=ink, width=width)
            if mask:
                self.mdraw.ellipse(box, outline=255, width=width)
        geometry = {"kind": "circle", "center": list(center), "radius": float(radius)}
        return self._object(kind, geometry, circle_outline(center, radius), importance, **extra)

    def ellipse(self, center, axes, importance="essential", width=None, **extra) -> str:
        width = width or self.stroke
        box = [center[0] - axes[0], center[1] - axes[1], center[0] + axes[0], center[1] + axes[1]]
        self.draw.ellipse(box, outline=self.ink, width=width)
        self.mdraw.ellipse(box, outline=255, width=width)
        geometry = {"kind": "ellipse", "center": list(center), "axes": [float(axes[0]), float(axes[1])], "angle": 0.0}
        return self._object("ellipse", geometry, ellipse_outline(center, axes), importance, **extra)

    def arc(self, center, radius, start, end, kind="arc", importance="essential", width=None, **extra) -> str:
        width = width or max(2, self.stroke - 1)
        box = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        self.draw.arc(box, start, end, fill=self.ink, width=width)
        self.mdraw.arc(box, start, end, fill=255, width=width)
        geometry = {"kind": "arc", "center": list(center), "radius": float(radius), "start_deg": start, "end_deg": end}
        return self._object(kind, geometry, circle_outline(center, radius, start, end), importance, **extra)

    def right_angle_marker(self, vertex, dir_a, dir_b, size=22, importance="supporting") -> str:
        ua = _unit(dir_a)
        ub = _unit(dir_b)
        p1 = (vertex[0] + ua[0] * size, vertex[1] + ua[1] * size)
        p3 = (vertex[0] + ub[0] * size, vertex[1] + ub[1] * size)
        p2 = (p1[0] + ub[0] * size, p1[1] + ub[1] * size)
        w = max(2, self.stroke - 2)
        self.draw.line([p1, p2, p3], fill=self.ink, width=w)
        self.mdraw.line([p1, p2, p3], fill=255, width=w)
        geometry = {"kind": "polyline", "points": [list(p1), list(p2), list(p3)], "vertex": list(vertex)}
        return self._object("right_angle_marker", geometry, [vertex, p1, p2, p3], importance)

    def point(self, xy, role="vertex", dot=False, importance="essential", radius=6) -> str:
        if dot:
            box = [xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius]
            self.draw.ellipse(box, fill=self.ink)
            self.mdraw.ellipse(box, fill=255)
        pid = self.new_id("point")
        self.points.append({"id": pid, "x": float(xy[0]), "y": float(xy[1]), "role": role,
                            "drawn_as_dot": dot, "semantic_importance": importance})
        return pid

    def text(self, xy, text, size=28, role="vertex_label", associated=None, face="sans",
             importance=None, ink=None, anchor="la") -> str:
        f = font(face, size)
        ink = self.ink if ink is None else ink
        self.draw.text(tuple(xy), text, font=f, fill=ink, anchor=anchor)
        left, top, right, bottom = self.draw.textbbox(tuple(xy), text, font=f, anchor=anchor)
        lid = self.new_id("label")
        if importance is None:
            importance = "omittable" if role in OMITTABLE_TEXT_ROLES else "essential"
        self.labels.append({
            "id": lid, "text": text, "role": role, "associated_object": associated,
            "font_px": size, "_box": [[left, top], [right, top], [right, bottom], [left, bottom]],
            "semantic_importance": importance,
            "preserve_in_tactile": importance != "omittable",
        })
        if associated:
            self.relate("LABELS", lid, associated)
        return lid

    def label_near(self, anchor, text, offset, size=28, role="vertex_label", associated=None, face="sans", **kw) -> str:
        return self.text((anchor[0] + offset[0], anchor[1] + offset[1]), text, size=size, role=role,
                         associated=associated, face=face, anchor="mm", **kw)

    def distractor(self, kind: str, outline: list, note: str) -> str:
        geometry = {"kind": "polyline", "points": [list(map(float, p)) for p in outline]}
        return self._object(kind, geometry, outline, "omittable", note=note)

    def dimension(self, value: str, unit: str, object_id: str, label_id: str) -> None:
        self.dimensions.append({"id": self.new_id("dim"), "value": value, "unit": unit,
                                "object": object_id, "label": label_id})
        self.relate("DIMENSION_OF", label_id, object_id)

    # ---------------------------------------------------------------- export
    def export(self, homography: np.ndarray, out_w: int, out_h: int) -> dict:
        """Ground truth in output-image coordinates after ``homography``."""

        def tp(p):
            v = homography @ np.array([p[0], p[1], 1.0])
            return [round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)]

        def local_scale(p):
            q = tp(p)
            dx = tp((p[0] + 1, p[1]))
            dy = tp((p[0], p[1] + 1))
            return math.sqrt(abs((dx[0] - q[0]) * (dy[1] - q[1]) - (dx[1] - q[1]) * (dy[0] - q[0])))

        def bbox(points):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return [round(min(xs), 1), round(min(ys), 1), round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)]

        objects = []
        for obj in self.objects:
            g = dict(obj["geometry"])
            if "points" in g:
                g["points"] = [tp(p) for p in g["points"]]
            if "vertex" in g:
                g["vertex"] = tp(g["vertex"])
            if "center" in g:
                s = local_scale(g["center"])
                g["center"] = tp(g["center"])
                if "radius" in g:
                    g["radius"] = round(g["radius"] * s, 2)
                if "axes" in g:
                    g["axes"] = [round(a * s, 2) for a in g["axes"]]
            outline = [tp(p) for p in obj["_outline"]]
            record = {k: v for k, v in obj.items() if not k.startswith("_")}
            record["geometry"] = g
            record["bbox"] = bbox(outline)
            if g["kind"] in ("circle", "ellipse", "arc"):
                record["outline"] = outline
            objects.append(record)
        points = [{**p, "x": tp((p["x"], p["y"]))[0], "y": tp((p["x"], p["y"]))[1]} for p in self.points]
        labels = []
        for label in self.labels:
            corners = [tp(p) for p in label["_box"]]
            record = {k: v for k, v in label.items() if not k.startswith("_")}
            record["bbox"] = bbox(corners)
            labels.append(record)
        return {
            "width": out_w,
            "height": out_h,
            "objects": objects,
            "points": points,
            "labels": labels,
            "relationships": self.relationships,
            "dimensions": self.dimensions,
        }


def _unit(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)
