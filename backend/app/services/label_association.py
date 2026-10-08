"""Label-to-geometry association.

Improves on ``label_mapping.py`` by associating an OCR label element with a
specific semantic geometry element (point, segment, line, circle, polygon)
rather than just the nearest raw vertex/edge. Association is not forced:
when uncertain, the text is retained with the candidate association exposed at
lower confidence so the teacher can correct it later.
"""
from __future__ import annotations

import math

from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    GeometryType,
    classify_confidence,
)


def _element_points(element: DetectedElement) -> list[tuple[float, float]]:
    geo = element.geometry
    points: list[tuple[float, float]] = []
    if "points" in geo:
        points.extend((float(p[0]), float(p[1])) for p in geo["points"])
    if "start" in geo:
        points.append((float(geo["start"][0]), float(geo["start"][1])))
    if "end" in geo:
        points.append((float(geo["end"][0]), float(geo["end"][1])))
    if "center" in geo:
        points.append((float(geo["center"][0]), float(geo["center"][1])))
    if "position" in geo:
        points.append((float(geo["position"][0]), float(geo["position"][1])))
    if "vertex" in geo:
        points.append((float(geo["vertex"][0]), float(geo["vertex"][1])))
    if "arms" in geo:
        for arm in geo["arms"]:
            points.append((float(arm[0]), float(arm[1])))
    return points


OUTLINE_SAMPLES = 72


def outline_points(element: DetectedElement) -> list[tuple[float, float]]:
    """Points on the drawn outline of ``element``, for distance checks.

    Circles and ellipses are sampled along their rim: a label beside the rim is
    close to the shape even though it is far from the centre.
    """
    geo = element.geometry
    if element.type is GeometryType.CIRCLE and geo.get("center") and geo.get("radius"):
        cx, cy = (float(v) for v in geo["center"])
        rx = ry = float(geo["radius"])
        angle = 0.0
    elif element.type is GeometryType.ELLIPSE and geo.get("center") and geo.get("semi_axes"):
        cx, cy = (float(v) for v in geo["center"])
        rx, ry = (float(v) for v in geo["semi_axes"])
        angle = math.radians(float(geo.get("angle") or 0.0))
    else:
        return _element_points(element)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for k in range(OUTLINE_SAMPLES):
        t = 2 * math.pi * k / OUTLINE_SAMPLES
        x, y = rx * math.cos(t), ry * math.sin(t)
        points.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return points


def _element_center(element: DetectedElement) -> tuple[float, float]:
    points = _element_points(element)
    if not points:
        return (0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _label_position(label: DetectedElement) -> tuple[float, float]:
    geo = label.geometry
    if "position" in geo:
        return (float(geo["position"][0]), float(geo["position"][1]))
    return _element_center(label)


CLOSED_TYPES = (GeometryType.TRIANGLE, GeometryType.RECTANGLE, GeometryType.POLYGON)
ROUND_TYPES = (GeometryType.CIRCLE, GeometryType.ELLIPSE)
POINT_PREFERENCE_PX = 40.0
TICK_PREFERENCE_PX = 40.0
TICK_ALIGNMENT_DEG = 30.0


def _segment_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq))
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def _segments(element: DetectedElement) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    geo = element.geometry
    if "start" in geo and "end" in geo:
        return [((float(geo["start"][0]), float(geo["start"][1])), (float(geo["end"][0]), float(geo["end"][1])))]
    if element.type in CLOSED_TYPES and len(geo.get("points") or []) >= 3:
        pts = [(float(p[0]), float(p[1])) for p in geo["points"]]
        return list(zip(pts, pts[1:] + pts[:1]))
    return []


def _distance_to_element(point: tuple[float, float], element: DetectedElement) -> float:
    segments = _segments(element)
    if segments:
        return min(_segment_distance(point, a, b) for a, b in segments)
    points = outline_points(element)
    if not points:
        return float("inf")
    return min(math.hypot(point[0] - p[0], point[1] - p[1]) for p in points)


def _enclosed_area(point: tuple[float, float], element: DetectedElement) -> float | None:
    """Area of ``element`` when it is a closed shape containing ``point``."""
    geo = element.geometry
    if element.type in ROUND_TYPES and geo.get("center"):
        cx, cy = (float(v) for v in geo["center"])
        if element.type is GeometryType.CIRCLE and geo.get("radius"):
            r = float(geo["radius"])
            return math.pi * r * r if math.hypot(point[0] - cx, point[1] - cy) < r else None
        if geo.get("semi_axes"):
            rx, ry = (float(v) for v in geo["semi_axes"])
            a = math.radians(float(geo.get("angle") or 0.0))
            x = (point[0] - cx) * math.cos(a) + (point[1] - cy) * math.sin(a)
            y = -(point[0] - cx) * math.sin(a) + (point[1] - cy) * math.cos(a)
            return math.pi * rx * ry if rx > 0 and ry > 0 and (x / rx) ** 2 + (y / ry) ** 2 < 1 else None
        return None
    if element.type not in CLOSED_TYPES or len(geo.get("points") or []) < 3:
        return None
    pts = [(float(p[0]), float(p[1])) for p in geo["points"]]
    inside = False
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > point[1]) != (y2 > point[1]) and point[0] < x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    if not inside:
        return None
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))) / 2


def _box_distance(point: tuple[float, float], box: tuple | None, fallback: tuple[float, float]) -> float:
    if not box:
        return math.hypot(point[0] - fallback[0], point[1] - fallback[1])
    x, y, w, h = (float(v) for v in box)
    dx = max(x - point[0], 0.0, point[0] - (x + w))
    dy = max(y - point[1], 0.0, point[1] - (y + h))
    return math.hypot(dx, dy)


def _tick_in_line(position: tuple[float, float], tick: DetectedElement) -> float | None:
    """Distance from a tick's nearer end to text continuing in the tick's direction."""
    if tick.semantic_properties.get("role") != "tick" or not _segments(tick):
        return None
    (a, b), = _segments(tick)
    near, far = (a, b) if math.dist(a, position) <= math.dist(b, position) else (b, a)
    direction = math.atan2(near[1] - far[1], near[0] - far[0])
    towards = math.atan2(position[1] - near[1], position[0] - near[0])
    off = abs((math.degrees(towards - direction) + 180) % 360 - 180)
    return math.dist(near, position) if off <= TICK_ALIGNMENT_DEG else None


def _choose(label: DetectedElement, position: tuple[float, float], candidates: list[DetectedElement]) -> tuple[DetectedElement, float, str]:
    """Best target and the evidence for it, most specific evidence first.

    A labelled point beside the text names that point; text continuing a tick
    names the tick; text inside a closed shape (a table cell, a region) names
    the smallest shape around it; otherwise the text names the geometry whose
    drawn outline is nearest.
    """
    distances = sorted(((e, _distance_to_element(position, e)) for e in candidates), key=lambda pair: pair[1])
    points = sorted(
        (
            (_box_distance((float(e.geometry["position"][0]), float(e.geometry["position"][1])), label.bbox, position), e)
            for e in candidates if e.type is GeometryType.POINT and e.geometry.get("position")
        ),
        key=lambda pair: pair[0],
    )
    if points and points[0][0] <= POINT_PREFERENCE_PX:
        return points[0][1], points[0][0], "point"
    ticks = sorted(
        ((d, e) for e in candidates if (d := _tick_in_line(position, e)) is not None and d <= TICK_PREFERENCE_PX),
        key=lambda pair: pair[0],
    )
    if ticks:
        return ticks[0][1], ticks[0][0], "tick"
    enclosing = [(area, e) for e in candidates if (area := _enclosed_area(position, e))]
    if enclosing:
        element = min(enclosing, key=lambda pair: pair[0])[1]
        return element, 0.0, "inside"
    element, distance = distances[0]
    return element, distance, "nearest_outline"


REASON_TEXT = {
    "point": "the labelled point beside it",
    "tick": "the tick it continues",
    "inside": "the smallest closed shape around it",
    "nearest_outline": "the nearest drawn outline",
}


def associate_label(label: DetectedElement, elements: list[DetectedElement], max_distance: float = 80.0) -> dict:
    """Associate an OCR ``label`` with the best candidate geometry element.

    Returns a dict with:
      - ``target_id``: the chosen element id (or None)
      - ``target_type``: the chosen element's geometry type
      - ``reason``: the evidence used (``point``, ``tick``, ``inside`` or ``nearest_outline``)
      - ``confidence``: association confidence (0..1)
      - ``confidence_level``
      - ``needs_review``: True when the association is uncertain
      - ``distance``: distance from the label to the candidate
      - ``explanation``: human-readable association note
    """
    position = _label_position(label)
    candidates = [e for e in elements if e.type is not GeometryType.TEXT_LABEL]
    if not candidates:
        return _no_association()

    best, best_distance, reason = _choose(label, position, candidates)

    # Confidence falls off with distance; far labels are not confidently bound.
    confidence = max(0.05, 1.0 - (best_distance / max_distance))
    needs_review = best_distance > max_distance / 2 or confidence < 0.5

    if best_distance > max_distance:
        # Retain text but do not claim an association.
        return _no_association(label_position=position)

    detail = (
        f"Text {label.geometry.get('text', '')} associated with {best.type.value} {best.id}: "
        f"{REASON_TEXT[reason]}."
    )
    if needs_review:
        detail += " Association may be uncertain; review suggested."
    return {
        "target_id": best.id,
        "target_type": best.type.value,
        "reason": reason,
        "confidence": round(confidence, 3),
        "confidence_level": classify_confidence(confidence).value,
        "needs_review": needs_review,
        "distance": round(best_distance, 2),
        "explanation": detail,
    }


def _no_association(label_position: tuple | None = None) -> dict:
    return {
        "target_id": None,
        "target_type": None,
        "reason": None,
        "confidence": 0.0,
        "confidence_level": ConfidenceLevel.LOW.value,
        "needs_review": True,
        "distance": None,
        "explanation": "No reliable geometry association found for label.",
    }
