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


def _distance_to_element(point: tuple[float, float], element: DetectedElement) -> float:
    points = _element_points(element)
    if not points:
        return float("inf")
    return min(math.hypot(point[0] - p[0], point[1] - p[1]) for p in points)


def associate_label(label: DetectedElement, elements: list[DetectedElement], max_distance: float = 80.0) -> dict:
    """Associate an OCR ``label`` with the best candidate geometry element.

    Returns a dict with:
      - ``target_id`` / ``target_none``: the best candidate element id (or None)
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

    distances = [(e, _distance_to_element(position, e)) for e in candidates]
    distances.sort(key=lambda pair: pair[1])
    best, best_distance = distances[0]

    # Confidence falls off with distance; far labels are not confidently bound.
    confidence = max(0.05, 1.0 - (best_distance / max_distance))
    needs_review = best_distance > max_distance / 2 or confidence < 0.5

    if best_distance > max_distance:
        # Retain text but do not claim an association.
        return _no_association(label_position=position)

    detail = f"Text {label.geometry.get('text', '')} associated with {best.type.value} {best.id}."
    if needs_review:
        detail += " Association may be uncertain; review suggested."
    return {
        "target_id": best.id,
        "confidence": round(confidence, 3),
        "confidence_level": classify_confidence(confidence).value,
        "needs_review": needs_review,
        "distance": round(best_distance, 2),
        "explanation": detail,
    }


def _no_association(label_position: tuple | None = None) -> dict:
    return {
        "target_id": None,
        "confidence": 0.0,
        "confidence_level": ConfidenceLevel.LOW.value,
        "needs_review": True,
        "distance": None,
        "explanation": "No reliable geometry association found for label.",
    }