"""Braille marker layout: collision-safe placement + reading order.

Reading order follows the top-to-bottom, left-to-right convention used in
tactile graphics, so a learner scans markers in a natural sequence.
"""
from math import hypot
from typing import Any

from app.models.geometry import DetectedElement, GeometryType
from app.services.tactile_rules import TACTILE_RULES


def _point_to_segment_distance(point: tuple[int, int], start: tuple[int, int], end: tuple[int, int]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def _line_segments(shapes: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    segments = []
    for shape in shapes:
        points = shape["points"]
        if shape["type"] == "line":
            segments.append((points[0], points[1]))
        elif len(points) > 1:
            segments.extend((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))
    return segments


def place_braille_markers(labels: list[dict], shapes: list[dict], minimum_clearance: int = None, **kwargs: Any) -> list[dict]:
    """Move labels off geometry, apart from other markers, and add reading order."""
    if minimum_clearance is None:
        minimum_clearance = int(TACTILE_RULES.braille_to_line_clearance_px)
    segments = _line_segments(shapes)
    positioned: list[dict] = []
    candidates = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    for label in labels:
        desired = label["desired_position"]
        position = desired
        maximum_search = minimum_clearance * 5 + 1
        step = max(1, minimum_clearance // 2)
        for radius in range(0, maximum_search + 1, step):
            for direction_x, direction_y in candidates:
                proposed = (desired[0] + direction_x * radius, desired[1] + direction_y * radius)
                clear_of_lines = all(_point_to_segment_distance(proposed, *segment) >= minimum_clearance for segment in segments)
                clear_of_markers = all(hypot(proposed[0] - prior["position"][0], proposed[1] - prior["position"][1]) >= minimum_clearance for prior in positioned)
                if clear_of_lines and clear_of_markers:
                    position = proposed
                    break
            else:
                continue
            break
        positioned.append({
            **label,
            "position": position,
            "offset": (position[0] - desired[0], position[1] - desired[1]),
            "collision_adjusted": position != desired,
        })

    return _with_reading_order(positioned)


def _with_reading_order(markers: list[dict]) -> list[dict]:
    """Annotate markers with a reading order (top-to-bottom, then left-to-right).

    The input list order is preserved; each marker gains a ranking attribute so
    downstream consumers can lay markers out in reading sequence without the
    function mutating the caller's ordering contract.
    """
    ranked = sorted(enumerate(markers), key=lambda pair: (pair[1]["position"][1], pair[1]["position"][0]))
    for rank, (original_index, _marker) in enumerate(ranked):
        markers[original_index]["reading_order"] = rank + 1
        markers[original_index]["reading_sequence"] = "top_to_bottom"
    return markers


def place_braille_markers_from_elements(labels: list[dict], elements: list[DetectedElement], minimum_clearance: int = None) -> list[dict]:
    """Variant that reuses semantic elements for collision geometry."""
    if minimum_clearance is None:
        minimum_clearance = int(TACTILE_RULES.braille_to_line_clearance_px)
    positioned: list[dict] = []
    candidates = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    geometry = [e for e in elements if e.type is not GeometryType.TEXT_LABEL]
    for label in labels:
        desired = label["desired_position"]
        position = desired
        maximum_search = minimum_clearance * 5 + 1
        step = max(1, minimum_clearance // 2)
        for radius in range(0, maximum_search + 1, step):
            for direction_x, direction_y in candidates:
                proposed = (desired[0] + direction_x * radius, desired[1] + direction_y * radius)
                clear_of_geometry = all(_marker_clear_of_element(proposed, element, minimum_clearance) for element in geometry)
                clear_of_markers = all(hypot(proposed[0] - prior["position"][0], proposed[1] - prior["position"][1]) >= minimum_clearance for prior in positioned)
                if clear_of_geometry and clear_of_markers:
                    position = proposed
                    break
            else:
                continue
            break
        positioned.append({
            **label,
            "position": position,
            "offset": (position[0] - desired[0], position[1] - desired[1]),
            "collision_adjusted": position != desired,
        })
    return _with_reading_order(positioned)


def _marker_clear_of_element(marker: tuple[int, int], element: DetectedElement, clearance: float) -> bool:
    geo = element.geometry
    if element.type is GeometryType.LINE_SEGMENT:
        return _point_to_segment_distance(marker, geo["start"], geo["end"]) >= clearance
    points = []
    if "points" in geo:
        points = geo["points"]
    elif "center" in geo:
        return hypot(marker[0] - geo["center"][0], marker[1] - geo["center"][1]) >= clearance + float(geo.get("radius", 0))
    elif "position" in geo:
        return hypot(marker[0] - geo["position"][0], marker[1] - geo["position"][1]) >= clearance
    if len(points) > 1:
        return all(_point_to_segment_distance(marker, points[i], points[(i + 1) % len(points)]) >= clearance for i in range(len(points)))
    return True