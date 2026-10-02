"""Braille marker layout: collision-safe placement + reading order.

Reading order follows the top-to-bottom, left-to-right convention used in
tactile graphics, so a learner scans markers in a natural sequence.
"""
import math
from math import hypot
from typing import Any

from app.models.geometry import DetectedElement, GeometryType
from app.services.segment_geometry import rect_around, rect_gap, segment_to_rect_distance
from app.services.tactile_rules import TACTILE_RULES
from app.services.tactile_svg import braille_text_extent, page_layout


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
        points = shape.get("points")
        if not points:
            continue
        if shape["type"] == "line":
            segments.append((points[0], points[1]))
        elif len(points) > 1:
            segments.extend((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))
    return segments


def _on_page(position: tuple[float, float], braille: str, bounds: tuple[int, int] | None) -> bool:
    if bounds is None:
        return True
    half_width, half_height = (value / 2 for value in braille_text_extent(braille, *bounds))
    return half_width <= position[0] <= bounds[0] - half_width and half_height <= position[1] <= bounds[1] - half_height


def _clamp_to_page(position: tuple[int, int], braille: str, bounds: tuple[int, int] | None) -> tuple[int, int]:
    if bounds is None:
        return position
    half_width, half_height = (value / 2 for value in braille_text_extent(braille, *bounds))
    x = min(max(position[0], math.ceil(half_width)), math.floor(bounds[0] - half_width))
    y = min(max(position[1], math.ceil(half_height)), math.floor(bounds[1] - half_height))
    return (x, y)


def place_braille_markers(
    labels: list[dict],
    shapes: list[dict],
    minimum_clearance: int = None,
    bounds: tuple[int, int] | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Move labels off geometry, apart from other markers, and add reading order.

    ``bounds`` is the page ``(width, height)``; when given, the whole rendered
    braille string must fit on the page, so no cell is clipped at the edge, and
    clearance is measured in millimetres from the embossed cell block.
    """
    if bounds is not None and minimum_clearance is None:
        return _place_on_page(labels, shapes, bounds)
    if minimum_clearance is None:
        minimum_clearance = int(TACTILE_RULES.braille_to_line_clearance_px)
    segments = _line_segments(shapes)
    positioned: list[dict] = []
    for label in labels:
        desired = label["desired_position"]
        braille = str(label.get("braille") or label.get("text") or "")
        position = _clamp_to_page(desired, braille, bounds)
        maximum_search = minimum_clearance * 5 + 1
        step = max(1, minimum_clearance // 2)
        for radius in range(0, maximum_search + 1, step):
            for direction_x, direction_y in _SEARCH_DIRECTIONS:
                proposed = (desired[0] + direction_x * radius, desired[1] + direction_y * radius)
                if not _on_page(proposed, braille, bounds):
                    continue
                clear_of_lines = all(_point_to_segment_distance(proposed, *segment) >= minimum_clearance for segment in segments)
                clear_of_markers = all(hypot(proposed[0] - prior["position"][0], proposed[1] - prior["position"][1]) >= minimum_clearance for prior in positioned)
                if clear_of_lines and clear_of_markers:
                    position = proposed
                    break
            else:
                continue
            break
        positioned.append(_positioned(label, desired, position))

    return _with_reading_order(positioned)


_SEARCH_DIRECTIONS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
# How far (as a multiple of the cell block's larger side) a marker may travel
# from its printed position before placement gives up and leaves it for QA.
_MAX_SEARCH_BLOCKS = 4


def _positioned(label: dict, desired: tuple[int, int], position: tuple[int, int]) -> dict:
    return {
        **label,
        "position": position,
        "offset": (position[0] - desired[0], position[1] - desired[1]),
        "collision_adjusted": position != desired,
    }


def _place_on_page(labels: list[dict], shapes: list[dict], bounds: tuple[int, int]) -> list[dict]:
    mm_per_px = page_layout(*bounds).mm_per_px
    line_clearance = TACTILE_RULES.braille_to_line_clearance_mm / mm_per_px
    marker_spacing = TACTILE_RULES.braille_to_braille_spacing_mm / mm_per_px
    segments = _line_segments(shapes)
    placed_rects: list[tuple[float, float, float, float]] = []
    positioned: list[dict] = []
    for label in labels:
        desired = label["desired_position"]
        braille = str(label.get("braille") or label.get("text") or "")
        width, height = braille_text_extent(braille, *bounds)
        position = _clamp_to_page(desired, braille, bounds)
        step = max(1, math.ceil(min(width, height) / 4))
        maximum_search = math.ceil(max(width, height) * _MAX_SEARCH_BLOCKS)
        found = False
        for radius in range(0, maximum_search + 1, step):
            for direction_x, direction_y in _SEARCH_DIRECTIONS:
                proposed = (desired[0] + direction_x * radius, desired[1] + direction_y * radius)
                if not _on_page(proposed, braille, bounds):
                    continue
                rect = rect_around(proposed, width, height)
                if all(segment_to_rect_distance(a, b, rect) >= line_clearance for a, b in segments) and all(
                    rect_gap(rect, prior) >= marker_spacing for prior in placed_rects
                ):
                    position = proposed
                    found = True
                    break
            if found:
                break
        placed_rects.append(rect_around(position, width, height))
        positioned.append(_positioned(label, desired, position))
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