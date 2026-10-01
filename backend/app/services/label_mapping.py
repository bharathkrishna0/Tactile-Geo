from math import dist

from app.services.ocr import OcrDetection


def _point_to_segment_distance(point: tuple[float, float], start: tuple[int, int], end: tuple[int, int]) -> tuple[float, tuple[int, int]]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return dist(point, start), start
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    projected = (round(start[0] + ratio * dx), round(start[1] + ratio * dy))
    return dist(point, projected), projected


def _geometry_candidates(shapes: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    vertices: list[tuple[int, int]] = []
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for shape in shapes:
        points = shape.get("points")
        if not points:
            continue
        vertices.extend(points)
        if shape["type"] == "line":
            edges.append((points[0], points[1]))
        elif len(points) > 1:
            edges.extend((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))
    return vertices, edges


def map_label_to_geometry(detection: OcrDetection, shapes: list[dict]) -> dict:
    vertices, edges = _geometry_candidates(shapes)
    if not vertices:
        raise ValueError("Cannot map labels because no vector geometry was detected.")
    center = (
        sum(point[0] for point in detection.bbox) / len(detection.bbox),
        sum(point[1] for point in detection.bbox) / len(detection.bbox),
    )
    vertex_distance, vertex = min((dist(center, vertex), vertex) for vertex in vertices)
    edge_distance, edge_anchor = min((_point_to_segment_distance(center, *edge) for edge in edges), default=(float("inf"), (0, 0)))
    anchor_type, anchor, anchor_distance = ("vertex", vertex, vertex_distance) if vertex_distance <= edge_distance else ("edge", edge_anchor, edge_distance)
    return {
        "text": detection.text,
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "anchor_type": anchor_type,
        "anchor": anchor,
        "anchor_distance": anchor_distance,
        "desired_position": (round(center[0]), round(center[1])),
    }
