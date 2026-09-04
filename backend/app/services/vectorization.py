import cv2
import numpy as np

def extract_shapes(binary_image: np.ndarray, edge_sensitivity: int = 50) -> list[dict]:
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, max(20, 110 - edge_sensitivity), minLineLength=max(20, 90 - edge_sensitivity), maxLineGap=12)
    shapes: list[dict] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            shapes.append({"type": "line", "points": [(int(x1), int(y1)), (int(x2), int(y2))]})
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        approximation = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        points = [(int(point[0][0]), int(point[0][1])) for point in approximation]
        if len(points) >= 3:
            shapes.append({"type": "contour", "points": points})
    return shapes

def shapes_to_svg(shapes: list[dict], width: int, height: int) -> str:
    elements: list[str] = []
    for shape in shapes:
        points = shape["points"]
        if shape["type"] == "line":
            (x1, y1), (x2, y2) = points
            elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />')
        else:
            elements.append('<polygon points="' + ' '.join(f"{x},{y}" for x, y in points) + '" />')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Raw geometry vector preview">'
            '<rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + ''.join(elements) + '</g></svg>')
