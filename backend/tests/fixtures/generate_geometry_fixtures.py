"""Creates the small raster worksheet fixtures used by the Milestone 1 tests."""
from pathlib import Path

import cv2
import numpy as np


FIXTURE_DIRECTORY = Path(__file__).parent


def create_fixtures() -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    triangle = np.full((240, 240, 3), 255, dtype=np.uint8)
    triangle_points = np.array([[120, 35], [35, 195], [205, 195]], dtype=np.int32)
    cv2.polylines(triangle, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=5)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "triangle_worksheet.png"), triangle)

    circle = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.circle(circle, (120, 120), 72, (0, 0, 0), thickness=5)
    cv2.line(circle, (48, 120), (192, 120), (0, 0, 0), thickness=5)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "circle_worksheet.png"), circle)

    labelled_triangle = triangle.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(labelled_triangle, "A", (112, 25), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(labelled_triangle, "B", (12, 215), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(labelled_triangle, "C", (208, 215), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "labelled_triangle_worksheet.png"), labelled_triangle)

    dark = np.full((240, 240, 3), 18, dtype=np.uint8)
    cv2.polylines(dark, [triangle_points], isClosed=True, color=(40, 40, 40), thickness=3)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "dark_worksheet.png"), dark)

    blurred = cv2.GaussianBlur(circle, (35, 35), 0)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "blurred_worksheet.png"), blurred)

    complex_image = np.full((500, 500, 3), 255, dtype=np.uint8)
    for k in range(6):
        cx = 80 + (k % 3) * 170
        cy = 90 + (k // 3) * 200
        cv2.circle(complex_image, (cx, cy), 45, (0, 0, 0), thickness=3)
    cv2.imwrite(str(FIXTURE_DIRECTORY / "complex_worksheet.png"), complex_image)


if __name__ == "__main__":
    create_fixtures()
