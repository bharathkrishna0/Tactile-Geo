import cv2
import numpy as np

from app.services.image_preprocessing import preprocess_image
from app.services.vectorization import extract_shapes


def test_preprocessing_thresholds_and_filters_triangle_fixture(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))

    result = preprocess_image(image, edge_sensitivity=80)

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})
    assert np.count_nonzero(result) > 500


def test_vectorization_detects_triangle_once_as_closed_contour(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))
    result = preprocess_image(image, edge_sensitivity=100)

    shapes = extract_shapes(result, edge_sensitivity=100)

    assert [len(shape["points"]) for shape in shapes if shape["type"] == "contour"] == [3]
    assert not any(shape["type"] in {"line", "ellipse"} for shape in shapes)


def test_vectorization_detects_circle_contour(fixture_directory):
    image = cv2.imread(str(fixture_directory / "circle_worksheet.png"))
    result = preprocess_image(image, edge_sensitivity=100)

    shapes = extract_shapes(result, edge_sensitivity=100)

    assert any(shape["type"] == "contour" and len(shape["points"]) >= 6 for shape in shapes)
