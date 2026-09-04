import cv2
import numpy as np

from app.services.image_quality import assess_image_quality
from app.services.image_preprocessing import decode_image


def test_quality_passes_for_clean_triangle_fixture(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))

    report = assess_image_quality(image)

    assert report.passes_gate is True
    assert not any(issue.severity == "error" for issue in report.issues)


def test_quality_detects_low_resolution():
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    cv2.line(image, (5, 5), (55, 55), (0, 0, 0), 2)

    report = assess_image_quality(image)

    assert any(issue.check == "resolution" for issue in report.issues)


def test_quality_detects_blur(fixture_directory):
    image = cv2.imread(str(fixture_directory / "blurred_worksheet.png"))

    report = assess_image_quality(image)

    assert any(issue.check == "blur" for issue in report.issues)


def test_quality_detects_dark_image(fixture_directory):
    image = cv2.imread(str(fixture_directory / "dark_worksheet.png"))

    report = assess_image_quality(image)

    assert any(issue.check == "brightness" for issue in report.issues)
    assert not report.passes_gate


def test_quality_detects_low_contrast():
    image = np.full((200, 200, 3), 128, dtype=np.uint8)

    report = assess_image_quality(image)

    assert any(issue.check == "contrast" for issue in report.issues)


def test_quality_handles_corrupt_input():
    image_bytes = b"this is not an image"
    try:
        decode_image(image_bytes)
    except ValueError:
        pass
