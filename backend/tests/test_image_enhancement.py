"""Tests for Milestone 3: Image enhancement + quality-gate integration."""
import cv2
import numpy as np

from app.services.image_enhancement import enhance_copy
from app.services.image_preprocessing import decode_image
from app.services.vectorization import extract_shapes
from app.services.image_preprocessing import preprocess_image


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class FakeLouis:
    def translateString(self, tables, text):
        return {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉"}.get(text, text)


# --- enhancement does not modify original ---

def test_enhance_copy_does_not_modify_original():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.line(image, (10, 10), (90, 90), (50, 50, 50), 3)
    original = image.copy()

    enhanced = enhance_copy(image)

    assert np.array_equal(image, original)
    assert not np.array_equal(enhanced, original)
    assert enhanced.shape == original.shape


def test_enhance_copy_preserves_grayscale_input():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    enhanced = enhance_copy(gray)

    assert enhanced.shape == (100, 100, 3)
    assert enhanced.dtype == np.uint8


def test_enhance_copy_increases_contrast_for_low_contrast():
    # A very low-contrast grey image.
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (160, 160), (120, 120, 120), 3)
    gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_before = float(np.std(gray_before))

    enhanced = enhance_copy(image)
    gray_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    std_after = float(np.std(gray_after))

    assert std_after > std_before


def test_enhance_copy_never_modifies_decoded_image():
    # Round-trip through decode to ensure the enhancement is on a copy.
    from pathlib import Path
    import sys
    fixture = Path(__file__).parent.parent / "tests" / "fixtures" / "triangle_worksheet.png"
    if not fixture.exists():
        fixture = Path(__file__).parent / "fixtures" / "triangle_worksheet.png"
    image = cv2.imread(str(fixture))
    original = image.copy()

    enhance_copy(image)

    assert np.array_equal(image, original)


# --- enhancement improves downstream detection ---

def test_enhanced_processing_still_detects_triangle(fixture_directory):
    image = cv2.imread(str(fixture_directory / "triangle_worksheet.png"))
    enhanced = enhance_copy(image)
    filtered = preprocess_image(enhanced, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    # The closed outline is the single representation of the triangle; Hough
    # lines along its stroke would double-strike the tactile edges.
    assert [len(s["points"]) for s in shapes if s["type"] == "contour"] == [3]
    assert not any(s["type"] == "line" for s in shapes)


def test_enhanced_processing_detects_low_contrast(fixture_directory):
    # Build a low-contrast copy of the dark fixture and confirm it still works.
    image = cv2.imread(str(fixture_directory / "dark_worksheet.png"))
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    original_shape = image.shape[:2]

    enhanced = enhance_copy(image)
    filtered = preprocess_image(enhanced, edge_sensitivity=100)
    shapes = extract_shapes(filtered, edge_sensitivity=100)

    assert enhanced.shape[:2] == original_shape
    assert any(s["type"] == "contour" for s in shapes)


def test_enhance_copy_maintains_dimensions():
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    enhanced = enhance_copy(image)

    assert enhanced.shape == (240, 320, 3)
    assert enhanced.dtype == np.uint8
