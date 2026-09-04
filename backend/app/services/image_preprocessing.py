import cv2
import numpy as np

def decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image

def preprocess_image(image: np.ndarray, edge_sensitivity: int = 50) -> np.ndarray:
    """Section 6.1: adaptive thresholding then 3x3 opening and erosion."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_offset = max(2, min(16, 16 - round(edge_sensitivity * 0.14)))
    thresholded = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, threshold_offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    return cv2.erode(opened, kernel, iterations=1)
