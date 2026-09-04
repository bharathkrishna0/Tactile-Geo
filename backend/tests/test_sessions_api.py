from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_create_session_accepts_a_valid_png(fixture_directory):
    image = (fixture_directory / "triangle_worksheet.png").read_bytes()

    response = client.post(
        "/api/sessions",
        files={"image": ("triangle.png", image, "image/png")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "triangle.png"
    assert payload["session_id"]
    assert payload["created_at"]


def test_create_session_rejects_a_wrong_file_type():
    response = client.post(
        "/api/sessions",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Upload a PNG, JPG, or JPEG image."


def test_create_session_rejects_an_oversized_file():
    oversized_image = b"0" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/sessions",
        files={"image": ("large.png", oversized_image, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "The uploaded image exceeds the 10 MB limit."

