import cv2
import numpy as np
import pytest

from app.services.figure_split import figure_image, find_figures
from app.services.braille import LouisBrailleTranslator
from app.services.ocr import EasyOcrProvider
from app.services.pipeline import FigureNotFound, build_full_analysis, list_figures


class _NoText:
    def readtext(self, image, detail, paragraph):
        return []


class _Louis:
    def translateString(self, tables, text):
        return text


def _worksheet() -> np.ndarray:
    page = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    for x in range(20, 980, 14):
        for y in (20, 1380):
            cv2.circle(page, (x, y), 2, (40, 40, 40), -1)
    cv2.putText(page, "Worksheet", (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.rectangle(page, (760, 50), (920, 110), (50, 50, 50), -1)
    cv2.rectangle(page, (160, 140), (500, 175), (0, 0, 0), 2)
    cv2.putText(page, "SAMPLE", (150, 800), cv2.FONT_HERSHEY_SIMPLEX, 6, (225, 225, 225), 25)
    # Figure 1: an angle with a blue arc and two letters; figure 2: a triangle.
    cv2.line(page, (150, 450), (400, 450), (0, 0, 0), 3)
    cv2.line(page, (150, 450), (330, 280), (0, 0, 0), 3)
    cv2.ellipse(page, (150, 450), (45, 45), 0, -43, 0, (255, 0, 0), 2)
    cv2.putText(page, "O", (130, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(page, "A", (410, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.polylines(page, [np.array([[600, 460], [850, 460], [720, 270]])], True, (0, 0, 0), 3)
    # Answer boxes with their prompts.
    for x in (250, 720):
        cv2.putText(page, "x =", (x - 70, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(page, (x, 515), (x + 100, 548), (0, 0, 0), 2)
    return page


def test_figures_are_found_in_reading_order_without_page_furniture():
    figures = find_figures(_worksheet())
    assert len(figures) == 2
    (ax0, ay0, ax1, ay1), (bx0, _, bx1, _) = figures[0].box, figures[1].box
    assert ax0 <= 150 and ax1 >= 400 and ay0 <= 280 and ay1 >= 460
    assert bx0 <= 600 and bx1 >= 850
    assert ax1 < bx0
    for box in (figures[0].box, figures[1].box):
        assert box[1] > 175 and box[3] < 515


def test_figure_image_keeps_coloured_arc_and_labels_and_blanks_the_rest():
    page = _worksheet()
    first = find_figures(page)[0]
    crop = figure_image(page, first)
    blue = (crop[:, :, 0] > 200) & (crop[:, :, 2] < 80)
    assert blue.sum() > 30
    assert crop.min() < 50
    watermark = (crop.min(axis=2) > 200) & (crop.min(axis=2) < 240)
    assert watermark.sum() < 0.01 * watermark.size


def test_single_figure_image_is_not_split():
    page = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.polylines(page, [np.array([[100, 500], [700, 500], [400, 100]])], True, (0, 0, 0), 3)
    cv2.line(page, (400, 100), (400, 500), (0, 0, 0), 2)
    assert find_figures(page) == []


def test_pipeline_reports_figures_and_rejects_a_missing_one():
    ok, encoded = cv2.imencode(".png", _worksheet())
    assert ok
    fake_ocr_provider = EasyOcrProvider(reader=_NoText())
    fake_braille_translator = LouisBrailleTranslator(bindings=_Louis())
    assert [f["number"] for f in list_figures(encoded.tobytes())] == [1, 2]
    with pytest.raises(FigureNotFound):
        build_full_analysis(encoded.tobytes(), ocr_provider=fake_ocr_provider,
                            braille_translator=fake_braille_translator, figure=5)
    result = build_full_analysis(encoded.tobytes(), ocr_provider=fake_ocr_provider,
                                 braille_translator=fake_braille_translator, figure=1)
    assert result.figure == 1
    assert [f["number"] for f in result.figures] == [1, 2]
    assert result.semantic_geometry.image_width < 400


def test_process_endpoint_defaults_to_first_figure_and_can_switch(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import sessions as sessions_api
    from app.main import app

    calls = []
    real = sessions_api.build_full_analysis

    def fake_analysis(image_bytes, edge_sensitivity=50, figure=None):
        calls.append(figure)
        return real(image_bytes, edge_sensitivity, ocr_provider=EasyOcrProvider(reader=_NoText()),
                    braille_translator=LouisBrailleTranslator(bindings=_Louis()), figure=figure)

    monkeypatch.setattr(sessions_api, "build_full_analysis", fake_analysis)
    client = TestClient(app)
    ok, encoded = cv2.imencode(".png", _worksheet())
    created = client.post("/api/sessions", files={"image": ("sheet.png", encoded.tobytes(), "image/png")})
    session_id = created.json()["session_id"]

    first = client.post(f"/api/sessions/{session_id}/process", json={})
    assert first.status_code == 200
    body = first.json()
    assert body["processing_params"]["figure"] == 0
    assert [f["number"] for f in body["figures"]] == [1, 2]
    assert body["figures"][0]["page_width"] == 1000

    second = client.post(f"/api/sessions/{session_id}/process", json={"figure": 1})
    assert second.json()["processing_params"]["figure"] == 1
    page = client.post(f"/api/sessions/{session_id}/process", json={"whole_page": True})
    assert page.json()["processing_params"]["figure"] is None
    missing = client.post(f"/api/sessions/{session_id}/process", json={"figure": 9})
    assert missing.status_code == 422
    assert calls == [0, 1, None, 9]
