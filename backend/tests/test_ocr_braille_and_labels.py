import os
from math import hypot

import cv2

from app.services.braille import (
    LouisBrailleTranslator,
    UEB_GRADE_2_TABLE,
    configure_tablepath,
    resolve_tablepath,
)
from app.services.braille_layout import place_braille_markers
from app.services.label_mapping import map_label_to_geometry
from app.services.ocr import EasyOcrProvider, OcrDetection
from app.services.pipeline import build_preview


class FakeReader:
    def readtext(self, image, detail, paragraph):
        return [
            ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
            ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
            ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
        ]


class FakeLouis:
    translations = {"A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉", "12": "⠼⠁⠃"}

    def __init__(self):
        self.calls = []

    def translateString(self, tables, text):
        self.calls.append((tables, text))
        return self.translations[text]


def triangle_shapes():
    return [
        {"type": "line", "points": [(120, 35), (35, 195)]},
        {"type": "line", "points": [(35, 195), (205, 195)]},
        {"type": "line", "points": [(205, 195), (120, 35)]},
    ]


def test_easyocr_adapter_normalizes_label_detections(fixture_directory):
    image = cv2.imread(str(fixture_directory / "labelled_triangle_worksheet.png"))

    labels = EasyOcrProvider(reader=FakeReader()).detect(image)

    assert [label.text for label in labels] == ["A", "B", "C"]
    assert labels[0].bbox == [(112, 8), (128, 8), (128, 28), (112, 28)]
    assert labels[0].confidence == 0.98


def test_label_maps_to_nearest_triangle_vertex():
    detection = OcrDetection("A", [(112, 8), (128, 8), (128, 28), (112, 28)], 0.98)

    mapped = map_label_to_geometry(detection, triangle_shapes())

    assert mapped["anchor_type"] == "vertex"
    assert mapped["anchor"] == (120, 35)


def test_liblouis_translator_uses_ueb_table_for_capitals_and_numbers():
    bindings = FakeLouis()
    translator = LouisBrailleTranslator(bindings=bindings)

    assert translator.translate("A") == "⠠⠁"
    assert translator.translate("12") == "⠼⠁⠃"
    assert bindings.calls == [([UEB_GRADE_2_TABLE], "A"), ([UEB_GRADE_2_TABLE], "12")]


def test_collision_layout_moves_markers_off_lines_and_apart():
    shapes = [{"type": "line", "points": [(20, 100), (220, 100)]}]
    labels = [
        {"text": "A", "braille": "⠠⠁", "desired_position": (120, 100)},
        {"text": "B", "braille": "⠠⠃", "desired_position": (120, 100)},
    ]

    placed = place_braille_markers(labels, shapes, minimum_clearance=20)

    assert all(label["collision_adjusted"] for label in placed)
    assert all(abs(label["position"][1] - 100) >= 20 for label in placed)
    assert hypot(placed[0]["position"][0] - placed[1]["position"][0], placed[0]["position"][1] - placed[1]["position"][1]) >= 20


def test_pipeline_returns_mapped_braille_labels_for_labelled_fixture(fixture_directory):
    image_bytes = (fixture_directory / "labelled_triangle_worksheet.png").read_bytes()

    _, _, labels = build_preview(
        image_bytes,
        edge_sensitivity=100,
        ocr_provider=EasyOcrProvider(reader=FakeReader()),
        braille_translator=LouisBrailleTranslator(bindings=FakeLouis()),
    )

    assert [label["text"] for label in labels] == ["A", "B", "C"]
    assert labels[0]["braille"] == "⠠⠁"
    assert all(label["anchor_type"] in {"vertex", "edge"} for label in labels)


def test_resolve_tablepath_uses_explicit_env_var(monkeypatch):
    monkeypatch.setenv("LOUIS_TABLEPATH", "C:/custom/liblouis/tables")

    resolved = resolve_tablepath()

    assert resolved == "C:/custom/liblouis/tables"


def test_resolve_tablepath_prefers_explicit_over_default(tmp_path, monkeypatch):
    default_dir = tmp_path / "default_tables"
    default_dir.mkdir()
    (default_dir / UEB_GRADE_2_TABLE).write_text("")
    explicit = tmp_path / "explicit_tables"
    explicit.mkdir()
    (explicit / UEB_GRADE_2_TABLE).write_text("")
    monkeypatch.setenv("LOUIS_TABLEPATH", str(explicit))
    monkeypatch.setattr("app.core.config.LOUIS_TABLEPATH", "")

    resolved = resolve_tablepath()

    assert resolved == str(explicit)


def test_configure_tablepath_sets_env_var_when_unset(tmp_path, monkeypatch):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / UEB_GRADE_2_TABLE).write_text("")
    monkeypatch.delenv("LOUIS_TABLEPATH", raising=False)
    monkeypatch.setattr("app.services.braille._DEFAULT_TABLEPATHS", (tables_dir,))

    configure_tablepath()

    assert os.environ["LOUIS_TABLEPATH"] == str(tables_dir)


def test_configure_tablepath_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("LOUIS_TABLEPATH", "C:/already/set")

    configure_tablepath()

    assert os.environ["LOUIS_TABLEPATH"] == "C:/already/set"
