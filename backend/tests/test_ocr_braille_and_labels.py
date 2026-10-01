import os
from math import hypot

import cv2
import pytest

from app.services.braille import (
    LouisBrailleTranslator,
    UEB_GRADE_2_TABLE,
    configure_tablepath,
    resolve_table_file,
    resolve_tablepath,
    to_unicode_braille,
)
from app.services.braille_layout import place_braille_markers
from app.services.label_mapping import map_label_to_geometry
from app.services.ocr import EasyOcrProvider, OcrDetection
from app.services.pipeline import build_preview

# Native pixel size of tests/fixtures/labelled_triangle_worksheet.png.
FIXTURE_SIZE = (240, 240)


class FakeReader:
    """Stands in for easyocr.Reader.

    The provider upscales small images before recognition, so a faithful fake
    must report boxes in the coordinate space of the image it was handed. The
    provider is then responsible for mapping them back to original-image space.
    """

    detections = [
        ([[112, 8], [128, 8], [128, 28], [112, 28]], "A", 0.98),
        ([[8, 202], [25, 202], [25, 222], [8, 222]], "B", 0.97),
        ([[205, 202], [224, 202], [224, 222], [205, 222]], "C", 0.96),
    ]
    base_size: tuple[int, int] | None = None
    received_size: tuple[int, int] | None = None

    def readtext(self, image, detail=1, paragraph=False):
        self.received_size = (image.shape[1], image.shape[0])
        self.base_size = FIXTURE_SIZE
        width, height = self.received_size
        base_width, base_height = FIXTURE_SIZE
        scale_x = width / base_width
        scale_y = height / base_height
        return [
            ([[x * scale_x, y * scale_y] for x, y in box], text, confidence)
            for box, text, confidence in self.detections
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
    # The table argument must be something liblouis can actually open: a resolved
    # absolute path when one is available, otherwise the bare UEB table name.
    expected = resolve_table_file(UEB_GRADE_2_TABLE) or UEB_GRADE_2_TABLE
    assert bindings.calls == [([expected], "A"), ([expected], "12")]


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
    # setenv first so monkeypatch restores the variable configure_tablepath() writes;
    # a bare delenv on an unset variable records nothing and the fake path leaks.
    monkeypatch.setenv("LOUIS_TABLEPATH", "")
    monkeypatch.delenv("LOUIS_TABLEPATH")
    monkeypatch.setattr("app.services.braille._DEFAULT_TABLEPATHS", (tables_dir,))

    configure_tablepath()

    assert os.environ["LOUIS_TABLEPATH"] == str(tables_dir)


def test_configure_tablepath_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("LOUIS_TABLEPATH", "C:/already/set")

    configure_tablepath()

    assert os.environ["LOUIS_TABLEPATH"] == "C:/already/set"


# --- table resolution and output encoding ----------------------------------

class RecordingLouis:
    def __init__(self, result="abc"):
        self.result = result
        self.calls = []

    def translateString(self, tables, text):
        self.calls.append((list(tables), text))
        return self.result


def test_resolve_table_file_returns_existing_file(tmp_path, monkeypatch):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    table_file = tables_dir / UEB_GRADE_2_TABLE
    table_file.write_text("")
    monkeypatch.setenv("LOUIS_TABLEPATH", str(tables_dir))
    monkeypatch.setattr("app.services.braille._DEFAULT_TABLEPATHS", ())

    assert resolve_table_file() == str(table_file)


def test_translator_passes_absolute_table_path(tmp_path, monkeypatch):
    # A bare table name only works when liblouis honours LOUIS_TABLEPATH, which the
    # 3.39 Windows build does not. The translator must hand over a real file path.
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / UEB_GRADE_2_TABLE).write_text("")
    monkeypatch.setenv("LOUIS_TABLEPATH", str(tables_dir))
    monkeypatch.setattr("app.services.braille._DEFAULT_TABLEPATHS", ())
    bindings = RecordingLouis()

    LouisBrailleTranslator(bindings=bindings).translate("A")

    (tables, text), = bindings.calls
    assert tables == [str(tables_dir / UEB_GRADE_2_TABLE)]
    assert text == "A"


def test_translator_converts_ascii_braille_to_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.braille.resolve_table_file", lambda table=None: None)

    result = LouisBrailleTranslator(bindings=RecordingLouis(",a")).translate("A")

    # Capital indicator (dot 6) followed by the letter a (dot 1).
    assert result == "\u2820\u2801"


def test_to_unicode_braille_leaves_unicode_braille_untouched():
    assert to_unicode_braille("\u2820\u2801") == "\u2820\u2801"


def test_to_unicode_braille_maps_braille_ascii_cells():
    assert to_unicode_braille("a") == "\u2801"  # dot 1
    assert to_unicode_braille(",") == "\u2820"  # dot 6
    assert to_unicode_braille("#") == "\u283c"  # dots 3-4-5-6, numeric indicator
    assert to_unicode_braille("=") == "\u283f"  # all six dots
    assert to_unicode_braille("~") == to_unicode_braille("^") == "\u2818"  # dots 4-5
    assert to_unicode_braille(" ") == "\u2800"
    assert to_unicode_braille("") == ""


def test_to_unicode_braille_rejects_non_braille_ascii():
    with pytest.raises(ValueError):
        to_unicode_braille("\x01")


def test_real_liblouis_translates_ueb_grade_2():
    translator = LouisBrailleTranslator()

    try:
        result = translator.translate("A")
    except Exception as error:  # pragma: no cover - depends on native install
        pytest.skip(f"native Liblouis unavailable: {error}")

    # A capital A in UEB is the capital indicator followed by the letter a cell.
    assert result == "\u2820\u2801"


@pytest.mark.parametrize("text", ["A", "B", "AB", "6 cm", "45\u00b0", "x = 3.5", "\u2220ABC", "r = 4 cm", "10 m\u00b2"])
def test_real_liblouis_matches_unicode_display_table(text):
    """The Braille-ASCII mapping must agree with Liblouis' own Unicode display table."""
    table = resolve_table_file(UEB_GRADE_2_TABLE)
    display = resolve_table_file("unicode.dis")
    try:
        import louis
    except ImportError:  # pragma: no cover - depends on native install
        pytest.skip("native Liblouis unavailable")
    if not table or not display:  # pragma: no cover - depends on native install
        pytest.skip("Liblouis tables unavailable")

    expected = louis.translateString([display, table], text)

    assert LouisBrailleTranslator().translate(text) == expected
