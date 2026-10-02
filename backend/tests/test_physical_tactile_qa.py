"""Physical-unit tactile QA, tactile density budget, and dark-page recovery."""
import cv2
import numpy as np

from app.models.geometry import ConfidenceLevel, DetectedElement, GeometryType, SemanticGeometry
from app.services.braille_layout import place_braille_markers
from app.services.image_enhancement import has_light_ink_on_dark_page, normalize_polarity
from app.services.tactile_qa import BLOCKING_CHECKS, run_tactile_qa
from app.services.tactile_rules import TACTILE_RULES
from app.services.tactile_simplification import simplify_geometry
from app.services.tactile_svg import page_layout


def _line(eid, start, end, confidence=0.9, label=None) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.LINE_SEGMENT,
        geometry={"start": start, "end": end},
        confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH if confidence >= 0.8 else ConfidenceLevel.MEDIUM,
        needs_review=False,
        source="hough",
        bbox=(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]), abs(end[1] - start[1])),
        associated_label_id=label,
    )


def _label(eid, position, text="A", associated=None) -> DetectedElement:
    return DetectedElement(
        id=eid,
        type=GeometryType.TEXT_LABEL,
        geometry={"text": text, "position": position},
        confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        needs_review=False,
        source="ocr",
        bbox=(position[0] - 5, position[1] - 5, 10, 10),
        associated_label_id=associated,
    )


def _scaled(elements, factor):
    scaled = []
    for e in elements:
        geo = dict(e.geometry)
        for key in ("start", "end", "position"):
            if key in geo:
                geo[key] = (geo[key][0] * factor, geo[key][1] * factor)
        bbox = tuple(v * factor for v in e.bbox) if e.bbox else None
        scaled.append(DetectedElement(**{**e.__dict__, "geometry": geo, "bbox": bbox}))
    return scaled


def _checks(report):
    return sorted({issue.check for issue in report.issues})


def test_qa_outcome_does_not_depend_on_upload_resolution():
    elements = [
        _line("el_0", (40, 40), (360, 40), label="el_2"),
        _line("el_1", (40, 60), (360, 60)),
        _line("el_3", (200, 300), (206, 300)),
        _label("el_2", (200, 20), associated="el_0"),
    ]
    low = run_tactile_qa(elements, 400, 400)
    high = run_tactile_qa(_scaled(elements, 4), 1600, 1600)
    assert _checks(low) == _checks(high)
    assert low.passes == high.passes


def test_minimum_feature_size_is_measured_on_paper():
    mm_per_px = page_layout(400, 400).mm_per_px
    small = round((TACTILE_RULES.minimum_feature_size_mm * 0.8) / mm_per_px)
    large = round((TACTILE_RULES.minimum_feature_size_mm * 1.5) / mm_per_px)
    tiny = run_tactile_qa([_line("el_0", (100, 100), (100 + small, 100))], 400, 400)
    fine = run_tactile_qa([_line("el_0", (100, 100), (100 + large, 100))], 400, 400)
    assert "feature_below_minimum_size" in _checks(tiny)
    assert "feature_below_minimum_size" not in _checks(fine)


def test_braille_cells_straddling_the_middle_of_a_line_are_blocking():
    line = _line("el_0", (100, 200), (300, 200), label="el_1")
    on_line = run_tactile_qa([line, _label("el_1", (200, 195), associated="el_0")], 400, 400)
    clear = run_tactile_qa([line, _label("el_1", (200, 170), associated="el_0")], 400, 400)
    assert "braille_on_line" in _checks(on_line) and not on_line.passes
    assert "braille_on_line" not in _checks(clear)


def test_density_above_limit_blocks_export():
    limit = TACTILE_RULES.tactile_feature_limit
    lines = [_line(f"el_{i}", (100, 20 + 30 * i), (700, 20 + 30 * i)) for i in range(limit + 1)]
    report = run_tactile_qa(lines, 2000, 2000)
    assert "exceeds_tactile_density" in _checks(report)
    assert "exceeds_tactile_density" in BLOCKING_CHECKS
    assert not report.passes


def test_density_between_target_and_limit_is_a_warning():
    count = TACTILE_RULES.tactile_feature_target + 1
    lines = [_line(f"el_{i}", (100, 20 + 40 * i), (700, 20 + 40 * i)) for i in range(count)]
    report = run_tactile_qa(lines, 2000, 2000)
    assert "complexity" in _checks(report)
    assert "exceeds_tactile_density" not in _checks(report)


def test_heavy_density_reduction_requires_review():
    lines = [_line(f"el_{i}", (100, 50 + 60 * i), (700, 50 + 60 * i)) for i in range(10)]
    heavy = run_tactile_qa(lines, 2000, 2000, density_removed=11)
    light = run_tactile_qa(lines, 2000, 2000, density_removed=3)
    assert "density_reduction_requires_review" in _checks(heavy) and not heavy.passes
    assert "density_reduced" in _checks(light)
    assert "density_reduction_requires_review" not in _checks(light)


def test_simplification_fits_the_tactile_budget_and_explains_every_drop():
    target = TACTILE_RULES.tactile_feature_target
    lines = [_line(f"el_{i}", (100, 20 + 30 * i), (700, 20 + 30 * i), confidence=0.6 + i * 0.001) for i in range(target + 20)]
    lines[0] = _line("el_0", (100, 20), (700, 20), confidence=0.6, label="lab")
    semantic = SemanticGeometry(elements=lines + [_label("lab", (400, 5), associated="el_0")], image_width=2000, image_height=2000)
    simplified = simplify_geometry(semantic)
    kept = {e.id for e in simplified.elements}
    embossed = [e for e in simplified.elements if e.type is not GeometryType.TEXT_LABEL]
    budget_drops = {a.element_id for a in simplified.actions if a.action == "removed_density_budget"}
    assert len(embossed) == target
    assert "el_0" in kept and "lab" in kept
    assert budget_drops == {e.id for e in lines} - kept
    # Lowest confidence goes first.
    assert "el_1" in budget_drops and f"el_{target + 19}" in kept


def test_simplification_removes_untouchably_small_unlabelled_features_only():
    mm_per_px = page_layout(400, 400).mm_per_px
    small = round((TACTILE_RULES.minimum_feature_size_mm * 0.5) / mm_per_px)
    semantic = SemanticGeometry(
        elements=[
            _line("el_0", (50, 50), (50 + small, 50)),
            _line("el_1", (50, 200), (50 + small, 200), label="lab"),
            _line("el_2", (50, 300), (350, 300)),
            _label("lab", (60, 230), associated="el_1"),
        ],
        image_width=400,
        image_height=400,
    )
    simplified = simplify_geometry(semantic)
    kept = {e.id for e in simplified.elements}
    assert "el_0" not in kept and {"el_1", "el_2", "lab"} <= kept
    assert any(a.element_id == "el_0" and a.action == "removed_below_tactile_size" for a in simplified.actions)


def test_on_page_braille_placement_satisfies_physical_clearance():
    shapes = [{"type": "line", "points": [(100, 200), (300, 200)]}]
    labels = [{"id": "el_1", "text": "A", "braille": "⠁", "desired_position": (200, 198)}]
    placed = place_braille_markers(labels, shapes, bounds=(400, 400))
    position = placed[0]["position"]
    assert placed[0]["collision_adjusted"]
    report = run_tactile_qa(
        [_line("el_0", (100, 200), (300, 200), label="el_1"), _label("el_1", position, associated="el_0")], 400, 400,
    )
    assert "braille_on_line" not in _checks(report)


def test_light_ink_on_dark_page_is_inverted():
    dark = np.full((200, 200, 3), 30, dtype=np.uint8)
    cv2.line(dark, (20, 100), (180, 100), (230, 230, 230), 3)
    light = 255 - dark
    assert has_light_ink_on_dark_page(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))
    assert not has_light_ink_on_dark_page(cv2.cvtColor(light, cv2.COLOR_BGR2GRAY))
    assert np.array_equal(normalize_polarity(dark), light)
    assert np.array_equal(normalize_polarity(light), light)
