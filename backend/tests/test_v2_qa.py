"""V2a export checks: figure structure lost to the density budget, and isolated specks."""
from app.models.geometry import (
    ConfidenceLevel,
    DetectedElement,
    ElementRelationship,
    GeometryType,
    RelationshipType,
    SemanticGeometry,
)
from app.services.tactile_qa import BLOCKING_CHECKS, run_tactile_qa
from app.services.tactile_rules import TACTILE_RULES
from app.services.tactile_svg import braille_text_extent, page_layout
from app.services.tactile_simplification import (
    density_removed_count,
    essential_density_removed_count,
    simplify_geometry,
)


def _element(eid, gtype, geometry, confidence=0.9, label=None, **props):
    return DetectedElement(
        id=eid, type=gtype, geometry=geometry, confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH if confidence >= 0.8 else ConfidenceLevel.MEDIUM,
        needs_review=False, source="hough", semantic_properties=dict(props), associated_label_id=label,
    )


def _line(eid, start, end, confidence=0.9, **props):
    return _element(eid, GeometryType.LINE_SEGMENT, {"start": list(start), "end": list(end)}, confidence, **props)


def _label(eid, position, text):
    return _element(eid, GeometryType.TEXT_LABEL, {"text": text, "position": list(position)})


def _connected(eids):
    return [
        ElementRelationship(
            id=f"rel_{a}_{b}", type=RelationshipType.CONNECTED_LINES, element_ids=[a, b],
            confidence=0.9, confidence_level=ConfidenceLevel.HIGH, needs_review=False,
        )
        for a, b in zip(eids, eids[1:])
    ]


def _zigzag(count, prefix="fig", confidence=0.6):
    """A connected chain of short sides: one figure, drawn with lower confidence than the clutter."""
    points = [(100 + 25 * i, 600 + (0 if i % 2 else 40)) for i in range(count + 1)]
    return [_line(f"{prefix}_{i}", points[i], points[i + 1], confidence) for i in range(count)]


def _clutter(count):
    return [_line(f"stray_{i}", (60 + 20 * i, 80 + 7 * i), (60 + 20 * i, 200 + 7 * i)) for i in range(count)]


def _report(semantic):
    simplified = simplify_geometry(semantic)
    qa = run_tactile_qa(
        simplified.elements, semantic.image_width, semantic.image_height,
        density_removed=density_removed_count(simplified),
        essential_removed=essential_density_removed_count(simplified),
    )
    return simplified, qa


def test_density_budget_removes_unconnected_clutter_before_figure_sides():
    figure = _zigzag(30)
    semantic = SemanticGeometry(
        elements=figure + _clutter(20),
        relationships=_connected([e.id for e in figure]),
        image_width=1000, image_height=1000,
    )
    simplified, qa = _report(semantic)
    kept = {e.id for e in simplified.elements}
    assert {e.id for e in figure} <= kept
    assert density_removed_count(simplified) == 50 - TACTILE_RULES.tactile_feature_target
    assert essential_density_removed_count(simplified) == 0
    assert qa.passes


def test_export_blocks_when_figure_structure_is_dropped_to_fit_the_page():
    figure = _zigzag(TACTILE_RULES.tactile_feature_limit + 10)
    semantic = SemanticGeometry(
        elements=figure,
        relationships=_connected([e.id for e in figure]),
        image_width=2000, image_height=1000,
    )
    simplified, qa = _report(semantic)
    assert essential_density_removed_count(simplified) == 10
    assert all(a.detail for a in simplified.actions if a.action == "removed_density_essential")
    assert "density_removed_essential_geometry" in BLOCKING_CHECKS
    assert "density_removed_essential_geometry" in {i.check for i in qa.issues}
    assert not qa.passes


def test_ticks_and_angle_markers_are_never_dropped_for_density():
    ticks = [_line(f"tick_{i}", (100 + 60 * i, 590), (100 + 60 * i, 610), 0.5, role="tick") for i in range(10)]
    arc = _element("arc", GeometryType.ARC, {"center": [500, 500], "radius": 30}, 0.5, angle_marker=True)
    semantic = SemanticGeometry(
        elements=ticks + [arc] + _clutter(40), image_width=1000, image_height=1000,
    )
    simplified, _ = _report(semantic)
    kept = {e.id for e in simplified.elements}
    assert {t.id for t in ticks} | {"arc"} <= kept


def _speck(eid, x, y, label):
    return _element(
        eid, GeometryType.POLYGON, {"points": [[x, y], [x + 5, y + 1], [x + 5, y - 1]], "area": 5.0}, 0.5, label=label,
    )


def test_isolated_labelled_speck_is_removed_as_auditable_noise():
    semantic = SemanticGeometry(
        elements=[
            _line("side_0", (200, 600), (800, 600)),
            _line("side_1", (800, 600), (500, 300)),
            _speck("speck", 340, 120, "header"),
            _label("header", (330, 110), "Name:"),
        ],
        relationships=_connected(["side_0", "side_1"]),
        image_width=1000, image_height=1000,
    )
    simplified, qa = _report(semantic)
    kept = {e.id for e in simplified.elements}
    assert "speck" not in kept and "header" in kept
    action = next(a for a in simplified.actions if a.element_id == "speck")
    assert action.action == "removed_isolated_speck" and "header" in action.detail
    assert "feature_below_minimum_size" not in {i.check for i in qa.issues}
    assert qa.passes


def test_labelled_small_outline_touching_the_figure_is_kept_and_blocks():
    semantic = SemanticGeometry(
        elements=[
            _line("side_0", (200, 600), (800, 600)),
            _speck("mark", 400, 603, "lab"),
            _label("lab", (400, 630), "A"),
        ],
        image_width=1000, image_height=1000,
    )
    simplified, qa = _report(semantic)
    assert "mark" in {e.id for e in simplified.elements}
    assert "feature_below_minimum_size" in {i.check for i in qa.issues}
    assert not qa.passes


def test_braille_beside_a_short_tick_moves_clear_of_the_lengthened_tick():
    # A 12 px tick on a number line, its number label placed just clear of the tick as drawn.
    width = height = 800
    mm_per_px = page_layout(width, height).mm_per_px
    clearance_px = TACTILE_RULES.braille_to_line_clearance_mm / mm_per_px
    _, cell_height = braille_text_extent("⠼⠑", width, height)
    label_y = round(606 + clearance_px + cell_height / 2 + 1)
    axis = _line("axis", (100, 600), (700, 600))
    tick = _line("tick", (400, 594), (400, 606), confidence=0.45, role="tick")
    label = _element("lab", GeometryType.TEXT_LABEL, {"text": "5", "braille": "⠼⠑", "position": [400, label_y]})
    semantic = SemanticGeometry(
        elements=[axis, tick, label], relationships=[], image_width=width, image_height=height,
    )
    simplified, qa = _report(semantic)
    assert "enlarged_tick" in {a.action for a in simplified.actions if a.element_id == "tick"}
    assert "moved_braille_from_tick" in {a.action for a in simplified.actions if a.element_id == "lab"}
    assert "braille_on_line" not in {issue.check for issue in qa.issues}


def test_density_budget_keeps_an_unrelated_closed_figure_over_clutter():
    octagon = _element(
        "octagon", GeometryType.POLYGON,
        {"points": [[500, 300], [570, 330], [600, 400], [570, 470], [500, 500], [430, 470], [400, 400], [430, 330]]},
        confidence=0.5,
    )
    semantic = SemanticGeometry(
        elements=[octagon] + _clutter(45), relationships=[], image_width=1000, image_height=1000,
    )
    simplified, qa = _report(semantic)
    assert "octagon" in {e.id for e in simplified.elements}
    assert essential_density_removed_count(simplified) == 0
