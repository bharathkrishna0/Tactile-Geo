"""Configurable tactile representation rules (BANA/ICEB-aligned, traceable).

Centralizes the tactile constants so they are not scattered as magic numbers
throughout the pipeline. Values can be overridden via the ``TACTILE_OVERRIDES``
environment variable (JSON) for experimentation without a code change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class TactileRules:
    # Stroke width (points) for tactile lines.
    stroke_width_min_pt: float = 1.5
    stroke_width_max_pt: float = 2.0

    # The stroke width actually used when rendering, i.e. the teacher-adjustable
    # value. Kept separate from the BANA band above so QA can genuinely validate
    # the rendered value against the band instead of validating itself.
    stroke_width_pt: float = 2.0

    # Minimum separation (pixels) between independent tactile features.
    minimum_feature_spacing_px: float = 18.0

    # Minimum clearance (pixels) between a Braille marker and geometry.
    braille_to_line_clearance_px: float = 18.0

    # Minimum clearance between two Braille markers.
    braille_to_braille_spacing_px: float = 18.0

    # A feature smaller than this extent (pixels) is flagged as suspiciously tiny.
    tiny_feature_px: float = 10.0

    # Element count above which the diagram is flagged as excessively complex.
    complexity_threshold: int = 50

    # Physical (on-paper) thresholds, applied after the diagram is scaled onto
    # the A4 page so QA does not depend on the upload's pixel resolution. These
    # are conservative, guideline-informed defaults, not a certified standard.
    # Smallest feature extent that can be resolved by touch.
    minimum_feature_size_mm: float = 5.0
    # Separation below which two independent features read as one.
    minimum_feature_spacing_mm: float = 6.0
    # Features closer than this physically touch once embossed.
    overlap_tolerance_mm: float = 1.5
    # Blank space required between a braille cell block and any line.
    braille_to_line_clearance_mm: float = 3.0
    # Blank space required between two braille cell blocks.
    braille_to_braille_spacing_mm: float = 3.0
    # An unassociated label further than this from every shape is dangling.
    label_isolation_mm: float = 50.0

    # Tactile density: simplification aims for at most ``tactile_feature_target``
    # embossed geometry features on one A4 sheet; more than
    # ``tactile_feature_limit`` blocks print-ready export.
    tactile_feature_target: int = 40
    tactile_feature_limit: int = 60

    # Printable boundary margin, as a fraction of image width/height.
    printable_margin_fraction: float = 0.1

    # A high-confidence element may not be removed as noise by simplification.
    high_confidence_threshold: float = 0.8

    # Geometry overlap tolerance (pixels) before it is flagged.
    overlap_tolerance_px: float = 6.0

    # Collinear merge tolerance (pixels) for duplicate line segments.
    collinear_merge_tolerance_px: float = 15.0

    properties: dict[str, object] = field(default_factory=dict)


def _load_rules() -> TactileRules:
    rules = TactileRules()
    overrides_raw = os.getenv("TACTILE_OVERRIDES", "")
    if not overrides_raw:
        return rules
    try:
        overrides = json.loads(overrides_raw)
    except json.JSONDecodeError:
        return rules
    for key, value in overrides.items():
        if hasattr(rules, key):
            setattr(rules, key, value)
        else:
            rules.properties[key] = value
    return rules


TACTILE_RULES = _load_rules()
