"""Shared canonical Model B payloads.

`valid_document()` is the single reference instance of a well-formed response.
Reusing it across phases keeps each test focused on one transformation instead
of re-deriving 200 lines of plausible-looking JSON.
"""

from __future__ import annotations

import copy
from typing import Any

from app.model_b.schema import SCHEMA_VERSION

# Triangle, side BC touching a circle at one point (tangent), apex occluded by
# a scrawl. Exercises mapped entities, an unmapped diagram relation, a
# null-text item, and an occluded entity in one document.
VALID_DOCUMENT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "image": {"width_px": 1000, "height_px": 800},
    "diagram_summary": {
        "kind": "geometric_construction",
        "description": "A labelled triangle with a circle tangent to side BC.",
    },
    "entities": [
        {
            "id": "e1",
            "kind": "triangle",
            "detection_kind": "observed",
            "bbox": [0.10, 0.10, 0.70, 0.80],
            "confidence": "certain",
            "evidence": "three straight sides meeting at three closed corners",
            "occluded": False,
            "label": None,
        },
        {
            "id": "e2",
            "kind": "circle",
            "detection_kind": "observed",
            "bbox": [0.55, 0.30, 0.85, 0.60],
            "confidence": "certain",
            "evidence": "closed curve of constant curvature",
            "occluded": False,
            "label": None,
        },
        {
            "id": "e3",
            "kind": "point",
            "detection_kind": "observed",
            "bbox": [0.38, 0.40, 0.42, 0.44],
            "confidence": "likely",
            "evidence": "filled dot where two sides meet",
            "occluded": True,
            "label": "A",
        },
    ],
    "relationships": [
        {
            "id": "r1",
            "kind": "connected_to",
            "from_id": "e1",
            "to_id": "e2",
            "detection_kind": "observed",
            "confidence": "likely",
            "evidence": "the circle rests on the triangle's lower side",
        }
    ],
    "diagram_relations": [
        {
            "id": "d1",
            "kind": "tangent",
            "subject_ids": ["e1", "e2"],
            "statement": "The circle touches side BC at exactly one point.",
            "detection_kind": "observed",
            "confidence": "likely",
            "evidence": "the curve meets the straight side at a single contact point",
        },
        {
            "id": "d2",
            "kind": "right_triangle",
            "subject_ids": ["e1"],
            "statement": "The triangle may be right-angled at A.",
            "detection_kind": "inferred",
            "confidence": "uncertain",
            "evidence": "apex angle looks near 90 degrees but no marker is present",
        },
    ],
    "text_items": [
        {
            "id": "t1",
            "text": "BC",
            "bbox": [0.20, 0.82, 0.26, 0.86],
            "role": "side_label",
            "confidence": "certain",
            "evidence": "two capital letters beneath the lower side",
        },
        {
            "id": "t2",
            "text": None,
            "bbox": [0.60, 0.62, 0.72, 0.70],
            "role": "measurement",
            "confidence": "unreadable",
            "evidence": "handwritten digits with an ambiguous middle character",
        },
    ],
    "uncertainties": [
        {
            "id": "u1",
            "subject_ids": ["e3"],
            "kind": "occluded_geometry",
            "note": "A scrawl crosses the apex, so the corner may be misread.",
            "severity": "worth_review",
        }
    ],
    "image_quality": {"readable": True, "issues": ["handwriting"]},
}


def valid_document() -> dict[str, Any]:
    """A fresh deep copy, safe for a test to mutate."""
    return copy.deepcopy(VALID_DOCUMENT)


def minimal_document() -> dict[str, Any]:
    """The smallest document the contract allows: an empty, readable page."""
    return {
        "schema_version": SCHEMA_VERSION,
        "image": {"width_px": 4, "height_px": 4},
        "diagram_summary": {
            "kind": "not_a_geometry_diagram",
            "description": "Blank page.",
        },
        "entities": [],
        "relationships": [],
        "diagram_relations": [],
        "text_items": [],
        "uncertainties": [],
        "image_quality": {"readable": True, "issues": ["none"]},
    }
