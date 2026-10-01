#!/usr/bin/env python
"""Model B evaluation runner (Phase 6).

Runs the real `analyze_image` pipeline over a set of ground-truth fixtures and
writes a JSON report to stdout or a file.

    # Default: hermetic. Uses a scripted client, so the run is reproducible
    # and needs no key, no network, and costs nothing.
    python scripts/evaluate_model_b.py

    # Live: real OpenRouter calls. Requires OPENROUTER_API_KEY.
    python scripts/evaluate_model_b.py --live --out report.json

The distinction matters and is recorded in the output. A run with the scripted
client proves the pipeline, the contract, and the scoring arithmetic; it says
nothing whatsoever about how well the vision model reads a worksheet. The report labels
which of the two produced it via `"evidence_source"`, so a hermetic run can
never be quoted as a provider result.

Latency from a scripted-client run is the pipeline's own cost (prepare, parse,
validate, normalise) and excludes network time. It is reported as
`latency_scope: "pipeline_only"` for that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_b.client import ProviderResponse  # noqa: E402
from app.model_b.errors import ModelBError  # noqa: E402
from app.model_b.evaluation import (  # noqa: E402
    EvidenceSource,
    FixtureOutcome,
    fusion_counts,
    score_detections,
    score_relations,
    summarise,
)
from app.model_b.fusion import reconcile  # noqa: E402
from app.model_b.service import ModelBSettings, analyze_image  # noqa: E402
from app.core.config import model_b_settings  # noqa: E402


class ScriptedClient:
    """Replays one canned document so the harness runs without a provider.

    It is a `ModelBClientProtocol` implementation in the same way the test fake
    is: it proves that the orchestration around a provider is correct, not that
    the provider reads diagrams well.

    It never sees, stores, or transmits an API key.
    """

    def __init__(self, text: str, finish_reason: str = "STOP") -> None:
        self._text = text
        self._finish_reason = finish_reason
        self.calls = 0

    def analyze(
        self,
        image_png: bytes,
        prepared_width: int,
        prepared_height: int,
    ) -> ProviderResponse:
        self.calls += 1
        if not image_png:
            raise RuntimeError("ScriptedClient received an empty image")
        return ProviderResponse(text=self._text, finish_reason=self._finish_reason)


def hermetic_settings() -> ModelBSettings:
    """Settings for a run with an injected client.

    The key is a literal placeholder. It is never read, never logged, and never
    leaves the process: `ScriptedClient` replaces the provider client entirely, so
    nothing here can become a credential. `enabled=True` is what lets
    `analyze_image` proceed past its availability gate in a hermetic run.
    """
    return ModelBSettings(
        enabled=True,
        api_key="hermetic-placeholder-unused",
        model="scripted",
        timeout_s=30.0,
    )


def _png(width: int, height: int, value: int = 220) -> bytes:
    import cv2
    import numpy as np

    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), value, dtype=np.uint8))
    if not ok:
        raise RuntimeError("failed to encode fixture image")
    return encoded.tobytes()


def _document(
    *,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    kind: str = "geometric_construction",
    description: str = "A single labelled triangle.",
    readable: bool = True,
) -> dict[str, Any]:
    """A hand-authored response document in the real nested contract shape.

    Written out rather than adapted from `valid_document()` so that the
    expected regions below are a human claim about the image, and the document
    here is an independent human claim about what a model would say. Deriving
    the expected boxes from the response would make the scoring vacuous.
    """
    return {
        "schema_version": "model_b.diag.v1",
        "image": {"width_px": 1000, "height_px": 800},
        "diagram_summary": {"kind": kind, "description": description},
        "entities": entities,
        "relationships": [],
        "diagram_relations": relations,
        "text_items": [],
        "uncertainties": [],
        "image_quality": {"readable": readable, "issues": []},
    }


def _entity(
    entity_id: str,
    kind: str,
    norm_bbox: list[float],
    *,
    confidence: str = "likely",
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "kind": kind,
        "detection_kind": "observed",
        "bbox": norm_bbox,
        "confidence": confidence,
        "evidence": f"visible {kind} outline",
        "occluded": False,
        "label": None,
    }


def _load_fixtures() -> list[dict[str, Any]]:
    """Ground truth plus the response the scripted client will replay.

    The expected boxes below are in Model A space (original image pixels). The
    image is 1000x800, so a normalised box maps back by multiplying by the
    image width and height. The values here and the normalised boxes in
    `_document` are written to correspond, which the runner's own assertions
    check in `tests/test_model_b_evaluation.py`.
    """
    return [
        {
            "name": "single_triangle",
            "image": _png(1000, 800),
            "expected_entities": [("triangle", [100.0, 80.0, 800.0, 720.0])],
            "expected_relations": [],
            "response": _document(
                entities=[_entity("e1", "triangle", [0.10, 0.10, 0.80, 0.90])],
                relations=[],
            ),
        },
        {
            "name": "blank_page",
            "image": _png(800, 600, value=255),
            "expected_entities": [],
            "expected_relations": [],
            "response": _document(
                entities=[],
                relations=[],
                kind="unclear",
                description="A blank worksheet page with no marks.",
            ),
        },
    ]


def run_fixture(
    fixture: dict[str, Any],
    client: Any,
    evidence_source: EvidenceSource,
    settings: ModelBSettings,
) -> FixtureOutcome:
    outcome = FixtureOutcome(fixture=fixture["name"], evidence_source=evidence_source)

    started = time.perf_counter()
    try:
        result = analyze_image(fixture["image"], settings, client=client)
    except ModelBError as error:
        outcome.ok = False
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        outcome.error_code = error.reason
        outcome.error_message = error.detail
        return outcome
    outcome.latency_ms = (time.perf_counter() - started) * 1000

    # Model B's regions are already projected back into original-image pixels by
    # the normaliser, so they are directly comparable to the hand-written
    # ground truth, which is also in original-image pixels.
    predicted_entities: list[tuple[str, Sequence[float]]] = [
        (entity.geometry_type.value if entity.geometry_type else "unmapped",
         entity.region.as_model_a_bbox())
        for entity in result.entities
    ]
    outcome.entity_score = score_detections(
        predicted_entities, list(fixture["expected_entities"])
    )
    outcome.relation_score = score_relations(
        [relation.statement for relation in result.diagram_relations],
        fixture["expected_relations"],
    )

    model_a = fixture.get("model_a")
    if model_a is not None:
        # Recorded because PROJECT.md section 13 asks for the fusion result, and
        # because an advisory layer whose agreements are never measured is an
        # advisory layer nobody can tell apart from noise.
        outcome.fusion = fusion_counts(reconcile(model_a, result))

    if result.truncated:
        outcome.notes.append("response was truncated")
    if not result.image_readable:
        outcome.notes.append("image reported unreadable")
    if outcome.relation_score and outcome.relation_score.false_negatives:
        outcome.notes.append("expected relations were missed")
    if outcome.entity_score and outcome.entity_score.false_negatives:
        outcome.notes.append("expected regions were missed")
    if outcome.evidence_source != evidence_source:
        # Defensive: outcome and run must agree on provenance.
        raise AssertionError("evidence source mismatch")
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the real OpenRouter API. Requires OPENROUTER_API_KEY.",
    )
    parser.add_argument("--out", type=Path, help="Write the JSON report here.")
    args = parser.parse_args(argv)

    fixtures = _load_fixtures()

    evidence_source: EvidenceSource = "fake_client" if not args.live else "live_provider"
    if args.live:
        settings = model_b_settings()
        try:
            settings.require_available()
        except ModelBError as error:
            # Refusing to continue is the whole point. A silently degraded
            # "live" run that quietly used the scripted client would produce a
            # report indistinguishable from a real one.
            print(
                json.dumps(
                    {
                        "error": "live evaluation unavailable",
                        "reason": error.reason,
                        "detail": error.detail,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        client = settings.build_client()

        def client_for(fixture: dict[str, Any]) -> Any:
            return client
    else:
        settings = hermetic_settings()

        def client_for(fixture: dict[str, Any]) -> Any:
            return ScriptedClient(json.dumps(fixture["response"]))

    outcomes = [
        run_fixture(fixture, client_for(fixture), evidence_source, settings)
        for fixture in fixtures
    ]

    report = {
        "evidence_source": evidence_source,
        "latency_scope": (
            "pipeline_only" if not args.live else "includes_provider_round_trip"
        ),
        "summary": summarise(outcomes),
        "fixtures": [outcome.as_dict() for outcome in outcomes],
    }

    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
