#!/usr/bin/env python
"""LIVE OpenRouter smoke test for Model B. Requires a real OPENROUTER_API_KEY.

    OPENROUTER_API_KEY=<key> python scripts/model_b_live_smoke.py

This is deliberately separate from the automated suite. Every normal test uses a
fake client, so the suite never needs a key, a network, or money. This script is
the one place a real provider call is made, and it is opt-in by construction:
without a key it refuses rather than falling back to anything fake.

What it does, end to end:

    image -> prepare -> OpenRouter -> structured JSON -> validate -> normalize

What it deliberately does NOT do: run Model A, write a session, or let any
output reach geometry. Model B is advisory, and the most important property to
demonstrate is that a real provider response can be turned into a validated
advisory result while leaving the authoritative pipeline untouched.

Safety:
  * the API key is read from the environment and never printed, logged, or
    written to disk, not even in truncated form;
  * the worksheet is generated in memory and never persisted;
  * only aggregate counts and short quoted statements reach stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import model_b_settings  # noqa: E402
from app.model_b.errors import ModelBError  # noqa: E402
from app.model_b.service import analyze_image  # noqa: E402


def synthetic_worksheet() -> bytes:
    """A representative geometry worksheet, drawn in memory.

    A triangle with an incircle, a right-angle marker at the base, a tick-marked
    pair of equal sides, and one deliberately ambiguous label. Chosen because it
    contains the features Model B exists to find, so a real run has something
    meaningful to report.
    """
    image = np.full((900, 1200, 3), 252, dtype=np.uint8)
    ink = (25, 25, 28)

    a = (330, 200)
    b = (900, 200)
    c = (610, 720)
    for start, end in ((a, b), (b, c), (c, a)):
        cv2.line(image, start, end, ink, 5)

    # Incircle.
    cv2.circle(image, (612, 380), 150, ink, 5)
    # Right-angle marker at the bottom-left vertex.
    cv2.line(image, c, (700, 720), ink, 3)
    cv2.line(image, (700, 720), (700, 630), ink, 3)
    # Single tick marks on the two slanted sides: an isosceles hint.
    for point, angle in ((a, 0.5), (c, -0.5)):
        cv2.line(image, (point[0] - 14, point[1] - 26), (point[0] + 14, point[1] - 8), ink, 3)

    for text, origin in (("A", (300, 175)), ("B", (905, 205)), ("C", (595, 765)), ("BC", (610, 760))):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.1, ink, 3)

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode the synthetic worksheet")
    return encoded.tobytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Override OPENROUTER_MODEL for this run. The name is printed, "
        "but never the key.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Override MODEL_B_TIMEOUT_S."
    )
    parser.add_argument(
        "--show-text", action="store_true", help="Print the raw Model B JSON body."
    )
    args = parser.parse_args(argv)

    # Presence check only. The value never leaves this process.
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        print(
            "OPENROUTER_API_KEY is not set.\n"
            "Live testing needs a real key. Every other test in this repository "
            "runs without one.",
            file=sys.stderr,
        )
        return 2

    settings = model_b_settings()
    model = args.model or settings.model
    timeout = args.timeout if args.timeout is not None else settings.timeout_s

    if not settings.enabled:
        print(
            "MODEL_B_ENABLED is not true; refusing to call the provider.\n"
            "Re-run with MODEL_B_ENABLED=true for a deliberate live test.",
            file=sys.stderr,
        )
        return 2

    # `analyze_image` requires an available configuration; build one explicitly
    # from the environment key so the run uses exactly the same code path
    # production uses, only with a model/timeout override if requested.
    #
    # `replace` rather than a field-by-field rebuild: constructing the settings
    # afresh silently resets every field not named here to its dataclass default,
    # so a run would quietly use 3 attempts and 30s even when the operator
    # configured 1 and 7. A live test that does not reflect production config is
    # worse than no test, because it looks like evidence.
    from dataclasses import replace

    live_settings = replace(
        settings,
        enabled=True,
        api_key=settings.api_key,
        model=model,
        timeout_s=timeout,
    )
    live_settings.require_available()

    image = synthetic_worksheet()
    print("Model B live smoke test")
    print(f"  provider        : OpenRouter")
    print(f"  requested model : {model}")
    print(f"  timeout         : {timeout}s")
    print(f"  max attempts    : {live_settings.max_attempts}")
    print(f"  image           : synthetic worksheet, {len(image):,} bytes PNG (not written to disk)")
    print(f"  api key         : present (value never printed)")

    started = time.perf_counter()
    try:
        result = analyze_image(image, live_settings)
    except ModelBError as error:
        elapsed = (time.perf_counter() - started) * 1000
        print(f"\nFAILED after {elapsed:.0f}ms")
        print(f"  typed error     : {type(error).__name__}")
        print(f"  reason          : {error.reason}")
        print(f"  teacher message : {error.detail}")
        print(f"  retryable       : {getattr(error, 'retryable', False)}")
        # Printed BEFORE returning. Previously `--show-text` sat at the end of the
        # success path, so a run that failed could never show what the model
        # actually said — which is exactly the run you need to inspect.
        diagnostic = getattr(error, "diagnostic", "")
        if diagnostic:
            print("\n--- diagnostics (redacted, MODEL_B_DIAGNOSTICS was on) ---")
            print(diagnostic)
        elif os.environ.get("MODEL_B_DIAGNOSTICS"):
            print(
                "\n--- diagnostics requested but none captured: the failure "
                "happened before the transport responded ---"
            )
        print(
            "\nModel A is unaffected by this: the failure is confined to the "
            "advisory layer and the request returns a typed error, not a crash."
        )
        return 1

    elapsed = (time.perf_counter() - started) * 1000

    print(f"\nOK in {elapsed:.0f}ms")
    print("\n--- model identity ---")
    print(f"  requested       : {result.requested_model}")
    # For a router like `openrouter/free` this differs from the requested id and
    # is the only way to know which model answered.
    print(f"  resolved        : {result.resolved_model or 'not reported'}")
    print(f"  upstream vendor : {result.upstream_provider or 'not reported'}")
    print(f"  request id      : {result.request_id or 'not reported'}")
    print(f"  usage           : {result.usage or 'not reported'}")

    print("\n--- Model B structured result (advisory only) ---")
    print(f"  schema_version        : {result.schema_version}")
    print(f"  diagram kind          : {result.diagram_kind}")
    print(f"  description           : {result.diagram_description}")
    print(f"  image (prepared)      : {result.image_width} x {result.image_height}")
    print(f"  schema validation     : PASSED ({len(result.validation_warnings)} warning(s))")
    for warning in result.validation_warnings:
        print(f"      - {warning}")
    print(f"  truncated             : {result.truncated}")
    print(f"  entities              : {len(result.entities)}")
    print(f"      mapped to Model A : {result.mapped_entity_count}")
    print(f"      advisory only      : {result.unmapped_entity_count}")
    print(f"      needs review       : {result.review_entity_count}")
    print(f"  diagram relations     : {len(result.diagram_relations)}")
    for relation in result.diagram_relations:
        print(f"      [{relation.detection_kind}/{relation.confidence_level}] {relation.statement}")
    print(f"  text items            : {len(result.text_items)}")
    for item in result.text_items:
        # A null text is the honest outcome for illegible text, and must never
        # be silently replaced with a guess.
        print(f"      {item.role}: {item.text!r} (confidence {item.confidence_level})")
    print(f"  uncertainties         : {len(result.uncertainties)}")
    for note in result.uncertainties:
        print(f"      [{note.severity}] {note.note}")
    print(f"  image readable        : {result.image_readable}")
    print(f"  image quality issues  : {result.image_quality_issues or 'none'}")

    # Model B is advisory-only. That is an architectural property, not a value
    # carried on the result: `ModelBResult` is a plain dataclass and has no
    # `advisory_only` field, because there is nothing for it to assert. The
    # guarantee is enforced where it is meaningful — Model A's geometry, the
    # SVG, and the export decision are never read or written by this layer — and
    # is asserted by tests rather than printed here.
    print("\n--- advisory-only guarantees ---")
    print(
        "  Advisory only            : by construction. This layer produced no\n"
        "                             point, radius, centre, or angle, so it\n"
        "                             supplied nothing that could become a\n"
        "                             measurement in Model A."
    )
    print(
        f"  Nothing applied          : {result.unmapped_entity_count} of "
        f"{len(result.entities)} region(s) have no Model A equivalent and remain\n"
        "                             advisory; no geometry, SVG, Braille, or\n"
        "                             export decision was produced or influenced."
    )

    if args.show_text:
        print("\n--- normalized result (dataclasses.asdict) ---")
        # `ModelBResult` is a dataclass, not a Pydantic model, so it has no
        # `model_dump`. `dataclasses.asdict` is the equivalent and recurses into
        # the nested dataclasses, which is what makes the entity dump readable.
        print(json.dumps(asdict(result), indent=2, default=str))
        print(
            "\n--- note ---\n"
            "The dump above is the normalized result, not the provider response.\n"
            "Set MODEL_B_DIAGNOSTICS=1 to also capture the raw envelope."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
