"""Run the real pipeline over every generated fixture and print comparable metrics.

Used to produce the before/after table for the Tier 1 + Tier 2 report.

    python scripts/pipeline_metrics.py
    python scripts/pipeline_metrics.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pipeline import build_full_analysis  # noqa: E402

FIXTURES = [
    "triangle_worksheet.png",
    "circle_worksheet.png",
    "ellipse_worksheet.png",
    "complex_worksheet.png",
    "labelled_triangle_worksheet.png",
    "blurred_worksheet.png",
    "dark_worksheet.png",
]


def count_types(elements) -> dict:
    counts: dict = {}
    for element in elements or []:
        key = str(getattr(element, "type", None) or (element.get("type") if isinstance(element, dict) else "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def element_type(element) -> str:
    raw = getattr(element, "type", None)
    if raw is None and isinstance(element, dict):
        raw = element.get("type")
    # DetectedElement.type is a GeometryType enum; normalise to its bare value.
    return str(getattr(raw, "value", raw) or "unknown")


def element_text(element) -> str:
    if isinstance(element, dict):
        return str(element.get("text", ""))
    return str(getattr(element, "text", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args()

    fixtures_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    rows = []

    for name in FIXTURES:
        path = fixtures_dir / name
        if not path.exists():
            continue
        result = build_full_analysis(path.read_bytes())
        semantic = result.semantic_geometry
        simplified = result.simplified_geometry
        qa = result.qa_report
        issues = qa.issues or []
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        row = {
            "fixture": name,
            "semantic_elements": semantic.element_count,
            "semantic_types": count_types(semantic.elements),
            "angles": sum(1 for e in semantic.elements if element_type(e) == "angle"),
            "angle_sides": sum(1 for e in semantic.elements if element_type(e) in ("line", "line_segment")),
            "simplified_elements": len(simplified.elements),
            "simplification_actions": len(simplified.actions),
            "removed_count": simplified.removed_count,
            "merged_count": simplified.merged_count,
            "raw_shapes": len(result.shapes),
            "ocr_labels": sorted(element_text(l) for l in result.labels if element_text(l)),
            "braille_labels": len([l for l in result.labels if (l.get("braille") if isinstance(l, dict) else getattr(l, "braille", None))]),
            "qa_score": qa.score_0_100,
            "qa_passes": qa.passes,
            "qa_errors": len(errors),
            "qa_warnings": len(warnings),
            "qa_issue_keys": len(issues),
            "blocking_checks": sorted({i.check for i in errors}),
            "has_svg": bool(result.tactile_svg),
            "braille_glyphs": sum(1 for ch in (result.tactile_svg or "") if "\u2800" <= ch <= "\u28ff"),
        }
        rows.append(row)

        if not args.json:
            print(f"\n=== {name} ===")
            print(f"  raw shapes         : {row['raw_shapes']}")
            print(f"  semantic elements  : {row['semantic_elements']} {row['semantic_types']}")
            print(f"  angles             : {row['angles']}")
            print(
                f"  simplified         : {row['simplified_elements']} "
                f"(actions={row['simplification_actions']} removed={row['removed_count']} "
                f"merged={row['merged_count']})"
            )
            print(f"  ocr labels         : {row['ocr_labels']}")
            print(
                f"  qa                 : score={row['qa_score']} passes={row['qa_passes']} "
                f"errors={row['qa_errors']} warnings={row['qa_warnings']} keys={row['qa_issue_keys']}"
            )
            if row["blocking_checks"]:
                print(f"  blocking checks    : {row['blocking_checks']}")
            print(f"  svg / braille glyphs: {row['has_svg']} / {row['braille_glyphs']}")

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("\n" + "=" * 78)
        print(
            f"{'fixture':<30}{'raw':>5}{'sem':>5}{'ang':>5}{'simp':>6}"
            f"{'score':>7}{'err':>5}{'warn':>6}"
        )
        for row in rows:
            print(
                f"{row['fixture']:<30}{row['raw_shapes']:>5}{row['semantic_elements']:>5}"
                f"{row['angles']:>5}{row['simplified_elements']:>6}{row['qa_score']:>7}"
                f"{row['qa_errors']:>5}{row['qa_warnings']:>6}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
