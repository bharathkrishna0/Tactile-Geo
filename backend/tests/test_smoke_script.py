"""The smoke script's reporting path must only read fields that exist.

A live smoke run is the one place in this repository where a wrong field name
costs a real provider call. `scripts/model_b_live_smoke.py` printed its report
*after* the request succeeded, so a single reference to a nonexistent attribute
crashed the process at the last possible moment — discarding a successful,
paid-for run and leaving only a traceback.

These tests are static. They parse the script and check every attribute it
reads against the dataclasses it reads them from, so a bad name fails in CI
rather than on a live run that has already worked.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from app.models.model_b_result import (
    DiagramRelationNote,
    ModelBResult,
    SuggestedEntity,
    SuggestedRelationship,
    TextNote,
    UncertaintyNote,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model_b_live_smoke.py"

#: Collection field on `ModelBResult` -> type of the items it yields.
_ITEM_TYPES = {
    "entities": SuggestedEntity,
    "relationships": SuggestedRelationship,
    "diagram_relations": DiagramRelationNote,
    "text_items": TextNote,
    "uncertainties": UncertaintyNote,
}


def _readable(cls: type) -> set[str]:
    """Attribute names that legitimately exist on a dataclass.

    Fields, properties, and methods all count. The bug this guards against is
    reading a name that exists nowhere, so anything genuinely reachable is fine.
    """
    names = {field.name for field in dataclasses.fields(cls)}
    names.update(
        name
        for name, value in vars(cls).items()
        if isinstance(value, property) or callable(value)
    )
    return names


def _resolve_variable_types(tree: ast.AST) -> dict[str, type]:
    """Infer the class of each variable the script iterates or holds.

    Only the patterns the script actually uses are inferred: `result` from
    `analyze_image`, and loop variables from `for x in result.<collection>`.
    A general type inference is not needed, and would obscure the intent.
    """
    types: dict[str, type] = {"result": ModelBResult}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        iterable = node.iter
        if not isinstance(iterable, ast.Attribute):
            continue
        if not isinstance(iterable.value, ast.Name) or iterable.value.id != "result":
            continue
        item_type = _ITEM_TYPES.get(iterable.attr)
        if item_type is not None:
            types[node.target.id] = item_type
    return types


def _bad_references(source: str) -> list[tuple[int, str, str]]:
    """Every `<var>.<attr>` in the script that the target class does not have."""
    tree = ast.parse(source)
    var_types = _resolve_variable_types(tree)
    problems: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        var = node.value.id
        cls = var_types.get(var)
        if cls is None or node.attr in _readable(cls):
            continue
        problems.append((node.lineno, var, node.attr))
    return problems


class TestSmokeScriptReporting:
    def test_every_attribute_the_script_reads_exists(self) -> None:
        problems = _bad_references(SCRIPT.read_text(encoding="utf-8"))
        assert not problems, (
            "scripts/model_b_live_smoke.py reads attributes that do not exist; "
            "a live run would crash after the provider call already succeeded: "
            + ", ".join(f"line {line} `{var}.{attr}`" for line, var, attr in problems)
        )

    def test_the_script_does_not_read_a_field_it_invented(self) -> None:
        """Regression guard for the specific defect.

        `ModelBResult` has no `advisory_only` attribute, and must not grow one:
        advisory-ness is an architectural property of the layer, not a per-result
        value. Adding the field to satisfy the script would invert the design.
        """
        assert not hasattr(ModelBResult, "advisory_only")

    def test_the_script_does_not_assume_pydantic_on_a_dataclass(self) -> None:
        """`ModelBResult` is a dataclass.

        `model_dump` is a Pydantic method and never existed on it. The script
        used it under `--show-text`, so that flag crashed on a successful run
        too, just less visibly than the unconditional one.
        """
        source = SCRIPT.read_text(encoding="utf-8")
        assert "result.model_dump" not in source
        assert not hasattr(ModelBResult, "model_dump")
        assert dataclasses.is_dataclass(ModelBResult)

    def test_the_script_reports_from_real_fields(self) -> None:
        """Spot-checks that the report is reading the normalised result.

        Cheap, and they fail loudly if a field is renamed.
        """
        source = SCRIPT.read_text(encoding="utf-8")
        for field in (
            "result.validation_warnings",
            "result.resolved_model",
            "result.requested_model",
            "result.unmapped_entity_count",
            "result.mapped_entity_count",
        ):
            assert field in source, f"report no longer reads {field}"

    @pytest.mark.parametrize("item_type", list(_ITEM_TYPES.values()))
    def test_each_reported_item_class_is_inspectable(self, item_type: type) -> None:
        """Every item the script iterates must be a real dataclass.

        The report dereferences fields on these directly, so an item model that
        is a dict or a Pydantic object would break the print rather than the
        analysis.
        """
        assert dataclasses.is_dataclass(item_type)
        assert _readable(item_type)
