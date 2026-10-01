"""Derive the provider-facing JSON Schema from the Pydantic contract.

`schema.py` owns the contract; this module projects it into the strict
subset that OpenRouter's `response_format: json_schema` accepts, and asserts
that projection stays inside the supported subset. The assertion is the point:
an unsupported keyword fails the test suite instead of failing silently in
production.

Deliberate projections away from raw Pydantic output:

* `title` - auto-generated noise ("Bbox", "Width Px") that only costs tokens.
* `default` - not part of the strict structured-output subset.
* `$defs`/`$ref` - inlined. Inlining removes a class of failure and keeps the
  payload flat enough for an upstream provider to honour.

Deliberate *non*-projections: `description` is retained. It is supported, and
the per-field contracts (e.g. "Null whenever any character is uncertain") are
exactly the kind of constraint that belongs in the schema rather than in a
duplicated prompt block.

The output is provider-neutral on purpose: OpenRouter normalises it to whichever
upstream model serves the request, so nothing here names a vendor.
"""

from __future__ import annotations

import copy
from typing import Any

from .schema import ModelBDocument

# Every keyword permitted by OpenRouter's documented strict structured-output
# subset.
# Anything outside this set is a bug, not a style preference.
_SUPPORTED_OBJECT_KEYWORDS = {"type", "properties", "required", "additionalProperties", "description"}
_SUPPORTED_ARRAY_KEYWORDS = {"type", "items", "prefixItems", "minItems", "maxItems", "description"}
_SUPPORTED_STRING_KEYWORDS = {"type", "enum", "description"}
_SUPPORTED_NUMBER_KEYWORDS = {"type", "enum", "minimum", "maximum", "description"}
_SUPPORTED_INTEGER_KEYWORDS = {"type", "enum", "minimum", "maximum", "description"}
_SUPPORTED_BOOLEAN_KEYWORDS = {"type", "description"}
_SUPPORTED_NULL_KEYWORDS = {"type"}
_SUPPORTED_COMBINATOR_KEYWORDS = {"anyOf"}
_SUPPORTED_KEYWORDS = (
    _SUPPORTED_OBJECT_KEYWORDS
    | _SUPPORTED_ARRAY_KEYWORDS
    | _SUPPORTED_STRING_KEYWORDS
    | _SUPPORTED_NUMBER_KEYWORDS
    | _SUPPORTED_INTEGER_KEYWORDS
    | _SUPPORTED_BOOLEAN_KEYWORDS
    | _SUPPORTED_NULL_KEYWORDS
    | _SUPPORTED_COMBINATOR_KEYWORDS
)

_TYPES_WITH_KEYWORDS = {
    "object": _SUPPORTED_OBJECT_KEYWORDS,
    "array": _SUPPORTED_ARRAY_KEYWORDS,
    "string": _SUPPORTED_STRING_KEYWORDS,
    "number": _SUPPORTED_NUMBER_KEYWORDS,
    "integer": _SUPPORTED_INTEGER_KEYWORDS,
    "boolean": _SUPPORTED_BOOLEAN_KEYWORDS,
    "null": _SUPPORTED_NULL_KEYWORDS,
}


class UnsupportedSchemaError(ValueError):
    """Raised when the derived schema leaves the strict structured-output subset."""


def _assert_supported(node: Any, path: str = "$") -> None:
    if isinstance(node, bool) or not isinstance(node, dict):
        return

    unknown = set(node) - _SUPPORTED_KEYWORDS
    if unknown:
        raise UnsupportedSchemaError(
            f"{path}: unsupported keyword(s) {sorted(unknown)}; "
            f"allowed here: {sorted(_SUPPORTED_KEYWORDS)}"
        )

    node_type = node.get("type")
    if node_type in _TYPES_WITH_KEYWORDS:
        illegal = set(node) - _TYPES_WITH_KEYWORDS[node_type] - _SUPPORTED_COMBINATOR_KEYWORDS
        if illegal:
            raise UnsupportedSchemaError(
                f"{path}: {sorted(illegal)} not valid for type {node_type!r}"
            )

    for combinator in ("anyOf",):
        for index, branch in enumerate(node.get(combinator, [])):
            _assert_supported(branch, f"{path}.{combinator}[{index}]")

    for name, child in node.get("properties", {}).items():
        _assert_supported(child, f"{path}.{name}")

    if isinstance(node.get("items"), dict):
        _assert_supported(node["items"], f"{path}.items")

    for index, child in enumerate(node.get("prefixItems", [])):
        _assert_supported(child, f"{path}.prefixItems[{index}]")


def _clean(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline $refs and drop Pydantic noise."""
    if isinstance(node, list):
        return [_clean(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/$defs/"):
            raise UnsupportedSchemaError(f"unexpected $ref target {ref!r}")
        target = ref.rsplit("/", 1)[-1]
        resolved = _clean(copy.deepcopy(defs[target]), defs)
        merged = {key: value for key, value in node.items() if key != "$ref"}
        if merged:
            resolved.update(merged)
        return resolved

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"$defs", "title", "default"}:
            continue
        if key == "const":
            # Pydantic emits `const` for a single-valued Literal; the strict
            # subset expresses that as a one-element `enum`.
            cleaned["enum"] = [value]
            continue
        cleaned[key] = _clean(value, defs)
    return cleaned


def build_response_json_schema() -> dict[str, Any]:
    """Return the response schema for `response_format.json_schema`.

    Raises:
        UnsupportedSchemaError: if the projection drifts out of the supported
            subset. Called on every request, so drift is caught immediately.
    """
    raw = ModelBDocument.model_json_schema(mode="serialization")
    defs = raw.get("$defs", {})
    schema = _clean(raw, defs)
    _assert_supported(schema)
    return schema


#: Name sent in `response_format.json_schema.name`. Also the schema's own
#: identity, so a provider error quoting it is traceable to this contract.
RESPONSE_SCHEMA_NAME = "model_b_document"
