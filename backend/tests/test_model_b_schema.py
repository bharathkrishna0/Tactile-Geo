"""Phase 0 contract tests: schema, derived response schema, prompt."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.model_b._fixtures import minimal_document, valid_document
from app.model_b.json_schema import UnsupportedSchemaError, _assert_supported, build_response_json_schema
from app.model_b.prompts import (
    COORDINATE_RULES,
    PRIORITY_RULES,
    TEXT_RULES,
    build_prompt,
)
from app.model_b.schema import (
    MAX_ENTITIES,
    SCHEMA_VERSION,
    Entity,
    ModelBDocument,
)


class TestPydanticContract:
    def test_canonical_document_validates(self) -> None:
        assert ModelBDocument.model_validate(valid_document()).schema_version == SCHEMA_VERSION

    def test_minimal_document_validates(self) -> None:
        assert ModelBDocument.model_validate(minimal_document()).entities == []

    def test_unknown_field_rejected(self) -> None:
        """`extra="forbid"` is what makes the contract closed rather than open."""
        doc = valid_document()
        doc["entities"][0]["radius_px"] = 12
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_bbox_must_be_four_normalized_numbers(self) -> None:
        doc = valid_document()
        doc["entities"][0]["bbox"] = [0.1, 0.2, 1.4, 0.9]
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_bbox_wrong_length_rejected(self) -> None:
        doc = valid_document()
        doc["entities"][0]["bbox"] = [0.1, 0.2, 0.3]
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_id_outside_bounded_enum_rejected(self) -> None:
        doc = valid_document()
        doc["entities"][0]["id"] = "entity_1"
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_confidence_band_is_closed(self) -> None:
        doc = valid_document()
        doc["entities"][0]["confidence"] = "very_high"
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_entity_count_bounded(self) -> None:
        doc = minimal_document()
        template = valid_document()["entities"][0]
        doc["entities"] = [{**template, "id": f"e{i + 1}"} for i in range(MAX_ENTITIES + 1)]
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)

    def test_text_item_may_be_null_but_must_be_present(self) -> None:
        doc = valid_document()
        del doc["text_items"][0]["text"]
        with pytest.raises(ValidationError):
            ModelBDocument.model_validate(doc)


class TestResponseSchemaDerivation:
    def test_derivation_is_self_contained(self) -> None:
        """$defs/$ref inlining is a projection choice, not an accident."""
        blob = json.dumps(build_response_json_schema())
        assert "$defs" not in blob
        assert "$ref" not in blob

    def test_strips_pydantic_noise(self) -> None:
        blob = json.dumps(build_response_json_schema())
        assert '"title"' not in blob
        assert '"const"' not in blob
        assert '"default"' not in blob

    def test_schema_version_is_becomes_one_element_enum(self) -> None:
        node = build_response_json_schema()["properties"]["schema_version"]
        assert node == {"enum": [SCHEMA_VERSION], "type": "string"}

    def test_ids_survive_as_bounded_enums(self) -> None:
        entity = build_response_json_schema()["properties"]["entities"]["items"]
        assert entity["properties"]["id"]["enum"] == [f"e{i + 1}" for i in range(MAX_ENTITIES)]

    def test_bbox_uses_prefixitems_with_bounds(self) -> None:
        bbox = build_response_json_schema()["properties"]["entities"]["items"]["properties"]["bbox"]
        assert bbox["minItems"] == bbox["maxItems"] == 4
        assert len(bbox["prefixItems"]) == 4
        for slot in bbox["prefixItems"]:
            assert slot == {"type": "number", "minimum": 0.0, "maximum": 1.0}

    def test_objects_forbid_extra_properties(self) -> None:
        schema = build_response_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["entities"]["items"]["additionalProperties"] is False

    def test_optional_label_is_nullable_not_required(self) -> None:
        entity = build_response_json_schema()["properties"]["entities"]["items"]
        assert "label" not in entity["required"]
        assert {"type": "null"} in entity["properties"]["label"]["anyOf"]

    def test_retains_meaningful_descriptions(self) -> None:
        """Descriptions carry the honesty rules, so they must survive."""
        schema = build_response_json_schema()
        assert "Null whenever any character is uncertain" in schema["properties"]["text_items"]["items"]["properties"]["text"]["description"]

    def test_derived_schema_matches_canonical_payload_keys(self) -> None:
        """Contract test: the shape we send is the shape we can read back."""
        entity_schema = build_response_json_schema()["properties"]["entities"]["items"]
        assert set(entity_schema["properties"]) == set(valid_document()["entities"][0])
        assert set(entity_schema["required"]) == set(
            key for key in valid_document()["entities"][0] if key != "label"
        )

    def test_whole_schema_serializes(self) -> None:
        assert isinstance(json.dumps(build_response_json_schema()), str)

    def test_subset_guard_rejects_pattern(self) -> None:
        with pytest.raises(UnsupportedSchemaError):
            _assert_supported({"type": "string", "pattern": "^e[0-9]+$"})

    def test_subset_guard_rejects_oneof(self) -> None:
        with pytest.raises(UnsupportedSchemaError):
            _assert_supported({"oneOf": [{"type": "string"}]})

    def test_subset_guard_rejects_wrong_keyword_for_type(self) -> None:
        with pytest.raises(UnsupportedSchemaError):
            _assert_supported({"type": "string", "minimum": 0.0})

    def test_subset_guard_recurses_into_arrays(self) -> None:
        with pytest.raises(UnsupportedSchemaError):
            _assert_supported(
                {"type": "array", "items": {"type": "string", "pattern": "x"}}
            )

    def test_subset_guard_accepts_current_schema(self) -> None:
        _assert_supported(build_response_json_schema())


class TestPrompt:
    def test_prompt_is_deterministic(self) -> None:
        assert build_prompt() == build_prompt()

    def test_prompt_states_top_left_origin(self) -> None:
        assert "TOP-LEFT" in build_prompt()
        assert "normalized" in build_prompt()

    def test_prompt_forbids_text_reconstruction(self) -> None:
        assert "set text to null" in build_prompt()
        assert "Never reconstruct" in TEXT_RULES

    def test_prompt_declares_advisory_scope(self) -> None:
        assert "advisory" in build_prompt()
        assert "embossing decisions" in PRIORITY_RULES + build_prompt()

    def test_prompt_does_not_duplicate_the_schema(self) -> None:
        """Re-sending the schema in the prompt degrades structured output."""
        prompt = build_prompt()
        assert '"properties"' not in prompt
        assert '"required"' not in prompt
        assert "prefixItems" not in prompt
        assert "additionalProperties" not in prompt

    def test_prompt_lists_entity_and_relation_vocabularies(self) -> None:
        prompt = build_prompt()
        assert "tangent" in prompt
        assert "line_segment" in prompt
        assert "illegible_text" in prompt

    def test_prompt_mentions_normalized_frame_for_image_dimensions(self) -> None:
        assert "image as you see it" in COORDINATE_RULES

    def test_prompt_does_not_request_measurements(self) -> None:
        assert "Do not measure" in build_prompt()


class TestEntityHelper:
    def test_entity_requires_explicit_detection_kind(self) -> None:
        """Silently defaulting observed/inferred would hide hallucination."""
        payload = valid_document()["entities"][0]
        payload.pop("detection_kind")
        with pytest.raises(ValidationError):
            Entity.model_validate(payload)
