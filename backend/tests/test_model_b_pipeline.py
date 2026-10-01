"""Phase 1 tests: preparation, client seam, parser, validator, normalizer, service."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.model_b._fixtures import minimal_document, valid_document
from app.model_b.client import (
    OpenRouterClient,
    ModelBClientProtocol,
    ProviderResponse,
)
from app.model_b.errors import (
    ModelBApiError,
    ModelBDisabled,
    ModelBError,
    ModelBMisconfigured,
    ModelBMalformedResponse,
    ModelBRateLimited,
    ModelBTimeout,
    ModelBValidationError,
)
from app.model_b.normalizer import normalize
from app.model_b.parser import extract_json_object
from app.model_b.preparation import (
    LONG_EDGE_TARGET,
    SHORT_EDGE_UPSCALE,
    PreparedImage,
    prepare_image,
)
from app.model_b.prompts import build_prompt
from app.model_b.service import ModelBSettings, analyze_image, is_available
from app.model_b.validator import validate_document
from app.models.model_b_result import (
    ENTITY_KIND_TO_GEOMETRY,
    RELATIONSHIP_KIND_TO_RELATIONSHIP,
    BoundingRegion,
    validate_mapping_tables,
)
from app.models.geometry import GeometryType, RelationshipType


def make_png(width: int, height: int, color: tuple[int, int, int] = (20, 20, 20)) -> bytes:
    array = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.line(array, (2, 2), (width - 3, height - 3), (255, 255, 255), 1)
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


class FakeClient:
    """Satisfies ModelBClientProtocol without an SDK, a network, or a key."""

    def __init__(self, text: str, finish_reason: str = "STOP", usage: dict | None = None) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.calls: list[dict] = []

    def analyze(self, *, image_png: bytes, prepared_width: int, prepared_height: int):
        self.calls.append(
            {"size": len(image_png), "width": prepared_width, "height": prepared_height}
        )
        return ProviderResponse(
            text=self.text, finish_reason=self.finish_reason, usage=self.usage
        )

    def raises(self, exc: Exception):
        """A client that fails, for error-propagation tests."""
        fake = FakeClient("{}")

        def _analyze(**_: object):
            raise exc

        fake.analyze = _analyze  # type: ignore[method-assign]
        return fake


ON = ModelBSettings(enabled=True, api_key="test-key", model="openrouter/free", timeout_s=5.0)


class TestPreparation:
    def test_produces_square_canvas(self) -> None:
        prepared = prepare_image(make_png(400, 700))
        assert prepared.width == prepared.height

    def test_preserves_original_dimensions(self) -> None:
        prepared = prepare_image(make_png(400, 700))
        assert (prepared.original_width, prepared.original_height) == (400, 700)

    def test_downscales_oversized_long_edge(self) -> None:
        prepared = prepare_image(make_png(3000, 1200))
        assert max(prepared.width, prepared.height) <= LONG_EDGE_TARGET

    def test_upscales_small_short_edge(self) -> None:
        prepared = prepare_image(make_png(200, 240))
        assert min(prepared.original_width, prepared.original_height) < SHORT_EDGE_UPSCALE
        assert prepared.content_width > prepared.original_width
        assert prepared.content_height > prepared.original_height

    def test_upscale_never_exceeds_long_edge_target(self) -> None:
        prepared = prepare_image(make_png(60, 60))
        assert prepared.width <= LONG_EDGE_TARGET

    def test_output_is_decodable_png(self) -> None:
        prepared = prepare_image(make_png(300, 300))
        decoded = cv2.imdecode(np.frombuffer(prepared.png_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (prepared.height, prepared.width)

    def test_rejects_undecodable_bytes(self) -> None:
        with pytest.raises(ValueError):
            prepare_image(b"not an image at all")

    def test_jpeg_input_accepted(self) -> None:
        array = np.full((300, 300, 3), 90, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", array)
        assert ok
        assert prepare_image(encoded.tobytes()).width > 0

    def test_identity_transform_when_no_resize_needed(self) -> None:
        prepared = PreparedImage(
            png_bytes=b"", width=100, height=100,
            original_width=100, original_height=100,
            content_width=100, content_height=100,
            pad_x=0, pad_y=0, scale_x=1.0, scale_y=1.0,
        )
        assert prepared.project_bbox((0.0, 0.0, 1.0, 1.0)) == (0, 0, 100, 100)
        assert prepared.project_bbox((0.25, 0.5, 0.75, 1.0)) == (25, 50, 50, 50)

    def test_projection_accounts_for_padding(self) -> None:
        """The bug this guards: treating a letterboxed norm coord as unscaled.

        Content occupies the middle 400px of a 1000px canvas, so the content
        starts at normalized 0.3, not 0.0. Scale is 1.0 here so the padding
        offset is the only thing under test.
        """
        prepared = PreparedImage(
            png_bytes=b"", width=1000, height=1000,
            original_width=400, original_height=400,
            content_width=400, content_height=400,
            pad_x=300, pad_y=300, scale_x=1.0, scale_y=1.0,
        )
        assert prepared.project_bbox((0.3, 0.3, 0.7, 0.7)) == (0, 0, 400, 400)
        assert prepared.project_bbox((0.4, 0.4, 0.5, 0.5)) == (100, 100, 100, 100)

    def test_projection_clamps_to_image(self) -> None:
        prepared = PreparedImage(
            png_bytes=b"", width=1000, height=1000,
            original_width=400, original_height=400,
            content_width=400, content_height=400,
            pad_x=300, pad_y=300, scale_x=1.0, scale_y=1.0,
        )
        x, y, w, h = prepared.project_bbox((0.0, 0.0, 1.0, 1.0))
        assert (x, y, w, h) == (0, 0, 400, 400)

    def test_projection_inside_padding_collapses_not_inverts(self) -> None:
        prepared = PreparedImage(
            png_bytes=b"", width=1000, height=1000,
            original_width=400, original_height=400,
            content_width=400, content_height=400,
            pad_x=300, pad_y=300, scale_x=1.0, scale_y=1.0,
        )
        x, y, w, h = prepared.project_bbox((0.0, 0.0, 0.1, 0.1))
        assert (x, y) == (0, 0)
        assert w >= 0 and h >= 0

    def test_projection_direction_is_upscale_not_downscale(self) -> None:
        """Guard against inverting the scale factor.

        Content is 800px wide for a 400px original, so the multiplier is 0.5.
        Getting the direction wrong yields 1600px for a 400px image, which
        clamping hides unless the box is strictly inside the content.
        """
        prepared = PreparedImage(
            png_bytes=b"", width=1400, height=1400,
            original_width=400, original_height=700,
            content_width=800, content_height=1400,
            pad_x=300, pad_y=0, scale_x=0.5, scale_y=0.5,
        )
        assert prepared.project_bbox((300 / 1400, 0.0, 1100 / 1400, 0.5)) == (
            0, 0, 400, 350
        )

    def test_letterbox_inverse_transform_is_self_consistent(self) -> None:
        """The projection must satisfy two independent identities.

        For a letterboxed image, (a) the whole normalized frame maps to the whole
        original image, and (b) the normalized span covering exactly the padded
        content maps to that same whole image. Identity (b) is the one that
        catches a mis-inverted scale, because it is the only case where padding
        and resizing are simultaneously non-trivial.
        """
        prepared = prepare_image(make_png(400, 700))
        assert prepared.pad_x + prepared.pad_y > 0  # genuinely letterboxed
        assert prepared.content_width != prepared.original_width  # genuinely rescaled

        assert prepared.project_bbox((0.0, 0.0, 1.0, 1.0)) == (
            0, 0, prepared.original_width, prepared.original_height
        )

        x0 = prepared.pad_x / prepared.width
        y0 = prepared.pad_y / prepared.height
        x1 = (prepared.pad_x + prepared.content_width) / prepared.width
        y1 = (prepared.pad_y + prepared.content_height) / prepared.height
        assert prepared.project_bbox((x0, y0, x1, y1)) == (
            0, 0, prepared.original_width, prepared.original_height
        )

    def test_projection_of_content_is_monotonic(self) -> None:
        """A bbox covering half the content must land at half the original."""
        prepared = prepare_image(make_png(400, 700))
        x0 = prepared.pad_x / prepared.width
        y0 = prepared.pad_y / prepared.height
        x1 = x0 + (prepared.content_width / 2) / prepared.width
        y1 = y0 + (prepared.content_height / 2) / prepared.height

        x, y, width, height = prepared.project_bbox((x0, y0, x1, y1))

        assert (x, y) == (0, 0)
        assert width == prepared.original_width // 2
        assert height == prepared.original_height // 2
        assert 0 < width < prepared.original_width
        assert 0 < height < prepared.original_height


class TestClientSeam:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeClient("{}"), ModelBClientProtocol)

    def test_empty_body_is_validation_error(self) -> None:
        with pytest.raises(ModelBValidationError):
            ProviderResponse(text="   ").payload()

    def test_non_object_json_rejected(self) -> None:
        with pytest.raises(ModelBValidationError):
            ProviderResponse(text="[1, 2, 3]").payload()

    def test_truncation_detected_from_finish_reason(self) -> None:
        assert ProviderResponse(text="{}", finish_reason="MAX_TOKENS").truncated
        assert not ProviderResponse(text="{}", finish_reason="STOP").truncated

    def test_missing_key_raises_at_construction(self) -> None:
        with pytest.raises(ModelBMisconfigured):
            OpenRouterClient(api_key="", model="m", timeout_s=5.0)

    def test_non_positive_timeout_rejected(self) -> None:
        with pytest.raises(ModelBMisconfigured):
            OpenRouterClient(api_key="k", model="m", timeout_s=0)

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ModelBMisconfigured):
            OpenRouterClient(api_key="k", model="", timeout_s=5.0)

    def test_zero_attempts_rejected(self) -> None:
        """Bounded means bounded; 0 attempts would silently disable Model B."""
        with pytest.raises(ModelBMisconfigured):
            OpenRouterClient(api_key="k", model="m", timeout_s=5.0, max_attempts=0)

    def test_negative_backoff_rejected(self) -> None:
        with pytest.raises(ModelBMisconfigured):
            OpenRouterClient(
                api_key="k", model="m", timeout_s=5.0, retry_base_delay_s=-1.0
            )

    def test_no_vendor_sdk_is_imported_at_all(self) -> None:
        """Model B must not need a vendor package installed.

        The whole point of the stdlib transport is that there is no SDK whose
        absence can be mistaken for a broken install, and no SDK that can change
        Model A's behaviour underneath it. Checked against the parsed import
        list rather than the source text, so prose about OpenAI compatibility
        does not fail the assertion.
        """
        import ast

        from app.model_b import client as client_module

        tree = ast.parse(Path(client_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        assert imported <= {
            "base64",
            "json",
            "logging",
            "socket",
            "time",
            "urllib",
            "dataclasses",
            "typing",
            "__future__",
            "errors",
            "json_schema",
            "prompts",
        }, imported - {"base64"}

    def test_error_mapping_prefers_configuration_over_retry(self) -> None:
        from app.model_b.client import _map_provider_error

        assert isinstance(_map_provider_error(Exception("401 unauthorized")), ModelBMisconfigured)
        assert isinstance(_map_provider_error(Exception("429 rate limit")), ModelBApiError)

    def test_retryable_flag_set_for_transient_failures(self) -> None:
        from app.model_b.client import _map_provider_error

        assert _map_provider_error(Exception("503 overloaded")).retryable is True
        assert _map_provider_error(Exception("weird failure")).retryable is True

    def test_timeout_classified(self) -> None:
        from app.model_b.client import _map_provider_error
        from app.model_b.errors import ModelBTimeout

        assert isinstance(_map_provider_error(TimeoutError("slow")), ModelBTimeout)

    def test_unknown_model_is_misconfiguration_not_retryable(self) -> None:
        """Section 10: a 404 must never become "try again".

        Retrying a bad model name only delays the message that would have
        helped, and on a free tier it costs the user's quota.
        """
        from app.model_b.client import _map_provider_error

        mapped = _map_provider_error(Exception("404 not_found"), "some/model:free")
        assert isinstance(mapped, ModelBMisconfigured)
        assert "some/model:free" in mapped.detail

    # --- HTTP status classification ---------------------------------------

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures_are_misconfiguration(self, status: int) -> None:
        from app.model_b.client import _error_from_status

        mapped = _error_from_status(status, b"{}")
        assert isinstance(mapped, ModelBMisconfigured)
        assert mapped.http_status == status
        # Not a ModelBApiError, so there is no retry path to fall into.
        assert not isinstance(mapped, ModelBApiError)

    def test_404_names_the_requested_model(self) -> None:
        from app.model_b.client import _error_from_status

        mapped = _error_from_status(404, b"{}", "qwen/qwen3.8-27b:free")
        assert isinstance(mapped, ModelBMisconfigured)
        assert "qwen/qwen3.8-27b:free" in mapped.detail

    def test_408_is_a_timeout(self) -> None:
        from app.model_b.client import _error_from_status
        from app.model_b.errors import ModelBTimeout

        assert isinstance(_error_from_status(408, b""), ModelBTimeout)

    def test_429_is_rate_limited_and_retryable(self) -> None:
        from app.model_b.client import _error_from_status
        from app.model_b.errors import ModelBRateLimited

        mapped = _error_from_status(429, b"", retry_after="12")
        assert isinstance(mapped, ModelBRateLimited)
        assert mapped.retryable is True
        assert mapped.retry_after_s == 12.0
        assert mapped.status_code == 429
        assert mapped.reason == "model_b_rate_limited"

    def test_429_without_a_retry_after_is_still_rate_limited(self) -> None:
        from app.model_b.client import _error_from_status
        from app.model_b.errors import ModelBRateLimited

        mapped = _error_from_status(429, b"")
        assert isinstance(mapped, ModelBRateLimited)
        assert mapped.retry_after_s is None
        assert "shortly" in mapped.detail

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_is_retryable(self, status: int) -> None:
        from app.model_b.client import _error_from_status

        mapped = _error_from_status(status, b"")
        assert isinstance(mapped, ModelBApiError)
        assert mapped.retryable is True
        assert mapped.http_status == status

    def test_400_is_not_retryable_and_quotes_the_provider_message(self) -> None:
        from app.model_b.client import _error_from_status

        body = b'{"error": {"message": "No endpoints found for vision"}}'
        mapped = _error_from_status(400, body)
        assert isinstance(mapped, ModelBMisconfigured)
        assert mapped.retryable if hasattr(mapped, "retryable") else True
        assert "No endpoints found for vision" in mapped.detail

    def test_unparseable_retry_after_is_ignored(self) -> None:
        """HTTP also permits an HTTP-date. Not worth a date parser here."""
        from app.model_b.client import _retry_after_seconds

        assert _retry_after_seconds("12") == 12.0
        assert _retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") is None
        assert _retry_after_seconds(None) is None
        assert _retry_after_seconds("") is None
        assert _retry_after_seconds("-5") is None

    def test_provider_detail_is_truncated(self) -> None:
        from app.model_b.client import _provider_detail

        assert _provider_detail(b'{"error":{"message":"' + b"x" * 5000 + b'"}}')
        assert len(_provider_detail(b'{"error":{"message":"' + b"x" * 5000 + b'"}}')) <= 240
        assert _provider_detail(None) == ""
        assert _provider_detail(b"") == ""

    # --- envelope extraction ------------------------------------------------

    def test_text_read_from_a_plain_string_content(self) -> None:
        from app.model_b.client import _extract_text

        assert _extract_text({"choices": [{"message": {"content": "{}"}}]}) == "{}"

    def test_text_read_from_a_typed_parts_list(self) -> None:
        """Several upstreams return parts even when only text is present."""
        from app.model_b.client import _extract_text

        envelope = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "{\"a\": "}, {"type": "text", "text": "1}"}]}}
            ]
        }
        assert _extract_text(envelope) == '{"a": 1}'

    def test_text_returns_empty_when_there_is_nothing_to_read(self) -> None:
        from app.model_b.client import _extract_text

        assert _extract_text({}) == ""
        assert _extract_text({"choices": []}) == ""
        assert _extract_text({"choices": [{"message": {}}]}) == ""
        assert _extract_text({"choices": [{"message": {"content": None}}]}) == ""
        # A tool-call-only response has no text and must not crash extraction.
        assert _extract_text({"choices": [{"message": {"tool_calls": []}}]}) == ""

    def test_finish_reason_extracted(self) -> None:
        from app.model_b.client import _extract_finish_reason

        assert _extract_finish_reason({"choices": [{"finish_reason": "stop"}]}) == "stop"
        assert _extract_finish_reason({"choices": [{"finish_reason": "length"}]}) == "length"
        assert _extract_finish_reason({"choices": [{}]}) == ""
        assert _extract_finish_reason({}) == ""

    def test_length_finish_reason_counts_as_truncated(self) -> None:
        """OpenRouter reports OpenAI's `length`; Gemini called it MAX_TOKENS."""
        assert ProviderResponse(text="{}", finish_reason="length").truncated
        assert ProviderResponse(text="{}", finish_reason="stop").truncated is False

    def test_usage_includes_reasoning_tokens(self) -> None:
        """Free-tier vision models are often reasoning models; their cost is
        otherwise invisible in the logs."""
        from app.model_b.client import _extract_usage

        usage = _extract_usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cost": 0.000012,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                }
            }
        )
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 7,
            "cost": 0.000012,
        }

    def test_usage_absent_is_not_an_error(self) -> None:
        from app.model_b.client import _extract_usage

        assert _extract_usage({}) == {}
        assert _extract_usage({"usage": None}) == {}
        assert _extract_usage({"usage": {"prompt_tokens": "many"}}) == {}

    def test_resolved_model_falls_back_to_the_requested_one(self) -> None:
        """Provenance must never be empty just because a gateway was terse."""
        assert (
            ProviderResponse(text="{}", requested_model="openrouter/free").resolved_model
            == "openrouter/free"
        )
        assert (
            ProviderResponse(
                text="{}", requested_model="openrouter/free", model="qwen/qwen3.8-27b"
            ).resolved_model
            == "qwen/qwen3.8-27b"
        )


class _RecordingTransport:
    """Stand-in for `urllib.request.urlopen` that records and replays.

    No socket, no key, no cost. This is the seam that lets the entire request
    construction path be asserted rather than mocked away.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, request, timeout=None):
        body = self._next_body()
        self.calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data),
                "timeout": timeout,
                "method": request.get_method(),
            }
        )
        return _FakeHTTPResponse(body)

    def _next_body(self) -> bytes:
        if not self._responses:
            raise AssertionError("transport was called more times than expected")
        value = self._responses[0]
        if isinstance(value, Exception):
            raise value
        # A list means "the same answer every attempt", which is how a
        # deterministic retry test is written.
        return value


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _envelope(
    content: str = "{}",
    *,
    model: str = "qwen/qwen3.8-27b",
    provider: str = "novita",
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> bytes:
    """A well-formed OpenAI-compatible response envelope."""
    return json.dumps(
        {
            "id": "gen-test",
            "model": model,
            "provider": provider,
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
    ).encode()


def _client(transport, **kwargs) -> OpenRouterClient:
    """A client wired to a fake transport and a no-op sleep.

    `sleep` is injected so a retry test costs microseconds instead of seconds
    and never actually waits.
    """
    params = {
        "api_key": "sk-or-v1-TESTKEYNOTREAL",
        "model": "openrouter/free",
        "timeout_s": 7.0,
        "transport": transport,
        "sleep": lambda _seconds: None,
    }
    params.update(kwargs)
    return OpenRouterClient(**params)


def _png_bytes(width: int = 64, height: int = 64) -> bytes:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


class TestOpenRouterRequest:
    """Section 5 and 16: request construction, headers, image encoding."""

    def test_posts_to_the_chat_completions_endpoint(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert transport.calls[0]["url"] == (
            "https://openrouter.ai/api/v1/chat/completions"
        )
        assert transport.calls[0]["method"] == "POST"

    def test_custom_base_url_is_honoured_without_a_double_slash(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(transport, base_url="https://proxy.internal/v1/")
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert transport.calls[0]["url"] == "https://proxy.internal/v1/chat/completions"

    def test_bearer_authorization_header(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        headers = {k.lower(): v for k, v in transport.calls[0]["headers"].items()}
        assert headers["authorization"] == "Bearer sk-or-v1-TESTKEYNOTREAL"
        assert headers["content-type"] == "application/json"

    def test_attribution_headers_only_when_configured(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(transport, site_url="", app_title="")
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        headers = {k.lower() for k in transport.calls[0]["headers"]}
        assert "http-referer" not in headers
        assert "x-title" not in headers

        transport = _RecordingTransport(_envelope())
        client = _client(
            transport, site_url="https://tactilegeo.example", app_title="TactileGeo"
        )
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        headers = {k.lower(): v for k, v in transport.calls[0]["headers"].items()}
        assert headers["http-referer"] == "https://tactilegeo.example"
        assert headers["x-title"] == "TactileGeo"

    def test_the_key_never_appears_in_the_request_body(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert "sk-or-v1-TESTKEYNOTREAL" not in json.dumps(transport.calls[0]["body"])

    def test_model_comes_from_configuration(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(transport, model="qwen/qwen3.8-27b:free")
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert transport.calls[0]["body"]["model"] == "qwen/qwen3.8-27b:free"

    def test_structured_output_is_requested(self) -> None:
        """Section 6: the contract travels in the request, not the prompt."""
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        response_format = transport.calls[0]["body"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "model_b_document"
        assert (
            response_format["json_schema"]["schema"]["properties"]["schema_version"]["enum"]
            == ["model_b.diag.v1"]
        )

    def test_determinism_settings(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        body = transport.calls[0]["body"]
        assert body["temperature"] == 0
        assert body["stream"] is False
        assert body["max_tokens"] > 0

    def test_image_is_sent_exactly_once_as_a_png_data_url(self) -> None:
        """Section 5: no duplicate copies of the image."""
        import base64

        png = _png_bytes()
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=png, prepared_width=64, prepared_height=64)
        parts = transport.calls[0]["body"]["messages"][0]["content"]
        images = [part for part in parts if part["type"] == "image_url"]
        assert len(images) == 1
        prefix = "data:image/png;base64,"
        url = images[0]["image_url"]["url"]
        assert url.startswith(prefix)
        assert base64.b64decode(url[len(prefix):]) == png

    def test_prompt_travels_beside_the_image_as_text(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        parts = transport.calls[0]["body"]["messages"][0]["content"]
        texts = [part for part in parts if part["type"] == "text"]
        assert len(texts) == 1
        assert "geometry-diagram analyst" in texts[0]["text"]
        assert texts[0]["text"] == build_prompt()

    def test_single_user_message(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        messages = transport.calls[0]["body"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_timeout_is_passed_through(self) -> None:
        transport = _RecordingTransport(_envelope())
        _client(transport, timeout_s=11.5).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert transport.calls[0]["timeout"] == 11.5

    def test_empty_image_fails_before_spending_a_round_trip(self) -> None:
        transport = _RecordingTransport(_envelope())
        with pytest.raises(ModelBValidationError):
            _client(transport).analyze(image_png=b"", prepared_width=64, prepared_height=64)
        assert transport.calls == []


class TestResolvedModelIdentity:
    """Sections 4 and 18: report the model that actually ran."""

    def test_resolved_model_comes_from_the_response_not_configuration(self) -> None:
        transport = _RecordingTransport(
            _envelope(model="google/gemma-4-31b-it", provider="google-vertex")
        )
        response = _client(transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert response.requested_model == "openrouter/free"
        assert response.resolved_model == "google/gemma-4-31b-it"
        assert response.upstream_provider == "google-vertex"

    def test_result_records_both_models_end_to_end(self) -> None:
        from app.model_b.service import analyze_image

        transport = _RecordingTransport(
            _envelope(
                json.dumps(valid_document()),
                model="qwen/qwen3.8-27b",
                provider="novita",
            )
        )
        client = _client(transport)
        result = analyze_image(_png_bytes(120, 90), ON, client=client)
        assert result.provider == "openrouter"
        assert result.requested_model == "openrouter/free"
        assert result.resolved_model == "qwen/qwen3.8-27b"
        assert result.upstream_provider == "novita"
        assert result.usage["prompt_tokens"] == 5

    def test_missing_model_in_the_response_falls_back_to_configuration(self) -> None:
        body = json.dumps(
            {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        ).encode()
        transport = _RecordingTransport(body)
        response = _client(transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert response.model == ""
        assert response.resolved_model == "openrouter/free"


class TestRetryPolicy:
    """Section 11: bounded retry, exponential backoff, no retry on config errors."""

    def test_429_is_retried_then_succeeds(self) -> None:
        sleeps: list[float] = []
        transport = _RecordingTransport(_envelope())
        client = OpenRouterClient(
            api_key="k",
            model="openrouter/free",
            timeout_s=5.0,
            transport=transport,
            sleep=sleeps.append,
        )
        flaky = _FlakyTransport(transport, status=429, times=1)
        client._transport = flaky
        response = client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert response.text == "{}"
        assert flaky.attempts == 2
        assert len(transport.calls) == 1  # only the successful attempt was recorded
        assert len(sleeps) == 1

    def test_retry_after_is_honoured_over_the_backoff_curve(self) -> None:
        sleeps: list[float] = []
        transport = _RecordingTransport(_envelope())
        client = OpenRouterClient(
            api_key="k",
            model="openrouter/free",
            timeout_s=5.0,
            transport=transport,
            sleep=sleeps.append,
            retry_base_delay_s=0.8,
            retry_max_delay_s=30.0,
        )
        client._transport = _FlakyTransport(transport, status=429, times=1, retry_after="7")
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert sleeps == [7.0]

    def test_retry_after_is_capped(self) -> None:
        sleeps: list[float] = []
        transport = _RecordingTransport(_envelope())
        client = OpenRouterClient(
            api_key="k",
            model="openrouter/free",
            timeout_s=5.0,
            transport=transport,
            sleep=sleeps.append,
            retry_max_delay_s=5.0,
        )
        client._transport = _FlakyTransport(transport, status=429, times=1, retry_after="900")
        client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert sleeps == [5.0]

    def test_backoff_grows_exponentially_and_is_capped(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(
            transport, max_attempts=5, retry_base_delay_s=0.5, retry_max_delay_s=2.0
        )
        assert [client._backoff_delay(n, None) for n in range(1, 6)] == [0.5, 1.0, 2.0, 2.0, 2.0]

    def test_5xx_is_retried(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(transport)
        client._transport = _FlakyTransport(transport, status=503, times=1)
        assert client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_attempts_are_bounded(self) -> None:
        transport = _RecordingTransport(_envelope())
        flaky = _FlakyTransport(transport, status=503, times=99)
        client = _client(transport, max_attempts=3)
        client._transport = flaky
        with pytest.raises(ModelBApiError):
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert flaky.attempts == 3

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 408])
    def test_configuration_errors_are_never_retried(self, status: int) -> None:
        """Section 11: do not retry 401/403 indefinitely."""
        transport = _RecordingTransport(_envelope())
        flaky = _FlakyTransport(transport, status=status, times=99)
        client = _client(transport)
        client._transport = flaky
        with pytest.raises(ModelBError):
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert flaky.attempts == 1

    def test_single_attempt_configuration_disables_retry_entirely(self) -> None:
        transport = _RecordingTransport(_envelope())
        flaky = _FlakyTransport(transport, status=503, times=99)
        client = _client(transport, max_attempts=1)
        client._transport = flaky
        with pytest.raises(ModelBApiError):
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert flaky.attempts == 1

    def test_rate_limit_exhaustion_surfaces_as_rate_limited(self) -> None:
        transport = _RecordingTransport(_envelope())
        client = _client(transport, max_attempts=2)
        client._transport = _FlakyTransport(transport, status=429, times=99)
        with pytest.raises(ModelBRateLimited):
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)


class _FlakyTransport:
    """Raises `status` for the first `times` attempts, then delegates.

    Built as a wrapper rather than a queue of pre-baked responses so a retry
    test states *how many times* it wants to fail and lets everything after
    succeed, instead of hard-coding a sequence that breaks the moment
    `max_attempts` changes.

    `attempts` counts every call, including the ones that raised. Counting on the
    inner transport instead would under-report, because a failing attempt never
    reaches the recorder.
    """

    def __init__(self, inner, *, status: int, times: int, retry_after: str | None = None):
        self._inner = inner
        self._status = status
        self._remaining = times
        self._retry_after = retry_after
        self.attempts = 0

    def __call__(self, request, timeout=None):
        self.attempts += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise urllib.error.HTTPError(
                request.full_url,
                self._status,
                "error",
                {"Retry-After": self._retry_after} if self._retry_after else {},
                None,
            )
        return self._inner(request, timeout=timeout)


class TestMalformedResponses:
    """Section 10: empty, malformed, and non-object bodies are rejected."""

    def test_empty_body_is_rejected(self) -> None:
        transport = _RecordingTransport(b"")
        with pytest.raises(ModelBValidationError):
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_non_json_envelope_is_rejected(self) -> None:
        transport = _RecordingTransport(b"<html>502 Bad Gateway</html>")
        with pytest.raises(ModelBMalformedResponse):
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_json_array_envelope_is_rejected(self) -> None:
        transport = _RecordingTransport(b"[1, 2, 3]")
        with pytest.raises(ModelBMalformedResponse):
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_missing_choices_without_an_error_is_rejected(self) -> None:
        """A 2xx with nothing usable must not read as "no geometry found".

        Caught at the transport boundary so the message names the real problem
        instead of blaming the worksheet.
        """
        transport = _RecordingTransport(b'{"id": "gen-1", "model": "m"}')
        with pytest.raises(ModelBMalformedResponse) as excinfo:
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert "no completion" in excinfo.value.detail

    def test_empty_choices_array_is_rejected(self) -> None:
        transport = _RecordingTransport(b'{"choices": []}')
        with pytest.raises(ModelBMalformedResponse):
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_provider_error_object_on_a_200_is_not_parsed_as_a_document(self) -> None:
        """A 2xx carrying an error envelope must not read as "found nothing".

        A gateway with no route to the requested model can answer 200 with an
        error body. Passing that to the parser would report an empty, valid,
        confident Model B result — the most misleading outcome possible.
        """
        transport = _RecordingTransport(
            b'{"error": {"message": "No endpoints found for openrouter/free"}}'
        )
        with pytest.raises(ModelBApiError) as excinfo:
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert "No endpoints found" in excinfo.value.detail

    def test_error_key_alongside_real_choices_is_not_treated_as_a_failure(self) -> None:
        transport = _RecordingTransport(
            b'{"error": null, "choices": [{"finish_reason": "stop",'
            b' "message": {"content": "{}"}}]}'
        )
        assert _client(transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        ).text == "{}"

    def test_socket_timeout_becomes_a_typed_timeout(self) -> None:
        import socket

        transport = _RecordingTransport(_envelope())
        client = _client(transport, max_attempts=1)

        def boom(request, timeout=None):
            raise socket.timeout("timed out")

        client._transport = boom
        with pytest.raises(ModelBTimeout):
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)

    def test_url_error_becomes_a_retryable_api_error(self) -> None:
        import urllib.error

        transport = _RecordingTransport(_envelope())
        client = _client(transport, max_attempts=1)

        def boom(request, timeout=None):
            raise urllib.error.URLError("connection reset by peer")

        client._transport = boom
        with pytest.raises(ModelBApiError) as excinfo:
            client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert excinfo.value.retryable is True


class TestObservabilitySafety:
    """Section 12: never log the key, headers, or image data."""

    def test_logs_never_contain_the_key_or_image_bytes(self, caplog) -> None:
        import logging

        png = _png_bytes()
        transport = _RecordingTransport(_envelope())
        with caplog.at_level(logging.DEBUG):
            _client(transport).analyze(
                image_png=png, prepared_width=64, prepared_height=64
            )
        blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "sk-or-v1-TESTKEYNOTREAL" not in blob
        assert "Bearer" not in blob
        assert blob

    def test_success_log_records_the_resolved_model_and_counts(self, caplog) -> None:
        import logging

        transport = _RecordingTransport(_envelope(model="qwen/qwen3.8-27b"))
        with caplog.at_level(logging.INFO):
            _client(transport).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "resolved_model=qwen/qwen3.8-27b" in blob
        assert "prompt_tokens=5" in blob

    def test_failure_log_records_the_category_not_the_prose(self, caplog) -> None:
        import logging

        transport = _RecordingTransport(_envelope())
        client = _client(transport, max_attempts=1)
        client._transport = _FlakyTransport(transport, status=401, times=1)
        with caplog.at_level(logging.INFO):
            with pytest.raises(ModelBError):
                client.analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "ModelBMisconfigured" in blob
        assert "status=401" in blob


class TestParser:
    def test_plain_object(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self) -> None:
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_surrounded_by_prose(self) -> None:
        assert extract_json_object('Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_brace_inside_string_does_not_break_matching(self) -> None:
        assert extract_json_object('{"evidence": "looks like {90 deg} here"}')["evidence"] == (
            "looks like {90 deg} here"
        )

    def test_escaped_quote_inside_string(self) -> None:
        assert extract_json_object('{"e": "a \\" b"}') == {"e": 'a " b'}

    def test_nested_objects(self) -> None:
        assert extract_json_object('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_empty_body_rejected(self) -> None:
        with pytest.raises(ModelBValidationError):
            extract_json_object("")

    def test_prose_without_json_rejected(self) -> None:
        with pytest.raises(ModelBValidationError):
            extract_json_object("I cannot analyze this image.")

    def test_unclosed_object_reported_as_truncation(self) -> None:
        with pytest.raises(ModelBMalformedResponse):
            extract_json_object('{"entities": [{"id": "e1"')

    def test_json_array_only_is_rejected_as_wrong_type(self) -> None:
        with pytest.raises(ModelBValidationError):
            extract_json_object("[1,2,3]")


class TestValidator:
    def test_valid_document_passes(self) -> None:
        document, warnings = validate_document(
            valid_document(), prepared_width=1000, prepared_height=800
        )
        assert document.schema_version.startswith("model_b")
        assert warnings == []

    def test_dangling_relationship_warns_but_does_not_fail(self) -> None:
        """One bookkeeping error must not discard a correct tangent finding."""
        doc = valid_document()
        doc["relationships"][0]["to_id"] = "e39"
        document, warnings = validate_document(
            doc, prepared_width=1000, prepared_height=800
        )
        assert any("no matching entity" in w for w in warnings)
        assert len(document.diagram_relations) == 2

    def test_self_referential_relationship_warns(self) -> None:
        doc = valid_document()
        doc["relationships"][0]["to_id"] = "e1"
        _, warnings = validate_document(doc, prepared_width=1000, prepared_height=800)
        assert any("references itself" in w for w in warnings)

    def test_size_mismatch_warns(self) -> None:
        _, warnings = validate_document(
            valid_document(), prepared_width=1600, prepared_height=1600
        )
        assert any("does not match" in w for w in warnings)

    def test_duplicate_entity_id_warns(self) -> None:
        doc = valid_document()
        doc["entities"].append(dict(doc["entities"][0]))
        _, warnings = validate_document(doc, prepared_width=1000, prepared_height=800)
        assert any("duplicate entity id" in w for w in warnings)

    def test_occluded_but_certain_warns(self) -> None:
        doc = valid_document()
        doc["entities"][0]["occluded"] = True
        doc["entities"][0]["confidence"] = "certain"
        _, warnings = validate_document(doc, prepared_width=1000, prepared_height=800)
        assert any("occluded but reported as certain" in w for w in warnings)

    def test_non_square_circle_warns(self) -> None:
        doc = valid_document()
        doc["entities"][1]["bbox"] = [0.0, 0.0, 1.0, 0.2]
        _, warnings = validate_document(doc, prepared_width=1000, prepared_height=800)
        assert any("non-square" in w for w in warnings)

    def test_schema_violation_raises_with_field_issues(self) -> None:
        doc = valid_document()
        doc["entities"][0]["confidence"] = "nope"
        with pytest.raises(ModelBValidationError) as excinfo:
            validate_document(doc, prepared_width=1000, prepared_height=800)
        assert excinfo.value.issues
        assert "entities.0.confidence" in excinfo.value.issues[0]


class TestNormalizer:
    def _normalize(self, doc=None, width=1000, height=800):
        document, warnings = validate_document(
            doc or valid_document(), prepared_width=width, prepared_height=height
        )
        prepared = prepare_image(make_png(width, height))
        return normalize(document, prepared, validation_warnings=warnings)

    def test_projects_into_original_frame(self) -> None:
        result = self._normalize()
        # Prepared image is square 800x800 padded from 1000x800, so the triangle
        # at norm x 0.10 must land left of centre in the original 1000px width.
        assert result.entities[0].region.x < result.image_width / 2

    def test_ids_are_namespaced(self) -> None:
        result = self._normalize()
        assert result.entities[0].id == "b_e1"
        assert result.relationships[0].from_id == "b_e1"

    def test_maps_mappable_entity_kinds(self) -> None:
        result = self._normalize()
        by_kind = {entity.kind: entity for entity in result.entities}
        assert by_kind["triangle"].geometry_type is GeometryType.TRIANGLE
        assert by_kind["circle"].geometry_type is GeometryType.CIRCLE
        assert by_kind["point"].geometry_type is GeometryType.POINT

    def test_mapped_flags_track_the_table(self) -> None:
        result = self._normalize()
        for entity in result.entities:
            assert entity.mapped == (entity.geometry_type is not None)

    def test_unmapped_kind_is_flagged_not_coerced(self) -> None:
        doc = valid_document()
        doc["entities"][0]["kind"] = "quadrilateral"
        result = self._normalize(doc)
        triangle = next(e for e in result.entities if e.kind == "quadrilateral")
        assert triangle.geometry_type is None
        assert triangle.mapped is False
        assert result.unmapped_entity_count == 1

    def test_dimension_annotation_is_unmapped(self) -> None:
        doc = valid_document()
        doc["entities"][0]["kind"] = "dimension_annotation"
        result = self._normalize(doc)
        assert next(e for e in result.entities if e.kind == "dimension_annotation").mapped is False

    def test_point_on_resolves_against_target_geometry(self) -> None:
        """`point_on` is polymorphic: the target decides the Model A relation."""
        doc = valid_document()
        doc["relationships"][0]["kind"] = "point_on"
        doc["relationships"][0]["from_id"] = "e3"
        doc["relationships"][0]["to_id"] = "e2"  # e2 is the circle
        result = self._normalize(doc)
        assert result.relationships[0].relationship_type is RelationshipType.POINT_ON_CIRCLE

    def test_point_on_line_when_target_is_a_segment(self) -> None:
        doc = valid_document()
        doc["entities"].append(
            {
                "id": "e4",
                "kind": "line_segment",
                "detection_kind": "observed",
                "bbox": [0.1, 0.5, 0.7, 0.55],
                "confidence": "certain",
                "evidence": "single straight stroke",
                "occluded": False,
                "label": None,
            }
        )
        doc["relationships"][0]["kind"] = "point_on"
        doc["relationships"][0]["from_id"] = "e3"
        doc["relationships"][0]["to_id"] = "e4"
        result = self._normalize(doc)
        assert result.relationships[0].relationship_type is RelationshipType.POINT_ON_LINE

    def test_point_on_ambiguous_target_is_unmapped(self) -> None:
        """A point on a triangle boundary has no Model A equivalent."""
        doc = valid_document()
        doc["relationships"][0]["kind"] = "point_on"
        doc["relationships"][0]["from_id"] = "e3"
        doc["relationships"][0]["to_id"] = "e1"  # e1 is the triangle
        result = self._normalize(doc)
        assert result.relationships[0].relationship_type is None
        assert result.relationships[0].mapped is False

    def test_symmetric_with_is_unmapped(self) -> None:
        doc = valid_document()
        doc["relationships"][0]["kind"] = "symmetric_with"
        result = self._normalize(doc)
        assert result.relationships[0].relationship_type is None
        assert result.relationships[0].mapped is False

    def test_occlusion_forces_review(self) -> None:
        result = self._normalize()
        apex = next(e for e in result.entities if e.id == "b_e3")
        assert apex.occluded is True
        assert apex.needs_review is True

    def test_null_text_item_is_always_review(self) -> None:
        result = self._normalize()
        unreadable = next(t for t in result.text_items if t.text is None)
        assert unreadable.needs_review is True

    def test_legible_text_item_not_review(self) -> None:
        result = self._normalize()
        legible = next(t for t in result.text_items if t.text == "BC")
        assert legible.needs_review is False

    def test_suggested_element_carries_no_geometry(self) -> None:
        """The advisory guarantee, asserted at the serialization boundary."""
        result = self._normalize()
        payload = result.entities[0].to_element_dict()
        assert payload["geometry"] == {}
        assert payload["source"] == "model_b"
        assert "x" not in payload["geometry"]

    def test_relationship_dict_has_no_coordinates(self) -> None:
        result = self._normalize()
        payload = result.relationships[0].to_relationship_dict()
        assert "geometry" not in payload
        assert payload["element_ids"] == ["b_e1", "b_e2"]

    def test_counts_and_dimensions(self) -> None:
        result = self._normalize(width=1000, height=800)
        assert (result.image_width, result.image_height) == (1000, 800)
        assert result.mapped_entity_count == 3
        assert len(result.diagram_relations) == 2
        assert len(result.uncertainties) == 1
        assert result.truncated is False

    def test_minimal_document_normalizes_to_empty(self) -> None:
        result = self._normalize(minimal_document(), width=4, height=4)
        assert result.entities == []
        assert result.diagram_kind == "not_a_geometry_diagram"
        assert result.mapped_entity_count == 0

    def test_truncation_flag_propagates(self) -> None:
        document, _ = validate_document(
            valid_document(), prepared_width=1000, prepared_height=800
        )
        result = normalize(
            document,
            prepare_image(make_png(1000, 800)),
            truncated=True,
            finish_reason="MAX_TOKENS",
        )
        assert result.truncated is True
        assert result.finish_reason == "MAX_TOKENS"


class TestMappingTables:
    def test_every_schema_vocabulary_is_mapped_or_explicitly_unmapped(self) -> None:
        assert validate_mapping_tables() == []

    def test_unmapped_entity_kinds_are_deliberate(self) -> None:
        unmapped = {k for k, v in ENTITY_KIND_TO_GEOMETRY.items() if v is None}
        assert unmapped == {"quadrilateral", "dimension_annotation", "unknown"}

    def test_unmapped_relationship_kinds_are_deliberate(self) -> None:
        unmapped = {k for k, v in RELATIONSHIP_KIND_TO_RELATIONSHIP.items() if v is None}
        assert unmapped == {"point_on", "contains", "adjacent_to", "symmetric_with", "dimension_of"}

    def test_confidence_bands_are_ordered(self) -> None:
        from app.models.model_b_result import CONFIDENCE_BAND_TO_FLOAT

        values = [CONFIDENCE_BAND_TO_FLOAT[b] for b in ("certain", "likely", "uncertain", "unreadable")]
        assert values == sorted(values, reverse=True)


class TestBoundingRegion:
    def test_iou_identical_is_one(self) -> None:
        a = BoundingRegion((0, 0, 1, 1), 0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)

    def test_iou_disjoint_is_zero(self) -> None:
        a = BoundingRegion((0, 0, 1, 1), 0, 0, 10, 10)
        b = BoundingRegion((0, 0, 1, 1), 50, 50, 10, 10)
        assert a.iou(b) == 0.0

    def test_iou_half_overlap(self) -> None:
        a = BoundingRegion((0, 0, 1, 1), 0, 0, 10, 10)
        b = BoundingRegion((0, 0, 1, 1), 5, 0, 10, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_contains_point(self) -> None:
        region = BoundingRegion((0, 0, 1, 1), 10, 20, 30, 40)
        assert region.contains_point(10, 20)
        assert region.contains_point(40, 60)
        assert not region.contains_point(9, 20)

    def test_model_a_bbox_convention(self) -> None:
        region = BoundingRegion((0, 0, 1, 1), 3, 4, 5, 6)
        assert region.as_model_a_bbox() == (3, 4, 5, 6)


class TestService:
    def test_disabled_raises(self) -> None:
        with pytest.raises(ModelBDisabled):
            analyze_image(make_png(100, 100), ModelBSettings(enabled=False))

    def test_enabled_without_key_raises_misconfigured(self) -> None:
        with pytest.raises(ModelBMisconfigured):
            analyze_image(make_png(100, 100), ModelBSettings(enabled=True, api_key=None))

    def test_missing_key_message_names_the_variable(self) -> None:
        with pytest.raises(ModelBMisconfigured) as excinfo:
            analyze_image(make_png(100, 100), ModelBSettings(enabled=True))
        assert "OPENROUTER_API_KEY" in str(excinfo.value)

    def test_disabled_message_names_the_flag(self) -> None:
        with pytest.raises(ModelBDisabled) as excinfo:
            analyze_image(make_png(100, 100), ModelBSettings(enabled=False))
        assert "MODEL_B_ENABLED" in str(excinfo.value)

    def test_is_available(self) -> None:
        assert is_available(ON)
        assert not is_available(ModelBSettings(enabled=True, api_key=None))
        assert not is_available(ModelBSettings(enabled=False, api_key="k"))

    def test_happy_path(self) -> None:
        doc = valid_document()
        doc["image"] = {"width_px": 1568, "height_px": 1568}
        client = FakeClient(json.dumps(doc))
        result = analyze_image(make_png(1000, 800), ON, client=client)
        assert len(result.entities) == 3
        assert result.schema_version.startswith("model_b")

    def test_passes_png_and_prepared_dims_to_client(self) -> None:
        doc = minimal_document()
        client = FakeClient(json.dumps(doc))
        analyze_image(make_png(300, 500), ON, client=client)
        call = client.calls[0]
        assert call["size"] > 0
        assert call["width"] == call["height"]

    def test_tolerates_fenced_response(self) -> None:
        client = FakeClient(f"```json\n{json.dumps(minimal_document())}\n```")
        result = analyze_image(make_png(100, 100), ON, client=client)
        assert result.entities == []

    def test_truncation_surfaces_in_result(self) -> None:
        client = FakeClient(json.dumps(minimal_document()), finish_reason="MAX_TOKENS")
        result = analyze_image(make_png(100, 100), ON, client=client)
        assert result.truncated is True

    def test_provider_error_propagates_typed(self) -> None:
        client = FakeClient("{}")
        with pytest.raises(ModelBApiError):
            analyze_image(
                make_png(100, 100), ON, client=client.raises(ModelBApiError("down", retryable=True))
            )

    def test_undecodable_image_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            analyze_image(b"garbage", ON, client=FakeClient(json.dumps(minimal_document())))

    def test_default_model_is_flash_tier(self) -> None:
        """The design forbids silent Pro escalation in v1."""
        assert ModelBSettings().model == "openrouter/free"


class TestDiagnostics:
    """The diagnostic path is the only way to see a rejected response.

    Two properties matter and they pull in opposite directions: it must capture
    the model's actual words, and it must never capture a credential. Both are
    tested here because either one failing is silent.
    """

    def _client(self, monkeypatch, **kwargs):
        monkeypatch.setenv("MODEL_B_DIAGNOSTICS", "1")
        return _client(**kwargs)

    def test_disabled_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("MODEL_B_DIAGNOSTICS", raising=False)
        from app.model_b import diagnostics

        assert diagnostics.enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "debug"])
    def test_truthy_values_enable(self, monkeypatch, value: str) -> None:
        from app.model_b import diagnostics

        monkeypatch.setenv("MODEL_B_DIAGNOSTICS", value)
        assert diagnostics.enabled() is True

    def test_nothing_is_captured_when_disabled(self, monkeypatch) -> None:
        monkeypatch.delenv("MODEL_B_DIAGNOSTICS", raising=False)
        transport = _RecordingTransport(_envelope())
        response = _client(transport=transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        # Empty, not a redacted summary: with the flag off the whole feature is
        # absent, so there is nothing to leak and nothing to pay for.
        assert response.diagnostic == ""

    def test_raw_content_is_captured_when_enabled(self, monkeypatch) -> None:
        prose = "I am unable to produce that document."
        body = json.dumps(
            {
                "id": "gen-1",
                "model": "google/gemma-4-26b-a4b-it:free",
                "provider": "google-vertex",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": prose}}
                ],
            }
        ).encode()
        transport = _RecordingTransport(body)
        response = self._client(monkeypatch, transport=transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert prose in response.diagnostic
        assert "google/gemma-4-26b-a4b-it:free" in response.diagnostic
        assert "http status       : 200" in response.diagnostic

    def test_the_key_is_never_captured(self, monkeypatch) -> None:
        secret = "sk-or-v1-" + "A1b2C3d4E5f6" * 4
        transport = _RecordingTransport(_envelope())
        response = self._client(
            monkeypatch, transport=transport, api_key=secret
        ).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert secret not in response.diagnostic
        assert "sk-or-v1" not in response.diagnostic

    def test_authorization_value_is_never_captured(self, monkeypatch) -> None:
        transport = _RecordingTransport(_envelope())
        response = self._client(monkeypatch, transport=transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        # The header is reported by name and its presence is confirmed, because
        # "was a token attached" is a real question. The value is not shown.
        assert "authorization set : True" in response.diagnostic
        assert "Bearer" not in response.diagnostic

    def test_a_model_that_echoes_the_key_is_redacted(self, monkeypatch) -> None:
        """Defence in depth: output is untrusted and gets the same redaction."""
        secret = "sk-or-v1-" + "Z9y8X7w6V5u4" * 4
        body = json.dumps(
            {
                "id": "gen-1",
                "model": "m",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": f"Authorization: Bearer {secret}"},
                    }
                ],
            }
        ).encode()
        response = self._client(
            monkeypatch, transport=_RecordingTransport(body)
        ).analyze(image_png=_png_bytes(), prepared_width=64, prepared_height=64)
        assert secret not in response.diagnostic
        assert "Bearer <redacted>" in response.diagnostic

    def test_the_image_is_never_captured(self, monkeypatch) -> None:
        """The worksheet is uploaded content; it is reported as a size only."""
        transport = _RecordingTransport(_envelope())
        response = self._client(monkeypatch, transport=transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert "base64," not in response.diagnostic
        assert "image_parts=1" in response.diagnostic

    def test_response_format_configuration_is_reported(self, monkeypatch) -> None:
        """The diagnostic exists to answer "did the provider honour our
        structured-output request", so it has to show what we asked for."""
        transport = _RecordingTransport(_envelope())
        response = self._client(monkeypatch, transport=transport).analyze(
            image_png=_png_bytes(), prepared_width=64, prepared_height=64
        )
        assert "response_format   : type=json_schema" in response.diagnostic
        assert "strict=True" in response.diagnostic

    def test_a_parse_failure_carries_the_diagnostic_on_the_error(self, monkeypatch) -> None:
        """The transport is out of scope by the time the parser rejects, so the
        diagnostic has to travel on the exception or it is unrecoverable."""
        from app.model_b.service import ModelBSettings, analyze_image

        prose = "Here is a description of the diagram instead of JSON."
        body = json.dumps(
            {
                "id": "gen-1",
                "model": "m",
                "choices": [{"finish_reason": "stop", "message": {"content": prose}}],
            }
        ).encode()
        client = self._client(
            monkeypatch, transport=_RecordingTransport(body), api_key="test-key"
        )
        with pytest.raises(ModelBValidationError) as excinfo:
            analyze_image(
                _png_bytes(),
                ModelBSettings(enabled=True, api_key="test-key"),
                client=client,
            )
        assert prose in excinfo.value.diagnostic

    def test_validation_still_rejects_prose(self, monkeypatch) -> None:
        """The diagnostic must not become a back door. Prose is still an error."""
        from app.model_b.service import ModelBSettings, analyze_image

        body = json.dumps(
            {
                "id": "gen-1",
                "model": "m",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "A circle labelled A sits above a line."},
                    }
                ],
            }
        ).encode()
        client = self._client(
            monkeypatch, transport=_RecordingTransport(body), api_key="test-key"
        )
        with pytest.raises(ModelBValidationError):
            analyze_image(
                _png_bytes(),
                ModelBSettings(enabled=True, api_key="test-key"),
                client=client,
            )
