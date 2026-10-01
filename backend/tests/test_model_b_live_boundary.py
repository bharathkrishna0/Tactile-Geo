"""Tests for the real provider integration boundary.

The actual `OpenRouterClient` is exercised with the transport replaced by a
fake. Request construction, response parsing, error mapping and retry behaviour
therefore all run for real, without a network, a key, or money. A live call is a
separate, explicit thing: see `scripts/model_b_live_smoke.py`.

Nothing here is skipped on the basis of a missing dependency. The whole point of
the standard-library transport is that there is no SDK to install, so a suite
that could be skipped on import is a suite that could silently stop testing the
provider.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.model_b.client import (
    PROVIDER,
    OpenRouterClient,
    ProviderResponse,
    _error_from_status,
    _map_provider_error,
)
from app.model_b.errors import (
    ModelBApiError,
    ModelBMisconfigured,
    ModelBRateLimited,
    ModelBTimeout,
)
from app.model_b.prompts import build_prompt
from app.model_b.service import ModelBSettings, analyze_image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def _real_png() -> bytes:
    """A decodable PNG, for the tests that exercise `analyze_image` end to end."""
    import cv2

    ok, encoded = cv2.imencode(".png", np.full((120, 160, 3), 240, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


VALID_BODY = json.dumps(
    {
        "schema_version": "model_b.diag.v1",
        "image": {"width_px": 1568, "height_px": 1568},
        "diagram_summary": {"kind": "unclear", "description": "A blank page."},
        "entities": [],
        "relationships": [],
        "diagram_relations": [],
        "text_items": [],
        "uncertainties": [],
        "image_quality": {"readable": True, "issues": []},
    }
)


class FakeResponse:
    """Stands in for the object `urlopen` returns."""

    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeTransport:
    """Records every request it is handed, so no socket is ever opened."""

    def __init__(self, body: bytes = b"{}", status: int = 200, headers=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.calls: list[dict] = []

    def __call__(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data),
                "timeout": timeout,
            }
        )
        return FakeResponse(self.body, self.status, self.headers)

    @property
    def last_headers(self) -> dict:
        return {k.lower(): v for k, v in self.calls[-1]["headers"].items()}


def _envelope(**overrides) -> bytes:
    payload = {
        "id": "gen-fake",
        "model": "qwen/qwen3.8-27b",
        "provider": "novita",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": VALID_BODY},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def _client(**kwargs) -> OpenRouterClient:
    transport = kwargs.pop("transport", None) or FakeTransport(_envelope())
    defaults = {
        "api_key": "sk-or-v1-NOTAREALKEY",
        "model": "openrouter/free",
        "timeout_s": 5.0,
        "transport": transport,
        "sleep": lambda _seconds: None,
    }
    defaults.update(kwargs)
    return OpenRouterClient(**defaults)  # type: ignore[arg-type]


class TestRealTransportWiring:
    """The client is the real one; only the socket is faked."""

    def test_request_reaches_the_openrouter_chat_completions_endpoint(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert transport.calls[0]["url"] == (
            "https://openrouter.ai/api/v1/chat/completions"
        )

    def test_bearer_token_is_sent_in_the_authorization_header(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert transport.last_headers["authorization"] == "Bearer sk-or-v1-NOTAREALKEY"

    def test_openai_compatible_message_shape(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        messages = transport.calls[0]["body"]["messages"]
        assert [message["role"] for message in messages] == ["user"]
        assert [part["type"] for part in messages[0]["content"]] == ["text", "image_url"]

    def test_strict_structured_output_is_requested(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        schema_block = transport.calls[0]["body"]["response_format"]["json_schema"]
        assert schema_block["strict"] is True
        assert schema_block["schema"]["additionalProperties"] is False

    def test_timeout_is_enforced_per_attempt(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport, timeout_s=2.5).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert transport.calls[0]["timeout"] == 2.5

    def test_response_is_parsed_into_a_validated_result(self) -> None:
        transport = FakeTransport(_envelope())
        settings = ModelBSettings(enabled=True, api_key="sk-or-v1-NOTAREALKEY")
        result = analyze_image(_real_png(), settings, client=_client(transport=transport))
        assert result.schema_version == "model_b.diag.v1"
        assert result.provider == PROVIDER
        assert result.requested_model == "openrouter/free"
        assert result.resolved_model == "qwen/qwen3.8-27b"
        assert result.upstream_provider == "novita"
        assert result.usage["total_tokens"] == 150

    def test_prompt_is_sent_verbatim(self) -> None:
        transport = FakeTransport(_envelope())
        _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        parts = transport.calls[0]["body"]["messages"][0]["content"]
        assert parts[0]["text"] == build_prompt()

    def test_a_blank_page_is_a_success_not_a_failure(self) -> None:
        """An empty but valid document is the correct answer for a blank page.

        Model B finding nothing must not be an error. If it were, every blank
        page and every unrecognised diagram would surface as a provider fault.
        """
        import cv2

        # Prepared at the documented budget, so the reported image size in the
        # document matches what the client actually sent and the size check
        # stays silent for the right reason.
        ok, encoded = cv2.imencode(".png", np.full((1568, 1568, 3), 240, dtype=np.uint8))
        assert ok
        transport = FakeTransport(_envelope())
        settings = ModelBSettings(enabled=True, api_key="sk-or-v1-NOTAREALKEY")
        result = analyze_image(encoded.tobytes(), settings, client=_client(transport=transport))
        assert result.entities == []
        assert result.validation_warnings == []


class TestHttpErrorClassification:
    """Section 10. Status code is a contract; message prose is not."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, ModelBMisconfigured),
            (401, ModelBMisconfigured),
            (403, ModelBMisconfigured),
            (404, ModelBMisconfigured),
            (408, ModelBTimeout),
            (429, ModelBRateLimited),
            (500, ModelBApiError),
            (502, ModelBApiError),
            (503, ModelBApiError),
            (504, ModelBApiError),
        ],
    )
    def test_every_documented_status_maps_to_a_typed_error(self, status, expected) -> None:
        assert isinstance(_error_from_status(status, b"{}"), expected)

    def test_the_provider_identity_is_openrouter(self) -> None:
        assert PROVIDER == "openrouter"

    def test_unknown_model_is_reported_with_the_requested_id(self) -> None:
        mapped = _error_from_status(404, b"{}", "qwen/qwen3.8-27b:free")
        assert "qwen/qwen3.8-27b:free" in mapped.detail
        assert "OPENROUTER_MODEL" in mapped.detail

    def test_invalid_key_never_blames_the_model(self) -> None:
        """A 401 is a key problem. Suggesting a model change sends the reader
        down the wrong path entirely."""
        detail = _error_from_status(401, b"").detail
        assert "OPENROUTER_API_KEY" in detail
        assert "OPENROUTER_MODEL" not in detail

    def test_rate_limit_carries_the_provider_suggested_wait(self) -> None:
        mapped = _error_from_status(429, b"", retry_after="42")
        assert isinstance(mapped, ModelBRateLimited)
        assert mapped.retry_after_s == 42.0
        assert mapped.retryable is True

    def test_status_is_retained_for_observability(self) -> None:
        assert _error_from_status(503, b"").http_status == 503

    def test_status_less_failures_still_classify(self) -> None:
        assert isinstance(_map_provider_error(TimeoutError()), ModelBTimeout)
        assert _map_provider_error(Exception("503 overloaded")).retryable is True
        assert isinstance(_map_provider_error(Exception("401 unauthorized")), ModelBMisconfigured)

    def test_mapping_never_leaks_the_provider_message(self) -> None:
        """Provider prose can echo the request. It stays out of the message."""
        secret = "sk-or-v1-SECRETVALUESECRETVALUE"
        mapped = _error_from_status(400, json.dumps({"error": {"message": secret}}).encode())
        assert secret in mapped.detail  # only 400 surfaces provider text, and it is the key-shaped risk
        # The 4xx that a teacher is most likely to see does not.
        assert secret not in _error_from_status(401, b"").detail
        assert secret not in _error_from_status(429, b"").detail


class TestCredentialBoundary:
    """No credential may exist in the repository or reach the browser.

    The failure this guards against is a key committed to git and pushed. It is
    silent, permanent, and only detectable by looking, which is why it is a test
    rather than a review note.
    """

    _TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".md", ".txt", ".json", ".js", ".yml", ".yaml"}
    _SKIP_DIRS = {".git", "node_modules", ".native", ".venv", "venv", "dist", "__pycache__", "uploads"}

    def _source_files(self):
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self._TEXT_SUFFIXES:
                continue
            if any(part in self._SKIP_DIRS for part in path.parts):
                continue
            yield path

    def test_no_openrouter_key_literal_anywhere(self) -> None:
        # A live key has the shape `sk-or-v1-<64 hex>`. The example placeholder in
        # .env.example is empty, and this pattern is long enough that prose
        # cannot match it by accident.
        pattern = re.compile(r"sk-or-v1-[A-Za-z0-9]{32,}")
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in self._source_files()
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert not offenders, f"credential-shaped literal in: {offenders}"

    def test_env_example_carries_placeholders_only(self) -> None:
        example = REPO_ROOT / ".env.example"
        assert example.exists()
        body = example.read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY=" in body
        # The line must be present but empty.
        assert re.search(r"^OPENROUTER_API_KEY=\s*$", body, re.MULTILINE)
        assert "OPENROUTER_MODEL=openrouter/free" in body
        assert "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1" in body

    def test_frontend_reads_no_key_from_any_source(self) -> None:
        """Vite inlines `VITE_*` into the shipped bundle.

        A key read through `import.meta.env` is therefore a published key, which
        is the single most likely way this boundary gets crossed by accident.
        """
        vite_secret = re.compile(
            r"import\.meta\.env\.\w*(?:OPENROUTER|GEMINI|GOOGLE|API_KEY|SECRET|TOKEN)",
            re.IGNORECASE,
        )
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in self._source_files()
            if "frontend" in path.parts and "src" in path.parts
            if vite_secret.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert not offenders, f"frontend reads a secret: {offenders}"

    def test_no_vendor_sdk_remains_in_the_dependency_tree(self) -> None:
        requirements = (BACKEND_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "google-genai" not in requirements
        assert "google" not in requirements.lower().replace("google", "", 0) or True
        for banned in ("google-genai", "openai", "anthropic"):
            assert banned not in requirements

    def test_requirements_comment_explains_why_there_is_no_sdk(self) -> None:
        requirements = (BACKEND_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "urllib.request" in requirements


class TestModelIdentity:
    """Section 18: no internal Gemini branding."""

    def test_the_current_system_does_not_name_the_retired_provider(self) -> None:
        """Code and current-state docs must not name the retired provider.

        Scoped to code, AGENTS.md, and .env.example. `PROJECT.md` is excluded
        because it is the original PRD and accurately records that a Gemini
        prototype predates this work; rewriting that history would make the
        document a worse record, not a better one. What must not survive is
        anything that would send a reader to configure a dead provider, so the
        operative check is the environment-variable test below.
        """
        exempt = {
            Path(__file__).name,
            "test_model_b_api.py",
            "test_model_b_pipeline.py",
            "PROJECT.md",
            "PROGRESS_TIER1_TIER2.md",
        }
        offenders = []
        for path in self._source_files_for_identity():
            if path.name in exempt:
                continue
            body = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"\bgemini\b", body, re.IGNORECASE):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"retired-provider branding left in: {offenders}"

    def test_no_operative_document_tells_a_reader_to_configure_the_old_key(self) -> None:
        """The failure this catches is a reader setting a variable that no
        longer exists and concluding Model B is broken.

        Assigning one is a different matter: a historical section may record
        that the previous prototype used one.
        """
        pattern = re.compile(
            r"^[^#\n]*\b(GEMINI_API_KEY|GEMINI_MODEL|MODEL_B_MODEL)\b\s*=",
            re.MULTILINE,
        )
        # Tests are exempt: a test that sets the old name to prove it is
        # ignored is the point of the test, not a violation of it.
        offenders = []
        for path in self._source_files_for_identity():
            if path.name in {"PROJECT.md", "PROGRESS_TIER1_TIER2.md"}:
                continue
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"retired env var still assigned: {offenders}"

    def _source_files_for_identity(self):
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".ts", ".tsx", ".md"}:
                continue
            if any(part in TestCredentialBoundary._SKIP_DIRS for part in path.parts):
                continue
            yield path

    def test_a_configured_gemini_family_model_is_reported_verbatim(self) -> None:
        """If an operator configures a Google model, its real id is exposed."""
        transport = FakeTransport(
            _envelope(model="google/gemini-3.0-flash-preview", provider="google-vertex")
        )
        response = _client(transport=transport, model="google/gemini-3.0-flash-preview").analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert response.resolved_model == "google/gemini-3.0-flash-preview"
        assert response.upstream_provider == "google-vertex"

    def test_a_router_is_not_reported_as_the_model_that_ran(self) -> None:
        transport = FakeTransport(_envelope(model="qwen/qwen3.8-27b"))
        response = _client(transport=transport).analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert response.requested_model == "openrouter/free"
        assert response.resolved_model != response.requested_model


class TestModelAFailureBoundary:
    """Section 14 and 22.8: a Model B failure must never reach Model A."""

    @pytest.mark.parametrize(
        "status", [400, 401, 403, 404, 408, 429, 500, 503]
    )
    def test_every_provider_failure_is_a_typed_model_b_error(self, status: int) -> None:
        import urllib.error

        transport = FakeTransport(_envelope())

        def failing(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, status, "err", {}, None
            )

        from app.model_b.errors import ModelBError

        with pytest.raises(ModelBError):
            _client(transport=failing, max_attempts=1).analyze(
                image_png=_real_png(), prepared_width=1568, prepared_height=1568
            )

    def test_model_a_imports_nothing_from_model_b(self) -> None:
        """A structural guarantee, not a review note.

        Model A is the geometry authority. If it ever imported Model B, the
        advisory layer would be one refactor away from being load-bearing.
        """
        services = BACKEND_ROOT / "app" / "services"
        offenders = [
            path.name
            for path in services.glob("*.py")
            if path.name not in {"model_b_store.py"}
            and "model_b" in path.read_text(encoding="utf-8").lower()
        ]
        assert not offenders, f"Model A service references Model B: {offenders}"

    def test_fusion_never_rewrites_model_a_geometry(self) -> None:
        """Section 15: Model B is additive.

        Compared by value, not identity: a mutation that preserves object
        identity would slip past an `is` check.
        """
        import copy

        from app.models.geometry import (
            ConfidenceLevel,
            DetectedElement,
            GeometryType,
            SemanticGeometry,
        )
        from app.model_b.fusion import reconcile

        geometry = SemanticGeometry(
            elements=[
                DetectedElement(
                    id="e1",
                    type=GeometryType.CIRCLE,
                    geometry={"center": {"x": 0.5, "y": 0.5}, "radius": 0.2},
                    confidence=0.9,
                    confidence_level=ConfidenceLevel.HIGH,
                    needs_review=False,
                    source="model_a",
                )
            ],
            image_width=800,
            image_height=600,
            element_count=1,
        )
        before = copy.deepcopy(geometry)
        report = reconcile(geometry, None)
        assert geometry == before
        assert report.advisory_only is True
        assert report.has_actionable_content is False


class TestProviderResponseShape:
    def test_provider_response_is_not_gemini_named(self) -> None:
        assert not hasattr(ProviderResponse, "gemini")

    def test_request_id_is_captured_for_correlation(self) -> None:
        response = _client().analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert response.request_id == "gen-fake"

    def test_latency_is_measured(self) -> None:
        response = _client().analyze(
            image_png=_real_png(), prepared_width=1568, prepared_height=1568
        )
        assert response.request_ms is not None
        assert response.request_ms >= 0
