"""Model B orchestration.

One function owns the whole sequence so the ordering constraints live in a
single readable place:

    prepare -> call -> parse -> validate -> normalize

`parse` runs before `validate` because validation needs a dict, and `validate`
runs before `normalize` because normalization trusts the contract. Each step
raises a distinct typed error, so the API layer can map failures to statuses
without inspecting messages.

Nothing here names an upstream vendor. The gateway is OpenRouter, but the model
behind it is configuration, and this layer must keep working unchanged when
`OPENROUTER_MODEL` changes to a different family.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models.model_b_result import ModelBResult
from .client import (
    PROVIDER,
    ModelBClientProtocol,
    OpenRouterClient,
    ProviderResponse,
)
from .errors import ModelBDisabled, ModelBError, ModelBMisconfigured
from .json_schema import UnsupportedSchemaError, build_response_json_schema
from .normalizer import normalize
from .parser import extract_json_object
from .preparation import prepare_image
from .validator import validate_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBSettings:
    """Resolved configuration for one Model B request.

    Held as a value object so tests can construct a valid configuration without
    touching environment variables, which are process-global and therefore
    hostile to parallel tests.
    """

    enabled: bool = False
    api_key: str | None = None
    #: Configured model or router, e.g. `openrouter/free`. Not necessarily the
    #: model that ends up serving the request.
    model: str = "openrouter/free"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 30.0
    max_attempts: int = 3
    retry_base_delay_s: float = 0.8
    retry_max_delay_s: float = 5.0
    site_url: str = ""
    app_title: str = "TactileGeo"

    @property
    def provider(self) -> str:
        """Gateway identity, for logs and result provenance."""
        return PROVIDER

    def require_available(self) -> None:
        """Raise the most specific reason Model B cannot run.

        Checked in escalating order so a teacher debugging setup is told about
        the missing key, not a downstream symptom.
        """
        if not self.enabled:
            raise ModelBDisabled(
                "Model B is turned off. Set MODEL_B_ENABLED=true to allow "
                "teachers to request an advisory vision-language analysis."
            )
        if not self.api_key:
            raise ModelBMisconfigured(
                "Model B is enabled but OPENROUTER_API_KEY is not set."
            )

    def build_client(self) -> ModelBClientProtocol:
        self.require_available()
        if not self.model:
            raise ModelBMisconfigured("OPENROUTER_MODEL is empty.")
        return OpenRouterClient(
            api_key=self.api_key or "",
            model=self.model,
            timeout_s=self.timeout_s,
            base_url=self.base_url,
            max_attempts=self.max_attempts,
            retry_base_delay_s=self.retry_base_delay_s,
            retry_max_delay_s=self.retry_max_delay_s,
            site_url=self.site_url,
            app_title=self.app_title,
        )


def is_available(settings: ModelBSettings) -> bool:
    """Whether Model B can serve a request right now, without raising."""
    return bool(settings.enabled and settings.api_key)


def analyze_image(
    image_bytes: bytes,
    settings: ModelBSettings,
    client: ModelBClientProtocol | None = None,
) -> ModelBResult:
    """Run the full Model B analysis.

    Args:
        image_bytes: raw uploaded file, in whatever format was uploaded.
        settings: resolved configuration.
        client: injectable for tests. When omitted, a real OpenRouter client is
            built from `settings`.

    Raises:
        ModelBDisabled, ModelBMisconfigured, ModelBRateLimited, ModelBTimeout,
        ModelBApiError, ModelBMalformedResponse, ModelBValidationError
    """
    settings.require_available()

    # Fail before spending an image encode on a contract we know is broken.
    try:
        build_response_json_schema()
    except UnsupportedSchemaError as exc:
        raise ModelBMisconfigured(
            "The Model B response contract is incompatible with the model provider."
        ) from exc

    prepared = prepare_image(image_bytes)
    active_client = client if client is not None else settings.build_client()

    response: ProviderResponse = active_client.analyze(
        image_png=prepared.png_bytes,
        prepared_width=prepared.width,
        prepared_height=prepared.height,
    )

    # A configured router is not the model that ran. Log the difference on every
    # job, because "which model produced this?" is otherwise unanswerable.
    if response.resolved_model and response.resolved_model != settings.model:
        logger.info(
            "Model B provider routed %s to %s (upstream=%s)",
            settings.model,
            response.resolved_model,
            response.upstream_provider or "<not reported>",
        )

    try:
        payload = extract_json_object(response.text)
        document, warnings = validate_document(
            payload, prepared_width=prepared.width, prepared_height=prepared.height
        )
    except ModelBError as error:
        # A contract failure is the one place the raw model content is
        # genuinely needed, and the transport is out of scope by now. Carrying
        # the redacted diagnostic on the error is what makes a rejected
        # response diagnosable at all. Empty unless MODEL_B_DIAGNOSTICS is on.
        if response.diagnostic:
            error.diagnostic = response.diagnostic  # type: ignore[attr-defined]
        raise

    result = normalize(
        document,
        prepared,
        provider=settings.provider,
        requested_model=response.requested_model or settings.model,
        resolved_model=response.resolved_model,
        upstream_provider=response.upstream_provider,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        truncated=response.truncated,
        usage=response.usage,
        validation_warnings=warnings,
    )

    if result.truncated:
        # A truncated document can still be internally consistent, so it is
        # reported rather than rejected. Saying so explicitly is the difference
        # between "found nothing" and "ran out of budget".
        logger.warning(
            "Model B response truncated: finish_reason=%s entities=%d",
            result.finish_reason,
            len(result.entities),
        )
    for warning in warnings:
        logger.info("Model B validation warning: %s", warning)

    return result
