"""OpenRouter client for Model B.

Transport only. The contract, the parser, the validator, the normalizer and the
fusion layer are all provider-agnostic and live in sibling modules; this file is
the one place that knows a vendor exists, which is what makes swapping the
gateway a one-file change.

Why the standard library rather than an SDK: OpenRouter speaks the
OpenAI-compatible `/chat/completions` shape over HTTPS, so `urllib.request` is
sufficient. Adding an HTTP SDK for one POST of one JSON body would be a
dependency that has to be patched, audited and kept alive forever in exchange
for nothing. It also means `requirements.txt` contains no provider SDK at all,
so neither Model A nor Model B can break because a vendor SDK changed.

Design notes that are load-bearing:

* **The resolved model is recorded, not assumed.** `OPENROUTER_MODEL` may be a
  router (`openrouter/free`) that picks a different upstream per request, so
  the model actually used comes from the response body. Reporting the
  *configured* model as if it were the model that ran would be a lie that only
  shows up when someone compares results against a different provider.
* **Status code, not message text, classifies failures.** An HTTP status is a
  contract; a provider's error prose is not. String matching survives only as a
  fallback for failures that never got a status.
* **Retries are bounded and never applied to configuration errors.** A 401, 403
  or 404 cannot become true by asking again. Retrying them only delays the
  message that would have helped.
* **`ModelBClientProtocol` is the seam the service depends on.** Every test in
  the suite fakes this one method, so the whole suite runs with no key, no
  network, and no cost.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import (
    ModelBApiError,
    ModelBError,
    ModelBMalformedResponse,
    ModelBMisconfigured,
    ModelBRateLimited,
    ModelBTimeout,
    ModelBValidationError,
)
from .json_schema import RESPONSE_SCHEMA_NAME, build_response_json_schema
from . import diagnostics
from .prompts import build_prompt

# Truncation is not an error condition the teacher caused, and a partial
# analysis is still worth showing, so it is reported rather than raised.
_TRUNCATING_FINISH_REASONS = frozenset({"MAX_TOKENS", "LENGTH", "max_tokens", "length"})

#: Fallback substring probes, used only for exceptions that carry no HTTP
#: status (a `URLError` wrapping a reset connection, for example). Prose is not
#: a contract, so this never overrides a status that was actually received.
_RETRYABLE_PATTERNS = (
    "429",
    "rate limit",
    "too many requests",
    "overloaded",
    "unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "502",
    "503",
    "504",
)
_MISCONFIGURED_PATTERNS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "no auth credentials",
    "permission",
)
_UNKNOWN_MODEL_PATTERNS = (
    "404",
    "not_found",
    "not found",
    "no endpoints found",
    "does not exist",
    "unsupported model",
    "unknown model",
)

#: Gateway identity. Recorded on results so a report can never claim a result
#: came from a provider that did not serve it.
PROVIDER = "openrouter"

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_CHAT_COMPLETIONS_PATH = "/chat/completions"

# Image placeholder for diagnostic summaries. Never sent, never logged.
_DIAGNOSTIC_IMAGE_PLACEHOLDER = b"PNG_DIAGNOSTIC_PLACEHOLDER"
_USER_AGENT = "TactileGeo/0.1 (Model B advisory layer)"
_MAX_OUTPUT_TOKENS = 8192

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Raw provider output, before any Model B interpretation.

    `requested_model` is what configuration asked for; `model` is what the
    gateway says actually served the request. They differ whenever a router is
    configured, and collapsing them would make provenance unfalsifiable.
    """

    text: str
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    requested_model: str = ""
    model: str = ""
    #: Upstream vendor behind the gateway, when the gateway reports one.
    upstream_provider: str = ""
    #: Provider request id. Safe to log and useful when correlating with a
    #: provider-side dashboard. Never a credential.
    request_id: str = ""
    #: Wall-clock time of the provider call, for development metrics only.
    request_ms: float | None = None
    #: Redacted developer diagnostics. Empty unless `MODEL_B_DIAGNOSTICS` is on,
    #: and carrying it on the response is what lets a *parse* failure be
    #: diagnosed: by the time the parser rejects the content, the transport is
    #: out of scope, so without this the raw text is unrecoverable.
    diagnostic: str = ""

    @property
    def resolved_model(self) -> str:
        """The model that actually ran, falling back to what was requested."""
        return self.model or self.requested_model

    @property
    def truncated(self) -> bool:
        return self.finish_reason in _TRUNCATING_FINISH_REASONS

    def payload(self) -> dict[str, Any]:
        """Parse the response body as JSON.

        `parser.extract_json_object` is what the service uses, because a
        structured-output request can still come back wrapped in prose or a
        markdown fence. This strict accessor exists for callers that genuinely
        want the raw body contract.

        Raises:
            ModelBValidationError: if the body is not a JSON object.
        """
        text = self.text.strip()
        if not text:
            raise ModelBValidationError("The model provider returned an empty response body.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelBValidationError(
                f"The model provider response was not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ModelBValidationError(
                f"The model provider response was {type(parsed).__name__}, "
                "expected an object."
            )
        return parsed


@runtime_checkable
class ModelBClientProtocol(Protocol):
    """The seam the service layer depends on."""

    def analyze(
        self, *, image_png: bytes, prepared_width: int, prepared_height: int
    ) -> ProviderResponse:
        ...


class OpenRouterClient:
    """Real provider client. Constructing it does no I/O."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        max_attempts: int = 3,
        retry_base_delay_s: float = 0.8,
        retry_max_delay_s: float = 5.0,
        site_url: str = "",
        app_title: str = "",
        transport: Any = None,
        sleep: Any = None,
    ) -> None:
        if not api_key:
            raise ModelBMisconfigured(
                "OPENROUTER_API_KEY is not set. Set it in the backend "
                "environment to enable Model B."
            )
        if not model:
            raise ModelBMisconfigured("OPENROUTER_MODEL is empty.")
        if timeout_s <= 0:
            raise ModelBMisconfigured("MODEL_B_TIMEOUT_S must be positive.")
        if max_attempts < 1:
            raise ModelBMisconfigured("MODEL_B_MAX_ATTEMPTS must be at least 1.")
        if retry_base_delay_s < 0 or retry_max_delay_s < 0:
            raise ModelBMisconfigured(
                "MODEL_B_RETRY_BASE_DELAY_S and MODEL_B_RETRY_MAX_DELAY_S must not be negative."
            )
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._max_attempts = max_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._site_url = site_url
        self._app_title = app_title
        # Injectable seams. Both default to the real thing; tests replace them so
        # the suite never opens a socket and never actually waits.
        self._transport = transport or _urlopen
        self._sleep = sleep or time.sleep

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}{_CHAT_COMPLETIONS_PATH}"

    def analyze(
        self, *, image_png: bytes, prepared_width: int, prepared_height: int
    ) -> ProviderResponse:
        """Send one image plus the Model B prompt and return the raw text.

        `prepared_width`/`prepared_height` are accepted but unused. They travel
        through the seam so a fake can assert the service handed it the frame it
        claimed to, which is how a mismatched-frame bug gets caught without
        inspecting the image.
        """
        if not image_png:
            # Fail locally rather than paying a round trip to be told the same.
            raise ModelBValidationError("No image was prepared for the Model B request.")

        request_body = self._build_request_body(image_png)
        headers = self._build_headers()

        started = time.perf_counter()
        for attempt in range(1, self._max_attempts + 1):
            logger.info(
                "Model B request start (provider=%s, model=%s, attempt=%d/%d, "
                "image_bytes=%d, timeout=%.0fs)",
                PROVIDER,
                self._model,
                attempt,
                self._max_attempts,
                len(image_png),
                self._timeout_s,
            )
            try:
                return self._attempt(request_body, headers, started)
            except ModelBApiError as error:
                # `ModelBRateLimited` is a `ModelBApiError` with retryable=True,
                # so throttling is retried here too, deferring to whatever wait
                # the provider asked for. `retry_after_s` is only present on that
                # subclass, hence the getattr.
                delay_s = self._backoff_delay(
                    attempt, getattr(error, "retry_after_s", None)
                )
                exhausted = attempt >= self._max_attempts
                if not error.retryable or exhausted:
                    logger.warning(
                        "Model B request failed (provider=%s, model=%s, %.0fms, "
                        "status=%s, mapped=%s, attempt=%d/%d, exhausted=%s)",
                        PROVIDER,
                        self._model,
                        (time.perf_counter() - started) * 1000,
                        getattr(error, "http_status", None),
                        type(error).__name__,
                        attempt,
                        self._max_attempts,
                        exhausted,
                    )
                    raise
                logger.info(
                    "Model B request retryable (provider=%s, model=%s, attempt=%d/%d, "
                    "wait=%.2fs, reason=%s)",
                    PROVIDER,
                    self._model,
                    attempt,
                    self._max_attempts,
                    delay_s,
                    error.detail,
                )
                self._sleep(delay_s)
            except ModelBError as error:
                # Disabled, misconfigured, timed out, malformed, or failed
                # validation. None of these improve by repeating the request.
                logger.warning(
                    "Model B request failed (provider=%s, model=%s, %.0fms, status=%s, "
                    "mapped=%s, attempt=%d/%d)",
                    PROVIDER,
                    self._model,
                    (time.perf_counter() - started) * 1000,
                    error.http_status,
                    type(error).__name__,
                    attempt,
                    self._max_attempts,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                mapped = _map_provider_error(exc, self._model)
                if not isinstance(mapped, ModelBApiError) or not mapped.retryable:
                    raise mapped from exc
                if attempt >= self._max_attempts:
                    raise mapped from exc
                delay_s = self._backoff_delay(attempt, None)
                logger.info(
                    "Model B request retryable (provider=%s, model=%s, attempt=%d/%d, "
                    "wait=%.2fs, exc=%s)",
                    PROVIDER,
                    self._model,
                    attempt,
                    self._max_attempts,
                    delay_s,
                    type(exc).__name__,
                )
                self._sleep(delay_s)

        # Unreachable: the loop either returns or raises. Kept as a hard guard so
        # a future edit cannot silently return None into the normalizer, which
        # would look to a teacher like "the model found nothing".
        raise ModelBApiError("The Model B request failed.")

    # -- request construction -------------------------------------------------

    def _build_request_body(self, image_png: bytes) -> dict[str, Any]:
        """Build the OpenAI-compatible multimodal body.

        Exactly one copy of the image is sent. A second copy would double the
        upload and, on a free tier with a per-request image limit, is the
        difference between a working request and a 400.
        """
        encoded = base64.b64encode(image_png).decode("ascii")
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            # The provider is asked for the contract, not for prose. This is a
            # request, not a guarantee: an upstream that ignores it is still
            # caught by parse -> validate, which rejects rather than repairs.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": RESPONSE_SCHEMA_NAME,
                    "strict": True,
                    "schema": build_response_json_schema(),
                },
            },
            # A vision model classifying geometry has no use for creativity, and
            # non-determinism would make a disagreement with Model A impossible
            # to interpret as noise.
            "temperature": 0,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "stream": False,
        }

    def _build_headers(self) -> dict[str, str]:
        """Headers for one request.

        The key appears here and nowhere else in the codebase. It is never
        logged, never placed in an exception message, and never returned by an
        endpoint.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        # Optional attribution headers. Not credentials, and omitted entirely
        # when unconfigured so a deployment controls its own identity.
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_title:
            headers["X-Title"] = self._app_title
        return headers

    # -- single attempt -------------------------------------------------------

    def _attempt(
        self, body: dict[str, Any], headers: dict[str, str], started: float
    ) -> ProviderResponse:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with self._transport(request, timeout=self._timeout_s) as response:
                raw = response.read()
                status = getattr(response, "status", None) or response.getcode()
                response_headers = dict(getattr(response, "headers", {}) or {})
        except urllib.error.HTTPError as exc:
            raise _error_from_http(exc, self._model) from exc
        except urllib.error.URLError as exc:
            raise _map_provider_error(exc, self._model) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ModelBTimeout(
                "The model provider did not respond before the configured timeout."
            ) from exc
        except OSError as exc:
            raise _map_provider_error(exc, self._model) from exc

        total_ms = (time.perf_counter() - started) * 1000

        # A 2xx with an error-shaped body is still a failure. Trusting the
        # status alone is how a JSON error object gets parsed as a Model B
        # document and reported as "found nothing".
        if isinstance(status, int) and status >= 400:
            raise _error_from_status(
                status, raw, self._model, response_headers.get("Retry-After")
            )

        envelope = _decode_envelope(raw)

        # A 2xx carrying an error envelope is still a failure, and it is not
        # hypothetical: a gateway that has no route to a model can answer 200
        # with `{"error": ...}`. Trusting the status alone would hand that
        # object to the parser, which would then report "the model found no
        # geometry" — the most misleading possible answer to a teacher.
        if _envelope_is_error(envelope):
            message = _provider_detail(raw)
            raise ModelBApiError(
                "The model provider returned an error"
                + (f": {message}" if message else ".")
                + " Model A is unaffected.",
                retryable=False,
                http_status=int(status) if isinstance(status, int) else None,
            )

        if not _has_choices(envelope):
            # A 2xx with neither a choice nor an error is an envelope we do not
            # understand. Caught here rather than downstream so the message says
            # "the provider sent nothing usable" instead of "the page had no
            # geometry", which points the teacher at the wrong problem.
            raise ModelBMalformedResponse(
                "The model provider returned a response with no completion."
            )

        text = _extract_text(envelope)
        finish_reason = _extract_finish_reason(envelope)
        model = _extract_str(envelope, "model")
        upstream = _extract_str(envelope, "provider")
        request_id = _extract_str(envelope, "id")

        logger.info(
            "Model B request ok (provider=%s, requested_model=%s, resolved_model=%s, "
            "upstream=%s, status=%s, total=%.0fms, response_bytes=%d, finish=%s, usage=%s)",
            PROVIDER,
            self._model,
            model or "<not reported>",
            upstream or "<not reported>",
            status,
            total_ms,
            len(raw),
            finish_reason or "<none>",
            _usage_summary(_extract_usage(envelope)),
        )

        return ProviderResponse(
            text=text,
            finish_reason=finish_reason,
            usage=_extract_usage(envelope),
            requested_model=self._model,
            model=model,
            upstream_provider=upstream,
            request_id=request_id,
            request_ms=total_ms,
            diagnostic=self._diagnostic(raw, status, envelope, text),
        )

    def _diagnostic(
        self,
        raw: bytes,
        status: Any,
        envelope: dict[str, Any],
        text: str,
    ) -> str:
        """Redacted request/response summary, or "" when diagnostics are off.

        Built here rather than in the caller because this is the last point at
        which both the request and the raw response are in scope together.
        """
        if not diagnostics.enabled():
            return ""
        lines = ["--- Model B request ---"]
        try:
            req_body = self._build_request_body(_DIAGNOSTIC_IMAGE_PLACEHOLDER)
            lines.append(
                diagnostics.summarize_request(
                    req_body, self._build_headers(), self.endpoint
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never mask errors
            lines.append(f"(failed to summarize request: {type(exc).__name__})")
        lines.append("--- Model B response ---")
        lines.append(
            diagnostics.summarize_response(
                status=status,
                raw=raw,
                envelope=envelope,
                content=text,
                content_source="choices[0].message.content",
            )
        )
        return "\n".join(lines)
    def _backoff_delay(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff, capped, deferring to a provider-supplied wait.

        A provider that tells us how long to wait knows better than our curve
        does. The cap keeps a 429 storm from holding a worker slot open.
        """
        if retry_after_s is not None and retry_after_s >= 0:
            return min(retry_after_s, self._retry_max_delay_s)
        delay = self._retry_base_delay_s * (2 ** (attempt - 1))
        return min(delay, self._retry_max_delay_s)


# --- module-level helpers ---------------------------------------------------
# Kept as plain functions rather than methods so tests can assert error
# classification directly, without constructing a client or a fake transport.


def _urlopen(request: Any, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _http_status_of(exc: Exception) -> int | None:
    """HTTP status from a provider error, or None.

    Read from the attribute rather than the message text so that a message
    containing a stray number cannot be mistaken for a status code.
    """
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_seconds(raw: Any) -> float | None:
    """Read a `Retry-After` header, tolerating both documented forms.

    OpenRouter sends seconds. HTTP also allows an HTTP-date, which is not worth
    a date parser in an advisory code path: an unparseable value is treated as
    absent and the normal backoff applies.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_from_http(exc: urllib.error.HTTPError, model: str | None = None) -> Exception:
    """Translate an HTTPError into a typed Model B error."""
    status = exc.code if isinstance(exc.code, int) else 502
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 - a broken body must not mask the status
        raw = b""
    retry_after = _retry_after_seconds(
        (getattr(exc, "headers", {}) or {}).get("Retry-After")
    )
    return _error_from_status(status, raw, model, retry_after)


def _error_from_status(
    status: int,
    raw: bytes | str | None,
    model: str | None = None,
    retry_after: Any = None,
) -> Exception:
    """Classify a failure by HTTP status.

    The ordering is the design. A misconfiguration is checked before a retryable
    condition, because a 403 can look transient to a substring match and telling
    a teacher to "try again" when the key is dead wastes a lesson.
    """
    requested = f" Requested: {model}." if model else ""

    if status in (401, 403):
        return ModelBMisconfigured(
            "OpenRouter rejected the request. Check that OPENROUTER_API_KEY is "
            "correct, active, and has credit or free-tier access.",
            http_status=status,
        )
    if status == 404:
        return ModelBMisconfigured(
            "OpenRouter does not offer the configured model, or no upstream "
            "provider is available for it right now. Check OPENROUTER_MODEL."
            + requested,
            http_status=status,
        )
    if status == 408:
        return ModelBTimeout(
            "The model provider did not respond before the configured timeout.",
            http_status=status,
        )
    if status == 429:
        seconds = _retry_after_seconds(retry_after)
        wait = (
            f" Try again in about {int(seconds)} seconds."
            if seconds is not None
            else " Try again shortly."
        )
        return ModelBRateLimited(
            "The model provider is rate limiting requests." + wait,
            retry_after_s=seconds,
            http_status=status,
        )
    if status >= 500:
        return ModelBApiError(
            "The model provider is temporarily unavailable. Model A is unaffected.",
            retryable=True,
            http_status=status,
        )
    # 400 and friends: the request itself is wrong. Not retryable.
    detail = _provider_detail(raw)
    return ModelBMisconfigured(
        "OpenRouter refused the request."
        + (f" {detail}" if detail else "")
        + " Check OPENROUTER_MODEL and that the model accepts images.",
        http_status=status,
    )


def _provider_detail(raw: bytes | str | None, limit: int = 240) -> str:
    """Best-effort provider error text, truncated.

    OpenRouter's error envelope carries a human-readable `message`. It is safe
    enough to surface for a 4xx because it describes the request, never the
    credential. Truncated because a provider can put an entire page in there.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - decode with 'replace' cannot fail
            return ""
    text = str(raw).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:limit]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()[:limit]
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:limit]
    return text[:limit]


def _map_provider_error(exc: Exception, model: str | None = None) -> Exception:
    """Translate a status-less provider exception into a typed Model B error.

    `model` is only ever a model identifier from configuration, never a
    credential, so naming it in the "unknown model" message is safe and is what
    makes that error diagnosable.
    """
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return ModelBTimeout(
            "The model provider did not respond before the configured timeout."
        )
    if isinstance(exc, urllib.error.HTTPError):
        return _error_from_http(exc, model)

    status = _http_status_of(exc)
    if status is not None:
        return _error_from_status(status, None, model)

    reason = getattr(exc, "reason", None)
    if isinstance(exc, urllib.error.URLError) and reason is not None:
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return ModelBTimeout(
                "The model provider did not respond before the configured timeout."
            )
        mapped = _classify_text(f"{type(reason).__name__} {reason}".lower(), model)
        if mapped is not None:
            return mapped

    mapped = _classify_text(f"{type(exc).__name__} {exc}".lower(), model)
    if mapped is not None:
        return mapped
    return ModelBApiError("The Model B request failed.", retryable=True)


def _classify_text(haystack: str, model: str | None) -> Exception | None:
    """Substring classification for failures with no HTTP status.

    Returns None when nothing matches, so the caller can decide the default.
    Order matters: misconfiguration wins over retryable.
    """
    requested = f" Requested: {model}." if model else ""
    if any(pattern in haystack for pattern in _MISCONFIGURED_PATTERNS):
        return ModelBMisconfigured(
            "OpenRouter rejected the request. Check that OPENROUTER_API_KEY is "
            "correct, active, and has credit or free-tier access."
        )
    if any(pattern in haystack for pattern in _UNKNOWN_MODEL_PATTERNS):
        return ModelBMisconfigured(
            "OpenRouter does not offer the configured model. Check OPENROUTER_MODEL."
            + requested
        )
    if any(pattern in haystack for pattern in _RETRYABLE_PATTERNS):
        return ModelBApiError(
            "The model provider is temporarily unavailable. Model A is unaffected.",
            retryable=True,
        )
    return None


def _decode_envelope(raw: bytes | str) -> dict[str, Any]:
    """Decode the response envelope, or fail with a clear error.

    An unparseable envelope is a malformed response, not an empty analysis, and
    the two must not be confused: one means "try again", the other means "the
    page had nothing on it".
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - 'replace' means this cannot fail
            raw = ""
    text = str(raw or "").strip()
    if not text:
        raise ModelBValidationError("The model provider returned an empty response body.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelBMalformedResponse(
            f"The model provider response was not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelBMalformedResponse(
            f"The model provider response was {type(parsed).__name__}, expected an object."
        )
    return parsed


def _envelope_is_error(envelope: dict[str, Any]) -> bool:
    """Whether a 2xx body is actually an error.

    Requires an `error` field *and* the absence of any choice, so a legitimate
    response that merely happens to carry an `error` key alongside content is
    not misread as a failure.
    """
    if not isinstance(envelope.get("error"), (dict, str)):
        return False
    choices = envelope.get("choices")
    return not (isinstance(choices, list) and choices)


def _has_choices(envelope: dict[str, Any]) -> bool:
    choices = envelope.get("choices")
    return isinstance(choices, list) and len(choices) > 0


def _extract_text(envelope: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible envelope.

    Handles both shapes in the wild: `content` as a plain string, and `content`
    as a list of typed parts. Returning "" rather than raising is deliberate —
    `service.analyze_image` already turns empty text into a clear error via the
    parser, and that path already has a test.
    """
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some upstreams return `[{"type": "text", "text": "..."}]` even when
        # only text parts are present.
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def _extract_finish_reason(envelope: dict[str, Any]) -> str:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    reason = first.get("finish_reason")
    return reason.strip() if isinstance(reason, str) else ""


def _extract_usage(envelope: dict[str, Any]) -> dict[str, Any]:
    """Best-effort token accounting. Never raises.

    Usage data is for the server log and cost tracking, so a missing field must
    not turn a successful analysis into a failed request. Reasoning tokens are
    included because several free-tier vision models are reasoning models and
    their cost is otherwise invisible.
    """
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return {}

    collected: dict[str, Any] = {}
    for field_name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        value = usage.get(field_name)
        if isinstance(value, int):
            collected[field_name] = value

    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
        if isinstance(reasoning, int):
            collected["reasoning_tokens"] = reasoning

    if usage.get("cost") is not None:
        # Numeric cost only, and only what the gateway chose to report.
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            collected["cost"] = float(cost)
    return collected


def _usage_summary(usage: dict[str, Any]) -> str:
    """Compact usage string for a log line."""
    if not usage:
        return "none"
    return ",".join(f"{key}={value}" for key, value in sorted(usage.items()))


def _extract_str(envelope: dict[str, Any], key: str) -> str:
    value = envelope.get(key)
    return value.strip() if isinstance(value, str) else ""
