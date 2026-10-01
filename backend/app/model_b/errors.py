"""Model B errors.

Each failure mode maps to a distinct HTTP status and a distinct teacher-facing
message, because "something went wrong" is useless during a lesson. A teacher
needs to know whether to retry, fix the key, or use Model A instead.
"""

from __future__ import annotations


class ModelBError(Exception):
    """Base for all Model B failures. Never leaks provider internals."""

    status_code = 502
    reason = "model_b_error"

    def __init__(self, detail: str, *, http_status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        #: The provider's HTTP status, when there was one. Carried for the
        #: server log and for deciding whether a retry is meaningful; it is not
        #: part of `detail` and never reaches the browser.
        self.http_status = http_status


class ModelBDisabled(ModelBError):
    """Model B was never enabled, or has no API key configured."""

    status_code = 503
    reason = "model_b_disabled"


class ModelBMisconfigured(ModelBError):
    """Enabled, but a required setting is missing or unusable."""

    status_code = 503
    reason = "model_b_misconfigured"


class ModelBTimeout(ModelBError):
    """Provider did not answer inside the configured budget."""

    status_code = 504
    reason = "model_b_timeout"


class ModelBApiError(ModelBError):
    """Provider returned an error, was throttled, or refused the request.

    `retryable` is surfaced to the client so the UI can offer a retry button
    only when retrying can actually help.
    """

    status_code = 502
    reason = "model_b_api_error"

    def __init__(
        self,
        detail: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(detail, http_status=http_status)
        self.retryable = retryable


class ModelBRateLimited(ModelBApiError):
    """The provider is throttling us and said when it will free up.

    A distinct type rather than a generic retryable API error, because it is the
    one Model B failure with a known, provider-supplied remedy: wait
    `retry_after_s`. The UI surfaces it as "rate limited" instead of "something
    went wrong", and `retry_after_s` is advisory — a provider that omits the
    header still produces a usable message.
    """

    status_code = 429
    reason = "model_b_rate_limited"

    def __init__(
        self,
        detail: str,
        *,
        retry_after_s: float | None = None,
        retryable: bool = True,
        http_status: int | None = None,
    ) -> None:
        super().__init__(detail, retryable=retryable, http_status=http_status)
        self.retry_after_s = retry_after_s


class ModelBMalformedResponse(ModelBError):
    """Response was not valid JSON, or was truncated mid-document."""

    status_code = 502
    reason = "model_b_malformed_response"


class ModelBValidationError(ModelBError):
    """Response parsed as JSON but violated the Model B contract.

    `issues` holds field-level messages for the server log. It is intentionally
    not returned to the client, because echoing provider output back to a
    browser is how you end up serving unescaped model text.
    """

    status_code = 502
    reason = "model_b_invalid_response"

    def __init__(self, detail: str, issues: list[str] | None = None) -> None:
        super().__init__(detail)
        self.issues = issues or []


class ModelBCancelled(ModelBError):
    """Job was cancelled by the teacher before it finished."""

    status_code = 409
    reason = "model_b_cancelled"
