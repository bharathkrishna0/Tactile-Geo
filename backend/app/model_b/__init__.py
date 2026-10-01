"""Model B: optional, on-demand, advisory geometry understanding.

Importing this package must stay cheap and dependency-free. The provider
transport in `client.py` uses nothing but the standard library, so the absence
of an API key — or of any network at all — can never break the Model A request
path.
"""

from .errors import (
    ModelBApiError,
    ModelBCancelled,
    ModelBDisabled,
    ModelBError,
    ModelBMalformedResponse,
    ModelBMisconfigured,
    ModelBRateLimited,
    ModelBTimeout,
    ModelBValidationError,
)

__all__ = [
    "ModelBApiError",
    "ModelBCancelled",
    "ModelBDisabled",
    "ModelBError",
    "ModelBMalformedResponse",
    "ModelBMisconfigured",
    "ModelBRateLimited",
    "ModelBTimeout",
    "ModelBValidationError",
]
