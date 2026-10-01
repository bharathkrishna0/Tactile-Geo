from pathlib import Path
import os

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
UPLOAD_DIRECTORY = Path(os.getenv("UPLOAD_DIRECTORY", "uploads"))
ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]

# Liblouis table path. If empty/unset, the braille service searches common
# install locations (see braille.py) for the UEB tables.
LOUIS_TABLEPATH = os.getenv("LOUIS_TABLEPATH", "").strip() or None

# --- Model B (optional OpenRouter-backed advisory layer) -----------------
# Model B is OFF unless explicitly enabled, and stays unavailable until a key is
# present. Two independent switches is deliberate: enabling the feature in config
# and forgetting the key should produce a clear "no key" message, not a crash on
# the first request.
#
# OpenRouter is the only gateway. It fronts many vision-language models behind
# one OpenAI-compatible API, so the upstream model is configuration rather than
# code. There is deliberately no second provider fallback: silently falling back
# to a different vendor would change cost, latency, and the accuracy profile
# without anyone deciding to.
#
# The key is backend-only. It is never returned by any endpoint, never logged,
# and never sent to the browser.
MODEL_B_ENABLED = os.getenv("MODEL_B_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_B_PROVIDER = "openrouter"
MODEL_B_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip() or None

# `openrouter/free` is the default because it needs no model research to try.
# It is a ROUTER, not a model: each request may be served by a different
# upstream. That is why the resolved model is recorded on every result rather
# than assumed. Set OPENROUTER_MODEL to a specific vision model (for example
# `qwen/qwen3.8-27b:free`) for reproducible output.
#
# No model id is hardcoded in business logic; this is only a default, and
# anything set in the environment wins.
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_MODEL = (
    os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_OPENROUTER_MODEL
)
OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL", "").strip().rstrip("/")
    or "https://openrouter.ai/api/v1"
)

# Per-attempt timeout. A free tier is slow and rate-limited, so this is kept
# short enough that a teacher sees a failure rather than a spinner.
MODEL_B_TIMEOUT_S = float(os.getenv("MODEL_B_TIMEOUT_S", "30").strip() or "30")

# One in-flight Model B job per session. Prevents a double-click from paying
# twice for the same image.
MODEL_B_MAX_CONCURRENT_JOBS = int(
    os.getenv("MODEL_B_MAX_CONCURRENT_JOBS", "4").strip() or "4"
)

# Bounded retry. Retries apply only to genuinely transient failures (429 and
# 5xx); a 401, 403 or 404 is a configuration problem that cannot become true by
# asking again, so it is never retried. The attempt count is capped because an
# uncontrolled loop against a free tier is indistinguishable from an attack.
MODEL_B_MAX_ATTEMPTS = int(os.getenv("MODEL_B_MAX_ATTEMPTS", "3").strip() or "3")
MODEL_B_RETRY_BASE_DELAY_S = float(
    os.getenv("MODEL_B_RETRY_BASE_DELAY_S", "0.8").strip() or "0.8"
)
MODEL_B_RETRY_MAX_DELAY_S = float(
    os.getenv("MODEL_B_RETRY_MAX_DELAY_S", "5").strip() or "5"
)

# Optional OpenRouter attribution headers. Not credentials; used only for
# OpenRouter's public request dashboard. Configure in the backend environment.
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "TactileGeo").strip()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, fallback: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _int_env(name: str, fallback: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def model_b_settings():
    """Build `ModelBSettings` from the current environment.

    The environment is read on every call rather than captured at import time.
    Capturing it at import time made the function lie: a test or a script that
    set `OPENROUTER_API_KEY` after this module loaded silently got the value
    from whenever the import happened, which is a genuinely confusing way to
    debug a "my key is ignored" report. The module-level constants above remain
    as the import-time snapshot for anything that wants a stable value.

    Numeric variables go through `_float_env`/`_int_env` so a typo produces the
    documented default rather than a ValueError on the request path, where it
    would look like a Model B bug instead of a configuration bug.

    The `ModelBSettings` import is local to keep `app.core` free of
    feature-package dependencies.
    """
    from app.model_b.service import ModelBSettings

    return ModelBSettings(
        enabled=_truthy(os.getenv("MODEL_B_ENABLED", "false")) or MODEL_B_ENABLED,
        api_key=(os.getenv("OPENROUTER_API_KEY", "").strip() or None) or MODEL_B_API_KEY,
        model=(
            os.getenv("OPENROUTER_MODEL", "").strip()
            or DEFAULT_OPENROUTER_MODEL
        ),
        base_url=(
            os.getenv("OPENROUTER_BASE_URL", "").strip().rstrip("/")
            or OPENROUTER_BASE_URL
        ),
        timeout_s=_float_env("MODEL_B_TIMEOUT_S", MODEL_B_TIMEOUT_S),
        max_attempts=_int_env("MODEL_B_MAX_ATTEMPTS", MODEL_B_MAX_ATTEMPTS),
        retry_base_delay_s=_float_env(
            "MODEL_B_RETRY_BASE_DELAY_S", MODEL_B_RETRY_BASE_DELAY_S
        ),
        retry_max_delay_s=_float_env(
            "MODEL_B_RETRY_MAX_DELAY_S", MODEL_B_RETRY_MAX_DELAY_S
        ),
        site_url=os.getenv("OPENROUTER_SITE_URL", "").strip() or OPENROUTER_SITE_URL,
        app_title=(
            os.getenv("OPENROUTER_APP_TITLE", "").strip() or OPENROUTER_APP_TITLE
        ),
    )
