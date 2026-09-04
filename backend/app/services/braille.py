import os
from pathlib import Path
from typing import Protocol

from app.core.config import LOUIS_TABLEPATH


UEB_GRADE_2_TABLE = "en-ueb-g2.ctb"

# Common locations where Liblouis table files ship on various installs.
# Used only when LOUIS_TABLEPATH is not explicitly configured.
_DEFAULT_TABLEPATHS = (
    Path("/usr/share/liblouis/tables"),          # Debian/Ubuntu (apt), Docker
    Path("/usr/local/share/liblouis/tables"),    # Liblouis built from source
    Path("/usr/share/liblouis"),                 # some distro layouts
)


class LouisBindings(Protocol):
    def translateString(self, tables: list[str], text: str): ...


def resolve_tablepath() -> str | None:
    """Return the Liblouis table path to use, or None if none can be found.

    Priority:
      1. Explicitly configured LOUIS_TABLEPATH (env or app config).
      2. First default location that actually contains the UEB Grade 2 table.
    """
    env_value = os.environ.get("LOUIS_TABLEPATH", "").strip()
    if env_value:
        return env_value
    if LOUIS_TABLEPATH:
        return LOUIS_TABLEPATH
    for candidate in _DEFAULT_TABLEPATHS:
        if (candidate / UEB_GRADE_2_TABLE).is_file():
            return str(candidate)
    return None


def configure_tablepath() -> None:
    """Make sure the LOUIS_TABLEPATH env var is set before the louis module loads.

    Liblouis reads LOUIS_TABLEPATH once at import time, so it must be present
    before ``import louis``. This mutates only the process environment.
    """
    if "LOUIS_TABLEPATH" in os.environ:
        return
    resolved = resolve_tablepath()
    if resolved:
        os.environ["LOUIS_TABLEPATH"] = resolved


class LouisBrailleTranslator:
    """UEB translator backed by the official Liblouis Python bindings."""
    def __init__(self, bindings: LouisBindings | None = None, table: str = UEB_GRADE_2_TABLE) -> None:
        self._bindings = bindings
        self.table = table

    def translate(self, text: str) -> str:
        bindings = self._bindings or self._load_bindings()
        translated = bindings.translateString([self.table], text)
        # Liblouis versions expose either a string or a tuple whose first item is the translation.
        return translated[0] if isinstance(translated, tuple) else translated

    def _load_bindings(self) -> LouisBindings:
        configure_tablepath()
        try:
            import louis
        except ImportError as error:
            raise RuntimeError(
                "Liblouis Python bindings are unavailable. Install the native Liblouis library and its official 'louis' bindings."
            ) from error
        return louis
