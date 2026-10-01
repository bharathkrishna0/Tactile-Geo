import os
from pathlib import Path
from typing import Protocol

from app.core.config import LOUIS_TABLEPATH


UEB_GRADE_2_TABLE = "en-ueb-g2.ctb"

# Where this file lives: backend/app/services/braille.py
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Common locations where Liblouis table files ship on various installs.
# Used only when LOUIS_TABLEPATH is not explicitly configured.
_DEFAULT_TABLEPATHS = (
    Path("/usr/share/liblouis/tables"),          # Debian/Ubuntu (apt), Docker
    Path("/usr/local/share/liblouis/tables"),    # Liblouis built from source
    Path("/usr/share/liblouis"),                 # some distro layouts
    # Windows: the native build vendored into this repo under backend/.native.
    # Resolved relative to the package so no machine-specific path is hardcoded.
    _BACKEND_ROOT / ".native" / "win64" / "share" / "liblouis" / "tables",
    _BACKEND_ROOT / ".native" / "win64" / "share" / "liblouis",
)


def _bundled_source_tablepath() -> Path | None:
    """Locate tables inside the vendored liblouis *source* tree, if present."""
    source_root = _BACKEND_ROOT / ".native" / "source"
    if not source_root.is_dir():
        return None
    candidates = sorted(source_root.glob("liblouis-*/tables"))
    return candidates[-1] if candidates else None


def _loaded_library_tablepath() -> Path | None:
    """Locate tables shipped alongside the native liblouis binary in use.

    Some liblouis builds (notably 3.39 on Windows) ignore LOUIS_TABLEPATH and
    resolve tables relative to the running executable instead, so the search path
    cannot be relied on. Distributions place the tables at
    ``<prefix>/share/liblouis/tables`` next to the loaded binary, so derive the
    candidate from the binary itself instead of hardcoding a machine-specific path.
    """
    try:
        import louis
    except ImportError:
        return None
    library = getattr(louis, "liblouis", None)
    library_path = getattr(library, "_name", None)
    if not library_path:
        return None
    bin_dir = Path(library_path).resolve().parent
    for relative in (Path("..") / "share" / "liblouis" / "tables", Path("..") / "share" / "liblouis"):
        candidate = (bin_dir / relative).resolve()
        if (candidate / UEB_GRADE_2_TABLE).is_file():
            return candidate
    return None


def resolve_table_file(table: str = UEB_GRADE_2_TABLE) -> str | None:
    """Return an absolute path to a Liblouis table file, or None if not found.

    Passing an explicit file path to ``translateString`` removes the dependency on
    liblouis honouring LOUIS_TABLEPATH, which not every build does.
    """
    candidates: list[Path] = []
    tablepath = resolve_tablepath()
    if tablepath:
        candidates.append(Path(tablepath) / table)
    for root in (*_DEFAULT_TABLEPATHS, _bundled_source_tablepath(), _loaded_library_tablepath()):
        if root is not None:
            candidates.append(root / table)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


class LouisBindings(Protocol):
    def translateString(self, tables: list[str], text: str): ...


# Liblouis emits North American Braille ASCII by default: each printable ASCII
# character stands for one six-dot cell (",a" is capital-indicator + a). The
# character's code is NOT the dot pattern, so cells are looked up in the standard
# Braille ASCII table, indexed by dot bitmask (dot 1 = bit 0 ... dot 6 = bit 5).
BRAILLE_ASCII = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="
UNICODE_BRAILLE_BASE = 0x2800
_BRAILLE_ASCII_TO_DOTS = {char: dots for dots, char in enumerate(BRAILLE_ASCII)}
# Liblouis writes the 0x40-0x5F cells in lowercase (0x60-0x7F), e.g. "~" for "^".
_BRAILLE_ASCII_TO_DOTS.update({
    chr(ord(char) + 0x20): dots for char, dots in list(_BRAILLE_ASCII_TO_DOTS.items()) if 0x40 <= ord(char) <= 0x5F
})


def to_unicode_braille(text: str) -> str:
    """Convert Liblouis Braille-ASCII output into Unicode braille cells.

    Characters that are already in the U+2800 block are left alone, so this is
    safe to apply to output that has been converted upstream.
    """
    cells: list[str] = []
    for char in text:
        if UNICODE_BRAILLE_BASE <= ord(char) <= UNICODE_BRAILLE_BASE + 0xFF:
            cells.append(char)
            continue
        dots = _BRAILLE_ASCII_TO_DOTS.get(char)
        if dots is None:
            raise ValueError(f"Liblouis returned {char!r}, which is not a Braille ASCII cell.")
        cells.append(chr(UNICODE_BRAILLE_BASE + dots))
    return "".join(cells)


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
    bundled = _bundled_source_tablepath()
    if bundled is not None and (bundled / UEB_GRADE_2_TABLE).is_file():
        return str(bundled)
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
        # Prefer an absolute path to the table file: liblouis resolves bare table
        # names through a search path that several builds (including the 3.39
        # Windows build) ignore, which made every real translation fail with
        # "Can't translate: tables ['en-ueb-g2.ctb']".
        table = resolve_table_file(self.table) or self.table
        translated = bindings.translateString([table], text)
        # Liblouis versions expose either a string or a tuple whose first item is the translation.
        result = translated[0] if isinstance(translated, tuple) else translated
        return to_unicode_braille(result)

    def _load_bindings(self) -> LouisBindings:
        configure_tablepath()
        try:
            import louis
        except ImportError as error:
            raise RuntimeError(
                "Liblouis Python bindings are unavailable. Install the native Liblouis library and its official 'louis' bindings."
            ) from error
        return louis
