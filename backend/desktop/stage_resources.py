"""Stage the read-only runtime assets the packaged backend needs.

Produces, under --out:
  liblouis/tables/          Liblouis translation tables (incl. en-ueb-g2.ctb)
  liblouis/liblouis.dll     the vendored native library
  louis-binding/louis/      the official Python binding, generated for that DLL
  ocr-models/               EasyOCR detector and English recogniser weights

Exits non-zero if anything required is missing, so the build stops early.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
NATIVE = BACKEND / ".native"
LIBLOUIS_DLL = "liblouis.dll"
REQUIRED_TABLE = "en-ueb-g2.ctb"
OCR_MODELS = {"craft_mlt_25k.pth": 50_000_000, "english_g2.pth": 10_000_000}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def find_binding_template() -> Path:
    matches = sorted((NATIVE / "source").glob("liblouis-*/python/louis/__init__.py.in"))
    if not matches:
        fail("Liblouis Python binding template not found under backend/.native/source")
    return matches[-1]


def stage_liblouis(out: Path) -> None:
    win64 = NATIVE / "win64"
    dll = win64 / "bin" / LIBLOUIS_DLL
    tables = win64 / "share" / "liblouis" / "tables"
    if not dll.is_file():
        fail(f"missing {dll}")
    if not (tables / REQUIRED_TABLE).is_file():
        fail(f"missing {tables / REQUIRED_TABLE}")
    target = out / "liblouis"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(tables, target / "tables")
    shutil.copy2(dll, target / LIBLOUIS_DLL)

    binding = out / "louis-binding" / "louis"
    binding.mkdir(parents=True, exist_ok=True)
    source = find_binding_template().read_text(encoding="utf-8")
    placeholder = "###LIBLOUIS_SONAME###"
    if placeholder not in source:
        fail("unexpected binding template: SONAME placeholder not found")
    (binding / "__init__.py").write_text(source.replace(placeholder, LIBLOUIS_DLL), encoding="utf-8")
    print(f"Liblouis: {LIBLOUIS_DLL}, {sum(1 for _ in (target / 'tables').iterdir())} tables, binding from {find_binding_template().parent.parent.parent.name}")


def stage_ocr_models(out: Path) -> None:
    import easyocr

    target = out / "ocr-models"
    target.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(["en"], gpu=False, model_storage_directory=str(target), download_enabled=True, verbose=False)
    for name, minimum in OCR_MODELS.items():
        path = target / name
        if not path.is_file() or path.stat().st_size < minimum:
            fail(f"EasyOCR model {name} missing or truncated in {target}")
        print(f"OCR model: {name} ({path.stat().st_size / 1e6:.1f} MB)")
    for extra in target.iterdir():
        if extra.name not in OCR_MODELS:
            if extra.is_dir():
                shutil.rmtree(extra)
            else:
                extra.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-ocr", action="store_true", help="stage Liblouis only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stage_liblouis(args.out)
    if not args.skip_ocr:
        stage_ocr_models(args.out)


if __name__ == "__main__":
    main()
