"""Entry point for the packaged desktop backend (tactilegeo-backend.exe).

The desktop shell starts this process with a free port and a per-user data
directory. Everything mutable (uploads, logs) goes under that directory; the
install directory is treated as read-only. The server only ever binds to
127.0.0.1.

``--self-check`` imports the full Model A stack, translates a Braille sample and
loads the OCR models, then writes a JSON report. The release build fails when
any required runtime asset is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

DESKTOP_ORIGINS = "http://tauri.localhost,https://tauri.localhost,tauri://localhost"
logger = logging.getLogger("tactilegeo.desktop")


def resource_root() -> Path:
    """Directory holding bundled read-only assets (PyInstaller's _MEIPASS when frozen)."""
    override = os.environ.get("TACTILEGEO_RESOURCES", "").strip()
    if override:
        return Path(override)
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent / "desktop" / "resources"


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "org.tactilegeo.desktop"


def configure_environment(data_dir: Path, resources: Path) -> Path:
    """Set process-wide paths before ``app`` is imported (its config is read at import)."""
    uploads = data_dir / "uploads"
    logs = data_dir / "logs"
    uploads.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    os.environ["UPLOAD_DIRECTORY"] = str(uploads)
    os.environ.setdefault("CORS_ORIGINS", DESKTOP_ORIGINS)
    tables = resources / "liblouis" / "tables"
    if tables.is_dir():
        os.environ["LOUIS_TABLEPATH"] = str(tables)
    models = resources / "ocr-models"
    if models.is_dir():
        os.environ["EASYOCR_MODEL_DIR"] = str(models)
    if hasattr(os, "add_dll_directory") and resources.is_dir():
        os.add_dll_directory(str(resources))
    return logs


def configure_logging(logs: Path) -> None:
    handler = logging.handlers.RotatingFileHandler(
        logs / "backend.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handlers: list[logging.Handler] = [handler]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def self_check() -> dict[str, object]:
    """Exercise every bundled runtime dependency; each entry is 'ok' or an error string."""
    results: dict[str, object] = {}

    def run(name: str, check) -> None:
        started = time.perf_counter()
        try:
            detail = check()
            results[name] = {"ok": True, "detail": detail, "ms": round((time.perf_counter() - started) * 1000)}
        except Exception as error:  # noqa: BLE001 - the report must capture every failure
            results[name] = {"ok": False, "detail": f"{type(error).__name__}: {error}"}

    def app_import() -> str:
        from app.main import app

        return app.title

    def braille() -> str:
        from app.services.braille import LouisBrailleTranslator

        translated = LouisBrailleTranslator().translate("triangle ABC")
        if not translated.strip():
            raise RuntimeError("empty translation")
        return translated

    def ocr_models() -> str:
        model_dir = os.environ.get("EASYOCR_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("EASYOCR_MODEL_DIR is not set; OCR models are not bundled")
        from app.services.ocr import _get_reader

        _get_reader(["en"])
        return model_dir

    def model_a() -> str:
        import cv2
        import numpy as np

        from app.services.pipeline import build_full_analysis

        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (100, 380), (540, 380), (0, 0, 0), 4)
        cv2.line(canvas, (100, 380), (100, 100), (0, 0, 0), 4)
        cv2.line(canvas, (100, 100), (540, 380), (0, 0, 0), 4)
        cv2.putText(canvas, "A", (70, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("could not encode test image")
        result = build_full_analysis(encoded.tobytes())
        if not result.tactile_svg or "<svg" not in result.tactile_svg:
            raise RuntimeError("no tactile SVG produced")
        return f"{len(result.shapes)} shapes, {len(result.labels)} labels"

    run("app_import", app_import)
    run("braille", braille)
    run("ocr_models", ocr_models)
    run("model_a", model_a)
    results["ok"] = all(isinstance(v, dict) and v.get("ok") for v in results.values())
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TactileGeo desktop backend")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--self-check", type=Path, metavar="REPORT_JSON", help="write a runtime asset report and exit")
    args = parser.parse_args(argv)

    resources = resource_root()
    logs = configure_environment(args.data_dir, resources)
    configure_logging(logs)
    logger.info("TactileGeo backend starting (frozen=%s, resources=%s)", getattr(sys, "frozen", False), resources)

    if args.self_check:
        report = self_check()
        args.self_check.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Self-check %s", "passed" if report["ok"] else "FAILED")
        return 0 if report["ok"] else 1

    import uvicorn

    from app.main import app

    logger.info("Serving on http://127.0.0.1:%d", args.port)
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_config=None, access_log=False)
    except Exception:
        logger.exception("Backend crashed")
        return 1
    logger.info("Backend stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
