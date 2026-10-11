import json
import os

import pytest

import desktop_main
from app.services import ocr

ENV_KEYS = ("UPLOAD_DIRECTORY", "LOUIS_TABLEPATH", "EASYOCR_MODEL_DIR", "CORS_ORIGINS", "TACTILEGEO_RESOURCES")


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_configure_environment_uses_per_user_dirs_and_bundled_assets(clean_env, tmp_path):
    resources = tmp_path / "resources"
    (resources / "liblouis" / "tables").mkdir(parents=True)
    (resources / "ocr-models").mkdir()
    data = tmp_path / "data"

    logs = desktop_main.configure_environment(data, resources)

    assert logs == data / "logs" and logs.is_dir()
    assert os.environ["UPLOAD_DIRECTORY"] == str(data / "uploads")
    assert (data / "uploads").is_dir()
    assert os.environ["LOUIS_TABLEPATH"] == str(resources / "liblouis" / "tables")
    assert os.environ["EASYOCR_MODEL_DIR"] == str(resources / "ocr-models")
    assert "http://tauri.localhost" in os.environ["CORS_ORIGINS"].split(",")


def test_configure_environment_leaves_asset_paths_unset_when_not_bundled(clean_env, tmp_path):
    desktop_main.configure_environment(tmp_path / "data", tmp_path / "missing")
    assert "LOUIS_TABLEPATH" not in os.environ
    assert "EASYOCR_MODEL_DIR" not in os.environ


def test_resource_root_override(clean_env, tmp_path):
    clean_env.setenv("TACTILEGEO_RESOURCES", str(tmp_path))
    assert desktop_main.resource_root() == tmp_path


def test_easyocr_downloads_disabled_only_with_bundled_models(clean_env, tmp_path):
    assert ocr._model_storage_options() == {}
    clean_env.setenv("EASYOCR_MODEL_DIR", str(tmp_path))
    assert ocr._model_storage_options() == {"model_storage_directory": str(tmp_path), "download_enabled": False}


def test_self_check_reports_missing_ocr_models(clean_env, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_main, "self_check", lambda: {"ocr_models": {"ok": False, "detail": "missing"}, "ok": False})
    report = tmp_path / "report.json"
    code = desktop_main.main(["--data-dir", str(tmp_path / "data"), "--self-check", str(report)])
    assert code == 1
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is False
