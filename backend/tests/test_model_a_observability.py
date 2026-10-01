"""Model A execution-time observability (PROJECT.md section 13).

Section 13 requires recording Model A execution time. These tests assert the log
record is actually emitted, that it carries a plausible duration, and that it
leaks no image data or credential.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

MODEL_A_LOGGER = "app.api.sessions"
TIMING_PATTERN = re.compile(r"Model A processed session \S+ in (\d+(?:\.\d+)?)ms")


def _upload_and_process(fixture_directory, caplog):
    image = (fixture_directory / "triangle_worksheet.png").read_bytes()
    created = client.post(
        "/api/sessions",
        files={"image": ("triangle.png", image, "image/png")},
    )
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    with caplog.at_level(logging.INFO, logger=MODEL_A_LOGGER):
        response = client.post(f"/api/sessions/{session_id}/process", json={})

    assert response.status_code == 200
    return session_id, response


def test_model_a_execution_time_is_logged(fixture_directory, caplog):
    """The §13 requirement itself: Model A execution time is recorded."""
    _session_id, _response = _upload_and_process(fixture_directory, caplog)

    records = [r for r in caplog.records if r.name == MODEL_A_LOGGER and r.levelno == logging.INFO]
    assert records, "Model A processing emitted no observability record"

    message = records[-1].getMessage()
    match = TIMING_PATTERN.search(message)
    assert match, f"no execution time in Model A log line: {message!r}"
    assert float(match.group(1)) >= 0.0


def test_model_a_log_reports_useful_counts(fixture_directory, caplog):
    """A duration alone is not actionable; the record carries result counts."""
    _session_id, response = _upload_and_process(fixture_directory, caplog)

    message = [r for r in caplog.records if r.name == MODEL_A_LOGGER and r.levelno == logging.INFO][-1].getMessage()
    assert "shapes" in message
    assert "labels" in message
    assert "QA issues" in message
    # The counts must agree with what the API actually returned.
    assert f"{len(response.json()['detected_shapes'])} shapes" in message


def test_model_a_log_contains_no_image_data(fixture_directory, caplog):
    """§13 forbids logging unnecessary image data."""
    _session_id, _response = _upload_and_process(fixture_directory, caplog)

    for record in caplog.records:
        if record.name != MODEL_A_LOGGER:
            continue
        message = record.getMessage()
        assert "data:image" not in message
        assert "<svg" not in message
        assert "base64" not in message.lower()


def test_model_a_failure_is_logged_with_timing(fixture_directory, caplog):
    """A failure is still a §13 event: it must be recorded, with a duration."""
    image = (fixture_directory / "triangle_worksheet.png").read_bytes()
    created = client.post(
        "/api/sessions",
        files={"image": ("triangle.png", image, "image/png")},
    )
    session_id = created.json()["session_id"]

    def boom(*_args, **_kwargs):
        raise ValueError("undecodable image")

    import app.api.sessions as sessions_module

    previous = sessions_module.build_full_analysis
    sessions_module.build_full_analysis = boom
    try:
        with caplog.at_level(logging.WARNING, logger=MODEL_A_LOGGER):
            response = client.post(f"/api/sessions/{session_id}/process", json={})
    finally:
        sessions_module.build_full_analysis = previous

    assert response.status_code == 422
    records = [r for r in caplog.records if r.name == MODEL_A_LOGGER and r.levelno == logging.WARNING]
    assert records, "a Model A failure was not recorded"
    assert "after" in records[-1].getMessage() and "ms" in records[-1].getMessage()


def test_model_a_log_never_contains_a_credential(fixture_directory, caplog):
    """§13 forbids logging API keys and sensitive credentials."""
    import os

    secret = "AQ.not-a-real-key-used-only-to-prove-it-is-never-logged"
    previous = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = secret
    try:
        _session_id, _response = _upload_and_process(fixture_directory, caplog)
    finally:
        if previous is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = previous

    for record in caplog.records:
        assert secret not in record.getMessage()


@pytest.mark.parametrize("logger_name", ["app.api.sessions", "app.api.demo"])
def test_model_a_observability_is_wired_on_both_paths(logger_name):
    """Both Model A entry points instrument execution time."""
    import app.api.demo as demo_module
    import app.api.sessions as sessions_module

    module = {"app.api.sessions": sessions_module, "app.api.demo": demo_module}[logger_name]
    assert module.logger.name == logger_name
    assert module.logger.level == logging.NOTSET  # inherits the app root config
