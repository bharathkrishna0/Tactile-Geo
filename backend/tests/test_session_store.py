from datetime import datetime, timedelta, timezone

from app.models.session import ConversionSession
from app.services.session_store import SessionStore


def _session(session_id, path, age_seconds, now):
    return ConversionSession(
        session_id=session_id,
        original_filename=path.name,
        original_image_path=path,
        created_at=datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(seconds=age_seconds),
    )


def test_expired_session_is_dropped_and_its_upload_deleted(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old_image = uploads / "old.png"
    new_image = uploads / "new.png"
    old_image.write_bytes(b"x")
    new_image.write_bytes(b"x")
    now = 1_800_000_000.0
    store = SessionStore(ttl_seconds=3600, upload_directory=uploads, clock=lambda: now)
    store.add(_session("old", old_image, 3601, now))
    store.add(_session("new", new_image, 10, now))

    assert store.get("old") is None
    assert not old_image.exists()
    assert store.get("new") is not None
    assert new_image.exists()


def test_expiry_never_deletes_files_outside_the_upload_directory(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    sample = tmp_path / "samples" / "demo.png"
    sample.parent.mkdir()
    sample.write_bytes(b"x")
    now = 1_800_000_000.0
    store = SessionStore(ttl_seconds=60, upload_directory=uploads, clock=lambda: now)
    store.add(_session("demo", sample, 120, now))

    assert store.prune() == 1
    assert sample.exists()


def test_zero_ttl_keeps_sessions(tmp_path):
    now = 1_800_000_000.0
    store = SessionStore(ttl_seconds=0, upload_directory=tmp_path, clock=lambda: now)
    store.add(_session("s", tmp_path / "a.png", 10**6, now))

    assert store.get("s") is not None
