"""Session persistence.

``SessionStore`` is the process-local store used for development and tests.
``PostgresSessionStore`` (``postgres_store.py``) implements the same interface
durably. ``session_store`` is chosen from the environment at import time.

Sessions are mutable documents: callers change fields and then call ``save``.
Processing runs, teacher edits and audit events are append-only history.
"""
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.core.config import (
    DATABASE_URL,
    SESSION_TTL_SECONDS,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_URL,
    UPLOAD_DIRECTORY,
)
from app.models.session import ConversionSession
from app.services.object_storage import (
    LocalObjectStorage,
    ObjectStorage,
    ObjectStorageError,
    SupabaseObjectStorage,
)


class SessionRepository(Protocol):
    def add(self, session: ConversionSession) -> None: ...

    def get(self, session_id: str) -> ConversionSession | None: ...

    def save(self, session: ConversionSession) -> None: ...

    def prune(self) -> int: ...

    def record_run(self, session_id: str, kind: str, params: dict, summary: dict, duration_ms: int | None) -> None: ...

    def record_edits(self, session_id: str, edits: list[dict]) -> None: ...

    def record_event(self, session_id: str, event_type: str, payload: dict | None = None) -> None: ...

    def events(self, session_id: str) -> list[dict[str, Any]]: ...


class SessionStore:
    """Process-local session store.

    Sessions expire ``ttl_seconds`` after creation. Expired sessions are dropped
    and their uploaded image is deleted, so student worksheets do not pile up on
    disk. Images outside ``upload_directory`` (for example bundled demo samples)
    are never deleted.
    """

    def __init__(
        self,
        ttl_seconds: float = SESSION_TTL_SECONDS,
        upload_directory: Path = UPLOAD_DIRECTORY,
        clock: Callable[[], float] = time.time,
        storage: ObjectStorage | None = None,
    ) -> None:
        self._sessions: dict[str, ConversionSession] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._ttl_seconds = ttl_seconds
        self._upload_directory = upload_directory
        self._clock = clock
        self._storage = storage

    def add(self, session: ConversionSession) -> None:
        self.prune()
        self._sessions[session.session_id] = session
        self.record_event(session.session_id, "session_created", {"filename": session.original_filename})

    def get(self, session_id: str) -> ConversionSession | None:
        self.prune()
        return self._sessions.get(session_id)

    def save(self, session: ConversionSession) -> None:
        self._sessions[session.session_id] = session

    def record_run(self, session_id: str, kind: str, params: dict, summary: dict, duration_ms: int | None) -> None:
        self._append(session_id, {"table": "processing_runs", "kind": kind, "params": params, "summary": summary, "duration_ms": duration_ms})

    def record_edits(self, session_id: str, edits: list[dict]) -> None:
        for edit in edits:
            self._append(session_id, {"table": "teacher_edits", "element_id": edit.get("element_id"), "edit": edit})

    def record_event(self, session_id: str, event_type: str, payload: dict | None = None) -> None:
        self._append(session_id, {"table": "audit_events", "event_type": event_type, "payload": payload or {}})

    def events(self, session_id: str) -> list[dict[str, Any]]:
        return [entry for entry in self._history.get(session_id, []) if entry["table"] == "audit_events"]

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._history.get(session_id, []))

    def _append(self, session_id: str, entry: dict[str, Any]) -> None:
        self._history.setdefault(session_id, []).append(
            {**entry, "created_at": datetime.fromtimestamp(self._clock(), tz=timezone.utc)}
        )

    def prune(self) -> int:
        """Drop expired sessions and delete their uploads; return how many expired."""
        if self._ttl_seconds <= 0:
            return 0
        cutoff = datetime.fromtimestamp(self._clock() - self._ttl_seconds, tz=timezone.utc)
        expired = [sid for sid, session in self._sessions.items() if session.created_at < cutoff]
        for session_id in expired:
            session = self._sessions.pop(session_id)
            self._history.pop(session_id, None)
            if session.source_storage_key and self._storage is not None:
                self._storage.delete([session.source_storage_key])
            self._delete_upload(session.original_image_path)
        return len(expired)

    def _delete_upload(self, path: Path) -> None:
        try:
            if path.resolve().is_relative_to(self._upload_directory.resolve()):
                path.unlink(missing_ok=True)
        except OSError:
            pass


def build_object_storage() -> ObjectStorage:
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        return SupabaseObjectStorage(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_STORAGE_BUCKET)
    return LocalObjectStorage(UPLOAD_DIRECTORY)


def build_session_store(storage: ObjectStorage) -> SessionRepository:
    if DATABASE_URL:
        from app.services.postgres_store import PostgresSessionStore

        return PostgresSessionStore(DATABASE_URL, storage=storage)
    return SessionStore(storage=storage)


def read_source_image(session: ConversionSession, storage: ObjectStorage) -> bytes:
    """The uploaded image bytes; raises ``ObjectStorageError`` when it is gone."""
    if session.source_storage_key:
        return storage.get(session.source_storage_key)
    try:
        return session.original_image_path.read_bytes()
    except OSError as error:
        raise ObjectStorageError("Source image is not available.") from error


object_storage = build_object_storage()
session_store = build_session_store(object_storage)
