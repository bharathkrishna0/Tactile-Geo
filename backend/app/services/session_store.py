import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import SESSION_TTL_SECONDS, UPLOAD_DIRECTORY
from app.models.session import ConversionSession


class SessionStore:
    """Process-local session store; replace with Postgres/object storage in deployment.

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
    ) -> None:
        self._sessions: dict[str, ConversionSession] = {}
        self._ttl_seconds = ttl_seconds
        self._upload_directory = upload_directory
        self._clock = clock

    def add(self, session: ConversionSession) -> None:
        self.prune()
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> ConversionSession | None:
        self.prune()
        return self._sessions.get(session_id)

    def prune(self) -> int:
        """Drop expired sessions and delete their uploads; return how many expired."""
        if self._ttl_seconds <= 0:
            return 0
        cutoff = datetime.fromtimestamp(self._clock() - self._ttl_seconds, tz=timezone.utc)
        expired = [sid for sid, session in self._sessions.items() if session.created_at < cutoff]
        for session_id in expired:
            self._delete_upload(self._sessions.pop(session_id).original_image_path)
        return len(expired)

    def _delete_upload(self, path: Path) -> None:
        try:
            if path.resolve().is_relative_to(self._upload_directory.resolve()):
                path.unlink(missing_ok=True)
        except OSError:
            pass


session_store = SessionStore()
