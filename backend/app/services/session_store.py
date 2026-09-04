from app.models.session import ConversionSession

class SessionStore:
    """Milestone 1 store; replace with Postgres/object storage in deployment."""
    def __init__(self) -> None:
        self._sessions: dict[str, ConversionSession] = {}
    def add(self, session: ConversionSession) -> None:
        self._sessions[session.session_id] = session
    def get(self, session_id: str) -> ConversionSession | None:
        return self._sessions.get(session_id)

session_store = SessionStore()
