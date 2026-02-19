"""In-memory session store keyed by session_id. No persistence."""


class SessionStore:
    def __init__(self) -> None:
        self.sessions: dict = {}

    def create(self, session_id: str, engine: object) -> None:
        self.sessions[session_id] = engine

    def get(self, session_id: str):
        return self.sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]


session_store = SessionStore()
