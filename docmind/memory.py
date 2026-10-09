"""Per-session chat history (in memory; swap for Redis/SQL history in production)."""
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

MAX_MESSAGES = 20  # keep the last 10 turns so the prompt doesn't grow forever


class SessionMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}

    def get(self, session_id: str) -> BaseChatMessageHistory:
        history = self._sessions.setdefault(session_id, InMemoryChatMessageHistory())
        if len(history.messages) > MAX_MESSAGES:
            history.messages = history.messages[-MAX_MESSAGES:]
        return history

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
