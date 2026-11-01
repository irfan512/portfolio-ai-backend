"""In-memory conversation sessions.

Privacy: turns live only in this process's memory, expire after `ttl_s` of
inactivity, are capped in number, and are never written to disk or logs.
Single-instance by design; multiple instances would need a shared store (Redis).
"""

import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from portfolio_ai.rag.service import Turn

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


@dataclass
class _Session:
    turns: list[Turn] = field(default_factory=list)
    last_seen: float = 0.0


class SessionStore:
    def __init__(
        self,
        *,
        ttl_s: float = 1_800,
        max_turns: int = 6,
        max_sessions: int = 2_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_s
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> tuple[str, list[Turn]]:
        """Existing session's history, or a brand-new session (unknown/expired ids are not errors)."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_seen = now
                self._sessions.move_to_end(session_id)
                return session_id, list(session.turns)
            new_id = secrets.token_urlsafe(24)  # 192 bits, unguessable
            self._sessions[new_id] = _Session(last_seen=now)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)  # least recently used
            return new_id, []

    def append(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.turns.append(turn)
            del session.turns[: -self._max_turns]
            session.last_seen = self._clock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _expire(self, now: float) -> None:
        while self._sessions:
            oldest_id, oldest = next(iter(self._sessions.items()))
            if now - oldest.last_seen < self._ttl:
                break
            del self._sessions[oldest_id]
