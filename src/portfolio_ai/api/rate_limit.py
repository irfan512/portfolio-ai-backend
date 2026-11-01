"""Per-client and global request limits (in memory, single instance).

Client keys are salted hashes of the IP address: raw IPs are never stored.
The global daily cap bounds Gemini cost even if someone rotates IPs.
"""

import hashlib
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from starlette.requests import Request


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after_s: int = 0
    scope: str = ""


class RateLimiter:
    def __init__(
        self,
        *,
        per_minute: int,
        per_day: int,
        global_per_day: int,
        global_per_minute: int = 1_000_000,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._global_per_day = global_per_day
        self._global_per_minute = global_per_minute
        self._global_window: deque[float] = deque()
        self._max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        self._minute: OrderedDict[str, deque[float]] = OrderedDict()
        self._daily: dict[str, int] = {}
        self._day = ""
        self._global = 0

    def check(self, client_key: str) -> RateDecision:
        now = self._clock()
        today = datetime.fromtimestamp(now, UTC).date().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._daily, self._global = today, {}, 0
            if self._global >= self._global_per_day:
                return RateDecision(False, _seconds_until_midnight(now), "global_daily")
            while self._global_window and now - self._global_window[0] >= 60:
                self._global_window.popleft()
            if len(self._global_window) >= self._global_per_minute:
                retry = max(1, int(60 - (now - self._global_window[0])) + 1)
                return RateDecision(False, retry, "global_minute")
            if self._daily.get(client_key, 0) >= self._per_day:
                return RateDecision(False, _seconds_until_midnight(now), "client_daily")

            window = self._minute.setdefault(client_key, deque())
            self._minute.move_to_end(client_key)
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self._per_minute:
                return RateDecision(False, max(1, int(60 - (now - window[0])) + 1), "client_minute")

            window.append(now)
            self._daily[client_key] = self._daily.get(client_key, 0) + 1
            self._global += 1
            self._global_window.append(now)
            while len(self._minute) > self._max_clients:
                self._minute.popitem(last=False)
            return RateDecision(True)


def _seconds_until_midnight(now: float) -> int:
    return max(1, 86_400 - int(now) % 86_400)


_SALT = secrets.token_bytes(16)  # per process: hashes can't be linked across restarts


def client_key(request: Request, trusted_proxy_hops: int) -> str:
    """Identify a client for rate limiting.

    Behind N trusted proxies the real client is the Nth entry from the *right*
    of X-Forwarded-For (entries further left are client-controlled and spoofable).
    """
    ip = request.client.host if request.client else "unknown"
    if trusted_proxy_hops > 0:
        forwarded = [
            p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()
        ]
        if len(forwarded) >= trusted_proxy_hops:
            ip = forwarded[-trusted_proxy_hops]
    return hashlib.sha256(_SALT + ip.encode()).hexdigest()[:32]
