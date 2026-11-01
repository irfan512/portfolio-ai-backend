from types import SimpleNamespace

from portfolio_ai.api.rate_limit import RateLimiter, client_key
from portfolio_ai.conversations.sessions import SESSION_ID_PATTERN, SessionStore
from portfolio_ai.rag.service import Turn


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_session_lifecycle_and_turn_cap() -> None:
    store = SessionStore(max_turns=2)
    session_id, history = store.get(None)
    assert SESSION_ID_PATTERN.match(session_id) and history == []
    for i in range(3):
        store.append(session_id, Turn(question=f"q{i}", answer="a"))
    _, history = store.get(session_id)
    assert [t.question for t in history] == ["q1", "q2"]


def test_sessions_expire_after_ttl() -> None:
    clock = Clock()
    store = SessionStore(ttl_s=60, clock=clock)
    session_id, _ = store.get(None)
    store.append(session_id, Turn(question="q", answer="a"))
    clock.now += 61
    new_id, history = store.get(session_id)
    assert new_id != session_id and history == []


def test_least_recently_used_sessions_are_evicted() -> None:
    store = SessionStore(max_sessions=2)
    first, _ = store.get(None)
    store.get(None)
    store.get(None)
    assert len(store) == 2
    assert store.get(first)[0] != first


def test_minute_and_daily_limits() -> None:
    clock = Clock()
    limiter = RateLimiter(per_minute=2, per_day=3, global_per_day=100, clock=clock)
    assert limiter.check("a").allowed and limiter.check("a").allowed
    blocked = limiter.check("a")
    assert not blocked.allowed and blocked.scope == "client_minute" and blocked.retry_after_s > 0
    assert limiter.check("b").allowed  # other clients unaffected
    clock.now += 61
    assert limiter.check("a").allowed
    clock.now += 61
    assert limiter.check("a").scope == "client_daily"


def test_global_limit_applies_across_clients() -> None:
    limiter = RateLimiter(per_minute=10, per_day=10, global_per_day=2, clock=Clock())
    assert limiter.check("a").allowed and limiter.check("b").allowed
    assert limiter.check("c").scope == "global_daily"


def request(host: str, forwarded: str | None = None):
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


def test_client_key_uses_trusted_hop_not_spoofable_left_entries() -> None:
    real = client_key(request("10.0.0.1", "1.1.1.1"), trusted_proxy_hops=1)
    spoofed = client_key(request("10.0.0.1", "6.6.6.6, 1.1.1.1"), trusted_proxy_hops=1)
    assert real == spoofed  # attacker-added left entries are ignored
    assert client_key(request("10.0.0.1", "1.1.1.1"), 0) == client_key(request("10.0.0.1"), 0)


def test_client_key_is_a_hash_not_the_ip() -> None:
    key = client_key(request("203.0.113.7"), 0)
    assert "203.0.113.7" not in key and len(key) == 32
