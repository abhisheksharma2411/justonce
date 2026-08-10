"""The properties this library exists to provide.

Each test maps to one of the verification checks in the idempotency skill this
project came out of. If any of these can be deleted without a failure, the
guarantee it names is not actually defended.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from justonce import (
    Idempotent,
    KeyReuseError,
    OnInFlight,
    OperationInFlightError,
    State,
    idempotent,
    operation_key,
)
from justonce.stores import SqliteStore


@pytest.fixture
def engine() -> Idempotent:
    return Idempotent(SqliteStore(":memory:"))


class Charger:
    """Stand-in for a payment provider. Counts real side effects."""

    def __init__(self) -> None:
        self.charges: list[str] = []
        self.lock = threading.Lock()

    def charge(self, order_id: str) -> dict:
        with self.lock:
            self.charges.append(order_id)
        return {"charge_id": f"ch_{len(self.charges)}", "order": order_id}


# 1. Replay -----------------------------------------------------------------

def test_replay_applies_the_effect_once(engine: Idempotent) -> None:
    charger = Charger()
    key = operation_key("charge", "order_1")

    first = engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
    second = engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})

    assert charger.charges == ["order_1"], "the effect ran more than once"
    assert first.executed is True
    assert second.executed is False
    assert second.value == first.value, "the replay did not return the recorded response"


def test_retry_storm_applies_the_effect_once(engine: Idempotent) -> None:
    charger = Charger()
    key = operation_key("charge", "order_1")
    for _ in range(50):
        engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
    assert len(charger.charges) == 1


# 2. Concurrency ------------------------------------------------------------

def test_concurrent_callers_apply_the_effect_once() -> None:
    """The case a check-then-act implementation silently fails."""
    engine = Idempotent(SqliteStore(":memory:"))
    charger = Charger()
    key = operation_key("charge", "order_1")
    workers = 24

    def attempt(_: int) -> str:
        try:
            engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
            return "ok"
        except OperationInFlightError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(attempt, range(workers)))

    assert len(charger.charges) == 1, f"charged {len(charger.charges)} times"
    assert outcomes.count("ok") >= 1
    assert outcomes.count("ok") + outcomes.count("conflict") == workers


# 3. Crash between effect and outcome ---------------------------------------

def test_crash_after_effect_leaves_the_key_unresolved() -> None:
    """The reason this library exists.

    The charge succeeded; recording it did not. The key must NOT be released —
    releasing it would let a retry charge the customer a second time.
    """
    store = SqliteStore(":memory:")
    engine = Idempotent(store)
    charger = Charger()
    key = operation_key("charge", "order_1")

    def exploding_complete(*_: object, **__: object) -> None:
        raise RuntimeError("process died before the outcome was written")

    store.complete = exploding_complete  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})

    assert charger.charges == ["order_1"]
    assert store.lookup(key).state is State.UNKNOWN

    # A later attempt must refuse rather than risk a duplicate.
    with pytest.raises(OperationInFlightError):
        engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
    assert charger.charges == ["order_1"], "retried an effect with an unknown outcome"


def test_unresolved_records_are_reconciliation_input() -> None:
    store = SqliteStore(":memory:")
    engine = Idempotent(store)
    store.claim("charge:v1:o1", "h", 60)
    store.mark_unknown("charge:v1:o1")
    assert [r.key for r in engine.unresolved()] == ["charge:v1:o1"]


# 4. Divergence -------------------------------------------------------------

def test_same_key_different_payload_fails_loudly(engine: Idempotent) -> None:
    """Never serve one request's response to a different request."""
    charger = Charger()
    key = operation_key("charge", "order_1")
    engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})

    with pytest.raises(KeyReuseError):
        engine.run(key, lambda: charger.charge("order_1"), payload={"amount": 9999})
    assert len(charger.charges) == 1


def test_payload_ordering_does_not_look_like_divergence(engine: Idempotent) -> None:
    charger = Charger()
    key = operation_key("charge", "order_1")
    engine.run(key, lambda: charger.charge("o"), payload={"a": 1, "b": 2})
    result = engine.run(key, lambda: charger.charge("o"), payload={"b": 2, "a": 1})
    assert result.deduplicated


# 5. Failure handling -------------------------------------------------------

def test_transient_failure_allows_a_later_retry(engine: Idempotent) -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("provider timeout")
        return "ok"

    key = operation_key("charge", "order_1")
    with pytest.raises(ConnectionError):
        engine.run(key, flaky, payload={"amount": 1})
    assert engine.run(key, flaky, payload={"amount": 1}).value == "ok"
    assert calls["n"] == 2


def test_terminal_failure_burns_the_key(engine: Idempotent) -> None:
    def rejected() -> str:
        raise ValueError("card declined")

    key = operation_key("charge", "order_1")
    with pytest.raises(ValueError):
        engine.run(key, rejected, payload={"amount": 1}, retry_on_failure=False)

    result = engine.run(key, rejected, payload={"amount": 1}, retry_on_failure=False)
    assert result.deduplicated, "a terminally failed key was retried"


# 6. In-flight policy -------------------------------------------------------

def test_wait_policy_returns_the_winner_result() -> None:
    store = SqliteStore(":memory:")
    engine = Idempotent(store, on_in_flight=OnInFlight.WAIT, wait_timeout=5.0)
    charger = Charger()
    key = operation_key("charge", "order_1")
    started = threading.Event()

    def slow() -> dict:
        started.set()
        time.sleep(0.3)
        return charger.charge("order_1")

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(engine.run, key, slow, payload={"amount": 1})
        started.wait(timeout=2)
        loser = pool.submit(engine.run, key, lambda: charger.charge("order_1"),
                            payload={"amount": 1})
        won, waited = winner.result(timeout=10), loser.result(timeout=10)

    assert len(charger.charges) == 1
    assert won.executed is True
    assert waited.executed is False
    assert waited.value == won.value


# 7. Retention --------------------------------------------------------------

def test_sweep_removes_only_expired_terminal_records() -> None:
    store = SqliteStore(":memory:")
    engine = Idempotent(store, ttl_seconds=0.01)
    engine.run(operation_key("charge", "old"), lambda: "x", payload={})
    time.sleep(0.05)
    assert engine.sweep() == 1
    assert store.lookup(operation_key("charge", "old")) is None


def test_sweep_never_removes_unresolved_records() -> None:
    store = SqliteStore(":memory:")
    engine = Idempotent(store, ttl_seconds=0.01)
    store.claim("k", "h", 0.01)
    store.mark_unknown("k")
    time.sleep(0.05)
    engine.sweep()
    assert store.lookup("k") is not None, "swept an unresolved outcome"


# 8. The decorator ----------------------------------------------------------

def test_decorator_applies_the_effect_once() -> None:
    engine = Idempotent(SqliteStore(":memory:"))
    charger = Charger()

    @idempotent(key=lambda order_id, amount: operation_key("charge", order_id),
                engine=engine)
    def charge(order_id: str, amount: int) -> dict:
        return charger.charge(order_id)

    a = charge("order_1", 500)
    b = charge("order_1", 500)
    assert charger.charges == ["order_1"]
    assert a == b


def test_decorator_detects_key_reuse() -> None:
    engine = Idempotent(SqliteStore(":memory:"))

    @idempotent(key=lambda order_id, amount: operation_key("charge", order_id),
                engine=engine)
    def charge(order_id: str, amount: int) -> str:
        return "ok"

    charge("order_1", 500)
    with pytest.raises(KeyReuseError):
        charge("order_1", 9999)


def test_decorator_can_report_whether_it_executed() -> None:
    engine = Idempotent(SqliteStore(":memory:"))

    @idempotent(key=lambda n: operation_key("op", n), engine=engine, return_result=True)
    def work(n: int) -> int:
        return n * 2

    assert work(21).executed is True
    assert work(21).deduplicated is True
