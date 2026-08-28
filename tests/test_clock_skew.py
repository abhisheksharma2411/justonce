"""Two hosts, two clocks, one effect.

The scenario from #47: host A claims a key with a fifteen-minute lease. Host B's
clock runs half an hour fast, so B considers A's claim long expired while A is
still running the effect. The claim is atomic, so only one row ever changes —
but if the lease is judged against each caller's own clock, *both* callers
believe they hold it, and the effect runs twice. Atomicity does not save you;
agreeing on the time does.

These drive the engine rather than the store, so what is asserted is the number
of times the side effect actually ran, which is the only thing the library
promises.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from justonce import Idempotent, OperationInFlightError, operation_key
from justonce.stores import MemoryStore, SqliteStore

TTL = 900.0          # fifteen minutes, the library default's order of magnitude
SKEW = 1800.0        # half an hour: far past NTP drift, ordinary after a VM resume


@contextmanager
def clock_ahead_by(seconds: float) -> Iterator[None]:
    """Move this process's `time.time()` forward for the duration of the block.

    Stands in for the second host. Patching the module attribute rather than
    injecting a clock is deliberate: a store must not be able to opt out of the
    skew, and one that took its time from a constructor argument could.
    """
    real = time.time
    try:
        time.time = lambda: real() + seconds
        yield
    finally:
        time.time = real


@pytest.fixture
def shared_db() -> Iterator[str]:
    """One database file, reachable by both 'hosts'."""
    path = os.path.join(tempfile.mkdtemp(), "justonce-skew.db")
    yield path


class Charger:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def charge(self) -> dict:
        with self.lock:
            self.calls += 1
            return {"charge_id": f"ch_{self.calls}"}


def test_a_fast_host_does_not_run_an_effect_a_slow_host_is_still_running(shared_db) -> None:
    """The whole of #47, end to end."""
    key = operation_key("charge", "order_1")
    charger = Charger()
    payload = {"amount": 500}

    host_a = Idempotent(SqliteStore(shared_db), ttl_seconds=TTL)
    host_b = Idempotent(SqliteStore(shared_db), ttl_seconds=TTL)

    # A takes the claim and is *still inside* the effect — modelled by claiming
    # through the store directly, which is the state the engine is in while the
    # effect runs. The fingerprint must match B's payload, or the divergence
    # guard rejects B first and the lease is never consulted at all.
    assert host_a.store.claim(key, _hash_of(payload), TTL).won

    # B's clock is half an hour ahead. A's lease has 15 minutes left in real
    # time, so B must not be able to take it.
    with clock_ahead_by(SKEW), pytest.raises(OperationInFlightError):
        host_b.run(key, charger.charge, payload=payload)

    assert charger.calls == 0, (
        f"the effect ran {charger.calls} time(s) while another host held a live "
        "claim — clock skew defeated the lease"
    )


def test_both_hosts_together_still_run_the_effect_once(shared_db) -> None:
    """A runs the effect normally; B, running fast, must replay rather than re-run."""
    key = operation_key("charge", "order_2")
    charger = Charger()
    payload = {"amount": 500}

    host_a = Idempotent(SqliteStore(shared_db), ttl_seconds=TTL)
    host_b = Idempotent(SqliteStore(shared_db), ttl_seconds=TTL)

    first = host_a.run(key, charger.charge, payload=payload)
    with clock_ahead_by(SKEW):
        second = host_b.run(key, charger.charge, payload=payload)

    assert charger.calls == 1
    assert first.executed is True
    assert second.executed is False
    assert second.value == first.value


def test_a_genuinely_expired_lease_is_still_reclaimable(shared_db) -> None:
    """The guard must not cost us the recovery it sits next to.

    Skew resistance is easy to get by never reclaiming at all, which would strand
    every key whose holder actually died. A real expiry must still be reclaimed.
    """
    key = operation_key("charge", "order_3")
    charger = Charger()
    payload = {"amount": 500}

    host_a = Idempotent(SqliteStore(shared_db), ttl_seconds=0.01)
    host_b = Idempotent(SqliteStore(shared_db), ttl_seconds=TTL)

    # Same fingerprint, so the reclaim's divergence guard permits it and what is
    # under test really is the expiry.
    assert host_a.store.claim(key, _hash_of(payload), 0.01).won
    time.sleep(0.05)

    result = host_b.run(key, charger.charge, payload=payload)
    assert result.executed is True
    assert charger.calls == 1


def test_the_sweeper_on_a_fast_host_does_not_delete_live_records(shared_db) -> None:
    """A swept record is a key the next delivery cannot find, so it re-runs.

    `sweep()` with no argument defers to the store's clock for exactly this
    reason — a fast sweeper would otherwise retire records still inside their
    retention window.
    """
    key = operation_key("charge", "order_4")
    charger = Charger()
    payload = {"amount": 500}

    engine = Idempotent(SqliteStore(shared_db), retention_seconds=TTL)
    engine.run(key, charger.charge, payload=payload)

    with clock_ahead_by(SKEW):
        deleted = engine.sweep()

    assert deleted == 0, "a fast host swept a record still inside its retention window"
    assert engine.run(key, charger.charge, payload=payload).executed is False
    assert charger.calls == 1


def test_a_single_process_store_is_exempt_and_says_so() -> None:
    """`MemoryStore` follows this process's clock, and cannot be reached by another.

    Asserted rather than skipped: if it ever becomes shareable, this fails and
    the exemption gets revisited instead of silently persisting.
    """
    store = MemoryStore()
    assert store.clock == "process"
    assert store.claim("k", "h", TTL).won

    with clock_ahead_by(SKEW):
        # It *does* follow the skew — harmless only because no second process
        # can reach this store to disagree with it.
        assert store.claim("k", "h", TTL).won is True


def _hash_of(payload: object) -> str:
    from justonce.keys import fingerprint

    return fingerprint(payload)
