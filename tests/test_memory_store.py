"""The in-memory store, and the ways a test double can lie.

Conformance already proves it satisfies the contract, concurrency test
included. What conformance cannot prove is the property that actually matters
for a double: that a handler passing its tests against this store will behave
the same way against a real one. A double that is *more* permissive than
production turns a green test suite into a false negative, which is worse than
having no double at all.
"""

from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from justonce import Idempotent, operation_key
from justonce.stores import MemoryStore, SqliteStore

TTL = 60.0


# -- parity with a real store ----------------------------------------------


def _both():
    return MemoryStore(), SqliteStore(":memory:")


def _comparable(record):
    """Everything but the clocks, which legitimately differ between stores."""
    if record is None:
        return None
    return (record.key, record.state, record.request_hash, record.response, record.attempts)


@pytest.mark.parametrize(
    "response",
    [
        {"charge_id": "ch_1", "amount": 250},
        "done",          # a bare string: decoded wrong, this comes back as `done` vs `"done"`
        [1, 2, 3],
        None,
        0,
        False,
        {"nested": {"list": [1, {"deep": True}]}},
    ],
)
def test_a_recorded_response_reads_back_the_same_as_from_sqlite(response) -> None:
    memory, sqlite = _both()
    for store in (memory, sqlite):
        store.claim("k", "h", TTL)
        store.complete("k", response)

    assert _comparable(memory.lookup("k")) == _comparable(sqlite.lookup("k"))
    assert memory.lookup("k").response == response  # type: ignore[union-attr]


def test_the_two_stores_agree_across_the_whole_lifecycle() -> None:
    """Replay, divergence, terminal failure, release, reclaim, unknown, sweep."""
    memory, sqlite = _both()
    trace = {}

    for name, store in (("memory", memory), ("sqlite", sqlite)):
        steps = []
        steps.append(("claim", store.claim("k", "h", TTL).won))
        steps.append(("second claim", store.claim("k", "h", TTL).won))
        steps.append(("diverging claim", store.claim("k", "other", TTL).won))
        store.complete("k", {"charge": 1}, retention_seconds=TTL)
        steps.append(("after complete", _comparable(store.lookup("k"))))
        steps.append(("claim a terminal key", store.claim("k", "h", TTL).won))

        steps.append(("claim b", store.claim("b", "h", TTL).won))
        store.fail("b", terminal=False)  # transient: releases the claim
        steps.append(("after release", _comparable(store.lookup("b"))))
        steps.append(("reclaim after release", store.claim("b", "h", TTL).won))
        store.fail("b", terminal=True, retention_seconds=TTL)
        steps.append(("after terminal failure", _comparable(store.lookup("b"))))

        steps.append(("claim c", store.claim("c", "h", TTL).won))
        store.mark_unknown("c")
        steps.append(("after unknown", _comparable(store.lookup("c"))))
        steps.append(("unresolved", [r.key for r in store.unresolved()]))

        # Far-future cutoff: terminal records go, UNKNOWN must not.
        steps.append(("swept", store.sweep(before=2 ** 40)))
        steps.append(("survivors", sorted(r.key for r in store.unresolved())))
        trace[name] = steps

    assert trace["memory"] == trace["sqlite"], (
        f"memory and sqlite diverge:\n  memory={trace['memory']}\n  sqlite={trace['sqlite']}"
    )


def test_an_expired_lease_is_reclaimable_by_the_same_request_only() -> None:
    memory, sqlite = _both()
    for store in (memory, sqlite):
        store.claim("k", "h", 0.01)
    import time

    time.sleep(0.05)

    assert memory.claim("k", "other", TTL).won is sqlite.claim("k", "other", TTL).won is False
    assert memory.claim("k", "h", TTL).won is sqlite.claim("k", "h", TTL).won is True
    assert _comparable(memory.lookup("k")) == _comparable(sqlite.lookup("k"))


# -- the ways a permissive double would lie --------------------------------


def test_an_unserialisable_response_is_refused_here_too() -> None:
    """Storing the object as handed in would accept what production rejects.

    A `datetime` in a response passes straight through a dict and fails against
    `SqliteStore` and `PostgresStore`. Encoding here moves that failure into the
    test, where it costs nothing.
    """
    memory, sqlite = _both()
    for store in (memory, sqlite):
        store.claim("k", "h", TTL)

    with pytest.raises(TypeError):
        sqlite.complete("k", {"at": dt.datetime(2026, 1, 1)})
    with pytest.raises(TypeError):
        memory.complete("k", {"at": dt.datetime(2026, 1, 1)})


def test_mutating_the_response_afterwards_does_not_rewrite_history() -> None:
    """A store handing back a live reference lets a test pass by editing it."""
    store = MemoryStore()
    response = {"charge_id": "ch_1", "items": [1, 2]}
    store.claim("k", "h", TTL)
    store.complete("k", response)

    response["charge_id"] = "ch_TAMPERED"
    response["items"].append(3)

    recorded = store.lookup("k")
    assert recorded is not None
    assert recorded.response == {"charge_id": "ch_1", "items": [1, 2]}

    # And the record handed out is detached from the store's own copy too.
    recorded.response["charge_id"] = "ch_ALSO_TAMPERED"
    assert store.lookup("k").response["charge_id"] == "ch_1"  # type: ignore[union-attr]


def test_two_instances_share_nothing() -> None:
    """The limitation, pinned. Two workers each hold a private view.

    This is why the docstring says tests only: both callers win the claim and
    both run the effect, which is the failure the library exists to prevent.
    """
    a, b = MemoryStore(), MemoryStore()
    assert a.claim("k", "h", TTL).won is True
    assert b.claim("k", "h", TTL).won is True
    assert a.lookup("k") is not None
    assert b.lookup("k") is not None


# -- it has to work under the engine, not just under the contract ----------


def test_the_engine_runs_an_effect_once_against_it() -> None:
    engine = Idempotent(MemoryStore())
    charges = []
    key = operation_key("charge", "order_1")

    first = engine.run(key, lambda: charges.append("x") or {"id": 1}, payload={"amount": 5})
    second = engine.run(key, lambda: charges.append("x") or {"id": 2}, payload={"amount": 5})

    assert charges == ["x"]
    assert first.executed is True
    assert second.executed is False
    assert second.value == first.value


def test_concurrent_callers_through_the_engine_run_the_effect_once() -> None:
    """The store is shared, so the guarantee must hold — unlike across processes."""
    engine = Idempotent(MemoryStore())
    lock = threading.Lock()
    charges = []

    def effect():
        with lock:
            charges.append(1)
        return {"n": len(charges)}

    def attempt(_):
        try:
            return engine.run("k", effect, payload={"a": 1}).executed
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        executed = list(pool.map(attempt, range(16)))

    assert sum(executed) == 1
    assert charges == [1]


def test_unresolved_is_ordered_oldest_first() -> None:
    """Reconciliation's work list. Out of order, the oldest unknown starves."""
    store = MemoryStore()
    for key in ("c", "a", "b"):
        store.claim(key, "h", TTL)
        store.mark_unknown(key)

    assert [r.key for r in store.unresolved()] == ["c", "a", "b"]
    assert [r.key for r in store.unresolved(limit=2)] == ["c", "a"]


def test_operations_on_a_missing_key_are_no_ops_not_errors() -> None:
    """Matching the SQL stores, whose UPDATE simply matches no rows."""
    memory, sqlite = _both()
    for store in (memory, sqlite):
        store.complete("nope", {"a": 1})
        store.fail("nope", terminal=True)
        store.fail("nope", terminal=False)
        store.mark_unknown("nope")
        assert store.lookup("nope") is None
        assert store.sweep(before=2 ** 40) == 0
