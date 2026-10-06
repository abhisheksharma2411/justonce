"""Engine-level ledger queries (#27).

The store-level contract lives in `justonce.conformance` and runs against every
store. What is left here is the part only the engine can get wrong: namespacing,
and what happens on a store that does not support queries at all.
"""

from __future__ import annotations

import pytest

from justonce.core import Idempotent
from justonce.errors import UnsupportedByStore
from justonce.stores.base import State
from justonce.stores.memory import MemoryStore
from justonce.stores.sqlite import SqliteStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request):
    return MemoryStore() if request.param == "memory" else SqliteStore(":memory:")


def test_a_tenant_cannot_see_another_tenants_keys(store):
    """The whole point of the namespace, on the surface people read in an incident."""
    a = Idempotent(store=store, namespace="tenant-a")
    b = Idempotent(store=store, namespace="tenant-b")
    a.run("order_1", lambda: "a1")
    b.run("order_1", lambda: "b1")
    b.run("order_2", lambda: "b2")

    assert [r.key for r in a.query()] == ["order_1"]
    assert sorted(r.key for r in b.query()) == ["order_1", "order_2"]


def test_summary_is_scoped_to_the_namespace_too(store):
    """A count that ignored the namespace would size an incident with a neighbour's rows."""
    a = Idempotent(store=store, namespace="tenant-a")
    b = Idempotent(store=store, namespace="tenant-b")
    a.run("order_1", lambda: "a1")
    for key in ("order_1", "order_2", "order_3"):
        b.run(key, lambda: "b")

    assert a.summary()[State.SUCCEEDED] == 1
    assert b.summary()[State.SUCCEEDED] == 3


def test_keys_come_back_caller_facing(store):
    """A key from `query` must be usable in `run` without the caller stripping anything."""
    engine = Idempotent(store=store, namespace="tenant-a")
    engine.run("order_1", lambda: "a1")

    (record,) = engine.query()
    assert record.key == "order_1"
    assert engine.lookup(record.key) is not None


def test_an_engine_with_no_namespace_sees_every_tenant(store):
    """The operator's view — the same asymmetry `unresolved` already has."""
    Idempotent(store=store, namespace="tenant-a").run("order_1", lambda: "a")
    Idempotent(store=store, namespace="tenant-b").run("order_1", lambda: "b")

    operator = Idempotent(store=store)
    assert sorted(r.key for r in operator.query()) == ["tenant-a:order_1", "tenant-b:order_1"]


def test_namespaced_prefix_is_still_literal(store):
    """Namespacing must not reintroduce the wildcard it is prefixed onto."""
    engine = Idempotent(store=store, namespace="tenant-a")
    engine.run("order_1", lambda: "a")
    engine.run("orderX1", lambda: "b")

    assert [r.key for r in engine.query(prefix="order_")] == ["order_1"]


def test_a_store_without_query_support_raises_rather_than_returning_nothing(store):
    """An empty list reads as "nobody was affected". That must not be guessable."""

    class Bare:
        def now(self) -> float:
            return 0.0

    engine = Idempotent(store=Bare())
    with pytest.raises(UnsupportedByStore) as seen:
        engine.query()
    assert "Bare" in str(seen.value)

    with pytest.raises(UnsupportedByStore):
        engine.summary()


def test_summary_reports_every_state_even_from_a_partial_store(store):
    """A third-party store returning only non-empty states must not KeyError the caller."""

    class Partial:
        def now(self) -> float:
            return 0.0

        def count_by_state(self, **_: object) -> dict[State, int]:
            return {State.SUCCEEDED: 2}

    counts = Idempotent(store=Partial()).summary()
    assert counts == {
        State.IN_PROGRESS: 0,
        State.SUCCEEDED: 2,
        State.FAILED: 0,
        State.UNKNOWN: 0,
    }


def test_unknowns_are_findable_by_state(store):
    """The incident question: which keys are unresolved under this prefix?

    The UNKNOWN is produced the way it happens in production — the effect ran
    and the process died before the outcome could be written — rather than by
    calling `mark_unknown` directly, so the test exercises the path an operator
    would actually be querying about.
    """
    engine = Idempotent(store=store)
    charged = []

    def exploding_complete(*_: object, **__: object) -> None:
        raise RuntimeError("process died before the outcome was written")

    store.complete = exploding_complete  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        engine.run("charge:v1:order_1", lambda: charged.append("order_1"))
    del store.complete  # type: ignore[attr-defined]

    assert charged == ["order_1"], "the effect must have run for this to be UNKNOWN"
    engine.run("charge:v1:order_2", lambda: "ok")

    unknown = engine.query(prefix="charge:v1:", states=[State.UNKNOWN])
    assert [r.key for r in unknown] == ["charge:v1:order_1"]
    assert engine.summary(prefix="charge:v1:")[State.UNKNOWN] == 1
