"""Async engine — same guarantees as the sync one, under an event loop.

The sync suite proves the state machine. These tests prove the async engine
does not weaken it, and that the two async-specific hazards are handled:
concurrent coroutines racing one key, and never blocking the loop while waiting.
"""

from __future__ import annotations

import asyncio

import pytest

from justonce import KeyReuseError, OnInFlight, OperationInFlightError, State, operation_key
from justonce.asyncio import (
    AsyncIdempotent,
    ThreadedStore,
    async_idempotent,
    configure_async,
)
from justonce.stores import SqliteStore

pytestmark = pytest.mark.asyncio


class AsyncCharger:
    def __init__(self) -> None:
        self.charges: list[str] = []

    async def charge(self, order_id: str) -> dict:
        await asyncio.sleep(0)  # a real await point, as a real client would have
        self.charges.append(order_id)
        return {"charge_id": f"ch_{len(self.charges)}", "order": order_id}


def engine(**kwargs) -> AsyncIdempotent:
    return AsyncIdempotent(SqliteStore(":memory:"), **kwargs)


# -- parity with the sync engine -------------------------------------------

async def test_replay_applies_the_effect_once() -> None:
    eng, charger = engine(), AsyncCharger()
    key = operation_key("charge", "order_1")

    first = await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
    second = await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})

    assert charger.charges == ["order_1"]
    assert first.executed is True
    assert second.executed is False
    assert second.value == first.value


async def test_same_key_different_payload_fails_loudly() -> None:
    eng, charger = engine(), AsyncCharger()
    key = operation_key("charge", "order_1")
    await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})

    with pytest.raises(KeyReuseError):
        await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 9999})
    assert len(charger.charges) == 1


async def test_transient_failure_allows_a_later_retry() -> None:
    eng = engine()
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("provider timeout")
        return "ok"

    key = operation_key("charge", "order_1")
    with pytest.raises(ConnectionError):
        await eng.run(key, flaky, payload={"amount": 1})
    assert (await eng.run(key, flaky, payload={"amount": 1})).value == "ok"
    assert calls["n"] == 2


# -- the async-specific hazards --------------------------------------------

async def test_concurrent_coroutines_apply_the_effect_once() -> None:
    """Many coroutines racing one key on a single event loop."""
    eng, charger = engine(), AsyncCharger()
    key = operation_key("charge", "order_1")

    async def attempt() -> str:
        try:
            await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 500})
            return "ok"
        except OperationInFlightError:
            return "conflict"

    outcomes = await asyncio.gather(*(attempt() for _ in range(24)))

    assert len(charger.charges) == 1, f"charged {len(charger.charges)} times"
    assert outcomes.count("ok") >= 1
    assert outcomes.count("ok") + outcomes.count("conflict") == 24


async def test_wait_policy_does_not_block_the_event_loop() -> None:
    """A waiting caller must not stall unrelated work on the same loop.

    If `_wait_for` used time.sleep, the ticker below would stop advancing while
    the loser waits — which is the bug this asserts against.
    """
    eng, charger = engine(on_in_flight=OnInFlight.WAIT, wait_timeout=5.0), AsyncCharger()
    key = operation_key("charge", "order_1")
    ticks = {"n": 0}
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    async def slow() -> dict:
        await asyncio.sleep(0.3)
        return await charger.charge("order_1")

    tick_task = asyncio.create_task(ticker())
    winner = asyncio.create_task(eng.run(key, slow, payload={"amount": 1}))
    await asyncio.sleep(0.05)
    loser = asyncio.create_task(
        eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 1})
    )
    w, r = await asyncio.gather(winner, loser)
    stop.set()
    await tick_task

    assert len(charger.charges) == 1
    assert w.executed is True
    assert r.executed is False
    assert r.value == w.value
    assert ticks["n"] > 5, "the event loop was blocked while waiting"


async def test_cancelled_effect_is_treated_as_a_failure() -> None:
    """A cancelled effect may have applied; the key must not silently succeed."""
    store = SqliteStore(":memory:")
    eng = AsyncIdempotent(store)
    key = operation_key("charge", "order_1")

    async def hangs() -> str:
        await asyncio.sleep(10)
        return "never"

    task = asyncio.create_task(eng.run(key, hangs, payload={"amount": 1}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # retry_on_failure defaults to True, so the claim is released for a retry.
    assert store.lookup(key) is None


async def test_crash_after_effect_leaves_the_key_unresolved() -> None:
    store = SqliteStore(":memory:")
    eng = AsyncIdempotent(store)
    charger = AsyncCharger()
    key = operation_key("charge", "order_1")

    def exploding_complete(*_: object, **__: object) -> None:
        raise RuntimeError("died before the outcome was written")

    store.complete = exploding_complete  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 1})

    assert charger.charges == ["order_1"]
    assert store.lookup(key).state is State.UNKNOWN

    with pytest.raises(OperationInFlightError):
        await eng.run(key, lambda: charger.charge("order_1"), payload={"amount": 1})
    assert charger.charges == ["order_1"]


# -- the decorator ----------------------------------------------------------

async def test_decorator_applies_the_effect_once() -> None:
    eng, charger = engine(), AsyncCharger()

    @async_idempotent(key=lambda oid: operation_key("charge", oid), engine=eng)
    async def charge(oid: str) -> dict:
        return await charger.charge(oid)

    a, b = await charge("order_1"), await charge("order_1")
    assert charger.charges == ["order_1"]
    assert a == b


async def test_decorator_rejects_a_sync_function() -> None:
    """Silently accepting one would return a coroutine nobody awaits."""
    with pytest.raises(TypeError, match="async def"):

        @async_idempotent(key=lambda n: operation_key("op", n), engine=engine())
        def not_async(n: int) -> int:
            return n


async def test_configure_async_sets_the_default_engine() -> None:
    configure_async(SqliteStore(":memory:"))
    charger = AsyncCharger()

    @async_idempotent(key=lambda oid: operation_key("charge", oid))
    async def charge(oid: str) -> dict:
        return await charger.charge(oid)

    await charge("order_9")
    await charge("order_9")
    assert len(charger.charges) == 1


# -- ThreadedStore ----------------------------------------------------------

async def test_threaded_store_wraps_a_sync_store() -> None:
    inner = SqliteStore(":memory:")
    eng = AsyncIdempotent(inner)
    assert isinstance(eng.store, ThreadedStore)
    assert eng.store.inner is inner


async def test_native_async_store_is_not_wrapped() -> None:
    """A store that is already async must be used directly, not re-wrapped."""

    class Native:
        async def claim(self, key, request_hash, ttl_seconds): ...
        async def complete(self, key, response): ...
        async def fail(self, key, *, terminal): ...
        async def mark_unknown(self, key): ...
        async def lookup(self, key): ...
        async def sweep(self, *, before): ...
        async def unresolved(self, *, older_than=None, limit=100): ...

    native = Native()
    assert AsyncIdempotent(native).store is native
