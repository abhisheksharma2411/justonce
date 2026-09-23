"""Observability hooks (#24).

The library knows things operators need and keeps them to itself: how often a
duplicate was actually suppressed, how many outcomes are unresolved, how old
the oldest one is. None of it is reachable without reading the store directly.

Two properties carry this feature, and both are the kind that a refactor
silently breaks:

  * **A hook must never change the outcome.** A metrics backend that is down
    must not fail a payment. Every callback is invoked defensively, and the
    tests below assert that a hook raising leaves the effect, the record and
    the return value exactly as they were.
  * **A hook must not fire before the thing it reports is durable.** Emitting
    `effect_finished` before `complete()` lands would report a success that a
    crash one line later makes untrue — the metric would say the effect was
    recorded when the ledger says UNKNOWN.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce import (
    Hooks,
    Idempotent,
    KeyReuseError,
    OnStoreUnavailable,
    OperationInFlightError,
    StoreError,
)
from justonce.asyncio import AsyncIdempotent
from justonce.stores import MemoryStore


class Recorder(Hooks):
    """Records every callback in order, so ordering can be asserted too."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def duplicate_suppressed(self, key: str, record: Any) -> None:
        self.events.append(("duplicate_suppressed", key))

    def claim_conflict(self, key: str) -> None:
        self.events.append(("claim_conflict", key))

    def key_reuse(self, key: str) -> None:
        self.events.append(("key_reuse", key))

    def unknown_recorded(self, key: str) -> None:
        self.events.append(("unknown_recorded", key))

    def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
        self.events.append(("effect_finished", (key, ok)))

    def ran_unguarded(self, key: str) -> None:
        self.events.append(("ran_unguarded", key))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class Exploding(Hooks):
    """Every callback raises. Stands in for a metrics backend that is down."""

    def duplicate_suppressed(self, key: str, record: Any) -> None:
        raise RuntimeError("metrics backend down")

    def claim_conflict(self, key: str) -> None:
        raise RuntimeError("metrics backend down")

    def key_reuse(self, key: str) -> None:
        raise RuntimeError("metrics backend down")

    def unknown_recorded(self, key: str) -> None:
        raise RuntimeError("metrics backend down")

    def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
        raise RuntimeError("metrics backend down")

    def ran_unguarded(self, key: str) -> None:
        raise RuntimeError("metrics backend down")


class TestEventsFire:
    def test_effect_finished_on_success(self) -> None:
        hooks = Recorder()
        Idempotent(MemoryStore(), hooks=hooks).run("k", lambda: "ok")
        assert hooks.events == [("effect_finished", ("k", True))]

    def test_effect_finished_reports_failure_too(self) -> None:
        hooks = Recorder()
        engine = Idempotent(MemoryStore(), hooks=hooks)

        def boom() -> None:
            raise RuntimeError("declined")

        with pytest.raises(RuntimeError):
            engine.run("k", boom)

        assert hooks.events == [("effect_finished", ("k", False))]

    def test_duplicate_suppressed_on_replay(self) -> None:
        hooks = Recorder()
        engine = Idempotent(MemoryStore(), hooks=hooks)
        engine.run("k", lambda: "ok")
        engine.run("k", lambda: "ok")
        assert hooks.names() == ["effect_finished", "claim_conflict", "duplicate_suppressed"]

    def test_key_reuse_is_reported(self) -> None:
        hooks = Recorder()
        engine = Idempotent(MemoryStore(), hooks=hooks)
        engine.run("k", lambda: "ok", payload={"a": 1})

        with pytest.raises(KeyReuseError):
            engine.run("k", lambda: "ok", payload={"a": 2})

        assert "key_reuse" in hooks.names()

    def test_unknown_recorded_when_the_outcome_write_fails(self) -> None:
        class CompleteFails(MemoryStore):
            def complete(self, key: str, response: Any, **kw: Any) -> None:
                raise StoreError("lost mid-write")

        hooks = Recorder()
        with pytest.raises(StoreError):
            Idempotent(CompleteFails(), hooks=hooks).run("k", lambda: "ok")

        assert "unknown_recorded" in hooks.names()

    def test_in_flight_rejection_is_a_claim_conflict(self) -> None:
        store = MemoryStore()
        store.claim("k", "wrong-hash", 60.0)
        hooks = Recorder()

        with pytest.raises((OperationInFlightError, KeyReuseError)):
            Idempotent(store, hooks=hooks).run("k", lambda: "ok")

        assert hooks.names()[0] == "claim_conflict"

    def test_ran_unguarded_fires_on_the_fail_open_path(self) -> None:
        """#64 gave callers `Result.guarded`; this is the metric behind it."""

        class Broken(MemoryStore):
            def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Any:
                raise StoreError("down")

        hooks = Recorder()
        engine = Idempotent(
            Broken(), hooks=hooks, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )
        result = engine.run("k", lambda: "ok")

        assert result.guarded is False
        assert "ran_unguarded" in hooks.names()


class TestAHookNeverChangesTheOutcome:
    """The property that makes it safe to call user code from the effect path."""

    def test_a_raising_hook_does_not_fail_the_effect(self) -> None:
        calls = []
        engine = Idempotent(MemoryStore(), hooks=Exploding())

        result = engine.run("k", lambda: calls.append(1) or "ok")

        assert calls == [1]
        assert result.value == "ok"
        assert result.executed is True

    def test_a_raising_hook_does_not_corrupt_the_record(self) -> None:
        store = MemoryStore()
        Idempotent(store, hooks=Exploding()).run("k", lambda: "ok")
        record = store.lookup("k")
        assert record is not None and record.state.value == "succeeded"

    def test_a_raising_hook_does_not_swallow_a_real_error(self) -> None:
        engine = Idempotent(MemoryStore(), hooks=Exploding())

        def boom() -> None:
            raise ValueError("the effect's own failure")

        with pytest.raises(ValueError, match="the effect's own failure"):
            engine.run("k", boom)

    def test_a_raising_hook_does_not_mask_key_reuse(self) -> None:
        engine = Idempotent(MemoryStore(), hooks=Exploding())
        engine.run("k", lambda: "ok", payload={"a": 1})

        with pytest.raises(KeyReuseError):
            engine.run("k", lambda: "ok", payload={"a": 2})

    def test_no_hooks_configured_is_the_default(self) -> None:
        engine = Idempotent(MemoryStore())
        assert engine.run("k", lambda: "ok").value == "ok"


class TestOrderingAgainstDurability:
    def test_effect_finished_fires_after_the_outcome_is_recorded(self) -> None:
        """Not before. A metric that says 'recorded' while the ledger says
        UNKNOWN is worse than no metric — it is a metric that lies in the
        direction of reassurance."""
        seen: list[str] = []

        class Watching(MemoryStore):
            def complete(self, key: str, response: Any, **kw: Any) -> None:
                seen.append("complete")
                super().complete(key, response, **kw)

        class H(Hooks):
            def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
                seen.append("effect_finished")

        Idempotent(Watching(), hooks=H()).run("k", lambda: "ok")
        assert seen == ["complete", "effect_finished"]

    def test_duration_is_measured_and_non_negative(self) -> None:
        seen: list[float] = []

        class H(Hooks):
            def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
                seen.append(duration_seconds)

        Idempotent(MemoryStore(), hooks=H()).run("k", lambda: "ok")
        assert len(seen) == 1 and seen[0] >= 0.0


class TestOldestUnresolvedAge:
    """The gauge the issue calls 'the one to alert on'."""

    def test_none_when_nothing_is_unresolved(self) -> None:
        assert Idempotent(MemoryStore()).oldest_unresolved_age() is None

    def test_reports_an_age_once_something_is_unresolved(self) -> None:
        class CompleteFails(MemoryStore):
            def complete(self, key: str, response: Any, **kw: Any) -> None:
                raise StoreError("lost mid-write")

        engine = Idempotent(CompleteFails())
        with pytest.raises(StoreError):
            engine.run("k", lambda: "ok")

        age = engine.oldest_unresolved_age()
        assert age is not None and age >= 0.0

    def test_age_is_measured_on_the_stores_clock(self) -> None:
        """Not the caller's. A gauge computed against a fast host's clock
        reports an age that never happened, and this one is alerted on."""
        store = MemoryStore()

        class CompleteFails(type(store)):  # type: ignore[misc]
            def complete(self, key: str, response: Any, **kw: Any) -> None:
                raise StoreError("lost mid-write")

        s = CompleteFails()
        engine = Idempotent(s)
        with pytest.raises(StoreError):
            engine.run("k", lambda: "ok")

        seen: list[str] = []
        real_now = s.now

        def spy() -> float:
            seen.append("store.now")
            return real_now()

        s.now = spy  # type: ignore[method-assign]
        engine.oldest_unresolved_age()
        assert seen == ["store.now"], "the age must come from the store's clock"


class TestAsyncParity:
    async def test_events_fire_on_the_async_engine(self) -> None:
        hooks = Recorder()

        async def effect() -> str:
            return "ok"

        await AsyncIdempotent(MemoryStore(), hooks=hooks).run("k", effect)
        assert hooks.events == [("effect_finished", ("k", True))]

    async def test_a_raising_hook_does_not_fail_the_async_effect(self) -> None:
        async def effect() -> str:
            return "ok"

        result = await AsyncIdempotent(MemoryStore(), hooks=Exploding()).run("k", effect)
        assert result.value == "ok"

    async def test_async_duplicate_suppressed(self) -> None:
        hooks = Recorder()

        async def effect() -> str:
            return "ok"

        engine = AsyncIdempotent(MemoryStore(), hooks=hooks)
        await engine.run("k", effect)
        await engine.run("k", effect)
        assert "duplicate_suppressed" in hooks.names()
