"""The database is down. Does the payment go through?

#49: today `claim()` raises `StoreError` and the request fails. That is the
right answer, but it was the right answer *by accident* — nothing stated it, so
nothing stopped it drifting. These tests pin it as a decision, and pin the
opt-out to being narrow enough to be safe to offer.

The dangerous version of the opt-out is a `try/except Exception` around the
claim. That would also swallow `KeyTooLongError` — a caller bug that means the
key is about to collide with a different intent — and turn the loudest signal
the library has into a silent unguarded run. So the tests below check not only
that fail-open runs the effect, but everything it must still refuse to run for.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce import (
    Idempotent,
    KeyTooLongError,
    OnStoreUnavailable,
    StoreError,
)
from justonce.asyncio import AsyncIdempotent
from justonce.stores import MemoryStore


class Effect:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "charged"

    async def acall(self) -> str:
        self.calls += 1
        return "charged"


class BrokenStore(MemoryStore):
    """A store that is up enough to be configured and down enough to be useless.

    Subclasses `MemoryStore` so every method beyond `claim` behaves normally —
    which is what makes the "fail-open must not write" assertions meaningful:
    a write that slipped through would succeed and be visible, not blow up.
    """

    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__()
        self.error = error or StoreError("connection refused")
        self.claim_attempts = 0
        self.writes: list[str] = []

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Any:
        self.claim_attempts += 1
        raise self.error

    def complete(self, key: str, response: Any, **kwargs: Any) -> None:
        self.writes.append("complete")
        super().complete(key, response, **kwargs)

    def fail(self, key: str, **kwargs: Any) -> None:
        self.writes.append("fail")
        super().fail(key, **kwargs)

    def mark_unknown(self, key: str) -> None:
        self.writes.append("mark_unknown")
        super().mark_unknown(key)


class TestFailClosedIsTheDefault:
    def test_store_error_reaches_the_caller(self) -> None:
        effect = Effect()
        engine = Idempotent(BrokenStore())

        with pytest.raises(StoreError):
            engine.run("k", effect)

        assert effect.calls == 0, "no claim, no effect"

    def test_default_is_fail_closed_by_name_not_by_omission(self) -> None:
        assert Idempotent(MemoryStore()).on_store_unavailable is OnStoreUnavailable.FAIL_CLOSED

    def test_fail_closed_stated_explicitly_behaves_the_same(self) -> None:
        effect = Effect()
        engine = Idempotent(
            BrokenStore(), on_store_unavailable=OnStoreUnavailable.FAIL_CLOSED
        )

        with pytest.raises(StoreError):
            engine.run("k", effect)

        assert effect.calls == 0


class TestFailOpen:
    def test_runs_the_effect_with_no_claim(self) -> None:
        effect = Effect()
        engine = Idempotent(
            BrokenStore(), on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )

        result = engine.run("k", effect)

        assert effect.calls == 1
        assert result.value == "charged"
        assert result.executed is True

    def test_result_says_it_was_unguarded(self) -> None:
        """The caller must be able to tell this run from a guarded one.

        Without this the only difference between "deduplicated correctly" and
        "ran with the dedup layer switched off" is a log line nobody reads.
        """
        engine = Idempotent(
            BrokenStore(), on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )

        result = engine.run("k", Effect())

        assert result.guarded is False
        assert result.record is None

    def test_a_guarded_run_says_so(self) -> None:
        engine = Idempotent(MemoryStore())
        assert engine.run("k", Effect()).guarded is True

    def test_a_replay_says_so(self) -> None:
        engine = Idempotent(MemoryStore())
        engine.run("k", Effect())
        assert engine.run("k", Effect()).guarded is True

    def test_writes_nothing_it_never_claimed(self) -> None:
        """No claim means no right to write the key.

        The store is down, so a write would most likely fail anyway — but "most
        likely" is not the argument. A partially-reachable store would accept
        the write, and a `complete` for a key this caller never won can
        overwrite the record of a holder that *did* win.
        """
        store = BrokenStore()
        engine = Idempotent(store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN)

        engine.run("k", Effect())

        assert store.writes == []
        assert store.lookup("k") is None

    def test_claim_is_attempted_before_giving_up_on_it(self) -> None:
        store = BrokenStore()
        engine = Idempotent(store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN)

        engine.run("k", Effect())

        assert store.claim_attempts == 1, "fail-open is a fallback, not a bypass"

    def test_a_raising_effect_still_raises(self) -> None:
        def boom() -> None:
            raise RuntimeError("gateway rejected")

        engine = Idempotent(
            BrokenStore(), on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )

        with pytest.raises(RuntimeError, match="gateway rejected"):
            engine.run("k", boom)


class TestFailOpenIsNarrow:
    """What fail-open must still refuse. These are the mutation targets."""

    def test_key_too_long_is_not_a_store_outage(self) -> None:
        """A key the column cannot hold is a collision waiting to happen.

        `check_key_length` raises this from inside `claim` specifically so it
        arrives at the engine on the same path a store failure does. Catching
        `Exception` there would run the effect for a key that is about to be
        truncated into another intent's key — the exact corruption the guard
        exists to prevent, now with the effect applied.
        """
        effect = Effect()
        store = BrokenStore(error=KeyTooLongError("k" * 300, 255, "test"))
        engine = Idempotent(store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN)

        with pytest.raises(KeyTooLongError):
            engine.run("k" * 300, effect)

        assert effect.calls == 0

    @pytest.mark.parametrize(
        "error",
        [ValueError("bad column"), RuntimeError("driver bug"), TypeError("nope")],
        ids=["value", "runtime", "type"],
    )
    def test_only_store_error_opens_the_gate(self, error: BaseException) -> None:
        """A store that raises something else has not said 'I am unavailable'.

        Every bundled store wraps its driver's failures in `StoreError`. An
        unwrapped exception escaping `claim` is a bug in that store, and a bug
        of unknown shape is not grounds for running an unguarded effect.
        """
        effect = Effect()
        engine = Idempotent(
            BrokenStore(error=error), on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )

        with pytest.raises(type(error)):
            engine.run("k", effect)

        assert effect.calls == 0

    def test_a_failure_after_the_claim_is_untouched(self) -> None:
        """Fail-open covers the claim only, never the outcome write.

        Once the claim is won the effect has run, and the question is no longer
        "may we run?" but "did we?". Leaving the key UNKNOWN is the honest
        answer and fail-open must not convert it into a silent success.
        """

        class CompleteFails(MemoryStore):
            def complete(self, key: str, response: Any, **kwargs: Any) -> None:
                raise StoreError("connection lost mid-write")

        store = CompleteFails()
        engine = Idempotent(store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN)
        effect = Effect()

        with pytest.raises(StoreError):
            engine.run("k", effect)

        assert effect.calls == 1
        record = store.lookup("k")
        assert record is not None and record.state.value == "unknown"

    def test_losing_the_claim_is_never_fail_open(self) -> None:
        """A lost claim is knowledge, not an outage.

        If the store answered "someone else holds this", running anyway is a
        duplicate we chose on purpose. Only a store that could not answer at
        all gets the benefit of the doubt.
        """
        store = MemoryStore()
        engine = Idempotent(store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN)
        engine.run("k", Effect())

        second = Effect()
        assert engine.run("k", second).deduplicated is True
        assert second.calls == 0


class TestAsyncParity:
    async def test_async_default_fails_closed(self) -> None:
        effect = Effect()
        engine = AsyncIdempotent(BrokenStore())

        with pytest.raises(StoreError):
            await engine.run("k", effect.acall)

        assert effect.calls == 0

    async def test_async_fail_open_runs_unguarded(self) -> None:
        effect = Effect()
        store = BrokenStore()
        engine = AsyncIdempotent(
            store, on_store_unavailable=OnStoreUnavailable.FAIL_OPEN
        )

        result = await engine.run("k", effect.acall)

        assert effect.calls == 1
        assert result.guarded is False
        assert store.writes == []

    async def test_async_key_too_long_is_not_an_outage(self) -> None:
        effect = Effect()
        engine = AsyncIdempotent(
            BrokenStore(error=KeyTooLongError("k" * 300, 255, "test")),
            on_store_unavailable=OnStoreUnavailable.FAIL_OPEN,
        )

        with pytest.raises(KeyTooLongError):
            await engine.run("k" * 300, effect.acall)

        assert effect.calls == 0
