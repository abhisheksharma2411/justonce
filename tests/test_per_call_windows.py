"""Per-call TTL and retention (#26).

A card charge and a nightly batch job are not the same shape. The charge takes
seconds and must stay replayable past the dispute window; the batch takes an
hour and is meaningless a day later. One engine-wide pair of numbers has to be
the max of both, which makes every claim's lease far longer than it needs to be
and every record's retention far longer than anyone wants to store.

Two properties carry the weight here, and both are easy to lose in a refactor:

  * The resolved values must reach *every* store call on the path — the claim,
    the completion, and the terminal failure. Missing one leaves that write on
    the engine default, silently.
  * A per-call TTL must not become a way to shorten someone else's live lease.
    It does not, because reclaim compares the *stored* `expires_at` — written
    by the holder, with the holder's TTL — and that is worth a test rather than
    an assumption.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce import Idempotent, OperationInFlightError, idempotent, operation_key
from justonce.asyncio import AsyncIdempotent, async_idempotent
from justonce.keys import fingerprint
from justonce.stores import MemoryStore, SqliteStore

ENGINE_TTL = 900.0
ENGINE_RETENTION = 30 * 24 * 3600.0


class RecordingStore(MemoryStore):
    """A real store that also writes down the windows it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.claim_ttls: list[float] = []
        self.complete_retentions: list[float | None] = []
        self.fail_retentions: list[float | None] = []

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Any:
        self.claim_ttls.append(ttl_seconds)
        return super().claim(key, request_hash, ttl_seconds)

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        self.complete_retentions.append(retention_seconds)
        super().complete(key, response, retention_seconds=retention_seconds)

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        self.fail_retentions.append(retention_seconds)
        super().fail(key, terminal=terminal, retention_seconds=retention_seconds)


def engine(store: RecordingStore) -> Idempotent:
    return Idempotent(store, ttl_seconds=ENGINE_TTL, retention_seconds=ENGINE_RETENTION)


def async_engine(store: RecordingStore) -> AsyncIdempotent:
    return AsyncIdempotent(store, ttl_seconds=ENGINE_TTL, retention_seconds=ENGINE_RETENTION)


class TestOverridesReachTheStore:
    def test_ttl_override_is_the_lease_that_is_claimed(self) -> None:
        store = RecordingStore()
        engine(store).run("k", lambda: "ok", ttl_seconds=5.0)
        assert store.claim_ttls == [5.0]

    def test_retention_override_is_what_completion_records(self) -> None:
        store = RecordingStore()
        engine(store).run("k", lambda: "ok", retention_seconds=60.0)
        assert store.complete_retentions == [60.0]

    def test_retention_override_reaches_a_terminal_failure(self) -> None:
        """The failure path writes a record too, and it is the one most likely
        to be left on the engine default by a partial refactor."""
        store = RecordingStore()

        def boom() -> None:
            raise RuntimeError("declined")

        with pytest.raises(RuntimeError):
            engine(store).run("k", boom, retry_on_failure=False, retention_seconds=60.0)

        assert store.fail_retentions == [60.0]

    def test_retention_override_reaches_a_transient_failure(self) -> None:
        store = RecordingStore()

        def boom() -> None:
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            engine(store).run("k", boom, retry_on_failure=True, retention_seconds=60.0)

        assert store.fail_retentions == [60.0]


class TestDefaultsRemainTheFallback:
    def test_omitting_both_uses_the_engine(self) -> None:
        store = RecordingStore()
        engine(store).run("k", lambda: "ok")
        assert store.claim_ttls == [ENGINE_TTL]
        assert store.complete_retentions == [ENGINE_RETENTION]

    def test_none_means_inherit_not_unbounded(self) -> None:
        """`None` is 'not given'. The store reads `None` as 'keep forever', and
        letting that leak through would silently make a key immortal."""
        store = RecordingStore()
        engine(store).run("k", lambda: "ok", ttl_seconds=None, retention_seconds=None)
        assert store.claim_ttls == [ENGINE_TTL]
        assert store.complete_retentions == [ENGINE_RETENTION]

    def test_one_override_does_not_disturb_the_other(self) -> None:
        store = RecordingStore()
        engine(store).run("k", lambda: "ok", ttl_seconds=5.0)
        assert store.complete_retentions == [ENGINE_RETENTION]

    def test_an_override_does_not_stick_to_the_engine(self) -> None:
        store = RecordingStore()
        eng = engine(store)
        eng.run("a", lambda: "ok", ttl_seconds=5.0, retention_seconds=60.0)
        eng.run("b", lambda: "ok")
        assert store.claim_ttls == [5.0, ENGINE_TTL]
        assert store.complete_retentions == [60.0, ENGINE_RETENTION]
        assert eng.ttl_seconds == ENGINE_TTL
        assert eng.retention_seconds == ENGINE_RETENTION


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, -0.5], ids=["zero", "negative", "fractional"])
    def test_non_positive_ttl_is_refused(self, bad: float) -> None:
        """A lease that has already expired is not a lease.

        The engine constructor has refused this since day one; a per-call
        override that skipped the check would be a second, unguarded door to
        the same state.
        """
        store = RecordingStore()
        with pytest.raises(ValueError, match="ttl_seconds"):
            engine(store).run("k", lambda: "ok", ttl_seconds=bad)

    def test_negative_retention_is_refused(self) -> None:
        store = RecordingStore()
        with pytest.raises(ValueError, match="retention_seconds"):
            engine(store).run("k", lambda: "ok", retention_seconds=-1)

    def test_the_engine_refuses_the_same_values(self) -> None:
        """One predicate, not two that happen to agree."""
        with pytest.raises(ValueError, match="ttl_seconds"):
            Idempotent(MemoryStore(), ttl_seconds=0)
        with pytest.raises(ValueError, match="retention_seconds"):
            Idempotent(MemoryStore(), retention_seconds=-1)

    def test_validation_happens_before_the_claim(self) -> None:
        """Not after, and certainly not after the effect.

        Claiming and then rejecting would leave a live claim for a key whose
        call never ran — an in-flight row nobody will ever resolve.
        """
        store = RecordingStore()
        ran = []

        with pytest.raises(ValueError):
            engine(store).run("k", lambda: ran.append(1), ttl_seconds=-1)

        assert store.claim_ttls == []
        assert ran == []
        assert store.lookup("k") is None


class TestDecorator:
    def test_windows_pass_through(self) -> None:
        store = RecordingStore()

        @idempotent(
            key=lambda oid: operation_key("charge", oid),
            engine=engine(store),
            ttl_seconds=5.0,
            retention_seconds=60.0,
        )
        def charge(oid: str) -> str:
            return "ok"

        charge("o1")
        assert store.claim_ttls == [5.0]
        assert store.complete_retentions == [60.0]

    def test_decorator_without_windows_uses_the_engine(self) -> None:
        store = RecordingStore()

        @idempotent(key=lambda oid: operation_key("charge", oid), engine=engine(store))
        def charge(oid: str) -> str:
            return "ok"

        charge("o1")
        assert store.claim_ttls == [ENGINE_TTL]
        assert store.complete_retentions == [ENGINE_RETENTION]


class TestAsyncParity:
    async def test_overrides_reach_the_store(self) -> None:
        store = RecordingStore()

        async def effect() -> str:
            return "ok"

        await async_engine(store).run("k", effect, ttl_seconds=5.0, retention_seconds=60.0)
        assert store.claim_ttls == [5.0]
        assert store.complete_retentions == [60.0]

    async def test_defaults_remain_the_fallback(self) -> None:
        store = RecordingStore()

        async def effect() -> str:
            return "ok"

        await async_engine(store).run("k", effect)
        assert store.claim_ttls == [ENGINE_TTL]
        assert store.complete_retentions == [ENGINE_RETENTION]

    async def test_non_positive_ttl_is_refused(self) -> None:
        store = RecordingStore()

        async def effect() -> str:
            return "ok"

        with pytest.raises(ValueError, match="ttl_seconds"):
            await async_engine(store).run("k", effect, ttl_seconds=0)

        assert store.claim_ttls == []

    async def test_decorator_windows_pass_through(self) -> None:
        store = RecordingStore()

        @async_idempotent(
            key=lambda oid: operation_key("charge", oid),
            engine=async_engine(store),
            ttl_seconds=5.0,
            retention_seconds=60.0,
        )
        async def charge(oid: str) -> str:
            return "ok"

        await charge("o1")
        assert store.claim_ttls == [5.0]
        assert store.complete_retentions == [60.0]


class TestAShortTtlCannotStealALiveClaim:
    """The property that makes per-call TTLs safe to offer at all.

    Two call sites using different TTLs for one key is now easy to write by
    accident. If the *challenger's* TTL decided whether the holder's lease had
    expired, a caller passing `ttl_seconds=0.01` could reclaim a fifteen-minute
    claim that is still running, and the effect would apply twice — which is
    the whole failure this library exists to prevent.

    It cannot, because reclaim compares the `expires_at` the *holder* wrote.
    Driven through a real `SqliteStore` because that comparison lives in SQL.
    """

    def test_challenger_with_a_tiny_ttl_still_loses(self) -> None:
        store = SqliteStore(":memory:")
        long_lease = Idempotent(store, ttl_seconds=900.0)
        calls = []

        # The holder claims with the long lease and is still "running". Same
        # payload fingerprint the engine will derive, so the challenger loses
        # on the *lease*, not on the divergence guard — which is the question
        # being asked here.
        assert store.claim("k", fingerprint(None), 900.0).won is True

        with pytest.raises(OperationInFlightError):
            long_lease.run("k", lambda: calls.append("stolen"), ttl_seconds=0.001)

        assert calls == [], "a live claim was reclaimed by a shorter-TTL caller"

    def test_the_holders_own_short_ttl_does_expire(self) -> None:
        """The mirror image, so the test above is not passing for free."""
        import time

        store = SqliteStore(":memory:")
        eng = Idempotent(store)
        assert store.claim("k", fingerprint(None), 0.01).won is True
        time.sleep(0.05)

        calls = []
        eng.run("k", lambda: calls.append("ran"), ttl_seconds=900.0)
        assert calls == ["ran"]
