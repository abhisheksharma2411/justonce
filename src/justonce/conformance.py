"""Executable definition of the store contract.

Adding a store? Subclass `StoreConformanceTests`, implement `make_store`, and
run it. If it passes, your store is correct by this project's definition — and
that is all a reviewer needs to check.

    from justonce.conformance import StoreConformanceTests

    class TestMyStore(StoreConformanceTests):
        def make_store(self):
            return MyStore(dsn=os.environ["MY_DSN"])

The concurrency test is the one that matters. A store that passes everything
else and fails that one is not a store — it is a cache with extra steps.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from .errors import KeyTooLongError
from .stores.base import Record, State, Store

TTL = 60.0


def _found(record: Record | None) -> Record:
    """Narrow an Optional lookup, failing the test if the record is missing."""
    assert record is not None, "expected a record for this key, store returned None"
    return record


@contextmanager
def _process_clock_ahead_by(seconds: float) -> Iterator[None]:
    """Run the block with this process's `time.time()` moved forward.

    Patches the module attribute the stores would read, so a store that takes
    its timestamps from the caller follows the skew and one that asks its
    database does not. Deliberately global and deliberately narrow: the point is
    that a store cannot opt out of it, which a clock injected through the
    constructor would let it do.
    """
    real = time.time
    try:
        time.time = lambda: real() + seconds
        yield
    finally:
        time.time = real


def _long_key(store: Store) -> str:
    """The longest key `store` claims to hold, or 300 characters if unbounded.

    300 is past MySQL's `VARCHAR(255)` and comfortably inside the Postgres
    btree index limit, so it exercises the interesting case without tripping a
    different one.
    """
    limit = getattr(store, "max_key_length", None)
    width = 300 if limit is None else int(limit)
    stem = "charge:v1:"
    return stem + "a" * (width - len(stem))


class StoreConformanceTests:
    """Contract tests every store must pass. Framework-agnostic; use with pytest."""

    def make_store(self) -> Store:  # pragma: no cover - overridden
        raise NotImplementedError("conformance subclasses must implement make_store()")

    # -- the property everything else rests on ------------------------------

    def test_only_one_concurrent_claimer_wins(self) -> None:
        """N threads claim one key; exactly one may win.

        This is the whole contract. If a store fails here, every guarantee the
        library makes above it is void.
        """
        store = self.make_store()
        workers = 24

        def attempt(_: int) -> bool:
            return store.claim("race", "hash", TTL).won

        with ThreadPoolExecutor(max_workers=workers) as pool:
            wins = list(pool.map(attempt, range(workers)))

        assert sum(wins) == 1, f"expected exactly 1 winner, got {sum(wins)}"

    def test_claim_is_not_check_then_act(self) -> None:
        """Repeated sequential claims never hand out a second win."""
        store = self.make_store()
        assert store.claim("k", "h", TTL).won is True
        for _ in range(5):
            assert store.claim("k", "h", TTL).won is False

    # -- lifecycle ----------------------------------------------------------

    def test_winner_sees_in_progress(self) -> None:
        store = self.make_store()
        claim = store.claim("k", "h", TTL)
        assert claim.won
        assert _found(store.lookup("k")).state is State.IN_PROGRESS

    def test_complete_records_response(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.complete("k", {"charge_id": "ch_1"})
        record = _found(store.lookup("k"))
        assert record.state is State.SUCCEEDED
        assert record.response == {"charge_id": "ch_1"}

    def test_loser_can_read_the_recorded_response(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.complete("k", {"charge_id": "ch_1"})
        claim = store.claim("k", "h", TTL)
        assert claim.lost
        assert _found(claim.record).response == {"charge_id": "ch_1"}

    def test_terminal_failure_burns_the_key(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.fail("k", terminal=True)
        assert _found(store.lookup("k")).state is State.FAILED
        assert store.claim("k", "h", TTL).lost

    def test_transient_failure_releases_the_key(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.fail("k", terminal=False)
        assert store.claim("k", "h", TTL).won is True

    # -- the state that reconciliation exists for ---------------------------

    def test_unknown_is_preserved(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.mark_unknown("k")
        assert _found(store.lookup("k")).state is State.UNKNOWN

    def test_unknown_is_never_swept(self) -> None:
        """An unresolved outcome that gets swept is an untraceable duplicate."""
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.mark_unknown("k")
        store.sweep(before=time.time() + 10_000)
        assert store.lookup("k") is not None, "sweep deleted an UNKNOWN record"

    def test_in_progress_is_never_swept(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.sweep(before=time.time() + 10_000)
        assert store.lookup("k") is not None, "sweep deleted an IN_PROGRESS record"

    def test_unresolved_lists_unknown_records(self) -> None:
        store = self.make_store()
        for i in range(3):
            store.claim(f"k{i}", "h", TTL)
            store.mark_unknown(f"k{i}")
        store.claim("done", "h", TTL)
        store.complete("done", None)
        keys = {r.key for r in store.unresolved(older_than=time.time() + 1)}
        assert keys == {"k0", "k1", "k2"}

    # -- reclaim ------------------------------------------------------------

    def test_expired_claim_is_reclaimable(self) -> None:
        """A holder that died must not hold the key forever."""
        store = self.make_store()
        assert store.claim("k", "h", ttl_seconds=-1).won is True
        assert store.claim("k", "h", TTL).won is True, "expired claim was not reclaimable"

    def test_live_claim_is_not_reclaimable(self) -> None:
        store = self.make_store()
        store.claim("k", "h", 3600)
        assert store.claim("k", "h", 3600).lost, "stole a claim that had not expired"

    def test_expired_claim_is_not_reclaimable_by_a_different_request(self) -> None:
        """An expired lease means the holder died, not that the key is free.

        Reclaiming with a different fingerprint would let one idempotency key
        execute two different requests — the failure the key exists to prevent.
        """
        store = self.make_store()
        store.claim("k", "original", ttl_seconds=-1)

        claim = store.claim("k", "different", TTL)

        assert claim.lost, "a different request reclaimed an expired lease"
        record = _found(claim.record)
        assert record.request_hash == "original", (
            "reclaim overwrote the stored fingerprint, so the caller above "
            "cannot tell that the request diverged"
        )

    def test_expired_claim_is_reclaimable_by_the_same_request(self) -> None:
        """The guard above must not block an honest retry of the same request."""
        store = self.make_store()
        store.claim("k", "same", ttl_seconds=-1)
        assert store.claim("k", "same", TTL).won is True

    # -- retention ----------------------------------------------------------
    #
    # Retention is not the claim lease. `expires_at` carries the lease deadline
    # while a record is IN_PROGRESS and the retention deadline once it is
    # terminal, so a store that leaves it alone on completion sweeps successful
    # records one claim-TTL after they are written. That deletes the evidence a
    # later delivery needs, and the effect runs a second time.

    def test_sweep_removes_terminal_records_past_their_retention(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.complete("k", "done", retention_seconds=-1)
        assert store.sweep(before=time.time()) == 1
        assert store.lookup("k") is None

    def test_sweep_keeps_terminal_records_inside_their_retention(self) -> None:
        store = self.make_store()
        store.claim("k", "h", TTL)
        store.complete("k", "done", retention_seconds=3600)
        assert store.sweep(before=time.time()) == 0
        assert store.lookup("k") is not None

    def test_retention_outlives_a_short_claim_lease(self) -> None:
        """The regression: a long retention must survive a short claim TTL."""
        store = self.make_store()
        store.claim("k", "h", ttl_seconds=-1)  # lease already expired
        store.complete("k", "done", retention_seconds=3600)

        assert store.sweep(before=time.time()) == 0, (
            "swept a completed record on its claim lease instead of its retention"
        )
        assert _found(store.lookup("k")).response == "done"

    def test_terminal_record_without_retention_is_never_swept(self) -> None:
        """No retention means keep it. Losing a record costs more than storage."""
        store = self.make_store()
        store.claim("k", "h", ttl_seconds=-1)
        store.complete("k", "done")
        assert store.sweep(before=time.time() + 10_000) == 0
        assert store.lookup("k") is not None

    def test_terminal_failure_honours_retention_too(self) -> None:
        store = self.make_store()
        store.claim("k", "h", ttl_seconds=-1)
        store.fail("k", terminal=True, retention_seconds=3600)
        assert store.sweep(before=time.time()) == 0
        assert _found(store.lookup("k")).state is State.FAILED

    # -- misc ---------------------------------------------------------------

    # -- clock ---------------------------------------------------------------
    #
    # A lease is written by one host and judged expired by another. If each
    # measures it against its own wall clock, the lease means different things
    # to each of them, and the atomic claim does not save you: host B decides
    # A's claim expired while A is still running the effect, B reclaims, and
    # the effect runs twice. NTP skew of seconds is normal; minutes happen
    # after a VM resume or with a broken time daemon.

    def test_a_lease_is_measured_against_a_clock_the_caller_does_not_control(self) -> None:
        """A caller whose clock runs fast must not be able to steal a live claim.

        Simulates the second host by moving this process's clock forward half an
        hour, which is skew far beyond anything NTP would leave but well inside
        what a resumed VM produces. The claim below has a fifteen-minute lease,
        so a store trusting the caller's clock sees it as long expired.
        """
        store = self.make_store()
        store.claim("skew", "hash", 900)

        with _process_clock_ahead_by(1800):
            stolen = store.claim("skew", "hash", 900).won

        if getattr(store, "clock", "store") == "process":
            # A single-process store has no clock but this one, so it does follow
            # the skew — and that is harmless, because no second process can
            # reach this store to disagree with it. Asserted rather than skipped,
            # so the exemption stays honest if the store ever becomes shared.
            assert stolen is True
            return

        assert stolen is False, (
            "a caller with a fast clock reclaimed a live lease — the store is "
            "measuring the lease against the caller's clock rather than its own, "
            "so two hosts will run the same effect"
        )

    def test_the_stores_clock_does_not_follow_the_callers(self) -> None:
        """The mechanism behind the test above, asserted directly."""
        store = self.make_store()
        if getattr(store, "clock", "store") == "process":
            # It *does* follow the caller, by construction. Asserted rather than
            # skipped so the claim stays a measurement instead of a comment.
            with _process_clock_ahead_by(1800):
                assert abs(store.now() - time.time()) < 5
            return

        with _process_clock_ahead_by(1800):
            drift = store.now() - time.time()
        # The store's clock stayed put while the caller's jumped 1800s forward.
        assert drift < -1500, f"the store's clock moved with the caller's (drift {drift:.0f}s)"

    # -- key width ---------------------------------------------------------
    #
    # A truncated key is a *collided* key. Two intents sharing a long prefix
    # collapse onto one, and the second is treated as a replay of the first —
    # so its effect never runs. That is the worst failure this library has: a
    # skipped payout, which no alert fires on, unlike a duplicate one.

    def test_a_long_key_survives_intact(self) -> None:
        """Whatever the store accepts must come back byte-identical.

        Namespaced, versioned keys are long by design — `operation_key` with a
        couple of UUIDs is already past 140 characters before anyone tries.
        """
        store = self.make_store()
        key = _long_key(store)

        assert store.claim(key, "hash", TTL).won
        assert _found(store.lookup(key)).key == key, "the store altered the key"

    def test_distinct_long_keys_do_not_collide(self) -> None:
        """Two keys differing only in their last character are two intents."""
        store = self.make_store()
        stem = _long_key(store)[:-1]
        first, second = stem + "1", stem + "2"

        assert store.claim(first, "hash", TTL).won
        assert store.claim(second, "hash", TTL).won, (
            "a distinct key was refused the claim — the store is treating two "
            "intents as one, and the second effect will never run"
        )

        # Resolving one must not resolve the other. If the keys collided, the
        # second lookup reads the first record and reports it terminal.
        store.complete(first, {"charge": 1})
        assert _found(store.lookup(second)).state is State.IN_PROGRESS

    def test_an_unstorable_key_is_refused_not_truncated(self) -> None:
        """A store that cannot hold a key whole must say so, loudly.

        Which way a fixed-width backend fails otherwise is a configuration
        detail — MySQL truncates or errors depending on `sql_mode` — and a
        correctness guarantee cannot rest on a session variable.
        """
        store = self.make_store()
        limit = getattr(store, "max_key_length", None)
        if limit is None:
            # Unbounded store: the same promise, made the other way. Nothing to
            # refuse, so prove instead that a key well past any fixed width
            # round-trips whole.
            key = "unbounded:" + "k" * 1000
            assert store.claim(key, "hash", TTL).won
            assert _found(store.lookup(key)).key == key
            return

        try:
            store.claim("x" * (limit + 1), "hash", TTL)
        except KeyTooLongError:
            return
        raise AssertionError(
            f"store accepted a {limit + 1}-character key with a declared limit "
            f"of {limit}; it will truncate and collide"
        )

    def test_lookup_missing_key_returns_none(self) -> None:
        assert self.make_store().lookup("nope") is None

    def test_request_hash_is_preserved(self) -> None:
        """The divergence guard depends on this surviving a round trip."""
        store = self.make_store()
        store.claim("k", "fingerprint-abc", TTL)
        assert _found(store.lookup("k")).request_hash == "fingerprint-abc"

    def test_keys_are_isolated(self) -> None:
        store = self.make_store()
        assert store.claim("a", "h", TTL).won
        assert store.claim("b", "h", TTL).won, "claiming one key blocked another"


def response_roundtrips(store: Store, value: Any) -> bool:
    """Helper for stores with unusual serialisation."""
    store.claim("rt", "h", TTL)
    store.complete("rt", value)
    return bool(_found(store.lookup("rt")).response == value)
