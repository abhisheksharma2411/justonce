"""The engine: claim, run, record.

The whole library is this sequence, and its correctness rests on one property —
the claim is atomic, so exactly one caller ever proceeds to the effect.

    claim ──won──> run effect ──> record outcome ──> return
      │
      └──lost──> terminal?  ──> return recorded response
                 in-flight? ──> reject / wait / raise

The case that earns the library its keep is the one in the middle: the process
dies between running the effect and recording the outcome. The key is left in
`UNKNOWN` rather than cleaned up, because "we do not know whether the customer
was charged" is a fact worth keeping.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable, TypeVar

from .codecs import ResponseCodec
from .errors import InFlightTimeout, KeyReuseError, ResponseDecodeError
from .hooks import Hooks, emit
from .keys import fingerprint
from .machine import (
    OnInFlight,
    OnStoreUnavailable,
    Result,
    check_windows,
    settle,
    unguarded_run_allowed,
)
from .namespacing import belongs, check_namespace, scoped, unscoped
from .stores.base import Claim, Record, Store

T = TypeVar("T")

#: Default claim lease. A claim older than this is presumed abandoned and may be
#: reclaimed. It must exceed the longest realistic runtime of the effect, or a
#: slow-but-alive holder gets its claim stolen while still running.
DEFAULT_TTL_SECONDS = 15 * 60

#: Default retention for terminal records. This is a *correctness* parameter,
#: not a storage optimisation: it must outlive the longest chain that can
#: re-deliver the same intent — including a dead-letter queue replayed a week
#: later, and any provider dispute window.
DEFAULT_RETENTION_SECONDS = 30 * 24 * 60 * 60


# `OnInFlight` and `Result` are re-exported here so existing imports keep
# working. Both now live in `justonce.machine`, alongside the decision logic the
# async engine shares with this one.
__all__ = [
    "DEFAULT_RETENTION_SECONDS",
    "DEFAULT_TTL_SECONDS",
    "Hooks",
    "Idempotent",
    "OnInFlight",
    "OnStoreUnavailable",
    "Result",
]


class Idempotent:
    """Runs effects at most once per key.

    Args:
        store: anything satisfying the `Store` protocol.
        ttl_seconds: claim lease. Must exceed the effect's worst-case runtime.
        on_in_flight: behaviour when another caller holds the claim.
        wait_timeout: bound for `OnInFlight.WAIT`.
        retention_seconds: how long terminal records are kept for `sweep`.
        on_store_unavailable: behaviour when the store cannot be reached at
            all. Fail-closed by default, and changing it is a decision about
            duplicate effects — read `OnStoreUnavailable` before you do.
        hooks: observability callbacks. See `justonce.hooks.Hooks`; a hook
            that raises is swallowed and cannot change an outcome.
        namespace: prefixed to every key, so tenants with their own id
            sequences cannot collide. `None` (the default) is the existing
            global keyspace. See `justonce.namespacing`.
        codec: shapes the response on its way into the store and back. The
            default stores it as-is, exactly as before. See `justonce.codecs`;
            this is where encryption at rest belongs.
        store_response: `False` records the outcome but not the body, so a
            replay is told the effect already ran and nothing else. Dedup is
            unchanged; the blast radius of the stored row is much smaller.
    """

    def __init__(
        self,
        store: Store,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        on_in_flight: OnInFlight = OnInFlight.RAISE,
        wait_timeout: float = 30.0,
        poll_interval: float = 0.05,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        on_store_unavailable: OnStoreUnavailable = OnStoreUnavailable.FAIL_CLOSED,
        hooks: Hooks | None = None,
        namespace: str | None = None,
        codec: ResponseCodec | None = None,
        store_response: bool = True,
    ) -> None:
        check_windows(ttl_seconds, retention_seconds)
        check_namespace(namespace)
        self.store = store
        self.hooks = hooks
        self.namespace = namespace
        self.codec = codec or ResponseCodec()
        self.store_response = store_response
        self.ttl_seconds = ttl_seconds
        self.on_in_flight = on_in_flight
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.retention_seconds = retention_seconds
        self.on_store_unavailable = on_store_unavailable

    def run(
        self,
        key: str,
        effect: Callable[[], T],
        *,
        payload: Any = None,
        retry_on_failure: bool = True,
        ttl_seconds: float | None = None,
        retention_seconds: float | None = None,
    ) -> Result:
        """Run `effect` at most once for `key`.

        Args:
            key: stable per intent. See `justonce.keys.operation_key`.
            effect: the side-effecting callable. Runs zero or one times.
            payload: request body, fingerprinted to detect key reuse.
            retry_on_failure: if the effect raises, whether a later attempt may
                retry this key. True releases the claim (transient failure);
                False records a terminal failure so the key is burned.
            ttl_seconds: claim lease for *this* call, overriding the engine's.
                A charge that takes seconds and a batch job that takes an hour
                need different leases, and one engine-wide value has to be the
                larger of the two.
            retention_seconds: replay window for *this* call's record.
                Likewise: a payment must stay replayable past the dispute
                window, and a nightly job is meaningless a day later.

        `None` for either means "not given, use the engine's" — it does *not*
        mean the store's "keep forever", which would silently make the key
        immortal. Set indefinite retention on the engine, where the choice is
        visible at configuration time.

        Raises:
            KeyReuseError: same key, different payload.
            OperationInFlightError: another caller holds the claim.
            StoreError: the store could not be reached, under the default
                `OnStoreUnavailable.FAIL_CLOSED`.
        """
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        retention = (
            self.retention_seconds if retention_seconds is None else retention_seconds
        )
        # Before the claim, not after: claiming and then rejecting would leave a
        # live IN_PROGRESS row for a call that never ran, and nothing resolves it
        # until the lease expires.
        check_windows(ttl, retention)

        request_hash = fingerprint(payload)
        stored = scoped(self.namespace, key)
        try:
            claim = self.store.claim(stored, request_hash, ttl)
        except BaseException as exc:
            if not unguarded_run_allowed(exc, self.on_store_unavailable):
                raise
            emit(self.hooks, "ran_unguarded", key)
            return self._run_unguarded(effect)

        if claim.lost:
            emit(self.hooks, "claim_conflict", key)
            return self._resolve_loser(stored, key, request_hash, claim.record)

        started = time.monotonic()
        try:
            value = effect()
        except BaseException:
            # The effect may or may not have applied. `retry_on_failure` says
            # which risk the caller prefers: a possible duplicate on retry, or
            # a possible lost effect. Never guess on their behalf.
            self.store.fail(
                stored,
                terminal=not retry_on_failure,
                retention_seconds=retention,
            )
            # After the outcome write, so the metric never claims an outcome the
            # ledger does not have.
            emit(self.hooks, "effect_finished", key, time.monotonic() - started, False)
            raise

        try:
            self.store.complete(
                stored,
                self.codec.encode(value) if self.store_response else None,
                retention_seconds=retention,
            )
        except BaseException:
            # The effect DID happen; we just could not record it. Leave the key
            # unresolved rather than releasing it — releasing would let a retry
            # apply the effect a second time.
            self.store.mark_unknown(stored)
            emit(self.hooks, "unknown_recorded", key)
            raise

        emit(self.hooks, "effect_finished", key, time.monotonic() - started, True)
        return Result(value=value, executed=True, record=self._strip(self.store.lookup(stored)))

    def claim_many(
        self, items: Sequence[tuple[str, str]], *, ttl_seconds: float | None = None
    ) -> dict[str, Claim]:
        """Claim many keys at once; return the outcome per key.

        Claiming 10,000 keys one round trip at a time makes bulk work — payout
        runs, nightly reconciliation, backfills — impractical (#28). A store
        that can do it in fewer statements says so by implementing
        `claim_many`; every other store, including third-party ones, is looped
        over here and keeps working unchanged.

        **Not atomic across keys.** Losing one key is an ordinary outcome, not
        a batch failure: rolling back the whole batch because one key was
        already held would discard claims the caller had legitimately won.

        Keys are namespaced and length-checked exactly as `claim` does, so a
        batch cannot smuggle in a key a single claim would have refused.
        """
        if len({key for key, _ in items}) != len(items):
            # Collapsing them would hide a divergence: two different payloads
            # under one key in one batch is precisely what the request hash
            # exists to catch, and a dict keyed by key can only report one.
            raise ValueError("claim_many received duplicate keys in one batch")

        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        stored = [(scoped(self.namespace, key), request_hash) for key, request_hash in items]

        batch = getattr(self.store, "claim_many", None)
        if callable(batch):
            claims = batch(stored, ttl)
        else:
            claims = {
                key: self.store.claim(key, request_hash, ttl)
                for key, request_hash in stored
            }
        # Back to caller-facing keys: the namespace is this engine's business,
        # and a caller that passed `order_1` must not get `tenant:order_1` back.
        return {
            unscoped(self.namespace, stored_key): replace(
                claim, record=self._strip(claim.record)
            )
            for stored_key, claim in claims.items()
        }

    def sweep(self, *, now: float | None = None) -> int:
        """Delete terminal records past their retention window.

        `now=None` lets the *store's* clock decide, which is the point: a
        sweeper running on a host whose clock is fast would otherwise delete
        records that have not actually expired, and a swept record is a key the
        next delivery of the same request cannot find.
        """
        return self.store.sweep(before=now)

    def oldest_unresolved_age(self) -> float | None:
        """Seconds since the oldest unresolved outcome was written, or `None`.

        The issue calls this the one to alert on, and it is right: a stuck
        reconciliation is invisible in a *count* that stays flat, because the
        count only moves when something new breaks. Age moves every second.

        Measured against the **store's** clock, for the same reason leases are
        (#47): computed against a caller whose clock runs fast, this gauge
        reports an age that never happened — and it is the number a pager is
        attached to.
        """
        # Through the scoped reader, not the store directly: a per-tenant
        # engine must not page an operator on another tenant's backlog.
        oldest = self.unresolved(limit=1)
        if not oldest:
            return None
        written = oldest[0].updated_at or oldest[0].created_at
        if written is None:
            return None
        return max(0.0, self.store.now() - written)

    def unresolved(self, *, limit: int = 100) -> list[Record]:
        """Effects whose outcome was never observed — reconciliation's input.

        Scoped to this engine's namespace, because the alternative hands one
        tenant another's list of "we do not know whether this customer was
        charged". An engine with no namespace sees everything, which is the
        operator's view rather than a tenant's.

        The paging matters. Fetching `limit` rows and filtering them is the
        obvious implementation and it under-reports: a noisy neighbour fills the
        first page and a scoped caller is told its queue is empty while its own
        records sit on page two. So it reads forward until it has `limit` of its
        own or the store is exhausted.
        """
        if self.namespace is None:
            return self.store.unresolved(limit=limit)

        found: list[Record] = []
        page = max(limit * 4, 100)
        seen = 0
        while len(found) < limit:
            batch = self.store.unresolved(limit=seen + page)[seen:]
            if not batch:
                break
            seen += len(batch)
            found.extend(
                self._rekey(r) for r in batch if belongs(self.namespace, r.key)
            )
        return found[:limit]

    # -- internals ----------------------------------------------------------

    def _run_unguarded(self, effect: Callable[[], T]) -> Result:
        """Run the effect with no claim behind it. See `OnStoreUnavailable`.

        Nothing is written afterwards, on either the success or the failure
        path. A `complete` for a key this caller never claimed is not a partial
        record, it is a false one: the store may be reachable again by then, or
        may have been reachable from another host all along, and the row this
        would overwrite could belong to a holder that actually won the claim.

        The consequence is that an unguarded run leaves no trace in the ledger,
        which is why `Result.guarded` is False — that flag is the only record
        the caller gets, so the caller has to be the one to keep it.
        """
        return Result(value=effect(), executed=True, record=None, guarded=False)

    def _decoded(self, key: str, record: Record | None) -> Record | None:
        """Run the stored response back through the codec before anyone reads it.

        Applied to the *loser's* view only. The winner already holds the live
        value and never needs the round trip — decoding there would turn a
        codec bug into a failure of the call that actually did the work.

        A codec that cannot read its own row raises rather than yielding
        `None`: see `ResponseDecodeError`.
        """
        if record is None or record.response is None or not self.store_response:
            return record
        try:
            return replace(record, response=self.codec.decode(record.response))
        except Exception as exc:
            raise ResponseDecodeError(key) from exc

    def _strip(self, record: Record | None) -> Record | None:
        """Hand a record back in the caller's keyspace, not the store's.

        Leaving the prefix on would mean a key read from `unresolved()` could
        not be passed straight back to `run()` — the caller would have to strip
        something the engine added, which is the kind of asymmetry that gets
        rediscovered during an incident.
        """
        return record if record is None else self._rekey(record)

    def _rekey(self, record: Record) -> Record:
        """The same record, with the namespace prefix taken back off its key."""
        if self.namespace is None:
            return record
        return replace(record, key=unscoped(self.namespace, record.key))

    def _resolve_loser(
        self, stored: str, key: str, request_hash: str, record: Record | None
    ) -> Result:
        try:
            # `key`, not `stored`: the error a caller catches names the key they
            # passed, not the one the engine derived from it.
            settled = settle(
                key, self._decoded(key, record), request_hash, self.on_in_flight,
                self.store_response,
            )
        except KeyReuseError:
            emit(self.hooks, "key_reuse", key)
            raise
        if settled is not None:
            emit(self.hooks, "duplicate_suppressed", key, settled.record)
            return replace(settled, record=self._strip(settled.record))
        return self._wait_for(stored, key, request_hash)

    def _wait_for(self, stored: str, key: str, request_hash: str) -> Result:
        deadline = time.monotonic() + self.wait_timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            # OnInFlight.WAIT, not self.on_in_flight: reaching here already
            # means waiting was chosen, and a still-in-progress holder must keep
            # us polling rather than raise.
            try:
                settled = settle(
                    key, self._decoded(key, self.store.lookup(stored)),
                    request_hash, OnInFlight.WAIT, self.store_response,
                )
            except KeyReuseError:
                emit(self.hooks, "key_reuse", key)
                raise
            if settled is not None:
                emit(self.hooks, "duplicate_suppressed", key, settled.record)
                return replace(settled, record=self._strip(settled.record))
        raise InFlightTimeout(key, self.wait_timeout)
