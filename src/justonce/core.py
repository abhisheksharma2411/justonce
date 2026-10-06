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
from typing import Any, Callable, TypeVar, cast

from .codecs import ResponseCodec
from .errors import (
    InFlightTimeout,
    KeyReuseError,
    ResponseDecodeError,
    UnsupportedByStore,
)
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
from .reconcile import Action, Applied, Plan, ReconciliationProvider, build_plan
from .stores.base import Claim, Record, State, Store

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

    def now(self) -> float:
        """The store's clock, not the caller's.

        Ages shown to an operator are computed against this for the same reason
        leases are: a host whose clock runs fast would otherwise report an age
        that never happened, and that age is the number someone decides whether
        to reconcile on.
        """
        return self.store.now()

    def lookup(self, key: str) -> Record | None:
        """What the store knows about one key, or `None`.

        Namespaced like everything else on the engine: a caller that ran
        `order_1` asks about `order_1`, not `tenant:order_1`. A per-tenant
        engine therefore cannot read another tenant's record by guessing a key,
        which is the same boundary `unresolved` enforces for lists.
        """
        record = self.store.lookup(scoped(self.namespace, key))
        return self._strip(record)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
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

        `older_than` is passed to the store rather than filtered here, for the
        same reason: filtering after the fact would page through records the
        store could have skipped, and would let a burst of recent unknowns hide
        the old ones that actually need reconciling.
        """
        if self.namespace is None:
            return self.store.unresolved(older_than=older_than, limit=limit)

        found: list[Record] = []
        page = max(limit * 4, 100)
        seen = 0
        while len(found) < limit:
            batch = self.store.unresolved(older_than=older_than, limit=seen + page)[seen:]
            if not batch:
                break
            seen += len(batch)
            found.extend(
                self._rekey(r) for r in batch if belongs(self.namespace, r.key)
            )
        return found[:limit]

    def query(
        self,
        *,
        prefix: str | None = None,
        states: Sequence[State] | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[Record]:
        """Read the ledger back by key prefix, state and time window (#27).

        The question this exists for is the incident one: *how many customers
        were affected in this window, which ones, and is it still happening?*
        Nothing else in the ecosystem can answer it, because nothing else keeps
        the record — so the answer has to be exact or it is worse than nothing.

        `prefix` is a literal prefix of the **caller's** key, not a pattern and
        not the stored key. `%` and `_` in it are matched literally; keys like
        `charge:v1:order_123` contain one by convention and a raw LIKE would
        quietly return strangers' records alongside the real ones.

        `since`/`until` bound `created_at` half-open, so windows tile.

        Scoped to this engine's namespace, which here costs nothing: the
        namespace simply becomes part of the prefix and the store filters on
        it. `unresolved` has to page-and-discard for the same guarantee because
        the store cannot filter its list; this one does not.

        Raises `UnsupportedByStore` if the store does not implement the
        optional query protocol, rather than returning an empty list — see that
        error for why a silent partial answer is the worse failure.
        """
        run = self._ledger_method("query")
        scoped_prefix = self._scoped_prefix(prefix)
        records = run(
            prefix=scoped_prefix, states=states, since=since, until=until, limit=limit
        )
        return [self._rekey(r) for r in records]

    def summary(
        self,
        *,
        prefix: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[State, int]:
        """How many records sit in each state, under `query`'s filters (#27).

        Counted in the store rather than by paging `query`, because "is it
        still happening" gets asked of windows much larger than any `limit` a
        caller would pass — and a count derived from a truncated page is a
        number that looks precise and is not.

        **This does not count suppressed duplicates**, which #27 also asks for,
        because the ledger does not record them. A lost claim returns
        `Claim(won=False)` and writes nothing at all; `attempts` counts
        *reclaims* of an expired lease, which is a different event and far
        rarer. Reporting `attempts - 1` as a duplicate count would be wrong in
        both directions at once, and wrong on exactly the screen someone sizes
        an incident from. Counting them needs a column that does not exist yet.
        """
        run = self._ledger_method("count_by_state")
        counts = run(prefix=self._scoped_prefix(prefix), since=since, until=until)
        # Re-assert the full set: a third-party store may return only the
        # states it found, and a caller reading counts[UNKNOWN] mid-incident
        # should see 0 rather than a KeyError.
        return {state: counts.get(state, 0) for state in State}

    def _ledger_method(self, name: str) -> Callable[..., Any]:
        method = getattr(self.store, name, None)
        if not callable(method):
            raise UnsupportedByStore(self.store, f"ledger queries ({name})")
        return cast("Callable[..., Any]", method)

    def _scoped_prefix(self, prefix: str | None) -> str | None:
        """The caller's prefix as the store sees it.

        `None` with no namespace means no filter at all. With a namespace it
        becomes the namespace prefix itself, which is what keeps one tenant's
        query off another tenant's keys.
        """
        if self.namespace is None:
            return prefix
        return scoped(self.namespace, prefix or "")
    def plan_reconciliation(
        self,
        provider: ReconciliationProvider,
        *,
        older_than: float | None = None,
        limit: int = 100,
    ) -> Plan:
        """Ask `provider` what happened to each unresolved key. Writes nothing (#23).

        This is the half an operator reads before anything is allowed to move.
        It is separate from `apply_reconciliation` rather than being a
        `dry_run=True` flag on one method, because a flag that defaults to safe
        is still one typo away from unsafe, and the unsafe direction here
        rewrites money-movement records.

        `older_than` is worth setting. A record that went UNKNOWN two seconds
        ago may simply be in flight — the effect is still running and the
        outcome write has not happened yet. Reconciling it races the process
        that owns it.
        """
        records = self.unresolved(older_than=older_than, limit=limit)
        return build_plan(records, provider)

    def apply_reconciliation(self, plan: Plan) -> Applied:
        """Carry out a plan built by `plan_reconciliation`.

        **Each record is re-read immediately before it is written**, and skipped
        if it is no longer `UNKNOWN`. Between planning and applying, the process
        that originally owned the key may have come back and recorded the real
        outcome, and overwriting that with one inferred from the provider would
        replace a fact with a guess.

        That re-read narrows the window; it does not close it. There is no
        compare-and-set in the `Store` contract — `complete` and `fail` write
        unconditionally — so a record that changes between the re-read and the
        write is still overwritten. Closing it properly needs a conditional
        write primitive, which would be a change to every store. Said plainly
        here because "reconciliation is safe" is the kind of claim that gets
        believed, and this one has a bound on it.

        A write that raises is recorded and the run continues. Stopping on the
        first failure leaves the batch half-applied with no record of where it
        got to, which is strictly worse than finishing and reporting.
        """
        completed = released = 0
        stale: list[str] = []
        failed: list[tuple[str, str]] = []

        for step in plan.effective:
            current = self.lookup(step.key)
            if current is None or current.state is not State.UNKNOWN:
                stale.append(step.key)
                continue
            stored = scoped(self.namespace, step.key)
            try:
                if step.action is Action.COMPLETE:
                    self.store.complete(
                        stored, step.response, retention_seconds=self.retention_seconds
                    )
                    completed += 1
                else:
                    # Released, not burned. The provider says the effect never
                    # landed, so the right outcome is that a retry may run it —
                    # a terminal failure would mean "this will never happen",
                    # which is a different and unrecoverable claim.
                    self.store.fail(stored, terminal=False)
                    released += 1
            except Exception as exc:
                failed.append((step.key, f"{type(exc).__name__}: {exc}"))

        return Applied(completed=completed, released=released, stale=stale, failed=failed)

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
