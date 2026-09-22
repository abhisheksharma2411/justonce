"""The state machine both engines share.

`Idempotent` and `AsyncIdempotent` differ in exactly one way: where they await.
Everything that *decides* — who replays a recorded outcome, who is told to
wait, who is refused — lives here, so there is one state machine rather than
two that happen to agree.

That mattered enough to factor out. The two engines were written together and
were identical on the day they shipped; nothing kept them that way. A fix
applied to one and forgotten in the other does not fail loudly — it quietly
gives async callers weaker guarantees than the documentation promises, which is
the worst failure shape this library has.

The decision is a pure function of three inputs::

    (record, request_hash, on_in_flight) -> Disposition

No clock, no store, no I/O. Every branch is reachable from a plain table, which
is what `tests/test_engine_parity.py` walks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, cast

from .errors import KeyReuseError, OperationInFlightError, StoreError
from .stores.base import Record, State


class OnInFlight(str, enum.Enum):
    """What to do when another caller holds the claim."""

    RAISE = "raise"
    """Reject immediately. Simplest and safest; maps to HTTP 409."""

    WAIT = "wait"
    """Poll until the holder reaches a terminal state, bounded by `wait_timeout`."""


class OnStoreUnavailable(str, enum.Enum):
    """What to do when the store cannot answer "has this run?" at all.

    This is not a tuning knob. It is the choice between two outages, and the
    library refuses to make it for you because the right answer depends on what
    the effect does, not on how the library is built.
    """

    FAIL_CLOSED = "fail_closed"
    """No store, no claim, no effect. The default, and it stays the default.

    An outage in the dedup layer becomes an outage in whatever it guards —
    payments stop while the database is down. That is a real cost, and it is
    the cost of the only behaviour that cannot produce a duplicate charge.
    """

    FAIL_OPEN = "fail_open"
    """Run the effect anyway, unguarded. **Duplicates become possible.**

    Not "unlikely" — possible, and concentrated exactly where it hurts. A store
    outage is when retries are most frequent, because the callers upstream are
    already seeing errors and retrying, and every one of those retries runs the
    effect again with nothing in the way.

    Choose this only for effects that genuinely tolerate being applied twice.
    "The customer would probably notice and call us" is not tolerance.
    """


@dataclass(frozen=True)
class Result:
    """Outcome of an idempotent execution."""

    value: Any
    executed: bool
    """True if this call ran the effect; False if a previous one did."""
    record: Record | None = None
    guarded: bool = True
    """False when the effect ran without a claim — see `OnStoreUnavailable`.

    Defaults to True because every other path in the library holds a claim.
    Only the fail-open branch sets it False, and it does so explicitly: a
    caller that alerts on `not result.guarded` learns it was running unguarded
    while it was happening, rather than from the duplicate-payment report.
    """

    @property
    def deduplicated(self) -> bool:
        return not self.executed


class Disposition(enum.Enum):
    """What a caller that did not win the claim must do.

    Deliberately not an exception and not a `Result`: the decision is separable
    from acting on it, so it can be tested as a table without running an engine.
    """

    REPLAY = "replay"
    """A terminal record exists. Return it; do not run the effect."""

    WAIT = "wait"
    """The holder is still working and the caller opted to wait."""

    IN_FLIGHT = "in_flight"
    """Refuse — raise `OperationInFlightError`."""

    KEY_REUSE = "key_reuse"
    """Same key, different payload — raise `KeyReuseError`."""


def classify(
    record: Record | None,
    request_hash: str,
    on_in_flight: OnInFlight,
) -> Disposition:
    """Decide what a losing caller does. Pure; the single source of truth.

    The ordering of these checks is itself the contract. In particular the
    payload comparison comes before any state check, because a mismatched
    payload makes the recorded outcome the *wrong answer* regardless of which
    state it is in — returning a succeeded record to a different request hands
    one caller another's response.
    """
    if record is None:
        # The holder finished and its record was swept between the failed claim
        # and this read. Treat as in-flight: the safe answer when we cannot
        # prove the effect did not run is to refuse, not to run it.
        return Disposition.IN_FLIGHT

    if record.request_hash != request_hash:
        return Disposition.KEY_REUSE

    if record.is_terminal:
        return Disposition.REPLAY

    if record.state is State.UNKNOWN:
        # Outcome genuinely unknown. Running again risks a duplicate; the
        # caller must resolve it through reconciliation, not by retrying.
        return Disposition.IN_FLIGHT

    if on_in_flight is OnInFlight.WAIT:
        return Disposition.WAIT
    return Disposition.IN_FLIGHT


def replay(record: Record) -> Result:
    """Build the result a losing caller sees for a terminal record.

    A failed record replays as `None` rather than re-raising: the exception
    belonged to the attempt that failed, and this caller did not make it.
    `record.state` is there for anyone who needs to tell the two apart.
    """
    return Result(
        value=record.response if record.state is State.SUCCEEDED else None,
        executed=False,
        record=record,
    )


def settle(
    key: str,
    record: Record | None,
    request_hash: str,
    on_in_flight: OnInFlight,
) -> Result | None:
    """Classify and act: return a `Result`, return `None` to wait, or raise.

    Both engines funnel every lost claim and every poll through this. `None`
    means "the holder is still working" — the sync engine sleeps, the async
    engine awaits, and that difference is the only one left between them.
    """
    disposition = classify(record, request_hash, on_in_flight)

    if disposition is Disposition.REPLAY:
        # classify returns REPLAY only for a non-None terminal record.
        return replay(cast("Record", record))
    if disposition is Disposition.WAIT:
        return None
    if disposition is Disposition.KEY_REUSE:
        raise KeyReuseError(key)
    raise OperationInFlightError(key)


def unguarded_run_allowed(exc: BaseException, policy: OnStoreUnavailable) -> bool:
    """May a caller whose claim raised `exc` run the effect without one?

    Both engines ask this and nothing else, so "unavailable" means one thing in
    both. Both conditions are deliberately narrow:

    * The policy must be `FAIL_OPEN`. It is never inferred from the exception,
      from a retry count, or from how long the store has been failing.
    * The exception must be `StoreError`, which is the stores' way of saying
      "the backend could not answer". `KeyTooLongError` also escapes `claim`
      and is emphatically not that — it says the key is about to be truncated
      into a collision with a different intent, so running the effect is the
      worst available response. Anything else escaping `claim` is a bug in that
      store, and a bug of unknown shape is not grounds for an unguarded effect.
    """
    return policy is OnStoreUnavailable.FAIL_OPEN and isinstance(exc, StoreError)


def check_windows(ttl_seconds: float, retention_seconds: float) -> None:
    """Refuse lease and retention windows that cannot mean what they say.

    Both engines call this from their constructor *and* from every `run()` that
    takes an override, so there is one predicate rather than two that happen to
    agree. A value rejected at startup is rejected per-call, which is the point:
    a per-call override that skipped the check would be a second, unguarded door
    into the state the constructor has refused since day one.

    What is *not* checked is the thing that actually bites — a TTL shorter than
    the effect's worst-case runtime. Nobody can know that runtime from here, and
    a number that looks generous next to the happy path is not generous next to
    a provider timing out. The lease expiring under a live holder is how one
    effect becomes two, so pick the TTL from the timeout you enforce on the
    effect, not from how long it usually takes.
    """
    if ttl_seconds <= 0:
        raise ValueError(
            f"ttl_seconds must be positive, got {ttl_seconds!r}; "
            "a lease that has already expired is not a lease"
        )
    if retention_seconds < 0:
        raise ValueError(
            f"retention_seconds must not be negative, got {retention_seconds!r}; "
            "a terminal record cannot expire before it is written"
        )
