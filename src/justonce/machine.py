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

from .errors import KeyReuseError, OperationInFlightError
from .stores.base import Record, State


class OnInFlight(str, enum.Enum):
    """What to do when another caller holds the claim."""

    RAISE = "raise"
    """Reject immediately. Simplest and safest; maps to HTTP 409."""

    WAIT = "wait"
    """Poll until the holder reaches a terminal state, bounded by `wait_timeout`."""


@dataclass(frozen=True)
class Result:
    """Outcome of an idempotent execution."""

    value: Any
    executed: bool
    """True if this call ran the effect; False if a previous one did."""
    record: Record | None = None

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
