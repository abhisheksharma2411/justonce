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

import enum
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .errors import InFlightTimeout, KeyReuseError, OperationInFlightError
from .keys import fingerprint
from .stores.base import Record, State, Store

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


class Idempotent:
    """Runs effects at most once per key.

    Args:
        store: anything satisfying the `Store` protocol.
        ttl_seconds: claim lease. Must exceed the effect's worst-case runtime.
        on_in_flight: behaviour when another caller holds the claim.
        wait_timeout: bound for `OnInFlight.WAIT`.
        retention_seconds: how long terminal records are kept for `sweep`.
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
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.on_in_flight = on_in_flight
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.retention_seconds = retention_seconds

    def run(
        self,
        key: str,
        effect: Callable[[], T],
        *,
        payload: Any = None,
        retry_on_failure: bool = True,
    ) -> Result:
        """Run `effect` at most once for `key`.

        Args:
            key: stable per intent. See `justonce.keys.operation_key`.
            effect: the side-effecting callable. Runs zero or one times.
            payload: request body, fingerprinted to detect key reuse.
            retry_on_failure: if the effect raises, whether a later attempt may
                retry this key. True releases the claim (transient failure);
                False records a terminal failure so the key is burned.

        Raises:
            KeyReuseError: same key, different payload.
            OperationInFlightError: another caller holds the claim.
        """
        request_hash = fingerprint(payload)
        claim = self.store.claim(key, request_hash, self.ttl_seconds)

        if claim.lost:
            return self._resolve_loser(key, request_hash, claim.record)

        try:
            value = effect()
        except BaseException:
            # The effect may or may not have applied. `retry_on_failure` says
            # which risk the caller prefers: a possible duplicate on retry, or
            # a possible lost effect. Never guess on their behalf.
            self.store.fail(
                key,
                terminal=not retry_on_failure,
                retention_seconds=self.retention_seconds,
            )
            raise

        try:
            self.store.complete(key, value, retention_seconds=self.retention_seconds)
        except BaseException:
            # The effect DID happen; we just could not record it. Leave the key
            # unresolved rather than releasing it — releasing would let a retry
            # apply the effect a second time.
            self.store.mark_unknown(key)
            raise

        return Result(value=value, executed=True, record=self.store.lookup(key))

    def sweep(self, *, now: float | None = None) -> int:
        """Delete terminal records past their retention window."""
        cutoff = (now if now is not None else time.time())
        return self.store.sweep(before=cutoff)

    def unresolved(self, *, limit: int = 100) -> list[Record]:
        """Effects whose outcome was never observed — reconciliation's input."""
        return self.store.unresolved(limit=limit)

    # -- internals ----------------------------------------------------------

    def _resolve_loser(self, key: str, request_hash: str, record: Record | None) -> Result:
        if record is None:
            # The holder finished and its record was swept between our failed
            # claim and this read. Treat as in-flight: the safe answer when we
            # cannot prove the effect did not run is to refuse, not to run it.
            raise OperationInFlightError(key)

        if record.request_hash != request_hash:
            raise KeyReuseError(key)

        if record.state is State.SUCCEEDED:
            return Result(value=record.response, executed=False, record=record)
        if record.state is State.FAILED:
            return Result(value=None, executed=False, record=record)
        if record.state is State.UNKNOWN:
            # Outcome genuinely unknown. Running again risks a duplicate; the
            # caller must resolve it through reconciliation, not by retrying.
            raise OperationInFlightError(key)

        if self.on_in_flight is OnInFlight.RAISE:
            raise OperationInFlightError(key)
        return self._wait_for(key, request_hash)

    def _wait_for(self, key: str, request_hash: str) -> Result:
        deadline = time.monotonic() + self.wait_timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            record = self.store.lookup(key)
            if record is None:
                raise OperationInFlightError(key)
            if record.request_hash != request_hash:
                raise KeyReuseError(key)
            if record.is_terminal:
                return Result(
                    value=record.response if record.state is State.SUCCEEDED else None,
                    executed=False,
                    record=record,
                )
            if record.state is State.UNKNOWN:
                raise OperationInFlightError(key)
        raise InFlightTimeout(key, self.wait_timeout)
