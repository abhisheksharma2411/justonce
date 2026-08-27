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
from typing import Any, Callable, TypeVar

from .errors import InFlightTimeout
from .keys import fingerprint
from .machine import OnInFlight, Result, settle
from .stores.base import Record, Store

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
    "Idempotent",
    "OnInFlight",
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
        settled = settle(key, record, request_hash, self.on_in_flight)
        if settled is not None:
            return settled
        return self._wait_for(key, request_hash)

    def _wait_for(self, key: str, request_hash: str) -> Result:
        deadline = time.monotonic() + self.wait_timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            # OnInFlight.WAIT, not self.on_in_flight: reaching here already
            # means waiting was chosen, and a still-in-progress holder must keep
            # us polling rather than raise.
            settled = settle(key, self.store.lookup(key), request_hash, OnInFlight.WAIT)
            if settled is not None:
                return settled
        raise InFlightTimeout(key, self.wait_timeout)
