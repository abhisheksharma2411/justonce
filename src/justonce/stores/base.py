"""The store contract.

A store is the only thing that makes justonce correct, and it has exactly one
job: **decide a winner atomically**. Everything else here is bookkeeping.

If your backend cannot enforce uniqueness on the key in a single operation, it
is not a store — a `SELECT` followed by an `INSERT` is a race, and two callers
will both believe they won.

Implementing a store? See `justonce.conformance`: it is an executable version of
this contract, and a store that passes it is correct by our definition.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


def decode_response(value: Any) -> Any:
    """Parse a stored response back into the object that was recorded.

    Callers must hand this JSON *text* — never a value a driver has already
    decoded. That distinction cannot be recovered by inspection: a recorded
    payload of `"done"` arrives as the string `'done'` once decoded and as the
    string `'"done"'` while still encoded, and only the second is parseable. So
    the stores select the column as text explicitly rather than letting each
    driver decide, and this function does one unconditional decode.

    Bytes are accepted because some drivers hand back a buffer for text
    columns.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    return json.loads(value)


class State(str, enum.Enum):
    """Lifecycle of a claimed effect.

    `UNKNOWN` is the state that matters. It means the effect was started and we
    never learned the outcome — a crash between the call and the outcome write.
    It is not a failure to be swept away; it is the specific state
    reconciliation exists to resolve, and the only honest answer to "did this
    happen?" is "go and look".
    """

    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Record:
    """What the store knows about one key."""

    key: str
    state: State
    request_hash: str
    response: Any | None = None
    attempts: int = 1
    created_at: float | None = None
    updated_at: float | None = None
    expires_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in (State.SUCCEEDED, State.FAILED)


@dataclass(frozen=True)
class Claim:
    """Outcome of trying to claim a key.

    `won` is the only field that decides whether the caller runs the effect.
    Exactly one concurrent caller may ever see `won=True` for a given key.
    """

    won: bool
    record: Record | None = None

    @property
    def lost(self) -> bool:
        return not self.won


@runtime_checkable
class Store(Protocol):
    """Persistence contract for claims and their outcomes."""

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        """Atomically claim `key`, or report that someone else holds it.

        Must be a single atomic operation — typically an insert guarded by a
        unique constraint. Implementations must not read-then-write.

        Returns a `Claim` with `won=True` for the single winner. Every other
        caller gets `won=False` and the existing `record`, so the caller layer
        can decide whether to return the recorded response, wait, or reject.

        A claim whose `expires_at` has passed is reclaimable: the previous
        holder is presumed dead, and reclaiming must also be atomic.

        Must NOT raise on losing a claim — losing is an expected outcome, not an
        error. Raise `StoreError` only when the store itself failed.
        """
        ...

    def complete(self, key: str, response: Any) -> None:
        """Record a successful outcome and its response."""
        ...

    def fail(self, key: str, *, terminal: bool) -> None:
        """Record a failed outcome.

        `terminal=True` means do not retry this key — the effect definitively
        did not happen. `terminal=False` releases the claim so a later attempt
        can retry, which is correct for transient failures.
        """
        ...

    def mark_unknown(self, key: str) -> None:
        """Record that the effect was attempted with an unresolved outcome.

        Called when the effect may have been applied but the result was never
        observed. These records are reconciliation's input queue.
        """
        ...

    def lookup(self, key: str) -> Record | None:
        """Return the record for `key`, or None."""
        ...

    def sweep(self, *, before: float) -> int:
        """Delete terminal records that expired before `before`; return the count.

        Must never delete `IN_PROGRESS` or `UNKNOWN` records — an unresolved
        outcome that gets swept is a duplicate charge nobody can trace.
        """
        ...

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        """Records in `UNKNOWN`, oldest first — the reconciliation work list."""
        ...
