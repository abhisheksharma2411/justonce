"""In-memory store — for tests, and only for tests.

A dict behind a lock, so a handler decorated with `@idempotent` can be unit
tested without standing up a database. It passes the full conformance suite,
including the 24-thread claim race, because an in-memory store that is not
thread-safe is worse than none: it passes a user's single-threaded tests and
tells them their concurrent code is fine.

Two things it deliberately does *not* do differently from a real store, even
though it easily could:

**Responses are stored as JSON text, not as the object handed in.** Keeping the
object would be faster and would happily record a `datetime`, a `Decimal`, or a
model instance — none of which survive `SqliteStore` or `PostgresStore`. The
test would pass and production would raise, which is precisely the class of
surprise a test double exists to prevent. Encoding here means an unserialisable
response fails in the test, where it is cheap.

**Recorded responses are detached from the caller's object.** The round trip
through JSON is also a deep copy, so mutating the response afterwards cannot
retroactively change what was recorded. A store that hands back a live
reference lets a test pass because the object was edited in place.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from .base import Claim, Record, State, check_key_length, decode_response


class MemoryStore:
    """Process-local, non-durable store. Tests only.

    Every key lives in one process's heap: nothing is shared between processes,
    nothing survives the interpreter exiting, and two workers each get their own
    private view in which they both win the claim. That is not a limitation to
    work around — it means this store cannot provide the guarantee the library
    is for. Use `SqliteStore` with a real path, `PostgresStore`, or
    `DjangoStore` anywhere an effect actually happens.
    """

    #: A dict key has no width, so nothing is ever truncated.
    max_key_length: int | None = None

    #: This process's clock, because the store *is* this process. Cross-host
    #: skew is not a hazard here for the same reason durability is not a
    #: feature: nothing outside this interpreter can reach the data.
    clock = "process"

    def __init__(self) -> None:
        # RLock rather than Lock: `claim` calls `_get` while holding it, and a
        # plain Lock would deadlock the moment either grows a second internal
        # call. The cost is nil and the failure it prevents is a hang.
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, Any]] = {}

    # -- contract -----------------------------------------------------------

    def now(self) -> float:
        """This process's clock. See `justonce.stores.base.Store.now`."""
        return time.time()

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        check_key_length(key, self.max_key_length, "memory")
        now = time.time()
        expires = now + ttl_seconds
        with self._lock:
            row = self._rows.get(key)

            if row is None:
                self._rows[key] = {
                    "key": key,
                    "state": State.IN_PROGRESS.value,
                    "request_hash": request_hash,
                    "response": None,
                    "attempts": 1,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": expires,
                }
                return Claim(won=True, record=self._get(key))

            # Someone holds it. Reclaim only if the lease expired — and only for
            # the same request.
            #
            # The `request_hash` comparison is the divergence guard, not an
            # optimisation. An expired lease says the previous holder died, not
            # that the key is free for a different payload; without this check a
            # diverging caller would inherit the key and execute, which is the
            # exact thing an idempotency key exists to prevent. A caller whose
            # hash differs must lose here and meet the original hash above.
            reclaimable = (
                row["state"] == State.IN_PROGRESS.value
                and row["request_hash"] == request_hash
                and row["expires_at"] is not None
                and row["expires_at"] < now
            )
            if reclaimable:
                row.update(
                    state=State.IN_PROGRESS.value,
                    request_hash=request_hash,
                    response=None,
                    attempts=row["attempts"] + 1,
                    updated_at=now,
                    expires_at=expires,
                )
                return Claim(won=True, record=self._get(key))

            return Claim(won=False, record=self._get(key))

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        self._set_terminal(key, State.SUCCEEDED, response, retention_seconds)

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        if terminal:
            self._set_terminal(key, State.FAILED, None, retention_seconds)
            return
        # Transient: drop the claim so a later attempt can retry cleanly.
        with self._lock:
            self._rows.pop(key, None)

    def mark_unknown(self, key: str) -> None:
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                return
            # `expires_at = None` so no sweep can ever reach it. An unresolved
            # outcome that gets swept is a duplicate charge nobody can trace.
            row.update(state=State.UNKNOWN.value, updated_at=time.time(), expires_at=None)

    def lookup(self, key: str) -> Record | None:
        with self._lock:
            return self._get(key)

    def sweep(self, *, before: float | None = None) -> int:
        cutoff = time.time() if before is None else before
        with self._lock:
            doomed = [
                k
                for k, row in self._rows.items()
                if row["state"] in (State.SUCCEEDED.value, State.FAILED.value)
                and row["expires_at"] is not None
                and row["expires_at"] < cutoff
            ]
            for k in doomed:
                del self._rows[k]
            return len(doomed)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = older_than if older_than is not None else time.time()
        with self._lock:
            rows = [
                row
                for row in self._rows.values()
                if row["state"] == State.UNKNOWN.value and row["updated_at"] <= cutoff
            ]
            rows.sort(key=lambda r: r["updated_at"])
            return [self._row(r) for r in rows[:limit]]

    # -- internals ----------------------------------------------------------

    def _set_terminal(
        self, key: str, state: State, response: Any, retention_seconds: float | None
    ) -> None:
        # `expires_at` is replaced, never left alone. Until it is, the row still
        # carries the *claim lease's* deadline, and `sweep` reads that field for
        # terminal records too — so a completed record would be deleted one
        # claim-TTL after it was written, however long retention was set to.
        now = time.time()
        # Encoded outside the lock: json.dumps runs arbitrary __str__ via the
        # default hook, and there is no reason to hold the store shut for it.
        encoded = json.dumps(response) if response is not None else None
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                return
            row.update(
                state=state.value,
                response=encoded,
                updated_at=now,
                expires_at=None if retention_seconds is None else now + retention_seconds,
            )

    def _get(self, key: str) -> Record | None:
        row = self._rows.get(key)
        return self._row(row) if row is not None else None

    @staticmethod
    def _row(row: dict[str, Any]) -> Record:
        return Record(
            key=row["key"],
            state=State(row["state"]),
            request_hash=row["request_hash"],
            response=decode_response(row["response"]),
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )
