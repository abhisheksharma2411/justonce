"""SQLite store.

Correct for single-process and multi-process-on-one-host use, and the default
for local development and tests because it needs no setup. For a fleet, use the
Postgres store — SQLite's writer lock does not span machines.

The atomic claim is `INSERT ... ON CONFLICT DO NOTHING` plus `changes()`: the
database decides the winner, and we ask it whether we were the one who inserted.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..errors import StoreError
from .base import Claim, Record, State, check_key_length, decode_response

#: SQLite's clock, as a Unix timestamp with subsecond resolution.
#:
#: `unixepoch('now','subsec')` would be tidier but landed in SQLite 3.42 (2023),
#: and this package supports Python 3.9, whose bundled SQLite can predate that.
#: The julian-day conversion works on every version and agrees with
#: `time.time()` to well under a millisecond.
#:
#: Evaluated by SQLite rather than passed in, so the value cannot come from a
#: caller whose clock is wrong. For a file-backed database every process shares
#: one host clock, so this is the shared authority; `:memory:` is single-process
#: and has nobody to disagree with.
_NOW = "((julianday('now') - 2440587.5) * 86400.0)"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS justonce_keys (
    key           TEXT    PRIMARY KEY,
    state         TEXT    NOT NULL,
    request_hash  TEXT    NOT NULL,
    response      TEXT,
    attempts      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL,
    expires_at    REAL
);
CREATE INDEX IF NOT EXISTS justonce_keys_state_updated
    ON justonce_keys (state, updated_at);
"""


class SqliteStore:
    """SQLite-backed store. Pass ``":memory:"`` for an ephemeral store."""

    #: Times come from SQLite, not the caller. For a file-backed database every
    #: process shares one host clock, which is the authority they have in common.
    clock = "store"

    #: SQLite's `TEXT` has no declared width, so no key is ever truncated.
    max_key_length: int | None = None

    def __init__(self, path: str | Path = ":memory:", *, timeout: float = 5.0) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        # check_same_thread=False + our own lock: the conformance suite drives
        # this from a thread pool to prove the claim is genuinely atomic.
        self._conn = sqlite3.connect(
            self._path, timeout=timeout, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    # -- contract -----------------------------------------------------------

    def now(self) -> float:
        """SQLite's clock. See `justonce.stores.base.Store.now`."""
        with self._lock:
            value: float = self._conn.execute(f"SELECT {_NOW}").fetchone()[0]
        return value

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        check_key_length(key, self.max_key_length, "sqlite")
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"""
                    INSERT INTO justonce_keys
                        (key, state, request_hash, created_at, updated_at, expires_at)
                    VALUES (?, ?, ?, {_NOW}, {_NOW}, {_NOW} + ?)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (key, State.IN_PROGRESS.value, request_hash, ttl_seconds),
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(key))

                # Lost the insert. The holder may be dead — reclaim only if its
                # lease expired, and only via a conditional UPDATE so two
                # reclaimers cannot both succeed.
                #
                # `request_hash = ?` in the WHERE clause is the divergence
                # guard, not an optimisation. Without it the UPDATE overwrites
                # the stored hash, and a *different* payload inherits the key
                # and executes — the exact thing an idempotency key exists to
                # prevent. A caller whose hash differs must lose here and meet
                # the original hash in `_resolve_loser`.
                cur = self._conn.execute(
                    f"""
                    UPDATE justonce_keys
                       SET state = ?, request_hash = ?, updated_at = {_NOW},
                           expires_at = {_NOW} + ?, attempts = attempts + 1,
                           response = NULL
                     WHERE key = ?
                       AND state = ?
                       AND request_hash = ?
                       AND expires_at IS NOT NULL
                       AND expires_at < {_NOW}
                    """,
                    (
                        State.IN_PROGRESS.value, request_hash, ttl_seconds,
                        key, State.IN_PROGRESS.value, request_hash,
                    ),
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(key))
                return Claim(won=False, record=self._get(key))
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                raise StoreError(f"sqlite claim failed for {key!r}: {exc}") from exc

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        self._set_terminal(key, State.SUCCEEDED, response, retention_seconds)

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        if terminal:
            self._set_terminal(key, State.FAILED, None, retention_seconds)
            return
        # Transient: drop the claim so a later attempt can retry cleanly.
        with self._lock:
            self._conn.execute("DELETE FROM justonce_keys WHERE key = ?", (key,))

    def mark_unknown(self, key: str) -> None:
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE justonce_keys SET state = ?, updated_at = {_NOW}, expires_at = NULL
                 WHERE key = ?
                """,
                (State.UNKNOWN.value, key),
            )

    def lookup(self, key: str) -> Record | None:
        with self._lock:
            return self._get(key)

    def sweep(self, *, before: float | None = None) -> int:
        # `before=None` means "the store's clock" rather than the caller's, so a
        # skewed sweeper cannot delete a record that has not actually expired.
        cutoff = _NOW if before is None else "?"
        params: list[Any] = [State.SUCCEEDED.value, State.FAILED.value]
        if before is not None:
            params.append(before)
        with self._lock:
            cur = self._conn.execute(
                f"""
                DELETE FROM justonce_keys
                 WHERE state IN (?, ?)
                   AND expires_at IS NOT NULL
                   AND expires_at < {cutoff}
                """,
                params,
            )
            return cur.rowcount

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = _NOW if older_than is None else "?"
        params: list[Any] = [State.UNKNOWN.value]
        if older_than is not None:
            params.append(older_than)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM justonce_keys
                 WHERE state = ? AND updated_at <= {cutoff}
                 ORDER BY updated_at ASC LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- internals ----------------------------------------------------------

    def _set_terminal(
        self, key: str, state: State, response: Any, retention_seconds: float | None = None
    ) -> None:
        # `expires_at` is replaced, never left alone. Until it is, the row still
        # carries the *claim lease's* deadline, and `sweep` reads that column
        # for terminal records too — so a completed record would be deleted one
        # claim-TTL after it was written, however long retention was set to.
        encoded = json.dumps(response) if response is not None else None
        expiry = "NULL" if retention_seconds is None else f"{_NOW} + ?"
        params: list[Any] = [state.value, encoded]
        if retention_seconds is not None:
            params.append(retention_seconds)
        params.append(key)
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE justonce_keys
                   SET state = ?, response = ?, updated_at = {_NOW}, expires_at = {expiry}
                 WHERE key = ?
                """,
                params,
            )

    def _get(self, key: str) -> Record | None:
        row = self._conn.execute(
            "SELECT * FROM justonce_keys WHERE key = ?", (key,)
        ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> Record:
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
