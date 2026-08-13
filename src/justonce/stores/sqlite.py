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
import time
from pathlib import Path
from typing import Any

from ..errors import StoreError
from .base import Claim, Record, State, decode_response

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

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        now = time.time()
        expires = now + ttl_seconds
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO justonce_keys
                        (key, state, request_hash, created_at, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (key, State.IN_PROGRESS.value, request_hash, now, now, expires),
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(key))

                # Lost the insert. The holder may be dead — reclaim only if its
                # lease expired, and only via a conditional UPDATE so two
                # reclaimers cannot both succeed.
                cur = self._conn.execute(
                    """
                    UPDATE justonce_keys
                       SET state = ?, request_hash = ?, updated_at = ?,
                           expires_at = ?, attempts = attempts + 1, response = NULL
                     WHERE key = ?
                       AND state = ?
                       AND expires_at IS NOT NULL
                       AND expires_at < ?
                    """,
                    (
                        State.IN_PROGRESS.value, request_hash, now, expires,
                        key, State.IN_PROGRESS.value, now,
                    ),
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(key))
                return Claim(won=False, record=self._get(key))
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                raise StoreError(f"sqlite claim failed for {key!r}: {exc}") from exc

    def complete(self, key: str, response: Any) -> None:
        self._set_terminal(key, State.SUCCEEDED, response)

    def fail(self, key: str, *, terminal: bool) -> None:
        if terminal:
            self._set_terminal(key, State.FAILED, None)
            return
        # Transient: drop the claim so a later attempt can retry cleanly.
        with self._lock:
            self._conn.execute("DELETE FROM justonce_keys WHERE key = ?", (key,))

    def mark_unknown(self, key: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE justonce_keys SET state = ?, updated_at = ?, expires_at = NULL
                 WHERE key = ?
                """,
                (State.UNKNOWN.value, time.time(), key),
            )

    def lookup(self, key: str) -> Record | None:
        with self._lock:
            return self._get(key)

    def sweep(self, *, before: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM justonce_keys
                 WHERE state IN (?, ?)
                   AND expires_at IS NOT NULL
                   AND expires_at < ?
                """,
                (State.SUCCEEDED.value, State.FAILED.value, before),
            )
            return cur.rowcount

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = older_than if older_than is not None else time.time()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM justonce_keys
                 WHERE state = ? AND updated_at <= ?
                 ORDER BY updated_at ASC LIMIT ?
                """,
                (State.UNKNOWN.value, cutoff, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- internals ----------------------------------------------------------

    def _set_terminal(self, key: str, state: State, response: Any) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE justonce_keys
                   SET state = ?, response = ?, updated_at = ?
                 WHERE key = ?
                """,
                (state.value, json.dumps(response) if response is not None else None,
                 time.time(), key),
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
