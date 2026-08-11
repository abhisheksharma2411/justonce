"""Postgres store — the reference implementation for multi-host deployments.

Requires the ``postgres`` extra::

    pip install justonce[postgres]

The atomic claim is `INSERT ... ON CONFLICT (key) DO NOTHING RETURNING key`.
Postgres decides the winner; a returned row means we were it. Reclaiming an
expired lease is a conditional `UPDATE` with the expiry in the `WHERE` clause,
so two reclaimers cannot both succeed.

Note the deliberate absence of `SELECT ... FOR UPDATE` anywhere: taking a lock
to decide who holds a lock just moves the race.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..errors import StoreError
from .base import Claim, Record, State

try:  # pragma: no cover - import guard
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresStore requires psycopg. Install with: pip install justonce[postgres]"
    ) from exc

SCHEMA = """
CREATE TABLE IF NOT EXISTS justonce_keys (
    key           TEXT        PRIMARY KEY,
    state         TEXT        NOT NULL,
    request_hash  TEXT        NOT NULL,
    response      JSONB,
    attempts      INTEGER     NOT NULL DEFAULT 1,
    created_at    DOUBLE PRECISION NOT NULL,
    updated_at    DOUBLE PRECISION NOT NULL,
    expires_at    DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS justonce_keys_state_updated
    ON justonce_keys (state, updated_at);
"""


class PostgresStore:
    """Postgres-backed store.

    Args:
        dsn: libpq connection string.
        create_schema: run `SCHEMA` on construction. Set False and manage the
            table through your own migrations in production — the unique
            constraint on `key` is the mechanism, so it belongs under review.
    """

    def __init__(self, dsn: str, *, create_schema: bool = True) -> None:
        self._dsn = dsn
        if create_schema:
            with self._connect() as conn:
                conn.execute(SCHEMA)

    def _connect(self) -> Any:
        try:
            return psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row)
        except psycopg.Error as exc:
            raise StoreError(f"could not connect to postgres: {exc}") from exc

    # -- contract -----------------------------------------------------------

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        now = time.time()
        expires = now + ttl_seconds
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    INSERT INTO justonce_keys
                        (key, state, request_hash, created_at, updated_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO NOTHING
                    RETURNING key
                    """,
                    (key, State.IN_PROGRESS.value, request_hash, now, now, expires),
                ).fetchone()
                if row is not None:
                    return Claim(won=True, record=self._get(conn, key))

                row = conn.execute(
                    """
                    UPDATE justonce_keys
                       SET state = %s, request_hash = %s, updated_at = %s,
                           expires_at = %s, attempts = attempts + 1, response = NULL
                     WHERE key = %s
                       AND state = %s
                       AND expires_at IS NOT NULL
                       AND expires_at < %s
                    RETURNING key
                    """,
                    (
                        State.IN_PROGRESS.value, request_hash, now, expires,
                        key, State.IN_PROGRESS.value, now,
                    ),
                ).fetchone()
                if row is not None:
                    return Claim(won=True, record=self._get(conn, key))
                return Claim(won=False, record=self._get(conn, key))
        except psycopg.Error as exc:
            raise StoreError(f"postgres claim failed for {key!r}: {exc}") from exc

    def complete(self, key: str, response: Any) -> None:
        self._terminal(key, State.SUCCEEDED, response)

    def fail(self, key: str, *, terminal: bool) -> None:
        if terminal:
            self._terminal(key, State.FAILED, None)
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM justonce_keys WHERE key = %s", (key,))

    def mark_unknown(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE justonce_keys
                   SET state = %s, updated_at = %s, expires_at = NULL
                 WHERE key = %s
                """,
                (State.UNKNOWN.value, time.time(), key),
            )

    def lookup(self, key: str) -> Record | None:
        with self._connect() as conn:
            return self._get(conn, key)

    def sweep(self, *, before: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM justonce_keys
                 WHERE state IN (%s, %s)
                   AND expires_at IS NOT NULL
                   AND expires_at < %s
                """,
                (State.SUCCEEDED.value, State.FAILED.value, before),
            )
            return int(cur.rowcount)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = older_than if older_than is not None else time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM justonce_keys
                 WHERE state = %s AND updated_at <= %s
                 ORDER BY updated_at ASC LIMIT %s
                """,
                (State.UNKNOWN.value, cutoff, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    # -- internals ----------------------------------------------------------

    def _terminal(self, key: str, state: State, response: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE justonce_keys
                   SET state = %s, response = %s, updated_at = %s
                 WHERE key = %s
                """,
                (state.value, json.dumps(response) if response is not None else None,
                 time.time(), key),
            )

    @staticmethod
    def _get(conn: Any, key: str) -> Record | None:
        row = conn.execute(
            "SELECT * FROM justonce_keys WHERE key = %s", (key,)
        ).fetchone()
        return PostgresStore._row(row) if row else None

    @staticmethod
    def _row(row: dict[str, Any]) -> Record:
        return Record(
            key=row["key"],
            state=State(row["state"]),
            request_hash=row["request_hash"],
            response=row["response"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )
