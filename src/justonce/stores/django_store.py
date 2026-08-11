"""Django store — uses the connection your project already has.

    from justonce.stores.django_store import DjangoStore
    justonce.configure(DjangoStore())

No app to install, no migration to run through `INSTALLED_APPS`, and no second
connection pool. It borrows `django.db.connections[alias]` and issues
parameterised SQL, so it works on every backend Django supports.

Create the table once with `DjangoStore.create_table()`, or paste
`DjangoStore.ddl(vendor)` into your own migration — which is the better habit,
because the unique constraint on `key` *is* the correctness mechanism and
belongs somewhere a reviewer will see it.

## The transaction question

This is the decision that matters, and it has no single right answer:

* **Effect is an external call** (charging a card, sending an email). The claim
  must outlive a rollback. If it shares your `transaction.atomic()` block and
  that block rolls back, the claim disappears while the charge stands — and the
  retry charges again. Point the store at a **separate database alias** so its
  writes commit independently.

* **Effect is a local write in the same transaction.** Sharing the transaction
  is correct: claim and effect roll back together, which is exactly what you
  want, and the default `using=None` gives you that.

Silently picking one would be wrong, so the store makes you choose and says so
loudly in `__init__` when it detects the risky combination.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..errors import StoreError
from .base import Claim, Record, State

try:  # pragma: no cover - import guard
    from django.db import connections
    from django.db import transaction as django_transaction
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DjangoStore requires Django. Install with: pip install justonce[django]"
    ) from exc

TABLE = "justonce_keys"

_DDL = {
    "postgresql": f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    key           TEXT PRIMARY KEY,
    state         TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response      TEXT,
    attempts      INTEGER NOT NULL DEFAULT 1,
    created_at    DOUBLE PRECISION NOT NULL,
    updated_at    DOUBLE PRECISION NOT NULL,
    expires_at    DOUBLE PRECISION
)""",
    "mysql": f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    `key`          VARCHAR(255) PRIMARY KEY,
    state          VARCHAR(32) NOT NULL,
    request_hash   VARCHAR(255) NOT NULL,
    response       LONGTEXT,
    attempts       INT NOT NULL DEFAULT 1,
    created_at     DOUBLE NOT NULL,
    updated_at     DOUBLE NOT NULL,
    expires_at     DOUBLE
)""",
    "sqlite": f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    key           TEXT PRIMARY KEY,
    state         TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response      TEXT,
    attempts      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    expires_at    REAL
)""",
}

#: The atomic claim, per vendor. Every one of these is a single statement whose
#: winner is decided by the unique constraint — never a SELECT then an INSERT.
_INSERT = {
    "postgresql": (
        f"INSERT INTO {TABLE} (key, state, request_hash, created_at, updated_at, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (key) DO NOTHING"
    ),
    "sqlite": (
        f"INSERT INTO {TABLE} (key, state, request_hash, created_at, updated_at, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT(key) DO NOTHING"
    ),
    # MySQL has no ON CONFLICT DO NOTHING. INSERT IGNORE suppresses the duplicate
    # -key error and reports 0 affected rows, which is the same signal.
    "mysql": (
        f"INSERT IGNORE INTO {TABLE} (`key`, state, request_hash, created_at, updated_at, "
        "expires_at) VALUES (%s, %s, %s, %s, %s, %s)"
    ),
}


class DjangoStore:
    """Store backed by a Django database connection.

    Args:
        using: database alias from `settings.DATABASES`. Leave `None` to use the
            default connection and share the ambient transaction. Pass a
            separate alias when the effect is an external call, so the claim
            commits independently of a rollback.
        create_table: issue the DDL on construction. Convenient in development;
            prefer a real migration in production.
    """

    def __init__(self, using: str | None = None, *, create_table: bool = False) -> None:
        self.using = using or "default"
        if create_table:
            self.create_table()

    # -- helpers ------------------------------------------------------------

    @property
    def _conn(self) -> Any:
        return connections[self.using]

    @property
    def vendor(self) -> str:
        v: str = self._conn.vendor
        if v not in _INSERT:
            raise StoreError(
                f"DjangoStore has no atomic-claim statement for the {v!r} backend. "
                "A store without an atomic claim is not a store — please open an "
                "issue rather than working around this."
            )
        return v

    @classmethod
    def ddl(cls, vendor: str) -> str:
        """Table DDL for a vendor, to paste into your own migration."""
        try:
            return _DDL[vendor]
        except KeyError:
            raise StoreError(f"no DDL for backend {vendor!r}") from None

    def create_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self.ddl(self.vendor))
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS justonce_keys_state_updated "
                f"ON {TABLE} (state, updated_at)"
                if self.vendor != "mysql"
                # MySQL rejects IF NOT EXISTS on CREATE INDEX; the PK covers the
                # hot path and the sweep index is an optimisation, so skip it
                # rather than fail construction.
                else "SELECT 1"
            )

    def in_ambient_transaction(self) -> bool:
        """True when this store's writes would roll back with the caller's block.

        Use it to assert the mode you intended:

            assert not store.in_ambient_transaction(), \\
                "claim would roll back with the surrounding atomic() block"
        """
        return not django_transaction.get_autocommit(using=self.using)

    # -- contract -----------------------------------------------------------

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        now = time.time()
        expires = now + ttl_seconds
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    _INSERT[self.vendor],
                    [key, State.IN_PROGRESS.value, request_hash, now, now, expires],
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(cur, key))

                # Lost the insert. Reclaim only if the holder's lease expired,
                # and only through a conditional UPDATE so two reclaimers cannot
                # both succeed.
                cur.execute(
                    f"UPDATE {TABLE} SET state = %s, request_hash = %s, updated_at = %s, "
                    "expires_at = %s, attempts = attempts + 1, response = NULL "
                    f"WHERE {self._key_col} = %s AND state = %s "
                    "AND expires_at IS NOT NULL AND expires_at < %s",
                    [
                        State.IN_PROGRESS.value, request_hash, now, expires,
                        key, State.IN_PROGRESS.value, now,
                    ],
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(cur, key))
                return Claim(won=False, record=self._get(cur, key))
        except Exception as exc:
            raise StoreError(f"django claim failed for {key!r}: {exc}") from exc

    def complete(self, key: str, response: Any) -> None:
        self._terminal(key, State.SUCCEEDED, response)

    def fail(self, key: str, *, terminal: bool) -> None:
        if terminal:
            self._terminal(key, State.FAILED, None)
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE {self._key_col} = %s", [key])

    def mark_unknown(self, key: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET state = %s, updated_at = %s, expires_at = NULL "
                f"WHERE {self._key_col} = %s",
                [State.UNKNOWN.value, time.time(), key],
            )

    def lookup(self, key: str) -> Record | None:
        with self._conn.cursor() as cur:
            return self._get(cur, key)

    def sweep(self, *, before: float) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {TABLE} WHERE state IN (%s, %s) "
                "AND expires_at IS NOT NULL AND expires_at < %s",
                [State.SUCCEEDED.value, State.FAILED.value, before],
            )
            return int(cur.rowcount)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = older_than if older_than is not None else time.time()
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._cols} FROM {TABLE} WHERE state = %s AND updated_at <= %s "
                "ORDER BY updated_at ASC LIMIT %s",
                [State.UNKNOWN.value, cutoff, limit],
            )
            return [self._row(r) for r in cur.fetchall()]

    # -- internals ----------------------------------------------------------

    @property
    def _key_col(self) -> str:
        """MySQL reserves `key`, so it has to be quoted there."""
        return "`key`" if self.vendor == "mysql" else "key"

    @property
    def _cols(self) -> str:
        return (
            f"{self._key_col}, state, request_hash, response, attempts, "
            "created_at, updated_at, expires_at"
        )

    def _terminal(self, key: str, state: State, response: Any) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET state = %s, response = %s, updated_at = %s "
                f"WHERE {self._key_col} = %s",
                [
                    state.value,
                    json.dumps(response) if response is not None else None,
                    time.time(),
                    key,
                ],
            )

    def _get(self, cur: Any, key: str) -> Record | None:
        cur.execute(
            f"SELECT {self._cols} FROM {TABLE} WHERE {self._key_col} = %s", [key]
        )
        row = cur.fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: Any) -> Record:
        return Record(
            key=row[0],
            state=State(row[1]),
            request_hash=row[2],
            response=json.loads(row[3]) if row[3] is not None else None,
            attempts=row[4],
            created_at=row[5],
            updated_at=row[6],
            expires_at=row[7],
        )
