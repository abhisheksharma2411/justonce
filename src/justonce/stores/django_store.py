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
from typing import Any, Literal

from ..errors import StoreError
from .base import Claim, Record, State, check_key_length, decode_response

try:  # pragma: no cover - import guard
    from django.db import connections
    from django.db import transaction as django_transaction
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DjangoStore requires Django. Install with: pip install justonce[django]"
    ) from exc

TABLE = "justonce_keys"

#: Width of the MySQL `key` column in the DDL below, and therefore the longest
#: key this store will accept on MySQL.
#:
#: MySQL is the only bundled backend with a *declared* key width; Postgres and
#: SQLite use unbounded `TEXT`. The number is not arbitrary — a `VARCHAR` index
#: is limited to 3072 bytes on InnoDB's DYNAMIC row format, and utf8mb4 costs
#: four bytes per character, so 768 is the widest a `VARCHAR` primary key can
#: be. Staying at 255 keeps the DDL working on the older COMPACT row format,
#: whose limit is 767 bytes.
#:
#: If 255 is too tight, widen the column and tell the store::
#:
#:     ALTER TABLE justonce_keys MODIFY `key` VARCHAR(768) NOT NULL;
#:     DjangoStore(max_key_length=768)
#:
#: The store cannot infer this: the DDL is `CREATE TABLE IF NOT EXISTS`, so an
#: existing table keeps whatever width it was created with, and guessing wide
#: would reintroduce exactly the silent truncation this guards against.
MYSQL_KEY_LENGTH = 255

#: DDL per vendor.
#:
#: The Postgres shape must stay byte-for-byte compatible with
#: `PostgresStore.SCHEMA` — same table name, same column types. Both DDLs are
#: `CREATE TABLE IF NOT EXISTS`, so in a database where both stores are used the
#: one that runs first wins and the other silently inherits its shape. When the
#: two disagreed on `response` (`JSONB` here, `TEXT` there) the loser read back
#: a JSON string where the contract promises the response object.
_DDL = {
    "postgresql": f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    key           TEXT PRIMARY KEY,
    state         TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response      JSONB,
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

#: Each vendor's own clock, as a Unix timestamp. Never the caller's — see
#: `justonce.stores.base.Store.now` for why that distinction is the whole point.
#:
#: Postgres uses `clock_timestamp()`, not `now()`/`CURRENT_TIMESTAMP`: those are
#: the *transaction's* start time and stay constant for its duration, and this
#: store shares the caller's ambient transaction by default, so a claim taken
#: inside a long-running transaction would measure its lease from whenever that
#: transaction opened.
#:
#: MySQL's `NOW(6)` is the statement's start time, which is what we want, and is
#: replication-safe in a way `SYSDATE(6)` is not.
_NOW = {
    "postgresql": "EXTRACT(EPOCH FROM clock_timestamp())",
    "mysql": "UNIX_TIMESTAMP(NOW(6))",
    "sqlite": "((julianday('now') - 2440587.5) * 86400.0)",
}

#: The atomic claim, per vendor. Every one of these is a single statement whose
#: winner is decided by the unique constraint — never a SELECT then an INSERT.
def _insert(vendor: str) -> str:
    """The atomic claim for `vendor`, with the timestamps taken server-side.

    Built per call rather than held in a dict because the clock expression is
    interpolated into it, and a `{TABLE}`-style constant that also carries SQL
    from `_NOW` is easier to read as a function than as a formatted literal.
    """
    now = _NOW[vendor]
    cols = "created_at, updated_at, expires_at"
    times = f"{now}, {now}, {now} + %s"
    if vendor == "mysql":
        # MySQL has no ON CONFLICT DO NOTHING. INSERT IGNORE suppresses the
        # duplicate-key error and reports 0 affected rows, which is the same signal.
        return (
            f"INSERT IGNORE INTO {TABLE} (`key`, state, request_hash, {cols}) "
            f"VALUES (%s, %s, %s, {times})"
        )
    conflict = (
        "ON CONFLICT (key) DO NOTHING" if vendor == "postgresql" else "ON CONFLICT(key) DO NOTHING"
    )
    return (
        f"INSERT INTO {TABLE} (key, state, request_hash, {cols}) "
        f"VALUES (%s, %s, %s, {times}) {conflict}"
    )


class DjangoStore:
    """Store backed by a Django database connection.

    Args:
        using: database alias from `settings.DATABASES`. Leave `None` to use the
            default connection and share the ambient transaction. Pass a
            separate alias when the effect is an external call, so the claim
            commits independently of a rollback.
        create_table: issue the DDL on construction. Convenient in development;
            prefer a real migration in production.
        max_key_length: longest key the `key` column holds whole. Leave
            `"auto"` to follow the shipped DDL — `MYSQL_KEY_LENGTH` on MySQL,
            unbounded elsewhere. Pass an int if you widened the column, or
            `None` to disable the check entirely, which re-exposes you to
            silent truncation.
    """

    def __init__(
        self,
        using: str | None = None,
        *,
        create_table: bool = False,
        max_key_length: int | Literal["auto"] | None = "auto",
    ) -> None:
        self.using = using or "default"
        self._max_key_length = max_key_length
        if create_table:
            self.create_table()

    #: Times come from the database server, which is the one clock every host
    #: talking to it shares.
    clock = "store"

    def now(self) -> float:
        """The database server's clock. See `justonce.stores.base.Store.now`."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_NOW[self.vendor]}")
            row = cur.fetchone()
        return float(row[0])

    @property
    def max_key_length(self) -> int | None:
        """Longest key this backend stores whole; `None` when unbounded.

        `"auto"` reports `MYSQL_KEY_LENGTH` on MySQL and `None` elsewhere,
        matching the DDL this store ships. Pass an explicit value after
        widening the column yourself.
        """
        if isinstance(self._max_key_length, str):
            return MYSQL_KEY_LENGTH if self.vendor == "mysql" else None
        return self._max_key_length

    # -- helpers ------------------------------------------------------------

    @property
    def _conn(self) -> Any:
        return connections[self.using]

    @property
    def vendor(self) -> str:
        v: str = self._conn.vendor
        if v not in _NOW:
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
        # Before the try: a key this store cannot hold is the caller's bug, not
        # a store failure, and must not be reported as `StoreError`.
        check_key_length(key, self.max_key_length, self.vendor)
        now_sql = _NOW[self.vendor]
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    _insert(self.vendor),
                    [key, State.IN_PROGRESS.value, request_hash, ttl_seconds],
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(cur, key))

                # Lost the insert. Reclaim only if the holder's lease expired,
                # and only through a conditional UPDATE so two reclaimers cannot
                # both succeed.
                #
                # `request_hash = %s` is the divergence guard, not an
                # optimisation. Without it this UPDATE overwrites the stored
                # hash and a *different* payload inherits the key and executes.
                cur.execute(
                    f"UPDATE {TABLE} SET state = %s, request_hash = %s, "
                    f"updated_at = {now_sql}, expires_at = {now_sql} + %s, "
                    "attempts = attempts + 1, response = NULL "
                    f"WHERE {self._key_col} = %s AND state = %s AND request_hash = %s "
                    f"AND expires_at IS NOT NULL AND expires_at < {now_sql}",
                    [
                        State.IN_PROGRESS.value, request_hash, ttl_seconds,
                        key, State.IN_PROGRESS.value, request_hash,
                    ],
                )
                if cur.rowcount == 1:
                    return Claim(won=True, record=self._get(cur, key))
                return Claim(won=False, record=self._get(cur, key))
        except Exception as exc:
            raise StoreError(f"django claim failed for {key!r}: {exc}") from exc

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        self._terminal(key, State.SUCCEEDED, response, retention_seconds)

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        if terminal:
            self._terminal(key, State.FAILED, None, retention_seconds)
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE {self._key_col} = %s", [key])

    def mark_unknown(self, key: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET state = %s, updated_at = {_NOW[self.vendor]}, "
                f"expires_at = NULL WHERE {self._key_col} = %s",
                [State.UNKNOWN.value, key],
            )

    def lookup(self, key: str) -> Record | None:
        with self._conn.cursor() as cur:
            return self._get(cur, key)

    def sweep(self, *, before: float | None = None) -> int:
        cutoff = _NOW[self.vendor] if before is None else "%s"
        params: list[Any] = [State.SUCCEEDED.value, State.FAILED.value]
        if before is not None:
            params.append(before)
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {TABLE} WHERE state IN (%s, %s) "
                f"AND expires_at IS NOT NULL AND expires_at < {cutoff}",
                params,
            )
            return int(cur.rowcount)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = _NOW[self.vendor] if older_than is None else "%s"
        params: list[Any] = [State.UNKNOWN.value]
        if older_than is not None:
            params.append(older_than)
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._cols} FROM {TABLE} WHERE state = %s "
                f"AND updated_at <= {cutoff} ORDER BY updated_at ASC LIMIT %s",
                params,
            )
            return [self._row(r) for r in cur.fetchall()]

    # -- internals ----------------------------------------------------------

    @property
    def _key_col(self) -> str:
        """MySQL reserves `key`, so it has to be quoted there."""
        return "`key`" if self.vendor == "mysql" else "key"

    @property
    def _cols(self) -> str:
        """Column list for reads.

        Postgres gets an explicit `::text` on `response` for the same reason
        `PostgresStore` does: the column may be `jsonb`, drivers disagree about
        whether they decode it, and a recorded payload of `"done"` is
        indistinguishable from JSON text once something has decoded it. Asking
        for text makes the decode happen exactly once, here.
        """
        response = "response::text AS response" if self.vendor == "postgresql" else "response"
        return (
            f"{self._key_col}, state, request_hash, {response}, attempts, "
            "created_at, updated_at, expires_at"
        )

    def _terminal(
        self, key: str, state: State, response: Any, retention_seconds: float | None = None
    ) -> None:
        # `expires_at` is replaced, never left alone — see the note in the
        # SQLite store. A terminal record still holding its claim lease gets
        # swept one claim-TTL after it was written, whatever retention says.
        now_sql = _NOW[self.vendor]
        expiry = "NULL" if retention_seconds is None else f"{now_sql} + %s"
        params: list[Any] = [
            state.value,
            json.dumps(response) if response is not None else None,
        ]
        if retention_seconds is not None:
            params.append(retention_seconds)
        params.append(key)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET state = %s, response = %s, updated_at = {now_sql}, "
                f"expires_at = {expiry} WHERE {self._key_col} = %s",
                params,
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
            response=decode_response(row[3]),
            attempts=row[4],
            created_at=row[5],
            updated_at=row[6],
            expires_at=row[7],
        )
