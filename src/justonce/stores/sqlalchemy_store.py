"""SQLAlchemy store — one store for every SQL backend SQLAlchemy can reach.

Requires the ``sqlalchemy`` extra::

    pip install justonce[sqlalchemy]

Use it when the application already owns a SQLAlchemy ``Engine`` and you would
rather justonce borrow it than open a second pool to the same database::

    from sqlalchemy import create_engine
    from justonce.stores import SQLAlchemyStore

    store = SQLAlchemyStore(create_engine("postgresql+psycopg://localhost/app"))

`PostgresStore` and `SqliteStore` stay the reference implementations and have no
dependency on this one. This exists for reach, not to replace them: if you are
on Postgres and have no Engine already, the driver-level store is one less layer
between you and the statement that decides the winner.

**What is dialect-specific, and why there is no generic fallback.**

Everything here is ordinary SQL except the one statement that matters, and that
one cannot be written portably. "Insert this row unless the key is already
there, and tell me whether I was the one who inserted it" is spelled differently
by every engine, and the portable-looking spellings are all wrong:

- ``SELECT`` then ``INSERT`` is the check-then-act race the library exists to
  prevent. Two callers both see nothing and both believe they won.
- A caught ``IntegrityError`` looks atomic and is not, in the way that counts:
  on Postgres the failed ``INSERT`` aborts the surrounding transaction, so a
  store sharing the caller's transaction poisons it on every lost claim — which
  is the *normal* path, not an error path.

So the claim is written per dialect, and an unrecognised dialect is refused at
construction rather than served by a generic statement that races. A store that
is wrong only under concurrency is worse than one that will not start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..errors import StoreError
from .base import (
    LIKE_ESCAPE,
    Claim,
    Record,
    State,
    check_key_length,
    decode_response,
    like_prefix,
)

try:  # pragma: no cover - import guard
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SQLAlchemyStore requires SQLAlchemy. Install with: pip install justonce[sqlalchemy]"
    ) from exc


class _Dialect:
    """What one SQL engine needs spelled its own way.

    `now` is an expression evaluated by the database, never a value this
    process computes. A lease measured against the caller's clock is a lease
    two hosts disagree about, and the one whose clock is fast reclaims a key
    another host is still working on.
    """

    def __init__(
        self,
        *,
        name: str,
        now: str,
        key_type: str,
        response_type: str,
        float_type: str,
        claim: str,
        max_key_length: int | None,
        quote: str,
        inline_index: bool,
        like_escape_sql: str,
    ) -> None:
        self.name = name
        self.now = now
        self.key_type = key_type
        self.response_type = response_type
        self.float_type = float_type
        self.claim = claim
        self.max_key_length = max_key_length
        #: `key` is a reserved word in MySQL, so the column cannot be named
        #: unquoted there -- the CREATE TABLE fails outright with a syntax
        #: error. Postgres and SQLite accept it bare, but quoting everywhere
        #: keeps one spelling of the statements rather than two.
        self.quote = quote
        #: MySQL 8 has no `CREATE INDEX IF NOT EXISTS`, so its index is declared
        #: inside the table instead of as a second statement.
        self.inline_index = inline_index
        #: How `LIKE_ESCAPE` has to be written *inside a string literal* for
        #: this engine. Standard SQL gives a backslash no special meaning in a
        #: literal, so Postgres and SQLite take `'\\'` verbatim. MySQL reads
        #: backslash as an escape inside literals, so the same clause is a
        #: syntax error there and the character has to be doubled.
        #:
        #: MySQL's `NO_BACKSLASH_ESCAPES` sql_mode turns that back off, which
        #: would make the doubled form match two literal backslashes instead.
        #: That is a session variable, and the prefix query is what an operator
        #: runs during an incident to find out who was affected, so it says so
        #: here rather than quietly returning the wrong blast radius.
        self.like_escape_sql = like_escape_sql

    def q(self, identifier: str) -> str:
        return f"{self.quote}{identifier}{self.quote}"

    def schema(self) -> list[str]:
        index = ", KEY justonce_keys_state_updated (state, updated_at)" if self.inline_index else ""
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS justonce_keys (
                {self.q("key")}  {self.key_type} PRIMARY KEY,
                state         VARCHAR(16) NOT NULL,
                request_hash  VARCHAR(128) NOT NULL,
                response      {self.response_type},
                attempts      INTEGER NOT NULL DEFAULT 1,
                created_at    {self.float_type} NOT NULL,
                updated_at    {self.float_type} NOT NULL,
                expires_at    {self.float_type}{index}
            )
            """
        ]
        if not self.inline_index:
            statements.append(
                """
                CREATE INDEX IF NOT EXISTS justonce_keys_state_updated
                    ON justonce_keys (state, updated_at)
                """
            )
        return statements


#: `clock_timestamp()` rather than `now()`: `now()` is the transaction's start
#: time and is constant for its duration, so a claim taken inside a long
#: transaction would measure its lease from whenever that transaction opened.
_POSTGRES = _Dialect(
    name="postgresql",
    now="EXTRACT(EPOCH FROM clock_timestamp())",
    key_type="TEXT",
    response_type="JSONB",
    float_type="DOUBLE PRECISION",
    quote='"',
    inline_index=False,
    like_escape_sql="'" + LIKE_ESCAPE + "'",
    # RETURNING names exactly the row that landed. Reading the state back
    # instead would be attribution by guesswork: a row another process inserted
    # between our write and our read looks identical, and crediting ourselves
    # with a claim we did not win is a double execution.
    claim=(
        "INSERT INTO justonce_keys "
        "({key}, state, request_hash, attempts, created_at, updated_at, expires_at) "
        "VALUES (:key, :state, :request_hash, 1, {now}, {now}, {now} + :ttl) "
        "ON CONFLICT ({key}) DO NOTHING RETURNING {key}"
    ),
    max_key_length=None,
)

#: SQLite's julian-day conversion rather than `unixepoch('now','subsec')`, which
#: landed in SQLite 3.42 (2023); this package supports Python 3.9, whose bundled
#: SQLite can predate that.
_SQLITE = _Dialect(
    name="sqlite",
    now="((julianday('now') - 2440587.5) * 86400.0)",
    key_type="TEXT",
    response_type="TEXT",
    float_type="REAL",
    quote='"',
    inline_index=False,
    like_escape_sql="'" + LIKE_ESCAPE + "'",
    claim=(
        "INSERT INTO justonce_keys "
        "({key}, state, request_hash, attempts, created_at, updated_at, expires_at) "
        "VALUES (:key, :state, :request_hash, 1, {now}, {now}, {now} + :ttl) "
        "ON CONFLICT ({key}) DO NOTHING RETURNING {key}"
    ),
    max_key_length=None,
)

#: MySQL cannot index an unbounded `TEXT` as a primary key, so the key column is
#: a bounded `VARCHAR` and `max_key_length` refuses anything longer. That refusal
#: is the point: MySQL otherwise truncates or errors depending on `sql_mode`, and
#: a truncated key is a *collided* key — the second intent is read as a replay of
#: the first and its effect never runs. A correctness guarantee cannot rest on a
#: session variable.
#:
#: 191 rather than 255: utf8mb4 is four bytes per character and the InnoDB index
#: prefix limit is 767 bytes on the older row formats still in the field.
_MYSQL = _Dialect(
    name="mysql",
    # `now()` in MySQL is also the statement's time, but unlike Postgres it does
    # not freeze for the transaction; `now(6)` adds the subsecond resolution a
    # lease needs.
    now="UNIX_TIMESTAMP(now(6))",
    key_type="VARCHAR(191)",
    response_type="JSON",
    float_type="DOUBLE",
    quote="`",
    inline_index=True,
    like_escape_sql="'" + LIKE_ESCAPE + LIKE_ESCAPE + "'",
    # MySQL has no `ON CONFLICT ... RETURNING`. `INSERT IGNORE` gives rowcount 1
    # when the row landed and 0 when it did not, which is the same question.
    # It also suppresses other errors, which is why every column is either NOT
    # NULL with a value supplied here or nullable: there is no third outcome for
    # IGNORE to hide.
    claim=(
        "INSERT IGNORE INTO justonce_keys "
        "({key}, state, request_hash, attempts, created_at, updated_at, expires_at) "
        "VALUES (:key, :state, :request_hash, 1, {now}, {now}, {now} + :ttl)"
    ),
    max_key_length=191,
)

_DIALECTS = {
    "postgresql": _POSTGRES,
    "sqlite": _SQLITE,
    "mysql": _MYSQL,
    "mariadb": _MYSQL,
}


@dataclass(frozen=True)
class _Result:
    """One statement's answer, read before its transaction closed."""

    rows: list[Any]
    rowcount: int


class SQLAlchemyStore:
    """Store backed by any SQLAlchemy `Engine` whose dialect is supported.

    Args:
        engine: a SQLAlchemy `Engine`. Not a `Session` and not a `Connection`:
            the store takes a connection per operation and commits it, because
            a claim that is not durable when `claim` returns is not a claim.
        create_schema: run `CREATE TABLE IF NOT EXISTS` at construction. Set
            `False` where migrations own the schema.

    Raises:
        StoreError: if the engine's dialect has no audited atomic claim here.
    """

    #: Times come from the database, not this process. See `_Dialect`.
    clock = "store"

    def __init__(self, engine: Engine, *, create_schema: bool = True) -> None:
        name = engine.dialect.name
        dialect = _DIALECTS.get(name)
        if dialect is None:
            raise StoreError(
                f"SQLAlchemyStore has no audited atomic claim for the {name!r} dialect. "
                f"Supported: {', '.join(sorted(_DIALECTS))}. "
                "Deciding the winner is the one thing a store must do, and a generic "
                "statement cannot do it, so this refuses rather than racing."
            )
        self._engine = engine
        self._d = dialect
        self.max_key_length: int | None = dialect.max_key_length
        if create_schema:
            with self._engine.begin() as conn:
                for statement in dialect.schema():
                    conn.execute(text(statement))

    # -- helpers ------------------------------------------------------------

    def _sql(self, template: str) -> Any:
        return text(template.format(now=self._d.now, key=self._d.q("key")))

    def _execute(self, statement: Any, params: dict[str, Any] | None = None) -> _Result:
        """Run one statement in its own transaction and materialise the answer.

        The rows and the row count are read *inside* the transaction on
        purpose. A `CursorResult` handed out of the `begin()` block is not
        usable: the commit happens on the way out, and a statement whose
        `RETURNING` rows have not been consumed fails there with "cannot commit
        transaction - SQL statements in progress". Materialising keeps the
        caller from having to know which statements return rows.
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(statement, params or {})
                rows = list(result) if result.returns_rows else []
                return _Result(rows=rows, rowcount=result.rowcount)
        except SQLAlchemyError as exc:  # pragma: no cover - driver-dependent
            raise StoreError(str(exc)) from exc

    def _fetchone(self, template: str, params: dict[str, Any] | None = None) -> Any:
        rows = self._execute(self._sql(template), params).rows
        return rows[0] if rows else None

    def _fetchall(self, template: str, params: dict[str, Any] | None = None) -> list[Any]:
        return self._execute(self._sql(template), params).rows

    def _record(self, row: Any) -> Record:
        mapping = row._mapping
        return Record(
            key=mapping["key"],
            state=State(mapping["state"]),
            request_hash=mapping["request_hash"],
            response=decode_response(mapping["response"]),
            attempts=mapping["attempts"],
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
            expires_at=mapping["expires_at"],
        )

    # -- contract -----------------------------------------------------------

    def now(self) -> float:
        """The database's clock. See `justonce.stores.base.Store.now`."""
        row = self._fetchone("SELECT {now} AS t")
        return float(row._mapping["t"])

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        """Claim `key`, or report who holds it. See the base protocol.

        Two statements, each atomic on its own, and never a read used to decide
        a write. The first takes the key if it is free. The second reclaims a
        lease that has *already* expired, with the expiry still in the `WHERE`
        clause so that two reclaimers racing cannot both match.
        """
        check_key_length(key, self.max_key_length, self._d.name)
        params = {
            "key": key,
            "state": State.IN_PROGRESS.value,
            "request_hash": request_hash,
            "ttl": float(ttl_seconds),
        }
        result = self._execute(self._sql(self._d.claim), params)
        if self._won(result):
            return Claim(won=True, record=self.lookup(key))

        # Not free. Either someone holds it, or their lease lapsed and nobody
        # has taken it back yet.
        reclaimed = self._execute(
            self._sql(
                "UPDATE justonce_keys SET "
                "  state = :state, request_hash = :request_hash, response = NULL, "
                "  attempts = attempts + 1, updated_at = {now}, expires_at = {now} + :ttl "
                "WHERE {key} = :key "
                # The expiry test belongs in the statement, not around it: read
                # it first and two reclaimers both see a lapsed lease.
                "  AND expires_at IS NOT NULL AND expires_at <= {now} "
                "  AND state = :in_progress "
                # A lapsed lease is only reclaimable for the *same* request. A
                # different payload under the same key is the caller reusing a
                # key for a new intent, which must surface as reuse rather than
                # quietly overwrite the first.
                "  AND request_hash = :request_hash"
            ),
            {**params, "in_progress": State.IN_PROGRESS.value},
        )
        if reclaimed.rowcount == 1:
            return Claim(won=True, record=self.lookup(key))
        return Claim(won=False, record=self.lookup(key))

    def _won(self, result: _Result) -> bool:
        """Did our INSERT land?

        Postgres and SQLite answer with `RETURNING`; MySQL has no such clause
        and answers with the row count from `INSERT IGNORE`. Both are the
        database's own account of what it did, not an inference from a later
        read.
        """
        if self._d is _MYSQL:
            return bool(result.rowcount == 1)
        return bool(result.rows)

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        """Record success. See the base protocol."""
        expiry = "NULL" if retention_seconds is None else "{now} + :retention"
        params: dict[str, Any] = {
            "key": key,
            "state": State.SUCCEEDED.value,
            "response": json.dumps(response),
        }
        if retention_seconds is not None:
            params["retention"] = float(retention_seconds)
        self._execute(
            self._sql(
                "UPDATE justonce_keys SET state = :state, response = :response, "
                f"updated_at = {{now}}, expires_at = {expiry} WHERE {{key}} = :key"
            ),
            params,
        )

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        """Record failure. See the base protocol.

        A transient failure deletes the row, which is what releases the key:
        the next caller takes a fresh claim rather than reclaiming a lease.
        """
        if not terminal:
            self._execute(self._sql("DELETE FROM justonce_keys WHERE {key} = :key"), {"key": key})
            return
        expiry = "NULL" if retention_seconds is None else "{now} + :retention"
        params: dict[str, Any] = {"key": key, "state": State.FAILED.value}
        if retention_seconds is not None:
            params["retention"] = float(retention_seconds)
        self._execute(
            self._sql(
                "UPDATE justonce_keys SET state = :state, "
                f"updated_at = {{now}}, expires_at = {expiry} WHERE {{key}} = :key"
            ),
            params,
        )

    def mark_unknown(self, key: str) -> None:
        """Record that the outcome was never learned. See the base protocol.

        `expires_at = NULL` so no sweep can remove it. An UNKNOWN record is the
        only evidence that something may have happened, and deleting it turns
        "go and look" into "nothing to see".
        """
        self._execute(
            self._sql(
                "UPDATE justonce_keys SET state = :state, updated_at = {now}, "
                "expires_at = NULL WHERE {key} = :key"
            ),
            {"key": key, "state": State.UNKNOWN.value},
        )

    def lookup(self, key: str) -> Record | None:
        """See the base protocol."""
        row = self._fetchone(
            "SELECT {key}, state, request_hash, "
            # Selected as text explicitly: a driver that has already decoded
            # JSON for us hands back `'done'` where the encoded form is
            # `'\"done\"'`, and only the second can be parsed. Which one arrives
            # is a driver detail, so the ambiguity is removed here rather than
            # guessed at in `decode_response`.
            f"{self._response_as_text()} AS response, "
            "attempts, created_at, updated_at, expires_at "
            "FROM justonce_keys WHERE {key} = :key",
            {"key": key},
        )
        return self._record(row) if row is not None else None

    def _response_as_text(self) -> str:
        if self._d is _POSTGRES:
            return "response::text"
        if self._d is _MYSQL:
            return "CAST(response AS CHAR)"
        return "response"

    def sweep(self, *, before: float | None = None) -> int:
        """Delete terminal records past their retention. See the base protocol.

        Only SUCCEEDED and FAILED are eligible. IN_PROGRESS is someone's live
        work and UNKNOWN is the record reconciliation exists to resolve, so
        neither is ever swept however old it is.
        """
        cutoff = "{now}" if before is None else ":before"
        params: dict[str, Any] = {
            "succeeded": State.SUCCEEDED.value,
            "failed": State.FAILED.value,
        }
        if before is not None:
            params["before"] = float(before)
        result = self._execute(
            self._sql(
                "DELETE FROM justonce_keys WHERE state IN (:succeeded, :failed) "
                f"AND expires_at IS NOT NULL AND expires_at <= {cutoff}"
            ),
            params,
        )
        return int(result.rowcount)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        """UNKNOWN records, oldest first. See the base protocol."""
        clause = "" if older_than is None else "AND updated_at <= :older_than "
        params: dict[str, Any] = {"state": State.UNKNOWN.value, "limit": int(limit)}
        if older_than is not None:
            params["older_than"] = float(older_than)
        rows = self._fetchall(
            "SELECT {key}, state, request_hash, "
            f"{self._response_as_text()} AS response, "
            "attempts, created_at, updated_at, expires_at "
            f"FROM justonce_keys WHERE state = :state {clause}"
            "ORDER BY updated_at ASC LIMIT :limit",
            params,
        )
        return [self._record(row) for row in rows]

    # -- ledger query (LedgerQueryStore) ------------------------------------

    def query(
        self,
        *,
        prefix: str | None = None,
        states: tuple[State, ...] | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[Record]:
        """Read the ledger back. See `justonce.stores.base.LedgerQueryStore`."""
        where, params = self._ledger_filter(prefix, states, since, until)
        if where is None:
            return []
        params["limit"] = int(limit)
        rows = self._fetchall(
            "SELECT {key}, state, request_hash, "
            f"{self._response_as_text()} AS response, "
            "attempts, created_at, updated_at, expires_at "
            f"FROM justonce_keys {where} ORDER BY updated_at ASC, {{key}} ASC LIMIT :limit",
            params,
        )
        return [self._record(row) for row in rows]

    def count_by_state(
        self,
        *,
        prefix: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[State, int]:
        """Count by state. See `justonce.stores.base.LedgerQueryStore`.

        Every state is present in the result, zeros included: an operator
        reading "0 unknown" has been told something, where a missing key leaves
        them to decide whether it means none or means the query was wrong.
        """
        where, params = self._ledger_filter(prefix, None, since, until)
        assert where is not None  # states=None never empties the filter
        rows = self._fetchall(
            f"SELECT state, COUNT(*) AS n FROM justonce_keys {where} GROUP BY state", params
        )
        counts = {state: 0 for state in State}
        for row in rows:
            counts[State(row._mapping["state"])] = int(row._mapping["n"])
        return counts

    def _ledger_filter(
        self,
        prefix: str | None,
        states: tuple[State, ...] | None,
        since: float | None,
        until: float | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Build the shared WHERE clause, or `None` when it can match nothing."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if prefix:
            # LIKE reads `%` and `_` as wildcards and justonce keys are built
            # from user data, so the literal prefix is escaped rather than
            # interpolated. See `like_prefix`.
            clauses.append(f"{{key}} LIKE :prefix ESCAPE {self._d.like_escape_sql}")
            params["prefix"] = like_prefix(prefix)
        if states is not None:
            if not states:
                # An empty tuple asks for no states. Returning everything would
                # be the opposite of what was asked.
                return None, params
            names = []
            for index, state in enumerate(states):
                name = f"state_{index}"
                names.append(f":{name}")
                params[name] = state.value
            clauses.append(f"state IN ({', '.join(names)})")
        # Half-open [since, until): a record exactly on `until` belongs to the
        # next window, so paging through windows neither repeats nor drops one.
        if since is not None:
            clauses.append("updated_at >= :since")
            params["since"] = float(since)
        if until is not None:
            clauses.append("updated_at < :until")
            params["until"] = float(until)
        return ("WHERE " + " AND ".join(clauses) if clauses else ""), params
