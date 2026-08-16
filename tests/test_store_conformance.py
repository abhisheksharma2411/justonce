"""Run the store conformance suite against every bundled store.

This file is also the template for contributors: adding a store means adding a
class here that returns it. Nothing else.
"""

from __future__ import annotations

import os

import pytest

from justonce.conformance import StoreConformanceTests
from justonce.stores import SqliteStore


class TestSqliteStore(StoreConformanceTests):
    def make_store(self) -> SqliteStore:
        return SqliteStore(":memory:")


class TestSqliteFileStore(StoreConformanceTests):
    """Same store on disk — WAL mode and the file path change locking behaviour."""

    def make_store(self) -> SqliteStore:
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "justonce.db")
        return SqliteStore(path)


POSTGRES_DSN = os.environ.get("JUSTONCE_POSTGRES_DSN")


@pytest.mark.skipif(not POSTGRES_DSN, reason="set JUSTONCE_POSTGRES_DSN to run")
class TestPostgresStore(StoreConformanceTests):
    def make_store(self):
        from justonce.stores.postgres import PostgresStore

        store = PostgresStore(POSTGRES_DSN)
        # Conformance assumes a clean namespace per store instance.
        with store._connect() as conn:
            conn.execute("TRUNCATE justonce_keys")
        return store


def test_the_two_postgres_ddls_declare_the_same_columns() -> None:
    """`PostgresStore` and `DjangoStore` share one table name in one database.

    Both create it with `CREATE TABLE IF NOT EXISTS`, so whichever runs first
    decides the column types and the other silently inherits them. They shipped
    disagreeing on `response` — `JSONB` in one, `TEXT` in the other — and the
    loser of the race read back a JSON string where the contract promises the
    response object.

    Needs no database: it is a spelling check on the two DDL strings, which is
    the cheapest place to catch this drifting again.
    """
    django = pytest.importorskip("justonce.stores.django_store")
    from justonce.stores.postgres import SCHEMA

    def columns(ddl: str) -> dict[str, str]:
        create_table = ddl.split(";")[0]  # SCHEMA also carries a CREATE INDEX
        body = create_table[create_table.index("(") + 1 : create_table.rindex(")")]
        found = {}
        for line in body.splitlines():
            parts = line.strip().rstrip(",").split()
            if len(parts) >= 2 and not parts[0].startswith(("CREATE", "--")):
                found[parts[0].strip('`"')] = parts[1].upper()
        return found

    assert columns(django.DjangoStore.ddl("postgresql")) == columns(SCHEMA)


@pytest.mark.skipif(not POSTGRES_DSN, reason="set JUSTONCE_POSTGRES_DSN to run")
class TestPostgresReadsWhicheverColumnTypeItFinds:
    """The reader must not depend on how `response` happens to be typed.

    Matching the DDLs stops *new* databases from drifting, but it does nothing
    for a database created by an older version, or by a hand-written migration,
    or by a `DjangoStore` that got there first. So the read path is pinned
    directly: whatever the column type, `lookup()` returns the recorded object.

    `TEXT` is the shape that actually broke. `JSON` is here because nobody
    promised an operator would pick `JSONB`.
    """

    @staticmethod
    def _table(column_type: str):
        from justonce.stores.postgres import PostgresStore

        store = PostgresStore(POSTGRES_DSN, create_schema=False)
        with store._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS justonce_keys")
            conn.execute(
                f"""
                CREATE TABLE justonce_keys (
                    key TEXT PRIMARY KEY, state TEXT NOT NULL,
                    request_hash TEXT NOT NULL, response {column_type},
                    attempts INTEGER NOT NULL DEFAULT 1,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION
                )
                """
            )
        return store

    @pytest.mark.parametrize("column_type", ["JSONB", "TEXT", "JSON"])
    @pytest.mark.parametrize(
        "payload",
        [
            {"charge_id": "ch_1", "amount": 250},
            # A bare string is the payload that makes guessing impossible:
            # decoded from `jsonb` it is `'done'`, which no longer parses.
            "done",
            [1, 2, 3],
            None,
            0,
            False,
        ],
    )
    def test_response_round_trips(self, column_type: str, payload: object) -> None:
        store = self._table(column_type)
        store.claim("k", "h", 60)
        store.complete("k", payload)

        record = store.lookup("k")
        assert record is not None
        assert record.response == payload

    @pytest.mark.parametrize("column_type", ["JSONB", "TEXT"])
    def test_the_loser_of_a_claim_reads_the_recorded_response(self, column_type: str) -> None:
        """The library's whole promise, across the schema it inherited."""
        store = self._table(column_type)
        store.claim("k", "h", 60)
        store.complete("k", {"charge_id": "ch_1"})

        claim = store.claim("k", "h", 60)
        assert claim.lost
        assert claim.record is not None
        assert claim.record.response == {"charge_id": "ch_1"}

    @pytest.mark.parametrize("column_type", ["JSONB", "TEXT"])
    def test_unresolved_decodes_the_same_way_lookup_does(self, column_type: str) -> None:
        """`unresolved()` is reconciliation's input; it reads its own SQL."""
        store = self._table(column_type)
        store.claim("k", "h", 60)
        store.complete("k", {"charge_id": "ch_1"})
        store.mark_unknown("k")

        pending = store.unresolved()
        assert [r.key for r in pending] == ["k"]
        assert pending[0].response == {"charge_id": "ch_1"}


@pytest.mark.skipif(not POSTGRES_DSN, reason="set JUSTONCE_POSTGRES_DSN to run")
class TestPostgresAndDjangoStoresShareOneTable:
    """One process claims through Django, another reads through psycopg."""

    def test_each_store_reads_the_others_writes(self) -> None:
        from justonce.stores.postgres import PostgresStore
        from test_django_store import TestDjangoStorePostgres

        scratch = PostgresStore(POSTGRES_DSN, create_schema=False)
        with scratch._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS justonce_keys")

        django_store = TestDjangoStorePostgres().make_store()
        postgres_store = PostgresStore(POSTGRES_DSN)

        django_store.claim("via-django", "h", 60)
        django_store.complete("via-django", {"charge_id": "ch_1"})
        postgres_store.claim("via-psycopg", "h", 60)
        postgres_store.complete("via-psycopg", {"charge_id": "ch_2"})

        for store in (django_store, postgres_store):
            assert store.lookup("via-django").response == {"charge_id": "ch_1"}  # type: ignore[union-attr]
            assert store.lookup("via-psycopg").response == {"charge_id": "ch_2"}  # type: ignore[union-attr]
