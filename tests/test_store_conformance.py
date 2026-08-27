"""Run the store conformance suite against every bundled store.

This file is also the template for contributors: adding a store means adding a
class here that returns it. Nothing else.
"""

from __future__ import annotations

import os

import pytest

from justonce import KeyTooLongError
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


class TestStoreWithAFixedWidthKeyColumn(StoreConformanceTests):
    """The whole contract, against a store that declares a key width.

    MySQL is the bundled backend with a fixed-width `key` column, and it has no
    service in CI. Declaring a width on SQLite exercises the same code path —
    the guard, and the width-aware conformance tests — everywhere the suite
    runs, rather than only where a MySQL server happens to exist.
    """

    LIMIT = 255

    def make_store(self) -> SqliteStore:
        class Narrow(SqliteStore):
            max_key_length = TestStoreWithAFixedWidthKeyColumn.LIMIT

        return Narrow(":memory:")

    def test_the_guard_actually_fires(self) -> None:
        store = self.make_store()
        with pytest.raises(KeyTooLongError) as caught:
            store.claim("x" * (self.LIMIT + 1), "hash", 60)

        # The message has to be actionable: someone reads it at 3am with a
        # payout stuck behind it.
        assert caught.value.limit == self.LIMIT
        assert "max_key_length" in str(caught.value)


def test_conformance_catches_a_store_that_truncates() -> None:
    """The contract must fail a store with the defect it was written for.

    A width test that passes against a truncating store proves nothing. This
    store is MySQL in non-strict `sql_mode`: it silently keeps the first 255
    characters of every key and declares no limit, so two distinct intents
    collapse onto one and the second effect is never applied.
    """

    class Truncating(SqliteStore):
        max_key_length = None  # claims to be unbounded, and is not

        def claim(self, key, request_hash, ttl_seconds):  # type: ignore[no-untyped-def]
            return super().claim(key[:255], request_hash, ttl_seconds)

        def lookup(self, key):  # type: ignore[no-untyped-def]
            return super().lookup(key[:255])

    class Contract(StoreConformanceTests):
        def make_store(self) -> SqliteStore:
            return Truncating(":memory:")

    contract = Contract()

    with pytest.raises(AssertionError, match=r"two intents as one|never run"):
        contract.test_distinct_long_keys_do_not_collide()

    with pytest.raises(AssertionError, match=r"altered the key"):
        contract.test_a_long_key_survives_intact()

    # And the same store passes everything that does not concern key width, so
    # the two new tests are what caught it — not incidental breakage.
    contract.test_only_one_concurrent_claimer_wins()
    contract.test_loser_can_read_the_recorded_response()
    contract.test_unknown_is_never_swept()
