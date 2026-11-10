"""SQLAlchemy store — full conformance, plus the parts only this store has.

Runs against SQLite by default so the suite needs no services. The claim
statement is the one dialect-specific thing in the store, so SQLite alone does
not prove the Postgres or MySQL spellings: both are gated on a URL in the
environment and skipped otherwise.

    JUSTONCE_TEST_PG_URL=postgresql+psycopg://user@localhost/justonce_test pytest
    JUSTONCE_TEST_MYSQL_URL=mysql+pymysql://root@localhost/justonce_test pytest
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="pip install justonce[sqlalchemy]")

from sqlalchemy import create_engine, text  # noqa: E402

from justonce.conformance import StoreConformanceTests  # noqa: E402
from justonce.errors import StoreError  # noqa: E402
from justonce.stores.base import State  # noqa: E402
from justonce.stores.sqlalchemy_store import SQLAlchemyStore  # noqa: E402

# File-backed rather than ":memory:": an in-memory SQLite database is scoped to
# a single connection, and the conformance concurrency test claims from a thread
# pool. Pooled connections to ":memory:" would each get their own empty
# database, so the test that matters most would pass without proving anything.
_TMP = Path(tempfile.mkdtemp(prefix="justonce-sqlalchemy-"))


def _sqlite_engine():
    path = _TMP / f"s{next(_counter)}.sqlite3"
    return create_engine(f"sqlite:///{path}", future=True)


def _counter_gen():
    n = 0
    while True:
        yield n
        n += 1


_counter = _counter_gen()


class TestSQLAlchemyStoreConformance(StoreConformanceTests):
    """The same contract every other store has to satisfy."""

    def make_store(self) -> SQLAlchemyStore:
        return SQLAlchemyStore(_sqlite_engine())


@pytest.mark.skipif(
    not os.environ.get("JUSTONCE_TEST_PG_URL"),
    reason="set JUSTONCE_TEST_PG_URL to run the Postgres dialect",
)
class TestPostgresDialectConformance(StoreConformanceTests):
    """The Postgres spelling of the claim, against a real Postgres.

    SQLite and Postgres share the `ON CONFLICT ... RETURNING` shape but not the
    clock expression, the JSON column type or the way the driver hands JSON
    back, so this is not a formality.
    """

    def make_store(self) -> SQLAlchemyStore:
        engine = create_engine(os.environ["JUSTONCE_TEST_PG_URL"], future=True)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS justonce_keys"))
        return SQLAlchemyStore(engine)


@pytest.mark.skipif(
    not os.environ.get("JUSTONCE_TEST_MYSQL_URL"),
    reason="set JUSTONCE_TEST_MYSQL_URL to run the MySQL dialect",
)
class TestMySQLDialectConformance(StoreConformanceTests):
    """The MySQL spelling, which is the one that differs most.

    MySQL has no `ON CONFLICT ... RETURNING`, so the winner is decided by the
    row count from `INSERT IGNORE`, and the key column is a bounded VARCHAR
    rather than TEXT. Both are exactly the kind of thing that looks right and
    is not, so the full contract runs against a real server or not at all.
    """

    def make_store(self) -> SQLAlchemyStore:
        engine = create_engine(os.environ["JUSTONCE_TEST_MYSQL_URL"], future=True)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS justonce_keys"))
        return SQLAlchemyStore(engine)


class TestSQLAlchemySpecifics:
    def test_an_unsupported_dialect_is_refused_at_construction(self) -> None:
        """Not served by a generic statement that races.

        The failure mode this avoids is the worst kind: a store that works in
        every test and loses its guarantee only under concurrency, in
        production, on a backend nobody exercised.

        Driven through a stub rather than a real engine on purpose. Naming an
        uninstalled dialect makes `create_engine` raise `NoSuchModuleError`
        first, so the test would pass without this check existing at all --
        which is the shape of a test that cannot fail.
        """

        class _Dialect:
            name = "oracle"

        class _Engine:
            dialect = _Dialect()

        with pytest.raises(StoreError) as excinfo:
            SQLAlchemyStore(_Engine())  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "oracle" in message
        # names what it does support, so the error is actionable
        assert "postgresql" in message and "sqlite" in message

    def test_a_supported_dialect_is_not_refused(self) -> None:
        """The other half of the guard: it must not refuse everything."""
        SQLAlchemyStore(_sqlite_engine())

    def test_create_schema_false_leaves_the_schema_alone(self) -> None:
        """For deployments where migrations own the table."""
        engine = _sqlite_engine()
        SQLAlchemyStore(engine, create_schema=False)
        with engine.begin() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        assert [t[0] for t in tables] == []

    def test_borrows_the_engine_rather_than_opening_its_own(self) -> None:
        """The reason to choose this store over the driver-level ones.

        Asserted by writing through the caller's engine and reading it back
        through the store: if the store had opened its own connection to
        somewhere else, the row would not be there.
        """
        engine = _sqlite_engine()
        store = SQLAlchemyStore(engine)
        store.claim("k", "h", 60.0)
        with engine.begin() as conn:
            row = conn.execute(text("SELECT state FROM justonce_keys WHERE key = 'k'")).fetchone()
        assert row is not None
        assert row[0] == State.IN_PROGRESS.value

    def test_the_claim_is_durable_when_claim_returns(self) -> None:
        """A claim still inside an uncommitted transaction is not a claim.

        If `claim` returned with the row only visible to its own transaction,
        a second process would see a free key and run the effect too. Checked
        from a connection the store never touched.
        """
        engine = _sqlite_engine()
        store = SQLAlchemyStore(engine)
        store.claim("durable", "h", 60.0)

        observer = create_engine(str(engine.url), future=True)
        with observer.begin() as conn:
            row = conn.execute(
                text("SELECT key FROM justonce_keys WHERE key = 'durable'")
            ).fetchone()
        assert row is not None

    def test_sqlite_has_no_key_limit_so_nothing_is_truncated(self) -> None:
        """TEXT has no declared width; MySQL's VARCHAR(191) is the case that does."""
        store = SQLAlchemyStore(_sqlite_engine())
        assert store.max_key_length is None
        long_key = "k" * 4000
        assert store.claim(long_key, "h", 60.0).won
        record = store.lookup(long_key)
        assert record is not None
        assert record.key == long_key
