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
