"""Django store — full conformance plus the transaction question.

Configured against SQLite so the suite runs anywhere. The claim statement is
vendor-specific, so Postgres and MySQL need their own runs; those are gated on
env vars in the same way the Postgres store's are.
"""

from __future__ import annotations

import os

import pytest

django = pytest.importorskip("django", reason="pip install justonce[django]")

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from django.conf import settings  # noqa: E402

# File-backed rather than ":memory:": an in-memory SQLite database is scoped to
# a single connection, so the conformance concurrency test — which claims from a
# thread pool, and therefore from several connections — would not even see the
# table. The threads must share one database for that test to mean anything.
_TMP = Path(tempfile.mkdtemp(prefix="justonce-django-"))

if not settings.configured:
    settings.configure(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": str(_TMP / "default.sqlite3"),
                "OPTIONS": {"timeout": 20},
            },
            # A second alias, used to prove that a store pointed elsewhere does
            # not share the caller's transaction.
            "effects": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": str(_TMP / "effects.sqlite3"),
                "OPTIONS": {"timeout": 20},
            },
        },
        INSTALLED_APPS=[],
        USE_TZ=True,
    )
    django.setup()

from django.db import connections, transaction  # noqa: E402

from justonce import Idempotent, operation_key  # noqa: E402
from justonce.conformance import StoreConformanceTests  # noqa: E402
from justonce.stores.django_store import TABLE, DjangoStore  # noqa: E402


def _fresh(using: str = "default") -> DjangoStore:
    store = DjangoStore(using=using)
    store.create_table()
    with connections[using].cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE}")
    return store


class TestDjangoStoreConformance(StoreConformanceTests):
    """The same contract every other store has to satisfy."""

    def make_store(self) -> DjangoStore:
        return _fresh()


class TestDjangoSpecifics:
    def test_ddl_is_available_per_vendor(self) -> None:
        for vendor in ("postgresql", "mysql", "sqlite"):
            assert TABLE in DjangoStore.ddl(vendor)

    def test_unknown_vendor_is_refused_loudly(self) -> None:
        """Better to fail construction than to run without an atomic claim."""
        from justonce.errors import StoreError

        with pytest.raises(StoreError, match="no DDL"):
            DjangoStore.ddl("oracle")

    def test_reports_when_it_shares_the_callers_transaction(self) -> None:
        store = _fresh()
        assert store.in_ambient_transaction() is False
        with transaction.atomic():
            assert store.in_ambient_transaction() is True

    def test_claim_rolls_back_with_the_surrounding_block(self) -> None:
        """Documents the sharp edge, so nobody discovers it in production.

        A store on the default alias participates in the caller's transaction.
        That is correct when the effect is a local write — and dangerous when
        the effect is an external charge, because a rollback erases the claim
        while the charge stands.
        """
        store = _fresh()
        key = operation_key("charge", "order_1")

        class Rollback(Exception):
            pass

        with pytest.raises(Rollback), transaction.atomic():
            assert store.claim(key, "hash", 60).won is True
            raise Rollback

        # The claim is gone: a retry would run the effect again.
        assert store.lookup(key) is None
        assert store.claim(key, "hash", 60).won is True

    def test_a_separate_alias_survives_the_callers_rollback(self) -> None:
        """The configuration to use when the effect is an external call."""
        store = _fresh("effects")
        key = operation_key("charge", "order_2")

        class Rollback(Exception):
            pass

        with pytest.raises(Rollback), transaction.atomic(using="default"):
            assert store.claim(key, "hash", 60).won is True
            raise Rollback

        # The claim outlived the rollback, so the retry is correctly refused.
        assert store.lookup(key) is not None
        assert store.claim(key, "hash", 60).lost


class TestDjangoEngineIntegration:
    def test_effect_runs_once_through_the_engine(self) -> None:
        engine = Idempotent(_fresh())
        charges: list[str] = []
        key = operation_key("charge", "order_1")

        for _ in range(3):
            engine.run(key, lambda: charges.append("x"), payload={"amount": 100})

        assert len(charges) == 1


POSTGRES_DSN = os.environ.get("JUSTONCE_POSTGRES_DSN")


@pytest.mark.skipif(not POSTGRES_DSN, reason="set JUSTONCE_POSTGRES_DSN to run")
class TestDjangoStorePostgres(StoreConformanceTests):
    """Same store, Postgres dialect — a different atomic-claim statement."""

    def make_store(self) -> DjangoStore:
        from urllib.parse import unquote, urlparse

        u = urlparse(POSTGRES_DSN)
        connections.databases["pg"] = {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": u.path.lstrip("/"),
            "USER": unquote(u.username or ""),
            "PASSWORD": unquote(u.password or ""),
            "HOST": u.hostname or "",
            "PORT": str(u.port or 5432),
            "OPTIONS": {},
            "TIME_ZONE": None,
            "CONN_MAX_AGE": 0,
            "CONN_HEALTH_CHECKS": False,
            "AUTOCOMMIT": True,
            "ATOMIC_REQUESTS": False,
        }
        return _fresh("pg")
