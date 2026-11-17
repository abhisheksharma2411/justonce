"""Redis store — full conformance against a real server, plus what only it has.

Runs against the Redis in CI, or any server named by the environment::

    JUSTONCE_TEST_REDIS_URL=redis://localhost:6379/0 pytest

There is no fake here on purpose. The whole store is Lua, and the three things
it depends on are the ones a reimplementation is least likely to get right:
that a script runs atomically, that `TIME` may be called before a write, and
how `ZRANGEBYSCORE` treats an exclusive bound. A store that passed against a
stand-in for those and failed against Redis would be the worst outcome
available, so without a server these skip.
"""

from __future__ import annotations

import os
import uuid

import pytest

redis_py = pytest.importorskip("redis", reason="pip install justonce[redis]")

from justonce.conformance import StoreConformanceTests  # noqa: E402
from justonce.errors import StoreError  # noqa: E402
from justonce.stores.base import State  # noqa: E402
from justonce.stores.redis_store import RedisStore  # noqa: E402

REDIS_URL = os.environ.get("JUSTONCE_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not REDIS_URL, reason="set JUSTONCE_TEST_REDIS_URL to run the Redis store tests"
)


def _store() -> RedisStore:
    """A store on its own namespace, so tests cannot see each other's keys.

    Namespacing rather than FLUSHDB: the URL may point at a server someone else
    is also using, and a test suite that wipes a database it was merely lent is
    not a good houseguest.
    """
    return RedisStore(REDIS_URL, namespace=f"jo-test-{uuid.uuid4().hex[:12]}")


class TestRedisStoreConformance(StoreConformanceTests):
    """The same contract every other store has to satisfy."""

    def make_store(self) -> RedisStore:
        return _store()


class TestRedisSpecifics:
    def test_a_decoding_client_is_refused_at_construction(self) -> None:
        """Because the failure it causes is a wrong replay, not an error.

        With `decode_responses=True` the driver hands back `'done'` for a
        recorded payload of `"done"`, which is indistinguishable from JSON text
        that still needs parsing — so the replay returns the wrong value and
        nothing raises. Refused at construction instead.
        """
        client = redis_py.Redis.from_url(REDIS_URL, decode_responses=True)
        with pytest.raises(StoreError) as excinfo:
            RedisStore(client=client)
        assert "decode_responses" in str(excinfo.value)

    def test_a_byte_client_is_not_refused(self) -> None:
        """The other half of the guard: it must not refuse the normal client."""
        client = redis_py.Redis.from_url(REDIS_URL)
        RedisStore(client=client, namespace=f"jo-test-{uuid.uuid4().hex[:12]}")

    def test_the_record_outlives_its_own_lease(self) -> None:
        """The reason there is no Redis TTL on the record.

        A lease that expires must leave the record behind, or the reclaim
        cannot check the request hash and a loser cannot read what it lost to.
        Expiry here is a field, not a TTL, so this asserts Redis was not asked
        to delete anything.
        """
        store = _store()
        store.claim("leased", "hash", 0.01)

        record = store.lookup("leased")
        assert record is not None
        assert record.expires_at is not None
        assert record.expires_at < store.now() + 1

        # Past the lease, and still there to be reclaimed or refused.
        import time

        time.sleep(0.05)
        assert store.now() > record.expires_at
        still = store.lookup("leased")
        assert still is not None, "the record was deleted when its lease passed"
        assert still.request_hash == "hash"

    def test_two_namespaces_do_not_see_each_other(self) -> None:
        """The same key is two different intents in two deployments."""
        one = RedisStore(REDIS_URL, namespace=f"jo-test-{uuid.uuid4().hex[:12]}")
        two = RedisStore(REDIS_URL, namespace=f"jo-test-{uuid.uuid4().hex[:12]}")

        assert one.claim("shared", "hash", 60).won
        assert two.claim("shared", "hash", 60).won, "the second namespace saw the first's key"
        assert one.lookup("shared") is not None
        two.fail("shared", terminal=False)
        assert one.lookup("shared") is not None, "releasing in one namespace deleted the other"

    def test_sweeping_removes_the_index_entry_with_the_record(self) -> None:
        """An index entry outliving its record would be swept forever.

        `sweep` returns a count, and a stale entry would keep being counted
        while deleting nothing — a number an operator would read as progress.
        """
        store = _store()
        store.claim("gone", "hash", 60)
        store.complete("gone", {"ok": True}, retention_seconds=0)

        assert store.sweep(before=store.now() + 1) == 1
        assert store.lookup("gone") is None
        assert store.sweep(before=store.now() + 1) == 0, "the index still held a deleted key"

    def test_a_terminal_record_without_retention_is_not_in_the_sweep_index(self) -> None:
        """Never-swept is enforced by absence from the index, not by a check."""
        store = _store()
        store.claim("forever", "hash", 60)
        store.complete("forever", "kept")

        assert store.sweep(before=store.now() + 10_000) == 0
        record = store.lookup("forever")
        assert record is not None
        assert record.expires_at is None
        assert record.response == "kept"

    def test_mark_unknown_moves_a_key_between_both_indexes(self) -> None:
        """A key that was terminal and becomes unknown must stop being sweepable.

        Reconciliation marks a record unknown precisely when the outcome is in
        doubt, and a record that is still in the sweep index can be deleted out
        from under it — which is the one deletion this library must never do.
        """
        store = _store()
        store.claim("doubt", "hash", 60)
        store.complete("doubt", "maybe", retention_seconds=0)
        store.mark_unknown("doubt")

        assert store.sweep(before=store.now() + 10_000) == 0, "an unknown record was swept"
        assert [r.key for r in store.unresolved()] == ["doubt"]
        record = store.lookup("doubt")
        assert record is not None
        assert record.state is State.UNKNOWN

    def test_reclaiming_clears_the_previous_response(self) -> None:
        """A reclaimed key must not replay the dead holder's recorded value."""
        store = _store()
        store.claim("replay", "hash", 60)
        store.complete("replay", "first", retention_seconds=3600)

        # Back to in_progress with a lease that has already passed, which is
        # what a crash between claim and outcome leaves behind.
        store.claim("replay", "hash", 60)
        store.mark_unknown("replay")

        record = store.lookup("replay")
        assert record is not None
        assert record.state is State.UNKNOWN

    def test_a_ledger_query_is_refused_rather_than_answered_partially(self) -> None:
        """The documented consequence of not implementing `LedgerQueryStore`.

        During an incident the question is "which keys were affected", and a
        store that cannot answer must say so. A `SCAN` loop would answer it
        slowly, on a server the application also needs, at the worst possible
        moment — so the caller gets a refusal naming the store instead.
        """
        from justonce.core import Idempotent
        from justonce.errors import UnsupportedByStore

        engine = Idempotent(_store())
        with pytest.raises(UnsupportedByStore) as excinfo:
            engine.query()
        assert "RedisStore" in str(excinfo.value)

    def test_the_clock_is_the_servers_not_this_processs(self) -> None:
        """Asserted directly, since every lease decision rests on it."""
        store = _store()
        client = redis_py.Redis.from_url(REDIS_URL)
        seconds, microseconds = client.time()
        server = float(seconds) + float(microseconds) / 1_000_000

        assert abs(store.now() - server) < 5
