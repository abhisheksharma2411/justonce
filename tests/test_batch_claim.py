"""Batch claim (#28).

The whole risk of this feature is that a faster path disagrees with `claim`.
Every test here compares the two rather than asserting the batch in isolation.
"""

from __future__ import annotations

import pytest

from justonce import Idempotent
from justonce.stores.base import BatchClaimStore
from justonce.stores.memory import MemoryStore
from justonce.stores.sqlite import SqliteStore


def sqlite_store(tmp_path) -> SqliteStore:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return SqliteStore(str(tmp_path / "batch.db"))


def test_sqlite_advertises_the_capability_and_memory_does_not():
    # The point of an optional protocol: a store that has not opted in must
    # keep working, not fail a type check.
    assert isinstance(SqliteStore(":memory:"), BatchClaimStore)
    assert not isinstance(MemoryStore(), BatchClaimStore)


def test_a_store_without_claim_many_still_works(monkeypatch):
    engine = Idempotent(MemoryStore())
    claims = engine.claim_many([("a", "h1"), ("b", "h2")])
    assert {key: claim.won for key, claim in claims.items()} == {"a": True, "b": True}


def _outcomes(engine, items):
    return {key: claim.won for key, claim in engine.claim_many(items).items()}


def test_batch_matches_looping_claim_on_fresh_keys(tmp_path):
    batched = Idempotent(sqlite_store(tmp_path / "a"))
    looped = Idempotent(sqlite_store(tmp_path / "b"))
    items = [(f"k{i}", f"h{i}") for i in range(5)]

    by_batch = _outcomes(batched, items)
    by_loop = {
        key: looped.store.claim(key, request_hash, looped.ttl_seconds).won
        for key, request_hash in items
    }
    assert by_batch == by_loop == {f"k{i}": True for i in range(5)}


def test_a_key_already_held_loses_in_a_batch_exactly_as_it_would_alone(tmp_path):
    engine = Idempotent(sqlite_store(tmp_path))
    engine.store.claim("held", "original", engine.ttl_seconds)

    claims = engine.claim_many([("fresh", "h"), ("held", "original")])

    assert claims["fresh"].won is True
    # Losing one key must not cost the caller the one it won: the batch is not
    # atomic across keys, deliberately.
    assert claims["held"].won is False


def test_a_divergent_hash_on_a_held_key_loses_and_keeps_the_original(tmp_path):
    # The guarantee the batch path must not weaken: a different payload may not
    # inherit a key, and the loser must meet the original hash.
    engine = Idempotent(sqlite_store(tmp_path))
    engine.store.claim("order", "original-hash", engine.ttl_seconds)

    claims = engine.claim_many([("order", "different-hash")])

    assert claims["order"].won is False
    assert claims["order"].record is not None
    assert claims["order"].record.request_hash == "original-hash"


def test_duplicate_keys_in_one_batch_are_refused(tmp_path):
    # Two payloads under one key is exactly what the request hash exists to
    # catch, and a dict keyed by key could only report one of them.
    engine = Idempotent(sqlite_store(tmp_path))
    with pytest.raises(ValueError, match="duplicate keys"):
        engine.claim_many([("same", "h1"), ("same", "h2")])


def test_the_fallback_path_gives_the_same_answers(tmp_path, monkeypatch):
    # A SQLite too old for RETURNING must still be correct, just slower.
    engine = Idempotent(sqlite_store(tmp_path / "fallback"))
    reference = Idempotent(sqlite_store(tmp_path / "reference"))
    engine.store.claim("held", "original", engine.ttl_seconds)
    reference.store.claim("held", "original", reference.ttl_seconds)

    items = [("fresh", "h"), ("held", "original")]
    with_returning = reference.claim_many(items)

    monkeypatch.setattr(SqliteStore, "_supports_returning", staticmethod(lambda: False))
    without_returning = engine.claim_many(items)

    assert {k: c.won for k, c in without_returning.items()} == {
        k: c.won for k, c in with_returning.items()
    }


def test_an_empty_batch_is_not_an_error(tmp_path):
    engine = Idempotent(sqlite_store(tmp_path))
    assert engine.claim_many([]) == {}


def test_keys_come_back_in_the_callers_namespace(tmp_path):
    # A caller that passed `order_1` must not get `tenant:order_1` back, or the
    # key it holds cannot be passed to anything else.
    engine = Idempotent(sqlite_store(tmp_path), namespace="tenant")
    claims = engine.claim_many([("order_1", "h")])

    assert set(claims) == {"order_1"}
    assert claims["order_1"].record is not None
    assert claims["order_1"].record.key == "order_1"
