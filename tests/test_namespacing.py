"""Multi-tenant key namespacing (#53).

`operation_key("charge", order_id)` is global to the store. That is fine while
order ids are globally unique and quietly wrong the moment they are not: two
merchants with their own sequential numbering collide, one tenant's payment is
deduplicated against another's, and the effect simply never runs. Silent, and
in a payments system, serious.

Three things have to hold, and two of them are easy to get wrong in a way that
reintroduces the exact bug being fixed:

  * **The delimiter must not be forgeable.** If `namespace="a:b"` + key `"c"`
    and `namespace="a"` + key `"b:c"` both produce `"a:b:c"`, the fix has built
    a new way for two tenants to collide.
  * **Reads must be scoped too.** An engine that namespaces writes but not
    `unresolved()` hands tenant A tenant B's reconciliation queue — which in
    this library is a list of "we do not know whether this customer was
    charged".
  * **A scoped read must not under-report.** Fetching `limit` rows and then
    filtering returns fewer than asked for while more exist, so a reconciliation
    worker walking a page at a time silently skips tenants.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce import Idempotent, StoreError, operation_key
from justonce.asyncio import AsyncIdempotent
from justonce.stores import MemoryStore, SqliteStore


def unresolving_store(base: type = MemoryStore) -> Any:
    """A store whose completion always fails, so every run lands in UNKNOWN."""

    class CompleteFails(base):  # type: ignore[valid-type,misc]
        def complete(self, key: str, response: Any, **kw: Any) -> None:
            raise StoreError("lost mid-write")

    return CompleteFails()


def strand(engine: Idempotent, key: str) -> None:
    with pytest.raises(StoreError):
        engine.run(key, lambda: "ok")


class TestTenantsDoNotCollide:
    def test_the_same_key_in_two_namespaces_runs_twice(self) -> None:
        """The bug from the issue: two merchants, one order number each."""
        store = MemoryStore()
        a = Idempotent(store, namespace="merchant-a")
        b = Idempotent(store, namespace="merchant-b")
        calls: list[str] = []

        key = operation_key("charge", "1001")
        a.run(key, lambda: calls.append("a"))
        b.run(key, lambda: calls.append("b"))

        assert calls == ["a", "b"], "one tenant's charge was deduplicated against another's"

    def test_within_one_namespace_dedup_still_works(self) -> None:
        store = MemoryStore()
        a = Idempotent(store, namespace="merchant-a")
        calls: list[str] = []

        a.run("k", lambda: calls.append("first"))
        a.run("k", lambda: calls.append("second"))

        assert calls == ["first"]

    def test_no_namespace_is_the_default_and_unchanged(self) -> None:
        store = MemoryStore()
        plain = Idempotent(store)
        plain.run("k", lambda: "ok")
        assert store.lookup("k") is not None, "an unnamespaced engine must store the bare key"


class TestTheDelimiterIsNotForgeable:
    """The failure mode the fix could introduce if the prefix were just concatenated."""

    def test_a_namespace_containing_the_delimiter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            Idempotent(MemoryStore(), namespace="merchant:a")

    def test_the_ambiguous_pair_cannot_be_constructed(self) -> None:
        """`ns="a:b" + key="c"` and `ns="a" + key="b:c"` would both make `a:b:c`.

        Only one of the two is now constructible, so the collision has no
        second half.
        """
        with pytest.raises(ValueError):
            Idempotent(MemoryStore(), namespace="a:b")

        store = MemoryStore()
        Idempotent(store, namespace="a").run("b:c", lambda: "ok")
        assert store.lookup("a:b:c") is not None

    def test_one_namespace_that_prefixes_another_does_not_match_it(self) -> None:
        """`"merchant"` must not claim `"merchant-a"`'s rows.

        A membership test written as `key.startswith(namespace)` passes every
        test above — `merchant-a` and `merchant-b` are not prefixes of each
        other, so nothing there can tell the two implementations apart. It fails
        here, which is the shape that matters: the delimiter is what makes the
        boundary a boundary, and an allowlist anchored to a bare prefix is the
        same over-broad-match bug in a different costume.
        """
        store = unresolving_store()
        broad = Idempotent(store, namespace="merchant")
        narrow = Idempotent(store, namespace="merchant-a")

        strand(narrow, "k1")

        assert narrow.unresolved() != []
        assert broad.unresolved() == [], "`merchant` claimed `merchant-a`'s record"

    def test_a_prefixing_namespace_does_not_dedup_against_its_neighbour(self) -> None:
        store = MemoryStore()
        calls: list[str] = []
        Idempotent(store, namespace="merchant").run("x", lambda: calls.append("broad"))
        Idempotent(store, namespace="merchant-a").run("x", lambda: calls.append("narrow"))
        assert calls == ["broad", "narrow"]

    @pytest.mark.parametrize("bad", ["", "   "], ids=["empty", "whitespace"])
    def test_an_empty_namespace_is_refused(self, bad: str) -> None:
        """`None` means "not namespaced"; `""` would silently mean the same
        thing while looking like a tenant, and `"" + ":" + key` would shift
        every key by one character instead."""
        with pytest.raises(ValueError, match="namespace"):
            Idempotent(MemoryStore(), namespace=bad)


class TestReadsAreScopedToo:
    def test_unresolved_does_not_leak_another_tenants_queue(self) -> None:
        store = unresolving_store()
        a = Idempotent(store, namespace="merchant-a")
        b = Idempotent(store, namespace="merchant-b")

        strand(a, "k1")
        strand(b, "k2")

        assert [r.key for r in a.unresolved()] == ["k1"]
        assert [r.key for r in b.unresolved()] == ["k2"]

    def test_unresolved_returns_keys_in_the_callers_keyspace(self) -> None:
        """Stripped, so a record can be fed straight back into `run()`."""
        store = unresolving_store()
        a = Idempotent(store, namespace="merchant-a")
        strand(a, "k1")

        record = a.unresolved()[0]
        assert record.key == "k1", "the namespace prefix must not leak to the caller"

    def test_an_unnamespaced_engine_sees_everything(self) -> None:
        """Including namespaced rows — it is the operator's view, not a tenant's."""
        store = unresolving_store()
        strand(Idempotent(store, namespace="merchant-a"), "k1")
        strand(Idempotent(store), "k2")

        assert len(Idempotent(store).unresolved()) == 2

    def test_the_gauge_is_scoped(self) -> None:
        store = unresolving_store()
        a = Idempotent(store, namespace="merchant-a")
        b = Idempotent(store, namespace="merchant-b")
        strand(a, "k1")

        assert a.oldest_unresolved_age() is not None
        assert b.oldest_unresolved_age() is None, "tenant b has nothing unresolved"


class TestAScopedReadDoesNotUnderReport:
    def test_limit_is_honoured_across_a_noisy_neighbour(self) -> None:
        """Fetch-then-filter is the naive implementation and it under-reports.

        Twenty of tenant B's records are written first, so a store returning
        oldest-first hands back only B's rows for the first page. Asking A for
        five must still yield five.
        """
        store = unresolving_store()
        noisy = Idempotent(store, namespace="merchant-b")
        quiet = Idempotent(store, namespace="merchant-a")

        for i in range(20):
            strand(noisy, f"noise-{i}")
        for i in range(5):
            strand(quiet, f"real-{i}")

        found = quiet.unresolved(limit=5)
        assert len(found) == 5, f"scoped read under-reported: got {len(found)}"
        assert all(r.key.startswith("real-") for r in found)

    def test_limit_is_still_a_ceiling(self) -> None:
        store = unresolving_store()
        a = Idempotent(store, namespace="merchant-a")
        for i in range(10):
            strand(a, f"k-{i}")

        assert len(a.unresolved(limit=3)) == 3


class TestAcrossARealStore:
    def test_sqlite_keeps_tenants_apart(self) -> None:
        store = SqliteStore(":memory:")
        a = Idempotent(store, namespace="merchant-a")
        b = Idempotent(store, namespace="merchant-b")
        calls: list[str] = []

        key = operation_key("charge", "1001")
        a.run(key, lambda: calls.append("a"))
        b.run(key, lambda: calls.append("b"))
        a.run(key, lambda: calls.append("a-again"))

        assert calls == ["a", "b"]


class TestAsyncParity:
    async def test_tenants_do_not_collide(self) -> None:
        store = MemoryStore()
        a = AsyncIdempotent(store, namespace="merchant-a")
        b = AsyncIdempotent(store, namespace="merchant-b")
        calls: list[str] = []

        async def eff(tag: str) -> str:
            calls.append(tag)
            return tag

        await a.run("k", lambda: eff("a"))
        await b.run("k", lambda: eff("b"))
        assert calls == ["a", "b"]

    async def test_async_refuses_a_forgeable_namespace(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            AsyncIdempotent(MemoryStore(), namespace="a:b")

    async def test_async_unresolved_is_scoped_and_stripped(self) -> None:
        store = unresolving_store()
        a = AsyncIdempotent(store, namespace="merchant-a")
        b = AsyncIdempotent(store, namespace="merchant-b")

        async def eff() -> str:
            return "ok"

        with pytest.raises(StoreError):
            await a.run("k1", eff)
        with pytest.raises(StoreError):
            await b.run("k2", eff)

        assert [r.key for r in await a.unresolved()] == ["k1"]
