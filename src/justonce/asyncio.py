"""Async engine.

`async def` effects are the common case in modern Python web stacks, and a
blocking claim in an event loop stalls every other request on that worker.

Two pieces:

  * `AsyncStore` — the async form of the store contract. Native async drivers
    (asyncpg, aiosqlite, motor) implement this directly.
  * `ThreadedStore` — wraps any sync `Store` and runs it on a worker thread, so
    every existing store works under the async engine today. Correctness is
    unchanged: the atomicity guarantee lives in the database, not in how we
    reach it.

The state machine is identical to the sync engine, deliberately. Two
implementations that drift are two sets of bugs.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable
from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from .core import DEFAULT_RETENTION_SECONDS, DEFAULT_TTL_SECONDS, OnInFlight, Result
from .errors import InFlightTimeout, KeyReuseError, OperationInFlightError
from .keys import fingerprint
from .stores.base import Claim, Record, State, Store

T = TypeVar("T")


@runtime_checkable
class AsyncStore(Protocol):
    """Async form of `justonce.stores.base.Store`.

    Same contract, same guarantees — in particular `claim` must still be a
    single atomic operation. Making it async does not make check-then-act safe;
    it just gives the race more opportunities to interleave.
    """

    async def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim: ...
    async def complete(
        self, key: str, response: Any, *, retention_seconds: float | None = None
    ) -> None: ...
    async def fail(
        self, key: str, *, terminal: bool, retention_seconds: float | None = None
    ) -> None: ...
    async def mark_unknown(self, key: str) -> None: ...
    async def lookup(self, key: str) -> Record | None: ...
    async def sweep(self, *, before: float) -> int: ...
    async def unresolved(
        self, *, older_than: float | None = None, limit: int = 100
    ) -> list[Record]: ...


class ThreadedStore:
    """Adapt a sync `Store` to `AsyncStore` by running it off the event loop.

    Every bundled store works under the async engine through this, with no new
    dependency and no second implementation to keep in step. A native async
    driver will be faster under load; this is correct today.

    The wrapped store must be safe to call from multiple threads — the bundled
    ones are, and the conformance suite proves it with a 24-thread claim race.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    @property
    def inner(self) -> Store:
        return self._store

    async def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        return await asyncio.to_thread(self._store.claim, key, request_hash, ttl_seconds)

    async def complete(
        self, key: str, response: Any, *, retention_seconds: float | None = None
    ) -> None:
        await asyncio.to_thread(
            functools.partial(
                self._store.complete, key, response, retention_seconds=retention_seconds
            )
        )

    async def fail(
        self, key: str, *, terminal: bool, retention_seconds: float | None = None
    ) -> None:
        await asyncio.to_thread(
            functools.partial(
                self._store.fail, key, terminal=terminal, retention_seconds=retention_seconds
            )
        )

    async def mark_unknown(self, key: str) -> None:
        await asyncio.to_thread(self._store.mark_unknown, key)

    async def lookup(self, key: str) -> Record | None:
        return await asyncio.to_thread(self._store.lookup, key)

    async def sweep(self, *, before: float) -> int:
        return await asyncio.to_thread(functools.partial(self._store.sweep, before=before))

    async def unresolved(
        self, *, older_than: float | None = None, limit: int = 100
    ) -> list[Record]:
        return await asyncio.to_thread(
            functools.partial(self._store.unresolved, older_than=older_than, limit=limit)
        )


class AsyncIdempotent:
    """Runs async effects at most once per key.

    Mirrors `justonce.core.Idempotent`. Accepts an `AsyncStore`, or a sync
    `Store` which is wrapped in `ThreadedStore` automatically.
    """

    def __init__(
        self,
        store: AsyncStore | Store,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        on_in_flight: OnInFlight = OnInFlight.RAISE,
        wait_timeout: float = 30.0,
        poll_interval: float = 0.05,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if _is_async_store(store):
            self.store: AsyncStore = cast("AsyncStore", store)
        else:
            self.store = ThreadedStore(cast("Store", store))
        self.ttl_seconds = ttl_seconds
        self.on_in_flight = on_in_flight
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.retention_seconds = retention_seconds

    async def run(
        self,
        key: str,
        effect: Callable[[], Awaitable[T]],
        *,
        payload: Any = None,
        retry_on_failure: bool = True,
    ) -> Result:
        """Run `effect` at most once for `key`. See `Idempotent.run`."""
        request_hash = fingerprint(payload)
        claim = await self.store.claim(key, request_hash, self.ttl_seconds)

        if claim.lost:
            return await self._resolve_loser(key, request_hash, claim.record)

        try:
            value = await effect()
        except BaseException:
            # Includes asyncio.CancelledError: a cancelled effect may have
            # already applied, so the caller's retry_on_failure choice governs
            # here exactly as it does for any other failure.
            await self.store.fail(
                key,
                terminal=not retry_on_failure,
                retention_seconds=self.retention_seconds,
            )
            raise

        try:
            await self.store.complete(key, value, retention_seconds=self.retention_seconds)
        except BaseException:
            # The effect DID happen and we could not record it. Leave the key
            # unresolved — releasing it would let a retry apply the effect twice.
            await self.store.mark_unknown(key)
            raise

        return Result(value=value, executed=True, record=await self.store.lookup(key))

    async def sweep(self, *, now: float | None = None) -> int:
        return await self.store.sweep(before=now if now is not None else time.time())

    async def unresolved(self, *, limit: int = 100) -> list[Record]:
        return await self.store.unresolved(limit=limit)

    # -- internals ----------------------------------------------------------

    async def _resolve_loser(
        self, key: str, request_hash: str, record: Record | None
    ) -> Result:
        if record is None:
            raise OperationInFlightError(key)
        if record.request_hash != request_hash:
            raise KeyReuseError(key)
        if record.state is State.SUCCEEDED:
            return Result(value=record.response, executed=False, record=record)
        if record.state is State.FAILED:
            return Result(value=None, executed=False, record=record)
        if record.state is State.UNKNOWN:
            raise OperationInFlightError(key)
        if self.on_in_flight is OnInFlight.RAISE:
            raise OperationInFlightError(key)
        return await self._wait_for(key, request_hash)

    async def _wait_for(self, key: str, request_hash: str) -> Result:
        deadline = time.monotonic() + self.wait_timeout
        while time.monotonic() < deadline:
            # asyncio.sleep, never time.sleep: blocking here would stall every
            # other request sharing this event loop.
            await asyncio.sleep(self.poll_interval)
            record = await self.store.lookup(key)
            if record is None:
                raise OperationInFlightError(key)
            if record.request_hash != request_hash:
                raise KeyReuseError(key)
            if record.is_terminal:
                return Result(
                    value=record.response if record.state is State.SUCCEEDED else None,
                    executed=False,
                    record=record,
                )
            if record.state is State.UNKNOWN:
                raise OperationInFlightError(key)
        raise InFlightTimeout(key, self.wait_timeout)


def _is_async_store(store: Any) -> bool:
    """True when `store.claim` is a coroutine function."""
    return asyncio.iscoroutinefunction(getattr(store, "claim", None))


_default: AsyncIdempotent | None = None


def configure_async(store: AsyncStore | Store, **kwargs: Any) -> AsyncIdempotent:
    """Set the engine used by `@async_idempotent` when none is passed."""
    global _default
    _default = AsyncIdempotent(store, **kwargs)
    return _default


def get_default_async() -> AsyncIdempotent:
    if _default is None:
        raise RuntimeError(
            "justonce async is not configured; call justonce.configure_async(store) "
            "at startup, or pass engine= to @async_idempotent"
        )
    return _default


def async_idempotent(
    *,
    key: Callable[..., str],
    payload: Callable[..., Any] | None = None,
    engine: AsyncIdempotent | None = None,
    retry_on_failure: bool = True,
    return_result: bool = False,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """`@idempotent` for `async def` effects.

        @async_idempotent(key=lambda order: operation_key("charge", order.id))
        async def charge_customer(order):
            return await payments.charge(order.customer, order.total)
    """

    def decorate(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"@async_idempotent expects an async def function; {func.__name__} is "
                "synchronous — use @idempotent instead"
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            eng = engine or get_default_async()
            from .decorators import _default_payload

            derived = key(*args, **kwargs)
            body = payload(*args, **kwargs) if payload else _default_payload(args, kwargs)
            result = await eng.run(
                derived,
                lambda: func(*args, **kwargs),
                payload=body,
                retry_on_failure=retry_on_failure,
            )
            return result if return_result else result.value

        wrapper.__justonce_key__ = key  # type: ignore[attr-defined]
        return wrapper

    return decorate
