"""`Idempotency-Key` middleware for ASGI applications (#10).

The FastAPI example in `examples/fastapi_payment.py` shows the pattern a handler
at a time: read the header, derive a key, wrap the call. That is fine for one
endpoint and turns into copied boilerplate at ten, with the interesting part —
what happens to the *HTTP response* on a replay — re-decided each time.

    from justonce.integrations.asgi import IdempotencyMiddleware
    from justonce.stores import SqliteStore

    app.add_middleware(IdempotencyMiddleware, store=SqliteStore("effects.db"))

This is plain ASGI, not Starlette's `BaseHTTPMiddleware`, so it imports nothing
outside the standard library and works the same under FastAPI, Starlette,
Litestar or a hand-written app. `add_middleware` above is FastAPI's way of
mounting it; `app = IdempotencyMiddleware(app, store=...)` works anywhere.

What a replay returns is the whole stored response — status, body, and the
headers you allowed — plus `Idempotent-Replay: true` so the client can tell.

**Server errors are deliberately not cached.** `cacheable` defaults to
`status < 500`, and a response that fails it releases the claim rather than
recording it, so the retry the client is about to send actually runs. Caching a
500 for the retention window would turn one bad minute into a key that can
never succeed — the opposite of what an idempotency key is for.

Two limits worth knowing before you mount this:

* The response body is **buffered** to be stored, so a streaming or
  server-sent-events endpoint loses its streaming. Scope the middleware with
  `methods` and `should_handle` to the routes that mutate state, which are the
  ones that need it anyway.
* Only the headers in `store_headers` survive a replay, `content-type` by
  default. This is a denylist-free design on purpose: `set-cookie`,
  `authorization` and anything else a response carries would otherwise be
  written to the ledger and handed to whoever replays the key next.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

from ..asyncio import AsyncIdempotent
from ..errors import KeyReuseError, OperationInFlightError
from ..keys import operation_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..asyncio import AsyncStore
    from ..stores.base import Store

__all__ = ["DEFAULT_METHODS", "IdempotencyMiddleware"]

#: Methods handled unless you say otherwise. GET and HEAD are absent because a
#: safe request has nothing to deduplicate, and buffering their responses would
#: cost every read in the application.
DEFAULT_METHODS = ("POST", "PUT", "PATCH", "DELETE")

REPLAY_HEADER = b"idempotent-replay"


class _Uncacheable(Exception):
    """Carries a response the engine must not record.

    Raised out of the effect on purpose: it is how the claim gets released, so
    the client's retry is allowed to run the handler again. Never escapes
    `__call__`.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__("response is not cacheable")
        self.response = response


class IdempotencyMiddleware:
    """Deduplicate mutating requests by their `Idempotency-Key` header.

    Args:
        app: the ASGI application to wrap.
        store: a `Store` or `AsyncStore`. A sync store is moved to a thread by
            `AsyncIdempotent`, so it will not block the event loop. Ignored if
            `engine` is given.
        engine: a pre-built `AsyncIdempotent`, when you want to share one with
            the rest of the application or configure retention, hooks or a codec.
        header: the header to read. Case-insensitive, as ASGI lowercases them.
        methods: which methods to handle.
        require: whether a handled request *must* carry the header. `False`
            passes an unkeyed request straight through, which is what lets you
            mount this on an existing application without breaking its clients.
            `True` rejects it with `400`, for an API that has promised the
            header is mandatory.
        should_handle: extra predicate on the ASGI scope, for scoping by path.
        cacheable: given the status code, whether this response may be recorded.
        store_headers: response headers kept and replayed, lowercase.
        key: builds the idempotency key from `(method, path, header_value)`.
            The default scopes by endpoint, so one key reused against two routes
            is two independent operations rather than a replay of the first.
    """

    def __init__(
        self,
        app: Callable[..., Any],
        *,
        store: Store | AsyncStore | None = None,
        engine: AsyncIdempotent | None = None,
        header: str = "Idempotency-Key",
        methods: Iterable[str] = DEFAULT_METHODS,
        require: bool = False,
        should_handle: Callable[[dict[str, Any]], bool] | None = None,
        cacheable: Callable[[int], bool] = lambda status: status < 500,
        store_headers: Iterable[str] = ("content-type",),
        key: Callable[[str, str, str], str] | None = None,
    ) -> None:
        if (store is None) == (engine is None):
            raise ValueError("pass exactly one of store= or engine=")
        self.app = app
        self.engine = engine if engine is not None else AsyncIdempotent(store)  # type: ignore[arg-type]
        # Matched lowercase because ASGI lowercases header names, but reported
        # back in the spelling the caller configured — an error that renames
        # the header reads like a different header.
        self.header_name = header
        self.header = header.lower().encode("latin-1")
        self.methods = {method.upper() for method in methods}
        self.require = require
        self.should_handle = should_handle
        self.cacheable = cacheable
        self.store_headers = {name.lower().encode("latin-1") for name in store_headers}
        self.key = key if key is not None else _default_key

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method", "").upper() not in self.methods:
            await self.app(scope, receive, send)
            return
        if self.should_handle is not None and not self.should_handle(scope):
            await self.app(scope, receive, send)
            return

        supplied = _header(scope, self.header)
        if supplied is None:
            if self.require:
                await _send(send, _problem(400, f"{self.header_name} header is required"))
                return
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        key = self.key(scope.get("method", ""), scope.get("path", ""), supplied)
        # The body is fingerprinted, never stored: this is what catches the same
        # key arriving for a different request, and `fingerprint` hashes it.
        payload = {"method": scope.get("method"), "path": scope.get("path"), "body": _text(body)}

        # The live response is sent as the handler wrote it, headers and all.
        # Only the copy that goes to the ledger is filtered, because a
        # middleware that strips a `Set-Cookie` from a first, non-replayed
        # response has broken the endpoint rather than protected anyone.
        live: dict[str, Any] = {}

        async def effect() -> dict[str, Any]:
            captured = await _call_capturing(self.app, scope, body)
            live.update(_encoded(captured, keep=None))
            storable = _encoded(captured, keep=self.store_headers)
            if not self.cacheable(captured["status"]):
                raise _Uncacheable(live)
            return storable

        try:
            result = await self.engine.run(key, effect, payload=payload)
        except _Uncacheable as uncacheable:
            await _send(send, uncacheable.response)
            return
        except OperationInFlightError as exc:
            await _send(send, _problem(409, str(exc)))
            return
        except KeyReuseError as exc:
            await _send(send, _problem(422, str(exc)))
            return

        if result.executed:
            await _send(send, live)
            return

        response = dict(result.value)
        response["headers"] = [*response.get("headers", []), (REPLAY_HEADER, b"true")]
        await _send(send, response)


def _default_key(method: str, path: str, supplied: str) -> str:
    return operation_key("http", method.upper(), path, supplied)


def _header(scope: dict[str, Any], name: bytes) -> str | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() == name:
            value = raw_value.decode("latin-1").strip()
            # An empty header is not a key. Treating it as one would collapse
            # every such request onto a single key and dedupe unrelated calls.
            return value or None
    return None


async def _read_body(receive: Any) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _call_capturing(
    app: Callable[..., Any],
    scope: dict[str, Any],
    body: bytes,
) -> dict[str, Any]:
    """Run the downstream app on a replayed body and collect its response."""
    captured: dict[str, Any] = {"status": 500, "headers": [], "body": b""}
    chunks: list[bytes] = []
    sent = False

    async def replay_receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            # The app asked for more after the single buffered message. ASGI
            # allows this; answering with a disconnect is how a consumed body
            # ends rather than hanging the handler.
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def capture(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
            captured["headers"] = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, replay_receive, capture)
    captured["body"] = b"".join(chunks)
    return captured


def _encoded(response: dict[str, Any], *, keep: set[bytes] | None) -> dict[str, Any]:
    """JSON-safe, so every bundled store can hold it without a custom codec.

    `keep=None` means every header, for the response going back to this caller.
    A set means only those, for the copy that is written to the ledger and
    handed to whoever replays the key later.
    """
    headers = [
        [name.decode("latin-1"), value.decode("latin-1")]
        for name, value in response["headers"]
        if keep is None or name.lower() in keep
    ]
    return {
        "status": response["status"],
        "headers": headers,
        "body_b64": base64.b64encode(response["body"]).decode("ascii"),
    }


def _decoded_headers(response: dict[str, Any]) -> list[tuple[bytes, bytes]]:
    out = []
    for name, value in response.get("headers", []):
        if isinstance(name, bytes):
            out.append((name, value))
        else:
            out.append((name.encode("latin-1"), value.encode("latin-1")))
    return out


async def _send(send: Any, response: dict[str, Any]) -> None:
    body = base64.b64decode(response["body_b64"])
    headers = _decoded_headers(response)
    # Recomputed rather than stored: a stale content-length from the original
    # response would truncate or hang the replay.
    headers = [(name, value) for name, value in headers if name.lower() != b"content-length"]
    headers.append((b"content-length", str(len(body)).encode("ascii")))
    await send({"type": "http.response.start", "status": response["status"], "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _problem(status: int, detail: str) -> dict[str, Any]:
    body = json.dumps({"detail": detail}).encode("utf-8")
    return {
        "status": status,
        "headers": [["content-type", "application/json"]],
        "body_b64": base64.b64encode(body).decode("ascii"),
    }


def _text(body: bytes) -> str:
    """A stable text form of the body for fingerprinting only."""
    return base64.b64encode(body).decode("ascii")
