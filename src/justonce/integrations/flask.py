"""Flask integration (#13).

    from flask import Flask
    from justonce.integrations.flask import JustOnce, idempotent_view
    from justonce.stores import SqliteStore

    app = Flask(__name__)
    JustOnce(app, store=SqliteStore("effects.db"))

    @app.post("/charges")
    @idempotent_view
    def create_charge():
        return {"id": charge(request.json)}, 201

Two pieces, on purpose. `JustOnce` puts one engine on the app so views do not
each build their own, and `idempotent_view` is opt-in per view — a decorator
rather than `before_request`, because an application that deduplicates every
route is almost never what anyone wants and is very hard to notice going wrong.

Unlike the ASGI middleware, which wraps a whole application and therefore
defaults to passing unkeyed requests through, a decorator here is an explicit
statement that *this* endpoint is not safe to run twice. So a missing header is
a `400` by default, and `require=False` is available for the migration case
where clients have not started sending one yet.

**The module is named `flask` and imports Flask.** Python 3 resolves `import
flask` absolutely, so this is the real package, not itself.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, cast

import flask
from flask import current_app, request

from ..core import Idempotent
from ..errors import KeyReuseError, OperationInFlightError
from ..keys import operation_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..stores.base import Store

__all__ = ["EXTENSION_KEY", "REPLAY_HEADER", "JustOnce", "idempotent_view"]

#: Where the extension stores itself, following Flask's convention.
EXTENSION_KEY = "justonce"

#: Set on a response that came from the ledger rather than from the view. The
#: same spelling the ASGI middleware uses, so a client talking to both halves
#: of a system sees one header.
REPLAY_HEADER = "Idempotent-Replay"


class _Uncacheable(Exception):
    """Carries a response the engine must not record.

    Raised out of the effect on purpose: that is how the claim is released, so
    the client's retry is allowed to run the view again. Never escapes
    `idempotent_view`.
    """

    def __init__(self, response: flask.Response) -> None:
        super().__init__("response is not cacheable")
        self.response = response


class JustOnce:
    """Flask extension holding one engine for the application.

    Args:
        app: the application, if you are not using the factory pattern.
        store: a `Store`. Ignored when `engine` is given.
        engine: a pre-built `Idempotent`, when you want to share one with the
            rest of the application or configure retention, hooks or a codec.
        header: the request header carrying the key.
        require: whether a decorated view *must* receive the header.
        cacheable: given the status code, whether the response may be recorded.
        store_headers: response headers recorded and replayed, lowercase.
        key: builds the key from `(method, path, header_value)`. The default
            scopes by endpoint, so one key reused against two routes is two
            operations rather than a replay of the first.
    """

    def __init__(
        self,
        app: flask.Flask | None = None,
        *,
        store: Store | None = None,
        engine: Idempotent | None = None,
        header: str = "Idempotency-Key",
        require: bool = True,
        cacheable: Callable[[int], bool] = lambda status: status < 500,
        store_headers: Iterable[str] = ("content-type",),
        key: Callable[[str, str, str], str] | None = None,
    ) -> None:
        if (store is None) == (engine is None):
            raise ValueError("pass exactly one of store= or engine=")
        self.engine = engine if engine is not None else Idempotent(cast("Store", store))
        self.header = header
        self.require = require
        self.cacheable = cacheable
        self.store_headers = {name.lower() for name in store_headers}
        self.key = key or _default_key
        if app is not None:
            self.init_app(app)

    def init_app(self, app: flask.Flask) -> None:
        """Register on an application. Safe to call for several apps."""
        app.extensions[EXTENSION_KEY] = self


def idempotent_view(view: Callable[..., Any]) -> Callable[..., Any]:
    """Deduplicate this view by its `Idempotency-Key` header.

    The view runs at most once per key. A later request with the same key is
    answered from the ledger with `Idempotent-Replay: true` and never reaches
    the view.
    """

    @functools.wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        extension = current_app.extensions.get(EXTENSION_KEY)
        if extension is None:
            raise RuntimeError(
                "idempotent_view needs the JustOnce extension: "
                "JustOnce(app, store=...) or JustOnce(app, engine=...)"
            )

        supplied = request.headers.get(extension.header)
        if not supplied:
            if extension.require:
                # The spelling the application configured, not the normalised
                # one -- a client told to send `X-Request-Key` should not be
                # told off about `x-request-key`.
                return _problem(400, f"{extension.header} header is required")
            return view(*args, **kwargs)

        key = extension.key(request.method.upper(), request.path, supplied)
        # The live response goes back exactly as the view wrote it, headers and
        # all. Only the copy bound for the ledger is filtered, because a
        # decorator that strips a `Set-Cookie` from a first, non-replayed
        # response has broken the endpoint rather than protected anyone.
        live: list[flask.Response] = []

        def effect() -> dict[str, Any]:
            response = flask.make_response(view(*args, **kwargs))
            live.append(response)
            if response.direct_passthrough:
                # A streamed or file response. `get_data()` on one either
                # raises or drains the iterator the client is about to be
                # handed, so there is nothing safe to record. Replaying a
                # stream is not something this can promise.
                raise _Uncacheable(response)
            if not extension.cacheable(response.status_code):
                raise _Uncacheable(response)
            return _recordable(response, extension.store_headers)

        try:
            result = extension.engine.run(key, effect, payload=_payload())
        except _Uncacheable as uncacheable:
            return uncacheable.response
        except OperationInFlightError as exc:
            return _problem(409, str(exc))
        except KeyReuseError as exc:
            return _problem(422, str(exc))

        if result.executed:
            return live[0]
        return _replayed(result.value)

    return wrapper


def _default_key(method: str, path: str, supplied: str) -> str:
    return operation_key("http", method, path, supplied)


def _payload() -> Any:
    """What the key is checked against, so the same key with a changed body is
    caught as reuse rather than replayed as if it were the first request.

    The raw bytes, decoded loosely: this is a fingerprint, not something anyone
    reads back, and a body that is not UTF-8 must still produce a stable one
    rather than raise inside the guard.
    """
    return request.get_data(cache=True).decode("utf-8", "replace")


def _recordable(response: flask.Response, keep: set[str]) -> dict[str, Any]:
    """The part of a response worth storing.

    Headers are filtered to `store_headers`. Everything else is per-exchange --
    `Set-Cookie`, `Date`, auth challenges -- and replaying a stale one is worse
    than omitting it.
    """
    return {
        "status": response.status_code,
        "headers": [
            [name, value] for name, value in response.headers.items() if name.lower() in keep
        ],
        "body": response.get_data(as_text=True),
    }


def _replayed(stored: Any) -> flask.Response:
    response = flask.make_response(stored["body"], stored["status"])
    # Replace rather than add: make_response has already guessed a content type
    # for the body, and the recorded one is what the first caller was told.
    for name, value in stored["headers"]:
        response.headers[name] = value
    response.headers[REPLAY_HEADER] = "true"
    return response


def _problem(status: int, detail: str) -> flask.Response:
    return flask.make_response(
        flask.jsonify({"title": _TITLES[status], "status": status, "detail": detail}),
        status,
    )


_TITLES = {
    400: "Missing idempotency key",
    409: "Operation in flight",
    422: "Idempotency key reused",
}
