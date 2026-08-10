"""The `@idempotent` decorator.

One line at the call site is the entire onboarding story:

    @idempotent(key=lambda order: operation_key("charge", order.id))
    def charge_customer(order):
        return stripe.charge(order.customer, order.total)

Everything else in the library is optional depth.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from .core import Idempotent, Result
from .stores.base import Store

F = TypeVar("F", bound=Callable[..., Any])

_default: Idempotent | None = None


def configure(store: Store, **kwargs: Any) -> Idempotent:
    """Set the engine used by `@idempotent` when none is passed explicitly.

    Call once at startup. Passing `engine=` to the decorator overrides it, which
    is what you want in tests and in libraries that should not depend on a
    process-global.
    """
    global _default
    _default = Idempotent(store, **kwargs)
    return _default


def get_default() -> Idempotent:
    if _default is None:
        raise RuntimeError(
            "justonce is not configured; call justonce.configure(store) at "
            "startup, or pass engine= to @idempotent"
        )
    return _default


def idempotent(
    *,
    key: Callable[..., str],
    payload: Callable[..., Any] | None = None,
    engine: Idempotent | None = None,
    retry_on_failure: bool = True,
    return_result: bool = False,
) -> Callable[[F], F]:
    """Make a side-effecting function run at most once per derived key.

    Args:
        key: derives the idempotency key from the call arguments. Must depend
            only on the *intent* — never on the attempt. A `uuid4()` or
            `time.time()` in here defeats the entire mechanism.
        payload: derives the request body used for the divergence guard.
            Defaults to the call arguments, which catches the common case of a
            key reused with different values.
        engine: engine to use; defaults to the one from `configure()`.
        retry_on_failure: whether a raised effect leaves the key retryable.
        return_result: return the full `Result` (with `executed`) instead of
            just the value. Useful when the caller needs to know whether this
            call was the one that did the work.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            eng = engine or get_default()
            derived = key(*args, **kwargs)
            body = payload(*args, **kwargs) if payload else _default_payload(args, kwargs)
            result: Result = eng.run(
                derived,
                lambda: func(*args, **kwargs),
                payload=body,
                retry_on_failure=retry_on_failure,
            )
            return result if return_result else result.value

        wrapper.__justonce_key__ = key  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def _default_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Fingerprint the call arguments, skipping `self`-like non-serialisables."""
    return {
        "args": [_safe(a) for a in args],
        "kwargs": {k: _safe(v) for k, v in sorted(kwargs.items())},
    }


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    # Objects: prefer an explicit, stable identity over repr(), which embeds a
    # memory address and would make every attempt look like a different payload.
    for attr in ("id", "pk", "uuid"):
        if hasattr(value, attr):
            return f"{type(value).__name__}:{getattr(value, attr)}"
    return type(value).__name__
