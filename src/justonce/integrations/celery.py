"""Celery integration (#11).

    from celery import Celery
    from justonce.integrations.celery import idempotent_task
    from justonce.keys import operation_key
    from justonce.stores import SqliteStore

    app = Celery(broker="redis://localhost")
    justonce.configure(SqliteStore("effects.db"))

    @app.task(acks_late=True)
    @idempotent_task(key=lambda order_id: operation_key("charge", order_id))
    def charge(order_id):
        return provider.charge(order_id)

A Celery task is delivered at least once, by design. With ``acks_late`` — which
is the setting you want if losing work matters — a worker that dies mid-task
hands the message back to the broker and another worker runs the body again.
The visibility timeout does the same thing to a task that simply takes longer
than expected. Celery has no opinion about whether running twice is safe; that
is the application's problem, and this is the application's answer.

**Decorator order matters.** ``@idempotent_task`` goes *under* ``@app.task``, so
that Celery registers the wrapped function. The other way round registers the
bare function and nothing is deduplicated — the common way to get this wrong,
and the reason `register_check` exists below.

**The module is named `celery` and imports Celery.** Python 3 resolves ``import
celery`` absolutely, so this is the real package, not itself.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from ..core import Idempotent
from ..keys import operation_key

try:  # pragma: no cover - import guard
    from celery import current_task
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the Celery integration requires celery. Install with: pip install justonce[celery]"
    ) from exc

__all__ = ["idempotent_task", "task_delivery_key"]


def task_delivery_key() -> str:
    """Key one *delivery* of the current task, from Celery's own task id.

    The default when no `key=` is given, and the weaker of the two guarantees.
    It deduplicates redeliveries of one message — the `acks_late` crash and the
    visibility-timeout case — because Celery keeps the task id across those.

    It does **not** deduplicate two publishes of the same intent: `charge.delay(
    order)` called twice gets two task ids and runs twice. If what you need is
    "this order is charged once however many times anything enqueues it", pass
    `key=` and build it from the order, not from the delivery.

    Raises:
        RuntimeError: called outside a task, where there is no id to use.
    """
    request = getattr(current_task, "request", None)
    task_id = getattr(request, "id", None)
    if not task_id:
        raise RuntimeError(
            "task_delivery_key() needs a running Celery task. Outside one there "
            "is no delivery to key on — pass key= instead."
        )
    name = getattr(current_task, "name", "task")
    return operation_key("celery", name, task_id)


def idempotent_task(
    func: Callable[..., Any] | None = None,
    *,
    key: Callable[..., str] | None = None,
    payload: Callable[..., Any] | None = None,
    engine: Idempotent | None = None,
    ttl_seconds: float | None = None,
    retention_seconds: float | None = None,
) -> Any:
    """Run this task's body at most once per key.

    A later delivery with the same key does not run the body; it returns the
    recorded return value, so `AsyncResult.get()` sees what the first run
    produced rather than `None`.

    Args:
        key: builds the key from the task's own arguments. Omit it to key on
            the delivery — see `task_delivery_key` for what that does and does
            not promise.
        payload: fingerprinted to catch the same key used for a changed intent.
            Defaults to the task's arguments, which is almost always right.
        engine: an `Idempotent`. Defaults to the one `justonce.configure` set.
        ttl_seconds: claim lease for this task. Should exceed the task's
            realistic runtime: a lease that expires while the body is still
            running is a lease another worker can take.
        retention_seconds: how long the result stays replayable.

    **There is deliberately no "burn the key on failure" option.** Celery owns
    the retry decision — `max_retries`, `autoretry_for`, the backoff, the
    dead-letter queue — and a key burned here would silently disable all of it:
    the retry Celery dutifully schedules would arrive, find a terminal record,
    and return the failure without ever running. So a raising body always
    releases the claim, and a task that must stop retrying stops it with
    Celery's own mechanisms, where the policy is visible.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = engine if engine is not None else _default_engine()
            effect_key = key(*args, **kwargs) if key is not None else task_delivery_key()
            fingerprint = payload(*args, **kwargs) if payload is not None else [args, kwargs]

            # retry_on_failure=True covers Celery's control-flow exceptions
            # without naming them. `self.retry()` raises `Retry`, which leaves
            # the body like any other exception and so releases the claim --
            # which is exactly what it needs to do, because `self.retry()`
            # keeps the same task id: a retry of a delivery-keyed task asks for
            # the very key this call is holding, and holding it means the retry
            # finds the key taken and cannot run.
            result = active.run(
                effect_key,
                lambda: target(*args, **kwargs),
                payload=fingerprint,
                retry_on_failure=True,
                ttl_seconds=ttl_seconds,
                retention_seconds=retention_seconds,
            )
            return result.value

        wrapper.justonce_idempotent = True  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


def _default_engine() -> Idempotent:
    from ..decorators import get_default

    return get_default()


def register_check(app: Any) -> list[str]:
    """Names of this app's tasks whose body is *not* wrapped by this decorator.

    The failure this exists for is silent. `@idempotent_task` has to sit under
    `@app.task`; put it above and Celery registers the undecorated function,
    every task runs exactly as it did before, and nothing anywhere says so.
    Call this at startup and log or raise on what comes back.

    Returns the task names in registration order. Celery's own internal tasks
    (``celery.*``) are excluded; they are not yours to decorate.
    """
    unwrapped = []
    for name, task in sorted(app.tasks.items()):
        if name.startswith("celery."):
            continue
        run = getattr(task, "run", None)
        if not getattr(run, "justonce_idempotent", False):
            unwrapped.append(name)
    return unwrapped
