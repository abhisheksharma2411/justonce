"""Celery integration — the at-least-once delivery this library exists for.

Driven with ``task_always_eager`` so the suite needs no broker. Eager mode runs
the task body in-process, which is all these tests need: the thing under test
is what happens when the *body* runs twice, and a real broker would only be a
slower way to arrange that.
"""

from __future__ import annotations

import pytest

celery = pytest.importorskip("celery", reason="pip install justonce[celery]")

from celery import Celery  # noqa: E402
from celery.exceptions import Retry  # noqa: E402

from justonce.core import Idempotent  # noqa: E402
from justonce.integrations.celery import (  # noqa: E402
    idempotent_task,
    register_check,
    task_delivery_key,
)
from justonce.keys import operation_key  # noqa: E402
from justonce.stores import MemoryStore  # noqa: E402


@pytest.fixture()
def app() -> Celery:
    application = Celery("tests")
    application.conf.task_always_eager = True
    application.conf.task_eager_propagates = True
    return application


@pytest.fixture()
def engine() -> Idempotent:
    return Idempotent(MemoryStore())


def test_a_redelivered_task_body_runs_once(app: Celery, engine: Idempotent) -> None:
    """The whole point: the broker delivers twice, the effect happens once."""
    runs = []

    @app.task(name="charge")
    @idempotent_task(key=lambda order_id: operation_key("charge", order_id), engine=engine)
    def charge(order_id: str) -> str:
        runs.append(order_id)
        return f"receipt-{order_id}"

    first = charge.apply(args=("order-1",)).get()
    second = charge.apply(args=("order-1",)).get()

    assert runs == ["order-1"]
    assert first == "receipt-order-1"
    assert second == first, "the replay must return the recorded value, not None"


def test_a_different_order_is_not_deduplicated(app: Celery, engine: Idempotent) -> None:
    """The key is the intent; two intents are two effects."""
    runs = []

    @app.task(name="charge2")
    @idempotent_task(key=lambda order_id: operation_key("charge", order_id), engine=engine)
    def charge(order_id: str) -> str:
        runs.append(order_id)
        return order_id

    charge.apply(args=("order-1",)).get()
    charge.apply(args=("order-2",)).get()

    assert runs == ["order-1", "order-2"]


def test_a_raising_body_releases_the_key(app: Celery, engine: Idempotent) -> None:
    """A failed effect must not burn the key.

    Celery owns the retry decision. If this held or burned the key, the retry
    Celery schedules would arrive, find the key taken, and never run the body
    -- the task would look retried and would not be.
    """
    attempts = []

    @app.task(name="flaky")
    @idempotent_task(key=lambda: operation_key("flaky", "k"), engine=engine)
    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("provider down")
        return "ok"

    with pytest.raises(RuntimeError):
        flaky.apply().get()

    assert flaky.apply().get() == "ok"
    assert len(attempts) == 2


def test_self_retry_does_not_block_its_own_retry(app: Celery, engine: Idempotent) -> None:
    """`self.retry()` keeps the task id, so it asks for the key it is holding.

    This is the case that makes a naive wrapper deadlock against itself: the
    retry arrives with the same delivery key, finds it claimed, and either
    fails or replays a result that was never produced. Driven through the real
    `Retry` exception rather than a stand-in, because `Retry` is what Celery
    actually raises and whether it is treated as a failure is the question.
    """
    bodies = []

    @app.task(bind=True, name="retrying", max_retries=1)
    @idempotent_task(key=lambda self: operation_key("retrying", "k"), engine=engine)
    def retrying(self) -> str:  # type: ignore[no-untyped-def]
        bodies.append(1)
        if len(bodies) == 1:
            raise Retry("try again")
        return "done"

    with pytest.raises(Retry):
        retrying.apply().get()

    # The key must be free again, or this second body never runs.
    assert retrying.apply().get() == "done"
    assert len(bodies) == 2


def test_the_arguments_are_the_default_fingerprint(app: Celery, engine: Idempotent) -> None:
    """Same key, changed arguments, is the caller reusing a key for a new intent."""
    from justonce.errors import KeyReuseError

    @app.task(name="fixedkey")
    @idempotent_task(key=lambda amount: operation_key("fixed", "k"), engine=engine)
    def fixed(amount: int) -> int:
        return amount

    assert fixed.apply(args=(100,)).get() == 100
    with pytest.raises(KeyReuseError):
        fixed.apply(args=(250,)).get()


def test_delivery_key_is_the_default_and_names_the_task(app: Celery, engine: Idempotent) -> None:
    """Without key=, the key comes from Celery's own task id."""
    seen = []

    @app.task(name="delivery")
    @idempotent_task(engine=engine)
    def delivery() -> str:
        seen.append(task_delivery_key())
        return "ok"

    delivery.apply().get()

    assert len(seen) == 1
    assert "celery" in seen[0] and "delivery" in seen[0]


def test_delivery_key_outside_a_task_is_refused(engine: Idempotent) -> None:
    """Rather than keying on nothing and deduplicating everything together."""
    with pytest.raises(RuntimeError, match="running Celery task"):
        task_delivery_key()


def test_register_check_names_an_undecorated_task(app: Celery, engine: Idempotent) -> None:
    """The decorator-order trap, which is otherwise completely silent.

    Put `@idempotent_task` above `@app.task` and Celery registers the bare
    function: every task runs exactly as before and nothing reports it.
    """

    @app.task(name="guarded")
    @idempotent_task(key=lambda: operation_key("g", "k"), engine=engine)
    def guarded() -> str:
        return "ok"

    @app.task(name="unguarded")
    def unguarded() -> str:
        return "ok"

    unwrapped = register_check(app)

    assert "unguarded" in unwrapped
    assert "guarded" not in unwrapped
    assert not any(name.startswith("celery.") for name in unwrapped), (
        "Celery's own internal tasks are not the application's to decorate"
    )


def test_register_check_catches_the_wrong_decorator_order(app: Celery, engine: Idempotent) -> None:
    """The specific mistake, arranged the way someone would actually make it."""

    # The wrong way round. @app.task runs first and registers the bare
    # function; @idempotent_task then wraps the Task object that comes back, so
    # the guard sits outside Celery entirely and the registered body is
    # undecorated. Nothing raises, and every task keeps working -- unguarded.
    @idempotent_task(key=lambda: operation_key("w", "k"), engine=engine)
    @app.task(name="wrong_order")
    def wrong_order() -> str:
        return "ok"

    assert "wrong_order" in register_check(app)

    # and the right way round is not flagged, so the check is not just
    # reporting everything
    @app.task(name="right_order")
    @idempotent_task(key=lambda: operation_key("r", "k"), engine=engine)
    def right_order() -> str:
        return "ok"

    assert "right_order" not in register_check(app)
