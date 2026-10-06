"""Reconciling UNKNOWN records against a provider (#23)."""

from __future__ import annotations

import pytest

from justonce.core import Idempotent
from justonce.reconcile import Action, Outcome, build_plan
from justonce.stores.base import State
from justonce.stores.memory import MemoryStore


@pytest.fixture
def engine():
    return Idempotent(store=MemoryStore())


def make_unknown(engine, *keys):
    """Produce UNKNOWN the way production does: effect ran, outcome write died."""
    store = engine.store
    real = store.complete

    def exploding(*_: object, **__: object) -> None:
        raise RuntimeError("process died before the outcome was written")

    store.complete = exploding  # type: ignore[method-assign]
    for key in keys:
        with pytest.raises(RuntimeError):
            engine.run(key, lambda: {"charged": True})
    store.complete = real  # type: ignore[method-assign]


class Provider:
    """Answers from a dict; records what it was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def outcome_for(self, record):
        self.asked.append(record.key)
        return self.answers[record.key]


def test_planning_writes_nothing(engine):
    """The whole reason plan and apply are separate methods."""
    make_unknown(engine, "a", "b")
    provider = Provider({"a": (Outcome.APPLIED, {"id": 1}), "b": (Outcome.NOT_APPLIED, None)})

    plan = engine.plan_reconciliation(provider)

    assert len(plan) == 2
    assert engine.lookup("a").state is State.UNKNOWN
    assert engine.lookup("b").state is State.UNKNOWN


def test_applied_completes_and_records_the_providers_response(engine):
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.APPLIED, {"provider_id": "ch_1"})})

    result = engine.apply_reconciliation(engine.plan_reconciliation(provider))

    assert result.completed == 1
    record = engine.lookup("a")
    assert record.state is State.SUCCEEDED
    assert record.response == {"provider_id": "ch_1"}


def test_not_applied_releases_the_key_rather_than_burning_it(engine):
    """The effect never landed, so a retry must be allowed to run it."""
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.NOT_APPLIED, None)})

    result = engine.apply_reconciliation(engine.plan_reconciliation(provider))

    assert result.released == 1
    assert engine.lookup("a") is None, "a released key must be free to claim again"

    ran = []
    engine.run("a", lambda: ran.append("charged"))
    assert ran == ["charged"]


def test_unresolved_is_never_written(engine):
    """"I cannot tell you" must not be recorded as either answer."""
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.UNRESOLVED, None)})

    plan = engine.plan_reconciliation(provider)
    assert plan.steps[0].action is Action.LEAVE

    result = engine.apply_reconciliation(plan)
    assert (result.completed, result.released) == (0, 0)
    assert engine.lookup("a").state is State.UNKNOWN


def test_a_raising_adapter_does_not_take_the_batch_down(engine):
    """One unreachable provider must not strand the other records."""
    make_unknown(engine, "a", "b")

    class Flaky:
        def outcome_for(self, record):
            if record.key == "a":
                raise ConnectionError("provider unreachable")
            return Outcome.APPLIED, {"id": 2}

    plan = engine.plan_reconciliation(Flaky())
    by_key = {step.key: step for step in plan.steps}

    assert by_key["a"].action is Action.ERROR
    assert "ConnectionError" in by_key["a"].error
    assert by_key["b"].action is Action.COMPLETE

    result = engine.apply_reconciliation(plan)
    assert result.completed == 1
    assert engine.lookup("a").state is State.UNKNOWN, "the errored key stays untouched"


def test_a_record_resolved_after_planning_is_skipped_not_overwritten(engine):
    """The owning process came back. Its fact must beat our inference."""
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.NOT_APPLIED, None)})
    plan = engine.plan_reconciliation(provider)

    # Between plan and apply, the original owner records the real outcome.
    engine.store.complete("a", {"the": "truth"})

    result = engine.apply_reconciliation(plan)

    assert result.stale == ["a"]
    assert result.released == 0
    record = engine.lookup("a")
    assert record.state is State.SUCCEEDED
    assert record.response == {"the": "truth"}


def test_a_failing_write_is_reported_and_the_run_continues(engine):
    make_unknown(engine, "a", "b")
    provider = Provider({
        "a": (Outcome.APPLIED, {"id": 1}),
        "b": (Outcome.APPLIED, {"id": 2}),
    })
    plan = engine.plan_reconciliation(provider)

    real = engine.store.complete

    def fail_on_a(key, response, **kwargs):
        if key == "a":
            raise RuntimeError("write failed")
        return real(key, response, **kwargs)

    engine.store.complete = fail_on_a  # type: ignore[method-assign]
    result = engine.apply_reconciliation(plan)
    engine.store.complete = real  # type: ignore[method-assign]

    assert result.completed == 1, "b must still have been written"
    assert [key for key, _ in result.failed] == ["a"]


def test_older_than_is_passed_through(engine):
    """A record that just went UNKNOWN may still be in flight."""
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.APPLIED, None)})

    assert len(engine.plan_reconciliation(provider, older_than=3600)) == 0
    assert provider.asked == [], "nothing should have been asked about"


def test_plan_counts_cover_every_action(engine):
    make_unknown(engine, "a")
    provider = Provider({"a": (Outcome.UNRESOLVED, None)})
    counts = engine.plan_reconciliation(provider).counts()
    assert set(counts) == set(Action)
    assert counts[Action.LEAVE] == 1
    assert counts[Action.COMPLETE] == 0


def test_build_plan_is_pure(engine):
    """No store at all — planning only needs records and a provider."""
    make_unknown(engine, "a")
    records = engine.unresolved()
    plan = build_plan(records, Provider({"a": (Outcome.APPLIED, {"id": 1})}))
    assert plan.steps[0].action is Action.COMPLETE
    assert engine.lookup("a").state is State.UNKNOWN
