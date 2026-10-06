"""Resolving `UNKNOWN` records against the system that actually knows (#23).

`UNKNOWN` is the state the library exists to keep: the effect was started and
the outcome was never observed. Nothing local can resolve it — the record says
"we do not know", and no amount of re-reading our own store turns that into an
answer. The only authority is the system the effect was applied to.

So reconciliation is a *question asked outward*, and this module is the shape of
that question. An adapter answers "what do you think happened for this key?" and
reconciliation writes the answer down.

**Planning and applying are separate on purpose.** A reconciliation tool that
acts before you have read its plan is not one anybody will run against
production, and the first thing an operator does with a new tool is check
whether it understood the situation. `plan()` is read-only; `apply()` takes a
plan it did not build.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .stores.base import Record, State


class Outcome(str, enum.Enum):
    """What the provider says happened.

    `UNRESOLVED` is not a failure of the adapter — it is a legitimate answer,
    and it is the one that must never be written down as either of the others.
    A provider that has genuinely lost the record, or whose retention window
    has passed, is saying "I cannot tell you", and guessing on its behalf is
    how a duplicate charge gets marked reconciled.
    """

    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    UNRESOLVED = "unresolved"


class Action(str, enum.Enum):
    """What reconciliation would do about an outcome."""

    #: Provider applied it. Record success so replays return the response.
    COMPLETE = "complete"
    #: Provider never applied it. Release the key so a retry can run the effect.
    RELEASE = "release"
    #: Nothing is safe to do. Left in UNKNOWN for a human or a later run.
    LEAVE = "leave"
    #: The adapter raised. Left untouched, with the error recorded.
    ERROR = "error"


#: The only mapping from an answer to an action. Written once, here, rather
#: than inline in the planner, because it is the whole safety argument of this
#: module and belongs somewhere a reviewer can read it in four lines.
_ACTION_FOR = {
    Outcome.APPLIED: Action.COMPLETE,
    Outcome.NOT_APPLIED: Action.RELEASE,
    Outcome.UNRESOLVED: Action.LEAVE,
}


@runtime_checkable
class ReconciliationProvider(Protocol):
    """An adapter that can say what happened for a key.

    One method, because the adapter's whole job is to answer one question. A
    Stripe adapter searches charges by idempotency key; a ledger adapter looks
    for the entry; an adapter over a system with no such lookup should return
    `UNRESOLVED` rather than inventing one.
    """

    def outcome_for(self, record: Record) -> tuple[Outcome, Any]:
        """What happened for `record`, and the response to record if applied.

        The second element is only read when the outcome is `APPLIED`, and is
        the response a later replay of this key should receive. Return `None`
        if the provider cannot supply one — the key still completes, and a
        replay gets a recorded success with no body, which is accurate.

        Raising is allowed and is not fatal to the run: the key is left in
        `UNKNOWN` with the error attached. An adapter that cannot reach its
        provider must not take the rest of the batch down with it.
        """
        ...


@dataclass(frozen=True)
class Step:
    """One record, the provider's answer, and what would be done about it."""

    key: str
    state: State
    outcome: Outcome | None
    action: Action
    response: Any = None
    error: str | None = None

    @property
    def changes_anything(self) -> bool:
        return self.action in (Action.COMPLETE, Action.RELEASE)


@dataclass(frozen=True)
class Plan:
    """What reconciliation intends to do, before it does any of it."""

    steps: list[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def effective(self) -> list[Step]:
        """Only the steps that would change something."""
        return [step for step in self.steps if step.changes_anything]

    def counts(self) -> dict[Action, int]:
        """How many steps fall to each action, including the zeroes."""
        counts = {action: 0 for action in Action}
        for step in self.steps:
            counts[step.action] += 1
        return counts


@dataclass(frozen=True)
class Applied:
    """What `apply` actually did, which is not always what the plan said."""

    completed: int = 0
    released: int = 0
    #: Steps skipped because the record stopped being UNKNOWN after planning.
    stale: list[str] = field(default_factory=list)
    #: Steps whose write raised.
    failed: list[tuple[str, str]] = field(default_factory=list)


def build_plan(
    records: Iterable[Record], provider: ReconciliationProvider
) -> Plan:
    """Ask `provider` about each record. Read-only — writes nothing.

    One provider call per record, deliberately sequential. Reconciliation runs
    against a provider that is usually rate-limited and often already unhealthy
    (being unhealthy is how these records got here), and a parallel fan-out
    turns a recovery tool into a second outage.
    """
    steps: list[Step] = []
    for record in records:
        try:
            outcome, response = provider.outcome_for(record)
        except Exception as exc:
            steps.append(
                Step(
                    key=record.key,
                    state=record.state,
                    outcome=None,
                    action=Action.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        outcome = Outcome(outcome)
        steps.append(
            Step(
                key=record.key,
                state=record.state,
                outcome=outcome,
                action=_ACTION_FOR[outcome],
                response=response if outcome is Outcome.APPLIED else None,
            )
        )
    return Plan(steps=steps)
