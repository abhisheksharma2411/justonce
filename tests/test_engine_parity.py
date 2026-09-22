"""The sync and async engines must not drift apart.

`Idempotent` and `AsyncIdempotent` used to implement the same state machine
twice. They agreed because they were written on the same afternoon, and nothing
kept them agreeing: a fix applied to one and forgotten in the other would have
handed async callers weaker guarantees than the README promises, silently.

Two guards, at different levels:

  * `classify` is pure, so every branch of the decision is walked as a table.
  * Each scenario below is run through *both* engines and the outcomes are
    compared to each other as well as to the expected answer. Comparing only
    against the expectation would let both engines be wrong together;
    comparing only against each other would let both be wrong identically.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from justonce import (
    AsyncIdempotent,
    Idempotent,
    InFlightTimeout,
    KeyReuseError,
    KeyTooLongError,
    OnInFlight,
    OperationInFlightError,
    State,
    StoreError,
    operation_key,
)
from justonce.keys import fingerprint
from justonce.machine import (
    Disposition,
    OnStoreUnavailable,
    classify,
    unguarded_run_allowed,
)
from justonce.stores import SqliteStore
from justonce.stores.base import Record

PAYLOAD = {"amount": 500, "currency": "USD"}
OTHER_PAYLOAD = {"amount": 9999, "currency": "USD"}


# -- the decision table -----------------------------------------------------
#
# `classify` is the single source of truth both engines call. It takes no clock
# and no store, so the whole contract fits in a table.

def _record(state: State, request_hash: str = "h") -> Record:
    return Record(key="k", state=state, request_hash=request_hash)


@pytest.mark.parametrize(
    ("record", "request_hash", "on_in_flight", "expected"),
    [
        # A record that vanished between the failed claim and the read. We
        # cannot prove the effect did not run, so we refuse rather than run it.
        (None, "h", OnInFlight.RAISE, Disposition.IN_FLIGHT),
        (None, "h", OnInFlight.WAIT, Disposition.IN_FLIGHT),
        # Payload divergence outranks every state: the recorded outcome belongs
        # to a different request, so it is the wrong answer even when terminal.
        (_record(State.SUCCEEDED), "other", OnInFlight.RAISE, Disposition.KEY_REUSE),
        (_record(State.IN_PROGRESS), "other", OnInFlight.WAIT, Disposition.KEY_REUSE),
        (_record(State.UNKNOWN), "other", OnInFlight.RAISE, Disposition.KEY_REUSE),
        # Terminal records replay, and waiting does not change that.
        (_record(State.SUCCEEDED), "h", OnInFlight.RAISE, Disposition.REPLAY),
        (_record(State.SUCCEEDED), "h", OnInFlight.WAIT, Disposition.REPLAY),
        (_record(State.FAILED), "h", OnInFlight.RAISE, Disposition.REPLAY),
        (_record(State.FAILED), "h", OnInFlight.WAIT, Disposition.REPLAY),
        # UNKNOWN is never waited on. Nobody is coming to resolve it; only
        # reconciliation can, so waiting would burn the timeout and then refuse.
        (_record(State.UNKNOWN), "h", OnInFlight.RAISE, Disposition.IN_FLIGHT),
        (_record(State.UNKNOWN), "h", OnInFlight.WAIT, Disposition.IN_FLIGHT),
        # Only a live holder is affected by the caller's in-flight policy.
        (_record(State.IN_PROGRESS), "h", OnInFlight.RAISE, Disposition.IN_FLIGHT),
        (_record(State.IN_PROGRESS), "h", OnInFlight.WAIT, Disposition.WAIT),
    ],
)
def test_classify_table(record, request_hash, on_in_flight, expected) -> None:
    assert classify(record, request_hash, on_in_flight) is expected


#: The other decision both engines share: may a caller whose claim raised run
#: the effect without one? Walked as a table for the same reason `classify` is
#: — it is pure, and a table shows the exceptions that must *not* open the gate
#: next to the one that may.
@pytest.mark.parametrize(
    ("exc", "policy", "expected"),
    [
        (StoreError("down"), OnStoreUnavailable.FAIL_OPEN, True),
        (StoreError("down"), OnStoreUnavailable.FAIL_CLOSED, False),
        # Not an outage: the key is about to be truncated into a collision.
        (KeyTooLongError("k", 8, "test"), OnStoreUnavailable.FAIL_OPEN, False),
        (KeyReuseError("k"), OnStoreUnavailable.FAIL_OPEN, False),
        # A bug in a store, of unknown shape. Not grounds for an unguarded run.
        (RuntimeError("driver bug"), OnStoreUnavailable.FAIL_OPEN, False),
        (KeyboardInterrupt(), OnStoreUnavailable.FAIL_OPEN, False),
    ],
    ids=["outage-open", "outage-closed", "key-too-long", "key-reuse", "bug", "interrupt"],
)
def test_unguarded_run_table(exc, policy, expected) -> None:
    assert unguarded_run_allowed(exc, policy) is expected


def test_only_one_policy_ever_runs_unguarded() -> None:
    """Fail-open must stay the single named opt-in, whatever policies exist."""
    opening = {p for p in OnStoreUnavailable if unguarded_run_allowed(StoreError("x"), p)}
    assert opening == {OnStoreUnavailable.FAIL_OPEN}


def test_every_disposition_is_reachable() -> None:
    """A table that cannot produce some outcome is not covering the contract."""
    reached = {
        classify(rec, "h", policy)
        for policy in OnInFlight
        for rec in (None, _record(State.SUCCEEDED), _record(State.IN_PROGRESS))
    } | {classify(_record(State.SUCCEEDED), "other", OnInFlight.RAISE)}
    assert reached == set(Disposition)


# -- driving both engines through one scenario body -------------------------


@dataclass
class Outcome:
    """A comparable summary of what a call did, however it ended."""

    kind: str
    value: Any = None
    executed: bool | None = None
    state: State | None = None


class SyncDriver:
    label = "sync"

    def __init__(self, store: SqliteStore, **kwargs: Any) -> None:
        self.store = store
        self.engine = Idempotent(store, **kwargs)

    async def run(self, key: str, effect: Callable[[], Any], **kwargs: Any) -> Outcome:
        return _capture(lambda: self.engine.run(key, effect, **kwargs))


class AsyncDriver:
    label = "async"

    def __init__(self, store: SqliteStore, **kwargs: Any) -> None:
        self.store = store
        self.engine = AsyncIdempotent(store, **kwargs)

    async def run(self, key: str, effect: Callable[[], Any], **kwargs: Any) -> Outcome:
        async def coro() -> Any:
            await asyncio.sleep(0)  # a real await point, as a real client has
            return effect()

        try:
            result = await self.engine.run(key, coro, **kwargs)
        # A raise is an outcome here, not a test error: which exception a
        # caller gets is exactly what the two engines must agree on.
        except Exception as exc:
            return Outcome(kind=type(exc).__name__)
        return Outcome("ok", result.value, result.executed, _state_of(result))


def _capture(call: Callable[[], Any]) -> Outcome:
    try:
        result = call()
    # A raise is an outcome here, not a test error.
    except Exception as exc:
        return Outcome(kind=type(exc).__name__)
    return Outcome("ok", result.value, result.executed, _state_of(result))


def _state_of(result: Any) -> State | None:
    return result.record.state if result.record is not None else None


class Charger:
    """Counts real side effects, so 'ran twice' is observable."""

    def __init__(self) -> None:
        self.calls = 0

    def charge(self) -> dict:
        self.calls += 1
        return {"charge_id": f"ch_{self.calls}"}


# -- the shared scenarios ---------------------------------------------------
#
# Each returns (outcomes, effect_count). They are written once and executed by
# both drivers; a divergence shows up as a mismatch between the two runs.

async def scenario_replay(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    first = await d.run(key, charger.charge, payload=PAYLOAD)
    second = await d.run(key, charger.charge, payload=PAYLOAD)
    return [first, second], charger.calls


async def scenario_key_reuse(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    first = await d.run(key, charger.charge, payload=PAYLOAD)
    second = await d.run(key, charger.charge, payload=OTHER_PAYLOAD)
    return [first, second], charger.calls


async def scenario_in_flight(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    d.store.claim(key, fingerprint(PAYLOAD), 900)  # another worker holds it
    return [await d.run(key, charger.charge, payload=PAYLOAD)], charger.calls


async def scenario_unknown_after_crash(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    d.store.claim(key, fingerprint(PAYLOAD), 900)
    d.store.mark_unknown(key)  # died between effect and outcome write
    return [await d.run(key, charger.charge, payload=PAYLOAD)], charger.calls


async def scenario_terminal_failure(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()

    def boom() -> Any:
        charger.calls += 1
        raise RuntimeError("declined")

    first = await d.run(key, boom, payload=PAYLOAD, retry_on_failure=False)
    second = await d.run(key, charger.charge, payload=PAYLOAD)
    return [first, second], charger.calls


async def scenario_transient_failure(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()

    def boom() -> Any:
        charger.calls += 1
        raise RuntimeError("timeout")

    first = await d.run(key, boom, payload=PAYLOAD, retry_on_failure=True)
    second = await d.run(key, charger.charge, payload=PAYLOAD)
    return [first, second], charger.calls


async def scenario_reclaim_expired(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    d.store.claim(key, fingerprint(PAYLOAD), 0.01)  # holder presumed dead
    time.sleep(0.05)
    return [await d.run(key, charger.charge, payload=PAYLOAD)], charger.calls


async def scenario_reclaim_wrong_payload(d: Any) -> tuple[list[Outcome], int]:
    """An expired lease frees the holder, not the key."""
    key, charger = operation_key("charge", "o1"), Charger()
    d.store.claim(key, fingerprint(PAYLOAD), 0.01)
    time.sleep(0.05)
    return [await d.run(key, charger.charge, payload=OTHER_PAYLOAD)], charger.calls


async def scenario_wait_times_out(d: Any) -> tuple[list[Outcome], int]:
    key, charger = operation_key("charge", "o1"), Charger()
    d.store.claim(key, fingerprint(PAYLOAD), 900)  # holder never finishes
    return [await d.run(key, charger.charge, payload=PAYLOAD)], charger.calls


async def scenario_wait_replays_terminal(d: Any) -> tuple[list[Outcome], int]:
    """Opting into WAIT must not delay a caller whose answer already exists."""
    key, charger = operation_key("charge", "o1"), Charger()
    first = await d.run(key, charger.charge, payload=PAYLOAD)
    started = time.monotonic()
    second = await d.run(key, charger.charge, payload=PAYLOAD)
    assert time.monotonic() - started < 0.5, "replayed through the poll loop"
    return [first, second], charger.calls


CH1 = {"charge_id": "ch_1"}
CH2 = {"charge_id": "ch_2"}
WAIT_SHORT = {"on_in_flight": OnInFlight.WAIT, "wait_timeout": 0.2, "poll_interval": 0.02}
WAIT_LONG = {"on_in_flight": OnInFlight.WAIT, "wait_timeout": 5.0}


def ok(value: Any, executed: bool, state: State) -> Outcome:
    return Outcome("ok", value, executed, state)


def raised(exc: type[BaseException]) -> Outcome:
    return Outcome(exc.__name__)


#: (scenario, engine kwargs, expected outcomes, expected number of effects run).
#: The effect count is the one that would catch a real duplicate charge; the
#: outcomes catch the two engines answering the same question differently.
SCENARIOS = [
    (scenario_replay, {},
     [ok(CH1, True, State.SUCCEEDED), ok(CH1, False, State.SUCCEEDED)], 1),
    (scenario_key_reuse, {},
     [ok(CH1, True, State.SUCCEEDED), raised(KeyReuseError)], 1),
    (scenario_in_flight, {},
     [raised(OperationInFlightError)], 0),
    (scenario_unknown_after_crash, {},
     [raised(OperationInFlightError)], 0),
    (scenario_terminal_failure, {},
     [raised(RuntimeError), ok(None, False, State.FAILED)], 1),
    (scenario_transient_failure, {},
     [raised(RuntimeError), ok(CH2, True, State.SUCCEEDED)], 2),
    (scenario_reclaim_expired, {},
     [ok(CH1, True, State.SUCCEEDED)], 1),
    (scenario_reclaim_wrong_payload, {},
     [raised(KeyReuseError)], 0),
    (scenario_wait_times_out, WAIT_SHORT,
     [raised(InFlightTimeout)], 0),
    (scenario_wait_replays_terminal, WAIT_LONG,
     [ok(CH1, True, State.SUCCEEDED), ok(CH1, False, State.SUCCEEDED)], 1),
]

IDS = [s[0].__name__.removeprefix("scenario_") for s in SCENARIOS]


@pytest.mark.parametrize(("scenario", "kwargs", "expected", "effects"), SCENARIOS, ids=IDS)
def test_engines_agree(scenario, kwargs, expected, effects) -> None:
    sync_outcomes, sync_effects = asyncio.run(
        scenario(SyncDriver(SqliteStore(":memory:"), **kwargs))
    )
    async_outcomes, async_effects = asyncio.run(
        scenario(AsyncDriver(SqliteStore(":memory:"), **kwargs))
    )

    # Against each other: catches a fix applied to one engine and not the other.
    assert sync_outcomes == async_outcomes, (
        f"engines disagree on {scenario.__name__}: "
        f"sync={sync_outcomes} async={async_outcomes}"
    )
    assert sync_effects == async_effects

    # Against the contract: catches both engines being wrong together.
    assert sync_outcomes == expected
    assert sync_effects == effects, (
        f"{scenario.__name__} ran the effect {sync_effects}x, expected {effects}x"
    )
