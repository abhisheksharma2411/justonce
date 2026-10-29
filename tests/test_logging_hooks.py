"""Structured logging hooks (#52).

The two things these tests exist to hold down are that a response never reaches
the log, and that every event can actually be emitted — `logging` raises
`KeyError` if an `extra` field collides with a `LogRecord` attribute, and that
would otherwise surface as a hook silently swallowed on the charge path.
"""

from __future__ import annotations

import logging

import pytest

import justonce
from justonce.logging import DEFAULT_LEVELS, LOGGER_NAME, LoggingHooks
from justonce.stores import MemoryStore
from justonce.stores.base import Record, State


@pytest.fixture
def logger(request):
    log = logging.getLogger(f"{LOGGER_NAME}.test.{request.node.name}")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield log
    log.handlers.clear()


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        # format() here on purpose: a %-style mismatch between the message and
        # its args only shows up at format time, and a handler that never
        # formats would let a broken message through these tests.
        record.formatted = self.format(record)
        self.records.append(record)


@pytest.fixture
def captured(logger):
    handler = Capture()
    logger.addHandler(handler)
    return handler


def events(captured) -> list[str]:
    return [r.event for r in captured.records]


def test_every_event_emits_without_colliding_with_logrecord(logger, captured):
    hooks = LoggingHooks(logger, levels=dict.fromkeys(DEFAULT_LEVELS, logging.CRITICAL))
    record = Record(key="k", state=State.SUCCEEDED, request_hash="h", attempts=2)

    hooks.duplicate_suppressed("k", record)
    hooks.claim_conflict("k")
    hooks.key_reuse("k")
    hooks.unknown_recorded("k")
    hooks.effect_finished("k", 0.25, True)
    hooks.effect_finished("k", 0.25, False)
    hooks.ran_unguarded("k")

    assert events(captured) == [
        "duplicate_suppressed",
        "claim_conflict",
        "key_reuse",
        "unknown_recorded",
        "effect_succeeded",
        "effect_failed",
        "ran_unguarded",
    ]


def test_the_response_never_reaches_the_log(logger, captured):
    secret = "4111111111111111"
    record = Record(
        key="k",
        state=State.SUCCEEDED,
        request_hash="fingerprint-of-the-payload",
        response={"card": secret},
        attempts=1,
    )

    LoggingHooks(logger).duplicate_suppressed("k", record)

    [line] = [r.formatted for r in captured.records]
    assert secret not in line
    assert "fingerprint-of-the-payload" not in line
    assert not hasattr(captured.records[0], "response")
    assert captured.records[0].state == "succeeded"


def test_hash_keys_replaces_the_key_everywhere_it_appears(logger, captured):
    key = "charge:customer@example.com"

    LoggingHooks(logger, hash_keys=True).claim_conflict(key)

    [record] = captured.records
    assert key not in record.formatted
    assert "example.com" not in record.formatted
    assert record.key == LoggingHooks(hash_keys=True).display_key(key)
    assert len(record.key) == 16


def test_levels_default_to_the_documented_map(logger, captured):
    hooks = LoggingHooks(logger)

    hooks.key_reuse("k")
    hooks.unknown_recorded("k")
    hooks.effect_finished("k", 0.1, True)

    assert [(r.event, r.levelno) for r in captured.records] == [
        ("key_reuse", logging.WARNING),
        ("unknown_recorded", logging.ERROR),
        ("effect_succeeded", logging.DEBUG),
    ]


def test_an_unknown_event_override_is_refused():
    with pytest.raises(ValueError, match="effect_finished"):
        LoggingHooks(levels={"effect_finished": logging.INFO})


def test_a_level_the_logger_rejects_is_not_emitted(logger, captured):
    logger.setLevel(logging.WARNING)

    LoggingHooks(logger).effect_finished("k", 0.1, True)

    assert captured.records == []


def test_the_engine_drives_the_hooks_end_to_end(logger, captured):
    engine = justonce.Idempotent(MemoryStore(), hooks=LoggingHooks(logger))

    first = engine.run("k", lambda: {"charge_id": "ch_1"})
    second = engine.run("k", lambda: pytest.fail("the effect must not run twice"))

    assert first.executed and second.deduplicated
    # claim_conflict first: a replay loses the claim before it replays, which is
    # what hooks.py means by "fires for every lost claim, including the ones
    # that go on to replay a terminal record".
    assert events(captured) == [
        "effect_succeeded",
        "claim_conflict",
        "duplicate_suppressed",
    ]


def test_a_broken_handler_cannot_break_the_effect(logger):
    class Exploding(logging.Handler):
        def emit(self, record):
            raise RuntimeError("the log host is full")

    logger.addHandler(Exploding())
    # logging swallows handler errors itself; raise from the hook instead so the
    # engine's own guard is what is under test.
    hooks = LoggingHooks(logger)
    hooks.display_key = lambda key: (_ for _ in ()).throw(RuntimeError("boom"))
    engine = justonce.Idempotent(MemoryStore(), hooks=hooks)

    result = engine.run("k", lambda: "charged")

    assert result.value == "charged"
    assert result.executed
