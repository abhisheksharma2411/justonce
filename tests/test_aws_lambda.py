"""AWS Lambda integration — the redelivery this library exists for.

Driven with plain dictionaries rather than a mocked AWS. The thing under test
is what happens when a *handler body* runs twice, and what the wrapper hands
back to SQS; neither needs a queue, and a mock of one would only be able to
agree with whatever this module already does.

The event shapes below are trimmed to the fields the wrapper reads, with the
field names taken from the real payloads.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce.core import Idempotent
from justonce.errors import KeyReuseError
from justonce.integrations.aws_lambda import (
    event_key,
    idempotent_handler,
    idempotent_sqs_handler,
    sqs_record_key,
)
from justonce.keys import operation_key
from justonce.stores import MemoryStore


@pytest.fixture()
def engine() -> Idempotent:
    return Idempotent(MemoryStore())


def sqs_event(*messages: tuple[str, str]) -> dict[str, Any]:
    return {"Records": [{"messageId": mid, "body": body} for mid, body in messages]}


# -- whole-invocation wrapper ------------------------------------------------


def test_a_retried_invocation_runs_the_body_once(engine: Idempotent) -> None:
    """The whole point: Lambda retries, the effect happens once."""
    runs = []

    @idempotent_handler(engine=engine)
    def handler(event: dict[str, Any], context: Any) -> str:
        runs.append(event["id"])
        return "receipt-" + event["id"]

    event = {"id": "evt-1", "detail-type": "order.placed"}
    first = handler(event, None)
    second = handler(event, None)

    assert runs == ["evt-1"]
    assert first == "receipt-evt-1"
    assert second == first, "the retry must return the recorded value, not None"


def test_a_raising_body_releases_the_key(engine: Idempotent) -> None:
    """A failed effect must not burn the key.

    Lambda owns the retry decision. If this held or burned the key, the retry
    Lambda schedules would find the key taken and never run the body -- the
    invocation would look retried and would not be.
    """
    attempts = []

    @idempotent_handler(engine=engine)
    def handler(event: dict[str, Any], context: Any) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("provider down")
        return "ok"

    event = {"id": "evt-flaky", "detail-type": "order.placed"}
    with pytest.raises(RuntimeError):
        handler(event, None)

    assert handler(event, None) == "ok"
    assert len(attempts) == 2


def test_two_events_are_two_effects(engine: Idempotent) -> None:
    runs = []

    @idempotent_handler(engine=engine)
    def handler(event: dict[str, Any], context: Any) -> str:
        runs.append(event["id"])
        return event["id"]

    handler({"id": "evt-1", "detail-type": "t"}, None)
    handler({"id": "evt-2", "detail-type": "t"}, None)

    assert runs == ["evt-1", "evt-2"]


def test_the_event_is_the_default_fingerprint(engine: Idempotent) -> None:
    """Same key, changed event, is a key reused for a new intent."""

    @idempotent_handler(engine=engine, key=lambda event, context: operation_key("fixed", "k"))
    def handler(event: dict[str, Any], context: Any) -> int:
        return event["amount"]

    assert handler({"amount": 100}, None) == 100
    with pytest.raises(KeyReuseError):
        handler({"amount": 250}, None)


# -- key derivation ----------------------------------------------------------


def test_event_key_reads_eventbridge() -> None:
    assert event_key({"id": "e1", "detail-type": "x"}) == operation_key("eventbridge", "e1")


def test_event_key_reads_a_dynamodb_stream_record() -> None:
    event = {"Records": [{"eventID": "d1", "eventSource": "aws:dynamodb"}]}
    assert event_key(event) == operation_key("dynamodbstream", "d1")


def test_event_key_reads_api_gateway() -> None:
    event = {"requestContext": {"requestId": "r1"}}
    assert event_key(event) == operation_key("apigateway", "r1")


def test_event_key_prefers_the_s3_request_id_over_the_sequencer() -> None:
    """Both identify the notification; the request id is the more specific."""
    event = {
        "Records": [
            {
                "responseElements": {"x-amz-request-id": "REQ1"},
                "s3": {
                    "bucket": {"name": "b"},
                    "object": {"key": "k.txt", "sequencer": "00SEQ"},
                },
            }
        ]
    }
    assert event_key(event) == operation_key("s3", "b", "k.txt", "REQ1")


def test_event_key_falls_back_to_the_s3_sequencer() -> None:
    event = {
        "Records": [
            {"s3": {"bucket": {"name": "b"}, "object": {"key": "k.txt", "sequencer": "00SEQ"}}}
        ]
    }
    assert event_key(event) == operation_key("s3", "b", "k.txt", "00SEQ")


def test_two_objects_in_one_bucket_are_two_keys() -> None:
    """The object key is in the derivation, not just the bucket."""

    def s3(key: str, sequencer: str) -> dict[str, Any]:
        return {
            "Records": [
                {"s3": {"bucket": {"name": "b"}, "object": {"key": key, "sequencer": sequencer}}}
            ]
        }

    assert event_key(s3("a.txt", "S1")) != event_key(s3("b.txt", "S2"))


def test_an_unrecognised_event_is_refused_not_guessed() -> None:
    """Rather than hashing the event and collapsing two identical payloads."""
    with pytest.raises(ValueError, match="could not derive a key"):
        event_key({"something": "else"})


def test_sqs_record_key_uses_the_message_id() -> None:
    assert sqs_record_key({"messageId": "m1"}) == operation_key("sqs", "m1")


def test_sqs_record_key_refuses_a_record_without_an_id() -> None:
    """Rather than keying on nothing and deduplicating the whole batch together."""
    with pytest.raises(ValueError, match="no messageId"):
        sqs_record_key({"body": "{}"})


# -- per-record SQS wrapper --------------------------------------------------


def test_each_message_in_a_batch_runs_once(engine: Idempotent) -> None:
    seen = []

    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        seen.append(record["messageId"])
        return "ok"

    result = handler(sqs_event(("m1", "a"), ("m2", "b")), None)

    assert seen == ["m1", "m2"]
    assert result == {"batchItemFailures": []}


def test_a_redelivered_message_does_not_run_again(engine: Idempotent) -> None:
    """SQS returns the unacknowledged message; the effect already happened."""
    seen = []

    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        seen.append(record["messageId"])
        return "ok"

    handler(sqs_event(("m1", "a"), ("m2", "b")), None)
    # The redelivery is a *different batch* -- m1 alongside a new message.
    result = handler(sqs_event(("m1", "a"), ("m3", "c")), None)

    assert seen == ["m1", "m2", "m3"]
    assert result == {"batchItemFailures": []}, "a replayed message must not be reported"


def test_one_failing_message_does_not_take_the_batch_with_it(engine: Idempotent) -> None:
    """The case the partial batch response exists for.

    Raising out of the handler instead would make SQS redeliver the whole
    batch, including the messages that succeeded — whose keys are now terminal,
    so their bodies would be skipped on the way back and the batch would make
    no further progress.
    """
    seen = []

    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        seen.append(record["messageId"])
        if record["messageId"] == "m2":
            raise RuntimeError("provider down")
        return "ok"

    result = handler(sqs_event(("m1", "a"), ("m2", "b"), ("m3", "c")), None)

    assert seen == ["m1", "m2", "m3"], "a failing message stopped the ones after it"
    assert result == {"batchItemFailures": [{"itemIdentifier": "m2"}]}


def test_a_reported_message_actually_runs_on_redelivery(engine: Idempotent) -> None:
    """The two halves have to agree, and this is the half that is easy to lose.

    Reporting the failure without releasing the key would mean the redelivery
    finds the key taken and skips the body: the message looks retried and is
    not. This asserts the body runs the second time and the batch then reports
    nothing.
    """
    attempts: list[str] = []

    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        attempts.append(record["messageId"])
        if attempts.count("m1") == 1:
            raise RuntimeError("transient")
        return "ok"

    first = handler(sqs_event(("m1", "a")), None)
    assert first == {"batchItemFailures": [{"itemIdentifier": "m1"}]}

    second = handler(sqs_event(("m1", "a")), None)
    assert attempts == ["m1", "m1"], "the redelivery never ran the body"
    assert second == {"batchItemFailures": []}


def test_a_record_without_a_message_id_raises_rather_than_vanishing(
    engine: Idempotent,
) -> None:
    """There is no id to report, so swallowing it would drop the message silently."""

    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        return "ok"

    with pytest.raises(ValueError, match="no messageId"):
        handler({"Records": [{"body": "a"}]}, None)


def test_an_empty_batch_reports_nothing(engine: Idempotent) -> None:
    @idempotent_sqs_handler(engine=engine)
    def handler(record: dict[str, Any], context: Any) -> str:
        return "ok"

    assert handler({"Records": []}, None) == {"batchItemFailures": []}


def test_a_custom_key_deduplicates_two_sends_of_one_intent(engine: Idempotent) -> None:
    """What `key=` is for, and what the messageId default deliberately is not.

    Two publishes of the same order are two message ids, so the default runs
    the effect twice. Keyed on the order, the second is a replay.
    """
    seen = []

    @idempotent_sqs_handler(
        engine=engine,
        key=lambda record, context: operation_key("charge", record["body"]),
    )
    def handler(record: dict[str, Any], context: Any) -> str:
        seen.append(record["messageId"])
        return "ok"

    handler(sqs_event(("m1", "order-7")), None)
    handler(sqs_event(("m2", "order-7")), None)

    assert seen == ["m1"], "a second send of the same intent ran the effect again"
