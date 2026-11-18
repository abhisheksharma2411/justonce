"""AWS Lambda handlers (#15).

    import justonce
    from justonce.integrations.aws_lambda import idempotent_sqs_handler
    from justonce.stores import PostgresStore

    # Module scope on purpose -- see "Cold starts" below.
    justonce.configure(PostgresStore(os.environ["DSN"]))

    @idempotent_sqs_handler()
    def handle_order(record, context):
        return charge(json.loads(record["body"])["order_id"])

Lambda retries a failed invocation on its own, and every event source in front
of it redelivers: SQS returns an unacknowledged message to the queue, S3 and
EventBridge retry on error, and a function with a dead-letter queue sees the
message again after the redrive. Handlers are invoked more than once by design,
and whether that is safe is the application's problem.

**Two wrappers, because an SQS batch is not one event.**

`idempotent_handler` keys the whole invocation. It suits a source where one
invocation really is one logical event: EventBridge, S3, a DynamoDB stream
record, an API Gateway request.

`idempotent_sqs_handler` keys each *record* and calls your function once per
message. For SQS this is the only correct shape, and the reason is worth
stating because the wrong one fails silently in both directions. SQS redelivers
*individual* messages, so the batch on a retry is a different set of messages
from the batch that failed. Key the invocation and one poisoned message either
blocks the nine beside it forever, or -- worse -- the retry replays the
invocation key and the nine that never ran are skipped as duplicates.

**Partial batch responses are not optional here.** `idempotent_sqs_handler`
returns ``{"batchItemFailures": [...]}`` and your event source mapping must have
`ReportBatchItemFailures` enabled, or SQS treats the whole batch as successful
and deletes messages whose effect never ran. The two halves have to agree: a
record whose body raised gets its key *released*, so the redelivery runs it, and
its id *reported*, so a redelivery happens at all. Doing one without the other
is the silent failure this wrapper exists to prevent -- release without report
deletes a message whose effect never happened, and report without release means
the redelivery finds the key taken and skips the body, so the message looks
retried and is not.

**Cold starts.** Build the store and call `justonce.configure` at module scope,
as above. Module scope runs once per execution environment; a store constructed
inside the handler opens a connection on every invocation, which on a busy
function is both the latency you are trying to avoid and a good way to exhaust
the database's connection limit. The decorators resolve the engine lazily and
cache nothing per call, so the only thing that has to be module-level is yours.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from ..core import Idempotent
from ..keys import operation_key

__all__ = [
    "event_key",
    "idempotent_handler",
    "idempotent_sqs_handler",
    "sqs_record_key",
]


def sqs_record_key(record: dict[str, Any]) -> str:
    """Key one SQS message, from the id SQS keeps across redeliveries.

    `messageId` is assigned when the message is *sent* and survives every
    redelivery of it: the visibility-timeout return, the explicit failure
    report, the redrive from a dead-letter queue. That is exactly the scope
    this deduplicates.

    It does **not** deduplicate two *sends* of the same intent. A producer that
    publishes the same order twice gets two message ids and the effect runs
    twice. If what you need is "this order is charged once however many times
    anything enqueues it", key on the order -- pass `key=` and read it out of
    the body.

    Not `md5OfBody`: two genuinely separate intents can carry identical bodies,
    and collapsing them means the second effect never runs.

    Raises:
        ValueError: if the record carries no `messageId`, rather than keying on
            nothing and deduplicating every message in the batch together.
    """
    message_id = record.get("messageId")
    if not message_id:
        raise ValueError(
            "SQS record has no messageId, so there is no delivery to key on. "
            "Pass key= and derive it from the message body instead."
        )
    return operation_key("sqs", message_id)


def event_key(event: dict[str, Any]) -> str:
    """Key one invocation, from whatever the event source calls its identity.

    Recognised sources, and what each id actually protects against -- which is
    the part worth reading, because every one of these deduplicates Lambda's
    own retries and only some of them deduplicate a retry by whoever sent it:

    * **EventBridge** (`id`) -- stable across every retry of that event, and a
      new id if the producer publishes again.
    * **S3** (`Records[0].responseElements."x-amz-request-id"` with the object
      key, falling back to `s3.object.sequencer`) -- stable across retries of
      that notification. `sequencer` is the token S3 documents for ordering and
      collapsing events on one object; `eTag` is not usable here because a
      multipart upload's eTag is not a digest of the content.
    * **DynamoDB streams** (`Records[0].eventID`) -- stable across retries of
      that stream record.
    * **API Gateway** (`requestContext.requestId`) -- stable across Lambda's
      retries of that HTTP request, and **different for every client retry**.
      A client that times out and resends gets a new request id and the effect
      runs again. For that you want the client's own `Idempotency-Key` header;
      see `justonce.integrations.asgi`.

    Raises:
        ValueError: if the event matches none of these. Guessing would be worse
            than refusing: the fallback everyone reaches for is hashing the
            whole event, and two genuinely separate events with identical
            payloads would then collapse onto one key and the second effect
            would never run.
    """
    if "id" in event and "detail-type" in event:
        return operation_key("eventbridge", event["id"])

    records = event.get("Records")
    if isinstance(records, list) and records:
        first = records[0]
        if "eventID" in first:
            return operation_key("dynamodbstream", first["eventID"])
        if "s3" in first:
            s3 = first["s3"]
            obj = s3.get("object", {})
            identity = (
                first.get("responseElements", {}).get("x-amz-request-id")
                or obj.get("sequencer")
            )
            if identity:
                return operation_key(
                    "s3", s3.get("bucket", {}).get("name", "-"), obj.get("key", "-"), identity
                )

    request_id = event.get("requestContext", {}).get("requestId")
    if request_id:
        return operation_key("apigateway", request_id)

    raise ValueError(
        "could not derive a key from this event: it is not an EventBridge, S3, "
        "DynamoDB stream or API Gateway event. Pass key= with a derivation you "
        "can defend -- hashing the whole event collapses two separate events "
        "that happen to carry the same payload."
    )


def _default_engine() -> Idempotent:
    from ..decorators import get_default

    return get_default()


def idempotent_handler(
    func: Callable[..., Any] | None = None,
    *,
    key: Callable[..., str] | None = None,
    payload: Callable[..., Any] | None = None,
    engine: Idempotent | None = None,
    ttl_seconds: float | None = None,
    retention_seconds: float | None = None,
) -> Any:
    """Run this handler's body at most once per key, per invocation.

    A later invocation with the same key does not run the body; it returns the
    recorded return value, so the caller sees what the first invocation
    produced rather than `None`.

    Args:
        key: builds the key from `(event, context)`. Defaults to `event_key`,
            which raises on an event shape it does not recognise rather than
            guessing.
        payload: fingerprinted to catch the same key used for a changed intent.
            Defaults to the event.
        engine: an `Idempotent`. Defaults to the one `justonce.configure` set.
        ttl_seconds: claim lease. Should exceed the function's realistic
            runtime, and there is no point setting it beyond the function
            timeout: Lambda will have killed the invocation by then.
        retention_seconds: how long the result stays replayable.

    **A raising body always releases the key**, so Lambda's own retry and the
    source's redelivery actually run. Burning the key on failure would silently
    disable them: the retry would arrive, find a terminal record, and return
    the failure without ever running. A handler that must stop retrying should
    stop it with the function's `MaximumRetryAttempts` and a dead-letter queue,
    where the policy is visible to whoever is on call.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(target)
        def wrapper(event: dict[str, Any], context: Any = None) -> Any:
            active = engine if engine is not None else _default_engine()
            effect_key = key(event, context) if key is not None else event_key(event)
            fingerprint = payload(event, context) if payload is not None else event
            result = active.run(
                effect_key,
                lambda: target(event, context),
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


def idempotent_sqs_handler(
    func: Callable[..., Any] | None = None,
    *,
    key: Callable[..., str] | None = None,
    payload: Callable[..., Any] | None = None,
    engine: Idempotent | None = None,
    ttl_seconds: float | None = None,
    retention_seconds: float | None = None,
) -> Any:
    """Run the body once per SQS *message*, and report the ones that failed.

    The decorated function takes `(record, context)` and is called once per
    message in the batch, not once per invocation. It returns the partial batch
    response SQS expects::

        {"batchItemFailures": [{"itemIdentifier": "<messageId>"}, ...]}

    **Enable `ReportBatchItemFailures` on the event source mapping.** Without
    it SQS ignores that return value, treats the batch as fully successful and
    deletes every message in it -- including the ones this wrapper just told it
    to redeliver.

    Args:
        key: builds the key from `(record, context)`. Defaults to
            `sqs_record_key`, which keys on `messageId` and so deduplicates
            redeliveries of one message but not two sends of one intent. Read
            its docstring before relying on the default.
        payload: fingerprinted against key reuse. Defaults to the record body.
        engine: an `Idempotent`. Defaults to the one `justonce.configure` set.
        ttl_seconds: claim lease per message. Worth keeping below the queue's
            visibility timeout: a lease that outlives the timeout is a message
            redelivered while the first attempt still holds its key, so the
            redelivery does nothing and the batch stalls on its own tail.
        retention_seconds: how long each message's result stays replayable.

    A message whose body raises gets its key released *and* its id reported, so
    the redelivery both happens and runs. A message that was already processed
    replays its recorded value and is **not** reported, because reporting it
    would redeliver a message whose effect has already happened, forever.

    One failing message never fails the others: the loop records it and carries
    on, which is the behaviour the partial batch response exists to express.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(target)
        def wrapper(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
            active = engine if engine is not None else _default_engine()
            failures: list[dict[str, str]] = []

            for record in event.get("Records", []):
                message_id = record.get("messageId")
                try:
                    effect_key = (
                        key(record, context) if key is not None else sqs_record_key(record)
                    )
                    fingerprint = (
                        payload(record, context) if payload is not None else record.get("body")
                    )
                    active.run(
                        effect_key,
                        # `partial` rather than a lambda with a default
                        # argument: the call is synchronous, so late binding is
                        # not a hazard here, but the default-argument idiom
                        # exists only to work around one and reads as if it
                        # were.
                        functools.partial(target, record, context),
                        payload=fingerprint,
                        retry_on_failure=True,
                        ttl_seconds=ttl_seconds,
                        retention_seconds=retention_seconds,
                    )
                except Exception:
                    # Reported rather than raised. Raising here would fail the
                    # whole invocation, which SQS reads as "redeliver every
                    # message in the batch" -- including the ones that
                    # succeeded, whose keys are now terminal, so their bodies
                    # would be skipped on the way back and the batch would make
                    # no further progress.
                    if message_id:
                        failures.append({"itemIdentifier": message_id})
                    else:
                        raise

            return {"batchItemFailures": failures}

        wrapper.justonce_idempotent = True  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
