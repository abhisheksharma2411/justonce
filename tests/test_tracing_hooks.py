"""OpenTelemetry tracing hooks (#25).

The property that carries this feature is the one an incident depends on: a
replayed call must say so **on the span the request already has**. A span of
our own, buried under the caller's, is not what someone staring at a slow
checkout looks at; `justonce.deduplicated` on the request span is.

The second property is the same one `test_logging_hooks` pins for logs: the
effect's return value is a charge, an account number, an address, and none of
it may reach a span. A trace backend is a searchable, long-retention,
widely-readable store, which makes an accidental attribute there worse than an
accidental log line.
"""

from __future__ import annotations

from typing import Any

import pytest

from justonce import Idempotent, KeyReuseError, OnStoreUnavailable, StoreError
from justonce.stores import MemoryStore
from justonce.stores.base import Record, State

trace = pytest.importorskip("opentelemetry.trace")
sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from justonce.tracing import (  # noqa: E402 - after importorskip
    ATTR_ATTEMPTS,
    ATTR_DEDUPLICATED,
    ATTR_GUARDED,
    ATTR_KEY,
    ATTR_STATE,
    EFFECT_SPAN,
    TracingHooks,
)


@pytest.fixture()
def exporter() -> Any:
    """A real SDK tracer writing to memory.

    Deliberately not a mock: the things most likely to break here are
    OpenTelemetry's own rules — an attribute type it rejects, a span ended
    twice, a start_time after an end_time — and a mock accepts all of them.
    """
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


def spans(exporter: Any) -> list[Any]:
    return list(exporter.get_finished_spans())


def by_name(exporter: Any, name: str) -> Any:
    found = [span for span in spans(exporter) if span.name == name]
    assert len(found) == 1, f"expected one {name!r} span, got {[s.name for s in spans(exporter)]}"
    return found[0]


class TestTheEffectSpan:
    def test_the_effect_gets_a_span_with_its_real_duration(self, exporter: Any) -> None:
        memory, tracer = exporter
        TracingHooks(tracer).effect_finished("k", 0.25, ok=True)

        span = by_name(memory, EFFECT_SPAN)
        elapsed_ns = span.end_time - span.start_time
        # Reconstructed from the duration the hook was handed, so it is exact
        # rather than approximate -- that is the point of building it this way.
        assert elapsed_ns == 250_000_000
        assert span.attributes[ATTR_KEY] == "k"
        assert span.attributes[ATTR_DEDUPLICATED] is False

    def test_a_failed_effect_marks_the_span_as_an_error(self, exporter: Any) -> None:
        memory, tracer = exporter
        TracingHooks(tracer).effect_finished("k", 0.01, ok=False)

        span = by_name(memory, EFFECT_SPAN)
        assert span.status.status_code is trace.StatusCode.ERROR

    def test_a_nonsensical_duration_cannot_produce_a_backwards_span(self, exporter: Any) -> None:
        # Not reachable from the engine. But a span whose end precedes its start
        # is a corrupt trace rather than a wrong number, and clamping costs one
        # max() on a path that already took a network round trip.
        memory, tracer = exporter
        TracingHooks(tracer).effect_finished("k", -5.0, ok=True)

        span = by_name(memory, EFFECT_SPAN)
        assert span.end_time >= span.start_time

    def test_the_effect_span_hangs_off_the_caller_s_span(self, exporter: Any) -> None:
        memory, tracer = exporter
        hooks = TracingHooks(tracer)
        with tracer.start_as_current_span("checkout") as caller:
            hooks.effect_finished("k", 0.01, ok=True)
            expected = caller.get_span_context().span_id

        effect = by_name(memory, EFFECT_SPAN)
        assert effect.parent is not None, "the effect span must not be a root span"
        assert effect.parent.span_id == expected


class TestTheCallersSpan:
    def test_a_replay_is_visible_on_the_request_s_own_span(self, exporter: Any) -> None:
        memory, tracer = exporter
        hooks = TracingHooks(tracer)
        record = Record(key="k", state=State.SUCCEEDED, request_hash="h", attempts=3)

        with tracer.start_as_current_span("checkout"):
            hooks.duplicate_suppressed("k", record)

        checkout = by_name(memory, "checkout")
        # The attribute is what makes this answerable across a whole service --
        # "what fraction of charges are replays" is a filter on it. The event is
        # what places it in the request's timeline.
        assert checkout.attributes[ATTR_DEDUPLICATED] is True
        event = next(e for e in checkout.events if e.name == "justonce.duplicate_suppressed")
        assert event.attributes[ATTR_STATE] == "succeeded"
        assert event.attributes[ATTR_ATTEMPTS] == 3

    def test_key_reuse_marks_the_span_as_an_error(self, exporter: Any) -> None:
        memory, tracer = exporter
        with tracer.start_as_current_span("checkout"):
            TracingHooks(tracer).key_reuse("k")

        checkout = by_name(memory, "checkout")
        # An intent is about to be dropped on the floor. A trace that showed
        # this as an ordinary successful request would be actively misleading.
        assert checkout.status.status_code is trace.StatusCode.ERROR

    def test_an_unrecorded_outcome_marks_the_span_as_an_error(self, exporter: Any) -> None:
        memory, tracer = exporter
        with tracer.start_as_current_span("checkout"):
            TracingHooks(tracer).unknown_recorded("k")

        assert by_name(memory, "checkout").status.status_code is trace.StatusCode.ERROR

    def test_running_unguarded_is_recorded_as_such(self, exporter: Any) -> None:
        memory, tracer = exporter
        with tracer.start_as_current_span("checkout"):
            TracingHooks(tracer).ran_unguarded("k")

        checkout = by_name(memory, "checkout")
        assert checkout.attributes[ATTR_GUARDED] is False
        assert any(e.name == "justonce.ran_unguarded" for e in checkout.events)

    def test_a_claim_conflict_is_an_event_not_an_error(self, exporter: Any) -> None:
        memory, tracer = exporter
        with tracer.start_as_current_span("checkout"):
            TracingHooks(tracer).claim_conflict("k")

        checkout = by_name(memory, "checkout")
        assert any(e.name == "justonce.claim_conflict" for e in checkout.events)
        # Losing a claim is the library working. Marking it an error would make
        # every healthy replay look like a failure.
        assert checkout.status.status_code is not trace.StatusCode.ERROR


class TestWhatNeverReachesATrace:
    def test_the_response_never_reaches_a_span(self, exporter: Any) -> None:
        memory, tracer = exporter
        record = Record(
            key="k",
            state=State.SUCCEEDED,
            request_hash="4111111111111111-hash",
            response={"card": "4111111111111111", "email": "a@b.example"},
            attempts=1,
        )

        with tracer.start_as_current_span("checkout"):
            TracingHooks(tracer).duplicate_suppressed("k", record)

        rendered = repr(
            [
                (
                    span.name,
                    dict(span.attributes),
                    [(e.name, dict(e.attributes)) for e in span.events],
                )
                for span in spans(memory)
            ]
        )
        assert "4111111111111111" not in rendered
        assert "a@b.example" not in rendered

    def test_hashing_replaces_the_key_but_keeps_duplicates_groupable(self, exporter: Any) -> None:
        memory, tracer = exporter
        hooks = TracingHooks(tracer, hash_keys=True)
        hooks.effect_finished("charge:alice@example.com", 0.01, ok=True)
        hooks.effect_finished("charge:alice@example.com", 0.01, ok=True)

        shown = {span.attributes[ATTR_KEY] for span in spans(memory)}
        assert len(shown) == 1, "the same key must still hash to one value"
        assert "alice@example.com" not in next(iter(shown))


class TestThroughTheEngine:
    def test_a_real_replay_produces_the_trace_an_operator_would_look_at(
        self, exporter: Any
    ) -> None:
        memory, tracer = exporter
        engine = Idempotent(MemoryStore(), hooks=TracingHooks(tracer))

        with tracer.start_as_current_span("first"):
            engine.run("k", lambda: {"charged": True})
        with tracer.start_as_current_span("second"):
            engine.run("k", lambda: {"charged": True})

        first, second = by_name(memory, "first"), by_name(memory, "second")
        assert first.attributes.get(ATTR_DEDUPLICATED) is None, "the first call ran the effect"
        assert second.attributes[ATTR_DEDUPLICATED] is True
        # One effect span in total: the replay did not run the effect, so it
        # must not have produced one.
        assert sum(1 for span in spans(memory) if span.name == EFFECT_SPAN) == 1

    def test_a_tracer_that_explodes_still_cannot_break_the_effect(self) -> None:
        class Exploding(TracingHooks):
            def __init__(self) -> None:
                super().__init__(tracer=object())

            def display_key(self, key: str) -> str:
                raise RuntimeError("trace exporter down")

        # The engine's guarantee, restated for this hook specifically, because
        # TracingHooks is the first one that reaches a network client.
        result = Idempotent(MemoryStore(), hooks=Exploding()).run("k", lambda: "ok")
        assert result.value == "ok"
        assert result.executed is True

    def test_fail_open_is_visible_as_an_unguarded_run(self, exporter: Any) -> None:
        memory, tracer = exporter

        class Broken(MemoryStore):
            def claim(self, *args: Any, **kwargs: Any) -> Any:
                raise StoreError("store down")

        with tracer.start_as_current_span("checkout"):
            Idempotent(
                Broken(),
                hooks=TracingHooks(tracer),
                on_store_unavailable=OnStoreUnavailable.FAIL_OPEN,
            ).run("k", lambda: "ok")

        assert by_name(memory, "checkout").attributes[ATTR_GUARDED] is False

    def test_key_reuse_through_the_engine_marks_the_request(self, exporter: Any) -> None:
        memory, tracer = exporter
        engine = Idempotent(MemoryStore(), hooks=TracingHooks(tracer))
        engine.run("k", lambda: "ok", payload={"amount": 100})

        with tracer.start_as_current_span("checkout"), pytest.raises(KeyReuseError):
            engine.run("k", lambda: "ok", payload={"amount": 999})

        assert by_name(memory, "checkout").status.status_code is trace.StatusCode.ERROR
