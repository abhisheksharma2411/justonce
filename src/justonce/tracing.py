"""OpenTelemetry tracing hooks (#25).

    from justonce import configure, idempotent
    from justonce.tracing import TracingHooks

    configure(store, hooks=TracingHooks())

The trace this produces answers the question people actually ask during an
incident: *did this request do anything?* A replayed call gets
`justonce.deduplicated = true` on the span that is already open for the
request, so the answer is visible from the request's own trace rather than
inferred from the absence of a downstream call.

This is a `Hooks` subclass like any other, so it composes with whatever else
you pass and it inherits the engine's guarantee that a hook cannot change an
outcome. OpenTelemetry is an optional extra (`pip install justonce[otel]`);
nothing here is imported unless you import this module.

**Why the effect span is created after the fact.** `Hooks` has no
`effect_started` — it reports events, not a lifecycle — and adding one would
put a second callback on the money path for the sake of a span. Instead
`effect_finished` already carries the duration, so the span is created with an
explicit `start_time` of `end - duration` and ended immediately. The result is
a real span with the real duration, parented correctly, because hooks run
synchronously inside the caller's context. What it cannot do is show a *partial*
effect: an effect that never finishes produces no span at all. The claim is
visible either way, so an effect that hangs shows as a claim with nothing under
it, which is the shape worth recognising.

**What never reaches a span.** `Record.response` is whatever the effect
returned — a charge, an account number — and `request_hash` identifies the
payload. Neither is ever read here. The key is, since a trace without it cannot
be searched; `hash_keys=True` replaces it with a digest for keys that carry an
email address or an account number of their own.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from .hooks import Hooks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .stores.base import Record

#: Span and attribute names. Prefixed, because these end up in a trace store
#: alongside every other library's attributes.
EFFECT_SPAN = "justonce.effect"
ATTR_KEY = "justonce.key"
ATTR_DEDUPLICATED = "justonce.deduplicated"
ATTR_STATE = "justonce.state"
ATTR_ATTEMPTS = "justonce.attempts"
ATTR_GUARDED = "justonce.guarded"

#: Events recorded on the *caller's* current span rather than one of our own.
#: Each is a thing that happened to a request that is already being traced, so
#: it belongs in that request's timeline, not in a span of its own.
DUPLICATE_SUPPRESSED = "justonce.duplicate_suppressed"
CLAIM_CONFLICT = "justonce.claim_conflict"
KEY_REUSE = "justonce.key_reuse"
UNKNOWN_RECORDED = "justonce.unknown_recorded"
RAN_UNGUARDED = "justonce.ran_unguarded"


class TracingHooks(Hooks):
    """Emit OpenTelemetry spans and span events for idempotency decisions.

    Args:
        tracer: an OpenTelemetry ``Tracer``. Defaults to one named ``justonce``
            from the global provider, which is the right default precisely
            because it follows whatever the application configured later.
        hash_keys: replace the key with a short SHA-256 digest. Use it when
            keys are derived from an email address or an account number;
            duplicates still group by digest, so the trace stays searchable.
    """

    def __init__(self, tracer: Any = None, *, hash_keys: bool = False) -> None:
        if tracer is None:
            # Imported here, not at module import, so the error names the extra
            # rather than arriving as a bare ModuleNotFoundError from a library
            # the caller did not know they were loading.
            try:
                from opentelemetry import trace
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "justonce.tracing needs OpenTelemetry: pip install justonce[otel]"
                ) from exc
            tracer = trace.get_tracer("justonce")
        self.tracer = tracer
        self.hash_keys = hash_keys

    # -- the effect itself ------------------------------------------------

    def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
        end = time.time_ns()
        # Negative durations are not reachable from the engine, but a hook that
        # hands OpenTelemetry a span ending before it starts is a corrupt trace
        # rather than a wrong number, so clamp rather than trust.
        start = end - max(int(duration_seconds * 1_000_000_000), 0)
        span = self.tracer.start_span(
            EFFECT_SPAN,
            start_time=start,
            attributes={ATTR_KEY: self.display_key(key), ATTR_DEDUPLICATED: False},
        )
        if not ok:
            self._mark_error(span, "the effect raised")
        span.end(end_time=end)

    # -- things that happened to the caller's span -------------------------

    def duplicate_suppressed(self, key: str, record: Record) -> None:
        span = self._current_span()
        # The attribute, not only the event: an attribute is what a trace
        # backend can filter a whole service on, which is how you find out that
        # a fifth of your charge requests are replays.
        span.set_attribute(ATTR_DEDUPLICATED, True)
        span.add_event(
            DUPLICATE_SUPPRESSED,
            attributes={
                ATTR_KEY: self.display_key(key),
                ATTR_STATE: str(getattr(record.state, "value", record.state)),
                ATTR_ATTEMPTS: record.attempts,
            },
        )

    def claim_conflict(self, key: str) -> None:
        self._event(CLAIM_CONFLICT, key)

    def ran_unguarded(self, key: str) -> None:
        span = self._current_span()
        span.set_attribute(ATTR_GUARDED, False)
        span.add_event(RAN_UNGUARDED, attributes={ATTR_KEY: self.display_key(key)})

    # Both of these leave the request worse than it found it, so they mark the
    # span as well as annotating it: a key collision means an intent will never
    # be applied, and an unrecorded outcome means someone has to reconcile it
    # by hand. Neither is visible anywhere else in the trace.

    def key_reuse(self, key: str) -> None:
        span = self._event(KEY_REUSE, key)
        self._mark_error(span, "the same key arrived with a different payload")

    def unknown_recorded(self, key: str) -> None:
        span = self._event(UNKNOWN_RECORDED, key)
        self._mark_error(span, "the effect ran and its outcome was not recorded")

    # -- helpers ----------------------------------------------------------

    def display_key(self, key: str) -> str:
        """The key as it should appear in a span. Overridable."""
        if not self.hash_keys:
            return key
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _current_span(self) -> Any:
        from opentelemetry import trace

        return trace.get_current_span()

    def _event(self, name: str, key: str) -> Any:
        span = self._current_span()
        span.add_event(name, attributes={ATTR_KEY: self.display_key(key)})
        return span

    def _mark_error(self, span: Any, description: str) -> None:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, description))
