"""Observability hooks (#24).

The library knows things operators need and currently keeps them to itself:
how often a duplicate was actually suppressed, how many outcomes were left
unresolved, how long effects take. Reading it back out means querying the store
directly, which every caller then reimplements slightly differently.

This is a hook interface rather than a metrics dependency. justonce does not
know whether you use Prometheus, OpenTelemetry, StatsD or a log line, and a
correctness library that drags in a metrics client is a library people vendor
around. Subclass `Hooks`, override what you care about, pass it to the engine.

**Nothing here may change an outcome.** These callbacks run on the path that
charges money, so the engine invokes every one of them defensively: a hook that
raises is swallowed, and the effect, the record and the return value are exactly
what they would have been with no hooks at all. That is not politeness, it is
the only way a metrics backend being down is allowed to matter less than a
payment.

The consequence worth stating: an exception inside a hook is *lost*. If you need
to know your metrics are broken, the hook body has to be the thing that reports
it. justonce will not surface it for you, because the alternative is a failed
charge caused by a full disk on a metrics host.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .stores.base import Record


class Hooks:
    """No-op base. Override the events you want; the rest stay free.

    Every method returns `None` and the engine ignores return values, so a hook
    can never redirect control flow — only observe.
    """

    def duplicate_suppressed(self, key: str, record: Record) -> None:
        """A caller was handed a recorded outcome instead of running the effect.

        This is the number that says what the library is worth: it is how often
        a duplicate actually arrived and was made harmless.
        """

    def claim_conflict(self, key: str) -> None:
        """A caller lost the claim — someone else holds or held this key.

        Fires for every lost claim, including the ones that go on to replay a
        terminal record. A sustained rate with few `duplicate_suppressed` means
        callers are contending on in-flight work rather than replaying.
        """

    def key_reuse(self, key: str) -> None:
        """The same key arrived with a different payload.

        Never routine. It means two distinct intents derived one key, so one of
        them is about to be treated as a replay of the other and never applied.
        """

    def unknown_recorded(self, key: str) -> None:
        """An effect ran and its outcome could not be recorded.

        The reconciliation queue just grew. Pair with `oldest_unresolved_age`.
        """

    def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
        """The effect ran and its outcome is durable. `ok=False` means it raised.

        Fires *after* the outcome write, never before — a duration reported
        before the record lands would say "recorded" about a key a crash one
        line later leaves UNKNOWN, and a metric that errs toward reassurance is
        worse than no metric.
        """

    def ran_unguarded(self, key: str) -> None:
        """An effect ran with no claim behind it — `OnStoreUnavailable.FAIL_OPEN`.

        Idempotency was off for this call. `Result.guarded` tells the caller;
        this tells the operator, which is the one who can see it happening
        across every caller at once rather than one request at a time.
        """


def emit(hooks: Hooks | None, event: str, *args: object) -> None:
    """Call one hook, swallowing anything it raises.

    Both engines funnel every callback through here, so "a hook cannot break
    the effect" is one line rather than a convention each call site has to
    remember. `BaseException` rather than `Exception` on purpose: a
    `KeyboardInterrupt` raised inside a metrics client is still not a reason to
    abandon a charge whose outcome has already been recorded.
    """
    if hooks is None:
        return
    with contextlib.suppress(BaseException):
        getattr(hooks, event)(*args)
