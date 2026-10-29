"""Structured logging hooks (#52).

During an incident the questions are immediate — *did this key dedupe? was it
refused? how long did the effect take?* — and until now the only way to answer
them was to query the store. `LoggingHooks` answers them from the log stream
instead, by implementing the existing `Hooks` interface rather than inventing a
second one, so metrics and logs see exactly the same events.

    import logging
    from justonce import Idempotent
    from justonce.logging import LoggingHooks

    logging.basicConfig(level=logging.INFO)
    engine = Idempotent(store, hooks=LoggingHooks())

Every record carries its fields in `extra`, so a JSON formatter emits them as
real fields and a plain formatter still prints a readable line. Nothing here
imports anything outside the standard library.

**Payloads and responses are never logged, and cannot be.** `duplicate_suppressed`
is handed a whole `Record`, whose `response` holds whatever the effect returned —
a charge, an account number, a customer's address. This module reads `state`,
`attempts` and the timestamps off that record and never touches `response` or
`request_hash`. If you want the response in your logs you have to go and get it
yourself, which is the right amount of friction for payment data.

The key itself is logged in full by default, because "did *this* key dedupe" is
unanswerable without it. Keys derived from an email address or a customer
reference carry that identifier into your log index, so `hash_keys=True`
replaces it with a truncated digest: still stable enough to correlate two lines
about one key, no longer the identifier itself.

Like every `Hooks` implementation, this one runs on the path that charges
money, and `justonce.hooks.emit` swallows whatever it raises. A broken log
handler cannot fail a payment — and equally, it will not tell you it is broken.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from .hooks import Hooks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .stores.base import Record

__all__ = ["DEFAULT_LEVELS", "LOGGER_NAME", "LoggingHooks"]

LOGGER_NAME = "justonce"

#: Default level per event. `effect_finished` is split in two: a successful
#: effect is the ordinary case and logging one line per call at INFO would
#: drown the events that matter, while a failed one is worth seeing.
DEFAULT_LEVELS: dict[str, int] = {
    "duplicate_suppressed": logging.INFO,
    "claim_conflict": logging.INFO,
    "key_reuse": logging.WARNING,
    "unknown_recorded": logging.ERROR,
    "effect_succeeded": logging.DEBUG,
    "effect_failed": logging.WARNING,
    "ran_unguarded": logging.WARNING,
}


class LoggingHooks(Hooks):
    """Emit one structured log record per engine event.

    Args:
        logger: where to send records. Defaults to `logging.getLogger("justonce")`,
            so an application can route or silence the library by name without
            touching this object.
        levels: per-event overrides, keyed by the names in `DEFAULT_LEVELS`. An
            unknown name raises, rather than being accepted and never used —
            a silently ignored override reads as "I turned that down" when
            nothing was turned down.
        hash_keys: log a truncated SHA-256 of each key instead of the key.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        levels: dict[str, int] | None = None,
        hash_keys: bool = False,
    ) -> None:
        self.logger = logger if logger is not None else logging.getLogger(LOGGER_NAME)
        self.levels = dict(DEFAULT_LEVELS)
        if levels:
            unknown = sorted(set(levels) - set(DEFAULT_LEVELS))
            if unknown:
                raise ValueError(
                    f"unknown event(s) {unknown}; expected one of {sorted(DEFAULT_LEVELS)}"
                )
            self.levels.update(levels)
        self.hash_keys = hash_keys

    def duplicate_suppressed(self, key: str, record: Record) -> None:
        self._log(
            "duplicate_suppressed",
            key,
            state=getattr(record.state, "value", record.state),
            attempts=record.attempts,
        )

    def claim_conflict(self, key: str) -> None:
        self._log("claim_conflict", key)

    def key_reuse(self, key: str) -> None:
        self._log("key_reuse", key)

    def unknown_recorded(self, key: str) -> None:
        self._log("unknown_recorded", key)

    def effect_finished(self, key: str, duration_seconds: float, ok: bool) -> None:
        self._log(
            "effect_succeeded" if ok else "effect_failed",
            key,
            duration_seconds=round(duration_seconds, 6),
        )

    def ran_unguarded(self, key: str) -> None:
        self._log("ran_unguarded", key)

    def display_key(self, key: str) -> str:
        """How `key` appears in the log. Override to redact differently."""
        if not self.hash_keys:
            return key
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _log(self, event: str, key: str, **fields: Any) -> None:
        level = self.levels[event]
        # A cost saving, not behaviour: `Logger.log` filters by level itself,
        # so removing this changes no output and no test pins it. What it saves
        # is the work below — a SHA-256 under `hash_keys`, the detail string and
        # the dict — on the common path where `effect_succeeded` sits at DEBUG
        # and nothing is listening, once per charge.
        if not self.logger.isEnabledFor(level):
            return
        shown = self.display_key(key)
        extra = {"event": event, "key": shown, **fields}
        detail = "".join(f" {name}={value}" for name, value in fields.items())
        self.logger.log(level, "justonce %s key=%s%s", event, shown, detail, extra=extra)
