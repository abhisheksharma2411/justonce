"""Exceptions raised by justonce.

Every exception here represents a decision the caller has to make. None of them
should be caught and ignored — an ignored `KeyReuseError` in particular means
serving one caller's response to a different request.
"""

from __future__ import annotations


class JustOnceError(Exception):
    """Base class for every justonce error."""


class KeyReuseError(JustOnceError):
    """The same idempotency key arrived with a different request payload.

    This is a caller bug, not a race. Two distinct intents were given the same
    key, so there is no correct response to return: the recorded result belongs
    to the *other* request. Surfacing this loudly is the whole point — silently
    returning the stored response is how one customer receives another's data.

    Map this to HTTP 422 at an API boundary.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            f"idempotency key {key!r} was reused with a different request payload; "
            "refusing to return the recorded response"
        )
        self.key = key


class OperationInFlightError(JustOnceError):
    """Another caller holds the claim and has not finished.

    The first attempt's fate is unknown, which is exactly when duplicating is
    most likely to cause real damage. Retry later rather than proceeding.

    Map this to HTTP 409 at an API boundary.
    """

    def __init__(self, key: str) -> None:
        super().__init__(f"idempotency key {key!r} is already in progress")
        self.key = key


class InFlightTimeout(OperationInFlightError):
    """Waited for the in-flight holder to finish and it did not."""

    def __init__(self, key: str, waited_seconds: float) -> None:
        JustOnceError.__init__(
            self,
            f"idempotency key {key!r} was still in progress after "
            f"{waited_seconds:.1f}s",
        )
        self.key = key
        self.waited_seconds = waited_seconds


class KeyTooLongError(JustOnceError):
    """The key is longer than the backing column can store whole.

    Raised instead of letting the store truncate. Truncation is the dangerous
    outcome, not the error: two distinct keys sharing a long prefix collapse
    onto one, so a second, genuinely different intent is deduplicated away and
    **its effect never runs**. A silently skipped payout is worse than a
    duplicate one, because nothing alerts on it.

    Map this to HTTP 400 at an API boundary — the key is the caller's to fix.
    """

    def __init__(self, key: str, limit: int, backend: str) -> None:
        super().__init__(
            f"idempotency key is {len(key)} characters and {backend} stores "
            f"{limit}; refusing to truncate it. Two keys sharing a "
            f"{limit}-character prefix would collapse onto one and the second "
            f"intent would never run. Either shorten the key, or widen the "
            f"column and tell the store its new size with "
            f"max_key_length=. Key begins {key[:48]!r}"
        )
        self.key = key
        self.limit = limit
        self.backend = backend


class ResponseDecodeError(JustOnceError):
    """A stored response could not be decoded back into the recorded value.

    Raised on replay when the engine's codec cannot read a row it once wrote —
    a rotated encryption key, a codec swapped for an incompatible one, a
    corrupted column.

    Deliberately an error rather than a `None` response. "We cannot tell you
    what the provider said" is the truth, and `None` would let the caller read
    a real prior charge as "no body" — the one answer that is actively
    dangerous here. The original exception is kept as `__cause__`.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            f"stored response for {key!r} could not be decoded; the recorded outcome "
            "exists but cannot be read back. Check the engine's codec against the one "
            "that wrote this record before retrying the effect."
        )
        self.key = key


class StoreError(JustOnceError):
    """The backing store could not be reached or returned something unusable.

    Deliberately *not* swallowed into "assume not seen". A store that cannot
    answer "has this run?" gives you no basis for running the effect.
    """
