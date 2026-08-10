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


class StoreError(JustOnceError):
    """The backing store could not be reached or returned something unusable.

    Deliberately *not* swallowed into "assume not seen". A store that cannot
    answer "has this run?" gives you no basis for running the effect.
    """
