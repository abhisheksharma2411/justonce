"""How a response is shaped on its way into the store, and back out (#48, #29).

`complete(key, response)` used to serialise whatever the effect returned
straight into the store. For the use case this library exists for, that
response is a payment provider's reply — last four digits, cardholder name,
billing address, email, sometimes a whole customer object. It landed in a table
in plaintext and stayed there for the retention window, which we correctly tell
people to make long. So the library was storing cardholder data for thirty days
without ever saying so.

A codec sits on the **engine**, not on each store. The stores keep JSON-encoding
whatever the codec hands them, so none of the four bundled ones needed a change
and a third-party store inherits this for free — which a per-store hook would
not have given.

Two uses, one mechanism, which is what #48 asked for:

* **Encryption at rest.** Hand back a ciphertext envelope; the store persists
  that and never sees the plaintext.
* **Values JSON cannot carry** (#29) — a `Decimal`, a `datetime`, a dataclass.
  Convert to a JSON-safe shape on the way in and back on the way out.

The rule for any codec: `decode(encode(x))` must equal `x` for every value the
effect can return. A codec that is lossy in either direction silently changes
what a replay hands a caller, and a replay is the answer to "did this already
happen, and what did it say".
"""

from __future__ import annotations

from typing import Any


class ResponseCodec:
    """Identity by default: what the engine did before this existed.

    Subclass and override both halves. The default is deliberately a no-op
    rather than JSON, because the stores already do the JSON step — a codec
    that encoded to text here would leave the store to encode the text again.
    """

    def encode(self, value: Any) -> Any:
        """The effect's return value, as the store should hold it."""
        return value

    def decode(self, stored: Any) -> Any:
        """The inverse of `encode`, applied to what the store gave back."""
        return stored


class JsonResponseCodec(ResponseCodec):
    """The default, named. Behaviourally identical to passing no codec at all.

    Exists so a caller can be explicit in configuration, and so the docs have
    something to point at when they say "this is what you get unless you change
    it". Asserting that it matches the default is a test, not a comment.
    """
