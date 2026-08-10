"""Idempotency key derivation and request fingerprinting.

The key must be **stable across retries of the same intent** and **different
across distinct intents**. Almost every idempotency bug in the wild is a key
that fails one of those two properties:

    uuid4()                 -> new key per attempt; every retry is a new charge
    f"{user}:{amount}"      -> two legitimate identical charges collapse into one
    hash(cart.contents)     -> key changes if the cart is edited mid-retry
    f"{order}:{time.time()}"-> a timestamp is just uuid4() wearing a hat

The rule: the key comes from the *initiating* event or the client, never from
the layer doing the retrying.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Bumping this invalidates every previously stored fingerprint, so the
#: divergence guard cannot compare across incompatible serialisations.
FINGERPRINT_VERSION = "1"


def operation_key(operation: str, *parts: Any, version: str = "v1") -> str:
    """Build a namespaced, versioned key from immutable identifiers.

        >>> operation_key("charge", "order_123", version="v1")
        'charge:v1:order_123'

    Namespacing by ``operation`` means the same order id can drive a charge and
    a refund independently. Versioning lets the derivation change later without
    colliding with historical keys.

    Raises:
        ValueError: if any part is empty or None — an empty component silently
            widens the key and lets unrelated intents collide.
    """
    if not operation:
        raise ValueError("operation must be a non-empty string")
    if not parts:
        raise ValueError(
            "at least one identifying part is required; a key of only an "
            "operation name would be shared by every caller"
        )
    rendered = []
    for part in parts:
        if part is None or part == "":
            raise ValueError(
                f"key part {part!r} is empty; an empty component makes distinct "
                "intents collide onto one key"
            )
        rendered.append(str(part))
    return ":".join([operation, version, *rendered])


def fingerprint(payload: Any) -> str:
    """Stable hash of a request payload, for the divergence guard.

    Dict ordering and whitespace must not change the fingerprint, or a caller
    that serialises differently on retry looks like a key-reuse bug.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_coerce,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{FINGERPRINT_VERSION}:{digest}"


def _coerce(value: Any) -> Any:
    """Best-effort JSON coercion for payloads holding non-primitive values."""
    for attr in ("isoformat", "__str__"):
        method = getattr(value, attr, None)
        if callable(method):
            return method()
    raise TypeError(f"cannot fingerprint value of type {type(value).__name__}")
