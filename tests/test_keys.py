"""Key derivation and fingerprinting.

The wrong-key patterns tested here are the ones that appear in real codebases
and look correct in review.
"""

from __future__ import annotations

import pytest

from justonce import fingerprint, operation_key


def test_key_is_namespaced_and_versioned() -> None:
    assert operation_key("charge", "order_1") == "charge:v1:order_1"


def test_same_id_under_different_operations_does_not_collide() -> None:
    """A charge and a refund for one order are distinct intents."""
    assert operation_key("charge", "o1") != operation_key("refund", "o1")


def test_version_isolates_a_changed_derivation() -> None:
    assert operation_key("charge", "o1", version="v2") != operation_key("charge", "o1")


def test_multiple_parts_are_ordered_and_joined() -> None:
    assert operation_key("charge", "o1", "attempt_2") == "charge:v1:o1:attempt_2"


@pytest.mark.parametrize("bad", ["", None])
def test_empty_part_is_rejected(bad: object) -> None:
    """An empty component silently widens the key so distinct intents collide."""
    with pytest.raises(ValueError):
        operation_key("charge", bad)


def test_operation_alone_is_rejected() -> None:
    with pytest.raises(ValueError):
        operation_key("charge")


def test_empty_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        operation_key("", "o1")


# -- fingerprinting ---------------------------------------------------------

def test_fingerprint_is_order_independent() -> None:
    """A client that serialises differently on retry is not a key-reuse bug."""
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_detects_a_changed_value() -> None:
    assert fingerprint({"amount": 500}) != fingerprint({"amount": 501})


def test_fingerprint_is_stable_across_calls() -> None:
    payload = {"amount": 500, "currency": "USD", "items": [1, 2, 3]}
    assert fingerprint(payload) == fingerprint(payload)


def test_fingerprint_distinguishes_nesting() -> None:
    assert fingerprint({"a": {"b": 1}}) != fingerprint({"a": {"b": 2}})


def test_fingerprint_is_versioned() -> None:
    """So a serialisation change cannot be compared against old records."""
    assert fingerprint({}).startswith("1:")
