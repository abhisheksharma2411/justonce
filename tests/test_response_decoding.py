"""Decoding a stored response back into the object that was recorded.

Every bundled store writes the response as `json.dumps(...)`, but drivers
disagree about who reads it back. psycopg decodes a `jsonb` column for you;
Django's Postgres backend leaves `jsonb` as text so its own field decoder can
run; a `TEXT` column is a string everywhere. A store that assumes exactly one of
those returns a JSON string where the contract promises the response object —
and the caller compares it to what they passed in and sees a mismatch with no
explanation.
"""

from __future__ import annotations

import json

import pytest

from justonce.stores.base import decode_response


def test_absent_response_stays_none() -> None:
    """`None` means no response was recorded — not the JSON literal `null`."""
    assert decode_response(None) is None


def test_json_text_is_parsed() -> None:
    """The TEXT-column case, and Django's Postgres backend."""
    assert decode_response('{"charge_id": "ch_1"}') == {"charge_id": "ch_1"}


def test_bytes_are_decoded_as_utf8_json() -> None:
    assert decode_response(b'{"note": "caf\xc3\xa9"}') == {"note": "café"}


def test_memoryview_is_decoded() -> None:
    """Some drivers hand back a buffer rather than bytes."""
    assert decode_response(memoryview(b'[1, 2, 3]')) == [1, 2, 3]


@pytest.mark.parametrize(
    "value",
    [
        {"nested": {"a": [1, 2, {"b": None}]}},
        [1, "two", 3.5, True, None],
        "plain string",
        42,
        3.5,
        True,
        None,
        [],
        {},
    ],
)
def test_every_json_shape_survives_a_full_round_trip(value: object) -> None:
    """What a store writes is what a caller must get back.

    This is the property the `response` column exists to hold: `complete()`
    takes an arbitrary JSON-serialisable payload, and a retry has to see that
    exact payload again.
    """
    assert decode_response(json.dumps(value)) == value


def test_a_recorded_string_survives_exactly_one_decode() -> None:
    """The case that makes type-sniffing impossible, pinned.

    A recorded payload of `"done"` is stored as the text `'"done"'` and decodes
    to `'done'` — which is itself a string, and not parseable as JSON. Any
    reader that guesses whether to decode based on the value's type gets this
    wrong in one direction or the other, which is why the stores select the
    column as text and decode here unconditionally.
    """
    assert decode_response(json.dumps("done")) == "done"


def test_json_text_stored_as_a_payload_is_not_parsed_twice() -> None:
    """A payload that happens to *look* like JSON is still just a string."""
    payload = '{"looks": "like json"}'
    assert decode_response(json.dumps(payload)) == payload


def test_malformed_stored_json_raises_rather_than_returning_garbage() -> None:
    """A corrupt row is worth a loud failure — this column decides retries."""
    with pytest.raises(json.JSONDecodeError):
        decode_response("{not json")
