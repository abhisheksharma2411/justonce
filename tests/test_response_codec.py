"""Pluggable response encoding (#48, and #29 with it).

`complete(key, response)` serialises whatever the effect returned straight into
the store. For the use case this library exists for, that response is a payment
provider's reply — last four digits, cardholder name, billing address, email —
and it lands in a table in plaintext for a retention window we correctly tell
people to make long. We were storing cardholder data for thirty days without
ever saying so.

A codec sits on the engine rather than on each store, so the four bundled
stores need no change: they keep JSON-encoding whatever the codec hands them.
That also means a third-party store inherits it for free, which a per-store
hook would not.

Three properties carry this, and each is a way for a codec to be worse than no
codec at all:

  * **The default must be byte-identical to today.** An encoding change that
    lands silently makes every stored record from before the upgrade
    unreadable.
  * **A codec must not touch dedup.** The divergence guard fingerprints the
    *request*; encrypting the *response* must not move a single claim decision.
  * **A failed decode must not look like an absent response.** If the key
    rotated, "we cannot tell you what the provider said" is the truth, and
    returning `None` would let a caller read a real prior charge as "no body".
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from justonce import (
    Idempotent,
    JsonResponseCodec,
    KeyReuseError,
    ResponseCodec,
    ResponseDecodeError,
    operation_key,
)
from justonce.asyncio import AsyncIdempotent
from justonce.stores import MemoryStore, SqliteStore

PROVIDER_REPLY = {"id": "ch_1", "last4": "4242", "name": "A Cardholder", "email": "a@b.test"}


class ReversingCodec(ResponseCodec):
    """Stands in for a real encryptor: cheap, reversible, obviously not JSON."""

    def encode(self, value: Any) -> Any:
        raw = json.dumps(value, sort_keys=True).encode()
        return {"v": 1, "ct": base64.b64encode(raw).decode()}

    def decode(self, stored: Any) -> Any:
        return json.loads(base64.b64decode(stored["ct"]))


class RotatedKeyCodec(ReversingCodec):
    """Encodes fine, cannot decode — a key that rotated under stored rows."""

    def decode(self, stored: Any) -> Any:
        raise ValueError("no key for ciphertext version 1")


def stored_blob(store: MemoryStore, key: str) -> Any:
    record = store.lookup(key)
    return None if record is None else record.response


class TestTheDefaultIsUnchanged:
    def test_no_codec_stores_exactly_what_it_used_to(self) -> None:
        store = MemoryStore()
        Idempotent(store).run("k", lambda: PROVIDER_REPLY)
        assert stored_blob(store, "k") == PROVIDER_REPLY

    def test_the_explicit_json_codec_matches_the_default(self) -> None:
        plain, explicit = MemoryStore(), MemoryStore()
        Idempotent(plain).run("k", lambda: PROVIDER_REPLY)
        Idempotent(explicit, codec=JsonResponseCodec()).run("k", lambda: PROVIDER_REPLY)
        assert stored_blob(plain, "k") == stored_blob(explicit, "k")

    def test_a_record_written_without_a_codec_still_replays(self) -> None:
        """The upgrade path: rows predate the codec and must keep working."""
        store = MemoryStore()
        Idempotent(store).run("k", lambda: PROVIDER_REPLY)
        replay = Idempotent(store).run("k", lambda: {"never": "runs"})
        assert replay.deduplicated is True
        assert replay.value == PROVIDER_REPLY


class TestEncodingAtRest:
    def test_the_plaintext_is_not_in_the_store(self) -> None:
        store = MemoryStore()
        Idempotent(store, codec=ReversingCodec()).run("k", lambda: PROVIDER_REPLY)

        blob = json.dumps(stored_blob(store, "k"))
        assert "4242" not in blob
        assert "A Cardholder" not in blob
        assert "a@b.test" not in blob

    def test_the_caller_still_gets_the_real_response_back(self) -> None:
        store = MemoryStore()
        engine = Idempotent(store, codec=ReversingCodec())
        engine.run("k", lambda: PROVIDER_REPLY)
        assert engine.run("k", lambda: {"never": "runs"}).value == PROVIDER_REPLY

    def test_the_executing_call_returns_the_live_value_unencoded(self) -> None:
        engine = Idempotent(MemoryStore(), codec=ReversingCodec())
        assert engine.run("k", lambda: PROVIDER_REPLY).value == PROVIDER_REPLY

    def test_a_codec_survives_a_real_store(self) -> None:
        store = SqliteStore(":memory:")
        engine = Idempotent(store, codec=ReversingCodec())
        engine.run("k", lambda: PROVIDER_REPLY)
        assert engine.run("k", lambda: {"never": "runs"}).value == PROVIDER_REPLY


class TestACodecDoesNotTouchDedup:
    def test_the_effect_still_runs_once(self) -> None:
        calls: list[int] = []
        engine = Idempotent(MemoryStore(), codec=ReversingCodec())
        engine.run("k", lambda: calls.append(1) or PROVIDER_REPLY)
        engine.run("k", lambda: calls.append(2) or PROVIDER_REPLY)
        assert calls == [1]

    def test_key_reuse_is_still_detected(self) -> None:
        """The divergence guard fingerprints the *request*. Encoding the
        response must not move a single claim decision."""
        engine = Idempotent(MemoryStore(), codec=ReversingCodec())
        engine.run("k", lambda: PROVIDER_REPLY, payload={"amount": 100})
        with pytest.raises(KeyReuseError):
            engine.run("k", lambda: PROVIDER_REPLY, payload={"amount": 200})

    def test_the_fingerprint_is_identical_with_and_without_a_codec(self) -> None:
        plain, coded = MemoryStore(), MemoryStore()
        key = operation_key("charge", "o1")
        Idempotent(plain).run(key, lambda: PROVIDER_REPLY, payload={"amount": 100})
        Idempotent(coded, codec=ReversingCodec()).run(
            key, lambda: PROVIDER_REPLY, payload={"amount": 100}
        )
        assert plain.lookup(key).request_hash == coded.lookup(key).request_hash  # type: ignore[union-attr]


class TestAFailedDecodeIsNotAnAbsentResponse:
    def test_a_rotated_key_raises_rather_than_replaying_none(self) -> None:
        """"We cannot tell you what the provider said" is the truth. Returning
        `None` would let a caller read a real prior charge as "no body", which
        is the one answer that is actively dangerous here."""
        store = MemoryStore()
        Idempotent(store, codec=ReversingCodec()).run("k", lambda: PROVIDER_REPLY)

        broken = Idempotent(store, codec=RotatedKeyCodec())
        with pytest.raises(ResponseDecodeError) as caught:
            broken.run("k", lambda: {"never": "runs"})
        assert "k" in str(caught.value)

    def test_the_underlying_error_is_kept_for_diagnosis(self) -> None:
        store = MemoryStore()
        Idempotent(store, codec=ReversingCodec()).run("k", lambda: PROVIDER_REPLY)
        with pytest.raises(ResponseDecodeError) as caught:
            Idempotent(store, codec=RotatedKeyCodec()).run("k", lambda: None)
        assert isinstance(caught.value.__cause__, ValueError)

    def test_a_failed_decode_does_not_re_run_the_effect(self) -> None:
        calls: list[int] = []
        store = MemoryStore()
        Idempotent(store, codec=ReversingCodec()).run("k", lambda: PROVIDER_REPLY)
        with pytest.raises(ResponseDecodeError):
            Idempotent(store, codec=RotatedKeyCodec()).run("k", lambda: calls.append(1))
        assert calls == []


class TestStoreResponseFalse:
    def test_nothing_of_the_response_reaches_the_store(self) -> None:
        store = MemoryStore()
        Idempotent(store, store_response=False).run("k", lambda: PROVIDER_REPLY)
        assert stored_blob(store, "k") is None

    def test_dedup_still_works(self) -> None:
        calls: list[int] = []
        engine = Idempotent(MemoryStore(), store_response=False)
        engine.run("k", lambda: calls.append(1) or PROVIDER_REPLY)
        engine.run("k", lambda: calls.append(2) or PROVIDER_REPLY)
        assert calls == [1]

    def test_the_executing_call_still_returns_the_live_value(self) -> None:
        result = Idempotent(MemoryStore(), store_response=False).run("k", lambda: PROVIDER_REPLY)
        assert result.value == PROVIDER_REPLY
        assert result.executed is True

    def test_a_replay_says_the_response_is_unavailable(self) -> None:
        """`value is None` on its own is ambiguous — the effect may genuinely
        have returned None. The flag is how a caller tells the two apart."""
        engine = Idempotent(MemoryStore(), store_response=False)
        engine.run("k", lambda: PROVIDER_REPLY)
        replay = engine.run("k", lambda: {"never": "runs"})

        assert replay.deduplicated is True
        assert replay.value is None
        assert replay.response_available is False

    def test_an_ordinary_replay_reports_the_response_as_available(self) -> None:
        engine = Idempotent(MemoryStore())
        engine.run("k", lambda: PROVIDER_REPLY)
        assert engine.run("k", lambda: None).response_available is True

    def test_an_effect_that_really_returned_none_is_not_confused_with_it(self) -> None:
        engine = Idempotent(MemoryStore())
        engine.run("k", lambda: None)
        replay = engine.run("k", lambda: "never")
        assert replay.value is None
        assert replay.response_available is True

    def test_the_executing_call_reports_availability_honestly(self) -> None:
        assert Idempotent(MemoryStore()).run("k", lambda: 1).response_available is True
        assert (
            Idempotent(MemoryStore(), store_response=False).run("k", lambda: 1).response_available
            is True
        ), "the live value is in hand on the call that ran the effect"


class TestAsyncParity:
    async def test_encoding_at_rest(self) -> None:
        store = MemoryStore()
        engine = AsyncIdempotent(store, codec=ReversingCodec())

        async def effect() -> Any:
            return PROVIDER_REPLY

        await engine.run("k", effect)
        assert "4242" not in json.dumps(stored_blob(store, "k"))
        assert (await engine.run("k", effect)).value == PROVIDER_REPLY

    async def test_rotated_key_raises(self) -> None:
        store = MemoryStore()

        async def effect() -> Any:
            return PROVIDER_REPLY

        await AsyncIdempotent(store, codec=ReversingCodec()).run("k", effect)
        with pytest.raises(ResponseDecodeError):
            await AsyncIdempotent(store, codec=RotatedKeyCodec()).run("k", effect)

    async def test_store_response_false(self) -> None:
        store = MemoryStore()
        engine = AsyncIdempotent(store, store_response=False)

        async def effect() -> Any:
            return PROVIDER_REPLY

        await engine.run("k", effect)
        replay = await engine.run("k", effect)
        assert stored_blob(store, "k") is None
        assert replay.response_available is False
