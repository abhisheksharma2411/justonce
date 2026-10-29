"""`Idempotency-Key` middleware for ASGI apps (#10).

The load-bearing tests here are the ones about what is *not* stored: a 5xx must
not be cached, and a header outside `store_headers` must not reach the ledger.
Both are places where a middleware that merely looks correct hands the next
caller something that belongs to the first one.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient

from justonce.integrations.asgi import IdempotencyMiddleware
from justonce.keys import operation_key
from justonce.stores import MemoryStore


def build(**options):
    """A FastAPI app counting how often each endpoint really ran."""
    calls = {"charge": 0, "flaky": 0, "read": 0, "cookie": 0}
    api = FastAPI()

    @api.post("/charge")
    def charge(body: dict):
        calls["charge"] += 1
        return {"charge_id": f"ch_{calls['charge']}", "amount": body.get("amount")}

    @api.post("/flaky")
    def flaky():
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise HTTPException(status_code=500, detail="provider timeout")
        return {"ok": True, "attempt": calls["flaky"]}

    @api.post("/cookie")
    def cookie(response: Response):
        calls["cookie"] += 1
        response.set_cookie("session", "super-secret")
        return {"ok": True}

    @api.get("/read")
    def read():
        calls["read"] += 1
        return {"calls": calls["read"]}

    options.setdefault("store", MemoryStore())
    return TestClient(IdempotencyMiddleware(api, **options)), calls


def post(client, path="/charge", key="k1", json=None):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post(path, json=json if json is not None else {"amount": 100}, headers=headers)


def test_a_repeated_key_replays_the_first_response_without_rerunning():
    client, calls = build()

    first = post(client)
    second = post(client)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"charge_id": "ch_1", "amount": 100}
    assert calls["charge"] == 1
    assert second.headers.get("idempotent-replay") == "true"
    assert "idempotent-replay" not in first.headers


def test_different_keys_are_different_operations():
    client, calls = build()

    assert post(client, key="k1").json()["charge_id"] == "ch_1"
    assert post(client, key="k2").json()["charge_id"] == "ch_2"
    assert calls["charge"] == 2


def test_one_key_on_two_routes_is_two_operations():
    client, calls = build()

    post(client, path="/charge", key="shared")
    post(client, path="/flaky", key="shared")

    assert calls["charge"] == 1
    assert calls["flaky"] == 1


def test_a_server_error_is_not_cached_so_the_retry_runs():
    client, calls = build()

    first = post(client, path="/flaky")
    second = post(client, path="/flaky")

    assert first.status_code == 500
    assert second.status_code == 200
    assert second.json() == {"ok": True, "attempt": 2}
    assert calls["flaky"] == 2
    # The successful retry is what gets recorded, so a third call replays it.
    assert post(client, path="/flaky").json() == {"ok": True, "attempt": 2}
    assert calls["flaky"] == 2


def test_the_same_key_with_a_different_body_is_refused():
    client, calls = build()

    post(client, json={"amount": 100})
    clash = post(client, json={"amount": 999})

    assert clash.status_code == 422
    assert calls["charge"] == 1


def test_set_cookie_is_neither_stored_nor_replayed():
    store = MemoryStore()
    client, _ = build(store=store)

    first = client.post("/cookie", headers={"Idempotency-Key": "c1"})
    second = client.post("/cookie", headers={"Idempotency-Key": "c1"})

    assert first.status_code == second.status_code == 200
    assert "session" in first.headers.get("set-cookie", "")
    # The replay must not hand the second caller the first caller's session.
    assert "super-secret" not in str(second.headers)

    # And it is absent because it was never written, not because the replay
    # path happened to drop it: read the ledger itself.
    record = store.lookup(operation_key("http", "POST", "/cookie", "c1"))
    assert record is not None, "the request should have been recorded"
    assert "super-secret" not in json.dumps(record.response)
    assert [name for name, _ in record.response["headers"]] == ["content-type"]


def test_an_unkeyed_request_passes_straight_through_by_default():
    client, calls = build()

    assert post(client, key=None).json()["charge_id"] == "ch_1"
    assert post(client, key=None).json()["charge_id"] == "ch_2"
    assert calls["charge"] == 2


def test_require_rejects_an_unkeyed_request():
    client, calls = build(require=True)

    response = post(client, key=None)

    assert response.status_code == 400
    assert "Idempotency-Key" in response.json()["detail"]
    assert calls["charge"] == 0


def test_an_empty_key_is_not_a_key():
    client, calls = build(require=True)

    assert post(client, key="   ").status_code == 400
    assert calls["charge"] == 0


def test_safe_methods_are_untouched():
    client, calls = build()

    # With a key, so this fails if GET is merely passing through for want of a
    # header rather than because the method is excluded.
    headers = {"Idempotency-Key": "same"}
    assert client.get("/read", headers=headers).json() == {"calls": 1}
    assert client.get("/read", headers=headers).json() == {"calls": 2}
    assert calls["read"] == 2


def test_should_handle_scopes_the_middleware_by_path():
    client, calls = build(should_handle=lambda scope: scope["path"].startswith("/charge"))

    # /cookie, not /flaky: a route that always succeeds, so "it ran twice" can
    # only mean the middleware left it alone. /flaky's 500 is not cached, which
    # would let it run twice either way and prove nothing.
    client.post("/cookie", headers={"Idempotency-Key": "s"})
    client.post("/cookie", headers={"Idempotency-Key": "s"})
    post(client, path="/charge", key="s")
    post(client, path="/charge", key="s")

    assert calls["cookie"] == 2
    assert calls["charge"] == 1


def test_content_type_survives_a_replay():
    client, _ = build()

    post(client)
    replay = post(client)

    assert replay.headers["content-type"] == "application/json"


def test_store_and_engine_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        IdempotencyMiddleware(FastAPI())
