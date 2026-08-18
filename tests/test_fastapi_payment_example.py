from examples.fastapi_payment import PaymentRequest, StubPaymentProvider, create_app
from fastapi.testclient import TestClient

import justonce
from justonce.stores import SqliteStore


def test_payment_is_executed_once_and_replayed():
    provider = StubPaymentProvider()
    client = TestClient(create_app(store=SqliteStore(":memory:"), provider=provider))
    headers = {"Idempotency-Key": "checkout-123"}
    payload = {"customer_id": "customer-1", "amount_cents": 2500}

    first = client.post("/payments", headers=headers, json=payload)
    replay = client.post("/payments", headers=headers, json=payload)

    assert first.status_code == 200
    assert first.json()["executed"] is True
    assert replay.status_code == 200
    assert replay.json() == {
        "charge": first.json()["charge"],
        "executed": False,
        "replayed": True,
    }
    assert len(provider.charges) == 1


def test_key_reuse_with_a_different_payment_returns_422():
    client = TestClient(create_app(store=SqliteStore(":memory:")))
    headers = {"Idempotency-Key": "checkout-123"}

    client.post(
        "/payments",
        headers=headers,
        json={"customer_id": "customer-1", "amount_cents": 2500},
    )
    response = client.post(
        "/payments",
        headers=headers,
        json={"customer_id": "customer-1", "amount_cents": 5000},
    )

    assert response.status_code == 422
    assert "different request payload" in response.json()["detail"]


def test_in_flight_payment_returns_409():
    store = SqliteStore(":memory:")
    client = TestClient(create_app(store=store))
    payment = PaymentRequest(customer_id="customer-1", amount_cents=2500)
    key = justonce.operation_key("payment", "checkout-123")
    store.claim(key, justonce.fingerprint(payment.model_dump()), ttl_seconds=60)

    response = client.post(
        "/payments",
        headers={"Idempotency-Key": "checkout-123"},
        json=payment.model_dump(),
    )

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]
