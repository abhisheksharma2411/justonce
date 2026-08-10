"""A copyable FastAPI payment endpoint with idempotent replay protection.

Run from the repository root:

    uv run --extra examples uvicorn examples.fastapi_payment:app --reload

Then open http://127.0.0.1:8000/docs. No provider credentials are required.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import justonce
from justonce.stores import SqliteStore


class PaymentRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    amount_cents: int = Field(gt=0)


class StubPaymentProvider:
    """A deterministic stand-in for a provider such as Stripe or Adyen."""

    def __init__(self) -> None:
        self.charges: list[dict[str, Any]] = []

    def charge(self, customer_id: str, amount_cents: int) -> dict[str, Any]:
        charge = {
            "charge_id": f"ch_{len(self.charges) + 1}",
            "customer_id": customer_id,
            "amount_cents": amount_cents,
        }
        self.charges.append(charge)
        return charge


def create_app(
    *,
    store: SqliteStore | None = None,
    provider: StubPaymentProvider | None = None,
) -> FastAPI:
    # In-memory keys are lost on restart, so a retry after a deploy will charge
    # again. Use SqliteStore("effects.db") or PostgresStore(...) for anything real.
    store = store or SqliteStore(":memory:")
    provider = provider or StubPaymentProvider()
    engine = justonce.Idempotent(store)
    api = FastAPI(title="justonce payment example")

    @justonce.idempotent(
        key=lambda idempotency_key, payment: justonce.operation_key(
            "payment", idempotency_key
        ),
        payload=lambda idempotency_key, payment: payment.model_dump(),
        engine=engine,
        return_result=True,
    )
    def charge(idempotency_key: str, payment: PaymentRequest) -> Any:
        return provider.charge(payment.customer_id, payment.amount_cents)

    @api.post("/payments")
    def create_payment(
        payment: PaymentRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
    ) -> dict[str, Any]:
        try:
            result = charge(idempotency_key, payment)
        except justonce.OperationInFlightError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except justonce.KeyReuseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "charge": result.value,
            "executed": result.executed,
            "replayed": result.deduplicated,
        }

    api.state.store = store
    api.state.provider = provider
    return api


app = create_app()
