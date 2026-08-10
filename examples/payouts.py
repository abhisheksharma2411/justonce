"""Partner payouts — the case this library was built for.

Run: python examples/payouts.py
"""

import justonce
from justonce.stores import SqliteStore

justonce.configure(SqliteStore(":memory:"))

transfers = []


class FlakyProvider:
    """Fails the first attempt the way a real one does: after doing the work."""

    def __init__(self):
        self.attempt = 0

    def transfer(self, partner, cents):
        self.attempt += 1
        transfers.append((partner, cents))          # the money has moved
        if self.attempt == 1:
            raise TimeoutError("gateway timed out after the transfer was applied")
        return {"transfer_id": f"tr_{self.attempt}"}


provider = FlakyProvider()


@justonce.idempotent(
    key=lambda payout_id, partner, cents: justonce.operation_key("payout", payout_id),
    retry_on_failure=True,
    return_result=True,
)
def issue_payout(payout_id, partner, cents):
    return provider.transfer(partner, cents)


print("attempt 1 — provider times out after moving the money")
try:
    issue_payout("p_1", "partner_a", 5_000)
except TimeoutError as exc:
    print(f"  raised: {exc}")

print("\nattempt 2 — the worker retries")
result = issue_payout("p_1", "partner_a", 5_000)
print(f"  executed={result.executed} value={result.value}")

print(f"\ntransfers actually sent: {len(transfers)}")
print("  ^ two, because `retry_on_failure=True` told us the failure was transient.")
print("    The provider lied: it had already moved the money before timing out.")
print("\nThis is why step 5 of the idempotency process matters: forward the key to")
print("the provider so IT can dedupe, and record intent BEFORE the call so an")
print("unknown outcome is visible to reconciliation instead of being retried blind.")
print("\nWith retry_on_failure=False the key is burned instead:")

provider2 = FlakyProvider()
transfers.clear()


@justonce.idempotent(
    key=lambda payout_id: justonce.operation_key("payout2", payout_id),
    retry_on_failure=False,
    return_result=True,
)
def strict_payout(payout_id):
    return provider2.transfer("partner_b", 5_000)


try:
    strict_payout("p_2")
except TimeoutError:
    print("  attempt 1 raised")
r = strict_payout("p_2")
print(f"  attempt 2: executed={r.executed} (key burned, no second transfer)")
print(f"  transfers sent: {len(transfers)}")
