# justonce

**Make side effects happen exactly once.**

Your code charges a customer. The network hiccups. Your retry logic fires. The customer is charged twice.

```python
def charge_customer(order):
    return payments.charge(order.customer, order.total)   # 💸 twice
```

```python
from justonce import idempotent, operation_key

@idempotent(key=lambda order: operation_key("charge", order.id))
def charge_customer(order):
    return payments.charge(order.customer, order.total)   # ✅ once
```

That's it. The function now runs **at most once per order** — across retries, restarts, queue replays, and concurrent workers on different machines. And the outcome is recorded, so later you can ask *"did anything get charged twice yesterday?"* and get a real answer instead of a guess.

---

## Why this exists

Every network call has three outcomes, not two: **success**, **failure**, and **unknown**. The unknown one — a timeout, a dropped connection, a process killed mid-write — is what creates duplicates, because the only safe response to "unknown" is to retry, and retrying something that already applied applies it twice.

"Exactly-once delivery" does not exist. What is achievable is **at-least-once delivery with idempotent processing**, which produces exactly-once *effects*. You cannot stop the duplicate arriving. This library makes it harmless.

The duplicates that matter are the irreversible ones: money moves, an email sends, inventory decrements, a webhook fires. Those aren't latency bugs — they're correctness bugs that reach customers, and they're usually discovered by finance rather than by monitoring.

## Install

```bash
pip install justonce                 # SQLite included, no dependencies
pip install justonce[postgres]       # + Postgres store
pip install justonce[django]         # + Django store, uses your existing connection
```

## Usage

```python
import justonce
from justonce.stores import SqliteStore

justonce.configure(SqliteStore("effects.db"))

@justonce.idempotent(key=lambda order: justonce.operation_key("charge", order.id))
def charge_customer(order):
    return payments.charge(order.customer, order.total)
```

For a fleet, swap the store — nothing else changes:

```python
from justonce.stores import PostgresStore
justonce.configure(PostgresStore("postgresql://localhost/app"))
```

Already on Django? Use the connection you have, on any backend Django supports:

```python
from justonce.stores.django_store import DjangoStore
justonce.configure(DjangoStore())
```

One caveat worth reading before you ship it: a store on the default alias joins
your ambient `transaction.atomic()` block. That is correct when the effect is a
local write — claim and effect roll back together. It is **wrong when the effect
is an external call**, because a rollback erases the claim while the charge
stands, and the retry charges again. Point the store at a separate database
alias in that case, and `store.in_ambient_transaction()` will tell you which
mode you are actually in.

### Knowing whether *this* call did the work

```python
@justonce.idempotent(key=..., return_result=True)
def charge_customer(order): ...

result = charge_customer(order)
if result.deduplicated:
    log.info("already charged", extra={"response": result.value})
```

### Handling the in-flight duplicate

Another worker holds the claim and hasn't finished. Choose deliberately:

```python
justonce.configure(store, on_in_flight=justonce.OnInFlight.RAISE)   # 409, default
justonce.configure(store, on_in_flight=justonce.OnInFlight.WAIT)    # block for the result
```

Never let a second caller proceed because the first "seems stuck" — a stalled attempt whose fate is unknown is exactly when duplicating is most expensive.

### Reconciliation

Prevention is never complete. When a process dies *between* the effect and recording it, the key is left `UNKNOWN` rather than cleaned up — because "we don't know whether the customer was charged" is a fact worth keeping.

```python
for record in engine.unresolved():
    outcome = payments.lookup(idempotency_key=record.key)   # ask the provider
    ...
```

Alert on the **age** of the oldest unresolved record, not the count. A stuck reconciliation is invisible in a count that stays flat.

### Retention

```python
justonce.configure(store, retention_seconds=30 * 24 * 3600)
engine.sweep()   # nightly
```

Retention is a correctness parameter, not a storage optimisation. It must outlive the longest chain that can re-deliver the same intent — including a dead-letter queue replayed a week later, and any provider dispute window. A 24-hour TTL behind a 7-day DLQ is a duplicate waiting to happen.

## Choosing a key

The key must be **stable across retries of the same intent** and **different across distinct intents**. Nearly every idempotency bug is a key that breaks one of those:

```python
uuid4()                        # ✗ new key per attempt — every retry is a new charge
f"{user_id}:{amount}"          # ✗ two legitimate $50 charges collapse into one
hash(cart.contents)            # ✗ key changes if the cart is edited mid-retry
f"{order.id}:{time.time()}"    # ✗ a timestamp is uuid4() wearing a hat

operation_key("charge", order.id)          # ✓ derived from an immutable identifier
request.headers["Idempotency-Key"]         # ✓ client-supplied, reused on retry
```

The key comes from the **initiating event or the client** — never from the layer doing the retrying.

## What it guarantees

| Situation | Behaviour |
|---|---|
| Same key, same payload, called again | Effect runs once; recorded response returned |
| Same key, **different** payload | `KeyReuseError` — never serves the wrong response |
| Two workers, same key, same instant | Exactly one runs the effect |
| Crash *after* effect, *before* recording | Key left `UNKNOWN`; retries refuse until reconciled |
| Effect raised a transient error | Claim released; a later attempt may retry |
| Effect raised a permanent error | Key burned; no retry |
| Holder died mid-flight | Claim reclaimable once its lease expires |

Each row is a test in [`tests/test_exactly_once.py`](tests/test_exactly_once.py). If a guarantee isn't defended by a test that fails when you remove the logic, it isn't a guarantee.

## How it works

```
claim ──won──> run effect ──> record outcome ──> return
  │
  └──lost──> terminal?  ──> return recorded response
             in-flight? ──> reject, or wait
             unknown?   ──> refuse; reconcile
```

The claim is a single atomic write guarded by a unique constraint — `INSERT ... ON CONFLICT DO NOTHING`. The database picks the winner. There is no `SELECT` before it, because a check followed by an act is a race:

```python
# ✗ TOCTOU: both callers read "not seen", both charge
if not db.exists(key):
    charge_card(amount)
    db.insert(key)
```

The unique constraint *is* the mechanism. If a backend can't enforce uniqueness atomically, it can't be a store.

## Compared to durable execution

Temporal, Restate, DBOS and friends solve a broader problem, and solve it well — but they ask you to restructure your application into workflows, which is why adoption stalls in existing codebases.

|  | Durable execution | justonce |
|---|---|---|
| Unit of protection | The workflow | One function call |
| Adoption cost | Rewrite the app | Add a decorator |
| Runtime required | A server or cluster | A table |
| "What did this actually do?" | Via workflow history | The core primitive |

Use justonce when you want *one dangerous call* made safe this afternoon. Use a workflow engine when you need orchestration, timers, and long-running state.

## Contributing

The core is small on purpose. Most of the value is at the edges, and that's where help is most useful:

- **A store for your database** — MySQL, DynamoDB, MongoDB, Redis, Spanner, D1. The contract is six methods, and [`justonce.conformance`](src/justonce/conformance.py) is an executable version of it. If your store passes the suite, it's correct by this project's definition.
- **A framework integration** — Django, Flask, Celery, Dramatiq, Airflow, FastAPI middleware.
- **A provider adapter** — map `justonce` keys onto Stripe, Adyen, Razorpay, PayPal native idempotency, so both sides agree on identity.

Adding a store is genuinely one file plus one conformance class. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [good first issues](https://github.com/abhisheksharma2411/justonce/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Licence

MIT
