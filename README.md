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


### Choosing a store

| Store | Use when | Not when |
|---|---|---|
| `SqliteStore("path.db")` | single host — local dev, tests, a single-instance deployment | you're running more than one machine or worker process |
| `PostgresStore` | a fleet of workers sharing state, no existing Django app | — |
| `DjangoStore` | you already run Django and want to reuse its connection | you haven't read the transaction caveat above |
| `MemoryStore()` | unit-testing a handler without a database | anything else — see below |

`MemoryStore` is a dict behind a lock. It passes the full conformance suite
including the 24-thread claim race, and it stores responses as JSON exactly as
the real stores do — so a `datetime` in a response fails in your test rather
than in production. But every key lives in one process's heap: two workers each
get a private view in which they both win the claim and both run the effect.
That is the failure this library exists to prevent, so it is for tests only.

**Durability matters more than it looks like it should, and the default is
not durable.** `SqliteStore()` defaults to `path=":memory:"`, so a store
constructed with no arguments forgets every key the moment the process
exits. That's fine for a REPL session or a test; it's a silent bug in
production, because it means the *one* scenario idempotency exists for — a
retry that arrives after a deploy or a crash — is exactly when the store
has no memory of what already happened.

If a key needs to survive process restarts, pass a real path
(`SqliteStore("justonce.db")`) or use `PostgresStore` or `DjangoStore`.

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

### When the store itself is down

The database is unreachable, so nothing can answer "has this already run?".
justonce **fails closed**: `claim()` raises `StoreError` and the request fails.

That is a stated decision, not an accident. It means an outage in the dedup
layer becomes an outage in whatever it guards — payments stop while the
database is down — and it is the only behaviour that cannot produce a duplicate
charge.

The alternative is available, opt-in, and named so nobody reaches it by
accident:

```python
justonce.configure(store, on_store_unavailable=justonce.OnStoreUnavailable.FAIL_OPEN)
```

**Fail-open runs the effect with nothing in the way, and duplicates become
possible.** Not unlikely — possible, and concentrated exactly where they hurt.
A store outage is when retries are most frequent, because everything upstream
is already erroring and retrying, and every one of those retries applies the
effect again. This is how a database outage becomes a finance incident. Choose
it only for effects that genuinely tolerate being applied twice.

Two things stay true even under fail-open:

* **It covers the claim only.** If the claim succeeded and the *outcome write*
  failed, the effect has already run and the honest answer is `UNKNOWN`, for
  reconciliation to resolve. Fail-open never converts that into a silent
  success.
* **It is never inferred.** Only `StoreError` opens the gate. A `KeyTooLongError`
  also comes out of `claim`, and it means the key is about to be truncated into
  a collision with a different intent — running the effect is the worst possible
  response to it, so it propagates exactly as it does today.

A run that happened without a claim says so, which is the only trace it leaves:

```python
result = engine.run(key, effect)
if not result.guarded:
    alert("effect ran with idempotency disabled", key=key)
```

### Clocks, and why the lease is not measured on yours

A claim's lease is written by one host and judged expired by another. If each
measured it against its own wall clock, the lease would mean different things to
each of them — and the atomic claim would not save you. Host B decides host A's
claim expired, A is still running the effect, both proceed, and the effect runs
twice.

So **every timestamp comes from the store's clock, never the caller's**. The
stores compute `now` inside SQL — `clock_timestamp()` on Postgres,
`UNIX_TIMESTAMP(NOW(6))` on MySQL, `julianday('now')` on SQLite — so the one
clock every host shares is the database's:

```python
store.now()          # the database server's clock, as a Unix timestamp
store.clock          # "store", or "process" for MemoryStore
```

NTP skew of a second or two is normal. Minutes happen after a VM resume or with
a broken time daemon, and a fifteen-minute default lease does not survive a
thirty-minute skew.

`sweep()` follows the same rule: called with no argument it defers to the store's
clock, because a sweeper on a fast host would otherwise delete records still
inside their retention window — and a swept record is a key the next delivery
cannot find, so the effect runs again.

`MemoryStore` is the exception and declares it with `clock = "process"`. It has
no clock but the caller's, which is harmless only because no second process can
reach it to disagree.

### Metrics, and the one to alert on

The library knows things you need and otherwise keeps them to itself. Subclass
`Hooks`, override what you care about, and wire it to whatever you already use —
there is no metrics dependency here, because a correctness library that drags in
a metrics client is one people vendor around.

```python
class Metrics(justonce.Hooks):
    def duplicate_suppressed(self, key, record):  # what the library is worth
        DUPES.inc()
    def ran_unguarded(self, key):                 # idempotency was OFF
        UNGUARDED.inc()
    def unknown_recorded(self, key):              # reconciliation queue grew
        UNKNOWN.inc()
    def effect_finished(self, key, duration_seconds, ok):
        DURATION.observe(duration_seconds)

justonce.configure(store, hooks=Metrics())
```

`claim_conflict` and `key_reuse` are there too. `key_reuse` is never routine: it
means two distinct intents derived one key, so one of them is about to be
treated as a replay of the other and never applied.

**A hook cannot change an outcome.** Every callback is invoked defensively, so a
metrics backend being down cannot fail a payment. The consequence is worth
stating plainly: an exception inside a hook is *lost*. If you need to know your
metrics are broken, the hook body has to be what reports it.

**Alert on the age, not the count:**

```python
age = engine.oldest_unresolved_age()   # seconds, or None when nothing is unresolved
```

A stuck reconciliation is invisible in a count that stays flat — the count only
moves when something new breaks. Age moves every second. It is measured on the
store's clock for the same reason leases are: computed against a host whose
clock runs fast, this gauge reports an age that never happened, and it is the
number the pager is attached to.

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

### Per-operation leases and replay windows

`ttl_seconds` and `retention_seconds` are engine-wide defaults, and a process
that runs effects of very different shapes should not have to pick one pair for
all of them — a card charge takes seconds and must stay replayable past the
dispute window; a nightly batch job takes an hour and is meaningless a day
later. One engine-wide value has to be the larger of the two, in both
directions.

Both can be overridden per call, and on the decorator:

```python
@justonce.idempotent(
    key=lambda order: operation_key("charge", order.id),
    ttl_seconds=60,                    # the charge times out well inside a minute
    retention_seconds=180 * 24 * 3600, # the dispute window
)
def charge_customer(order): ...

engine.run(key, rebuild_report, ttl_seconds=2 * 3600, retention_seconds=24 * 3600)
```

`None` — the default for both — means *not given*, so the engine's value is
used. It does not mean the store's "keep forever"; indefinite retention is a
decision that belongs at configuration time where it is visible.

Two things worth knowing before you reach for these:

* **The TTL to pick is the timeout you enforce on the effect, not how long it
  usually takes.** The check here only refuses a lease that is already expired
  (`ttl_seconds <= 0`). Nothing can tell from the outside that 30 seconds is
  too short for a call that hangs for 90, and a lease expiring under a live
  holder is how one effect becomes two.
* **Different TTLs at different call sites for the same key are safe.** Reclaim
  compares the `expires_at` the *holder* wrote, so a caller passing a short TTL
  cannot decide that someone else's long lease has expired. Its own TTL only
  sets the new expiry if it wins.

### Multi-tenancy

`operation_key("charge", order_id)` is global to the store. That is fine while
order ids are globally unique and **quietly wrong** the moment they are not: two
merchants with their own sequential numbering collide, one tenant's payment is
deduplicated against another's, and the effect never runs. No error, no log
line.

```python
justonce.configure(store, namespace=f"merchant:{merchant_id}")  # ✗ refused
justonce.configure(store, namespace=f"merchant-{merchant_id}")  # ✓
```

A namespace is prefixed to every key the engine touches, and it may not contain
`:`. That restriction is the feature, not a limitation of it — if `"a:b" + "c"`
and `"a" + "b:c"` both produced `a:b:c`, two tenants would collide again through
the very mechanism meant to keep them apart. Keys may contain as many colons as
they like; only the namespace may not.

Reads are scoped too. `unresolved()` and `oldest_unresolved_age()` return only
this tenant's records, with the prefix stripped so a key can go straight back
into `run()`. An engine with **no** namespace sees everything — that is the
operator's view, and a reconciliation worker that could not see every unresolved
effect would be worse than none.

`sweep()` is deliberately **not** scoped: it deletes records past their
retention window regardless of tenant, because retention is a property of the
store and a per-tenant sweeper would leave other tenants' expired rows to
accumulate forever.

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

### Key length

`SqliteStore` and `PostgresStore` store keys in unbounded `TEXT`, so length is not a concern. **MySQL is the exception**: the shipped DDL declares `key VARCHAR(255)`, and namespaced keys get long faster than you'd expect — `operation_key("charge", tenant_id, order_id, attempt_id)` with UUIDs is already past 140 characters.

Over-length keys are refused rather than stored, because the alternative is worse than an error:

```python
KeyTooLongError: idempotency key is 300 characters and mysql stores 255;
refusing to truncate it. Two keys sharing a 255-character prefix would
collapse onto one and the second intent would never run.
```

A truncated key is a *collided* key. The second intent is treated as a replay of the first, so its effect is **never applied** — a silently skipped payout, which nothing alerts on, unlike a duplicate one. Whether MySQL truncates or errors on its own depends on `sql_mode`, and a correctness guarantee cannot rest on a session variable.

If 255 is too tight, widen the column and tell the store:

```sql
ALTER TABLE justonce_keys MODIFY `key` VARCHAR(768) NOT NULL;
```

```python
DjangoStore(max_key_length=768)
```

768 is the widest a `VARCHAR` primary key can be under utf8mb4 — InnoDB caps index keys at 3072 bytes and utf8mb4 costs four bytes per character. The store can't detect the width for you: the DDL is `CREATE TABLE IF NOT EXISTS`, so an existing table keeps whatever it was created with, and guessing wide would reintroduce the truncation this prevents.

Writing your own store? Declare `max_key_length` if your key column has a fixed width, and leave it unset if it doesn't. [`justonce.conformance`](src/justonce/conformance.py) checks both cases.

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

## FastAPI payment example

The copyable [`examples/fastapi_payment.py`](examples/fastapi_payment.py) endpoint accepts an
`Idempotency-Key`, uses a credential-free payment provider, and returns the recorded response on
replay. Run it from the repository root:

```bash
uv run --extra examples uvicorn examples.fastapi_payment:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the successful, replay, key-reuse (422), and concurrent
in-flight (409) paths.

## Contributing

The core is small on purpose. Most of the value is at the edges, and that's where help is most useful:

- **A store for your database** — MySQL, DynamoDB, MongoDB, Redis, Spanner, D1. The contract is six methods, and [`justonce.conformance`](src/justonce/conformance.py) is an executable version of it. If your store passes the suite, it's correct by this project's definition.
- **A framework integration** — Django, Flask, Celery, Dramatiq, Airflow, FastAPI middleware.
- **A provider adapter** — map `justonce` keys onto Stripe, Adyen, Razorpay, PayPal native idempotency, so both sides agree on identity.

Adding a store is genuinely one file plus one conformance class. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [good first issues](https://github.com/abhisheksharma2411/justonce/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Licence

MIT
