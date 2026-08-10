# Contributing

The core of this library is deliberately small — around 450 lines. Most of the value is at the edges, and that is where contributions matter most.

## The best first contribution: a store

Someone runs MySQL. Someone runs DynamoDB. Each of those is one file, and the contract is six methods.

1. Read [`src/justonce/stores/base.py`](src/justonce/stores/base.py) — the `Store` protocol, with the reasoning for each method.
2. Copy [`sqlite.py`](src/justonce/stores/sqlite.py) as a starting shape.
3. Add a class to `tests/test_store_conformance.py`:

```python
class TestMyStore(StoreConformanceTests):
    def make_store(self):
        return MyStore(dsn=os.environ["MY_DSN"])
```

4. Run it. If [`conformance.py`](src/justonce/conformance.py) passes, your store is correct by this project's definition, and that is all a reviewer needs to check.

Gate the test on an environment variable so the suite still runs for people without your database, as the Postgres store does.

### The one requirement that is not negotiable

**The claim must be atomic.** A `SELECT` followed by an `INSERT` is a race — two callers both read "not seen", both proceed, and the duplicate this library exists to prevent happens anyway. Use a unique constraint and let the database pick the winner.

`test_only_one_concurrent_claimer_wins` runs 24 threads at one key and asserts exactly one win. A store that fails it is not a store.

If your backend cannot enforce uniqueness atomically, say so in the PR rather than working around it. A store that is *nearly* correct is worse than no store, because someone will trust it with real money.

## Other useful work

- **Framework integrations** — Django, Flask, Celery, Dramatiq, Airflow, FastAPI middleware. The pattern is deriving a key from that framework's request or task context.
- **Provider adapters** — map our key onto Stripe, Adyen, Razorpay, PayPal, Braintree native idempotency, so both sides agree on identity. Without this, our dedup does not protect you from a duplicate you already sent.
- **A reconciliation CLI** — resolve `UNKNOWN` records against a provider's view.
- **Docs and examples** — a worked example against a real framework is worth more than API reference prose.

## Development

```bash
git clone https://github.com/abhisheksharma2411/justonce
cd justonce
uv venv && uv pip install -e ".[dev,postgres]"

pytest                    # SQLite only
ruff check src tests
mypy

# Postgres conformance
docker run -d --name justonce-pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
JUSTONCE_POSTGRES_DSN="postgresql://postgres:pw@localhost:5432/postgres" pytest
```

## Standards

**Every guarantee needs a test that fails when you remove the logic.** This is the project's central rule. If you cannot point at such a test, the behaviour is undefended and will be refactored away by someone who does not know it mattered.

**Tests describe the failure, not the mechanism.** `test_crash_after_effect_leaves_the_key_unresolved` tells a reader what breaks in production. `test_mark_unknown_called` does not.

**Comments explain why, not what.** The code says what it does. A comment earns its place by recording the reasoning that is not visible — why `<=` rather than `<`, why this cannot be a `SELECT ... FOR UPDATE`.

**No silent degradation.** If the library cannot determine whether an effect ran, it must say so. Guessing "probably fine" in a library that guards money movement is the one unforgivable design choice here.

## Pull requests

- One concern per PR. A store, or an integration, not both.
- Include the reproduction if you are fixing a bug — a failing test first, then the fix.
- State what you ran. `pytest`, `ruff check`, and the conformance suite for your store.
- Small is fine. A single well-tested store is a great contribution.

## Reporting a correctness bug

If you can produce a duplicate effect that this library should have prevented, that is the highest-value issue you can file. Include the store, the key derivation, and the sequence of calls. A reproducible duplicate will be treated as the most urgent thing in the tracker.
