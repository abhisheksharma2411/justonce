# Changelog

All notable changes to `justonce` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-21

### Fixed

**If you use the SQLite or Postgres store, upgrade.** Both defects below could
let a retry produce a second effect — the failure mode this library exists to
prevent.

- **`retention_seconds` was ignored, so keys were forgotten early.** A completed
  record kept the *claim lease's* `expires_at`, and `sweep()` reads that column
  for terminal records too. A record was therefore deleted one claim-TTL after it
  was written, however long retention was configured for. Once the key is gone, a
  retry arriving inside the intended retention window is indistinguishable from a
  first attempt and executes again. `expires_at` is now replaced when a record
  becomes terminal, and a terminal record written with no retention is never
  swept ([#39], [#59]).
- **An expired lease could be reclaimed by a different payload.** The reclaim
  `UPDATE` did not compare `request_hash`, so a caller whose request differed
  from the original could inherit the key and run — precisely what an
  idempotency key exists to stop. Reclaim now requires a matching fingerprint;
  a diverging caller loses the race and meets the recorded hash ([#40], [#59]).
- **Postgres responses failed to decode depending on which store created the
  table.** `PostgresStore` declared `response` as `JSONB` while `DjangoStore`
  declared it `TEXT`, so whichever ran first won and the other's reads broke.
  The DDL is now aligned, and both stores read `response::text` so the value is
  decoded exactly once regardless of column type ([#57]).

### Added

- **`DjangoStore`** — reuses Django's existing database connection, for projects
  that already run Django. Reads `store.in_ambient_transaction()` so you can tell
  whether the store shares a transaction with your business writes.
- **Runnable FastAPI payment example** under `examples/`, exercising the retry,
  crash, divergence and concurrency paths end to end ([#38]).
- **Store selection guidance** in the README: which store to use, which not to,
  and why `SqliteStore()` with no path — the default — is not durable ([#58]).

### Changed

- `Store.complete()` and `Store.fail()` take `retention_seconds`, so retention is
  applied where the record becomes terminal rather than inferred later.
- Conformance suite additions covering retention that outlives a short claim
  lease, reclaim under a diverging fingerprint, and terminal records with no
  retention. **Two existing conformance tests were asserting the [#39] bug** —
  they swept a terminal record on its claim TTL and called it retention — and
  have been rewritten.

### Fixed (internal)

- Strict `mypy` now passes across the package.

## [0.1.0] — 2026-08-11

Initial release: the `@idempotent` decorator, `SqliteStore` and `PostgresStore`,
the claim → run → record protocol with atomic claims, divergence detection via
request fingerprint, `UNKNOWN` state and reconciliation, and the store
conformance suite.

[#38]: https://github.com/abhisheksharma2411/justonce/pull/38
[#39]: https://github.com/abhisheksharma2411/justonce/issues/39
[#40]: https://github.com/abhisheksharma2411/justonce/issues/40
[#57]: https://github.com/abhisheksharma2411/justonce/pull/57
[#58]: https://github.com/abhisheksharma2411/justonce/pull/58
[#59]: https://github.com/abhisheksharma2411/justonce/pull/59
[0.2.0]: https://github.com/abhisheksharma2411/justonce/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/abhisheksharma2411/justonce/releases/tag/v0.1.0
