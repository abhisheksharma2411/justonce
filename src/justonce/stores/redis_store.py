"""Redis store (#3).

Requires the ``redis`` extra::

    pip install justonce[redis]

    from justonce.stores import RedisStore
    justonce.configure(RedisStore("redis://localhost:6379/0"))

**Every mutation is one Lua script.** Redis executes a script atomically, and
that is the only property this store needs from it: `claim` has to read the
record, decide, and write in one indivisible step, exactly as the SQL stores
get from `INSERT ... ON CONFLICT`. `WATCH`/`MULTI` would also work and would be
slower and racier to reason about, since the retry loop is then ours.

Three things here are not the obvious implementation, and each is a correctness
decision rather than a style one.

**The record carries no Redis TTL.** Expiring the key in Redis is the first
thing anyone reaches for, and it is wrong twice over. A claim whose lease has
passed must stay readable: the contract says a reclaim requires a *matching*
`request_hash`, so the record has to survive its own lease in order to refuse
the wrong one — and a caller that loses must find the original hash on the
record, which it cannot do if Redis has already deleted it. It is wrong again
for terminal records, which outlive the claim lease by their retention, and
which `sweep` is specified to delete only once that longer window passes.
So `expires_at` is a field, and deletion is `sweep`'s job alone.

**The clock is Redis's, never the caller's.** `now()` is the `TIME` command and
the scripts take their timestamps from `redis.call('TIME')`, so a lease means
the same thing to every host talking to this server. Measured against caller
clocks instead, a host running half an hour fast declares a live lease expired
and runs the effect a second time. `TIME` inside a script needs Redis 5.0 or
newer, where scripts replicate by effect; before that it was refused outright.

**Two sorted sets stand in for the indexes Redis does not have.** `unresolved`
and `sweep` both have to enumerate records by state, and Redis can only find a
key by name. `idx:unknown` holds unresolved keys scored by `updated_at`, and
`idx:terminal` holds terminal keys scored by `expires_at`. Both are maintained
inside the same scripts that write the record, so an index entry cannot survive
the record it points at.

That second index is also how "a terminal record without retention is never
swept" is enforced: such a record is simply never added to it. The rule is
structural rather than a condition someone can forget to write.

Not implemented: `LedgerQueryStore`. `query` filters on a key prefix, a state
and a `created_at` window at once, and Redis offers no index that answers that
without walking the keyspace. A `SCAN` loop would do it, and would be slowest
during exactly the incident the query exists for, on a server the application
also needs. A caller that asks gets `UnsupportedByStore` rather than a slow
partial answer -- point the ledger query at one of the SQL stores.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import StoreError
from .base import Claim, Record, State, check_key_length, decode_response

try:  # pragma: no cover - import guard
    import redis as redis_py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RedisStore requires redis. Install with: pip install justonce[redis]"
    ) from exc


#: Sentinel for a field that is absent. `expires_at` and `response` are both
#: nullable, and Lua cannot put a nil in the middle of a returned array without
#: truncating everything after it. The empty string is unambiguous for both:
#: `expires_at` is always a number when set, and `response` always holds JSON
#: *text*, whose shortest possible value is the two characters `""`.
_ABSENT = ""

#: Reads the record into the flat array every script returns. Kept as one
#: string and pasted into each script so the field order cannot drift between
#: the script that writes a record and the one that reads it back.
_READ_RECORD = """
local function read_record(rec)
  local f = redis.call('HMGET', rec,
    'state', 'request_hash', 'response', 'attempts',
    'created_at', 'updated_at', 'expires_at')
  for i = 1, 7 do
    if f[i] == false then f[i] = '' end
  end
  return f
end
"""

#: `TIME` returns seconds and microseconds as separate strings.
_NOW = """
local function store_now()
  local t = redis.call('TIME')
  return tonumber(t[1]) + tonumber(t[2]) / 1000000
end
"""

#: KEYS: record, unknown index, terminal index. ARGV: key, request_hash, ttl.
#:
#: The reclaim branch mirrors the SQL stores exactly, including the part that
#: looks redundant: `stored_hash == hash` is the divergence guard. An expired
#: lease says the previous holder died, not that the key is free for a
#: different request, and a reclaim that overwrote the hash would let a second
#: payload inherit the key and execute under it.
_CLAIM = (
    _NOW
    + _READ_RECORD
    + """
local rec, unknown_idx, terminal_idx = KEYS[1], KEYS[2], KEYS[3]
local key, hash, ttl = ARGV[1], ARGV[2], tonumber(ARGV[3])
local now = store_now()
local won = 0

if redis.call('EXISTS', rec) == 0 then
  redis.call('HSET', rec,
    'state', 'in_progress', 'request_hash', hash, 'attempts', 1,
    'created_at', tostring(now), 'updated_at', tostring(now),
    'expires_at', tostring(now + ttl))
  won = 1
else
  local state = redis.call('HGET', rec, 'state')
  local stored_hash = redis.call('HGET', rec, 'request_hash')
  local expires = redis.call('HGET', rec, 'expires_at')
  if state == 'in_progress' and stored_hash == hash
     and expires ~= false and expires ~= '' and tonumber(expires) < now then
    local attempts = tonumber(redis.call('HGET', rec, 'attempts')) + 1
    redis.call('HSET', rec,
      'request_hash', hash, 'attempts', attempts,
      'updated_at', tostring(now), 'expires_at', tostring(now + ttl))
    redis.call('HDEL', rec, 'response')
    won = 1
  end
end

if won == 1 then
  redis.call('ZREM', unknown_idx, key)
  redis.call('ZREM', terminal_idx, key)
end

local out = read_record(rec)
table.insert(out, 1, tostring(won))
return out
"""
)

#: KEYS: record, unknown index, terminal index.
#: ARGV: key, state, response-or-sentinel, retention-or-sentinel.
#:
#: `expires_at` is rewritten rather than left alone. A terminal record still
#: carrying its claim lease would be swept one claim-TTL after it was written,
#: whatever retention says, and the next delivery would find nothing and run
#: the effect again.
_TERMINAL = (
    _NOW
    + """
local rec, unknown_idx, terminal_idx = KEYS[1], KEYS[2], KEYS[3]
local key, state, response, retention = ARGV[1], ARGV[2], ARGV[3], ARGV[4]

if redis.call('EXISTS', rec) == 0 then
  return 0
end

local now = store_now()
redis.call('HSET', rec, 'state', state, 'updated_at', tostring(now))

if response == '' then
  redis.call('HDEL', rec, 'response')
else
  redis.call('HSET', rec, 'response', response)
end

redis.call('ZREM', unknown_idx, key)
if retention == '' then
  -- No retention means never swept, enforced by absence from the index
  -- rather than by a condition inside sweep.
  redis.call('HDEL', rec, 'expires_at')
  redis.call('ZREM', terminal_idx, key)
else
  local expires = now + tonumber(retention)
  redis.call('HSET', rec, 'expires_at', tostring(expires))
  redis.call('ZADD', terminal_idx, expires, key)
end
return 1
"""
)

#: KEYS: record, unknown index, terminal index. ARGV: key.
_RELEASE = """
local rec, unknown_idx, terminal_idx = KEYS[1], KEYS[2], KEYS[3]
redis.call('DEL', rec)
redis.call('ZREM', unknown_idx, ARGV[1])
redis.call('ZREM', terminal_idx, ARGV[1])
return 1
"""

#: KEYS: record, unknown index, terminal index. ARGV: key.
_MARK_UNKNOWN = (
    _NOW
    + """
local rec, unknown_idx, terminal_idx = KEYS[1], KEYS[2], KEYS[3]
local key = ARGV[1]
if redis.call('EXISTS', rec) == 0 then
  return 0
end
local now = store_now()
redis.call('HSET', rec, 'state', 'unknown', 'updated_at', tostring(now))
redis.call('HDEL', rec, 'expires_at')
redis.call('ZADD', unknown_idx, now, key)
redis.call('ZREM', terminal_idx, key)
return 1
"""
)

#: KEYS: terminal index. ARGV: cutoff-or-sentinel, record key prefix.
#:
#: The whole sweep is one script so a record cannot be deleted while its index
#: entry survives, or the reverse. Only terminal records with a retention are
#: in this index, so there is no state check to get wrong here.
_SWEEP = (
    _NOW
    + """
local terminal_idx = KEYS[1]
local cutoff = ARGV[1]
local prefix = ARGV[2]
if cutoff == '' then
  cutoff = store_now()
else
  cutoff = tonumber(cutoff)
end

local doomed = redis.call('ZRANGEBYSCORE', terminal_idx, '-inf', '(' .. tostring(cutoff))
for _, key in ipairs(doomed) do
  redis.call('DEL', prefix .. key)
  redis.call('ZREM', terminal_idx, key)
end
return #doomed
"""
)

#: KEYS: unknown index. ARGV: cutoff-or-sentinel, limit, record key prefix.
_UNRESOLVED = (
    _NOW
    + _READ_RECORD
    + """
local unknown_idx = KEYS[1]
local cutoff = ARGV[1]
local limit = tonumber(ARGV[2])
local prefix = ARGV[3]
if cutoff == '' then
  cutoff = store_now()
else
  cutoff = tonumber(cutoff)
end

local keys = redis.call('ZRANGEBYSCORE', unknown_idx, '-inf', cutoff, 'LIMIT', 0, limit)
local out = {}
for _, key in ipairs(keys) do
  local rec = prefix .. key
  if redis.call('EXISTS', rec) == 1 then
    local fields = read_record(rec)
    table.insert(fields, 1, key)
    table.insert(out, fields)
  end
end
return out
"""
)


class RedisStore:
    """Redis-backed store.

    Args:
        url: a Redis URL, e.g. ``redis://localhost:6379/0``. Ignored when
            `client` is given.
        client: an existing `redis.Redis`. Pass this to share the application's
            connection pool rather than opening a second one.
        namespace: prefix for every key this store owns. Change it to run two
            independent justonce deployments against one Redis, or to keep the
            records out of the way of an application already using the server.
        decode_responses: not a parameter, and deliberately so -- see below.

    A client configured with ``decode_responses=True`` is rejected at
    construction. This store reads `response` as JSON *text* and decodes it
    exactly once; a client that has already decoded for us makes a recorded
    payload of ``"done"`` indistinguishable from the text ``'"done"'``, and
    only the second is parseable. That is the same trap the SQL stores avoid by
    selecting the column as text, and it is worth a loud error rather than a
    corrupted replay.
    """

    #: Times come from the Redis server, which is the one clock every host
    #: talking to it shares.
    clock = "store"

    #: Redis keys are bounded at 512 MB, which no justonce key approaches, and
    #: Redis stores them whole -- there is no width to truncate against.
    max_key_length: int | None = None

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Any | None = None,
        namespace: str = "justonce",
    ) -> None:
        if client is not None:
            self._redis = client
        elif url is not None:
            self._redis = redis_py.Redis.from_url(url)
        else:
            raise ValueError("RedisStore needs either a url or a client")
        self._check_client_returns_bytes()

        self._namespace = namespace
        self._record_prefix = f"{namespace}:rec:"
        self._unknown_index = f"{namespace}:idx:unknown"
        self._terminal_index = f"{namespace}:idx:terminal"

        register = self._redis.register_script
        self._claim_script = register(_CLAIM)
        self._terminal_script = register(_TERMINAL)
        self._release_script = register(_RELEASE)
        self._mark_unknown_script = register(_MARK_UNKNOWN)
        self._sweep_script = register(_SWEEP)
        self._unresolved_script = register(_UNRESOLVED)

    def _check_client_returns_bytes(self) -> None:
        """Refuse a client that decodes for us. See the class docstring."""
        options = getattr(self._redis, "connection_pool", None)
        kwargs = getattr(options, "connection_kwargs", {}) if options else {}
        if kwargs.get("decode_responses"):
            raise StoreError(
                "RedisStore needs a client with decode_responses=False: it reads "
                "the recorded response as JSON text and decodes it once itself, "
                "and a pre-decoded string cannot be told apart from JSON text "
                "that still needs parsing."
            )

    def _record_key(self, key: str) -> str:
        return self._record_prefix + key

    def _index_keys(self, key: str) -> list[str]:
        return [self._record_key(key), self._unknown_index, self._terminal_index]

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("utf-8")
        return str(value)

    def _record(self, key: str, fields: list[Any]) -> Record:
        """Build a `Record` from the seven-field array the scripts return."""
        state, request_hash, response, attempts, created, updated, expires = (
            self._text(f) for f in fields
        )
        return Record(
            key=key,
            state=State(state),
            request_hash=request_hash,
            response=decode_response(response) if response != _ABSENT else None,
            attempts=int(float(attempts)),
            created_at=float(created) if created != _ABSENT else None,
            updated_at=float(updated) if updated != _ABSENT else None,
            expires_at=float(expires) if expires != _ABSENT else None,
        )

    def _run(self, script: Any, keys: list[str], args: list[Any]) -> Any:
        try:
            return script(keys=keys, args=args, client=self._redis)
        except redis_py.RedisError as exc:
            raise StoreError(f"redis call failed: {exc}") from exc

    # -- contract -----------------------------------------------------------

    def now(self) -> float:
        """The Redis server's clock. See `justonce.stores.base.Store.now`."""
        try:
            seconds, microseconds = self._redis.time()
        except redis_py.RedisError as exc:
            raise StoreError(f"could not read the redis clock: {exc}") from exc
        return float(seconds) + float(microseconds) / 1_000_000

    def claim(self, key: str, request_hash: str, ttl_seconds: float) -> Claim:
        check_key_length(key, self.max_key_length, "redis")
        result = self._run(
            self._claim_script, self._index_keys(key), [key, request_hash, ttl_seconds]
        )
        won = self._text(result[0]) == "1"
        return Claim(won=won, record=self._record(key, result[1:]))

    def complete(self, key: str, response: Any, *, retention_seconds: float | None = None) -> None:
        self._terminal(key, State.SUCCEEDED, response, retention_seconds)

    def fail(self, key: str, *, terminal: bool, retention_seconds: float | None = None) -> None:
        if terminal:
            self._terminal(key, State.FAILED, None, retention_seconds)
            return
        self._run(self._release_script, self._index_keys(key), [key])

    def _terminal(
        self, key: str, state: State, response: Any, retention_seconds: float | None
    ) -> None:
        encoded = json.dumps(response) if response is not None else _ABSENT
        retention = _ABSENT if retention_seconds is None else str(retention_seconds)
        self._run(
            self._terminal_script,
            self._index_keys(key),
            [key, state.value, encoded, retention],
        )

    def mark_unknown(self, key: str) -> None:
        self._run(self._mark_unknown_script, self._index_keys(key), [key])

    def lookup(self, key: str) -> Record | None:
        try:
            fields = self._redis.hmget(
                self._record_key(key),
                ["state", "request_hash", "response", "attempts",
                 "created_at", "updated_at", "expires_at"],
            )
        except redis_py.RedisError as exc:
            raise StoreError(f"redis lookup failed for {key!r}: {exc}") from exc
        if fields[0] is None:
            return None
        return self._record(key, [_ABSENT if f is None else f for f in fields])

    def sweep(self, *, before: float | None = None) -> int:
        cutoff = _ABSENT if before is None else str(before)
        count = self._run(
            self._sweep_script, [self._terminal_index], [cutoff, self._record_prefix]
        )
        return int(count)

    def unresolved(self, *, older_than: float | None = None, limit: int = 100) -> list[Record]:
        cutoff = _ABSENT if older_than is None else str(older_than)
        rows = self._run(
            self._unresolved_script,
            [self._unknown_index],
            [cutoff, limit, self._record_prefix],
        )
        return [self._record(self._text(row[0]), row[1:]) for row in rows]
