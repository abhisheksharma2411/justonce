"""`justonce` command line entry point.

Sweeping exists as `Engine.sweep()` and works, but nothing calls it, so in
practice the table grows until somebody notices (#45). Retention here is a
correctness parameter — it has to outlive the longest replay chain — so leaving
it to "the operator will remember" is not good enough.

A console script rather than a Django management command, for two reasons. The
Django store is deliberately app-less ("No app to install, no migration to run
through `INSTALLED_APPS`"), and a management command would require exactly the
app that design avoids. And a script serves every store, not only Django, which
is what the "cron one-liner for everyone else" in #45 actually needs.

The engine is named the way gunicorn names an app — `package.module:attribute` —
because the sweeper has to operate on the *same* store configuration as the
application, and re-deriving that from flags would let the two drift apart.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any

TARGET_HELP = "import path of the engine to sweep, as package.module:attribute"


class EngineLoadError(Exception):
    """The `module:attribute` target could not be resolved to an engine."""


def load_engine(target: str, requires: Sequence[str] = ("sweep", "oldest_unresolved_age")) -> Any:
    """Import `package.module:attribute` and return it.

    Failures are raised as one error type carrying the reason, because the
    three ways this goes wrong — bad syntax, unimportable module, missing
    attribute — are indistinguishable to the operator from the outside, and a
    bare ImportError traceback from cron tells them nothing actionable.
    """
    if target.count(":") != 1:
        raise EngineLoadError(
            f"expected package.module:attribute, got {target!r}"
        )
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise EngineLoadError(
            f"expected package.module:attribute, got {target!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineLoadError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        engine = getattr(module, attribute)
    except AttributeError as exc:
        raise EngineLoadError(
            f"{module_name!r} has no attribute {attribute!r}"
        ) from exc
    # Per command, not one global list. A Store has `sweep` too, so checking
    # only that accepts the likeliest mistake — naming the store instead of the
    # engine — and then dies on an unbound-method TypeError from inside cron.
    # But demanding every method any subcommand might want would reject a
    # legitimate minimal engine from a command that never calls them, so each
    # command asks for what it actually uses.
    missing = [name for name in requires if not callable(getattr(engine, name, None))]
    if missing:
        raise EngineLoadError(
            f"{target!r} is {type(engine).__name__}, which has no "
            f"{' or '.join(missing)}(); point this at a justonce engine, "
            "not at a store"
        )
    return engine


_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> float:
    """`90`, `30s`, `15m`, `1h`, `7d` as seconds.

    A bare number is seconds. Anything else is rejected rather than guessed:
    an operator who types `1hr` during an incident should be told, not silently
    given one second and an empty list that looks like good news.
    """
    raw = text.strip().lower()
    if not raw:
        raise ValueError("empty duration")
    unit = _DURATION_UNITS.get(raw[-1])
    if unit is None:
        number, unit = raw, 1
    else:
        number = raw[:-1]
    try:
        value = float(number)
    except ValueError:
        raise ValueError(
            f"expected a duration like 30s, 15m, 1h, 7d — got {text!r}"
        ) from None
    if value < 0:
        raise ValueError(f"duration cannot be negative: {text!r}")
    return value * unit


def _age(record: Any, now: float) -> float | None:
    written = record.updated_at or record.created_at
    return None if written is None else max(0.0, now - written)


def _record_row(record: Any, now: float) -> str:
    age = _age(record, now)
    age_text = "unknown" if age is None else f"{age:.0f}s"
    return f"{record.key}\t{record.state.value}\tattempts={record.attempts}\tage={age_text}"


def _unresolved(args: argparse.Namespace, out: Any) -> int:
    try:
        engine = load_engine(args.engine, requires=("unresolved", "now"))
    except EngineLoadError as exc:
        print(f"justonce unresolved: {exc}", file=sys.stderr)
        return 2

    older_than = None
    if args.older_than is not None:
        try:
            older_than = parse_duration(args.older_than)
        except ValueError as exc:
            print(f"justonce unresolved: {exc}", file=sys.stderr)
            return 2

    records = engine.unresolved(older_than=older_than, limit=args.limit)
    now = engine.now()
    for record in records:
        print(_record_row(record, now), file=out)
    # A count on stderr so piping stdout into something stays clean, and an
    # empty list is still visibly an answer rather than a command that did
    # nothing.
    print(f"{len(records)} unresolved record(s)", file=sys.stderr)
    return 0


def _inspect(args: argparse.Namespace, out: Any) -> int:
    try:
        engine = load_engine(args.engine, requires=("lookup", "now"))
    except EngineLoadError as exc:
        print(f"justonce inspect: {exc}", file=sys.stderr)
        return 2

    record = engine.lookup(args.key)
    if record is None:
        # Distinct exit code: "no such key" and "the key is fine" are different
        # answers during an incident, and a script should be able to tell them
        # apart without parsing text.
        print(f"no record for {args.key!r}", file=sys.stderr)
        return 1

    now = engine.now()
    age = _age(record, now)
    print(f"key:          {record.key}", file=out)
    print(f"state:        {record.state.value}", file=out)
    print(f"attempts:     {record.attempts}", file=out)
    print(f"request_hash: {record.request_hash}", file=out)
    print(f"age:          {'unknown' if age is None else f'{age:.0f}s'}", file=out)
    print(f"has_response: {record.response is not None}", file=out)
    return 0


def _sweep(args: argparse.Namespace, out: Any) -> int:
    try:
        engine = load_engine(args.engine)
    except EngineLoadError as exc:
        print(f"justonce sweep: {exc}", file=sys.stderr)
        return 2

    removed = engine.sweep()
    # Reported together on purpose. A removal count alone looks healthy while a
    # backlog of unresolved outcomes sits behind it untouched — sweep never
    # deletes IN_PROGRESS or UNKNOWN, so a growing age is invisible in the
    # count that an operator would otherwise be watching.
    age = engine.oldest_unresolved_age()
    age_text = "none" if age is None else f"{age:.0f}s"
    print(f"swept {removed} record(s); oldest unresolved: {age_text}", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="justonce",
        description="Operate on a justonce store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser(
        "sweep",
        help="delete terminal records past their retention window",
        description=(
            "Deletes terminal records whose retention window has passed. "
            "IN_PROGRESS and UNKNOWN records are never deleted: an unresolved "
            "outcome that gets swept is a duplicate charge nobody can trace."
        ),
    )
    sweep.add_argument("--engine", required=True, help=TARGET_HELP)
    sweep.set_defaults(func=_sweep)

    unresolved = sub.add_parser(
        "unresolved",
        help="list effects whose outcome was never observed",
        description=(
            "Lists records in UNKNOWN — effects that ran, or may have run, with "
            "no recorded outcome. This is reconciliation's work list. Read-only."
        ),
    )
    unresolved.add_argument("--engine", required=True, help=TARGET_HELP)
    unresolved.add_argument(
        "--older-than",
        default=None,
        help="only records older than this, e.g. 30s, 15m, 1h, 7d",
    )
    unresolved.add_argument("--limit", type=int, default=100, help="maximum rows (default 100)")
    unresolved.set_defaults(func=_unresolved)

    inspect = sub.add_parser(
        "inspect",
        help="show what the store knows about one key",
        description=(
            "Prints the stored record for a key. Read-only. Exits 1 when no "
            "record exists, so a script can tell 'nothing here' from 'fine'."
        ),
    )
    inspect.add_argument("--engine", required=True, help=TARGET_HELP)
    inspect.add_argument("key", help="the idempotency key, in the caller's namespace")
    inspect.set_defaults(func=_inspect)

    return parser


def main(argv: Sequence[str] | None = None, out: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args, out or sys.stdout))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
