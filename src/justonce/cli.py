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


def load_engine(target: str) -> Any:
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
    # Both, not just `sweep`. A Store has `sweep` too, so checking only that
    # accepts the likeliest mistake — naming the store instead of the engine —
    # and then dies on an unbound-method TypeError from inside cron. The pair
    # is what actually distinguishes the two.
    missing = [
        name
        for name in ("sweep", "oldest_unresolved_age")
        if not callable(getattr(engine, name, None))
    ]
    if missing:
        raise EngineLoadError(
            f"{target!r} is {type(engine).__name__}, which has no "
            f"{' or '.join(missing)}(); point this at a justonce engine, "
            "not at a store"
        )
    return engine


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
    return parser


def main(argv: Sequence[str] | None = None, out: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args, out or sys.stdout))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
