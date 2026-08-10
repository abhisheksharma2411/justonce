"""Storage backends.

`SqliteStore` is the default and needs no setup. `PostgresStore` is the
reference implementation for multi-host deployments.

Adding a backend is the most useful contribution to this project, and the
contract is small — see `justonce.stores.base.Store` and prove it with
`justonce.conformance.StoreConformanceTests`.
"""

from .base import Claim, Record, State, Store
from .sqlite import SqliteStore

__all__ = ["Claim", "Record", "SqliteStore", "State", "Store"]


def __getattr__(name: str):  # pragma: no cover - import shim
    # Postgres needs psycopg, which is an optional extra. Import lazily so the
    # base package installs with no database driver at all.
    if name == "PostgresStore":
        from .postgres import PostgresStore

        return PostgresStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
