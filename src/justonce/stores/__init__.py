"""Storage backends.

`SqliteStore` is the default and needs no setup. `PostgresStore` is the
reference implementation for multi-host deployments. `MemoryStore` is for
tests, and its docstring says why it is not for anything else.

Adding a backend is the most useful contribution to this project, and the
contract is small — see `justonce.stores.base.Store` and prove it with
`justonce.conformance.StoreConformanceTests`.
"""

from .base import Claim, Record, State, Store
from .memory import MemoryStore
from .sqlite import SqliteStore

__all__ = ["Claim", "MemoryStore", "Record", "SqliteStore", "State", "Store"]


def __getattr__(name: str) -> object:  # pragma: no cover - import shim
    # Postgres needs psycopg, which is an optional extra. Import lazily so the
    # base package installs with no database driver at all.
    if name == "PostgresStore":
        from .postgres import PostgresStore

        return PostgresStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
