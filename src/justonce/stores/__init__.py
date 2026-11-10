"""Storage backends.

`SqliteStore` is the default and needs no setup. `PostgresStore` is the
reference implementation for multi-host deployments. `SQLAlchemyStore` reaches
any SQL backend SQLAlchemy supports and an audited claim exists for, and is the
one to pick when the application already owns an `Engine`. `MemoryStore` is for
tests, and its docstring says why it is not for anything else.

Adding a backend is the most useful contribution to this project, and the
contract is small — see `justonce.stores.base.Store` and prove it with
`justonce.conformance.StoreConformanceTests`.
"""

from .base import Claim, Record, State, Store
from .memory import MemoryStore
from .sqlite import SqliteStore

__all__ = [
    "Claim",
    "MemoryStore",
    "Record",
    "SQLAlchemyStore",
    "SqliteStore",
    "State",
    "Store",
]


def __getattr__(name: str) -> object:  # pragma: no cover - import shim
    # Postgres needs psycopg, which is an optional extra. Import lazily so the
    # base package installs with no database driver at all.
    if name == "PostgresStore":
        from .postgres import PostgresStore

        return PostgresStore
    if name == "SQLAlchemyStore":
        from .sqlalchemy_store import SQLAlchemyStore

        return SQLAlchemyStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
