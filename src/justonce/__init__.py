"""justonce — make side effects happen exactly once.

    from justonce import configure, idempotent, operation_key
    from justonce.stores import SqliteStore

    configure(SqliteStore("effects.db"))

    @idempotent(key=lambda order: operation_key("charge", order.id))
    def charge_customer(order):
        return payments.charge(order.customer, order.total)

Exactly-once *delivery* does not exist. At-least-once delivery with idempotent
processing produces exactly-once *effects*. You cannot stop the duplicate
arriving — this library makes it harmless.
"""

from .core import (
    DEFAULT_RETENTION_SECONDS,
    DEFAULT_TTL_SECONDS,
    Idempotent,
    OnInFlight,
    Result,
)
from .decorators import configure, get_default, idempotent
from .errors import (
    InFlightTimeout,
    JustOnceError,
    KeyReuseError,
    OperationInFlightError,
    StoreError,
)
from .keys import fingerprint, operation_key
from .stores.base import Claim, Record, State, Store

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RETENTION_SECONDS",
    "DEFAULT_TTL_SECONDS",
    "Claim",
    "Idempotent",
    "InFlightTimeout",
    "JustOnceError",
    "KeyReuseError",
    "OnInFlight",
    "OperationInFlightError",
    "Record",
    "Result",
    "State",
    "Store",
    "StoreError",
    "__version__",
    "configure",
    "fingerprint",
    "get_default",
    "idempotent",
    "operation_key",
]
