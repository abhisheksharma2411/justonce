"""Multi-tenant key namespacing (#53).

`operation_key("charge", order_id)` produces a key that is global to the store.
That is fine while order ids are globally unique, and quietly wrong the moment
they are not: two merchants with their own sequential numbering collide, one
tenant's payment is deduplicated against another's, and the effect simply never
runs. No error, no log line — the second charge just does not happen.

A namespace prefixes every key the engine touches. The whole design rests on one
property, and losing it rebuilds the bug the feature exists to prevent:

    **the delimiter must not be forgeable.**

If `namespace="a:b"` with key `"c"` and `namespace="a"` with key `"b:c"` both
produce `"a:b:c"`, two tenants collide again — through the mechanism that was
supposed to keep them apart, which is a worse place to have the bug than where
it started. So the namespace is validated to contain no delimiter, and the split
at the *first* delimiter is therefore unambiguous. Keys may contain as many as
they like; `operation_key` already produces `charge:v1:order_123`.
"""

from __future__ import annotations

#: Chosen to match `operation_key`'s existing separator, so a namespaced key
#: reads as one key rather than two conventions stacked on each other.
DELIMITER = ":"


def check_namespace(namespace: str | None) -> None:
    """Refuse a namespace that cannot be separated back out of a key.

    `None` means "not namespaced" and is the default. An empty or blank string
    is refused rather than treated as `None`: it looks like a tenant at the call
    site, and prefixing with it would shift every key by one delimiter instead
    of leaving them alone — a silent, total change of keyspace.
    """
    if namespace is None:
        return
    if not namespace.strip():
        raise ValueError(
            "namespace must be a non-empty string, or None for no namespace; "
            f"got {namespace!r}"
        )
    if DELIMITER in namespace:
        raise ValueError(
            f"namespace must not contain {DELIMITER!r}, got {namespace!r}: a namespace "
            "carrying the delimiter can be split two ways, so two tenants could "
            "produce one stored key — the exact collision namespacing prevents"
        )


def scoped(namespace: str | None, key: str) -> str:
    """The key as the store sees it."""
    return key if namespace is None else f"{namespace}{DELIMITER}{key}"


def unscoped(namespace: str | None, key: str) -> str:
    """The stored key as the caller sees it — the inverse of `scoped`.

    Records handed back to a caller carry their own keys, so leaving the prefix
    on would mean a key read from `unresolved()` could not be passed back to
    `run()` without the caller stripping something the engine added.
    """
    if namespace is None:
        return key
    prefix = f"{namespace}{DELIMITER}"
    return key[len(prefix) :] if key.startswith(prefix) else key


def belongs(namespace: str | None, key: str) -> bool:
    """Whether a stored key is inside this namespace.

    An unnamespaced engine matches everything on purpose: it is the operator's
    view of the store, not a tenant's, and a reconciliation worker that could
    not see every unresolved effect would be worse than no worker.
    """
    return namespace is None or key.startswith(f"{namespace}{DELIMITER}")
