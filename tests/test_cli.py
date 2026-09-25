"""Tests for the `justonce` console script (#45)."""

from __future__ import annotations

import io
import sys
import types

import pytest

from justonce.cli import EngineLoadError, load_engine, main


class FakeEngine:
    """Minimal stand-in: the CLI only needs sweep() and oldest_unresolved_age()."""

    def __init__(self, removed: int = 0, age: float | None = None) -> None:
        self._removed = removed
        self._age = age
        self.swept = 0

    def sweep(self) -> int:
        self.swept += 1
        return self._removed

    def oldest_unresolved_age(self) -> float | None:
        return self._age


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_sweep_reports_the_count_and_the_unresolved_age(monkeypatch):
    engine = FakeEngine(removed=7, age=125.4)
    _install(monkeypatch, "fake_app", engine=engine)
    out = io.StringIO()

    assert main(["sweep", "--engine", "fake_app:engine"], out=out) == 0

    assert engine.swept == 1
    assert "swept 7 record(s)" in out.getvalue()
    # Reported beside the count on purpose: a healthy-looking removal count can
    # sit in front of a backlog sweep is forbidden to touch.
    assert "oldest unresolved: 125s" in out.getvalue()


def test_no_unresolved_backlog_reads_as_none_not_zero(monkeypatch):
    # "0s" would mean something was written this instant, which is the opposite
    # of an empty backlog.
    _install(monkeypatch, "fake_empty", engine=FakeEngine(removed=0, age=None))
    out = io.StringIO()

    assert main(["sweep", "--engine", "fake_empty:engine"], out=out) == 0
    assert "oldest unresolved: none" in out.getvalue()


@pytest.mark.parametrize(
    "target",
    ["no_colon", "too:many:colons", ":missing_module", "missing_attr:"],
)
def test_a_malformed_target_is_rejected_with_its_shape(target):
    with pytest.raises(EngineLoadError) as excinfo:
        load_engine(target)
    assert "package.module:attribute" in str(excinfo.value)


def test_an_unimportable_module_names_the_module(monkeypatch):
    with pytest.raises(EngineLoadError) as excinfo:
        load_engine("justonce_no_such_module:engine")
    assert "justonce_no_such_module" in str(excinfo.value)


def test_a_missing_attribute_names_the_attribute(monkeypatch):
    _install(monkeypatch, "fake_bare")
    with pytest.raises(EngineLoadError) as excinfo:
        load_engine("fake_bare:engine")
    assert "engine" in str(excinfo.value)


def test_pointing_at_something_that_is_not_an_engine_says_so(monkeypatch):
    _install(monkeypatch, "fake_wrong", engine="a string")
    with pytest.raises(EngineLoadError) as excinfo:
        load_engine("fake_wrong:engine")
    assert "point this at a justonce engine" in str(excinfo.value)


def test_pointing_at_a_store_is_rejected_rather_than_crashing(monkeypatch):
    # Found by running the real command: a Store has `sweep` too, so a guard
    # that checked only that accepted the store and then died on an unbound
    # `MemoryStore.sweep() missing 1 required positional argument` traceback
    # from inside cron. `oldest_unresolved_age` is what separates them.
    from justonce.stores.memory import MemoryStore

    _install(monkeypatch, "fake_store_mod", store=MemoryStore(), cls=MemoryStore)

    for attribute in ("store", "cls"):
        with pytest.raises(EngineLoadError) as excinfo:
            load_engine(f"fake_store_mod:{attribute}")
        assert "oldest_unresolved_age" in str(excinfo.value)
        assert "not at a store" in str(excinfo.value)


def test_attributes_that_are_not_callable_are_rejected(monkeypatch):
    # `callable()` rather than `hasattr()`: a config object carrying a `sweep`
    # value would otherwise pass the guard and fail at the call, which is the
    # same class of late failure the store case produced.
    class NotAnEngine:
        sweep = "nightly"
        oldest_unresolved_age = None

    _install(monkeypatch, "fake_not_callable", engine=NotAnEngine())
    with pytest.raises(EngineLoadError) as excinfo:
        load_engine("fake_not_callable:engine")
    assert "sweep" in str(excinfo.value)


def test_a_real_engine_is_accepted():
    # The other side of the guard: it must not reject the thing it exists for.
    import sys
    import types

    from justonce import Idempotent
    from justonce.stores.memory import MemoryStore

    module = types.ModuleType("fake_real_engine")
    module.engine = Idempotent(MemoryStore())
    sys.modules["fake_real_engine"] = module
    try:
        assert load_engine("fake_real_engine:engine") is module.engine
    finally:
        del sys.modules["fake_real_engine"]


def test_a_bad_target_exits_non_zero_without_sweeping(monkeypatch, capsys):
    engine = FakeEngine(removed=3)
    _install(monkeypatch, "fake_guard", engine=engine)
    out = io.StringIO()

    assert main(["sweep", "--engine", "fake_guard:nope"], out=out) == 2

    assert engine.swept == 0
    assert "justonce sweep:" in capsys.readouterr().err
