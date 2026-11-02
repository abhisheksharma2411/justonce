"""Flask integration (#13).

Two properties carry this, and both are the kind a refactor breaks quietly:

  * **A replay must not reach the view.** The whole point is that the second
    POST does not charge the card again, so every test that claims a replay
    asserts on a side-effect counter, not only on the response body.
  * **A response the ledger must not keep must also release the claim.** A 500
    that got recorded would answer every retry with the same 500 forever, and
    the client would have no way to ask again.

The `Set-Cookie` case has its own test because it is the one where the obvious
implementation is wrong: filtering the headers *before* sending strips a cookie
from the first, non-replayed response, which breaks the endpoint for the very
caller the library is supposed to leave alone.
"""

from __future__ import annotations

from typing import Any

import pytest

flask = pytest.importorskip("flask")

from justonce import Idempotent  # noqa: E402
from justonce.integrations.flask import (  # noqa: E402
    REPLAY_HEADER,
    JustOnce,
    idempotent_view,
)
from justonce.keys import fingerprint, operation_key  # noqa: E402
from justonce.stores import MemoryStore  # noqa: E402


@pytest.fixture()
def app() -> Any:
    application = flask.Flask(__name__)
    application.config["TESTING"] = True
    store = MemoryStore()
    JustOnce(application, store=store)
    application.store = store  # type: ignore[attr-defined]
    application.ran = []  # type: ignore[attr-defined]

    @application.post("/charges")
    @idempotent_view
    def charge() -> Any:
        application.ran.append(flask.request.get_json(silent=True))  # type: ignore[attr-defined]
        return {"id": "ch_1"}, 201

    @application.post("/flaky")
    @idempotent_view
    def flaky() -> Any:
        application.ran.append("flaky")  # type: ignore[attr-defined]
        return {"error": "upstream"}, 500

    @application.post("/cookie")
    @idempotent_view
    def cookie() -> Any:
        response = flask.make_response({"ok": True}, 201)
        response.set_cookie("session", "abc123")
        return response

    @application.post("/stream")
    @idempotent_view
    def stream() -> Any:
        return flask.Response(iter([b"a", b"b"]), direct_passthrough=True)

    @application.get("/health")
    @idempotent_view
    def health() -> Any:
        return {"ok": True}

    return application


@pytest.fixture()
def client(app: Any) -> Any:
    return app.test_client()


class TestTheSecondRequestDoesNotRunTheView:
    def test_a_repeat_is_answered_from_the_ledger(self, app: Any, client: Any) -> None:
        headers = {"Idempotency-Key": "k1"}
        first = client.post("/charges", json={"amount": 100}, headers=headers)
        second = client.post("/charges", json={"amount": 100}, headers=headers)

        assert first.status_code == second.status_code == 201
        assert first.get_json() == second.get_json() == {"id": "ch_1"}
        # The assertion that matters: the card was charged once.
        assert len(app.ran) == 1
        assert REPLAY_HEADER not in first.headers
        assert second.headers[REPLAY_HEADER] == "true"

    def test_a_different_key_is_a_different_operation(self, app: Any, client: Any) -> None:
        client.post("/charges", json={"amount": 100}, headers={"Idempotency-Key": "k1"})
        client.post("/charges", json={"amount": 100}, headers={"Idempotency-Key": "k2"})
        assert len(app.ran) == 2

    def test_the_same_key_on_another_route_is_not_a_replay(self, app: Any) -> None:
        # The default key is scoped by method and path. Without that, a client
        # that reuses one key per request id across two endpoints would get the
        # first endpoint's response back from the second.
        store = MemoryStore()
        application = flask.Flask(__name__)
        application.config["TESTING"] = True
        JustOnce(application, store=store)
        seen = []

        @application.post("/a")
        @idempotent_view
        def a() -> Any:
            seen.append("a")
            return {"route": "a"}, 200

        @application.post("/b")
        @idempotent_view
        def b() -> Any:
            seen.append("b")
            return {"route": "b"}, 200

        client = application.test_client()
        first = client.post("/a", headers={"Idempotency-Key": "same"})
        second = client.post("/b", headers={"Idempotency-Key": "same"})

        assert first.get_json() == {"route": "a"}
        assert second.get_json() == {"route": "b"}
        assert seen == ["a", "b"]


class TestWhatIsNotRecorded:
    def test_a_server_error_is_not_cached_so_the_retry_runs(self, app: Any, client: Any) -> None:
        headers = {"Idempotency-Key": "k1"}
        first = client.post("/flaky", headers=headers)
        second = client.post("/flaky", headers=headers)

        assert first.status_code == second.status_code == 500
        # Both ran. A cached 500 would answer every retry with the same failure
        # forever, and the claim would never be released.
        assert app.ran == ["flaky", "flaky"]
        assert REPLAY_HEADER not in second.headers

    def test_a_streamed_response_is_passed_through_not_recorded(
        self, app: Any, client: Any
    ) -> None:
        headers = {"Idempotency-Key": "k1"}
        first = client.post("/stream", headers=headers)
        second = client.post("/stream", headers=headers)

        assert first.get_data() == b"ab"
        # Reading the body of a passthrough response would drain the iterator
        # the client is about to be handed, so there is nothing to record and
        # the claim is released instead.
        assert second.get_data() == b"ab"
        assert REPLAY_HEADER not in second.headers

    def test_set_cookie_is_neither_stored_nor_replayed(self, app: Any, client: Any) -> None:
        headers = {"Idempotency-Key": "c1"}
        first = client.post("/cookie", headers=headers)
        second = client.post("/cookie", headers=headers)

        # The live response keeps its cookie: filtering before sending would
        # break the endpoint for the caller that did nothing wrong.
        assert "session=abc123" in first.headers["Set-Cookie"]
        # The replay does not carry it, and neither does the ledger.
        assert "Set-Cookie" not in second.headers
        stored = app.store.lookup(operation_key("http", "POST", "/cookie", "c1"))
        assert "Set-Cookie" not in repr(stored.response)
        assert "abc123" not in repr(stored.response)


class TestTheHeaderItself:
    def test_a_missing_key_is_rejected_by_default(self, app: Any, client: Any) -> None:
        response = client.post("/charges", json={"amount": 100})
        assert response.status_code == 400
        assert app.ran == [], "the view must not run when the key is required"

    def test_the_error_names_the_header_as_configured(self) -> None:
        application = flask.Flask(__name__)
        application.config["TESTING"] = True
        JustOnce(application, store=MemoryStore(), header="X-Request-Key")

        @application.post("/x")
        @idempotent_view
        def x() -> Any:
            return {"ok": True}

        response = application.test_client().post("/x")
        # The spelling the application chose. A client told to send
        # `X-Request-Key` should not be corrected about `x-request-key`.
        assert "X-Request-Key" in response.get_json()["detail"]

    def test_require_false_lets_an_unkeyed_request_through(self) -> None:
        application = flask.Flask(__name__)
        application.config["TESTING"] = True
        JustOnce(application, store=MemoryStore(), require=False)
        ran = []

        @application.post("/x")
        @idempotent_view
        def x() -> Any:
            ran.append(1)
            return {"ok": True}

        client = application.test_client()
        assert client.post("/x").status_code == 200
        assert client.post("/x").status_code == 200
        # Unkeyed means unguarded, not deduplicated: there is no key to
        # deduplicate on, and silently collapsing two of them would be worse.
        assert len(ran) == 2

    def test_a_safe_method_is_still_only_as_guarded_as_its_key(self, app: Any, client: Any) -> None:
        # The decorator is opt-in per view, so a GET it is applied to is
        # deduplicated like anything else -- that is the author's choice, and
        # it still needs a key.
        assert client.get("/health").status_code == 400
        assert client.get("/health", headers={"Idempotency-Key": "g1"}).status_code == 200


class TestKeyReuseAndContention:
    def test_the_same_key_with_a_different_body_is_rejected(self, app: Any, client: Any) -> None:
        headers = {"Idempotency-Key": "k1"}
        client.post("/charges", json={"amount": 100}, headers=headers)
        response = client.post("/charges", json={"amount": 999}, headers=headers)

        assert response.status_code == 422
        # Two distinct intents derived one key. Replaying the first as if it
        # were the second would silently drop a 999 charge.
        assert len(app.ran) == 1

    def test_a_key_held_in_flight_is_answered_with_409(self) -> None:
        application = flask.Flask(__name__)
        application.config["TESTING"] = True
        store = MemoryStore()
        JustOnce(application, store=store)

        @application.post("/slow")
        @idempotent_view
        def slow() -> Any:
            return {"ok": True}

        # Claim the key out of band, the way a concurrent request on another
        # worker would hold it, and never complete it.
        key = operation_key("http", "POST", "/slow", "k1")
        claim = store.claim(key, request_hash=fingerprint(""), ttl_seconds=60)
        assert claim.won, "the fixture must hold the claim for this to mean anything"

        response = application.test_client().post("/slow", headers={"Idempotency-Key": "k1"})
        assert response.status_code == 409


class TestWiring:
    def test_the_extension_is_required(self) -> None:
        application = flask.Flask(__name__)
        application.config["TESTING"] = True

        @application.post("/x")
        @idempotent_view
        def x() -> Any:
            return {"ok": True}

        with pytest.raises(RuntimeError, match="JustOnce extension"):
            application.test_client().post("/x", headers={"Idempotency-Key": "k"})

    def test_store_and_engine_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            JustOnce(store=MemoryStore(), engine=Idempotent(MemoryStore()))
        with pytest.raises(ValueError, match="exactly one"):
            JustOnce()

    def test_the_factory_pattern_works(self) -> None:
        extension = JustOnce(store=MemoryStore())
        application = flask.Flask(__name__)
        application.config["TESTING"] = True
        extension.init_app(application)

        @application.post("/x")
        @idempotent_view
        def x() -> Any:
            return {"ok": True}

        client = application.test_client()
        first = client.post("/x", headers={"Idempotency-Key": "k"})
        second = client.post("/x", headers={"Idempotency-Key": "k"})
        assert first.status_code == 200
        assert second.headers[REPLAY_HEADER] == "true"

    def test_the_view_keeps_its_name_so_url_for_still_works(self, app: Any) -> None:
        # functools.wraps, not cosmetics: Flask registers an endpoint under the
        # function's __name__, so losing it collapses every decorated view into
        # one endpoint called "wrapper" and the second registration raises.
        with app.test_request_context():
            assert flask.url_for("charge") == "/charges"
