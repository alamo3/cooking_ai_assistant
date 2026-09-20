"""Reconnecting with a session id that no longer exists must not look like losing the evening.

The tablet remembers its session id in localStorage. If that id is not in memory when it
reconnects - pruned for being older than COOK_SESSION_MAX_AGE, or the server restarted before
it was worth saving - the socket quietly created a new empty session. Every recipe the cook
had loaded appeared to vanish, with nothing said, while their work sat in memory under another
id and their library was untouched.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cooking_assistant_ai.api.app import create_app


@pytest.fixture
def client(store):
    """An explicit in-memory store. Settings reads COOK_DB when it is built, but passing the
    store leaves no doubt: an earlier version of this file relied on the environment and,
    because that default was evaluated at import, wrote a session into the real database."""
    from cooking_assistant_ai.api.app import Settings
    from cooking_assistant_ai.llm.client import ScriptedLLM

    app = create_app(settings=Settings(no_llm=True, warm_model=False, mdns=False,
                                       stt="none", tts="none"),
                     llm=ScriptedLLM(), store=store)
    with TestClient(app) as c:
        yield c


def open_session(client, session_id=None):
    url = "/ws/session" + (f"?session_id={session_id}" if session_id else "")
    return client.websocket_connect(url)


def test_a_fresh_tablet_just_gets_a_session(client):
    with open_session(client) as ws:
        assert ws.receive_json()["type"] == "session"
        assert ws.receive_json()["type"] == "state"


def test_reconnecting_with_a_live_id_keeps_that_session(client):
    with open_session(client) as ws:
        sid = ws.receive_json()["session_id"]
        ws.receive_json()
    with open_session(client, sid) as ws:
        assert ws.receive_json()["session_id"] == sid
        assert ws.receive_json()["type"] == "state"          # no notice: nothing went wrong


def test_an_unknown_id_does_not_silently_open_an_empty_kitchen(client):
    """The bug: a stale id got a blank session and the cook's recipes appeared to vanish."""
    with open_session(client) as ws:
        sid = ws.receive_json()["session_id"]
        ws.receive_json()
        client.post(f"/session/{sid}/recipes", json={"recipe": "r001"})

    with open_session(client, "a-session-that-expired") as ws:
        assert ws.receive_json()["session_id"] == sid        # their own cook, not a new one
        notice = ws.receive_json()
        assert notice["type"] == "notice"
        assert "expired" in notice["text"] and "picked up" in notice["text"]
        state = ws.receive_json()
        assert [r["id"] for r in state["progress"]["recipes"]] == ["r001"]


def test_when_there_is_genuinely_nothing_it_says_so(client):
    """No previous work anywhere: a fresh session is right, but the cook is told why."""
    with open_session(client, "a-session-that-expired") as ws:
        ws.receive_json()
        notice = ws.receive_json()
        assert notice["type"] == "notice"
        assert "fresh one" in notice["text"] and "library is untouched" in notice["text"]


def test_an_empty_session_is_never_offered_as_a_resume(client):
    """An untouched session is not "their evening"; resuming into it is just confusing."""
    with open_session(client) as ws:
        ws.receive_json()
        ws.receive_json()
    with open_session(client, "gone") as ws:
        ws.receive_json()
        assert "fresh one" in ws.receive_json()["text"]


def test_the_library_is_never_what_was_lost(client):
    """Whatever happens to a session, the recipes on disk are a separate thing."""
    before = client.get("/recipes").json()
    with open_session(client, "gone") as ws:
        ws.receive_json(); ws.receive_json(); ws.receive_json()
    assert client.get("/recipes").json() == before
    assert len(before) > 0
