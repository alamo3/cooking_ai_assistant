"""Restarting from the tablet.

The contract with start.ps1's supervisor loop is a single number: exit 42 means "start me
again". If that drifts, the restart button silently becomes a stop button, so it is pinned
here along with the guard that keeps a restart from throwing away a cook in progress.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from cooking_assistant_ai.api import admin
from cooking_assistant_ai.api.app import Settings, create_app
from cooking_assistant_ai.llm.client import ScriptedLLM
from cooking_assistant_ai.storage.db import Store


@pytest.fixture
def client():
    app = create_app(Settings(no_llm=True, idle_interval_s=0, warm_model=False),
                     llm=ScriptedLLM(), store=Store(":memory:"))
    with TestClient(app) as c:
        yield c


def wait_for_stop(calls, timeout=3.0):
    """The endpoint stops the server from a background task, so the 202 can reach the tablet
    before the socket closes. Give that task a moment to run."""
    deadline = time.time() + timeout
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    return calls


@pytest.fixture
def supervised():
    """A server that has a supervisor behind it, as start.ps1 provides."""
    calls = []
    admin.reset()
    admin.register_stopper(lambda: calls.append("stop"))
    yield calls
    admin._state.stop_server = None
    admin.reset()


def test_exit_code_is_the_supervisor_contract():
    assert admin.RESTART_EXIT_CODE == 42, "start.ps1 checks for 42; change both or neither"


def test_status_reports_what_the_tablet_needs(client):
    s = client.get("/admin/status").json()
    assert s["pid"] > 0 and s["uptime_s"] >= 0
    assert set(("code_changed", "changed_files", "can_restart", "active_sessions")) <= set(s)


def test_code_changes_are_noticed(client, tmp_path):
    admin.reset()
    assert client.get("/admin/status").json()["code_changed"] is False
    (admin._PACKAGE / "core" / "render.py").touch()
    s = client.get("/admin/status").json()
    assert s["code_changed"] and "core/render.py" in s["changed_files"]


def test_restart_needs_a_supervisor(client):
    admin._state.stop_server = None
    r = client.post("/admin/restart")
    assert r.status_code == 503 and "supervisor" in r.json()["detail"]


def test_restart_stops_the_server_and_flags_the_exit(client, supervised):
    r = client.post("/admin/restart")
    assert r.status_code == 202 and r.json()["restarting"]
    assert wait_for_stop(supervised) == ["stop"], "the server was never asked to shut down"
    assert admin.restart_requested(), "the exit code would be 0, so the supervisor would stop"


def test_a_cook_in_progress_is_not_thrown_away_silently(client, supervised):
    sid = client.post("/session", json={"recipe_ids": ["r001"]}).json()["session_id"]

    r = client.post("/admin/restart", json={})
    assert r.status_code == 409 and "loses their timers" in r.json()["detail"]
    time.sleep(0.4)  # long enough that a mistaken stop would have landed
    assert supervised == [], "the server was stopped despite the guard"
    assert not admin.restart_requested()

    forced = client.post("/admin/restart", json={"force": True})
    assert forced.status_code == 202 and forced.json()["lost_sessions"] == [sid]
    assert wait_for_stop(supervised) == ["stop"]


def test_an_empty_session_does_not_block_a_restart(client, supervised):
    client.post("/session")  # opened the page, chose nothing
    assert client.post("/admin/restart", json={}).status_code == 202
