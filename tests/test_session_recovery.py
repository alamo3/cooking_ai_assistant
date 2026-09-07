"""Surviving a crash, a restart or a power cut.

Session state — the plan, the timers, what has been done — is the only thing that was never
in the database, and losing it mid-cook is the worst failure this system has. These tests
cover the snapshot round trip and the awkward parts of coming back: id counters that must not
restart, and timers that ran out while the power was off.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from cooking_assistant_ai.api.app import Settings, create_app
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.llm.client import ScriptedLLM
from cooking_assistant_ai.storage import sessions as snap
from cooking_assistant_ai.storage.db import Store


def _busy_session(ctx):
    """A cook part-way through: two recipes, a swap, a task under way and a timer running."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "load_recipe", {"recipe": "r002"})
    dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "butter",
                                 "replacement": "olive oil"})
    dispatch(ctx, "scale", {"recipe_id": "r002", "factor": 2.0})
    r = dispatch(ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                                   "step_ids": ["r001-s5"]})
    assert r.ok, r.reason
    dispatch(ctx, "set_timer", {"label": "rice simmering", "duration_s": 900,
                                "on_complete_hint": "take it off the heat"})
    dispatch(ctx, "mark_complete", {"step_id": "r001-s1"})
    dispatch(ctx, "remember", {"fact": "less salt than the recipe says"})
    return ctx.session


def test_a_cook_in_progress_round_trips(ctx):
    before = _busy_session(ctx)
    after = snap.from_dict(snap.to_dict(before))

    assert after.id == before.id and after.started_at == before.started_at
    assert list(after.recipes) == list(before.recipes)
    assert after.completed_steps == before.completed_steps
    assert after.notes == before.notes
    # the plan survives with its resolved times, so the cook plan renders the same
    assert list(after.tasks) == list(before.tasks)
    task, original = next(iter(after.tasks.values())), next(iter(before.tasks.values()))
    assert (task.start_at, task.end_at, task.status) == (original.start_at, original.end_at,
                                                         original.status)
    # and so do the overlays
    assert after.overlays["r002"].scale_factor == 2.0
    subs = after.overlays["r001"].substitutions
    assert subs and subs[0].replacement == "olive oil" and subs[0].original == "butter"
    timer = next(iter(after.timers.values()))
    assert timer.label == "rice simmering" and timer.on_complete_hint == "take it off the heat"


def test_new_ids_do_not_collide_after_a_restore(ctx):
    """A fresh counter would hand out t_001 again and overwrite the first task of the cook."""
    before = _busy_session(ctx)
    dispatch(ctx, "set_timer", {"label": "second", "duration_s": 60})
    existing_tasks, existing_timers = set(before.tasks), set(before.timers)

    after = snap.from_dict(snap.to_dict(before))
    assert after.new_id("t") not in existing_tasks
    assert after.new_id("tm") not in existing_timers


def test_timers_that_ran_out_while_the_power_was_off_do_not_all_fire(ctx):
    session = _busy_session(ctx)
    restored = snap.from_dict(snap.to_dict(session))
    later = ctx.clock.now() + timedelta(hours=2)

    late = snap.expire_timers(restored, later)
    assert [t.label for t in late] == ["rice simmering"]
    assert all(t.status != "running" for t in restored.timers.values())
    # a second pass finds nothing left to retire
    assert snap.expire_timers(restored, later) == []


def test_a_timer_still_counting_is_left_alone(ctx):
    session = _busy_session(ctx)
    restored = snap.from_dict(snap.to_dict(session))
    soon = ctx.clock.now() + timedelta(minutes=1)

    assert snap.expire_timers(restored, soon) == []
    assert all(t.status == "running" for t in restored.timers.values())


def test_an_empty_session_is_not_worth_saving(ctx):
    from cooking_assistant_ai.model.types import Session

    fresh = Session(id="empty", started_at=ctx.clock.now())   # opened the page, chose nothing
    assert not snap.is_worth_saving(fresh)
    fresh.add_recipe(ctx.store.get_recipe("r001"))
    assert snap.is_worth_saving(fresh)


def test_unreadable_snapshots_never_stop_the_server(tmp_path):
    store = Store(str(tmp_path / "c.db"))
    store.save_session("good", {"v": 1, "id": "good", "started_at": datetime.now().isoformat()})
    store.conn.execute("INSERT OR REPLACE INTO sessions (id, updated_at, body) VALUES (?,?,?)",
                       ("bad", datetime.now().isoformat(), "{not json"))
    store.conn.commit()
    assert [row[0] for row in store.load_sessions()] == ["good"]


def test_stale_snapshots_are_pruned(tmp_path):
    store = Store(str(tmp_path / "c.db"))
    store.save_session("fresh", {"v": 1})
    store.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = 'fresh'",
                       ((datetime.now() - timedelta(days=3)).isoformat(),))
    store.conn.commit()
    assert store.load_sessions(max_age_s=3600) == []
    assert store.prune_sessions(3600) == 1


# --------------------------------------------------------------------------- through the API

@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cooking.db")


def _app(db):
    return create_app(Settings(no_llm=True, idle_interval_s=0, warm_model=False, db_path=db,
                               autosave_s=0),  # drive persistence explicitly, no timing races
                      llm=ScriptedLLM(), store=Store(db))


def test_a_cook_survives_a_restart(db):
    with TestClient(_app(db)) as c:
        sid = c.post("/session", json={"recipe_ids": ["r001"]}).json()["session_id"]
        live = c.app.state.manager.get(sid)
        dispatch(live.orchestrator.ctx, "set_timer", {"label": "roast", "duration_s": 3600})
        c.app.state.manager.persist()

    # a brand new process, as the supervisor would start after a crash
    with TestClient(_app(db)) as c2:
        assert sid in c2.app.state.restored_sessions
        state = c2.get(f"/session/{sid}/state").json()
        assert [r["id"] for r in state["progress"]["recipes"]] == ["r001"]
        assert [t["label"] for t in state["timers"]] == ["roast"]


def test_a_session_the_cook_ended_is_not_resurrected(db):
    with TestClient(_app(db)) as c:
        sid = c.post("/session", json={"recipe_ids": ["r001"]}).json()["session_id"]
        c.app.state.manager.persist()
        assert c.delete(f"/session/{sid}").status_code == 204

    with TestClient(_app(db)) as c2:
        assert c2.app.state.restored_sessions == []


def test_shutdown_snapshots_so_a_restart_loses_nothing(db):
    with TestClient(_app(db)) as c:
        sid = c.post("/session", json={"recipe_ids": ["r002"]}).json()["session_id"]
        # no explicit persist: the lifespan must do it on the way out

    with TestClient(_app(db)) as c2:
        assert c2.app.state.restored_sessions == [sid]
