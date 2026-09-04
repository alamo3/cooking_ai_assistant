from __future__ import annotations

from datetime import timedelta

import pytest

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.tools import dispatch
from tests.conftest import T0


def ok(ctx, name, **args):
    r = dispatch(ctx, name, args)
    assert r.ok, r.reason
    return r


def rejected(ctx, name, **args):
    r = dispatch(ctx, name, args)
    assert not r.ok, f"expected rejection, got: {r.message}"
    return r


def test_asap_default_starts_now(ctx):
    ok(ctx, "add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3"])
    t = ctx.session.find_task("rice")
    assert t.start_at == T0
    assert t.end_at == T0 + timedelta(seconds=1200)
    assert t.appliance == "stovetop:2"  # inferred from the steps


def test_alap_to_plating(ctx):
    ok(ctx, "set_target_plating", time="7:30 PM")
    ok(ctx, "add_task", label="chicken roast", recipe_id="r001", step_ids=["r001-s5"], must_finish_by="plating")
    t = ctx.session.find_task("chicken roast")
    assert t.end_at == ctx.session.target_plating
    assert t.start_at == ctx.session.target_plating - timedelta(seconds=1500)


def test_after_dependency_and_shift_on_actual_start(ctx, clock):
    ok(ctx, "add_task", label="sear", recipe_id="r001", step_ids=["r001-s3"])
    ok(ctx, "add_task", label="roast", recipe_id="r001", step_ids=["r001-s5"], after="sear")
    sear, roast = ctx.session.find_task("sear"), ctx.session.find_task("roast")
    assert roast.start_at == sear.end_at
    # The sear runs long: start it 4 minutes late and the roast shifts.
    clock.advance(240)
    ok(ctx, "start_task", task_id="sear")
    assert roast.start_at == sear.end_at == T0 + timedelta(seconds=240 + 300)


def test_must_finish_by_task_is_alap_against_that_task(ctx):
    ok(ctx, "set_target_plating", time="7:30 PM")
    ok(ctx, "add_task", label="roast", recipe_id="r001", step_ids=["r001-s5"], must_finish_by="plating")
    ok(ctx, "add_task", label="sear", recipe_id="r001", step_ids=["r001-s3"], must_finish_by="roast")
    roast, sear = ctx.session.find_task("roast"), ctx.session.find_task("sear")
    assert sear.end_at == roast.start_at


def test_after_chain_feeding_alap_task_is_just_in_time(ctx):
    """prep -> sear -> roast -> rest(must_finish_by plating): the whole chain backs up to plating."""
    ok(ctx, "set_target_plating", time="7:30 PM")
    ok(ctx, "add_task", label="prep", recipe_id="r001", step_ids=["r001-s2"])
    ok(ctx, "add_task", label="sear", recipe_id="r001", step_ids=["r001-s3", "r001-s4"], after="prep")
    ok(ctx, "add_task", label="roast", recipe_id="r001", step_ids=["r001-s5"], after="sear")
    ok(ctx, "add_task", label="rest", recipe_id="r001", step_ids=["r001-s6"], after="roast", must_finish_by="plating")
    ok(ctx, "add_task", label="salad", duration_s=300)  # no deadline, no dependents: stays ASAP
    plating = ctx.session.target_plating
    rest, roast, sear, prep = (ctx.session.find_task(x) for x in ("rest", "roast", "sear", "prep"))
    assert rest.end_at == plating
    assert roast.end_at == rest.start_at == plating - timedelta(seconds=600)
    assert sear.end_at == roast.start_at
    assert prep.end_at == sear.start_at
    assert ctx.session.find_task("salad").start_at == T0
    # searing early pins the sear; the roast stays just-in-time for the rest
    ok(ctx, "start_task", task_id="sear")
    assert sear.end_at == T0 + timedelta(seconds=360)
    assert roast.start_at == rest.start_at - timedelta(seconds=1500)


def test_oven_shared_at_same_temperature_but_not_different(ctx):
    ok(ctx, "add_task", label="chicken", recipe_id="r001", step_ids=["r001-s5"])
    ok(ctx, "add_task", label="potatoes", appliance="oven", temp_f=425, duration_s=1200)
    r = rejected(ctx, "add_task", label="brussels", appliance="oven", temp_f=375, duration_s=720)
    assert "425°F" in r.reason and "375°F" in r.reason and "chicken" in r.reason
    assert ctx.session.find_task("brussels") is None  # rolled back


def test_burner_is_exclusive_and_after_fixes_it(ctx):
    ok(ctx, "add_task", label="rice", appliance="stovetop:2", duration_s=900)
    r = rejected(ctx, "add_task", label="sauce", appliance="stovetop:2", duration_s=600)
    assert "after='rice'" in r.reason
    ok(ctx, "add_task", label="sauce", appliance="stovetop:2", duration_s=600, after="rice")
    assert ctx.session.find_task("sauce").start_at == ctx.session.find_task("rice").end_at


def test_reject_when_cannot_finish_by_plating(ctx):
    ok(ctx, "set_target_plating", minutes_from_now=10)
    r = rejected(ctx, "add_task", label="roast", recipe_id="r001", step_ids=["r001-s5"], must_finish_by="plating")
    assert "after plating" in r.reason


def test_cycle_rejected(ctx):
    ok(ctx, "add_task", label="a", duration_s=60)
    ok(ctx, "add_task", label="b", duration_s=60, after="a")
    r = rejected(ctx, "move_task", task_id="a", after="b")
    assert "circular" in r.reason
    # state unchanged
    assert ctx.session.find_task("a").depends_on == []


def test_move_task_delay_and_remove_reschedules_dependents(ctx):
    ok(ctx, "add_task", label="a", duration_s=600)
    ok(ctx, "add_task", label="b", duration_s=600, after="a")
    ok(ctx, "move_task", task_id="a", delay_minutes=5)
    a, b = ctx.session.find_task("a"), ctx.session.find_task("b")
    assert a.start_at == T0 + timedelta(minutes=5)
    assert b.start_at == a.end_at
    ok(ctx, "remove_task", task_id="a")
    assert b.depends_on == [] and b.start_at == T0


def test_next_action_and_timeline_render(ctx):
    ok(ctx, "set_target_plating", time="7:30 PM")
    ok(ctx, "add_task", label="chicken roast", recipe_id="r001", step_ids=["r001-s5"], must_finish_by="plating")
    ok(ctx, "add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3", "r002-s4"], must_finish_by="plating")
    nxt = scheduler.next_action(ctx.session, ctx.now)
    assert nxt[1] == "start rice"
    plan = dispatch(ctx, "get_plan").state
    assert "Next action: 7:00 PM - start rice" in plan
    assert "Target plating: 7:30 PM" in plan
    assert "Conflicts: none" in plan


def test_replan_clears_only_pending(ctx):
    ok(ctx, "add_task", label="a", duration_s=600)
    ok(ctx, "add_task", label="b", duration_s=600)
    ok(ctx, "start_task", task_id="a")
    ok(ctx, "replan", reason="chicken ran long")
    assert set(ctx.session.tasks) == {"t_001"}
