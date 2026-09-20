"""Moving the cook on is an action, and it can only move one step.

advance_step used to record every earlier step of the recipe as done, so one call to step 3
silently claimed steps 1 and 2 had happened. Over a real cook that marched through a jambalaya
nobody had made, and in the opening seconds it advanced three different recipes before
anything had been cooked at all. A step is done because the cook did it, not because the model
moved past it.
"""
from __future__ import annotations

from cooking_assistant_ai.core.tools import dispatch


def call(ctx, **args):
    return dispatch(ctx, "advance_step", args)


def only(ctx):
    """Leave one recipe loaded, so a bare step number is unambiguous."""
    for rid in ("r002", "r003"):
        dispatch(ctx, "unload_recipe", {"recipe_id": rid})
    return ctx


# ----------------------------------------------------------------- it cannot invent progress

def test_it_will_not_jump_over_open_steps(ctx):
    """The jambalaya failure: a jump to step 3 from a cold start recorded steps 1 and 2."""
    out = call(ctx, step_id="3", recipe_id="r001")
    assert not out.ok
    assert "still open" in out.reason
    assert ctx.session.completed_steps == set()


def test_the_refusal_names_what_is_in_the_way(ctx):
    out = call(ctx, step_id="4", recipe_id="r001")
    assert "1, 2, 3" in out.reason
    assert "mark_complete" in out.reason and "skip_step" in out.reason


def test_moving_one_step_records_exactly_one(ctx):
    assert call(ctx, step_id="1", recipe_id="r001").ok
    assert ctx.session.completed_steps == set()        # standing on it is not doing it
    assert call(ctx, step_id="2", recipe_id="r001").ok
    assert ctx.session.completed_steps == {"r001-s1"}
    assert call(ctx, step_id="3", recipe_id="r001").ok
    assert ctx.session.completed_steps == {"r001-s1", "r001-s2"}


def test_staying_on_the_current_step_records_nothing(ctx):
    call(ctx, step_id="2", recipe_id="r001")
    before = set(ctx.session.completed_steps)
    assert call(ctx, step_id="2", recipe_id="r001").ok
    assert ctx.session.completed_steps == before


def test_a_skipped_step_does_not_block_the_one_after_it(ctx):
    dispatch(ctx, "skip_step", {"step_id": "r001-s1", "reason": "oven already hot"})
    assert call(ctx, step_id="2", recipe_id="r001").ok
    assert "r001-s1" not in ctx.session.completed_steps


def test_skipping_clears_the_way_for_a_further_step(ctx):
    for sid in ("r001-s1", "r001-s2"):
        dispatch(ctx, "skip_step", {"step_id": sid, "reason": "done earlier"})
    assert call(ctx, step_id="3", recipe_id="r001").ok
    assert ctx.session.completed_steps == set()


def test_going_backwards_is_refused_with_a_way_forward(ctx):
    call(ctx, step_id="2", recipe_id="r001")
    out = call(ctx, step_id="1", recipe_id="r001")
    assert not out.ok
    assert "already recorded as done" in out.reason and "mark_complete" in out.reason


# ------------------------------------------------------------------ what it hands back

def test_the_step_it_returns_is_the_one_to_read_out(ctx):
    dispatch(ctx, "mark_complete", {"step_ids": ["r001-s1", "r001-s2"]})
    out = call(ctx, step_id="3", recipe_id="r001")
    assert "step 3 of 7" in out.message and "Sear the thighs" in out.message


def test_it_carries_the_amounts_so_they_need_not_be_recalled(ctx):
    """"How much, and for what dish" was the thing the cook kept having to ask for."""
    dispatch(ctx, "mark_complete", {"step_ids": ["r001-s1", "r001-s2"]})
    out = call(ctx, step_id="3", recipe_id="r001")
    assert "needs" in out.message and "butter" in out.message


def test_amounts_follow_a_scaled_recipe(ctx):
    assert dispatch(ctx, "scale", {"recipe_id": "r001", "factor": 2}).ok
    dispatch(ctx, "mark_complete", {"step_ids": ["r001-s1", "r001-s2"]})
    out = call(ctx, step_id="3", recipe_id="r001")
    assert out.ok and "4 tbsp butter" in out.message


# ------------------------------------------------------------------------- how it is called

def test_a_step_id_works_without_naming_the_recipe(ctx):
    assert call(ctx, step_id="r001-s1").ok


def test_an_integer_step_id_is_accepted(ctx):
    """Models send numbers as numbers; a type mismatch should not be a refusal."""
    assert call(ctx, step_id=1, recipe_id="r001").ok


def test_a_bare_step_number_works_when_one_dish_is_on(ctx):
    assert call(only(ctx), step_id=1).ok


def test_a_bare_step_number_is_refused_when_it_is_genuinely_ambiguous(ctx):
    out = call(ctx, step_id=1)
    assert not out.ok and "ambiguous" in out.reason


def test_an_unknown_step_is_refused(ctx):
    assert not call(ctx, step_id="99", recipe_id="r001").ok


# --------------------------------------------------------------------- tasks and timers

def test_finishing_a_tasks_last_step_completes_it(ctx, prepped):
    add = dispatch(ctx, "add_task", {"recipe_id": "r001", "label": "sear thighs",
                                     "step_ids": ["r001-s3"], "duration_s": 300,
                                     "appliance": "stovetop"})
    assert add.ok
    task = next(t for t in ctx.session.tasks.values() if t.label == "sear thighs")
    dispatch(ctx, "mark_complete", {"step_ids": ["r001-s1", "r001-s2"]})
    dispatch(ctx, "start_task", {"task_id": task.id})
    assert call(ctx, step_id="4", recipe_id="r001").ok
    assert ctx.session.tasks[task.id].status == "complete"
    assert not [t for t in ctx.session.timers.values() if t.status == "running"]
