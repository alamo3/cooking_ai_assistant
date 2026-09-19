"""Moving the cook on is an action, not something inferred from what was said.

Word matching against step text could notice the model had advanced without recording it, but
only after the fact and only when the wording happened to overlap. advance_step makes it the
same kind of thing as setting a timer: the model does it, and the state changes because it
did.
"""
from __future__ import annotations


from cooking_assistant_ai.core.tools import dispatch


def call(ctx, **args):
    return dispatch(ctx, "advance_step", args)


def test_advancing_records_every_earlier_step(ctx):
    out = call(ctx, step_id="3", recipe_id="r001")
    assert out.ok
    assert ctx.session.completed_steps == {"r001-s1", "r001-s2"}


def test_the_step_it_returns_is_the_one_to_read_out(ctx):
    out = call(ctx, step_id="3", recipe_id="r001")
    assert "step 3 of 7" in out.message
    assert "Sear the thighs" in out.message


def test_it_carries_the_amounts_so_they_need_not_be_recalled(ctx):
    """"How much, and for what dish" was the thing the cook kept having to ask for."""
    out = call(ctx, step_id="3", recipe_id="r001")
    assert "needs" in out.message and "butter" in out.message


def test_amounts_follow_a_scaled_recipe(ctx):
    assert dispatch(ctx, "scale", {"recipe_id": "r001", "factor": 2}).ok
    out = call(ctx, step_id="3", recipe_id="r001")
    assert out.ok
    assert "4 tbsp butter" in out.message


def test_advancing_again_to_the_same_step_is_harmless(ctx):
    call(ctx, step_id="3", recipe_id="r001")
    before = set(ctx.session.completed_steps)
    out = call(ctx, step_id="3", recipe_id="r001")
    assert out.ok and ctx.session.completed_steps == before


def test_going_backwards_is_refused_with_a_way_forward(ctx):
    call(ctx, step_id="4", recipe_id="r001")
    out = call(ctx, step_id="2", recipe_id="r001")
    assert not out.ok
    assert "already recorded as done" in out.reason
    assert "mark_complete" in out.reason          # the rejection says what to do instead
    assert "r001-s2" in ctx.session.completed_steps  # and changed nothing


def test_a_skipped_step_is_refused_rather_than_silently_reopened(ctx):
    dispatch(ctx, "skip_step", {"step_id": "r001-s2", "reason": "no thyme"})
    out = call(ctx, step_id="2", recipe_id="r001")
    assert not out.ok and "skipped" in out.reason


def test_skipped_steps_are_not_swept_up_as_done(ctx):
    dispatch(ctx, "skip_step", {"step_id": "r001-s2", "reason": "no thyme"})
    assert call(ctx, step_id="4", recipe_id="r001").ok
    assert "r001-s2" not in ctx.session.completed_steps
    assert "r001-s1" in ctx.session.completed_steps


def test_a_step_id_works_without_naming_the_recipe(ctx):
    out = call(ctx, step_id="r001-s3")
    assert out.ok and ctx.session.completed_steps == {"r001-s1", "r001-s2"}


def test_an_unknown_step_is_refused(ctx):
    out = call(ctx, step_id="99", recipe_id="r001")
    assert not out.ok


def test_advancing_past_a_task_completes_it(ctx, prepped):
    """The searing task is finished by walking off the end of its steps, not only by
    mark_complete, or the timeline would still show it running."""
    add = dispatch(ctx, "add_task", {"recipe_id": "r001", "label": "sear thighs",
                                     "step_ids": ["r001-s3"], "duration_s": 300,
                                     "appliance": "stovetop"})
    assert add.ok
    task = next(t for t in ctx.session.tasks.values() if t.label == "sear thighs")
    dispatch(ctx, "start_task", {"task_id": task.id})
    out = call(ctx, step_id="5", recipe_id="r001")
    assert out.ok
    assert ctx.session.tasks[task.id].status == "complete"


def test_its_timers_stop_when_it_does(ctx, prepped):
    add = dispatch(ctx, "add_task", {"recipe_id": "r001", "label": "sear thighs",
                                     "step_ids": ["r001-s3"], "duration_s": 300,
                                     "appliance": "stovetop"})
    task = next(t for t in ctx.session.tasks.values() if t.label == "sear thighs")
    dispatch(ctx, "start_task", {"task_id": task.id})
    dispatch(ctx, "set_timer", {"label": "thighs searing", "duration_s": 300,
                                "task_id": task.id, "on_complete_hint": "flip them"})
    assert call(ctx, step_id="5", recipe_id="r001").ok
    assert not [t for t in ctx.session.timers.values() if t.status == "running"]


def test_a_bare_step_number_works_when_one_dish_is_on(ctx):
    """The obvious call when there is only one recipe; refusing it would be pedantry."""
    for rid in ("r002", "r003"):
        dispatch(ctx, "unload_recipe", {"recipe_id": rid})
    out = call(ctx, step_id=3)
    assert out.ok and ctx.session.completed_steps == {"r001-s1", "r001-s2"}


def test_a_bare_step_number_is_refused_when_it_is_genuinely_ambiguous(ctx):
    out = call(ctx, step_id=3)
    assert not out.ok
    assert "ambiguous" in out.reason and "r001" in out.reason


def test_an_integer_step_id_is_accepted(ctx):
    """Models send numbers as numbers; a type mismatch should not be a refusal."""
    assert call(ctx, step_id=3, recipe_id="r001").ok
