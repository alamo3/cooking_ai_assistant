"""Fixes from the first real cook.

Each test here corresponds to something that went wrong with a person standing at a hob,
which makes them the most valuable tests in the suite.
"""
from __future__ import annotations

from datetime import datetime

from cooking_assistant_ai.core.context import SYSTEM_PROMPT
from cooking_assistant_ai.core.plan import build_plan, render_cook_plan
from cooking_assistant_ai.core.render import render_timeline
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.speech.stt import kitchen_prompt


def test_substituting_updates_the_timeline_not_just_the_steps(ctx):
    """The cook plan said olive oil while the timeline still said 'butter sear'."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    assert dispatch(ctx, "add_task", {"label": "butter sear", "recipe_id": "r001",
                                      "step_ids": ["r001-s3"]}).ok
    assert dispatch(ctx, "set_timer", {"label": "butter sear timer", "duration_s": 300,
                                       "on_complete_hint": "flip once the butter browns"}).ok
    assert dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "butter",
                                        "replacement": "olive oil"}).ok

    timeline = render_timeline(ctx.session, ctx.clock.now())
    assert "olive oil sear" in timeline and "butter" not in timeline

    timer = next(iter(ctx.session.timers.values()))
    assert "olive oil" in timer.label and "butter" not in timer.label
    assert "butter" not in (timer.on_complete_hint or "")


def test_a_pinned_substitution_leaves_labels_alone(ctx):
    """Only a whole-recipe swap is a rename; a one-step swap must not rewrite the plan."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "add_task", {"label": "butter sear", "recipe_id": "r001",
                               "step_ids": ["r001-s3"]})
    dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "butter",
                                 "replacement": "ghee", "at_step": "r001-s3"})
    assert next(iter(ctx.session.tasks.values())).label == "butter sear"


def test_the_plan_carries_amounts_so_the_cook_need_not_ask(ctx):
    """'Add the garlic' is useless with both hands full: how much, and for which dish?"""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    plan = render_cook_plan(ctx.session, ctx.clock.now())
    seasoning = next(l for l in plan.splitlines() if "season all over" in l)
    assert "uses:" in seasoning
    assert "1 1/2 tsp kosher salt" in seasoning     # the amount
    assert "Roast Chicken Thighs" in seasoning      # and which dish

    items = [i for i in build_plan(ctx.session, ctx.clock.now()).items if i.id == "r001-s2"]
    assert items and any("kosher salt" in x for x in items[0].ingredients)


def test_amounts_follow_scaling(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r002"})
    dispatch(ctx, "scale", {"recipe_id": "r002", "factor": 2.0})
    plan = render_cook_plan(ctx.session, ctx.clock.now())
    assert "3 cup jasmine rice" in plan, "the plan still quotes the unscaled amount"


def test_the_prompt_asks_for_one_instruction_at_a_time():
    """The old wording invited 'what to do now and what comes next', which is two."""
    assert "exactly one instruction" in SYSTEM_PROMPT
    assert "Never chain two steps" in SYSTEM_PROMPT
    assert "say the amount and which dish" in SYSTEM_PROMPT


def test_the_speech_vocabulary_follows_what_is_being_cooked(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    hint = kitchen_prompt(ctx.session)
    assert "Roast Chicken Thighs" in hint and "chicken thighs" in hint
    assert "timer" in hint and "preheat" in hint          # the generic kitchen words too
    assert len(hint) <= 900                              # whisper ignores an over-long prompt
    assert "smashed" not in hint                          # qualifiers add noise, not signal


def test_the_vocabulary_hint_works_without_a_session():
    assert "rice cooker" in kitchen_prompt()
