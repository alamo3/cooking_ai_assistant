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


# --------------------------------------------------- appliances that finish when they finish

def test_a_rice_cooker_task_is_untimed(ctx):
    r = dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice cooker", "duration_s": 1800})
    assert r.ok, r.reason
    task = next(iter(ctx.session.tasks.values()))
    assert task.appliance == "rice_cooker" and task.awaits_cook


def test_a_deadline_on_an_untimed_appliance_is_refused(ctx):
    r = dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice_cooker",
                                   "duration_s": 1800, "must_finish_by": "plating"})
    assert not r.ok and "finishes when it finishes" in r.reason


def test_no_timer_for_something_that_cannot_be_timed(ctx):
    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice_cooker", "duration_s": 1800})
    r = dispatch(ctx, "set_timer", {"label": "rice", "duration_s": 1800, "task_id": "rice"})
    assert not r.ok
    assert "finishes when it finishes" in r.reason and "say when it is done" in r.reason


def test_running_past_the_estimate_is_not_an_overrun(ctx):
    from datetime import timedelta

    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice_cooker", "duration_s": 1800})
    dispatch(ctx, "add_task", {"label": "serve", "after": "rice", "duration_s": 120})
    dispatch(ctx, "start_task", {"task_id": "rice"})
    ctx.clock.set(ctx.clock.now() + timedelta(minutes=45))          # 15 past the estimate

    timeline = render_timeline(ctx.session, ctx.clock.now())
    assert "tell me when it's done" in timeline
    rice = ctx.session.find_task("rice")
    # the window is dragged along so what follows stays in the future, not the past
    assert rice.end_at >= ctx.clock.now()
    serve = ctx.session.find_task("serve")
    assert serve.start_at >= ctx.clock.now()

    assert dispatch(ctx, "mark_complete", {"task_id": "rice"}).ok
    assert ctx.session.find_task("rice").status == "complete"


def test_untimed_survives_a_restart(ctx):
    from cooking_assistant_ai.storage import sessions as snap

    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice_cooker", "duration_s": 1800})
    restored = snap.from_dict(snap.to_dict(ctx.session))
    assert next(iter(restored.tasks.values())).awaits_cook


def test_the_model_is_told_to_search_before_inventing():
    assert "find_recipes" in SYSTEM_PROMPT and "import_recipe" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("find_recipes") < SYSTEM_PROMPT.index("create_recipe")
    assert "rice cooker" in SYSTEM_PROMPT and "never set a timer for one" in SYSTEM_PROMPT


# ------------------------------------------------------------------- the appliance board

def test_the_board_shows_what_is_on_what(ctx):
    from cooking_assistant_ai.core.appliances import board, render_board

    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                               "step_ids": ["r001-s5"], "appliance": "oven",
                               "temp_f": 425, "duration_s": 1500})
    dispatch(ctx, "add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3"],
                               "appliance": "stovetop", "duration_s": 300, "before": "chicken roast"})
    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice cooker", "duration_s": 1800})
    dispatch(ctx, "start_task", {"task_id": "sear"})
    dispatch(ctx, "start_task", {"task_id": "rice"})

    rows = {r["slot"]: r for r in board(ctx.session, ctx.clock.now())}
    assert rows["stovetop:1"]["status"] == "active" and rows["stovetop:1"]["current"]["label"] == "sear"
    assert rows["oven"]["status"] == "reserved" and rows["oven"]["temp_f"] == 425
    assert rows["stovetop:4"]["status"] == "free" and rows["stovetop:4"]["current"] is None
    # an untimed appliance says so rather than counting down to a time it cannot know
    assert rows["rice_cooker"]["untimed"] and "tell me when" in rows["rice_cooker"]["detail"]

    text = render_board(ctx.session, ctx.clock.now())
    assert "Burner 1" in text and "Oven at 425" in text and "free: Burner 2" in text


def test_an_idle_kitchen_says_so(ctx):
    from cooking_assistant_ai.core.appliances import render_board

    assert render_board(ctx.session, ctx.clock.now()) == "APPLIANCES: all free"


def test_the_board_reaches_the_tablet(ctx):
    from cooking_assistant_ai.core.render import state_dict

    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice_cooker", "duration_s": 1800})
    rows = state_dict(ctx.session, ctx.clock.now())["appliances"]
    assert any(r["slot"] == "rice_cooker" for r in rows)
    assert any(r["slot"] == "oven" for r in rows), "standard appliances show even when idle"


def test_the_hostname_is_forgiving():
    from cooking_assistant_ai.api.mdns import hostname

    assert hostname("kitchen") == hostname("kitchen.local") == "kitchen.local"
    assert hostname("Chef.Local.") == "chef.local"


def test_the_board_shows_appliances_a_recipe_needs_before_anything_is_planned(ctx):
    """'What's expected to be in use' means the recipe's needs, not only the schedule."""
    from cooking_assistant_ai.core.appliances import board, render_board

    dispatch(ctx, "create_recipe", {
        "title": "Steamed Pudding", "servings": 2, "ingredients": ["flour", "butter"],
        "steps": [{"text": "Mix the batter.", "duration_s": 300},
                  {"text": "Steam in the rice cooker.", "duration_s": 2400, "appliance": "rice_cooker"},
                  {"text": "Blast the sauce.", "duration_s": 60, "appliance": "microwave"}]})

    rows = {r["slot"]: r for r in board(ctx.session, ctx.clock.now())}
    assert rows["rice_cooker"]["status"] == "needed"
    assert rows["rice_cooker"]["needed_by"] == ["Steamed Pudding"]
    assert rows["microwave"]["status"] == "needed"
    assert "needed for Steamed Pudding" in render_board(ctx.session, ctx.clock.now())

    # once a task claims it, it is scheduled rather than merely needed
    assert dispatch(ctx, "add_task", {"label": "steam", "recipe_id": "r004",
                                      "appliance": "rice_cooker", "duration_s": 2400}).ok
    rows = {r["slot"]: r for r in board(ctx.session, ctx.clock.now())}
    assert rows["rice_cooker"]["status"] in ("due", "reserved")
    assert rows["rice_cooker"]["needed_by"] == []


def test_a_finished_step_stops_asking_for_its_appliance(ctx):
    from cooking_assistant_ai.core.appliances import board

    dispatch(ctx, "create_recipe", {
        "title": "Zap", "servings": 1, "ingredients": ["soup", "bread"],
        "steps": [{"text": "Warm the soup.", "duration_s": 120, "appliance": "microwave"},
                  {"text": "Serve.", "duration_s": 60}]})
    step = ctx.store.get_recipe("r004").steps[0]
    assert any(r["slot"] == "microwave" for r in board(ctx.session, ctx.clock.now()))

    dispatch(ctx, "mark_complete", {"step_id": step.id})
    assert not any(r["slot"] == "microwave" for r in board(ctx.session, ctx.clock.now()))


def test_an_appliance_listed_as_needed_is_not_also_listed_as_free(ctx):
    from cooking_assistant_ai.core.appliances import render_board

    dispatch(ctx, "create_recipe", {
        "title": "Zap", "servings": 1, "ingredients": ["soup", "bread"],
        "steps": [{"text": "Warm the soup.", "duration_s": 120, "appliance": "microwave"},
                  {"text": "Serve.", "duration_s": 60}]})
    text = render_board(ctx.session, ctx.clock.now())
    assert text.count("Microwave") == 1, "listed twice"
    assert "Microwave" not in text.split("free:")[-1]
