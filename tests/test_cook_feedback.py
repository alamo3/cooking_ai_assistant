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

def test_the_board_shows_what_is_on_what(ctx, prepped):
    from cooking_assistant_ai.core.appliances import board, render_board

    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                               "step_ids": ["r001-s5"], "appliance": "oven",
                               "temp_f": 425, "duration_s": 1500})
    dispatch(ctx, "add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3"],
                               "appliance": "stovetop", "duration_s": 300, "before": "chicken roast"})
    dispatch(ctx, "add_task", {"label": "rice", "appliance": "rice cooker", "duration_s": 1800})
    prepped(ctx, "sear")
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
    """Nothing loaded and nothing planned: every appliance is genuinely free."""
    from cooking_assistant_ai.core.appliances import render_board
    from cooking_assistant_ai.model.types import Session

    empty = Session(id="empty", started_at=ctx.clock.now())
    assert render_board(empty, ctx.clock.now(), ctx.store) == "APPLIANCES: all free"


def test_an_owned_appliance_a_recipe_wants_still_reads_as_needed(ctx):
    """Owning a rice cooker must not cost you the 'needed for X' signal."""
    from cooking_assistant_ai.core.appliances import board

    rows = {r["slot"]: r for r in board(ctx.session, ctx.clock.now(), ctx.store)}
    assert rows["air_fryer"]["status"] == "needed"      # r003 wants it, nothing planned
    assert rows["air_fryer"]["owned"] is True


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
    """An owned appliance goes back to free rather than vanishing off the board."""
    from cooking_assistant_ai.core.appliances import board

    dispatch(ctx, "create_recipe", {
        "title": "Zap", "servings": 1, "ingredients": ["soup", "bread"],
        "steps": [{"text": "Warm the soup.", "duration_s": 120, "appliance": "microwave"},
                  {"text": "Serve.", "duration_s": 60}]})
    step = ctx.store.get_recipe("r004").steps[0]

    status = lambda: {r["slot"]: r["status"] for r in board(ctx.session, ctx.clock.now(), ctx.store)}
    assert status()["microwave"] == "needed"
    dispatch(ctx, "mark_complete", {"step_id": step.id})
    assert status()["microwave"] == "free"


def test_an_appliance_is_never_both_needed_and_free(ctx):
    from cooking_assistant_ai.core.appliances import board, render_board

    dispatch(ctx, "create_recipe", {
        "title": "Zap", "servings": 1, "ingredients": ["soup", "bread"],
        "steps": [{"text": "Warm the soup.", "duration_s": 120, "appliance": "microwave"},
                  {"text": "Serve.", "duration_s": 60}]})
    rows = board(ctx.session, ctx.clock.now(), ctx.store)
    assert len({r["slot"] for r in rows}) == len(rows), "an appliance is drawn twice"

    text = render_board(ctx.session, ctx.clock.now(), ctx.store)
    free_list = text.split("free:")[-1] if "free:" in text else ""
    for r in rows:
        if r["status"] == "needed":
            assert r["label"] not in free_list, f"{r['label']} is listed as needed and free"


# ------------------------------------------------------------------- my kitchen

def test_the_board_only_draws_what_the_cook_owns(ctx):
    from cooking_assistant_ai.core.appliances import board

    ctx.store.set_appliances(["oven", "stovetop", "rice_cooker"], burners=2)
    slots = [r["slot"] for r in board(ctx.session, ctx.clock.now(), ctx.store)]
    assert slots[:4] == ["oven", "stovetop:1", "stovetop:2", "rice_cooker"]
    assert "stovetop:3" not in slots, "drew burners this hob does not have"
    assert not any(s == "microwave" for s in slots)


def test_a_task_on_an_appliance_you_do_not_have_is_refused(ctx):
    ctx.store.set_appliances(["oven", "stovetop"])
    r = dispatch(ctx, "add_task", {"label": "wings", "appliance": "air_fryer", "duration_s": 900})
    assert not r.ok
    assert "no air fryer" in r.reason and "Oven, Hob" in r.reason
    assert dispatch(ctx, "add_task", {"label": "roast", "appliance": "oven",
                                      "duration_s": 900}).ok


def test_the_model_is_told_what_the_kitchen_has(ctx):
    from cooking_assistant_ai.core.appliances import describe_kitchen

    ctx.store.set_appliances(["oven", "stovetop", "rice_cooker"], burners=2)
    text = describe_kitchen(ctx.store)
    assert "a hob (2 burners)" in text and "rice cooker" in text
    assert "air fryer" not in text
    assert "Do not plan a task on anything else" in text


def test_an_unowned_appliance_a_recipe_wants_is_still_shown(ctx):
    """Hiding the mismatch would leave the cook wondering why a step never appears."""
    from cooking_assistant_ai.core.appliances import board

    ctx.store.set_appliances(["oven", "stovetop"])
    rows = {r["slot"]: r for r in board(ctx.session, ctx.clock.now(), ctx.store)}
    assert rows["air_fryer"]["status"] == "needed"    # r003 wants one
    assert rows["air_fryer"]["owned"] is False


def test_unknown_appliances_are_rejected(ctx):
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown appliance"):
        ctx.store.set_appliances(["oven", "teleporter"])


def test_the_kitchen_survives_and_defaults_sanely(tmp_path):
    from cooking_assistant_ai.core.appliances import DEFAULT_OWNED, burner_count, owned
    from cooking_assistant_ai.storage.db import Store

    path = str(tmp_path / "k.db")
    fresh = Store(path)
    assert owned(fresh) == list(DEFAULT_OWNED) and burner_count(fresh) == 4
    fresh.set_appliances(["oven", "grill"], burners=6)
    assert owned(Store(path)) == ["oven", "grill"] and burner_count(Store(path)) == 6


def test_a_recipe_wanting_the_hob_flags_one_burner_not_all(ctx):
    """The hob is a pool: needing it means needing *a* ring."""
    from cooking_assistant_ai.core.appliances import board

    ctx.store.set_appliances(["oven", "stovetop"], burners=5)
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    rows = board(ctx.session, ctx.clock.now(), ctx.store)
    burners = [r for r in rows if r["slot"].startswith("stovetop:")]
    assert len(burners) == 5
    assert [r["status"] for r in burners].count("needed") == 1
    assert all(r["status"] == "free" for r in burners[1:])
    # and no stray bare "stovetop" slot alongside the numbered ones
    assert [r["slot"] for r in rows].count("stovetop") == 0
    assert len({r["slot"] for r in rows}) == len(rows)


# --------------------------------------------------------- the hob is not four free rings

def _hob(ctx, sizes="large,large,medium,small", cap="2", burners=4):
    from cooking_assistant_ai.core.appliances import CATALOGUE

    ctx.store.set_appliances(list(CATALOGUE), burners=burners)
    ctx.store.set_hob(sizes.split(","), int(cap))


def test_only_as_many_pans_as_physically_fit(ctx):
    """Four rings is not four pans: two large ones already fill most hobs."""
    _hob(ctx)
    assert dispatch(ctx, "add_task", {"label": "a", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "large"}).ok
    assert dispatch(ctx, "add_task", {"label": "b", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "small"}).ok
    third = dispatch(ctx, "add_task", {"label": "c", "appliance": "stovetop",
                                       "duration_s": 1200, "pan": "medium"})
    assert not third.ok
    assert "only 2 fit at once" in third.reason and "after=" in third.reason
    # chaining it is the way through, exactly as the rejection says
    assert dispatch(ctx, "add_task", {"label": "c", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "medium", "after": "a"}).ok


def test_a_big_pot_needs_a_big_ring(ctx):
    _hob(ctx, sizes="large,small,small,small", cap="4")
    assert dispatch(ctx, "add_task", {"label": "stockpot", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "large"}).ok
    second = dispatch(ctx, "add_task", {"label": "another stockpot", "appliance": "stovetop",
                                        "duration_s": 1200, "pan": "large"})
    assert not second.ok and "no free burner" in second.reason
    # a small pan still fits alongside it
    assert dispatch(ctx, "add_task", {"label": "sauce", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "small"}).ok
    assert ctx.session.find_task("stockpot").appliance == "stovetop:1"


def test_unknown_ring_sizes_never_invent_a_constraint(ctx):
    """A kitchen that has not configured its hob must not have large pans refused."""
    from cooking_assistant_ai.core.appliances import CATALOGUE

    ctx.store.set_appliances(list(CATALOGUE), burners=4)
    assert dispatch(ctx, "add_task", {"label": "stockpot", "appliance": "stovetop",
                                      "duration_s": 1200, "pan": "large"}).ok


def test_the_model_is_told_the_shape_of_the_hob(ctx):
    from cooking_assistant_ai.core.appliances import describe_kitchen

    _hob(ctx)
    text = describe_kitchen(ctx.store)
    assert "burner 1 large" in text and "burner 4 small" in text
    assert "only 2 pans fit on it at once" in text


def test_pan_size_survives_a_restart(ctx):
    from cooking_assistant_ai.storage import sessions as snap

    _hob(ctx)
    dispatch(ctx, "add_task", {"label": "a", "appliance": "stovetop",
                               "duration_s": 1200, "pan": "large"})
    restored = snap.from_dict(snap.to_dict(ctx.session))
    assert next(iter(restored.tasks.values())).pan == "large"


# ------------------------------------------------- do not send anyone to a pan unprepared

def test_a_task_will_not_start_with_prep_outstanding(ctx):
    """"It asked me to start dumping things into a pan I didn't have ready.\""""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3"],
                               "appliance": "stovetop", "duration_s": 300})
    r = dispatch(ctx, "start_task", {"task_id": "sear"})
    assert not r.ok
    assert "not ready to start" in r.reason and "season" in r.reason
    assert "mark_complete" in r.reason and "skip_step" in r.reason   # a way through, not a wall

    dispatch(ctx, "mark_complete", {"step_id": "r001-s2"})
    assert dispatch(ctx, "start_task", {"task_id": "sear"}).ok


def test_skipping_the_prep_also_unblocks_it(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3"],
                               "appliance": "stovetop", "duration_s": 300})
    dispatch(ctx, "skip_step", {"step_id": "r001-s2", "reason": "already seasoned"})
    assert dispatch(ctx, "start_task", {"task_id": "sear"}).ok


def test_the_plan_says_what_is_not_ready_before_it_is_due(ctx):
    """The gate stops the worst case; this is the looking-ahead the cook asked for."""
    from cooking_assistant_ai.core.plan import render_readiness

    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    dispatch(ctx, "load_recipe", {"recipe": "r002"})
    dispatch(ctx, "add_task", {"label": "sear chicken", "recipe_id": "r001",
                               "step_ids": ["r001-s3"], "appliance": "stovetop", "duration_s": 300})
    dispatch(ctx, "add_task", {"label": "simmer rice", "recipe_id": "r002",
                               "step_ids": ["r002-s3"], "appliance": "stovetop", "duration_s": 900})
    text = render_readiness(ctx.session)
    assert "sear chicken" in text and "season" in text
    assert "simmer rice" in text and "Rinse the rice" in text

    for sid in ("r001-s2", "r002-s1"):
        dispatch(ctx, "mark_complete", {"step_id": sid})
    assert render_readiness(ctx.session) == ""


def test_prep_detection_is_strict_for_gating_and_generous_for_grouping():
    """"Tuck the garlic, thyme and lemon halves around them" is not prep: 'halves' is a noun."""
    from cooking_assistant_ai.core.plan import is_prep_step, looks_like_prep
    from cooking_assistant_ai.model.types import Step

    tucking = Step(id="s", text="Flip the thighs. Tuck the garlic, thyme and lemon halves around them.")
    assert not is_prep_step(tucking), "would block a searing task from ever starting"
    assert looks_like_prep(tucking), "but still worth grouping the garlic with other prep"

    for text in ("Pat the chicken dry and season all over.", "Rinse the rice in a sieve.",
                 "Finely chop the shallots.", "Then mince the garlic."):
        assert is_prep_step(Step(id="s", text=text)), text
    for text in ("Preheat the oven to 425°F.", "Rest the chicken 10 minutes."):
        assert not is_prep_step(Step(id="s", text=text)), text


def test_a_stored_judgement_beats_the_heuristic(ctx):
    """Verb matching guesses; the model was asked once and the answer is on the step."""
    from cooking_assistant_ai.core.plan import is_prep_step, looks_like_prep
    from cooking_assistant_ai.model.types import Step

    # the regexes would both say prep: "Cut ..." opens with a prep verb
    assembling = Step(id="s", text="Cut the cooked potatoes in half and combine with the sauce.",
                      prep=False)
    assert not is_prep_step(assembling) and not looks_like_prep(assembling)

    # and both would miss this one: "Press" is not in the verb list at all
    pressing = Step(id="s", text="Press and cube the tofu, then toss with soy sauce.", prep=True)
    assert is_prep_step(pressing) and looks_like_prep(pressing)


def test_an_unjudged_step_still_falls_back(ctx):
    """Recipes stored before the field existed must keep working."""
    from cooking_assistant_ai.core.plan import is_prep_step
    from cooking_assistant_ai.model.types import Step

    assert Step(id="s", text="Rinse the rice.").prep is None
    assert is_prep_step(Step(id="s", text="Rinse the rice."))
    assert not is_prep_step(Step(id="s", text="Rest the chicken 10 minutes."))


def test_the_judgement_survives_storage_and_a_restart(ctx):
    from dataclasses import replace
    from cooking_assistant_ai.storage import sessions as snap

    recipe = ctx.store.get_recipe("r001")
    ctx.store.put_recipe(replace(recipe, steps=tuple(
        replace(s, prep=(n == 2)) for n, s in enumerate(recipe.steps, start=1))))
    assert [s.prep for s in ctx.store.get_recipe("r001").steps][:3] == [False, True, False]

    # the fixture already has r001 loaded, and a loaded recipe keeps the copy it was loaded
    # with, so reload it to pick the edit up
    dispatch(ctx, "unload_recipe", {"recipe_id": "r001"})
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    restored = snap.from_dict(snap.to_dict(ctx.session))
    assert [s.prep for s in restored.recipes["r001"].steps][:3] == [False, True, False]
