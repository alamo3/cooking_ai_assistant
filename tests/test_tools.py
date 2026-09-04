from __future__ import annotations

import json

from cooking_assistant_ai.core.fmt import fmt_amount, fmt_dur, parse_clock_time, parse_duration
from cooking_assistant_ai.core.render import render_progress, render_recipe
from cooking_assistant_ai.core.tools import dispatch
from tests.conftest import T0


def test_envelope_shapes(ctx):
    good = dispatch(ctx, "get_plan").envelope()
    assert good["ok"] is True and "Now:" in good["state"]
    bad = dispatch(ctx, "remove_task", {"task_id": "nope"}).envelope()
    assert bad["ok"] is False and "no task 'nope'" in bad["reason"] and "state" in bad
    unknown = dispatch(ctx, "make_coffee").envelope()
    assert not unknown["ok"] and "unknown tool" in unknown["reason"]
    json.dumps(good), json.dumps(bad)  # serializable


def test_nullish_strings_in_optional_fields_are_treated_as_absent(ctx):
    """Smaller models send after="none" / appliance="none" / temp_f=0 instead of omitting them."""
    r = dispatch(ctx, "add_task", {"label": "roast", "recipe_id": "r001", "step_ids": ["r001-s5"],
                                   "after": "none", "before": "null", "must_finish_by": "",
                                   "appliance": "oven", "temp_f": 0})
    assert r.ok, r.reason
    t = ctx.session.find_task("roast")
    assert t.depends_on == [] and t.must_finish_by is None and t.temp_f is None
    r = dispatch(ctx, "add_task", {"label": "prep", "recipe_id": "r001", "step_ids": ["r001-s2"],
                                   "appliance": "none", "after": "N/A"})
    assert r.ok and ctx.session.find_task("prep").appliance is None
    # a genuinely unknown reference is still rejected, with the known task list
    bad = dispatch(ctx, "add_task", {"label": "x", "duration_s": 60, "after": "sear"})
    assert not bad.ok and "no task 'sear'" in bad.reason
    assert not dispatch(ctx, "move_task", {"task_id": "roast", "after": "none"}).ok  # nothing to do


def test_extra_args_are_ignored_and_strings_coerced(ctx):
    r = dispatch(ctx, "add_task", {"label": "x", "duration_s": "15 minutes", "appliance": "Stove:1", "bogus": 1})
    assert r.ok, r.reason
    t = ctx.session.find_task("x")
    assert t.duration_s == 900 and t.appliance == "stovetop:1"


def test_timer_label_validation_and_lifecycle(ctx):
    assert not dispatch(ctx, "set_timer", {"label": "timer 2", "duration_s": 60}).ok
    assert not dispatch(ctx, "set_timer", {"label": "rice", "duration_s": 0}).ok
    r = dispatch(ctx, "set_timer", {"label": "rice simmering", "duration_s": 900, "on_complete_hint": "take off heat"})
    assert r.ok and "rice simmering" in r.state and "take off heat" in r.state
    dup = dispatch(ctx, "set_timer", {"label": "Rice Simmering", "duration_s": 60})
    assert not dup.ok and "already running" in dup.reason
    c = dispatch(ctx, "cancel_timer", {"timer_id": "rice"})  # by label substring
    assert c.ok and "none running" in c.state
    assert not dispatch(ctx, "cancel_timer", {"timer_id": "tm_001"}).ok  # already cancelled


def test_mark_complete_cascades_to_task_and_cancels_timers(ctx):
    dispatch(ctx, "add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3", "r001-s4"]})
    dispatch(ctx, "start_task", {"task_id": "sear"})
    dispatch(ctx, "set_timer", {"label": "sear skin side", "duration_s": 300, "task_id": "sear"})
    r = dispatch(ctx, "mark_complete", {"step_ids": ["r001-s3", "r001-s4"]})
    assert r.ok
    assert ctx.session.find_task("sear").status == "complete"
    assert ctx.session.timers["tm_001"].status == "cancelled"
    assert "steps 3-4 done" in render_progress(ctx.session)
    assert not dispatch(ctx, "mark_complete", {"task_id": "sear"}).ok  # already complete


def test_step_number_shorthand_and_skip(ctx):
    r = dispatch(ctx, "skip_step", {"step_id": "r001-s6", "reason": "in a hurry"})
    assert r.ok and "[skipped]" in r.state and "in a hurry" in r.state
    n = dispatch(ctx, "add_note", {"recipe_id": "r001", "step_id": "4", "note": "pan smoking"})
    assert n.ok and "note: pan smoking" in n.state


def test_overlay_rendering(ctx):
    dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "butter", "replacement": "olive oil", "note": "1:1"})
    dispatch(ctx, "scale", {"recipe_id": "r001", "factor": 2})
    txt = render_recipe(ctx.session.recipes["r001"], ctx.session.overlays["r001"])
    assert "4 tbsp olive oil (instead of butter, 1:1)" in txt
    assert "serves 8" in txt and "16 chicken thighs" in txt
    prog = render_progress(ctx.session)
    assert "butter -> olive oil" in prog and "scaled x2" in prog


def test_remember_and_inventory(ctx):
    assert dispatch(ctx, "remember", {"note": "no cilantro"}).ok
    assert "no cilantro" in render_progress(ctx.session)
    r = dispatch(ctx, "check_stock", {"items": ["butter", "saffron"]})
    assert r.ok and "saffron: NOT in stock" in r.message and "250 g butter" in r.message
    d = dispatch(ctx, "deduct", {"items": [{"name": "butter", "amount": 50, "unit": "g"}]})
    assert d.ok and "200" in d.message
    bad = dispatch(ctx, "deduct", {"items": [{"name": "butter", "amount": 1, "unit": "cup"}]})
    assert not bad.ok and "tracked in g" in bad.reason


def test_load_and_unload_recipe(ctx):
    del ctx.session.recipes["r003"]
    r = dispatch(ctx, "load_recipe", {"recipe": "brussels"})
    assert r.ok and "r003" in ctx.session.recipes
    dispatch(ctx, "add_task", {"label": "sprouts", "recipe_id": "r003", "step_ids": ["r003-s2"]})
    dispatch(ctx, "start_task", {"task_id": "sprouts"})
    assert not dispatch(ctx, "unload_recipe", {"recipe_id": "r003"}).ok
    dispatch(ctx, "mark_complete", {"task_id": "sprouts"})
    assert dispatch(ctx, "unload_recipe", {"recipe_id": "r003"}).ok


def test_set_target_plating_forms(ctx):
    assert dispatch(ctx, "set_target_plating", {"time": "19:15"}).ok
    assert ctx.session.target_plating.hour == 19 and ctx.session.target_plating.minute == 15
    assert dispatch(ctx, "set_target_plating", {"minutes_from_now": 45}).ok
    assert ctx.session.target_plating == T0.replace(hour=19, minute=15)
    assert not dispatch(ctx, "set_target_plating", {"time": "sometime"}).ok


def test_fmt_helpers():
    assert fmt_amount(1.5) == "1 1/2" and fmt_amount(0.25) == "1/4" and fmt_amount(2) == "2"
    assert fmt_dur(1980) == "33m" and fmt_dur(3900) == "1h05m" and fmt_dur(45) == "45s"
    assert parse_duration("1h30m") == 5400 and parse_duration("15 minutes") == 900
    assert parse_clock_time("7:15pm", T0).hour == 19
