from __future__ import annotations

from datetime import timedelta

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.plan import build_plan, ingredient_key, prep_groups, render_cook_plan
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.model.types import Recipe
from tests.conftest import T0

BEANS = {
    "id": "r009", "title": "Garlic Green Beans", "servings": 4,
    "ingredients": [
        {"id": "r009-i1", "name": "green beans", "amount": 400, "unit": "g"},
        {"id": "r009-i2", "name": "garlic, minced", "amount": 2, "unit": None},
        {"id": "r009-i3", "name": "kosher salt", "amount": 0.5, "unit": "tsp"},
    ],
    "steps": [
        {"id": "r009-s1", "text": "Trim the beans and mince the garlic.", "duration_s": 240, "ingredient_ids": ["r009-i1", "r009-i2"]},
        {"id": "r009-s2", "text": "Saute the beans with the garlic and salt.", "duration_s": 480, "appliance": "stovetop", "ingredient_ids": ["r009-i1", "r009-i2", "r009-i3"]},
    ],
}


def ok(ctx, name, **args):
    r = dispatch(ctx, name, args)
    assert r.ok, r.reason
    return r


def test_ingredient_keys_normalize_descriptors_plurals_and_aliases():
    assert ingredient_key("Garlic cloves, smashed") == ingredient_key("garlic, minced") == "garlic"
    assert ingredient_key("2 large yellow onions") == ingredient_key("Onions, thinly sliced") == "onion"
    assert ingredient_key("fresh thyme sprigs") == "thyme"
    assert ingredient_key("kosher salt") == "salt"


def test_auto_burner_assignment_and_shortage(ctx):
    ok(ctx, "add_task", label="rice", appliance="stovetop", duration_s=1200)
    ok(ctx, "add_task", label="sauce", appliance="stovetop", duration_s=600)
    ok(ctx, "add_task", label="beans", appliance="stovetop", duration_s=600, after="sauce")
    tasks = {t.label: t for t in ctx.session.tasks.values()}
    assert tasks["rice"].appliance == "stovetop:1" and tasks["sauce"].appliance == "stovetop:2"
    assert tasks["beans"].appliance == "stovetop:2"  # burner 2 is free again once the sauce is done
    # an active task keeps its burner even if a re-resolve would prefer another
    ok(ctx, "start_task", task_id="sauce")
    assert tasks["sauce"].appliance == "stovetop:2"
    # with only two burners, a fourth overlapping pot is a shortage with a specific reason
    old = scheduler.BURNERS
    scheduler.BURNERS = 2
    try:
        r = dispatch(ctx, "add_task", {"label": "soup", "appliance": "stovetop", "duration_s": 900})
        assert not r.ok and "no free burner for soup" in r.reason and "2 burners" in r.reason
        assert ctx.session.find_task("soup") is None
    finally:
        scheduler.BURNERS = old


def test_prep_groups_merge_shared_ingredients_across_recipes(ctx):
    ctx.session.add_recipe(Recipe.from_dict(BEANS))
    groups = prep_groups(ctx.session)
    assert [g.id for g in groups] == ["prep_garlic"]  # salt is a staple, rice is unique
    g = groups[0]
    assert g.summary() == "Garlic: 4 (Roast Chicken Thighs) + 2 (Garlic Green Beans) = 6 total"
    assert set(g.step_ids) == {"r001-s4", "r009-s1"} and g.status == "pending"
    ok(ctx, "scale", recipe_id="r009", factor=2)
    assert "+ 4 (Garlic Green Beans) = 8 total" in prep_groups(ctx.session)[0].summary()
    r = ok(ctx, "complete_prep", group="garlic")
    assert "2 step(s) done" in r.message
    assert {"r001-s4", "r009-s1"} <= ctx.session.completed_steps
    assert prep_groups(ctx.session)[0].status == "done"
    assert not dispatch(ctx, "complete_prep", {"group": "garlic"}).ok


def test_merged_plan_orders_steps_across_recipes_on_one_clock(ctx):
    ctx.session.add_recipe(Recipe.from_dict(BEANS))
    ok(ctx, "set_target_plating", time="7:30 PM")
    ok(ctx, "add_task", label="chicken roast", recipe_id="r001", step_ids=["r001-s5"], must_finish_by="plating")
    ok(ctx, "add_task", label="chicken sear", recipe_id="r001", step_ids=["r001-s3", "r001-s4"], appliance="stovetop", before="chicken roast")
    ok(ctx, "add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3", "r002-s4"], appliance="stovetop", must_finish_by="plating")
    ok(ctx, "add_task", label="beans", recipe_id="r009", step_ids=["r009-s2"], must_finish_by="plating")
    plan = build_plan(ctx.session, ctx.now)
    assert plan.now_id == "prep_garlic"  # mise en place comes first
    steps = [i for i in plan.items if i.kind == "step"]
    ids = [i.id for i in steps]
    assert "r001-s4" not in ids and "r009-s1" not in ids  # grouped prep steps appear once, as the group
    # chronological across recipes: sear (6:59) before rice boil (7:00) before roast (7:05) before beans (7:22)
    order = [ids.index(x) for x in ("r001-s3", "r002-s2", "r001-s5", "r009-s2")]
    assert order == sorted(order)
    # prep steps that no task covers are scheduled before their recipe's first task, not after
    assert ids.index("r001-s1") < ids.index("r001-s3") and ids.index("r002-s1") < ids.index("r002-s2")
    # steps after a task's steps (rest, serve) follow the task, in order
    assert ids.index("r001-s6") > ids.index("r001-s5") and ids.index("r001-s7") > ids.index("r001-s6")
    # an uncovered step between two covered ones is scheduled just before the later task,
    # never after it (seasoning belongs before the sear, not after the roast)
    ok(ctx, "add_task", label="preheat", recipe_id="r001", step_ids=["r001-s1"], appliance="oven", temp_f=425)
    ok(ctx, "move_task", task_id="preheat", delay_minutes=0)
    plan2 = build_plan(ctx.session, ctx.now)
    by = {i.id: i for i in plan2.items if i.kind == "step"}
    assert by["r001-s2"].at is not None and by["r001-s2"].at <= by["r001-s3"].at
    ids2 = [i.id for i in plan2.items if i.kind == "step"]
    assert ids2.index("r001-s2") < ids2.index("r001-s3")
    by_id = {i.id: i for i in steps}
    assert by_id["r001-s6"].at == by_id["r001-s5"].at + timedelta(seconds=1500)
    assert by_id["r002-s4"].appliance is None  # resting off the heat is not "on stovetop:2"
    assert by_id["r009-s2"].appliance == "stovetop:1"  # generic step took the task's assigned burner
    text = render_cook_plan(ctx.session, ctx.now)
    assert "NOW [prep_garlic]" in text and "Roast Chicken Thighs step 3" in text
    # completing the group and the first steps moves NOW along the merged sequence
    ok(ctx, "complete_prep", group="prep_garlic")
    ok(ctx, "mark_complete", step_ids=["r001-s1", "r001-s2"])
    plan = build_plan(ctx.session, ctx.now)
    assert plan.now_id == "r002-s1"
    assert plan.to_dict()["done_count"] == 3  # the garlic group plus two chicken steps


def test_plan_summary_merges_ingredients_and_reports_appliances(ctx):
    from cooking_assistant_ai.core.plan import merged_ingredients, plan_summary, render_plan_summary

    ing = {i["key"]: i for i in merged_ingredients(ctx.session)}
    assert ing["salt"]["text"] == "2 1/2 tsp kosher salt" and ing["salt"]["shared"]
    assert ing["chicken thigh"]["text"] == "8 chicken thighs" and not ing["chicken thigh"]["shared"]
    s = plan_summary(ctx.session, ctx.now)
    assert not s["planned"] and s["steps"] == 14
    ok(ctx, "set_target_plating", minutes_from_now=90)
    ok(ctx, "add_task", label="sear", recipe_id="r001", step_ids=["r001-s3"], appliance="stovetop")
    ok(ctx, "add_task", label="roast", recipe_id="r001", step_ids=["r001-s5"], after="sear", must_finish_by="plating")
    ok(ctx, "add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3"], appliance="stovetop", must_finish_by="plating")
    ok(ctx, "add_task", label="sprouts", recipe_id="r003", step_ids=["r003-s2"], must_finish_by="plating")
    s = plan_summary(ctx.session, ctx.now)
    assert s["planned"] and s["plating_at"].endswith("20:00:00") and 0 < s["span_s"] <= 90 * 60
    assert s["appliances"] == ["air fryer 375°F", "oven 425°F", "stovetop x1 (burner 1)"]
    text = render_plan_summary(ctx.session, ctx.now)
    assert "plating 8:00 PM" in text and "(shared)" in text and "14 steps in all" in text


def test_plan_without_tasks_lists_steps_as_unscheduled(ctx):
    plan = build_plan(ctx.session, ctx.now)
    assert plan.items and all(i.status in ("now", "unscheduled") for i in plan.items)
    assert plan.now_id == "r001-s1"
