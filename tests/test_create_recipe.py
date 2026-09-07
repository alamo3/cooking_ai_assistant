from __future__ import annotations

from cooking_assistant_ai.core.mealplan import render_meal_options
from cooking_assistant_ai.core.tools import dispatch

GOOD = {
    "title": "Garlic Butter Rice Skillet", "servings": 4,
    "ingredients": [
        {"name": "jasmine rice", "amount": 1.5, "unit": "cup"},
        {"name": "butter", "amount": 2, "unit": "tbsp"},
        {"name": "garlic cloves", "amount": 3},
    ],
    "steps": [
        {"text": "Mince the garlic.", "duration_s": 120, "uses": ["garlic cloves"]},
        {"text": "Melt the butter and soften the garlic.", "duration_s": 180,
         "appliance": "stovetop", "uses": ["butter", "garlic cloves"]},
        {"text": "Add the rice, cover and simmer.", "duration_s": 900, "appliance": "stovetop"},
    ],
}


def test_created_recipe_is_saved_loaded_and_schedulable(ctx):
    r = dispatch(ctx, "create_recipe", GOOD)
    assert r.ok, r.reason
    rid = "r004"
    assert ctx.store.get_recipe(rid) is not None          # persisted to the library
    assert rid in ctx.session.recipes                      # and usable straight away
    recipe = ctx.store.get_recipe(rid)
    assert recipe.title == "Garlic Butter Rice Skillet" and recipe.servings == 4
    # steps carry what the scheduler needs
    assert recipe.steps[1].appliance == "stovetop" and recipe.steps[1].duration_s == 180
    assert recipe.steps[0].appliance is None
    # 'uses' resolved to real ingredient ids, so shared-prep grouping works
    garlic = next(i for i in recipe.ingredients if i.name == "garlic cloves")
    assert garlic.id in recipe.steps[0].ingredient_ids
    # and it can immediately be scheduled
    assert dispatch(ctx, "add_task", {"label": "skillet", "recipe_id": rid,
                                      "step_ids": [recipe.steps[1].id]}).ok


def test_ids_do_not_collide_with_existing_recipes(ctx):
    first = dispatch(ctx, "create_recipe", GOOD)
    second = dispatch(ctx, "create_recipe", dict(GOOD, title="Second Dish"))
    assert first.ok and second.ok
    ids = [r.id for r in ctx.store.list_recipes()]
    assert len(ids) == len(set(ids)) and "r004" in ids and "r005" in ids


def test_diet_is_enforced_at_creation(ctx):
    ctx.store.set_diet("vegan")
    bad = dispatch(ctx, "create_recipe", {
        "title": "Chicken Pilaf", "servings": 4,
        "ingredients": [{"name": "chicken thighs", "amount": 4}, {"name": "rice", "amount": 1, "unit": "cup"}],
        "steps": [{"text": "Brown the chicken.", "duration_s": 300, "appliance": "stovetop"},
                  {"text": "Add rice and simmer.", "duration_s": 900, "appliance": "stovetop"}]})
    assert not bad.ok
    assert "chicken is not vegan" in bad.reason and "invent something without chicken" in bad.reason
    assert len(ctx.store.list_recipes()) == 3  # nothing was written


def test_thin_recipes_are_rejected(ctx):
    one_step = dispatch(ctx, "create_recipe", dict(GOOD, steps=[{"text": "Cook it."}]))
    assert not one_step.ok and "at least 2 steps" in one_step.reason
    one_ing = dispatch(ctx, "create_recipe", dict(GOOD, ingredients=[{"name": "rice", "amount": 1}]))
    assert not one_ing.ok and "at least 2 ingredients" in one_ing.reason
    assert dispatch(ctx, "create_recipe", dict(GOOD, title="")).ok is False


def test_plain_string_ingredients_and_steps_are_accepted(ctx):
    """Models often send bare strings instead of objects."""
    r = dispatch(ctx, "create_recipe", {
        "title": "Simple Toast", "servings": 2,
        "ingredients": ["bread", "butter"],
        "steps": ["Toast the bread.", "Butter it."]})
    assert r.ok, r.reason
    recipe = ctx.store.get_recipe("r004")
    assert [i.name for i in recipe.ingredients] == ["bread", "butter"]
    assert len(recipe.steps) == 2


def test_meal_options_send_the_model_to_the_web_first(ctx):
    """Invented recipes were the weakest part of the first real cook: search, then invent."""
    text = render_meal_options(ctx.store, meals=8)
    assert "find_recipes" in text and "import_recipe" in text
    assert text.index("find_recipes") < text.index("create_recipe"), "invention should be the fallback"
    assert "nothing usable" in text
    assert "never propose a long shopping list" in text
