"""A substitution is a real edit to the recipe, and it has to show up everywhere.

Two bugs live here. The first: `substitute` rewrote the ingredient list while every step
still said "melt the butter", so from the cook's chair the swap looked like it did nothing.
The second: the swap only lived in the session, so it was gone by the next cook.
"""
from __future__ import annotations

from cooking_assistant_ai.core.plan import apply_substitution, render_cook_plan, substitute_text
from cooking_assistant_ai.core.render import recipe_view, render_all_recipes
from cooking_assistant_ai.core.tools import ToolContext, dispatch
from cooking_assistant_ai.model.types import Overlay, Recipe, Session, Step, Substitution


def _swap(ctx, **kw):
    assert dispatch(ctx, "load_recipe", {"recipe": "r001"}).ok
    r = dispatch(ctx, "substitute", dict({"recipe_id": "r001", "ingredient_id": "butter",
                                          "replacement": "olive oil"}, **kw))
    assert r.ok, r.reason
    return r


def test_step_text_follows_the_substitution_everywhere(ctx):
    _swap(ctx)
    rendered = render_all_recipes(ctx.session)               # what the model reads
    plan = render_cook_plan(ctx.session, ctx.clock.now())    # the NOW line, and what is spoken
    card = recipe_view(ctx.session, ctx.session.recipes["r001"])
    steps = " ".join(s["text"] for s in card["steps"])       # the tablet's step list

    for surface, text in (("recipe", rendered), ("cook plan", plan), ("tablet", steps)):
        assert "olive oil" in text, f"{surface} never mentions the replacement"
        assert "melt the butter" not in text.lower(), f"{surface} still tells the cook to use butter"

    assert any(s["substituted"] for s in card["steps"])      # the UI can badge it
    assert "butter -> olive oil" in card["changes"]          # and the swap is still traceable
    assert card["ingredients"][1]["substituted_for"] == "butter"


def test_substitution_survives_the_session(ctx):
    before = ctx.store.get_recipe("r001")
    step_ids = [s.id for s in before.steps]
    ingredient_ids = [i.id for i in before.ingredients]

    r = _swap(ctx)
    assert "saved to the recipe" in (r.message or "")
    stored = ctx.store.get_recipe("r001")
    assert stored.ingredient("r001-i2").name == "olive oil"
    assert not any(i.name == "butter" for i in stored.ingredients)
    assert "olive oil" in stored.step("r001-s3").text and "butter" not in stored.step("r001-s3").text
    # ids are untouched, so tasks, timers and completed steps still line up
    assert [s.id for s in stored.steps] == step_ids
    assert [i.id for i in stored.ingredients] == ingredient_ids

    # a fresh cook picks the recipe up already swapped, with no overlay involved
    later = Session(id="later", started_at=ctx.clock.now())
    later.add_recipe(ctx.store.get_recipe("r001"))
    assert "olive oil" in render_all_recipes(later)
    assert "butter" not in render_all_recipes(later)


def test_a_second_swap_resolves_the_new_name(ctx):
    """'Actually, ghee instead of the olive oil' has to find the ingredient by its new name."""
    _swap(ctx)
    r = dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "olive oil",
                                     "replacement": "ghee"})
    assert r.ok, r.reason
    stored = ctx.store.get_recipe("r001")
    assert stored.ingredient("r001-i2").name == "ghee"
    assert "ghee" in stored.step("r001-s3").text


def test_substitution_pinned_to_one_step_stays_in_the_session(ctx):
    r = _swap(ctx, ingredient_id="butter", replacement="ghee", at_step="r001-s3")
    assert "for this step only" in (r.message or "")
    assert ctx.store.get_recipe("r001").ingredient("r001-i2").name == "butter"  # library untouched
    card = recipe_view(ctx.session, ctx.session.recipes["r001"])
    assert "ghee" in next(s["text"] for s in card["steps"] if s["id"] == "r001-s3")


def test_pinned_substitution_leaves_the_other_steps_alone(ctx):
    recipe = Recipe(id="rx", title="Twice Buttered", servings=2,
                    ingredients=ctx.store.get_recipe("r001").ingredients,
                    steps=(Step(id="rx-s1", text="Melt the butter."),
                           Step(id="rx-s2", text="Finish with butter.")))
    ov = Overlay(recipe_id="rx")
    ov.substitutions.append(Substitution(ingredient_id="r001-i2", replacement="ghee",
                                         at_step="rx-s2"))
    assert substitute_text(recipe.steps[0].text, recipe, ov, "rx-s1") == "Melt the butter."
    assert substitute_text(recipe.steps[1].text, recipe, ov, "rx-s2") == "Finish with ghee."


def test_matching_is_conservative(ctx):
    """Plurals match; unrelated words that merely contain the name do not."""
    recipe = Recipe(id="ry", title="T", servings=1,
                    ingredients=ctx.store.get_recipe("r001").ingredients,
                    steps=(Step(id="ry-s1", text="Add butter and butters."),
                           Step(id="ry-s2", text="Spread the buttermilk.")))
    edited = apply_substitution(recipe, "r001-i2", "ghee")
    assert edited.step("ry-s1").text == "Add ghee and ghee."
    assert edited.step("ry-s2").text == "Spread the buttermilk."
