from __future__ import annotations

import pytest

from cooking_assistant_ai.core.mealplan import (
    match_all,
    match_recipe,
    options_dict,
    render_meal_options,
    render_shopping_list,
    shopping_list,
)
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.storage.db import Store


@pytest.fixture
def stocked() -> Store:
    return Store(":memory:")  # seeded pantry covers most of the seed recipes


def test_a_fully_stocked_recipe_is_cookable_now(stocked):
    m = match_recipe(stocked, stocked.get_recipe("r001"))
    assert m.cookable_now and not m.missing_key
    assert m.have == len(m.needs) and m.servings == 4


def test_missing_ingredient_blocks_and_is_named(stocked):
    stocked.remove_stock("brussels sprouts")
    m = match_recipe(stocked, stocked.get_recipe("r003"))
    assert not m.cookable_now
    assert [n.name for n in m.missing_key] == ["brussels sprouts"]
    assert "450 g brussels sprouts" in m.line()


def test_staples_never_block_a_recipe(stocked):
    """Water and salt missing should not make a recipe uncookable."""
    stocked.remove_stock("kosher salt")
    m = match_recipe(stocked, stocked.get_recipe("r002"))
    assert m.cookable_now and m.missing  # short of staples, still cookable
    assert all(n.key in {"salt", "water"} for n in m.missing)


def test_recipe_names_match_pantry_names_loosely(stocked):
    """'Garlic cloves, smashed' in a recipe must find 'garlic' in the pantry."""
    m = match_recipe(stocked, stocked.get_recipe("r001"))
    garlic = next(n for n in m.needs if n.key == "garlic")
    assert garlic.have_enough and garlic.in_stock == 6


def test_scaling_increases_what_is_needed(stocked):
    stocked.set_stock("chicken thighs", 8, None)
    assert match_recipe(stocked, stocked.get_recipe("r001"), scale=1.0).cookable_now
    m2 = match_recipe(stocked, stocked.get_recipe("r001"), scale=3.0)
    assert m2.servings == 12
    assert any(n.key == "chicken thigh" and not n.have_enough for n in m2.needs)


def test_best_stocked_recipes_come_first(stocked):
    stocked.remove_stock("brussels sprouts")
    order = [m.recipe.id for m in match_all(stocked)]
    assert order[-1] == "r003"  # the one needing shopping sorts last


def test_shopping_list_subtracts_stock_and_merges(stocked):
    stocked.remove_stock("brussels sprouts")
    stocked.set_stock("chicken thighs", 2, None)
    needs = shopping_list(stocked, [stocked.get_recipe("r001"), stocked.get_recipe("r003")])
    by = {n.key: n for n in needs}
    assert "brussel sprout" in by
    assert by["chicken thigh"].short_by == 6  # need 8, have 2
    assert "6 more chicken thighs" in by["chicken thigh"].shortfall_text()
    text = render_shopping_list(stocked, [stocked.get_recipe("r001")])
    assert "chicken thighs" in text


def test_nothing_to_buy_says_so(stocked):
    assert "nothing needed" in render_shopping_list(stocked, [stocked.get_recipe("r001")])


def test_suggest_meals_tool_reports_portions_and_options(ctx):
    ctx.store.remove_stock("brussels sprouts")
    r = dispatch(ctx, "suggest_meals", {"meals": 8})
    assert r.ok
    assert "about 8 portion(s)" in r.state
    assert "COOKABLE NOW" in r.state and "Would need shopping" in r.state
    assert "brussels sprouts" in r.state


def test_shopping_list_tool_rejects_unknown_recipes(ctx):
    bad = dispatch(ctx, "shopping_list", {"recipe_ids": ["nope"]})
    assert not bad.ok and "no stored recipe 'nope'" in bad.reason
    ok = dispatch(ctx, "shopping_list", {"recipe_ids": ["r001"]})
    assert ok.ok and "SHOPPING LIST" in ok.state


def test_options_dict_shape(stocked):
    d = options_dict(stocked, meals=6)
    assert d["meals_wanted"] == 6
    r1 = next(x for x in d["recipes"] if x["id"] == "r001")
    assert r1["cookable_now"] and r1["servings"] == 4 and r1["total"] == 7
