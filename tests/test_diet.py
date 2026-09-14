from __future__ import annotations

import pytest

from cooking_assistant_ai.core.diet import check_ingredient, check_recipe, summarize
from cooking_assistant_ai.core.mealplan import match_all, match_recipe, render_meal_options
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.storage.db import Store


def v(name, diet):
    got = check_ingredient(name, diet)
    return got.severity if got else None


def test_none_allows_everything():
    for name in ("pork belly", "bacon", "red wine", "parmesan", "eggs"):
        assert check_ingredient(name, "none") is None


def test_vegetarian_and_vegan_exclusions():
    assert v("chicken thighs, bone-in", "vegetarian") == "excluded"
    assert v("anchovy fillets", "vegetarian") == "excluded"
    assert v("parmesan", "vegetarian") is None          # dairy is fine for vegetarians
    assert v("parmesan", "vegan") == "excluded"
    assert v("free-range eggs", "vegan") == "excluded"
    assert v("honey", "vegan") == "excluded"
    assert v("jasmine rice", "vegan") is None


def test_plant_based_products_are_not_mistaken_for_dairy():
    """'coconut cream' and 'vegan butter' must not be flagged."""
    for name in ("coconut cream", "vegan butter", "almond milk", "oat milk", "soy yoghurt"):
        assert check_ingredient(name, "vegan") is None, name


def test_word_boundaries_and_known_false_friends():
    assert v("beef tomato", "vegetarian") is None
    assert v("vegetable stock", "vegetarian") is None
    assert v("beef brisket", "vegetarian") == "excluded"


def test_halal_separates_forbidden_from_needs_checking():
    assert v("pork shoulder", "halal") == "excluded"
    assert v("smoked bacon", "halal") == "excluded"
    assert v("dry white wine", "halal") == "excluded"
    # meat is allowed but its sourcing cannot be known from a name
    assert v("chicken thighs", "halal") == "check"
    assert v("jasmine rice", "halal") is None
    assert "must be halal-certified" in summarize(check_recipe(["chicken thighs"], "halal"))


def test_diet_persists_in_the_store():
    store = Store(":memory:")
    assert store.diet == "none"
    assert store.set_diet("VEGAN ") == "vegan"
    assert store.diet == "vegan"
    with pytest.raises(ValueError):
        store.set_diet("paleo")


def test_meal_options_exclude_and_explain_forbidden_recipes():
    store = Store(":memory:")
    store.set_diet("vegetarian")
    text = render_meal_options(store)
    assert "DIET: vegetarian" in text
    assert "NOT ALLOWED on a vegetarian diet" in text
    assert "Roast Chicken Thighs" in text.split("NOT ALLOWED")[1]
    # and the allowed section must not contain it
    assert "Roast Chicken Thighs" not in text.split("NOT ALLOWED")[0]
    assert "Jasmine Rice" in text.split("NOT ALLOWED")[0]


def test_forbidden_recipes_sort_last_and_are_flagged(ctx):
    ctx.store.set_diet("vegan")
    matches = match_all(ctx.store)
    assert not matches[-1].diet_ok
    chicken = next(m for m in matches if m.recipe.id == "r001")
    assert not chicken.diet_ok and "not vegan" in chicken.diet_note
    # butter in the chicken recipe is also flagged for vegans
    assert any("butter" in x.reason for x in chicken.violations)


def test_set_diet_tool_validates_and_reports(ctx):
    bad = dispatch(ctx, "set_diet", {"diet": "keto"})
    assert not bad.ok and "must be one of" in bad.reason
    ok = dispatch(ctx, "set_diet", {"diet": "halal"})
    assert ok.ok and ok.message == "diet set to halal"
    assert ctx.store.diet == "halal"
    assert "DIET: halal" in ok.state


def test_diet_reaches_the_model_prompt(session, store):
    from cooking_assistant_ai.core.context import assemble_context

    store.set_diet("vegan")
    msgs = assemble_context(session, "what should I cook", session.started_at, store=store)
    system = msgs[0]["content"]
    assert "DIET: vegan" in system and "without exception" in system
    # and it is absent when unrestricted, so we do not waste context
    store.set_diet("none")
    assert "DIET:" not in assemble_context(session, "hi", session.started_at, store=store)[0]["content"]


# ------------------------------------------- what an ingredient is, not how it is spelled

def test_hidden_animal_products_a_word_list_cannot_see():
    """Every one of these was allowed before, because the name contains no offending word."""
    from cooking_assistant_ai.core.diet import check_ingredient

    hidden = [
        ("caesar dressing", ("seafood",), "vegetarian"),   # anchovies
        ("kimchi", ("seafood",), "vegetarian"),            # fish sauce
        ("parmesan", ("dairy", "meat"), "vegetarian"),     # animal rennet
        ("rennet", ("meat",), "vegetarian"),
        ("bone broth", ("meat",), "vegetarian"),
        ("marshmallows", ("meat",), "vegan"),              # gelatin
        ("refried beans", ("meat",), "vegan"),             # lard
    ]
    for name, contains, diet in hidden:
        v = check_ingredient(name, diet, contains)
        # Surfaced, not banned: the model is unreliable enough at recalling what a product is
        # made of that it is only ever allowed to raise a warning.
        assert v is not None and v.severity == "check", f"{name} slipped past {diet}"


def test_a_stored_judgement_also_prevents_false_alarms():
    from cooking_assistant_ai.core.diet import check_ingredient

    # the name says beef, the food is a tomato
    assert check_ingredient("beef tomato", "halal", ()) is None
    assert check_ingredient("vegan butter", "vegan", ()) is None
    assert check_ingredient("oat milk", "vegan", ()) is None


def test_the_model_can_warn_but_never_ban():
    """Three passes over a real vegan library produced "tofu contains meat", "flour contains
    meat" and "baguette contains meat". Trusting that to exclude would have refused the cook
    their own recipes, so a model judgement can only raise a check-the-label note."""
    from cooking_assistant_ai.core.diet import check_ingredient

    wrong = check_ingredient("tofu", "vegan", ("meat",))
    assert wrong is not None and wrong.severity == "check"

    # the word lists keep the power to exclude, because they are precise where it is wrong
    assert check_ingredient("chicken stock", "vegan", ()).severity == "excluded"
    assert check_ingredient("lard", "halal", ()).severity == "excluded"
    assert check_ingredient("olive oil", "vegan", ()) is None


def test_no_real_recipe_is_refused_by_a_model_guess(ctx):
    from cooking_assistant_ai.core.diet import check_recipe
    from dataclasses import replace

    recipe = ctx.store.get_recipe("r002")
    sabotaged = tuple(replace(i, contains=("meat",)) for i in recipe.ingredients)
    verdicts = check_recipe(list(sabotaged), "vegan")
    assert verdicts and all(v.severity == "check" for v in verdicts)


def test_halal_separates_forbidden_from_unverifiable():
    from cooking_assistant_ai.core.diet import check_ingredient

    assert check_ingredient("lard", "halal", ("pork",)).severity == "excluded"
    assert check_ingredient("mirin", "halal", ("alcohol",)).severity == "excluded"
    # meat is allowed but no ingredient list can certify how it was slaughtered
    assert check_ingredient("chicken breast", "halal", ("meat",)).severity == "check"


def test_unjudged_ingredients_still_fall_back_to_the_word_lists():
    from cooking_assistant_ai.core.diet import check_ingredient

    assert check_ingredient("chicken stock", "vegan") is not None   # contains=None
    assert check_ingredient("olive oil", "vegan") is None


def test_the_judgement_travels_with_the_ingredient(ctx):
    """check_recipe takes Ingredients, so the stored answer is used rather than the name."""
    from dataclasses import replace
    from cooking_assistant_ai.core.diet import check_recipe
    from cooking_assistant_ai.model.types import Ingredient

    plain = [Ingredient(id="i1", name="caesar dressing", amount=1)]
    assert check_recipe(plain, "vegetarian") == []           # unjudged: the name hides it
    judged = [replace(plain[0], contains=("seafood",))]
    assert check_recipe(judged, "vegetarian"), "the stored judgement was ignored"
