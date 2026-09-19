"""The rule that decides whether a model's composition claim is allowed to ban a dish.

Backfilling the real library caught the model, inside the big per-recipe call, calling tofu
meat, flour and a baguette dairy, sesame oil seafood and soy sauce dairy. Every one of those
would have sat in the database quietly refusing a vegan recipe. A second isolated pass cleared
all of them, which is what these tests pin down.
"""
import pytest

from cooking_assistant_ai.core.diet import check_ingredient, reconcile


def test_both_passes_agree_so_it_bans():
    assert reconcile((["dairy"], []), (["dairy"], [])) == (("dairy",), ())


def test_one_pass_alone_never_bans():
    """The tofu-is-meat case: first pass condemns, second clears, nothing is stored."""
    assert reconcile((["meat"], []), ([], [])) == ((), ())


def test_second_pass_alone_never_bans():
    assert reconcile(([], []), (["meat"], [])) == ((), ())


def test_disagreement_on_certainty_is_demoted_not_dropped():
    """One says definite, one says brand-dependent: warn, do not ban, and do not forget."""
    assert reconcile((["seafood"], []), ([], ["seafood"])) == ((), ("seafood",))


def test_agreed_maybe_stays_maybe():
    assert reconcile(([], ["alcohol"]), ([], ["alcohol"])) == ((), ("alcohol",))


def test_categories_are_judged_independently():
    contains, maybe = reconcile((["dairy", "egg"], []), (["dairy"], ["egg"]))
    assert contains == ("dairy",)
    assert maybe == ("egg",)


def test_casing_and_padding_do_not_split_a_category():
    assert reconcile(([" Dairy "], []), (["dairy"], [])) == (("dairy",), ())


@pytest.mark.parametrize("name", [
    "tofu", "flour", "baguette", "sesame oil", "soy sauce", "extra firm tofu, pressed",
])
def test_the_real_false_positives_are_not_stored(name):
    """Exactly the names the combined call got wrong, cleared by the second pass."""
    assert reconcile((["meat"], ["dairy", "seafood"]), ([], [])) == ((), ())
    # and the word lists must not object to them either, or the fix above is moot
    assert check_ingredient(name, "vegan") is None


@pytest.mark.parametrize("name,diet", [
    ("non-dairy milk", "vegan"),
    ("dairy-free creamer", "vegan"),
    ("mock eggs", "vegan"),
    ("vegan mayo", "vegan"),
    ("plant-based mince", "vegan"),
    ("flax egg", "vegan"),
    ("meatless crumbles", "vegetarian"),
])
def test_negated_names_are_not_blocked(name, diet):
    assert check_ingredient(name, diet) is None


@pytest.mark.parametrize("name,diet", [
    ("whole milk", "vegan"),
    ("pork belly", "halal"),
    ("eggs", "vegan"),
])
def test_the_real_thing_is_still_blocked(name, diet):
    assert check_ingredient(name, diet) is not None
