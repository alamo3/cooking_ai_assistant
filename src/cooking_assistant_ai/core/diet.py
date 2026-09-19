"""Dietary restrictions, checked against ingredient names.

Two severities, because they are genuinely different questions:

* ``excluded``  the ingredient itself breaks the restriction (pork in a halal kitchen,
  chicken in a vegetarian one). Never recommend the recipe as it stands.
* ``check``     the ingredient is allowed but depends on sourcing. Meat can be halal, but
  only if it was slaughtered that way, and no ingredient list can tell you. Say so rather
  than pretending to certify it.

What an ingredient contains is decided by the model when the recipe is imported and stored on
the Ingredient, because it is a fact about food rather than about spelling. A word list can
catch "chicken stock"; it cannot know that caesar dressing has anchovies in it, that
marshmallows are gelatin, or that refried beans are traditionally lard, since none of those
names contain the offending word. The lists below remain as the fallback for ingredients
nobody has judged, with plant qualifiers so "coconut cream" and "vegan butter" are not
mistaken for dairy - a patch that only exists because spelling was being used as evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DIETS = ("none", "halal", "vegetarian", "vegan")

MEAT = (
    "chicken", "beef", "pork", "lamb", "mutton", "veal", "duck", "turkey", "goose", "rabbit",
    "venison", "bacon", "ham", "prosciutto", "pancetta", "chorizo", "salami", "sausage",
    "pepperoni", "lard", "tallow", "suet", "gelatin", "gelatine", "liver", "oxtail", "brisket",
)
SEAFOOD = (
    "fish", "salmon", "tuna", "cod", "haddock", "anchovy", "anchovies", "sardine", "mackerel",
    "prawn", "prawns", "shrimp", "crab", "lobster", "oyster", "oysters", "mussel", "mussels",
    "clam", "clams", "squid", "octopus", "scallop", "scallops", "worcestershire",
)
DAIRY_AND_EGG = (
    "milk", "cream", "butter", "cheese", "parmesan", "mozzarella", "cheddar", "feta", "ricotta",
    "yoghurt", "yogurt", "egg", "eggs", "honey", "ghee", "mayonnaise", "custard", "creme",
    "buttermilk", "mascarpone",
)
PORK = ("pork", "bacon", "ham", "prosciutto", "pancetta", "chorizo", "salami", "pepperoni",
        "lard", "gelatin", "gelatine")
ALCOHOL = ("wine", "beer", "lager", "ale", "rum", "brandy", "vodka", "whisky", "whiskey",
           "sherry", "mirin", "sake", "vermouth", "kirsch", "bourbon", "cider", "liqueur")

# "coconut cream" and "vegan butter" are not dairy; "beef tomato" is not beef.
PLANT_QUALIFIERS = ("coconut", "almond", "soy", "soya", "oat", "cashew", "rice", "hemp",
                    "vegan", "plant", "nut", "pea protein")
NOT_MEAT_PHRASES = ("beef tomato", "beefsteak tomato", "chicken of the woods",
                    "vegetable stock", "vegetable broth", "mock", "vegetarian")

# A name that denies the thing it names. The word lists read "non-dairy milk" as milk and
# "flax egg" as egg, and once the lists became the only thing allowed to exclude, that
# blocked a cook's own vegan recipes. Negation is the one bit of grammar worth encoding here,
# because it inverts the meaning of every term that follows it.
NEGATIONS = ("non-dairy", "non dairy", "nondairy", "dairy-free", "dairy free", "dairyfree",
             "egg-free", "egg free", "eggless", "meat-free", "meat free", "meatless",
             "free-from", "free from", "imitation", "faux", "mock", "vegan", "plant-based",
             "plant based", "substitute", "replacer", "replacement", "alternative",
             "-style", " style", "analogue", "analog", "not-", "no-dairy")

# Plant-based egg and dairy stand-ins whose names give no other clue.
NOT_ANIMAL_PHRASES = ("flax egg", "chia egg", "aquafaba", "just egg", "tofu scramble",
                      "nutritional yeast", "cashew cream", "oat cream", "soy cream",
                      "coconut yoghurt", "coconut yogurt", "nut milk", "seed milk")


# What an ingredient actually contains, decided by the model when the recipe is imported and
# stored on the Ingredient. The word lists below can catch "chicken stock"; they cannot know
# that caesar dressing has anchovies in it, that marshmallows are gelatin, or that refried
# beans are traditionally lard, because none of those names contain the offending word. Those
# are facts about food, not about spelling.
ANIMAL_CATEGORIES = ("meat", "pork", "seafood", "dairy", "egg", "honey", "alcohol")

_EXCLUDED_BY_DIET: Dict[str, Tuple[str, ...]] = {
    "vegan": ("meat", "pork", "seafood", "dairy", "egg", "honey"),
    "vegetarian": ("meat", "pork", "seafood"),
    "halal": ("pork", "alcohol"),
}
# Allowed, but not certifiable from an ingredient list.
_CHECK_BY_DIET: Dict[str, Tuple[str, ...]] = {"halal": ("meat",)}

_CATEGORY_REASON = {
    "meat": "meat", "pork": "pork or pork-derived", "seafood": "fish or shellfish",
    "dairy": "dairy", "egg": "egg", "honey": "honey", "alcohol": "alcoholic",
}


@dataclass(frozen=True)
class Violation:
    ingredient: str
    term: str
    reason: str
    severity: str  # "excluded" | "check"


def _denied(name: str) -> bool:
    """True when the name itself says it is not the animal product it mentions."""
    low = name.lower()
    return (any(p in low for p in NEGATIONS)
            or any(p in low for p in NOT_ANIMAL_PHRASES))


def _has(name: str, terms: Tuple[str, ...]) -> Optional[str]:
    low = name.lower()
    if _denied(low):
        return None
    for phrase in NOT_MEAT_PHRASES:
        if phrase in low:
            return None
    for t in terms:
        if re.search(rf"\b{re.escape(t)}\b", low):
            return t
    return None


def _plant_based(name: str) -> bool:
    low = name.lower()
    return any(q in low for q in PLANT_QUALIFIERS) or _denied(low)


def check_known(name: str, contains: Tuple[str, ...], diet: str,
                may_contain: Tuple[str, ...] = ()) -> Optional[Violation]:
    """Judge an ingredient from what it is known to contain, not from how it is spelled.

    `may_contain` is the brand-dependent case - anchovy in some gochujang, fish sauce in some
    kimchi. Treating those as definite would refuse a cook their own recipes, so they get the
    same "check the label" severity that halal meat gets.
    """
    diet = (diet or "none").lower()
    if diet == "none":
        return None
    have = {c.strip().lower() for c in contains}
    maybe = {c.strip().lower() for c in may_contain} - have
    for category in _EXCLUDED_BY_DIET.get(diet, ()):
        if category in have:
            reason = _CATEGORY_REASON.get(category, category)
            return Violation(name, category, f"{name} contains {reason}", "excluded")
    for category in _CHECK_BY_DIET.get(diet, ()):
        if category in have:
            return Violation(name, category, f"{name} must be halal-certified", "check")
    for category in _EXCLUDED_BY_DIET.get(diet, ()):
        if category in maybe:
            reason = _CATEGORY_REASON.get(category, category)
            return Violation(name, category,
                             f"some {name} contains {reason}; check the label", "check")
    return None


def check_ingredient(name: str, diet: str, contains: Optional[Tuple[str, ...]] = None,
                     may_contain: Optional[Tuple[str, ...]] = None) -> Optional[Violation]:
    """`contains` is the model's judgement, stored on the Ingredient. When it is present it
    is the answer; the word lists below are only for ingredients nobody has judged."""
    # The model's judgement only ever *adds* a warning; it never bans a dish. Measured over
    # three passes of a real vegan library it produced "tofu contains meat", "flour contains
    # meat" and "baguette contains meat", which would have refused the cook their own
    # recipes. It is good at judging a sentence and unreliable at recalling what a product is
    # made of, so the word lists below - which are precise, if narrow - keep the power to
    # exclude, and anything the model raises becomes a check-the-label note.
    # A clearance overrides the word lists; a condemnation does not. That is the asymmetry the
    # measurements support: the model is reliable at reading a name ("non-dairy milk is not
    # dairy", "Oatly is oat milk") and unreliable at recalling what a product is made of (it
    # claimed tofu contains meat). No word list will ever know the brands, and a false
    # clearance costs a missed warning where a false exclusion costs the cook their dinner.
    if contains is not None and not contains and not may_contain:
        return None
    advisory = None
    if contains is not None or may_contain is not None:
        found = check_known(name, contains or (), diet, may_contain or ())
        if found is not None:
            advisory = Violation(found.ingredient, found.term,
                                 f"{name} may contain {_CATEGORY_REASON.get(found.term, found.term)}; "
                                 f"check the label", "check")
    diet = (diet or "none").lower()
    if diet == "none":
        return None
    if diet in ("vegetarian", "vegan"):
        term = _has(name, MEAT) or _has(name, SEAFOOD)
        if term:
            return Violation(name, term, f"{term} is not {diet}", "excluded")
        if diet == "vegan" and not _plant_based(name):
            term = _has(name, DAIRY_AND_EGG)
            if term:
                return Violation(name, term, f"{term} is not vegan", "excluded")
        return advisory
    if diet == "halal":
        term = _has(name, PORK)
        if term:
            return Violation(name, term, f"{term} is pork or pork-derived", "excluded")
        term = _has(name, ALCOHOL)
        if term:
            return Violation(name, term, f"{term} is alcoholic", "excluded")
        term = _has(name, MEAT)
        if term:
            return Violation(name, term, f"{term} must be halal-certified", "check")
        return advisory
    return advisory


def check_recipe(ingredients: List[Any], diet: str) -> List[Violation]:
    """Accepts names or Ingredient objects; an Ingredient brings its stored judgement along."""
    out: List[Violation] = []
    for item in ingredients:
        name = getattr(item, "name", item)
        v = check_ingredient(str(name), diet, getattr(item, "contains", None),
                             getattr(item, "may_contain", None))
        if v is not None:
            out.append(v)
    return out


def summarize(violations: List[Violation]) -> str:
    excluded = [v for v in violations if v.severity == "excluded"]
    check = [v for v in violations if v.severity == "check"]
    bits = []
    if excluded:
        bits.append("not allowed: " + ", ".join(sorted({v.reason for v in excluded})))
    if check:
        bits.append("check sourcing: " + ", ".join(sorted({v.reason for v in check})))
    return "; ".join(bits)


def describe(diet: str) -> str:
    diet = (diet or "none").lower()
    return {
        "none": "No dietary restriction.",
        "halal": ("Halal kitchen: never suggest pork, pork-derived ingredients or alcohol. Meat "
                  "is fine but remind the cook it must be halal-certified."),
        "vegetarian": "Vegetarian kitchen: never suggest meat, poultry, fish or shellfish.",
        "vegan": ("Vegan kitchen: never suggest meat, poultry, fish, shellfish, dairy, eggs "
                  "or honey."),
    }.get(diet, "No dietary restriction.")
