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


def _has(name: str, terms: Tuple[str, ...]) -> Optional[str]:
    low = name.lower()
    for phrase in NOT_MEAT_PHRASES:
        if phrase in low:
            return None
    for t in terms:
        if re.search(rf"\b{re.escape(t)}\b", low):
            return t
    return None


def _plant_based(name: str) -> bool:
    low = name.lower()
    return any(q in low for q in PLANT_QUALIFIERS)


def check_known(name: str, contains: Tuple[str, ...], diet: str) -> Optional[Violation]:
    """Judge an ingredient from what it is known to contain, not from how it is spelled."""
    diet = (diet or "none").lower()
    if diet == "none":
        return None
    have = {c.strip().lower() for c in contains}
    for category in _EXCLUDED_BY_DIET.get(diet, ()):
        if category in have:
            reason = _CATEGORY_REASON.get(category, category)
            return Violation(name, category, f"{name} contains {reason}", "excluded")
    for category in _CHECK_BY_DIET.get(diet, ()):
        if category in have:
            return Violation(name, category, f"{name} must be halal-certified", "check")
    return None


def check_ingredient(name: str, diet: str,
                     contains: Optional[Tuple[str, ...]] = None) -> Optional[Violation]:
    """`contains` is the model's judgement, stored on the Ingredient. When it is present it
    is the answer; the word lists below are only for ingredients nobody has judged."""
    if contains is not None:
        return check_known(name, contains, diet)
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
        return None
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
        return None
    return None


def check_recipe(ingredients: List[Any], diet: str) -> List[Violation]:
    """Accepts names or Ingredient objects; an Ingredient brings its stored judgement along."""
    out: List[Violation] = []
    for item in ingredients:
        name = getattr(item, "name", item)
        v = check_ingredient(str(name), diet, getattr(item, "contains", None))
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
