"""Dietary restrictions, checked against ingredient names.

Two severities, because they are genuinely different questions:

* ``excluded``  the ingredient itself breaks the restriction (pork in a halal kitchen,
  chicken in a vegetarian one). Never recommend the recipe as it stands.
* ``check``     the ingredient is allowed but depends on sourcing. Meat can be halal, but
  only if it was slaughtered that way, and no ingredient list can tell you. Say so rather
  than pretending to certify it.

Plant-based products are excluded from the dairy rules by qualifier, so "coconut cream" and
"vegan butter" are not mistaken for dairy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def check_ingredient(name: str, diet: str) -> Optional[Violation]:
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


def check_recipe(ingredient_names: List[str], diet: str) -> List[Violation]:
    out: List[Violation] = []
    for name in ingredient_names:
        v = check_ingredient(name, diet)
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
