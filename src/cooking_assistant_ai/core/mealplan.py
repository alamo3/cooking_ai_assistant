"""Meal planning from what is actually in the pantry.

Code answers the factual questions (what does each recipe need, what is in stock, what is
missing, how many portions does it make) and the model does the choosing, because "what
would make a good week of food" is judgement, not arithmetic.

Ingredient names are matched with the same normalizer the mise en place grouping uses, so
"Garlic cloves, smashed" in a recipe finds "garlic" in the pantry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cooking_assistant_ai.core.diet import Violation, check_recipe, describe, summarize
from cooking_assistant_ai.core.fmt import fmt_amount, fmt_ingredient
from cooking_assistant_ai.core.plan import ingredient_key
from cooking_assistant_ai.model.types import Recipe
from cooking_assistant_ai.storage.db import Store

# Things almost every kitchen has; missing ones are worth mentioning but never block a recipe.
STAPLES = {"salt", "pepper", "water", "oil", "olive oil", "sugar", "flour", "butter"}


@dataclass
class Need:
    name: str
    key: str
    needed: float
    unit: Optional[str]
    in_stock: float = 0.0
    stock_unit: Optional[str] = None

    @property
    def comparable(self) -> bool:
        """Amounts only mean anything when the units agree."""
        return bool(self.unit) == bool(self.stock_unit) and (
            (self.unit or "").lower() == (self.stock_unit or "").lower())

    @property
    def have_enough(self) -> bool:
        if self.in_stock <= 0:
            return False
        return self.in_stock >= self.needed if self.comparable else True

    @property
    def short_by(self) -> float:
        return max(0.0, self.needed - self.in_stock) if self.comparable else 0.0

    def text(self) -> str:
        return fmt_ingredient(self.name, self.needed, self.unit)

    def shortfall_text(self) -> str:
        if self.in_stock <= 0:
            return self.text()
        if self.comparable and self.short_by > 0:
            return f"{fmt_amount(self.short_by)}{(' ' + self.unit) if self.unit else ''} more {self.name}"
        return self.text()


@dataclass
class Match:
    recipe: Recipe
    scale: float
    needs: List[Need] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)

    @property
    def diet_ok(self) -> bool:
        return not any(v.severity == "excluded" for v in self.violations)

    @property
    def diet_note(self) -> str:
        return summarize(self.violations)

    @property
    def servings(self) -> float:
        return self.recipe.servings * self.scale

    @property
    def missing(self) -> List[Need]:
        return [n for n in self.needs if not n.have_enough]

    @property
    def missing_key(self) -> List[Need]:
        return [n for n in self.missing if n.key not in STAPLES]

    @property
    def have(self) -> int:
        return sum(1 for n in self.needs if n.have_enough)

    @property
    def cookable_now(self) -> bool:
        return not self.missing_key

    def line(self) -> str:
        bits = [f"{self.recipe.title} [{self.recipe.id}] serves {fmt_amount(self.servings)}",
                f"{self.have}/{len(self.needs)} ingredients in stock"]
        if self.cookable_now:
            bits.append("COOKABLE NOW" if not self.missing else "cookable (only staples short)")
        else:
            bits.append("need: " + ", ".join(n.shortfall_text() for n in self.missing_key))
        if self.diet_note:
            bits.append(self.diet_note)
        return " | ".join(bits)


def match_recipe(store: Store, recipe: Recipe, scale: float = 1.0,
                 diet: Optional[str] = None) -> Match:
    diet = store.diet if diet is None else diet
    m = Match(recipe=recipe, scale=scale,
              violations=check_recipe([i.name for i in recipe.ingredients], diet))
    for ing in recipe.ingredients:
        key = ingredient_key(ing.name) or ing.name.lower()
        stock = store.get_stock(key) or store.get_stock(ing.name)
        need = Need(name=ing.name.split(",")[0].strip(), key=key,
                    needed=ing.amount * scale, unit=ing.unit)
        if stock:
            need.in_stock = float(stock["amount"])
            need.stock_unit = stock["unit"]
        m.needs.append(need)
    return m


def match_all(store: Store, scale: float = 1.0, diet: Optional[str] = None) -> List[Match]:
    diet = store.diet if diet is None else diet
    matches = [match_recipe(store, r, scale, diet) for r in store.list_recipes()]
    # Diet-compatible first, then best-stocked: what the cook can start today, and is
    # allowed to eat, is the most useful thing to see.
    matches.sort(key=lambda m: (not m.diet_ok, len(m.missing_key), -m.have))
    return matches


def shopping_list(store: Store, recipes: List[Recipe], scale: float = 1.0) -> List[Need]:
    """Everything short across the chosen recipes, merged by ingredient."""
    merged: Dict[str, Need] = {}
    for r in recipes:
        for need in match_recipe(store, r, scale).needs:
            existing = merged.get(need.key)
            if existing is None:
                merged[need.key] = Need(need.name, need.key, need.needed, need.unit,
                                        need.in_stock, need.stock_unit)
            elif (existing.unit or "").lower() == (need.unit or "").lower():
                existing.needed += need.needed
    return [n for n in merged.values() if not n.have_enough]


def render_meal_options(store: Store, meals: Optional[int] = None, scale: float = 1.0,
                        diet: Optional[str] = None) -> str:
    diet = store.diet if diet is None else diet
    matches = match_all(store, scale, diet)
    lines = ["MEAL OPTIONS (what the pantry supports)"]
    if diet and diet != "none":
        lines.append(f"DIET: {diet}. {describe(diet)}")
    if meals:
        lines.append(f"The cook wants about {meals} portion(s). Recipe servings are listed; "
                     f"combine or scale recipes to reach it.")
    inventory = store.inventory()
    lines.append("In stock: " + (", ".join(fmt_ingredient(i["name"], i["amount"], i["unit"])
                                           for i in inventory) or "nothing recorded"))
    allowed = [m for m in matches if m.diet_ok]
    banned = [m for m in matches if not m.diet_ok]
    ready = [m for m in allowed if m.cookable_now]
    short = [m for m in allowed if not m.cookable_now]
    lines.append("")
    lines.append(f"Cookable with what you have ({len(ready)}):")
    lines.extend(f"  {m.line()}" for m in ready) if ready else lines.append("  none")
    if short:
        lines.append(f"Would need shopping ({len(short)}):")
        lines.extend(f"  {m.line()}" for m in short)
    if banned:
        lines.append(f"NOT ALLOWED on a {diet} diet, do not recommend these ({len(banned)}):")
        lines.extend(f"  {m.recipe.title} [{m.recipe.id}] - {m.diet_note}" for m in banned)
    lines.append("")
    lines.append("Recommend a set of meals and say how many portions it makes. You are not "
                 "limited to the list above: invent dishes around what is in stock and save each "
                 "one with create_recipe, which adds it to the library permanently. A dish may "
                 "need one or two easy things from the shop; never propose a long shopping list. "
                 "Once the cook agrees, call create_recipe for anything new, then shopping_list "
                 "for what they must buy.")
    if banned:
        lines.append(f"Never recommend anything from the not-allowed list. You may suggest a "
                     f"{diet} substitution for one if it genuinely works, and say what you changed.")
    return "\n".join(lines)


def render_shopping_list(store: Store, recipes: List[Recipe], scale: float = 1.0) -> str:
    needs = shopping_list(store, recipes, scale)
    titles = ", ".join(r.title for r in recipes)
    if not needs:
        return f"SHOPPING LIST for {titles}\nnothing needed, everything is in stock"
    key = [n for n in needs if n.key not in STAPLES]
    staples = [n for n in needs if n.key in STAPLES]
    lines = [f"SHOPPING LIST for {titles}"]
    lines.extend(f"  {n.shortfall_text()}" for n in key)
    if staples:
        lines.append("  staples to check: " + ", ".join(n.name for n in staples))
    return "\n".join(lines)


def options_dict(store: Store, meals: Optional[int] = None, scale: float = 1.0,
                 diet: Optional[str] = None) -> Dict[str, Any]:
    diet = store.diet if diet is None else diet
    return {
        "meals_wanted": meals,
        "diet": diet,
        "recipes": [
            {
                "id": m.recipe.id, "title": m.recipe.title, "servings": m.servings,
                "have": m.have, "total": len(m.needs), "cookable_now": m.cookable_now,
                "diet_ok": m.diet_ok, "diet_note": m.diet_note,
                "missing": [{"name": n.name, "text": n.shortfall_text(), "staple": n.key in STAPLES}
                            for n in m.missing],
            }
            for m in match_all(store, scale, diet)
        ],
    }
