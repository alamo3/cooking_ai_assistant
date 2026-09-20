"""What is missing before a single pan comes out.

Finding out you have no coconut milk forty minutes in, with three pans going, is the worst
time to find out. This checks every loaded recipe against the pantry up front.

Deliberately derived, never stored: the report recomputes from the inventory, the loaded
recipes and their overlays, so it clears itself the moment the cook tops up the pantry or the
model substitutes something. That is what lets it sit in the model's context every turn
without becoming a stale nag it learns to ignore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cooking_assistant_ai.core.mealplan import STAPLES, Need, match_recipe
from cooking_assistant_ai.model.types import Session

# Swaps that hold in almost any recipe. Offered as a starting point so the assistant can be
# concrete immediately; it is expected to judge whether one suits the dish, and to propose
# something better when it does not.
COMMON_SWAPS = {
    "butter": ["olive oil", "ghee", "neutral oil"],
    "olive oil": ["any neutral oil", "butter"],
    "heavy cream": ["coconut cream", "full-fat milk plus a little butter"],
    "milk": ["any plant milk", "water plus a little butter"],
    "coconut milk": ["heavy cream", "cashew cream"],
    "shallot": ["small onion"],
    "onion": ["shallot", "leek"],
    "garlic": ["garlic powder, a quarter teaspoon per clove"],
    "lemon": ["lime", "a splash of vinegar"],
    "lime": ["lemon"],
    "white wine": ["stock plus a splash of vinegar"],
    "stock": ["water plus a stock cube", "water and extra seasoning"],
    "soy sauce": ["tamari", "coconut aminos"],
    "parsley": ["cilantro", "chives"],
    "cilantro": ["parsley"],
    "thyme": ["oregano", "rosemary"],
    "buttermilk": ["milk with a squeeze of lemon"],
    "cornstarch": ["flour, double the amount"],
    "breadcrumbs": ["crushed crackers", "rolled oats"],
    "gochujang": ["sriracha with a little miso"],
    "udon": ["any thick noodle"],
}


def swaps_for(name: str) -> List[str]:
    """Suggestions for an ingredient, matched on whole words so 'garlic cloves' finds garlic."""
    low = name.lower()
    for key, options in COMMON_SWAPS.items():
        if key == low or key in low.split() or low.startswith(key + " ") or low.endswith(" " + key):
            return options
    return []


@dataclass
class Gap:
    recipe_id: str
    recipe_title: str
    need: Need

    @property
    def staple(self) -> bool:
        return self.need.key in STAPLES

    @property
    def swaps(self) -> List[str]:
        return swaps_for(self.need.name)

    def line(self) -> str:
        text = self.need.shortfall_text()
        swaps = self.swaps
        tail = f" (try {' or '.join(swaps[:2])})" if swaps else ""
        return f"{text} for {self.recipe_title}{tail}"


@dataclass
class Report:
    gaps: List[Gap] = field(default_factory=list)
    checked: int = 0

    @property
    def blocking(self) -> List[Gap]:
        """Missing things that are not store-cupboard staples: the ones worth stopping for."""
        return [g for g in self.gaps if not g.staple]

    @property
    def staples(self) -> List[Gap]:
        return [g for g in self.gaps if g.staple]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            # key rides along so the tablet can draw the same glyph it uses everywhere else
            # for this ingredient, rather than a second guess from the display name.
            "missing": [{"recipe_id": g.recipe_id, "recipe": g.recipe_title,
                         "name": g.need.name, "key": g.need.key,
                         "text": g.need.shortfall_text(),
                         "staple": g.staple, "swaps": g.swaps}
                        for g in self.gaps],
        }


def check(store, session: Session, scale: float = 1.0) -> Report:
    """Every loaded recipe against the pantry, with substitutions already applied.

    Overlays matter here: once the cook swaps butter for olive oil the recipe no longer needs
    butter, and a report that kept asking for it would be worse than none at all.
    """
    report = Report()
    for recipe in session.recipes.values():
        overlay = session.overlays.get(recipe.id)
        factor = scale * (overlay.scale_factor if overlay else 1.0)
        match = match_recipe(store, recipe, factor)
        skip = _not_needed(recipe, overlay)
        report.checked += len(match.needs)
        for need, ingredient in zip(match.needs, recipe.ingredients):
            if need.have_enough or ingredient.id in skip:
                continue
            report.gaps.append(Gap(recipe.id, recipe.title, need))
    return report


def _not_needed(recipe, overlay) -> set:
    """Ingredients the cook no longer has to own.

    Two cases, both from watching a real exchange go wrong. An ingredient whose every step
    has been skipped is not needed at all. And one the cook chose as a substitute is assumed
    to be in the kitchen — they just said they would use it — otherwise accepting a swap
    immediately produces a fresh warning about the thing you swapped to, which is maddening.
    """
    out = set()
    if overlay is None:
        return out
    out.update(sub.ingredient_id for sub in overlay.substitutions)
    if overlay.skipped_steps:
        for ingredient in recipe.ingredients:
            used_by = [st for st in recipe.steps if ingredient.id in st.ingredient_ids]
            if used_by and all(st.id in overlay.skipped_steps for st in used_by):
                out.add(ingredient.id)
    return out


def render(report: Report, staples_too: bool = False) -> str:
    """The block the model sees. Silent when there is nothing worth stopping for.

    Staples alone do not earn a block in every turn's context: "water is not recorded" is
    noise, and a warning that cries wolf is one the model learns to skip past. They are still
    reported when the cook explicitly asks for a check.
    """
    if not report.gaps or (not report.blocking and not staples_too):
        return ""
    lines = ["MISSING INGREDIENTS (checked against the pantry)"]
    for gap in report.blocking:
        lines.append(f"  - {gap.line()}")
    if report.staples:
        lines.append("  staples not recorded, probably fine: "
                     + ", ".join(g.need.name for g in report.staples))
    if report.blocking:
        lines.append("Tell the cook what is missing before they start anything, and offer a "
                     "substitution for each. If they accept one, call substitute; if they say "
                     "they have it after all, call add_stock. Do not plan around an ingredient "
                     "that is not there.")
    return "\n".join(lines)
