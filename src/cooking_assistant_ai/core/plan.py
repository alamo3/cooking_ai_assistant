"""The merged cook plan: one ordered step sequence across every loaded recipe, plus
mise en place groups for ingredients that several recipes need prepped.

Everything here is derived from tasks, recipes, overlays and completed_steps. Nothing is
stored, so it can never drift from the state the tools maintain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.fmt import fmt_amount, fmt_dur, fmt_ingredient, fmt_time
from cooking_assistant_ai.model.types import Ingredient, Recipe, Session, Step

# --------------------------------------------------------------------------- ingredient matching

_DESCRIPTORS = {
    "fresh", "large", "small", "medium", "big", "chopped", "diced", "sliced", "minced", "smashed", "crushed",
    "peeled", "halved", "trimmed", "whole", "ground", "dried", "raw", "ripe", "boneless", "skinless", "bone-in",
    "extra", "virgin", "fine", "coarse", "unsalted", "salted", "low-sodium", "reduced-sodium", "cooked",
    "uncooked", "plain", "thinly", "finely", "roughly", "thick", "thin", "of", "the", "and", "a", "an", "to",
    "taste", "for", "serving", "optional", "divided", "packed", "heaping", "level", "room", "temperature",
    "softened", "melted", "cold", "warm", "hot", "sprigs", "sprig", "leaves", "leaf", "cloves", "clove",
    "pieces", "piece", "slices", "slice", "cups", "cup", "handful",
}
_ALIASES = {
    "garlic clove": "garlic", "yellow onion": "onion", "white onion": "onion", "brown onion": "onion",
    "spanish onion": "onion", "scallion": "green onion", "spring onion": "green onion",
    "coriander": "cilantro", "kosher salt": "salt", "sea salt": "salt", "table salt": "salt",
    "black pepper": "pepper", "ground pepper": "pepper", "olive oil": "olive oil", "evoo": "olive oil",
    "chicken thigh": "chicken thigh", "vegan butter": "butter", "italian parsley": "parsley",
    "flat-leaf parsley": "parsley", "roma tomato": "tomato", "plum tomato": "tomato",
}


# Pantry staples every recipe uses a pinch of; grouping them is noise, not mise en place.
_STAPLES = {"salt", "pepper", "water", "oil", "olive oil", "vegetable oil", "sunflower oil", "canola oil",
            "sugar", "black pepper", "white pepper", "ice", "cooking spray"}


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def ingredient_key(name: str) -> str:
    """Normalize an ingredient name so 'Garlic cloves, smashed' and 'garlic' match."""
    text = name.lower()
    text = re.split(r",|\(|\bor\b|\bfor\b", text, maxsplit=1)[0]
    words = [w for w in re.findall(r"[a-z][a-z-]*", text) if w not in _DESCRIPTORS]
    words = [_singular(w) for w in words]
    key = " ".join(words).strip()
    return _ALIASES.get(key, key)


def _sub_pattern(name: str) -> Optional[re.Pattern]:
    """Match an ingredient's head name inside step text ('garlic cloves, smashed' -> garlic clove/cloves)."""
    head = re.split(r",|\(", name, maxsplit=1)[0].strip()
    words = re.findall(r"[A-Za-z][A-Za-z-]*", head)
    if not words:
        return None
    parts = [re.escape(w) for w in words]
    parts[-1] = re.escape(_singular(words[-1])) + r"e?s?"  # match singular and plural alike
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.I)


def rename_in_text(text: str, name: str, replacement: str) -> str:
    """Swap one ingredient name for another inside a sentence of instructions."""
    pattern = _sub_pattern(name)
    return pattern.sub(replacement, text) if pattern is not None else text


def mentions_ingredient(text: str, name: str) -> bool:
    pattern = _sub_pattern(name)
    return bool(pattern is not None and pattern.search(text))


def apply_substitution(recipe: Recipe, ingredient_id: str, replacement: str) -> Recipe:
    """A copy of the recipe with the swap baked in: the ingredient renamed and every step
    that named it reworded. Ids are untouched, so tasks, timers and completed steps still
    point at the same things."""
    ing = recipe.ingredient(ingredient_id)
    if ing is None:
        return recipe
    return replace(
        recipe,
        ingredients=tuple(replace(i, name=replacement) if i.id == ingredient_id else i
                          for i in recipe.ingredients),
        steps=tuple(replace(s, text=rename_in_text(s.text, ing.name, replacement))
                    for s in recipe.steps),
    )


def substitute_text(text: str, recipe: Recipe, overlay, step_id: Optional[str] = None) -> str:
    """Rewrite substituted ingredient names inside a step's text.

    Without this the ingredient list says olive oil while the instruction the cook reads
    (and the assistant speaks) still says butter, so a substitution looks like it did nothing.
    A whole-recipe swap is written into the recipe itself, so this only has work to do for
    one pinned to a single step; the rest is a no-op safety net.
    """
    for sub in overlay.substitutions:
        if sub.at_step and step_id is not None and sub.at_step != step_id:
            continue
        ing = recipe.ingredient(sub.ingredient_id)
        if ing is None or ing.name.lower() == sub.replacement.lower():
            continue  # already baked into the recipe
        text = rename_in_text(text, ing.name, sub.replacement)
    return text


_PREP_VERBS = re.compile(
    r"\b(chop|dice|slice|mince|peel|trim|halve|quarter|rinse|wash|cut|grate|zest|juice|crush|smash|pat|"
    r"season|measure|soak|marinate|shred|cube|core|seed|devein|debone|prep|prepare|mix|whisk|combine|toss)\w*",
    re.I,
)


def is_prep_step(step: Step) -> bool:
    return step.appliance is None and bool(_PREP_VERBS.search(step.text))


# --------------------------------------------------------------------------- prep groups

@dataclass
class PrepEntry:
    recipe_id: str
    recipe_title: str
    ingredient: Ingredient
    amount: float  # overlay-scaled
    step_ids: List[str]
    step_numbers: List[int]


@dataclass
class PrepGroup:
    id: str
    key: str
    label: str
    entries: List[PrepEntry]
    total_text: Optional[str]
    step_ids: List[str]
    status: str  # pending | partial | done

    def summary(self) -> str:
        parts = [f"{fmt_amount(e.amount)}{(' ' + e.ingredient.unit) if e.ingredient.unit else ''} ({e.recipe_title})"
                 for e in self.entries]
        total = f" = {self.total_text} total" if self.total_text else ""
        return f"{self.label}: " + " + ".join(parts) + total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "key": self.key, "label": self.label, "summary": self.summary(),
            "total": self.total_text, "status": self.status, "step_ids": list(self.step_ids),
            "entries": [
                {"recipe_id": e.recipe_id, "recipe_title": e.recipe_title, "ingredient_id": e.ingredient.id,
                 "text": fmt_ingredient(e.ingredient.name, e.amount, e.ingredient.unit),
                 "step_ids": list(e.step_ids), "step_numbers": list(e.step_numbers)}
                for e in self.entries
            ],
        }


def prep_groups(session: Session) -> List[PrepGroup]:
    """Ingredients used by two or more loaded recipes, where at least one has a prep step."""
    by_key: Dict[str, List[PrepEntry]] = {}
    for r in session.recipes.values():
        ov = session.overlays[r.id]
        subs = {s.ingredient_id: s for s in ov.substitutions}
        for ing in r.ingredients:
            name = subs[ing.id].replacement if ing.id in subs else ing.name
            key = ingredient_key(name)
            if not key or key in _STAPLES:
                continue
            steps = [s for s in r.steps if ing.id in s.ingredient_ids and is_prep_step(s) and s.id not in ov.skipped_steps]
            entry = PrepEntry(
                recipe_id=r.id, recipe_title=r.title,
                ingredient=Ingredient(ing.id, name, ing.amount, ing.unit), amount=ing.amount * ov.scale_factor,
                step_ids=[s.id for s in steps], step_numbers=[r.step_index(s.id) for s in steps],
            )
            by_key.setdefault(key, []).append(entry)

    groups: List[PrepGroup] = []
    for key, entries in by_key.items():
        recipes = {e.recipe_id for e in entries}
        step_ids = [sid for e in entries for sid in e.step_ids]
        if len(recipes) < 2 or not step_ids:
            continue
        units = {(e.ingredient.unit or "").lower() for e in entries}
        total_text: Optional[str] = None
        if len(units) == 1:
            unit = entries[0].ingredient.unit
            total_text = fmt_ingredient("", sum(e.amount for e in entries), unit).strip()
        done = [sid for sid in step_ids if sid in session.completed_steps]
        status = "done" if len(done) == len(step_ids) else "partial" if done else "pending"
        gid = "prep_" + re.sub(r"[^a-z0-9]+", "_", key).strip("_")
        # Label with the shortest original name ("garlic" rather than "garlic cloves, smashed").
        label = min((e.ingredient.name.split(",")[0].strip() for e in entries), key=len).capitalize()
        groups.append(PrepGroup(id=gid, key=key, label=label, entries=entries,
                                total_text=total_text, step_ids=step_ids, status=status))
    groups.sort(key=lambda g: (g.status == "done", g.label))
    return groups


def find_prep_group(session: Session, ref: str) -> Optional[PrepGroup]:
    ref = ref.strip().lower()
    groups = prep_groups(session)
    for g in groups:
        if g.id == ref or g.key == ref:
            return g
    for g in groups:
        if ref and (ref in g.key or ingredient_key(ref) == g.key):
            return g
    return None


# --------------------------------------------------------------------------- merged plan

@dataclass
class PlanItem:
    kind: str  # "prep" | "step"
    id: str  # prep group id or step id
    text: str
    status: str  # done | skipped | active | now | pending | unscheduled
    recipe_id: Optional[str] = None
    recipe_title: Optional[str] = None
    step_n: Optional[int] = None
    at: Optional[datetime] = None
    duration_s: Optional[int] = None
    appliance: Optional[str] = None
    temp_f: Optional[int] = None
    task_id: Optional[str] = None
    task_label: Optional[str] = None
    note: Optional[str] = None
    # Scaled amounts for the ingredients this step uses, so the assistant can say "two
    # tablespoons of olive oil" without the cook having to ask how much, every time.
    ingredients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "id": self.id, "text": self.text, "status": self.status,
            "recipe_id": self.recipe_id, "recipe_title": self.recipe_title, "step_n": self.step_n,
            "at": self.at.isoformat() if self.at else None, "duration_s": self.duration_s,
            "appliance": self.appliance, "temp_f": self.temp_f, "task_id": self.task_id,
            "task_label": self.task_label, "note": self.note,
            "ingredients": list(self.ingredients),
        }


@dataclass
class Plan:
    items: List[PlanItem] = field(default_factory=list)
    groups: List[PrepGroup] = field(default_factory=list)
    now_id: Optional[str] = None

    @property
    def now(self) -> Optional[PlanItem]:
        return next((i for i in self.items if i.id == self.now_id), None)

    def upcoming(self, n: int = 8) -> List[PlanItem]:
        out = [i for i in self.items if i.status in ("now", "active", "pending", "unscheduled")]
        return out[:n]

    def to_dict(self) -> Dict[str, Any]:
        done = sum(1 for i in self.items if i.status in ("done", "skipped"))
        return {"items": [i.to_dict() for i in self.items], "groups": [g.to_dict() for g in self.groups],
                "now_id": self.now_id, "done_count": done, "total": len(self.items)}


def build_plan(session: Session, now: datetime) -> Plan:
    try:
        scheduler.resolve(session, now)
    except scheduler.ScheduleError:
        pass
    groups = prep_groups(session)
    grouped: Set[str] = {sid for g in groups for sid in g.step_ids}
    items: List[PlanItem] = []
    for g in groups:
        items.append(PlanItem(kind="prep", id=g.id, text=g.summary(),
                              status="done" if g.status == "done" else "pending",
                              note="prep for several recipes"))

    def step_status(r: Recipe, s: Step, task=None) -> str:
        ov = session.overlays[r.id]
        if s.id in session.completed_steps:
            return "done"
        if s.id in ov.skipped_steps:
            return "skipped"
        if task is not None and task.status == "active":
            return "active"
        return "pending"

    def make(r: Recipe, s: Step, at: Optional[datetime], task=None, status: Optional[str] = None) -> PlanItem:
        ov = session.overlays[r.id]
        appliance = s.appliance
        if appliance and task and task.appliance and appliance.split(":")[0] == task.appliance.split(":")[0]:
            appliance = task.appliance  # the step said "stovetop"; the task knows which burner
        return PlanItem(
            kind="step", id=s.id, text=substitute_text(s.text, r, ov, s.id), status=status or step_status(r, s, task),
            recipe_id=r.id, recipe_title=r.title, step_n=r.step_index(s.id) if r.step_index(s.id) > 0 else None,
            at=at, duration_s=s.duration_s, appliance=appliance,
            temp_f=s.temp_f or (task.temp_f if task and appliance else None), task_id=task.id if task else None,
            task_label=task.label if task else None, note=ov.step_notes.get(s.id),
            ingredients=[fmt_ingredient(i.name, i.amount * ov.scale_factor, i.unit)
                         for i in (r.ingredient(iid) for iid in s.ingredient_ids) if i],
        )

    # Steps covered by tasks get the task's clock; each step's time follows the previous one.
    covered: Dict[str, PlanItem] = {}
    tasks = sorted(session.tasks.values(), key=lambda t: (t.start_at or now, t.id))
    for t in tasks:
        cursor = t.start_at
        for sid in t.step_ids:
            found = session.find_step(sid)
            if not found:
                continue
            r, s = found
            item = make(r, s, cursor, t)
            if t.status == "complete" and item.status == "pending":
                item.status = "done"
            covered[sid] = item
            if cursor is not None and s.duration_s:
                cursor = cursor + timedelta(seconds=s.duration_s)

    # Walk each recipe in order. Uncovered steps before the first covered one are prep,
    # scheduled backwards from that task; uncovered steps after a covered one follow it.
    timed: List[PlanItem] = []
    loose: List[PlanItem] = []
    for r in session.recipes.values():
        ov = session.overlays[r.id]
        all_steps = [s for s in list(r.steps) + list(ov.added_steps) if s.id not in grouped]
        first_covered = next((i for i, s in enumerate(all_steps) if s.id in covered), None)
        if first_covered is None:
            for s in all_steps:
                st = step_status(r, s)
                loose.append(make(r, s, None, status=st if st != "pending" else "unscheduled"))
            continue
        pre = all_steps[:first_covered]
        anchor = covered[all_steps[first_covered].id].at or now
        total = sum(s.duration_s or 0 for s in pre if step_status(r, s) == "pending")
        cursor = max(now, anchor - timedelta(seconds=total))
        for s in pre:
            st = step_status(r, s)
            if st == "pending":
                timed.append(make(r, s, cursor))
                cursor = cursor + timedelta(seconds=s.duration_s or 0)
            else:
                timed.append(make(r, s, None, status=st))
        # Runs of steps no task covers are scheduled just in time for the next task that
        # does, so seasoning between a preheat and a sear lands before the sear, not after.
        rest = all_steps[first_covered:]
        cursor: Optional[datetime] = None
        i = 0
        while i < len(rest):
            s = rest[i]
            if s.id in covered:
                item = covered[s.id]
                timed.append(item)
                cursor = (item.at + timedelta(seconds=s.duration_s or 0)) if item.at else None
                i += 1
                continue
            j = i
            while j < len(rest) and rest[j].id not in covered:
                j += 1
            run = rest[i:j]
            next_at = covered[rest[j].id].at if j < len(rest) else None
            total = sum(x.duration_s or 0 for x in run if step_status(r, x) == "pending")
            if next_at is not None:
                start = next_at - timedelta(seconds=total)
                if cursor is not None and cursor > start:
                    start = cursor
                if start < now:
                    start = now
            else:
                start = cursor
            for x in run:
                st = step_status(r, x)
                if st != "pending":
                    timed.append(make(r, x, None, status=st))
                elif start is None:
                    timed.append(make(r, x, None, status="unscheduled"))
                else:
                    timed.append(make(r, x, start))
                    start = start + timedelta(seconds=x.duration_s or 0)
            cursor = start if start is not None else cursor
            i = j

    timed.sort(key=lambda i: (i.status in ("done", "skipped"), i.at or now, i.recipe_id or "", i.step_n or 0))
    items.extend(timed)
    items.extend(loose)

    plan = Plan(items=items, groups=groups)
    first = next((i for i in items if i.status in ("active", "pending", "unscheduled")), None)
    if first is not None:
        first.status = "now"
        plan.now_id = first.id
    return plan


# --------------------------------------------------------------------------- summary

def merged_ingredients(session: Session) -> List[Dict[str, Any]]:
    """Every ingredient across loaded recipes, merged by normalized name with totals where
    the units agree. Overlay scaling and substitutions applied."""
    merged: Dict[str, Dict[str, Any]] = {}
    for r in session.recipes.values():
        ov = session.overlays[r.id]
        subs = {s.ingredient_id: s for s in ov.substitutions}
        for ing in r.ingredients:
            name = subs[ing.id].replacement if ing.id in subs else ing.name
            key = ingredient_key(name) or name.lower()
            amount = ing.amount * ov.scale_factor
            unit = (ing.unit or "").lower() or None
            entry = merged.setdefault(key, {"key": key, "name": name.split(",")[0].strip(), "parts": [], "recipes": []})
            entry["parts"].append((amount, unit))
            entry["recipes"].append(r.title)
            if len(entry["name"]) > len(name.split(",")[0].strip()):
                entry["name"] = name.split(",")[0].strip()
    out = []
    for e in merged.values():
        units = {u for _, u in e["parts"]}
        if len(units) == 1:
            unit = e["parts"][0][1]
            total = sum(a for a, _ in e["parts"])
            text = fmt_ingredient(e["name"], total, unit)
        else:
            text = e["name"] + ": " + " + ".join(fmt_ingredient("", a, u).strip() for a, u in e["parts"])
        out.append({"key": e["key"], "name": e["name"], "text": text, "recipes": sorted(set(e["recipes"])),
                    "shared": len(set(e["recipes"])) > 1})
    out.sort(key=lambda x: (not x["shared"], x["name"]))
    return out


def plan_summary(session: Session, now: datetime) -> Dict[str, Any]:
    plan = build_plan(session, now)
    tasks = [t for t in session.tasks.values() if t.is_open or t.status == "complete"]
    times = [i.at for i in plan.items if i.at]
    ends = [t.end_at for t in tasks if t.end_at]
    start = min(times) if times else None
    end = session.target_plating or (max(ends) if ends else None)
    appliances: Dict[str, Set[str]] = {}
    for t in tasks:
        if not t.appliance:
            continue
        fam, _, idx = t.appliance.partition(":")
        detail = f"{t.temp_f}°F" if t.temp_f else (f"burner {idx}" if idx else "")
        appliances.setdefault(fam, set())
        if detail:
            appliances[fam].add(detail)
    appliance_text = []
    for fam, details in sorted(appliances.items()):
        label = fam.replace("_", " ")
        if fam == "stovetop":
            n = len([d for d in details if d.startswith("burner")])
            label = f"stovetop x{n} ({', '.join(sorted(details))})" if n else "stovetop"
        elif details:
            label = f"{label} {', '.join(sorted(details))}"
        appliance_text.append(label)
    ingredients = merged_ingredients(session)
    steps_total = sum(1 for i in plan.items if i.kind == "step")
    return {
        "recipes": [{"id": r.id, "title": r.title, "steps": len(r.steps) + len(session.overlays[r.id].added_steps)}
                    for r in session.recipes.values()],
        "start_at": start.isoformat() if start else None,
        "plating_at": end.isoformat() if end else None,
        "span_s": int((end - start).total_seconds()) if start and end and end > start else None,
        "steps": steps_total,
        "appliances": appliance_text,
        "prep_groups": [g.summary() for g in plan.groups],
        "ingredients": ingredients,
        "planned": bool(tasks),
    }


def render_plan_summary(session: Session, now: datetime, max_ingredients: int = 24) -> str:
    s = plan_summary(session, now)
    if not s["recipes"]:
        return "PLAN SUMMARY\nno recipes loaded"
    lines = ["PLAN SUMMARY"]
    lines.append("Recipes: " + ", ".join(f"{r['title']} ({r['steps']} steps)" for r in s["recipes"]))
    if s["planned"]:
        start = datetime.fromisoformat(s["start_at"]) if s["start_at"] else None
        end = datetime.fromisoformat(s["plating_at"]) if s["plating_at"] else None
        span = f", {fmt_dur(s['span_s'])} from first step to plating" if s["span_s"] else ""
        lines.append(f"Timing: start {fmt_time(start)}, plating {fmt_time(end)}{span}; {s['steps']} steps in all")
        lines.append("Appliances: " + (", ".join(s["appliances"]) or "none"))
    else:
        lines.append("Timing: not planned yet (no tasks)")
    if s["prep_groups"]:
        lines.append("Prep first: " + "; ".join(s["prep_groups"]))
    shown = s["ingredients"][:max_ingredients]
    more = len(s["ingredients"]) - len(shown)
    lines.append("Ingredients to get out: " + "; ".join(i["text"] + (" (shared)" if i["shared"] else "") for i in shown)
                 + (f"; and {more} more" if more > 0 else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- rendering

def render_cook_plan(session: Session, now: datetime, upcoming: int = 10) -> str:
    plan = build_plan(session, now)
    lines = ["COOK PLAN (every recipe merged, in the order to do things)"]
    if not plan.items:
        lines.append("nothing to do yet: load recipes and add tasks")
        return "\n".join(lines)
    if plan.groups:
        lines.append("Prep first (shared across recipes; complete_prep marks all their steps):")
        for g in plan.groups:
            tag = "NOW " if plan.now_id == g.id else "    "
            lines.append(f"{tag}[{g.id}] {g.summary()}  ({g.status})")
    if session.completed_steps:
        lines.append(f"Done: {len(session.completed_steps)} step(s) so far")
    shown = 0
    steps_left = [i for i in plan.items if i.kind == "step" and i.status not in ("done", "skipped")]
    for item in steps_left:
        if shown >= upcoming:
            lines.append(f"  ... {len(steps_left) - shown} more")
            break
        tag = "NOW  " if item.status == "now" else "next " if (shown == 0 or (shown == 1 and steps_left[0].status == "now")) else "     "
        when = fmt_time(item.at) if item.at else "--:--"
        bits = []
        if item.duration_s:
            bits.append(fmt_dur(item.duration_s))
        if item.appliance:
            bits.append(f"{item.appliance} {item.temp_f}°F" if item.temp_f else item.appliance)
        if item.status == "unscheduled":
            bits.append("not scheduled")
        meta = f" ({', '.join(bits)})" if bits else ""
        who = f"{item.recipe_title} step {item.step_n}: " if item.recipe_title else ""
        uses = f" [uses: {', '.join(item.ingredients)}]" if item.ingredients else ""
        lines.append(f"{tag}{when:>8}  {who}{item.text}{meta}{uses}  [{item.id}]")
        shown += 1
    return "\n".join(lines)
