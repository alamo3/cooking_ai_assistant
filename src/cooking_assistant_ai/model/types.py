"""State models. See spec section 1.

Recipes are immutable. Everything mutable lives in Overlay / Task / Timer / Session,
and is only ever mutated through tool calls (spec invariant 1).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# --------------------------------------------------------------------------- recipe

@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    amount: float
    unit: Optional[str] = None  # None for countable items


@dataclass(frozen=True)
class Step:
    id: str
    text: str
    duration_s: Optional[int] = None
    appliance: Optional[str] = None  # "oven" | "stovetop:1" | "air_fryer" | None
    temp_f: Optional[int] = None
    ingredient_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    servings: int
    ingredients: Tuple[Ingredient, ...]
    steps: Tuple[Step, ...]

    def step(self, step_id: str) -> Optional[Step]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def ingredient(self, ingredient_id: str) -> Optional[Ingredient]:
        for i in self.ingredients:
            if i.id == ingredient_id:
                return i
        return None

    def step_index(self, step_id: str) -> int:
        """1-based index of a step, or -1."""
        for n, s in enumerate(self.steps, start=1):
            if s.id == step_id:
                return n
        return -1

    # -- serialization (storage + HTTP) -------------------------------------

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Recipe":
        rid = str(d["id"])
        ingredients = []
        for n, raw in enumerate(d.get("ingredients", []), start=1):
            ingredients.append(Ingredient(
                id=str(raw.get("id") or f"{rid}-i{n}"),
                name=str(raw["name"]),
                amount=float(raw.get("amount", 1)),
                unit=raw.get("unit"),
            ))
        steps = []
        for n, raw in enumerate(d.get("steps", []), start=1):
            steps.append(Step(
                id=str(raw.get("id") or f"{rid}-s{n}"),
                text=str(raw["text"]),
                duration_s=int(raw["duration_s"]) if raw.get("duration_s") is not None else None,
                appliance=raw.get("appliance"),
                temp_f=int(raw["temp_f"]) if raw.get("temp_f") is not None else None,
                ingredient_ids=tuple(str(x) for x in raw.get("ingredient_ids", [])),
            ))
        return Recipe(
            id=rid,
            title=str(d["title"]),
            servings=int(d.get("servings", 2)),
            ingredients=tuple(ingredients),
            steps=tuple(steps),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "servings": self.servings,
            "ingredients": [
                {"id": i.id, "name": i.name, "amount": i.amount, "unit": i.unit}
                for i in self.ingredients
            ],
            "steps": [
                {
                    "id": s.id, "text": s.text, "duration_s": s.duration_s,
                    "appliance": s.appliance, "temp_f": s.temp_f,
                    "ingredient_ids": list(s.ingredient_ids),
                }
                for s in self.steps
            ],
        }


# --------------------------------------------------------------------------- overlay

@dataclass
class Substitution:
    ingredient_id: str
    replacement: str
    note: Optional[str] = None
    at_step: Optional[str] = None


@dataclass
class Overlay:
    recipe_id: str
    scale_factor: float = 1.0
    substitutions: List[Substitution] = field(default_factory=list)
    step_notes: Dict[str, str] = field(default_factory=dict)  # step_id -> note
    skipped_steps: Set[str] = field(default_factory=set)
    skip_reasons: Dict[str, str] = field(default_factory=dict)  # step_id -> reason
    added_steps: List[Step] = field(default_factory=list)

    def is_clean(self) -> bool:
        return (
            self.scale_factor == 1.0
            and not self.substitutions
            and not self.step_notes
            and not self.skipped_steps
            and not self.added_steps
        )


# --------------------------------------------------------------------------- task

@dataclass
class Task:
    id: str
    label: str
    recipe_id: str
    step_ids: List[str]
    appliance: Optional[str]
    temp_f: Optional[int]
    duration_s: int
    appliance_auto: bool = False  # "stovetop" without a burner: the resolver picks a free one
    start_at: Optional[datetime] = None  # resolved by code, never by the model
    end_at: Optional[datetime] = None
    status: str = "pending"  # pending | active | complete | skipped
    depends_on: List[str] = field(default_factory=list)
    # Scheduling intent (resolver inputs). Also code-owned.
    must_finish_by: Optional[str] = None  # "plating" | task id | None
    not_before: Optional[datetime] = None  # set by move_task(delay_minutes)
    actual_start: Optional[datetime] = None  # set by start_task
    actual_end: Optional[datetime] = None  # set by mark_complete

    @property
    def is_open(self) -> bool:
        return self.status in ("pending", "active")


# --------------------------------------------------------------------------- timer

@dataclass
class Timer:
    id: str
    label: str
    task_id: Optional[str]
    step_id: Optional[str]
    end_at: datetime
    on_complete_hint: Optional[str] = None
    status: str = "running"  # running | fired | cancelled
    created_at: Optional[datetime] = None


# --------------------------------------------------------------------------- session

@dataclass
class Turn:
    role: str  # "user" | "assistant" | "system"
    text: str
    at: datetime


@dataclass
class Session:
    id: str
    started_at: datetime
    recipes: Dict[str, Recipe] = field(default_factory=dict)
    overlays: Dict[str, Overlay] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    timers: Dict[str, Timer] = field(default_factory=dict)
    completed_steps: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    target_plating: Optional[datetime] = None
    transcript: List[Turn] = field(default_factory=list)
    proactivity: float = 0.5
    last_turn_at: Optional[datetime] = None
    _counters: Dict[str, Iterator[int]] = field(default_factory=dict, repr=False)

    # -- ids ----------------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        c = self._counters.setdefault(prefix, itertools.count(1))
        return f"{prefix}_{next(c):03d}"

    # -- recipes ------------------------------------------------------------
    def add_recipe(self, recipe: Recipe) -> None:
        self.recipes[recipe.id] = recipe
        self.overlays.setdefault(recipe.id, Overlay(recipe_id=recipe.id))

    def find_step(self, step_id: str) -> Optional[Tuple[Recipe, Step]]:
        for r in self.recipes.values():
            s = r.step(step_id)
            if s is not None:
                return r, s
            for extra in self.overlays[r.id].added_steps:
                if extra.id == step_id:
                    return r, extra
        return None

    def find_recipe(self, ref: str) -> Optional[Recipe]:
        """Look up by id, then exact title, then title substring (case-insensitive)."""
        if ref in self.recipes:
            return self.recipes[ref]
        low = ref.strip().lower()
        for r in self.recipes.values():
            if r.title.lower() == low:
                return r
        for r in self.recipes.values():
            if low and low in r.title.lower():
                return r
        return None

    # -- tasks --------------------------------------------------------------
    def find_task(self, ref: str) -> Optional[Task]:
        """Look up by id, then exact label, then label substring (case-insensitive)."""
        if ref in self.tasks:
            return self.tasks[ref]
        low = ref.strip().lower()
        for t in self.tasks.values():
            if t.label.lower() == low:
                return t
        for t in self.tasks.values():
            if low and low in t.label.lower():
                return t
        return None

    def open_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_open]

    # -- timers -------------------------------------------------------------
    def running_timers(self) -> List[Timer]:
        return [t for t in self.timers.values() if t.status == "running"]
