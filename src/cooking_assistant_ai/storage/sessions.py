"""Making a cooking session survive a crash, a restart or a power cut.

A session is the only state that is not already in SQLite: the plan, the timers, what has
been done and what was said. Losing it mid-cook is the worst failure this system has, so it
is snapshotted to the database every few seconds and restored on startup.

The snapshot is self-contained — recipes are stored in full rather than by id — so a session
restores intact even if the recipe was edited or deleted while the server was down.
"""
from __future__ import annotations

import itertools
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from cooking_assistant_ai.model.types import (
    Overlay,
    Recipe,
    Session,
    Step,
    Substitution,
    Task,
    Timer,
    Turn,
)

# Bumped when a change makes old snapshots unreadable; older ones are then dropped rather
# than half-restored into something the scheduler cannot reason about.
SCHEMA_VERSION = 1


def _dt(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _step_dict(s: Step) -> Dict[str, Any]:
    return {"id": s.id, "text": s.text, "duration_s": s.duration_s, "appliance": s.appliance,
            "temp_f": s.temp_f, "ingredient_ids": list(s.ingredient_ids)}


def _step(d: Dict[str, Any]) -> Step:
    return Step(id=d["id"], text=d["text"], duration_s=d.get("duration_s"),
                appliance=d.get("appliance"), temp_f=d.get("temp_f"),
                ingredient_ids=tuple(d.get("ingredient_ids") or ()))


def _overlay_dict(o: Overlay) -> Dict[str, Any]:
    return {
        "recipe_id": o.recipe_id,
        "scale_factor": o.scale_factor,
        "substitutions": [{"ingredient_id": s.ingredient_id, "replacement": s.replacement,
                           "note": s.note, "at_step": s.at_step, "original": s.original}
                          for s in o.substitutions],
        "step_notes": dict(o.step_notes),
        "skipped_steps": sorted(o.skipped_steps),
        "skip_reasons": dict(o.skip_reasons),
        "added_steps": [_step_dict(s) for s in o.added_steps],
    }


def _overlay(d: Dict[str, Any]) -> Overlay:
    return Overlay(
        recipe_id=d["recipe_id"],
        scale_factor=d.get("scale_factor", 1.0),
        substitutions=[Substitution(ingredient_id=s["ingredient_id"], replacement=s["replacement"],
                                    note=s.get("note"), at_step=s.get("at_step"),
                                    original=s.get("original", ""))
                       for s in d.get("substitutions", [])],
        step_notes=dict(d.get("step_notes") or {}),
        skipped_steps=set(d.get("skipped_steps") or ()),
        skip_reasons=dict(d.get("skip_reasons") or {}),
        added_steps=[_step(s) for s in d.get("added_steps", [])],
    )


def _task_dict(t: Task) -> Dict[str, Any]:
    return {
        "id": t.id, "label": t.label, "recipe_id": t.recipe_id, "step_ids": list(t.step_ids),
        "appliance": t.appliance, "temp_f": t.temp_f, "duration_s": t.duration_s,
        "appliance_auto": t.appliance_auto, "start_at": _dt(t.start_at), "end_at": _dt(t.end_at),
        "status": t.status, "depends_on": list(t.depends_on), "must_finish_by": t.must_finish_by,
        "not_before": _dt(t.not_before), "actual_start": _dt(t.actual_start),
        "actual_end": _dt(t.actual_end), "awaits_cook": t.awaits_cook,
    }


def _task(d: Dict[str, Any]) -> Task:
    return Task(
        id=d["id"], label=d["label"], recipe_id=d["recipe_id"], step_ids=list(d["step_ids"]),
        appliance=d.get("appliance"), temp_f=d.get("temp_f"), duration_s=d.get("duration_s", 0),
        appliance_auto=d.get("appliance_auto", False), start_at=_parse_dt(d.get("start_at")),
        end_at=_parse_dt(d.get("end_at")), status=d.get("status", "pending"),
        depends_on=list(d.get("depends_on") or ()), must_finish_by=d.get("must_finish_by"),
        not_before=_parse_dt(d.get("not_before")), actual_start=_parse_dt(d.get("actual_start")),
        actual_end=_parse_dt(d.get("actual_end")), awaits_cook=d.get("awaits_cook", False),
    )


def _timer_dict(t: Timer) -> Dict[str, Any]:
    return {"id": t.id, "label": t.label, "task_id": t.task_id, "step_id": t.step_id,
            "end_at": _dt(t.end_at), "on_complete_hint": t.on_complete_hint,
            "status": t.status, "created_at": _dt(t.created_at)}


def _timer(d: Dict[str, Any]) -> Timer:
    return Timer(id=d["id"], label=d["label"], task_id=d.get("task_id"), step_id=d.get("step_id"),
                 end_at=_parse_dt(d["end_at"]), on_complete_hint=d.get("on_complete_hint"),
                 status=d.get("status", "running"), created_at=_parse_dt(d.get("created_at")))


def to_dict(session: Session) -> Dict[str, Any]:
    return {
        "v": SCHEMA_VERSION,
        "id": session.id,
        "started_at": _dt(session.started_at),
        "recipes": [r.to_dict() for r in session.recipes.values()],
        "overlays": [_overlay_dict(o) for o in session.overlays.values()],
        "tasks": [_task_dict(t) for t in session.tasks.values()],
        "timers": [_timer_dict(t) for t in session.timers.values()],
        "completed_steps": sorted(session.completed_steps),
        "notes": list(session.notes),
        "target_plating": _dt(session.target_plating),
        # The transcript is for continuity of conversation, not state; the tail is plenty and
        # keeps the snapshot small enough to write every few seconds.
        "transcript": [{"role": t.role, "text": t.text, "at": _dt(t.at)}
                       for t in session.transcript[-40:]],
        "proactivity": session.proactivity,
        "last_turn_at": _dt(session.last_turn_at),
    }


def _rebuild_counters(session: Session) -> None:
    """Restart id generation past the highest id already used.

    Counters are iterators and cannot be serialized. Deriving them from the ids that exist is
    better than storing them anyway: it is self-correcting, and a fresh counter would hand out
    t_001 again and quietly overwrite the first task of the cook.
    """
    highest: Dict[str, int] = {}
    for ident in list(session.tasks) + list(session.timers):
        m = re.fullmatch(r"([a-z]+)_(\d+)", ident)
        if m:
            prefix, n = m.group(1), int(m.group(2))
            highest[prefix] = max(highest.get(prefix, 0), n)
    for prefix, n in highest.items():
        session._counters[prefix] = itertools.count(n + 1)


def from_dict(d: Dict[str, Any]) -> Session:
    session = Session(id=d["id"], started_at=_parse_dt(d["started_at"]))
    for raw in d.get("recipes", []):
        session.add_recipe(Recipe.from_dict(raw))
    for raw in d.get("overlays", []):
        session.overlays[raw["recipe_id"]] = _overlay(raw)
    for raw in d.get("tasks", []):
        t = _task(raw)
        session.tasks[t.id] = t
    for raw in d.get("timers", []):
        t = _timer(raw)
        session.timers[t.id] = t
    session.completed_steps = set(d.get("completed_steps") or ())
    session.notes = list(d.get("notes") or ())
    session.target_plating = _parse_dt(d.get("target_plating"))
    session.transcript = [Turn(role=t["role"], text=t["text"], at=_parse_dt(t["at"]))
                          for t in d.get("transcript", [])]
    session.proactivity = d.get("proactivity", 0.5)
    session.last_turn_at = _parse_dt(d.get("last_turn_at"))
    _rebuild_counters(session)
    return session


def expire_timers(session: Session, now: datetime) -> List[Timer]:
    """Retire timers that ran out while the server was down.

    Left alone they would all fire at once the moment the session is restored, so the cook
    would get a burst of alarms for things that finished twenty minutes ago. Marking them
    fired and saying so is more use than pretending they are still counting.
    """
    late = [t for t in session.timers.values() if t.status == "running" and t.end_at <= now]
    for t in late:
        t.status = "fired"
    return late


def is_worth_saving(session: Session) -> bool:
    """An empty session (someone opened the page and chose nothing) is not worth restoring."""
    return bool(session.recipes or session.tasks or session.timers or session.completed_steps)
