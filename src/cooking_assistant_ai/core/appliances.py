"""What is on what, right now.

The scheduler already knows which task owns which burner and when. What it never did was
answer the question a cook actually asks while standing in the kitchen: is that ring free,
and what is the oven doing? This turns the task list inside out to answer it per appliance
rather than per dish.

Derived entirely from tasks, like everything else here, so it cannot drift.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.fmt import fmt_dur, fmt_time
from cooking_assistant_ai.model.types import Session, Task

# Shown even when nothing is using them, so the cook can see what is free at a glance.
# Anything else a recipe calls for is added when a task needs it.
ALWAYS_SHOWN = ("oven", "stovetop", "air_fryer")

LABELS = {
    "oven": "Oven",
    "air_fryer": "Air fryer",
    "rice_cooker": "Rice cooker",
    "pressure_cooker": "Pressure cooker",
    "bread_maker": "Bread maker",
    "grill": "Grill",
    "microwave": "Microwave",
}


def _label(slot: str) -> str:
    if slot.startswith("stovetop:"):
        return f"Burner {slot.split(':', 1)[1]}"
    return LABELS.get(slot, slot.replace("_", " ").capitalize())


def _slots(session: Session) -> List[str]:
    """Every appliance worth drawing: the standard set, plus whatever the tasks need."""
    slots: List[str] = ["oven"]
    slots += [f"stovetop:{n}" for n in range(1, scheduler.BURNERS + 1)]
    slots.append("air_fryer")
    for t in session.tasks.values():
        if not t.appliance:
            continue
        family = t.appliance.split(":", 1)[0]
        if family in ALWAYS_SHOWN:
            continue
        if t.appliance not in slots:
            slots.append(t.appliance)
    return slots


def _owns(task: Task, slot: str) -> bool:
    if not task.appliance:
        return False
    if task.appliance == slot:
        return True
    # A task still waiting for a burner assignment shows on the hob generally, not on a ring.
    return task.appliance == "stovetop" and slot.startswith("stovetop:") and task.appliance_auto


def _task_view(t: Task, now: datetime) -> Dict[str, Any]:
    remaining = None
    if t.status == "active" and t.end_at and not t.awaits_cook:
        remaining = max(0, int((t.end_at - now).total_seconds()))
    return {
        "id": t.id, "label": t.label, "recipe_id": t.recipe_id, "status": t.status,
        "temp_f": t.temp_f, "awaits_cook": t.awaits_cook,
        "start_at": t.start_at.isoformat() if t.start_at else None,
        "end_at": t.end_at.isoformat() if t.end_at else None,
        "starts_in_s": int((t.start_at - now).total_seconds()) if t.start_at else None,
        "remaining_s": remaining,
    }


def board(session: Session, now: datetime) -> List[Dict[str, Any]]:
    """One entry per appliance: what is on it now, and what is queued for it next."""
    out: List[Dict[str, Any]] = []
    for slot in _slots(session):
        mine = [t for t in session.tasks.values() if _owns(t, slot) and t.is_open]
        active = [t for t in mine if t.status == "active"]
        # Scheduled but not started; the soonest is the one worth showing.
        upcoming = sorted((t for t in mine if t.status == "pending" and t.start_at),
                          key=lambda t: t.start_at)  # type: ignore[arg-type]

        if active:
            status = "active"
        elif upcoming and upcoming[0].start_at and upcoming[0].start_at <= now:
            status = "due"          # should already be going
        elif upcoming:
            status = "reserved"
        else:
            status = "free"

        current = active[0] if active else None
        nxt = upcoming[0] if upcoming and not active else None
        temp = None
        if current is not None:
            temp = current.temp_f
        elif nxt is not None:
            temp = nxt.temp_f

        out.append({
            "slot": slot,
            "label": _label(slot),
            "family": slot.split(":", 1)[0],
            "status": status,
            "temp_f": temp,
            "untimed": bool(current and current.awaits_cook),
            "current": _task_view(current, now) if current else None,
            "next": _task_view(nxt, now) if nxt else None,
            "detail": _detail(current, nxt, now),
        })
    return out


def _detail(current: Optional[Task], nxt: Optional[Task], now: datetime) -> str:
    if current is not None:
        if current.awaits_cook:
            return f"{current.label} - tell me when it's done"
        if current.end_at:
            left = int((current.end_at - now).total_seconds())
            return f"{current.label} - {fmt_dur(max(0, left))} left" if left > 0 else f"{current.label} - due now"
        return current.label
    if nxt is not None:
        if nxt.start_at and nxt.start_at <= now:
            return f"{nxt.label} - start now"
        if nxt.start_at:
            return f"{nxt.label} at {fmt_time(nxt.start_at)}"
        return nxt.label
    return "free"


def render_board(session: Session, now: datetime) -> str:
    """Text form, for the model's context: it should know the hob is full before promising."""
    rows = board(session, now)
    busy = [r for r in rows if r["status"] != "free"]
    if not busy:
        return "APPLIANCES: all free"
    lines = ["APPLIANCES"]
    for r in rows:
        if r["status"] == "free":
            continue
        temp = f" at {r['temp_f']}°F" if r["temp_f"] else ""
        lines.append(f"  {r['label']}{temp}: {r['detail']}")
    free = [r["label"] for r in rows if r["status"] == "free"]
    if free:
        lines.append("  free: " + ", ".join(free))
    return "\n".join(lines)
