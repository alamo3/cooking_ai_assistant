"""Deterministic scheduler. Turns task *intent* (after / before / must_finish_by /
delay) into wall-clock windows, and detects conflicts.

Semantics (documented to the model in the tool descriptions):

* Default is ASAP: a task starts now, or as soon as everything it depends on ends.
* ``must_finish_by="plating"`` makes the task ALAP: it is scheduled to end exactly at
  target plating (but never before its dependencies allow).
* ``must_finish_by=<task>`` makes it ALAP against that task's start.
* ``after=<task>`` adds a dependency edge; ``before=<task>`` adds the reverse edge.
* A chain of ``after`` dependencies feeding an ALAP task is itself scheduled just-in-time.
* Active tasks are pinned to their actual start; dependents shift automatically.

Resolution is recomputed from scratch on every call, so state can never drift.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from cooking_assistant_ai.core.fmt import fmt_time, fmt_window
from cooking_assistant_ai.model.types import Session, Task

SHARED_APPLIANCES = ("oven",)  # can hold several tasks at once if the temperature matches
BURNERS = int(os.environ.get("COOK_BURNERS", "4"))  # stovetop burners available for auto-assignment


class ScheduleError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalize_appliance(appliance: Optional[str]) -> Optional[str]:
    if appliance is None:
        return None
    a = str(appliance).strip().lower().replace(" ", "_").replace("-", "_")
    if not a or a in ("none", "null"):
        return None
    if a in ("stove", "hob", "burner", "range", "cooktop", "stovetop:any", "stovetop:*", "stove:any"):
        a = "stovetop"
    if a.startswith("stove:") or a.startswith("burner:") or a.startswith("hob:"):
        a = "stovetop:" + a.split(":", 1)[1]
    if a in ("airfryer", "air_fryer", "air-fryer"):
        a = "air_fryer"
    return a


def is_generic_stovetop(appliance: Optional[str]) -> bool:
    return appliance == "stovetop"


def assign_burners(session: Session, burners: Optional[int] = None) -> None:
    """Give every auto-stovetop task a concrete free burner for its window.

    Tasks that named a burner, and auto tasks already active (the pan is physically on
    it), are fixed. Pending auto tasks get the lowest burner free for their whole window,
    earliest start first. If none is free the task keeps the generic "stovetop" and the
    shortage is reported as a conflict.
    """
    burners = burners or BURNERS
    fixed: List[Tuple[int, Task]] = []
    auto: List[Task] = []
    for t in session.tasks.values():
        if not t.is_open or not t.appliance or _appliance_family(t.appliance) != "stovetop":
            continue
        if t.appliance_auto and t.status == "pending":
            auto.append(t)
        elif ":" in t.appliance:
            try:
                fixed.append((int(t.appliance.split(":", 1)[1]), t))
            except ValueError:
                pass
    auto.sort(key=lambda t: (t.start_at or datetime.max, t.id))
    for t in auto:
        chosen: Optional[int] = None
        for n in range(1, burners + 1):
            if all(not (m == n and _overlaps(t, other)) for m, other in fixed):
                chosen = n
                break
        if chosen is None:
            t.appliance = "stovetop"
        else:
            t.appliance = f"stovetop:{chosen}"
            fixed.append((chosen, t))


def burner_shortages(session: Session, burners: Optional[int] = None) -> List[str]:
    burners = burners or BURNERS
    out = []
    for t in session.tasks.values():
        if t.is_open and is_generic_stovetop(t.appliance):
            out.append(f"no free burner for {t.label} ({fmt_window(t.start_at, t.end_at)}); all {burners} burners are busy then. "
                       f"Use after='<task>' or move_task to shift it.")
    return out


def _appliance_family(appliance: str) -> str:
    return appliance.split(":", 1)[0]


def topo_order(tasks: Dict[str, Task]) -> List[Task]:
    """Kahn's algorithm over depends_on. Raises on unknown deps or cycles."""
    indeg: Dict[str, int] = {tid: 0 for tid in tasks}
    children: Dict[str, List[str]] = {tid: [] for tid in tasks}
    for t in tasks.values():
        for dep in t.depends_on:
            if dep not in tasks:
                raise ScheduleError(f"task {t.label} depends on unknown task '{dep}'")
            indeg[t.id] += 1
            children[dep].append(t.id)
    ready = [tid for tid, d in indeg.items() if d == 0]
    out: List[Task] = []
    while ready:
        ready.sort()  # deterministic
        tid = ready.pop(0)
        out.append(tasks[tid])
        for c in children[tid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(out) != len(tasks):
        stuck = [tasks[tid].label for tid, d in indeg.items() if d > 0]
        raise ScheduleError("circular ordering between tasks: " + ", ".join(stuck))
    return out


def _deadline_for(t: Task, session: Session, order: List[Task], alap: Dict[str, datetime]) -> Optional[datetime]:
    deadline: Optional[datetime] = None
    if t.must_finish_by == "plating":
        deadline = session.target_plating
    elif t.must_finish_by:
        ref = session.tasks.get(t.must_finish_by)
        if ref is not None and ref.is_open and ref.start_at is not None:
            deadline = ref.start_at
    # Dependents that are themselves pinned (active) or ALAP-scheduled constrain us.
    for d in order:
        if t.id not in d.depends_on or not d.is_open:
            continue
        pin: Optional[datetime] = None
        if d.status == "active":
            pin = d.start_at
        elif d.id in alap:
            pin = alap[d.id]
        if pin is not None:
            deadline = pin if deadline is None else min(deadline, pin)
    return deadline


def _feeds_alap(t: Task, order: List[Task], alap: Dict[str, datetime]) -> bool:
    """True if some pending dependent of t is itself ALAP-scheduled."""
    return any(t.id in d.depends_on and d.status == "pending" and d.id in alap for d in order)


def resolve(session: Session, now: datetime) -> None:
    """Recompute start_at/end_at for every task from intent. Mutates tasks in place."""
    order = topo_order(session.tasks)
    alap: Dict[str, datetime] = {}

    for _ in range(6):
        # forward pass: earliest feasible start, lifted to any ALAP target
        for t in order:
            dur = timedelta(seconds=t.duration_s)
            if t.status in ("complete", "skipped"):
                if t.actual_start is not None:
                    t.start_at = t.actual_start
                if t.actual_end is not None:
                    t.end_at = t.actual_end
                elif t.start_at is not None:
                    t.end_at = t.start_at + dur
                continue
            if t.status == "active":
                t.start_at = t.actual_start or t.start_at or now
                t.end_at = t.start_at + dur
                continue
            earliest = now
            if t.not_before is not None and t.not_before > earliest:
                earliest = t.not_before
            for dep_id in t.depends_on:
                dep = session.tasks[dep_id]
                if dep.status == "skipped":
                    continue
                if dep.end_at is not None and dep.end_at > earliest:
                    earliest = dep.end_at
            start = earliest
            if t.id in alap and alap[t.id] > start:
                start = alap[t.id]
            t.start_at = start
            t.end_at = start + dur

        # backward pass: compute ALAP targets for tasks that carry a deadline, and for
        # the dependency chain feeding an ALAP task (prep -> sear -> roast -> rest is
        # just-in-time all the way back, which is what "ready together" means).
        changed = False
        for t in reversed(order):
            if t.status != "pending":
                continue
            if not t.must_finish_by and not _feeds_alap(t, order, alap):
                if t.id in alap:
                    del alap[t.id]
                    changed = True
                continue
            deadline = _deadline_for(t, session, order, alap)
            if deadline is None:
                if t.id in alap:
                    del alap[t.id]
                    changed = True
                continue
            target = deadline - timedelta(seconds=t.duration_s)
            if alap.get(t.id) != target:
                alap[t.id] = target
                changed = True
        if not changed:
            break
    assign_burners(session)


def _overlaps(a: Task, b: Task) -> bool:
    if None in (a.start_at, a.end_at, b.start_at, b.end_at):
        return False
    return a.start_at < b.end_at and b.start_at < a.end_at  # type: ignore[operator]


def appliance_conflicts(session: Session) -> List[Tuple[Task, Task, str]]:
    """Pairs of open tasks fighting over an appliance, with a specific reason."""
    out: List[Tuple[Task, Task, str]] = []
    open_tasks = [t for t in session.tasks.values() if t.is_open and t.appliance and not is_generic_stovetop(t.appliance)]
    for i, a in enumerate(open_tasks):
        for b in open_tasks[i + 1:]:
            if a.appliance != b.appliance or not _overlaps(a, b):
                continue
            fam = _appliance_family(a.appliance or "")
            if fam in SHARED_APPLIANCES:
                if a.temp_f == b.temp_f:
                    continue
                reason = (
                    f"{a.appliance} is at {a.temp_f}°F for {a.label} until {fmt_time(a.end_at)}; "
                    f"{b.label} needs {b.temp_f}°F ({fmt_window(b.start_at, b.end_at)})"
                )
            else:
                reason = (
                    f"{a.appliance} is busy with {a.label} ({fmt_window(a.start_at, a.end_at)}); "
                    f"{b.label} would overlap ({fmt_window(b.start_at, b.end_at)}). "
                    f"Use after='{a.label}' or a different appliance."
                )
            out.append((a, b, reason))
    return out


def deadline_misses(session: Session) -> List[Tuple[Task, str]]:
    out: List[Tuple[Task, str]] = []
    for t in session.tasks.values():
        if not t.is_open or t.end_at is None:
            continue
        if t.must_finish_by == "plating" and session.target_plating and t.end_at > session.target_plating:
            late = (t.end_at - session.target_plating).total_seconds()
            out.append((t, f"{t.label} would end at {fmt_time(t.end_at)}, {int(late // 60)}m after plating at {fmt_time(session.target_plating)}"))
        elif t.must_finish_by and t.must_finish_by != "plating":
            ref = session.tasks.get(t.must_finish_by)
            if ref is not None and ref.is_open and ref.start_at and t.end_at > ref.start_at:
                out.append((t, f"{t.label} would end at {fmt_time(t.end_at)} but must finish before {ref.label} starts at {fmt_time(ref.start_at)}"))
        elif session.target_plating and t.end_at > session.target_plating:
            late = (t.end_at - session.target_plating).total_seconds()
            out.append((t, f"{t.label} ends at {fmt_time(t.end_at)}, {int(late // 60)}m after target plating {fmt_time(session.target_plating)}"))
    return out


def all_violations(session: Session) -> List[str]:
    reasons = [r for _, _, r in appliance_conflicts(session)]
    reasons += burner_shortages(session)
    reasons += [r for _, r in deadline_misses(session)]
    return reasons


def next_action(session: Session, now: datetime) -> Optional[Tuple[datetime, str]]:
    """Earliest upcoming thing the cook must do: start a pending task or finish an active one."""
    best: Optional[Tuple[datetime, str]] = None
    for t in session.tasks.values():
        cand: Optional[Tuple[datetime, str]] = None
        if t.status == "pending" and t.start_at is not None:
            cand = (t.start_at, f"start {t.label}")
        elif t.status == "active" and t.end_at is not None:
            cand = (t.end_at, f"finish {t.label}")
        if cand is not None and (best is None or cand[0] < best[0]):
            best = cand
    return best
