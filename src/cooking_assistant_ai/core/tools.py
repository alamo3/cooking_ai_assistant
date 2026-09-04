"""Tool registry (spec section 2). The only path through which state changes.

Each tool:
* takes a ToolContext and keyword args exactly as the model sends them,
* raises ToolError(reason) to reject (reason is written to be re-prompt-friendly),
* returns (domain, message); the dispatcher renders `domain` into the envelope's `state`.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.clock import Clock
from cooking_assistant_ai.core.fmt import fmt_dur, fmt_ingredient, fmt_time, parse_clock_time, parse_duration
from cooking_assistant_ai.core.render import (
    render_changes,
    render_progress,
    render_recipe,
    render_state,
    render_timeline,
    render_timers,
)
from cooking_assistant_ai.model.types import Session, Step, Substitution, Task, Timer
from cooking_assistant_ai.storage.db import Store


class ToolError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ToolContext:
    session: Session
    clock: Clock
    store: Store

    @property
    def now(self) -> datetime:
        return self.clock.now()


@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    ok: bool
    state: str
    message: Optional[str] = None
    reason: Optional[str] = None
    domain: str = "state"

    def envelope(self) -> Dict[str, Any]:
        if self.ok:
            out: Dict[str, Any] = {"ok": True, "state": self.state}
            if self.message:
                out["message"] = self.message
            return out
        return {"ok": False, "reason": self.reason or "rejected", "state": self.state}

    def to_json(self) -> str:
        return json.dumps(self.envelope(), ensure_ascii=False)


ToolFn = Callable[..., Tuple[str, Optional[str]]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: ToolFn
    mid_cook: bool = False  # latency-sensitive; kept single round-trip

    def ollama_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


REGISTRY: Dict[str, ToolSpec] = {}


def _params(props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def tool(name: str, description: str, params: Dict[str, Any], mid_cook: bool = False):
    def deco(fn: ToolFn) -> ToolFn:
        REGISTRY[name] = ToolSpec(name, description, params, fn, mid_cook)
        return fn
    return deco


def tool_schemas(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    specs = REGISTRY.values() if names is None else [REGISTRY[n] for n in names]
    return [s.ollama_schema() for s in specs]


# --------------------------------------------------------------------------- coercion helpers

# Smaller models fill optional fields with these instead of omitting them, so treat them
# as absent rather than as a literal task label or appliance name.
_NULLISH = {"", "none", "null", "nil", "n/a", "na", "-", "undefined", "nothing", "no", "false"}


def _str(v: Any, name: str, required: bool = False) -> Optional[str]:
    text = "" if v is None else str(v).strip()
    if text.lower() in _NULLISH:
        if required:
            raise ToolError(f"'{name}' is required")
        return None
    return text


def _int(v: Any, name: str, required: bool = False, minimum: Optional[int] = None) -> Optional[int]:
    if v is None or v == "":
        if required:
            raise ToolError(f"'{name}' is required")
        return None
    try:
        out = int(float(v))
    except (TypeError, ValueError):
        if isinstance(v, str) and name.endswith("_s"):
            parsed = parse_duration(v)
            if parsed is not None:
                out = parsed
            else:
                raise ToolError(f"'{name}' must be a number of seconds, got {v!r}")
        else:
            raise ToolError(f"'{name}' must be a number, got {v!r}")
    if minimum is not None and out < minimum:
        raise ToolError(f"'{name}' must be >= {minimum}, got {out}")
    return out


def _float(v: Any, name: str, required: bool = False) -> Optional[float]:
    if v is None or v == "":
        if required:
            raise ToolError(f"'{name}' is required")
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ToolError(f"'{name}' must be a number, got {v!r}")


def _list(v: Any, name: str) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, (list, tuple)):
        return list(v)
    raise ToolError(f"'{name}' must be a list")


def _recipe(ctx: ToolContext, ref: Optional[str], required: bool = True):
    if not ref:
        if required:
            raise ToolError("'recipe_id' is required. Loaded recipes: " + (", ".join(f"{r.id} ({r.title})" for r in ctx.session.recipes.values()) or "none"))
        return None
    r = ctx.session.find_recipe(ref)
    if r is None:
        loaded = ", ".join(f"{x.id} ({x.title})" for x in ctx.session.recipes.values()) or "none"
        raise ToolError(f"recipe '{ref}' is not loaded in this session. Loaded: {loaded}. Use load_recipe first.")
    return r


def _task(ctx: ToolContext, ref: Optional[str], field: str = "task_id") -> Task:
    ref = _str(ref, field, required=True)
    t = ctx.session.find_task(ref or "")
    if t is None:
        known = ", ".join(f"{x.id} ({x.label})" for x in ctx.session.tasks.values()) or "none"
        raise ToolError(f"no task '{ref}'. Known tasks: {known}")
    return t


def _step_id(ctx: ToolContext, step_ref: str, recipe=None) -> str:
    """Accept a step id, or a 1-based step number when a recipe is known."""
    found = ctx.session.find_step(step_ref)
    if found:
        return found[1].id
    if recipe is not None and str(step_ref).isdigit():
        n = int(step_ref)
        if 1 <= n <= len(recipe.steps):
            return recipe.steps[n - 1].id
    raise ToolError(f"no step '{step_ref}' in the loaded recipes")


def _recipe_state(ctx: ToolContext, recipe) -> str:
    ov = ctx.session.overlays[recipe.id]
    txt = render_recipe(recipe, ov, ctx.session.completed_steps)
    ch = render_changes(recipe, ov)
    return txt + ("\n" + ch if ch else "")


def _validate_change(ctx: ToolContext, before: List[str], subject: str) -> None:
    """Compare violations before/after a mutation; raise (caller rolls back) if new ones appeared."""
    scheduler.resolve(ctx.session, ctx.now)
    after = scheduler.all_violations(ctx.session)
    new = [r for r in after if r not in before]
    if new:
        raise ToolError(f"cannot {subject}: " + " | ".join(new))


class _Snapshot:
    """Roll back task state if a scheduling mutation is rejected."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tasks = copy.deepcopy(ctx.session.tasks)
        self.plating = ctx.session.target_plating

    def restore(self) -> None:
        self.ctx.session.tasks = self.tasks
        self.ctx.session.target_plating = self.plating
        try:
            scheduler.resolve(self.ctx.session, self.ctx.now)
        except scheduler.ScheduleError:
            pass


# =========================================================================== recipe catalog

@tool(
    "list_recipes",
    "List recipes available in storage and which are loaded into this session.",
    _params({}),
)
def list_recipes(ctx: ToolContext) -> Tuple[str, Optional[str]]:
    return "catalog", None


@tool(
    "load_recipe",
    "Bring a stored recipe into the session so it can be cooked. Accepts a recipe id or (part of) its title.",
    _params({"recipe": {"type": "string", "description": "recipe id or title"}}, ["recipe"]),
)
def load_recipe(ctx: ToolContext, recipe: Any = None, recipe_id: Any = None, title: Any = None) -> Tuple[str, Optional[str]]:
    ref = _str(recipe or recipe_id or title, "recipe", required=True) or ""
    r = ctx.store.find_recipe(ref)
    if r is None:
        names = ", ".join(f"{x.id} ({x.title})" for x in ctx.store.list_recipes())
        raise ToolError(f"no stored recipe matches '{ref}'. Available: {names}")
    if r.id in ctx.session.recipes:
        return f"recipe:{r.id}", f"{r.title} is already loaded"
    ctx.session.add_recipe(r)
    return f"recipe:{r.id}", f"loaded {r.title}"


@tool(
    "unload_recipe",
    "Remove a recipe from the session. Rejected if it has active tasks.",
    _params({"recipe_id": {"type": "string"}}, ["recipe_id"]),
)
def unload_recipe(ctx: ToolContext, recipe_id: Any = None) -> Tuple[str, Optional[str]]:
    r = _recipe(ctx, _str(recipe_id, "recipe_id", required=True))
    active = [t.label for t in ctx.session.tasks.values() if t.recipe_id == r.id and t.status == "active"]
    if active:
        raise ToolError(f"{r.title} has active tasks ({', '.join(active)}); finish or remove them first")
    for t in [t for t in ctx.session.tasks.values() if t.recipe_id == r.id]:
        _remove_task_internal(ctx, t)
    del ctx.session.recipes[r.id]
    ctx.session.overlays.pop(r.id, None)
    return "progress", f"unloaded {r.title}"


# =========================================================================== planning tools

@tool(
    "set_target_plating",
    "Set when the meal should be on the table. Give either a clock time ('7:15 PM') or minutes_from_now.",
    _params({
        "time": {"type": "string", "description": "clock time like '7:15 PM' or '19:15'"},
        "minutes_from_now": {"type": "integer"},
    }),
)
def set_target_plating(ctx: ToolContext, time: Any = None, minutes_from_now: Any = None) -> Tuple[str, Optional[str]]:
    now = ctx.now
    mins = _int(minutes_from_now, "minutes_from_now")
    if mins is not None:
        target = (now + timedelta(minutes=mins)).replace(second=0, microsecond=0)
    else:
        text = _str(time, "time")
        if not text:
            raise ToolError("give either time ('7:15 PM') or minutes_from_now")
        parsed = parse_clock_time(text, now)
        if parsed is None:
            raise ToolError(f"could not understand time '{text}'. Use a form like '7:15 PM' or '19:15'.")
        if parsed < now - timedelta(minutes=1):
            parsed += timedelta(days=1)
            if parsed - now > timedelta(hours=12):
                raise ToolError(f"{fmt_time(parsed)} is in the past")
        target = parsed
    snap = _Snapshot(ctx)
    before = scheduler.all_violations(ctx.session)
    ctx.session.target_plating = target
    try:
        _validate_change(ctx, before, f"set plating to {fmt_time(target)}")
    except ToolError:
        snap.restore()
        raise
    return "timeline", f"target plating set to {fmt_time(target)} ({fmt_dur((target - now).total_seconds())} from now)"


_TASK_PARAMS = _params({
    "label": {"type": "string", "description": "short unique name, e.g. 'rice', 'chicken roast'"},
    "recipe_id": {"type": "string"},
    "step_ids": {"type": "array", "items": {"type": "string"}, "description": "recipe steps this task covers"},
    "appliance": {"type": "string", "description": "oven | stovetop (a free burner is assigned automatically) | stovetop:2 (a specific burner) | air_fryer | null"},
    "temp_f": {"type": "integer"},
    "duration_s": {"type": "integer", "description": "defaults to the sum of the steps' durations"},
    "after": {"type": "string", "description": "task label/id that must finish before this starts"},
    "before": {"type": "string", "description": "task label/id that must start after this finishes"},
    "must_finish_by": {"type": "string", "description": "'plating' to schedule as late as possible so it ends at plating, or a task label/id"},
}, ["label"])


@tool(
    "add_task",
    "Add a cooking task to the timeline. Express ordering intent (after/before/must_finish_by); code computes the clock times. Rejected on appliance or temperature conflicts, or if it cannot finish by plating.",
    _TASK_PARAMS,
)
def add_task(ctx: ToolContext, label: Any = None, recipe_id: Any = None, step_ids: Any = None,
             appliance: Any = None, temp_f: Any = None, duration_s: Any = None,
             after: Any = None, before: Any = None, must_finish_by: Any = None) -> Tuple[str, Optional[str]]:
    s = ctx.session
    label_s = _str(label, "label", required=True) or ""
    if s.find_task(label_s) is not None and any(t.label.lower() == label_s.lower() for t in s.tasks.values()):
        raise ToolError(f"a task labelled '{label_s}' already exists; use move_task or a different label")

    recipe = _recipe(ctx, _str(recipe_id, "recipe_id"), required=False)
    steps: List[str] = []
    for ref in _list(step_ids, "step_ids"):
        sid = _step_id(ctx, str(ref), recipe)
        found = s.find_step(sid)
        if recipe is None and found:
            recipe = found[0]
        steps.append(sid)

    appl = scheduler.normalize_appliance(_str(appliance, "appliance"))
    auto = scheduler.is_generic_stovetop(appl)
    temp = _int(temp_f, "temp_f") or None  # 0 means "no temperature", not absolute zero
    dur = _int(duration_s, "duration_s", minimum=1)
    if dur is None:
        total = 0
        for sid in steps:
            found = s.find_step(sid)
            if found and found[1].duration_s:
                total += found[1].duration_s
        if total <= 0:
            raise ToolError("'duration_s' is required (the listed steps have no durations)")
        dur = total
    if appl is None and steps:
        for sid in steps:
            found = s.find_step(sid)
            if found and found[1].appliance:
                appl = scheduler.normalize_appliance(found[1].appliance)
                auto = scheduler.is_generic_stovetop(appl)
                temp = temp if temp is not None else found[1].temp_f
                break

    after_s, before_s, mfb_s = _str(after, "after"), _str(before, "before"), _str(must_finish_by, "must_finish_by")
    deps: List[str] = []
    if after_s:
        deps.append(_task(ctx, after_s, "after").id)
    before_task = _task(ctx, before_s, "before") if before_s else None
    mfb: Optional[str] = None
    if mfb_s:
        ref = mfb_s
        if ref.lower() in ("plating", "target", "target_plating", "serve", "serving"):
            if s.target_plating is None:
                raise ToolError("must_finish_by='plating' but no target plating is set; call set_target_plating first")
            mfb = "plating"
        else:
            mfb = _task(ctx, ref, "must_finish_by").id

    snap = _Snapshot(ctx)
    before_v = scheduler.all_violations(s)
    task = Task(
        id=s.new_id("t"), label=label_s, recipe_id=recipe.id if recipe else "",
        step_ids=steps, appliance=appl, temp_f=temp, duration_s=dur, appliance_auto=auto,
        depends_on=deps, must_finish_by=mfb,
    )
    s.tasks[task.id] = task
    if before_task is not None:
        before_task.depends_on.append(task.id)
    try:
        try:
            scheduler.resolve(s, ctx.now)
        except scheduler.ScheduleError as e:
            raise ToolError(f"cannot add {label_s}: {e.reason}")
        _validate_change(ctx, before_v, f"add {label_s}")
    except ToolError:
        snap.restore()
        raise
    where = f" on {task.appliance}" if auto and task.appliance and ":" in task.appliance else ""
    return "plan", f"added {label_s} ({task.id}): {fmt_time(task.start_at)} to {fmt_time(task.end_at)}{where}"


def _remove_task_internal(ctx: ToolContext, t: Task) -> None:
    s = ctx.session
    del s.tasks[t.id]
    for other in s.tasks.values():
        other.depends_on = [d for d in other.depends_on if d != t.id]
        if other.must_finish_by == t.id:
            other.must_finish_by = None
    for tm in s.timers.values():
        if tm.task_id == t.id and tm.status == "running":
            tm.status = "cancelled"


@tool(
    "remove_task",
    "Remove a task from the timeline. Dependents are rescheduled.",
    _params({"task_id": {"type": "string", "description": "task id or label"}}, ["task_id"]),
)
def remove_task(ctx: ToolContext, task_id: Any = None) -> Tuple[str, Optional[str]]:
    t = _task(ctx, task_id)
    _remove_task_internal(ctx, t)
    scheduler.resolve(ctx.session, ctx.now)
    return "timeline", f"removed {t.label}"


@tool("get_plan", "Return the timeline and the merged cook plan (every recipe's steps in the order to do them). Cheap; call freely.", _params({}))
def get_plan(ctx: ToolContext) -> Tuple[str, Optional[str]]:
    return "plan", None


@tool(
    "complete_prep",
    "The cook finished a shared prep group from the cook plan (e.g. all the garlic for every recipe). Marks every step in that group done.",
    _params({"group": {"type": "string", "description": "group id like prep_garlic, or the ingredient name"}}, ["group"]),
    mid_cook=True,
)
def complete_prep(ctx: ToolContext, group: Any = None) -> Tuple[str, Optional[str]]:
    from cooking_assistant_ai.core.plan import find_prep_group, prep_groups

    ref = _str(group, "group", required=True) or ""
    g = find_prep_group(ctx.session, ref)
    if g is None:
        known = ", ".join(f"{x.id} ({x.label})" for x in prep_groups(ctx.session)) or "none"
        raise ToolError(f"no prep group '{ref}'. Groups: {known}")
    if g.status == "done":
        raise ToolError(f"{g.label} prep is already done")
    mark_complete(ctx, step_ids=list(g.step_ids))
    return "plan", f"{g.label} prepped for all recipes ({len(g.step_ids)} step(s) done)"


@tool(
    "replan",
    "Escape hatch: clear every pending task (active/complete ones stay) so the plan can be rebuilt with add_task.",
    _params({"reason": {"type": "string"}}, ["reason"]),
)
def replan(ctx: ToolContext, reason: Any = None) -> Tuple[str, Optional[str]]:
    pending = [t for t in ctx.session.tasks.values() if t.status == "pending"]
    for t in pending:
        _remove_task_internal(ctx, t)
    scheduler.resolve(ctx.session, ctx.now)
    why = _str(reason, "reason") or "no reason given"
    return "timeline", f"cleared {len(pending)} pending task(s) ({why}). Rebuild with add_task."


# =========================================================================== mid-cook tools

@tool(
    "start_task",
    "Mark a task as started right now (the cook put it on / in). Pins its window and shifts dependents.",
    _params({"task_id": {"type": "string", "description": "task id or label"}}, ["task_id"]),
    mid_cook=True,
)
def start_task(ctx: ToolContext, task_id: Any = None) -> Tuple[str, Optional[str]]:
    t = _task(ctx, task_id)
    if t.status == "active":
        raise ToolError(f"{t.label} is already active (started {fmt_time(t.actual_start)})")
    if t.status == "complete":
        raise ToolError(f"{t.label} is already complete")
    t.status = "active"
    t.actual_start = ctx.now
    t.start_at = ctx.now
    scheduler.resolve(ctx.session, ctx.now)
    return "timeline", f"{t.label} started at {fmt_time(ctx.now)}, ends {fmt_time(t.end_at)}"


@tool(
    "move_task",
    "Reschedule a pending task: after='<task>' to order it after another task, or delay_minutes=N to push it back. Dependents shift automatically.",
    _params({
        "task_id": {"type": "string", "description": "task id or label"},
        "after": {"type": "string"},
        "before": {"type": "string"},
        "delay_minutes": {"type": "integer"},
    }, ["task_id"]),
    mid_cook=True,
)
def move_task(ctx: ToolContext, task_id: Any = None, after: Any = None, before: Any = None,
              delay_minutes: Any = None) -> Tuple[str, Optional[str]]:
    s = ctx.session
    t = _task(ctx, task_id)
    if t.status != "pending":
        raise ToolError(f"{t.label} is {t.status}; only pending tasks can be moved")
    delay = _int(delay_minutes, "delay_minutes")
    after_s, before_s = _str(after, "after"), _str(before, "before")
    if not after_s and not before_s and delay is None:
        raise ToolError("give after=, before= or delay_minutes=")
    snap = _Snapshot(ctx)
    before_v = scheduler.all_violations(s)
    what: List[str] = []
    try:
        if after_s:
            ref = _task(ctx, after_s, "after")
            if ref.id == t.id:
                raise ToolError("a task cannot come after itself")
            t.depends_on = [ref.id]
            t.not_before = None
            what.append(f"after {ref.label}")
        if before_s:
            ref = _task(ctx, before_s, "before")
            if ref.id == t.id:
                raise ToolError("a task cannot come before itself")
            if t.id not in ref.depends_on:
                ref.depends_on.append(t.id)
            what.append(f"before {ref.label}")
        if delay is not None:
            base = t.start_at if t.start_at and t.start_at > ctx.now else ctx.now
            t.not_before = base + timedelta(minutes=delay)
            what.append(f"delayed {delay}m")
        try:
            scheduler.resolve(s, ctx.now)
        except scheduler.ScheduleError as e:
            raise ToolError(f"cannot move {t.label}: {e.reason}")
        _validate_change(ctx, before_v, f"move {t.label}")
    except ToolError:
        snap.restore()
        raise
    return "timeline", f"moved {t.label} {', '.join(what)}: now {fmt_time(t.start_at)} to {fmt_time(t.end_at)}"


@tool(
    "mark_complete",
    "Record that steps are done (step_ids) or that a whole task is done (task_id). Cancels that task's timers.",
    _params({
        "step_ids": {"type": "array", "items": {"type": "string"}},
        "task_id": {"type": "string", "description": "task id or label"},
    }),
    mid_cook=True,
)
def mark_complete(ctx: ToolContext, step_ids: Any = None, task_id: Any = None, step_id: Any = None) -> Tuple[str, Optional[str]]:
    s = ctx.session
    done: List[str] = []
    refs = _list(step_ids, "step_ids")
    if step_id:
        refs.append(step_id)
    for ref in refs:
        sid = _step_id(ctx, str(ref))
        s.completed_steps.add(sid)
        done.append(sid)
    finished: List[str] = []
    if task_id:
        t = _task(ctx, task_id)
        if t.status == "complete":
            raise ToolError(f"{t.label} is already complete")
        t.status = "complete"
        t.actual_end = ctx.now
        if t.actual_start is None:
            t.actual_start = t.start_at if t.start_at and t.start_at < ctx.now else ctx.now - timedelta(seconds=t.duration_s)
        for sid in t.step_ids:
            s.completed_steps.add(sid)
        finished.append(t.label)
    elif not done:
        raise ToolError("give step_ids or task_id")
    # A task whose steps are all complete is complete too.
    for t in s.tasks.values():
        if t.is_open and t.step_ids and all(sid in s.completed_steps for sid in t.step_ids):
            t.status = "complete"
            t.actual_end = ctx.now
            if t.actual_start is None:
                t.actual_start = ctx.now - timedelta(seconds=t.duration_s)
            finished.append(t.label)
    for t in s.tasks.values():
        if t.status == "complete":
            for tm in s.timers.values():
                if tm.task_id == t.id and tm.status == "running":
                    tm.status = "cancelled"
    scheduler.resolve(s, ctx.now)
    msg_bits = []
    if done:
        names = []
        for sid in done:
            found = s.find_step(sid)
            if found:
                names.append(f"{found[0].title} step {found[0].step_index(sid) if found[0].step_index(sid) > 0 else sid}")
        msg_bits.append("completed " + ", ".join(names))
    if finished:
        msg_bits.append("task(s) done: " + ", ".join(finished))
    return "state", "; ".join(msg_bits)


@tool(
    "skip_step",
    "Skip a recipe step, recording why.",
    _params({"step_id": {"type": "string"}, "reason": {"type": "string"}}, ["step_id"]),
    mid_cook=True,
)
def skip_step(ctx: ToolContext, step_id: Any = None, reason: Any = None) -> Tuple[str, Optional[str]]:
    sid = _step_id(ctx, _str(step_id, "step_id", required=True) or "")
    recipe, step = ctx.session.find_step(sid)  # type: ignore[misc]
    ov = ctx.session.overlays[recipe.id]
    ov.skipped_steps.add(sid)
    why = _str(reason, "reason")
    if why:
        ov.skip_reasons[sid] = why
    return f"recipe:{recipe.id}", f"skipped step {recipe.step_index(sid)} of {recipe.title}" + (f" ({why})" if why else "")


# =========================================================================== timer tools

_GENERIC_LABEL = re.compile(r"^(timer|alarm|countdown|reminder)?\s*\d*$", re.I)


@tool(
    "set_timer",
    "Start a countdown. label must say what it is for ('chicken resting', not 'timer'). on_complete_hint is what the cook should do when it fires.",
    _params({
        "label": {"type": "string"},
        "duration_s": {"type": "integer"},
        "task_id": {"type": "string"},
        "step_id": {"type": "string"},
        "on_complete_hint": {"type": "string"},
    }, ["label", "duration_s"]),
    mid_cook=True,
)
def set_timer(ctx: ToolContext, label: Any = None, duration_s: Any = None, task_id: Any = None,
              step_id: Any = None, on_complete_hint: Any = None, duration_minutes: Any = None) -> Tuple[str, Optional[str]]:
    s = ctx.session
    label_s = _str(label, "label", required=True) or ""
    if _GENERIC_LABEL.match(label_s):
        raise ToolError(f"label '{label_s}' is too generic; say what it is for, e.g. 'rice simmering'")
    dur = _int(duration_s, "duration_s", minimum=1)
    if dur is None:
        mins = _int(duration_minutes, "duration_minutes", minimum=1)
        if mins is None:
            raise ToolError("'duration_s' is required")
        dur = mins * 60
    tid = _task(ctx, task_id).id if task_id else None
    sid = _step_id(ctx, str(step_id)) if step_id else None
    for existing in s.running_timers():
        if existing.label.lower() == label_s.lower():
            raise ToolError(f"a timer labelled '{label_s}' is already running (ends {fmt_time(existing.end_at)}); cancel it or use a different label")
    timer = Timer(
        id=s.new_id("tm"), label=label_s, task_id=tid, step_id=sid,
        end_at=ctx.now + timedelta(seconds=dur),
        on_complete_hint=_str(on_complete_hint, "on_complete_hint"),
        created_at=ctx.now,
    )
    s.timers[timer.id] = timer
    return "timers", f"timer '{label_s}' set for {fmt_dur(dur)}, ends {fmt_time(timer.end_at)}"


@tool(
    "cancel_timer",
    "Cancel a running timer by id or label.",
    _params({"timer_id": {"type": "string"}}, ["timer_id"]),
    mid_cook=True,
)
def cancel_timer(ctx: ToolContext, timer_id: Any = None, label: Any = None) -> Tuple[str, Optional[str]]:
    ref = _str(timer_id or label, "timer_id", required=True) or ""
    s = ctx.session
    timer = s.timers.get(ref)
    if timer is None:
        low = ref.lower()
        for t in s.running_timers():
            if t.label.lower() == low or low in t.label.lower():
                timer = t
                break
    if timer is None:
        running = ", ".join(f"{t.id} ({t.label})" for t in s.running_timers()) or "none"
        raise ToolError(f"no timer '{ref}'. Running: {running}")
    if timer.status != "running":
        raise ToolError(f"timer '{timer.label}' is already {timer.status}")
    timer.status = "cancelled"
    return "timers", f"cancelled '{timer.label}'"


@tool("get_timers", "List running timers.", _params({}), mid_cook=True)
def get_timers(ctx: ToolContext) -> Tuple[str, Optional[str]]:
    return "timers", None


# =========================================================================== recipe tools

def _ingredient(recipe, ref: str):
    ing = recipe.ingredient(ref)
    if ing:
        return ing
    low = ref.strip().lower()
    for i in recipe.ingredients:
        if i.name.lower() == low:
            return i
    for i in recipe.ingredients:
        if low in i.name.lower():
            return i
    names = ", ".join(f"{i.id} ({i.name})" for i in recipe.ingredients)
    raise ToolError(f"no ingredient '{ref}' in {recipe.title}. Ingredients: {names}")


@tool(
    "substitute",
    "Replace an ingredient in a loaded recipe (by id or name).",
    _params({
        "recipe_id": {"type": "string"},
        "ingredient_id": {"type": "string", "description": "ingredient id or name"},
        "replacement": {"type": "string"},
        "note": {"type": "string", "description": "e.g. '1:1 by volume'"},
        "at_step": {"type": "string"},
    }, ["recipe_id", "ingredient_id", "replacement"]),
)
def substitute(ctx: ToolContext, recipe_id: Any = None, ingredient_id: Any = None, replacement: Any = None,
               note: Any = None, at_step: Any = None, ingredient: Any = None) -> Tuple[str, Optional[str]]:
    recipe = _recipe(ctx, _str(recipe_id, "recipe_id"))
    ing = _ingredient(recipe, _str(ingredient_id or ingredient, "ingredient_id", required=True) or "")
    rep = _str(replacement, "replacement", required=True) or ""
    step = _step_id(ctx, str(at_step), recipe) if at_step else None
    ov = ctx.session.overlays[recipe.id]
    ov.substitutions = [x for x in ov.substitutions if x.ingredient_id != ing.id]
    ov.substitutions.append(Substitution(ingredient_id=ing.id, replacement=rep, note=_str(note, "note"), at_step=step))
    return f"recipe:{recipe.id}", f"{ing.name} -> {rep} in {recipe.title}"


@tool(
    "scale",
    "Scale a recipe's ingredient amounts by a factor (2.0 doubles). Durations are NOT scaled; reason about timing yourself.",
    _params({"recipe_id": {"type": "string"}, "factor": {"type": "number"}}, ["recipe_id", "factor"]),
)
def scale(ctx: ToolContext, recipe_id: Any = None, factor: Any = None, servings: Any = None) -> Tuple[str, Optional[str]]:
    recipe = _recipe(ctx, _str(recipe_id, "recipe_id"))
    f = _float(factor, "factor")
    if f is None:
        sv = _float(servings, "servings")
        if sv is None:
            raise ToolError("'factor' is required")
        f = sv / recipe.servings
    if f <= 0:
        raise ToolError("factor must be positive")
    ctx.session.overlays[recipe.id].scale_factor = f
    return f"recipe:{recipe.id}", f"{recipe.title} scaled x{f:g} (now serves {recipe.servings * f:g})"


@tool(
    "add_note",
    "Attach a note to a recipe step (e.g. 'pan was smoking, reduced heat').",
    _params({"recipe_id": {"type": "string"}, "step_id": {"type": "string"}, "note": {"type": "string"}}, ["recipe_id", "step_id", "note"]),
)
def add_note(ctx: ToolContext, recipe_id: Any = None, step_id: Any = None, note: Any = None) -> Tuple[str, Optional[str]]:
    recipe = _recipe(ctx, _str(recipe_id, "recipe_id"))
    sid = _step_id(ctx, _str(step_id, "step_id", required=True) or "", recipe)
    text = _str(note, "note", required=True) or ""
    ov = ctx.session.overlays[recipe.id]
    ov.step_notes[sid] = (ov.step_notes[sid] + " " + text) if sid in ov.step_notes else text
    return f"recipe:{recipe.id}", f"note added to step {recipe.step_index(sid)}"


@tool(
    "add_step",
    "Append a new step to a loaded recipe (for improvised additions).",
    _params({
        "recipe_id": {"type": "string"}, "text": {"type": "string"},
        "duration_s": {"type": "integer"}, "appliance": {"type": "string"}, "temp_f": {"type": "integer"},
    }, ["recipe_id", "text"]),
)
def add_step(ctx: ToolContext, recipe_id: Any = None, text: Any = None, duration_s: Any = None,
             appliance: Any = None, temp_f: Any = None) -> Tuple[str, Optional[str]]:
    recipe = _recipe(ctx, _str(recipe_id, "recipe_id"))
    ov = ctx.session.overlays[recipe.id]
    n = len(recipe.steps) + len(ov.added_steps) + 1
    step = Step(
        id=f"{recipe.id}-s{n}", text=_str(text, "text", required=True) or "",
        duration_s=_int(duration_s, "duration_s"), appliance=scheduler.normalize_appliance(_str(appliance, "appliance")),
        temp_f=_int(temp_f, "temp_f"),
    )
    ov.added_steps.append(step)
    return f"recipe:{recipe.id}", f"added step {n} to {recipe.title} ({step.id})"


# =========================================================================== memory

@tool(
    "remember",
    "Keep a conversational fact for the rest of the session (preferences, allergies, deviations).",
    _params({"note": {"type": "string"}}, ["note"]),
)
def remember(ctx: ToolContext, note: Any = None) -> Tuple[str, Optional[str]]:
    text = _str(note, "note", required=True) or ""
    if text not in ctx.session.notes:
        ctx.session.notes.append(text)
    return "progress", "noted"


# =========================================================================== inventory

def _inventory_text(ctx: ToolContext) -> str:
    items = ctx.store.inventory()
    if not items:
        return "INVENTORY\nempty"
    return "INVENTORY\n" + "\n".join(f"{fmt_ingredient(i['name'], i['amount'], i['unit'])}" for i in items)


@tool(
    "set_diet",
    "Record the household's dietary restriction. It persists and governs every suggestion.",
    _params({"diet": {"type": "string", "enum": ["none", "halal", "vegetarian", "vegan"]}}, ["diet"]),
)
def set_diet(ctx: ToolContext, diet: Any = None) -> Tuple[str, Optional[str]]:
    try:
        d = ctx.store.set_diet(_str(diet, "diet", required=True) or "")
    except ValueError as e:
        raise ToolError(str(e))
    return "meals", f"diet set to {d}"


@tool(
    "suggest_meals",
    "What can I cook? Returns every stored recipe with how many of its ingredients are in the "
    "pantry, which are missing, and how many portions it makes. Use it to recommend a set of "
    "recipes; say what to buy only when it is one or two easy items.",
    _params({
        "meals": {"type": "integer", "description": "how many portions the cook wants in total"},
        "scale": {"type": "number", "description": "batch multiplier for every recipe, default 1"},
    }),
)
def suggest_meals(ctx: ToolContext, meals: Any = None, scale: Any = None) -> Tuple[str, Optional[str]]:
    n = _int(meals, "meals")
    factor = _float(scale, "scale") or 1.0
    if factor <= 0:
        raise ToolError("scale must be positive")
    ctx.meal_options = (n, factor)  # type: ignore[attr-defined]
    return "meals", None


@tool(
    "shopping_list",
    "What to buy for a chosen set of recipes, after subtracting what is already in the pantry.",
    _params({
        "recipe_ids": {"type": "array", "items": {"type": "string"}},
        "scale": {"type": "number"},
    }, ["recipe_ids"]),
)
def shopping_list_tool(ctx: ToolContext, recipe_ids: Any = None, scale: Any = None) -> Tuple[str, Optional[str]]:
    refs = _list(recipe_ids, "recipe_ids")
    if not refs:
        raise ToolError("'recipe_ids' is required")
    recipes = []
    for ref in refs:
        r = ctx.store.find_recipe(str(ref))
        if r is None:
            known = ", ".join(f"{x.id} ({x.title})" for x in ctx.store.list_recipes())
            raise ToolError(f"no stored recipe '{ref}'. Available: {known}")
        recipes.append(r)
    factor = _float(scale, "scale") or 1.0
    ctx.shopping_for = (recipes, factor)  # type: ignore[attr-defined]
    return "shopping", None


@tool(
    "check_stock",
    "Check whether ingredients are in the pantry inventory.",
    _params({"items": {"type": "array", "items": {"type": "string"}}}, ["items"]),
)
def check_stock(ctx: ToolContext, items: Any = None) -> Tuple[str, Optional[str]]:
    names = [str(x) for x in _list(items, "items")]
    if not names:
        raise ToolError("'items' is required")
    lines = []
    for name in names:
        item = ctx.store.get_stock(name)
        if item is None or item["amount"] <= 0:
            lines.append(f"{name}: NOT in stock")
        else:
            lines.append(f"{name}: {fmt_ingredient(item['name'], item['amount'], item['unit'])}")
    return "inventory", "; ".join(lines)


@tool(
    "deduct",
    "Deduct used quantities from the inventory.",
    _params({"items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "amount": {"type": "number"}, "unit": {"type": "string"}}}}}, ["items"]),
)
def deduct(ctx: ToolContext, items: Any = None) -> Tuple[str, Optional[str]]:
    rows = _list(items, "items")
    if not rows:
        raise ToolError("'items' is required")
    done = []
    for row in rows:
        if isinstance(row, str):
            row = {"name": row, "amount": 1}
        name = _str(row.get("name"), "name", required=True) or ""
        amount = _float(row.get("amount"), "amount", required=True) or 0.0
        try:
            new = ctx.store.deduct(name, amount, row.get("unit"))
        except ValueError as e:
            raise ToolError(str(e))
        done.append(f"{new['name']} now {fmt_ingredient('', new['amount'], new['unit']).strip()}")
    return "inventory", "; ".join(done)


# =========================================================================== dispatch

def render_domain(ctx: ToolContext, domain: str) -> str:
    now = ctx.now
    s = ctx.session
    if domain == "timeline":
        return render_timeline(s, now)
    if domain == "plan":
        from cooking_assistant_ai.core.plan import render_cook_plan, render_plan_summary

        return render_timeline(s, now) + "\n\n" + render_plan_summary(s, now) + "\n\n" + render_cook_plan(s, now)
    if domain == "timers":
        return render_timers(s, now)
    if domain == "progress":
        return render_progress(s)
    if domain.startswith("recipe:"):
        r = s.recipes.get(domain.split(":", 1)[1])
        return _recipe_state(ctx, r) if r else render_progress(s)
    if domain == "inventory":
        return _inventory_text(ctx)
    if domain == "meals":
        from cooking_assistant_ai.core.mealplan import render_meal_options

        wanted, factor = getattr(ctx, "meal_options", (None, 1.0))
        return render_meal_options(ctx.store, wanted, factor)
    if domain == "shopping":
        from cooking_assistant_ai.core.mealplan import render_shopping_list

        recipes, factor = getattr(ctx, "shopping_for", ([], 1.0))
        return render_shopping_list(ctx.store, recipes, factor)
    if domain == "catalog":
        lines = ["RECIPES IN STORAGE"]
        for r in ctx.store.list_recipes():
            flag = " (loaded)" if r.id in s.recipes else ""
            lines.append(f"{r.id}  {r.title}  serves {r.servings}, {len(r.steps)} steps{flag}")
        return "\n".join(lines)
    return render_state(s, now)


def dispatch(ctx: ToolContext, name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
    args = dict(args or {})
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult(name, args, ok=False, state=render_state(ctx.session, ctx.now),
                          reason=f"unknown tool '{name}'. Available: {', '.join(sorted(REGISTRY))}")
    sig = inspect.signature(spec.fn)
    accepted = {k: v for k, v in args.items() if k in sig.parameters}
    domain = "state"
    try:
        domain, message = spec.fn(ctx, **accepted)
        return ToolResult(name, args, ok=True, state=render_domain(ctx, domain), message=message, domain=domain)
    except ToolError as e:
        return ToolResult(name, args, ok=False, state=render_domain(ctx, domain), reason=e.reason, domain=domain)
    except scheduler.ScheduleError as e:
        return ToolResult(name, args, ok=False, state=render_domain(ctx, domain), reason=e.reason, domain=domain)
