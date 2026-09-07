"""One renderer, two consumers: context assembly and tool results (spec invariant 5).

Everything here is derived from structured state. Nothing is LLM-generated.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from cooking_assistant_ai.core import scheduler
from cooking_assistant_ai.core.fmt import fmt_dur, fmt_ingredient, fmt_time, fmt_window
from cooking_assistant_ai.core.plan import mentions_ingredient, substitute_text
from cooking_assistant_ai.model.types import Overlay, Recipe, Session, Step, Task


# --------------------------------------------------------------------------- recipe

def _step_detail(s: Step) -> str:
    bits = []
    if s.duration_s:
        bits.append(fmt_dur(s.duration_s))
    if s.appliance:
        bits.append(f"{s.appliance} {s.temp_f}°F" if s.temp_f else s.appliance)
    return f" ({', '.join(bits)})" if bits else ""


def render_recipe(recipe: Recipe, overlay: Optional[Overlay] = None,
                  completed_steps: Iterable[str] = ()) -> str:
    overlay = overlay or Overlay(recipe_id=recipe.id)
    done = set(completed_steps)
    scale = overlay.scale_factor
    subs = {s.ingredient_id: s for s in overlay.substitutions}

    servings = recipe.servings * scale
    servings_txt = str(int(servings)) if servings == int(servings) else f"{servings:.1f}"
    head = f"RECIPE: {recipe.title}  [{recipe.id}]  serves {servings_txt}"
    if scale != 1.0:
        head += f"  (scaled x{scale:g} from {recipe.servings})"
    lines = [head, "Ingredients:"]
    for ing in recipe.ingredients:
        sub = subs.get(ing.id)
        if sub:
            base = fmt_ingredient(sub.replacement, ing.amount * scale, ing.unit)
            was = sub.original or ing.name
            extra = f" (instead of {was}" + (f", {sub.note}" if sub.note else "") + ")"
            lines.append(f"  - {base}{extra}  [{ing.id}]")
        else:
            lines.append(f"  - {fmt_ingredient(ing.name, ing.amount * scale, ing.unit)}  [{ing.id}]")
    lines.append("Steps:")
    for n, s in enumerate(recipe.steps, start=1):
        if s.id in done:
            mark = "[done]"
        elif s.id in overlay.skipped_steps:
            mark = "[skipped]"
        else:
            mark = "[    ]"
        text = substitute_text(s.text, recipe, overlay, s.id)
        lines.append(f"  {n:>2}. {mark} {text}{_step_detail(s)}  [{s.id}]")
        note = overlay.step_notes.get(s.id)
        if note:
            lines.append(f"        note: {note}")
        reason = overlay.skip_reasons.get(s.id)
        if s.id in overlay.skipped_steps and reason:
            lines.append(f"        skipped: {reason}")
    for n, s in enumerate(overlay.added_steps, start=len(recipe.steps) + 1):
        mark = "[done]" if s.id in done else "[    ]"
        lines.append(f"  {n:>2}. {mark} {s.text}{_step_detail(s)}  [{s.id}] (added)")
    return "\n".join(lines)


def render_changes(recipe: Recipe, overlay: Overlay) -> str:
    """Short 'changes made' list so the model knows what was altered. '' if nothing."""
    if overlay.is_clean():
        return ""
    items: List[str] = []
    if overlay.scale_factor != 1.0:
        items.append(f"scaled x{overlay.scale_factor:g}")
    for sub in overlay.substitutions:
        ing = recipe.ingredient(sub.ingredient_id)
        name = sub.original or (ing.name if ing else sub.ingredient_id)
        where = ""
        if sub.at_step:
            idx = recipe.step_index(sub.at_step)
            where = f" (step {idx})" if idx > 0 else f" ({sub.at_step})"
        note = f", {sub.note}" if sub.note else ""
        items.append(f"{name} -> {sub.replacement}{where}{note}")
    for sid in sorted(overlay.skipped_steps, key=lambda x: recipe.step_index(x)):
        idx = recipe.step_index(sid)
        reason = overlay.skip_reasons.get(sid)
        items.append(f"step {idx} skipped" + (f" ({reason})" if reason else ""))
    for sid, note in overlay.step_notes.items():
        idx = recipe.step_index(sid)
        items.append(f"note on step {idx}: {note}")
    for s in overlay.added_steps:
        items.append(f"added step: {s.text}")
    return f"Changes to {recipe.title}: " + "; ".join(items)


def render_all_recipes(session: Session) -> str:
    if not session.recipes:
        return "RECIPES: none loaded. Use list_recipes / load_recipe to bring one into the session."
    blocks = []
    for r in session.recipes.values():
        blocks.append(render_recipe(r, session.overlays.get(r.id), session.completed_steps))
    return "\n\n".join(blocks)


def render_all_changes(session: Session) -> str:
    out = []
    for r in session.recipes.values():
        txt = render_changes(r, session.overlays[r.id])
        if txt:
            out.append(txt)
    return "\n".join(out)


# --------------------------------------------------------------------------- timeline

def _task_status(t: Task, now: datetime) -> str:
    if t.status == "complete":
        return "done"
    if t.status == "skipped":
        return "skipped"
    if t.awaits_cook and t.status == "active":
        return "running, tell me when it's done"
    if t.awaits_cook:
        return "waiting to start, ~est, ends when you say"
    if t.status == "active":
        left = (t.end_at - now).total_seconds() if t.end_at else 0
        return f"active, {fmt_dur(left)} left" if left >= 0 else f"active, overdue {fmt_dur(-left)}"
    # pending
    if t.start_at is None:
        return "pending"
    delta = (t.start_at - now).total_seconds()
    return "pending, start now" if delta <= 30 else f"pending, starts in {fmt_dur(delta)}"


def _appliance_col(t: Task) -> str:
    if not t.appliance:
        return "-"
    return f"{t.appliance} {t.temp_f}°F" if t.temp_f else t.appliance


def render_timeline(session: Session, now: datetime) -> str:
    try:
        scheduler.resolve(session, now)
    except scheduler.ScheduleError as e:  # pragma: no cover - state should never be left invalid
        return f"TIMELINE ERROR: {e.reason}"

    head = f"Now: {fmt_time(now)}".ljust(32)
    if session.target_plating:
        head += f"Target plating: {fmt_time(session.target_plating)}"
    lines = [head.rstrip(), ""]
    if not session.tasks:
        lines.append("No tasks planned yet.")
    else:
        lines.append(f"{'APPLIANCE':<18} {'WINDOW':<13} {'TASK':<20} {'STATUS':<24} ID")
        ordered = sorted(session.tasks.values(),
                         key=lambda t: (t.status == "complete", t.start_at or now))
        for t in ordered:
            lines.append(
                f"{_appliance_col(t):<18} {fmt_window(t.start_at, t.end_at):<13} "
                f"{t.label[:20]:<20} {_task_status(t, now):<24} {t.id}"
            )
    lines.append("")
    nxt = scheduler.next_action(session, now)
    if nxt is None:
        lines.append("Next action: none scheduled")
    else:
        when = "now" if nxt[0] <= now else fmt_time(nxt[0])
        lines.append(f"Next action: {when} - {nxt[1]}")
    problems = scheduler.all_violations(session)
    lines.append("Conflicts: " + ("none" if not problems else "; ".join(problems)))
    for warning in scheduler.drift_warnings(session, now):
        lines.append(f"CHECK: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- timers

def render_timers(session: Session, now: datetime) -> str:
    running = sorted(session.running_timers(), key=lambda t: t.end_at)
    lines = ["TIMERS"]
    if not running:
        lines.append("none running")
        return "\n".join(lines)
    for t in running:
        left = (t.end_at - now).total_seconds()
        left_txt = f"({fmt_dur(left)} left)" if left >= 0 else "(due)"
        hint = f"   -> {t.on_complete_hint}" if t.on_complete_hint else ""
        lines.append(f"{t.label[:22]:<22} ends {fmt_time(t.end_at):<9} {left_txt:<12}{hint}  [{t.id}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- progress

def _compress_indices(idx: List[int]) -> str:
    if not idx:
        return ""
    idx = sorted(set(idx))
    runs: List[List[int]] = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def render_progress(session: Session) -> str:
    lines = ["PROGRESS"]
    if not session.recipes:
        lines.append("No recipes loaded.")
    for r in session.recipes.values():
        ov = session.overlays[r.id]
        all_steps = list(r.steps) + list(ov.added_steps)
        done_idx = [n for n, s in enumerate(all_steps, start=1) if s.id in session.completed_steps]
        skipped_idx = [n for n, s in enumerate(all_steps, start=1) if s.id in ov.skipped_steps]
        nxt = next((s for s in all_steps
                    if s.id not in session.completed_steps and s.id not in ov.skipped_steps), None)
        if not done_idx and not skipped_idx:
            status = "not started"
        elif nxt is None:
            status = "all steps done"
        else:
            status = f"steps {_compress_indices(done_idx)} done"
            if skipped_idx:
                status += f", {_compress_indices(skipped_idx)} skipped"
        if nxt is not None and (done_idx or skipped_idx):
            nidx = all_steps.index(nxt) + 1
            status += f'. Next: step {nidx} "{nxt.text[:60]}"'
        lines.append(f"{r.title} ({r.id}): {status}.")
    changes = render_all_changes(session)
    if changes:
        lines.append(changes)
    if session.notes:
        lines.append("Notes: " + " ".join(n.rstrip(".") + "." for n in session.notes))
    return "\n".join(lines)


# --------------------------------------------------------------------------- whole state

def render_state(session: Session, now: datetime) -> str:
    return "\n\n".join([
        render_timeline(session, now),
        render_timers(session, now),
        render_progress(session),
    ])


def state_dict(session: Session, now: datetime) -> Dict[str, Any]:
    """Structured state for the tablet UI (websocket 'state' message)."""
    from cooking_assistant_ai.core.plan import build_plan, plan_summary

    scheduler.resolve(session, now)
    nxt = scheduler.next_action(session, now)
    plan = build_plan(session, now).to_dict()
    plan["summary"] = plan_summary(session, now)
    return {
        "plan": plan,
        "now": now.isoformat(),
        "target_plating": session.target_plating.isoformat() if session.target_plating else None,
        "proactivity": session.proactivity,
        "timeline": {
            "tasks": [
                {
                    "id": t.id, "label": t.label, "recipe_id": t.recipe_id,
                    "appliance": t.appliance, "temp_f": t.temp_f, "status": t.status,
                    "start_at": t.start_at.isoformat() if t.start_at else None,
                    "end_at": t.end_at.isoformat() if t.end_at else None,
                    "step_ids": list(t.step_ids), "depends_on": list(t.depends_on),
                }
                for t in session.tasks.values()
            ],
            "next_action": {"at": nxt[0].isoformat(), "text": nxt[1]} if nxt else None,
            "conflicts": scheduler.all_violations(session),
        },
        "timers": [
            {
                "id": t.id, "label": t.label, "end_at": t.end_at.isoformat(),
                "seconds_left": max(0, int((t.end_at - now).total_seconds())),
                "hint": t.on_complete_hint, "status": t.status,
                "task_id": t.task_id, "step_id": t.step_id,
            }
            for t in session.timers.values() if t.status == "running"
        ],
        "progress": {
            "recipes": [recipe_view(session, r) for r in session.recipes.values()],
            "notes": list(session.notes),
            "text": render_progress(session),
        },
        "text": render_state(session, now),
    }


def recipe_view(session: Session, r: Recipe) -> Dict[str, Any]:
    """Structured recipe view with the overlay applied, for the tablet UI."""
    ov = session.overlays[r.id]
    subs = {s.ingredient_id: s for s in ov.substitutions}
    scale = ov.scale_factor
    ingredients = []
    for ing in r.ingredients:
        sub = subs.get(ing.id)
        name = sub.replacement if sub else ing.name
        ingredients.append({
            "id": ing.id, "name": name, "amount": ing.amount * scale, "unit": ing.unit,
            "text": fmt_ingredient(name, ing.amount * scale, ing.unit),
            "substituted_for": (sub.original or ing.name) if sub else None,
            "note": sub.note if sub else None,
        })
    steps = []
    current = None
    for n, s in enumerate(list(r.steps) + list(ov.added_steps), start=1):
        if s.id in session.completed_steps:
            status = "done"
        elif s.id in ov.skipped_steps:
            status = "skipped"
        else:
            status = "pending"
            if current is None:
                current = n
        added = n > len(r.steps)
        text = s.text if added else substitute_text(s.text, r, ov, s.id)
        # Flag a step whose wording came out of a swap the cook made, so the UI can badge it.
        subbed = not added and any(
            sub.at_step in (None, s.id) and mentions_ingredient(text, sub.replacement)
            for sub in ov.substitutions)
        steps.append({
            "id": s.id, "n": n, "text": text, "substituted": subbed, "status": status,
            "duration_s": s.duration_s, "appliance": s.appliance, "temp_f": s.temp_f,
            "note": ov.step_notes.get(s.id), "skip_reason": ov.skip_reasons.get(s.id),
            "added": added,
        })
    servings = r.servings * scale
    return {
        "id": r.id, "title": r.title,
        "servings": int(servings) if servings == int(servings) else round(servings, 1),
        "ingredients": ingredients, "steps": steps, "current_step": current,
        "completed_steps": [s.id for s in r.steps if s.id in session.completed_steps],
        "skipped_steps": sorted(ov.skipped_steps),
        "changes": render_changes(r, ov),
        "rendered": render_recipe(r, ov, session.completed_steps),
    }
