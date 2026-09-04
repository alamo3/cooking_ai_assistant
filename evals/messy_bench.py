"""The messy benchmark: what happens when cooking does not go to plan.

plan_bench.py measures whether a model can build a correct plan. That test saturated: every
capable model scores full marks, so it can no longer tell models apart. This one measures the
part that actually makes an assistant worth talking to, and it is deliberately harder to pass.

Every scenario starts from the same deterministically-built state (no model involved in
setup), says one awkward thing, and is scored on an objective consequence: state that must
change, or a reply that must ask rather than guess. Each runs in a fresh session.

usage: python evals/messy_bench.py <model> [repeats]
       COOK_LLM=openrouter COOK_OPENROUTER_MODEL=... python evals/messy_bench.py <model>
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from cooking_assistant_ai.core.clock import Clock
from cooking_assistant_ai.core.tools import ToolContext, dispatch
from cooking_assistant_ai.llm.factory import build_llm
from cooking_assistant_ai.llm.llm_orchestrator import Notice, Orchestrator, SpeechEnd, ToolCalled
from cooking_assistant_ai.model.types import Session
from cooking_assistant_ai.storage.db import Store

Result = Tuple[bool, str]


def setup_meal(ctx: ToolContext) -> None:
    """A chicken-and-rice meal, mid-cook: chicken roasting, rice not started."""
    for rid in ("r001", "r002"):
        dispatch(ctx, "load_recipe", {"recipe": rid})
    dispatch(ctx, "set_target_plating", {"minutes_from_now": 60})
    dispatch(ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                               "step_ids": ["r001-s5"], "appliance": "oven", "temp_f": 425,
                               "must_finish_by": "plating"})
    dispatch(ctx, "add_task", {"label": "rice", "recipe_id": "r002",
                               "step_ids": ["r002-s2", "r002-s3"], "appliance": "stovetop",
                               "must_finish_by": "plating"})
    dispatch(ctx, "start_task", {"task_id": "chicken roast"})
    dispatch(ctx, "set_timer", {"label": "chicken roasting", "duration_s": 1500,
                                "task_id": "chicken roast",
                                "on_complete_hint": "rest it for 10 minutes"})


def two_timers(ctx: ToolContext) -> None:
    setup_meal(ctx)
    dispatch(ctx, "start_task", {"task_id": "rice"})
    dispatch(ctx, "set_timer", {"label": "rice simmering", "duration_s": 900, "task_id": "rice"})


@dataclass
class Scenario:
    name: str
    setup: Callable[[ToolContext], None]
    say: str
    check: Callable[[Session, str, List[str], ToolContext], Result]
    why: str


def _asks(reply: str) -> bool:
    return "?" in reply


SCENARIOS: List[Scenario] = [
    Scenario(
        "eat_earlier", setup_meal,
        "Change of plan, we need to eat twenty five minutes earlier than we said.",
        lambda s, r, t, c: (
            (s.target_plating is not None
             and 15 * 60 <= (c.session_plating0 - s.target_plating).total_seconds() <= 35 * 60),
            f"plating now {s.target_plating:%H:%M} (was {c.session_plating0:%H:%M})"),
        "a hard constraint moved; the whole plan depends on it",
    ),
    Scenario(
        "runs_long", setup_meal,
        "The chicken isn't browning, it needs about fifteen more minutes in the oven.",
        lambda s, r, t, c: (
            # a NEW timer, a longer task, or an explicit reschedule. The clock is frozen, so
            # "created after the turn started" cannot identify new timers: compare ids.
            (any(x.duration_s > 1500 for x in s.tasks.values() if x.label.startswith("chicken"))
             or bool(set(s.timers) - c.timers0)
             or "replan" in t or "move_task" in t),
            f"tools={t} new_timers={sorted(set(s.timers) - c.timers0)}"),
        "reality diverged from the plan; the timeline must absorb it",
    ),
    Scenario(
        "out_of_butter", setup_meal,
        "I've run out of butter, I'm going to use olive oil instead.",
        lambda s, r, t, c: (
            any(sub.replacement.lower().startswith("olive")
                for ov in s.overlays.values() for sub in ov.substitutions),
            f"substitutions={[(x.ingredient_id, x.replacement) for ov in s.overlays.values() for x in ov.substitutions]}"),
        "a change to the recipe the cook must not have to repeat later",
    ),
    Scenario(
        "skip_resting", setup_meal,
        "We're in a real hurry, let's skip resting the chicken.",
        lambda s, r, t, c: (
            ("r001-s6" in s.overlays["r001"].skipped_steps or "r001-s6" in s.completed_steps),
            f"skipped={sorted(s.overlays['r001'].skipped_steps)}"),
        "an instruction to drop a step, which changes the plan",
    ),
    Scenario(
        "ambiguous_that", two_timers,
        "How much longer on that one?",
        lambda s, r, t, c: (
            _asks(r) or ("chicken" in r.lower() and "rice" in r.lower()),
            f"reply={r[:110]!r}"),
        "two timers are running: guessing is worse than asking",
    ),
    Scenario(
        "false_progress", setup_meal,
        "The rice is done, what's next?",
        lambda s, r, t, c: (
            bool({"r002-s2", "r002-s3"} & s.completed_steps) or _asks(r),
            f"rice steps done={sorted({'r002-s2','r002-s3'} & s.completed_steps)} reply={r[:90]!r}"),
        "the cook contradicts the state: record it or question it, never ignore it",
    ),
    Scenario(
        "dropped_garlic", setup_meal,
        "I just dropped half the garlic on the floor.",
        lambda s, r, t, c: (
            "garlic" in r.lower() and len(r.split()) >= 5,
            f"reply={r[:110]!r}"),
        "a mishap with no tool for it: it still needs a useful answer",
    ),
    Scenario(
        "vegetarian_guest", setup_meal,
        "One of the guests just told me they're vegetarian.",
        lambda s, r, t, c: (
            "remember" in t or any(w in r.lower() for w in ("chicken", "vegetarian", "meat")),
            f"tools={t} reply={r[:90]!r}"),
        "a new constraint that conflicts with the meal in the oven",
    ),
]


async def run_scenario(sc: Scenario, model: str) -> Tuple[bool, str, float, str]:
    clock = Clock(datetime.now().replace(second=0, microsecond=0))
    store = Store(":memory:")
    session = Session(id=sc.name, started_at=clock.now())
    llm = build_llm(model=model)
    spoken: List[str] = []
    tools: List[str] = []
    notices: List[str] = []

    async def sink(ev):
        if isinstance(ev, ToolCalled):
            if ev.result.get("ok"):
                tools.append(ev.name)
        elif isinstance(ev, Notice):
            notices.append(ev.text)
        elif isinstance(ev, SpeechEnd) and ev.full_text:
            spoken.append(ev.full_text)

    orch = Orchestrator(session, llm, store, sink, clock=clock, idle_interval_s=0)
    orch.start()
    sc.setup(orch.ctx)
    orch.ctx.session_plating0 = session.target_plating  # type: ignore[attr-defined]
    orch.ctx.timers0 = set(session.timers)               # type: ignore[attr-defined]
    orch.ctx.tasks0 = set(session.tasks)                 # type: ignore[attr-defined]
    t0 = time.time()
    await orch.submit(sc.say)
    await orch.wait_idle(timeout=600)
    elapsed = time.time() - t0
    reply = " ".join(spoken)
    # A model that errored produced no turn at all: never score that as pass or fail.
    failed = [n for n in notices if "turn failed" in n]
    if failed and not reply and not tools:
        await orch.stop()
        close = getattr(llm, "aclose", None)
        if close:
            await close()
        return None, failed[0][:140], time.time() - t0, ""  # type: ignore[return-value]
    try:
        ok, detail = sc.check(session, reply, tools, orch.ctx)
    except Exception as e:  # a check that blows up is a failure, not a crash
        ok, detail = False, f"check error: {e}"
    await orch.stop()
    close = getattr(llm, "aclose", None)
    if close:
        await close()
    return ok, detail, elapsed, reply


async def main() -> None:
    model = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    passed = 0
    total = 0
    errors = 0
    for rep in range(repeats):
        for sc in SCENARIOS:
            ok, detail, elapsed, reply = await run_scenario(sc, model)
            if ok is None:
                errors += 1
                print(f"  [ERR ] {sc.name:<18} {elapsed:5.1f}s  {detail[:120]}")
                continue
            total += 1
            passed += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {sc.name:<18} {elapsed:5.1f}s  {detail[:120]}")
            if not ok:
                print(f"         why it matters: {sc.why}")
                print(f"         said: {reply[:150]!r}")
    print(f"\n==== {model}: {passed}/{total} messy scenarios passed"
          + (f", {errors} errored" if errors else ""))


asyncio.run(main())
