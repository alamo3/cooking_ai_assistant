"""Grade the 'Start cooking' planning turn for a model, over N trials.

usage: plan_bench.py <model> [trials]
Scores each trial on: tasks created, tool calls accepted, steps planned twice,
whether every recipe got at least one task, whether a briefing was spoken, and time.
"""
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from cooking_assistant_ai.api.app import Settings, create_app
from cooking_assistant_ai.core.plan import render_cook_plan
from cooking_assistant_ai.core.render import render_timeline
from cooking_assistant_ai.storage.db import Store

MODEL = sys.argv[1]
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
results = []

for trial in range(1, TRIALS + 1):
    app = create_app(Settings(model=MODEL, idle_interval_s=0, warm_model=False), store=Store(":memory:"))
    with TestClient(app) as c:
        sid = c.post("/session", json={}).json()["session_id"]
        with c.websocket_connect(f"/ws/session?session_id={sid}") as ws:
            ws.receive_json(); ws.receive_json()
            t0 = time.time()
            # No serving deadline: batch cooking has none and the app no longer sets one.
            c.post(f"/session/{sid}/start", json={"recipe_ids": ["r001", "r002"]})
            calls = ok = 0
            briefing = ""
            print(f"\n===== trial {trial}")
            while True:
                m = ws.receive_json()
                if m["type"] == "tool_call" and m.get("source") != "ui":
                    calls += 1
                    ok += 1 if m["ok"] else 0
                    flag = "ok" if m["ok"] else "REJECTED: " + str(m.get("message"))[:70]
                    print(f"  [{time.time()-t0:5.1f}s] {m['name']}({str(m['args'])[:150]}) -> {flag}")
                elif m["type"] == "notice":
                    print(f"  [{time.time()-t0:5.1f}s] notice: {m['text'][:110]}")
                elif m["type"] == "speech_end":
                    briefing = m.get("text") or ""
                    break
        live = app.state.manager.sessions[sid]
        session = live.session
        now = live.orchestrator.clock.now()
        elapsed = time.time() - t0
        dup = [s for s in set(x for t in session.tasks.values() for x in t.step_ids)
               if sum(1 for t in session.tasks.values() if s in t.step_ids) > 1]
        covered = {t.recipe_id for t in session.tasks.values()}
        both = covered >= {"r001", "r002"}

        # Ordered: within a recipe, a task covering later steps must not start before one
        # covering earlier steps (searing and roasting the same chicken at once is wrong).
        ordered = True
        for rid in covered:
            rec = session.recipes.get(rid)
            if not rec:
                continue
            ts = [t for t in session.tasks.values() if t.recipe_id == rid and t.step_ids and t.start_at]
            ts.sort(key=lambda t: min(rec.step_index(s) for s in t.step_ids if rec.step_index(s) > 0) or 0)
            starts = [t.start_at for t in ts]
            ordered = ordered and starts == sorted(starts) and len(set(starts)) == len(starts)

        # Prompt: with no deadline everything should start as soon as it can, so the first
        # task begins now rather than being parked in the future. (When a serving time is
        # set, the meal should instead land within 20 minutes of it.)
        starts = [t.start_at for t in session.tasks.values() if t.start_at]
        if session.target_plating is not None:
            ends = [t.end_at for t in session.tasks.values() if t.end_at]
            timed = bool(ends) and abs((max(ends) - session.target_plating).total_seconds()) <= 20 * 60
        else:
            timed = bool(starts) and (min(starts) - now).total_seconds() <= 5 * 60
        print(f"\n  BRIEFING: {briefing[:260]}")
        if trial == 1:
            print(render_timeline(session, now))
            print(render_cook_plan(session, now, upcoming=5))
        results.append({"tasks": len(session.tasks), "ok": ok, "calls": calls, "dup": len(dup),
                        "both": both, "ordered": ordered, "timed": timed,
                        "brief": len(briefing.split()) >= 8, "s": elapsed})
        print(f"  trial {trial}: {len(session.tasks)} tasks, {ok}/{calls} ok, {len(dup)} dup, "
              f"both={both}, ordered={ordered}, timed={timed}, "
              f"briefed={len(briefing.split()) >= 8}, {elapsed:.0f}s")

good = [r for r in results if r["tasks"] >= 3 and r["dup"] == 0 and r["both"]
        and r["ordered"] and r["timed"] and r["brief"]]
print(f"\n==== {MODEL}: {len(good)}/{TRIALS} clean trials")
print(f"     tasks {[r['tasks'] for r in results]}  dup {[r['dup'] for r in results]}  "
      f"both {[r['both'] for r in results]}  ordered {[r['ordered'] for r in results]}  "
      f"timed {[r['timed'] for r in results]}  briefed {[r['brief'] for r in results]}")
print(f"     median {statistics.median(r['s'] for r in results):.0f}s")
