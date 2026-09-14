"""Terminal driver.

Phase 1: `cooking-assistant-ai repl --no-llm` and drive the state machine with /tool calls.
Phase 2: `cooking-assistant-ai repl` and just type; the model drives the same tools.
`--sim` freezes the clock so you can jump time with /now and watch timers fire.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cooking_assistant_ai.core.clock import Clock
from cooking_assistant_ai.core.fmt import fmt_time, parse_clock_time, parse_duration
from cooking_assistant_ai.core.render import render_recipe, render_state
from cooking_assistant_ai.core.tools import REGISTRY, dispatch
from cooking_assistant_ai.llm.client import DEFAULT_MODEL, OllamaLLM, ScriptedLLM
from cooking_assistant_ai.llm.llm_orchestrator import Orchestrator, StateChanged, print_sink
from cooking_assistant_ai.model.types import Session
from cooking_assistant_ai.storage.db import Store

HELP = """commands:
  /recipes                 list stored recipes        /load <id|title>   load into session
  /state                   rendered state             /recipe <id>       rendered recipe
  /tools                   list tools                 /tool <name> {json args}
  /plating 7:15 PM         set target plating         /timers            running timers
  /now +10m | /now 7:05 PM (sim only) move the clock  /pro 0.3           proactivity
  /idle                    force an idle tick         /quit
anything else is sent to the model (unless --no-llm)."""


async def _sink(ev) -> None:
    if isinstance(ev, StateChanged):
        return
    await print_sink(ev)


async def _readline(prompt: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    print(prompt, end="", flush=True)
    line = await loop.run_in_executor(None, sys.stdin.readline)
    return None if line == "" else line.rstrip("\n")


async def repl(args: argparse.Namespace) -> None:
    # An explicit flag wins; otherwise fall back to the environment, then the built-in default.
    args.db = args.db or os.environ.get("COOK_DB", "cooking.db")
    args.model = args.model or os.environ.get("COOK_MODEL", DEFAULT_MODEL)
    clock = Clock(datetime.now().replace(second=0, microsecond=0)) if args.sim else Clock()
    store = Store(args.db)
    session = Session(id="repl", started_at=clock.now())
    for ref in args.load or []:
        r = store.find_recipe(ref)
        if r is None:
            print(f"no recipe '{ref}'")
        else:
            session.add_recipe(r)
            print(f"loaded {r.title}")
    if args.no_llm:
        llm = ScriptedLLM()
    else:
        llm = OllamaLLM(model=args.model)
        print(f"model: {args.model} (warming up...)", flush=True)
        try:
            await llm.warm()
        except Exception as e:
            print(f"warning: could not reach Ollama ({e}); turns will fail until it is up")
    orch = Orchestrator(session, llm, store, _sink, clock=clock,
                        idle_interval_s=0 if args.sim else args.idle_interval)
    orch.start()
    print(HELP)
    try:
        while True:
            await orch.wait_idle(timeout=600)
            line = await _readline(f"\n[{fmt_time(clock.now())}] > ")
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not await _command(line, orch, clock, store, session, args):
                    break
                continue
            if args.no_llm:
                print("no LLM in this mode; use /tool <name> {json}")
                continue
            await orch.submit(line)
            await orch.wait_idle(timeout=600)
    finally:
        await orch.stop()
        if hasattr(llm, "aclose"):
            await llm.aclose()  # type: ignore[union-attr]


async def _command(line: str, orch: Orchestrator, clock: Clock, store: Store, session: Session,
                   args: argparse.Namespace) -> bool:
    parts = line.split(" ", 2)
    cmd = parts[0].lower()
    rest = line[len(parts[0]):].strip()
    if cmd in ("/quit", "/exit", "/q"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/recipes":
        for r in store.list_recipes():
            flag = " (loaded)" if r.id in session.recipes else ""
            print(f"  {r.id}  {r.title}  serves {r.servings}{flag}")
    elif cmd == "/load":
        _show(dispatch(orch.ctx, "load_recipe", {"recipe": rest}))
    elif cmd == "/recipe":
        r = session.find_recipe(rest)
        print(render_recipe(r, session.overlays[r.id], session.completed_steps) if r else f"not loaded: {rest}")
    elif cmd == "/state":
        print(render_state(session, clock.now()))
    elif cmd == "/timers":
        _show(dispatch(orch.ctx, "get_timers"))
    elif cmd == "/tools":
        for name, spec in REGISTRY.items():
            props = ", ".join(spec.parameters.get("properties", {}).keys())
            print(f"  {name}({props})")
    elif cmd == "/tool":
        bits = rest.split(" ", 1)
        name = bits[0]
        raw = bits[1].strip() if len(bits) > 1 else "{}"
        try:
            call_args = json.loads(raw) if raw else {}
        except ValueError:
            try:  # allow key=value form: /tool add_task label=rice duration_s=900
                call_args = dict(kv.split("=", 1) for kv in shlex.split(raw))
            except ValueError:
                print("args must be JSON or key=value pairs")
                return True
        _show(dispatch(orch.ctx, name, call_args))
        orch.sync_timers()
    elif cmd == "/plating":
        _show(dispatch(orch.ctx, "set_target_plating", {"time": rest}))
    elif cmd == "/now":
        if not clock.is_simulated:
            print("clock is real; start with --sim to move time")
        elif rest.startswith("+"):
            secs = parse_duration(rest[1:])
            if secs is None:
                print("usage: /now +10m")
            else:
                clock.advance(secs)
                print(f"now {fmt_time(clock.now())}")
        else:
            when = parse_clock_time(rest, clock.now())
            if when is None:
                print("usage: /now 7:05 PM")
            else:
                clock.set(when)
                print(f"now {fmt_time(clock.now())}")
        await asyncio.sleep(0.1)  # let due timers fire
    elif cmd == "/pro":
        orch.set_proactivity(float(rest or 0.5))
        print(f"proactivity {session.proactivity}")
    elif cmd == "/idle":
        from cooking_assistant_ai.model.events import IdleTick
        since = int((clock.now() - (session.last_turn_at or session.started_at)).total_seconds())
        session.proactivity = max(session.proactivity, 0.99)
        orch._last_idle_at = None
        session.last_turn_at = None
        await orch.queue.put(IdleTick(since))
    else:
        print(f"unknown command {cmd}; /help")
    return True


def _show(result) -> None:
    env = result.envelope()
    print(("OK  " + (env.get("message") or "")) if env["ok"] else ("REJECTED  " + env["reason"]))
    print(env["state"])


async def classify_steps(args: argparse.Namespace) -> None:
    """Backfill the model's judgements onto stored recipes: which steps are prep, what each
    ingredient contains, and its plain grocery name.

    All three are semantic questions a word list cannot answer. A verb heuristic cannot tell
    "Tuck the garlic and lemon halves around the thighs" from real prep; no list of meat words
    knows caesar dressing has anchovies in it. Asked once per recipe and stored, so nothing
    pays for it at cook time.

    Every answer carries the name or number it is about, and anything that does not match is
    discarded. An earlier version asked for four parallel arrays indexed by position, and one
    slip produced "Crushed Garlic contains meat, pork, seafood, dairy, egg, honey, alcohol"
    with nothing able to notice.
    """
    from dataclasses import replace as _replace

    from cooking_assistant_ai.llm.factory import build_llm
    from cooking_assistant_ai.storage.db import Store

    CATEGORIES = ["meat", "pork", "seafood", "dairy", "egg", "honey", "alcohol"]
    store = Store(args.db or os.environ.get("COOK_DB", "cooking.db"))
    llm = build_llm()
    schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "text": {"type": "string"},
                               "prep": {"type": "boolean"}},
                "required": ["n", "prep"]}},
            "ingredients": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "name": {"type": "string"},
                    "key": {"type": "string"},
                    "contains": {"type": "array", "items": {"type": "string", "enum": CATEGORIES}},
                    "may_contain": {"type": "array", "items": {"type": "string", "enum": CATEGORIES}},
                },
                "required": ["n", "name", "key", "contains", "may_contain"]}},
        },
        "required": ["steps", "ingredients"],
    }

    for recipe in store.list_recipes():
        judged = (all(s.prep is not None for s in recipe.steps)
                  and all(i.contains is not None for i in recipe.ingredients))
        if not args.all and judged:
            print(f"  {recipe.id} {recipe.title}: already judged")
            continue

        step_lines = "\n".join(f"{n}. {s.text}" for n, s in enumerate(recipe.steps, start=1))
        item_lines = "\n".join(f"{n}. {i.name}" for n, i in enumerate(recipe.ingredients, start=1))
        raw = await llm.complete([{"role": "user", "content":
            f"Recipe: {recipe.title}\n\n"
            "STEPS - repeat each step's number and answer prep: is it preparation done before "
            "anything is on the heat (chopping, rinsing, peeling, measuring, seasoning raw "
            "ingredients, making a marinade)? Cooking, resting, assembling and serving are "
            "not prep. Judge each step whole: 'Tuck the garlic and lemon halves around the "
            "thighs' is cooking despite the word halves.\n\n"
            f"{step_lines}\n\n"
            "INGREDIENTS - repeat each ingredient's number and name exactly as written, then:\n"
            "key: the plain grocery name, lowercase, no amount, preparation or brand. "
            "'Medium Onion (White, Yellow or Brown, Chopped)' and 'onions' are both 'onion'; "
            "'1 19oz can black beans' is 'black beans'. Same item, same key; different items, "
            "different keys - ground coriander seed is not fresh coriander leaf.\n"
            "contains: which categories the ordinary product genuinely contains, empty when "
            "none. Judge the food, not the word: caesar dressing contains seafood, "
            "marshmallows contain meat, parmesan contains dairy and meat. But soy sauce has "
            "no seafood, vinegar is not alcoholic, and a vegetable is just a vegetable - do "
            "not guess.\n"
            "may_contain: categories only SOME brands use, such as anchovy in some gochujang "
            "or fish sauce in some kimchi. These raise a check-the-label warning instead of "
            "banning the dish, so anything brand-dependent belongs here.\n\n"
            f"{item_lines}"}], json_schema=schema)

        try:
            answer = json.loads(raw)
        except ValueError:
            print(f"  {recipe.id} {recipe.title}: unreadable answer, left alone")
            continue

        by_n = {}
        for row in answer.get("steps") or []:
            if isinstance(row, dict) and isinstance(row.get("n"), int):
                by_n[row["n"]] = bool(row.get("prep"))
        new_steps = tuple(_replace(s, prep=by_n[n]) if n in by_n else s
                          for n, s in enumerate(recipe.steps, start=1))

        # Matched on the name the model echoed back, so a shifted or partial answer drops the
        # rows it got wrong instead of relabelling the wrong ingredient.
        answers, mismatched = {}, 0
        for row in answer.get("ingredients") or []:
            if not isinstance(row, dict):
                continue
            n, name = row.get("n"), str(row.get("name", "")).strip().lower()
            if not isinstance(n, int) or not 1 <= n <= len(recipe.ingredients):
                continue
            if recipe.ingredients[n - 1].name.strip().lower() != name:
                mismatched += 1
                continue
            answers[n] = row

        new_items = []
        for n, item in enumerate(recipe.ingredients, start=1):
            row = answers.get(n)
            if row is None:
                new_items.append(item)
                continue
            clean = lambda field: tuple(  # noqa: E731
                str(c).strip().lower() for c in (row.get(field) or [])
                if str(c).strip().lower() in CATEGORIES)
            key = str(row.get("key") or "").strip().lower()
            new_items.append(_replace(item, key=key or item.key,
                                      contains=clean("contains"),
                                      may_contain=clean("may_contain")))

        store.put_recipe(_replace(recipe, steps=new_steps, ingredients=tuple(new_items)))
        marks = "".join("P" if s.prep else "." for s in new_steps)
        found = sorted({c for i in new_items for c in (i.contains or ())})
        maybe = sorted({c for i in new_items for c in (i.may_contain or ())})
        note = f"   contains: {', '.join(found)}" if found else ""
        note += f"   may contain: {', '.join(maybe)}" if maybe else ""
        if mismatched:
            note += f"   ({mismatched} answer(s) did not match an ingredient, skipped)"
        print(f"  {recipe.id} {recipe.title}: {marks}{note}")

    close = getattr(llm, "aclose", None)
    if close:
        await close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cooking-assistant-ai")
    sub = p.add_subparsers(dest="cmd")

    # --model / --db default to None so an explicit flag can be told apart from "not given".
    # Otherwise the argparse default silently overwrites COOK_MODEL / COOK_DB in the
    # environment, and setting those has no effect.
    r = sub.add_parser("repl", help="terminal driver (Phase 1/2)")
    r.add_argument("--no-llm", action="store_true", help="drive tools by hand, no model")
    r.add_argument("--model", default=None, help=f"default: $COOK_MODEL or {DEFAULT_MODEL}")
    r.add_argument("--db", default=None, help="default: $COOK_DB or cooking.db")
    r.add_argument("--sim", action="store_true", help="frozen clock; move it with /now")
    r.add_argument("--load", nargs="*", help="recipe ids/titles to load at start")
    r.add_argument("--idle-interval", type=float, default=60.0)

    s = sub.add_parser("serve", help="FastAPI server (HTTP + websocket)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", default=None, help=f"default: $COOK_MODEL or {DEFAULT_MODEL}")
    s.add_argument("--db", default=None, help="default: $COOK_DB or cooking.db")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--reload", action="store_true", help="restart when a .py file under src/ changes (drops the session)")
    s.add_argument("--https", action="store_true",
                   help="serve TLS with a self-signed certificate; required for the microphone on a tablet")
    s.add_argument("--cert", help="certificate file (default: generated under ~/.cooking-assistant)")
    s.add_argument("--key", help="private key file")

    cs = sub.add_parser("classify-steps",
                        help="ask the model which steps are prep and what each ingredient contains")
    cs.add_argument("--db", default=None)
    cs.add_argument("--all", action="store_true", help="redo recipes that already have it")

    lg = sub.add_parser("log", help="read back a cook: what was said and every tool call")
    lg.add_argument("which", nargs="?", help="session id, or a log filename; default the latest")
    lg.add_argument("--list", action="store_true", help="list the logs and stop")
    lg.add_argument("--summary", action="store_true", help="counts and rejections only")
    lg.add_argument("--no-tools", action="store_true", help="just the conversation")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    args = build_parser().parse_args(argv)
    if args.cmd == "classify-steps":
        asyncio.run(classify_steps(args))
        return

    if args.cmd == "log":
        from cooking_assistant_ai.storage.journal import list_logs, render, summarize

        logs = list_logs()
        if not logs:
            print("no logs yet. They are written to ./logs as you cook (COOK_LOG_DIR moves them).")
            return
        if args.list:
            for path in logs:
                info = summarize(path)
                print(f"  {path.name}  {info['turns']} turns, {info['tool_calls']} tool calls, "
                      f"{info['rejected']} rejected")
            return
        chosen = logs[0]
        if args.which:
            matches = [p for p in logs if args.which in p.name]
            if not matches:
                print(f"no log matching '{args.which}'. Try --list.")
                return
            chosen = matches[0]
        if args.summary:
            info = summarize(chosen)
            print(chosen.name)
            print(f"  {info['first']} -> {info['last']}")
            print(f"  {info['turns']} turns, {info['tool_calls']} tool calls, {info['rejected']} rejected")
            for name, n in sorted(info["by_tool"].items(), key=lambda kv: -kv[1]):
                print(f"    {n:4} {name}")
            for r in info["rejections"][:20]:
                print(f"    REJECTED {r}")
            return
        print(render(chosen, tools=not args.no_tools))
        return

    if args.cmd == "serve":
        import os

        import uvicorn

        if args.model:
            os.environ["COOK_MODEL"] = args.model
        if args.db:
            os.environ["COOK_DB"] = args.db
        if args.no_llm:
            os.environ["COOK_NO_LLM"] = "1"
        # Actually serving, so a fixed name is worth advertising. COOK_MDNS=0 opts out.
        os.environ.setdefault("COOK_MDNS", "1")
        os.environ["COOK_PORT"] = str(args.port)
        os.environ["COOK_HTTPS"] = "1" if (args.https or args.cert) else "0"
        ssl_args = {}
        if args.https or args.cert:
            if args.cert and args.key:
                cert, key = Path(args.cert), Path(args.key)
            else:
                from cooking_assistant_ai.api.mdns import hostname
                from cooking_assistant_ai.api.tls import ensure_cert, lan_addresses

                # The friendly name goes in the certificate too: without it the tablet
                # gets a name mismatch, which is a worse warning than the IP produced.
                cert, key = ensure_cert(extra_hosts=[hostname()])
                lan = lan_addresses()  # best first: the LAN, not a VPN tunnel
                print(f"  Tablet: https://{hostname()}:{args.port}/  (if mDNS resolves there)")
                print(f"TLS on (self-signed certificate at {cert}).")
                for a in lan:
                    print(f"  Tablet: https://{a}:{args.port}/")
                print("  The tablet must accept this certificate once. In Fully Kiosk, turn on the")
                print("  setting that ignores SSL/certificate errors, or install the .crt above on the device.")
            ssl_args = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

        src_dir = str(Path(__file__).resolve().parent)
        if args.reload:
            # uvicorn's own reloader owns the process tree here, so the restart endpoint
            # stays unavailable; file changes are picked up automatically instead.
            uvicorn.run("cooking_assistant_ai.main:app", host=args.host, port=args.port,
                        reload=True, reload_dirs=[src_dir], reload_includes=["*.py"], **ssl_args)
            return

        # Drive the server by hand rather than uvicorn.run(), so /admin/restart has something
        # to stop. Exiting with RESTART_EXIT_CODE tells the supervisor in start.ps1 to
        # relaunch us; any other non-zero code is a crash and gets backed off.
        from cooking_assistant_ai.api import admin

        config = uvicorn.Config("cooking_assistant_ai.main:app", host=args.host, port=args.port,
                                **ssl_args)
        server = uvicorn.Server(config)

        def _stop() -> None:
            server.should_exit = True

        admin.register_stopper(_stop)
        admin.reset()
        server.run()
        if admin.restart_requested():
            print("restart requested; handing back to the supervisor")
            sys.exit(admin.RESTART_EXIT_CODE)
    else:
        if args.cmd is None:
            args = build_parser().parse_args(["repl"] + list(argv or sys.argv[1:]))
        asyncio.run(repl(args))
