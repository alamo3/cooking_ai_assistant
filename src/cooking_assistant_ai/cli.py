"""Terminal driver.

Phase 1: `cooking-assistant-ai repl --no-llm` and drive the state machine with /tool calls.
Phase 2: `cooking-assistant-ai repl` and just type; the model drives the same tools.
`--sim` freezes the clock so you can jump time with /now and watch timers fire.
"""
from __future__ import annotations

import argparse
import asyncio
import json
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cooking-assistant-ai")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("repl", help="terminal driver (Phase 1/2)")
    r.add_argument("--no-llm", action="store_true", help="drive tools by hand, no model")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--db", default="cooking.db")
    r.add_argument("--sim", action="store_true", help="frozen clock; move it with /now")
    r.add_argument("--load", nargs="*", help="recipe ids/titles to load at start")
    r.add_argument("--idle-interval", type=float, default=60.0)

    s = sub.add_parser("serve", help="FastAPI server (HTTP + websocket)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--db", default="cooking.db")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--reload", action="store_true", help="restart when a .py file under src/ changes (drops the session)")
    s.add_argument("--https", action="store_true",
                   help="serve TLS with a self-signed certificate; required for the microphone on a tablet")
    s.add_argument("--cert", help="certificate file (default: generated under ~/.cooking-assistant)")
    s.add_argument("--key", help="private key file")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        import os

        import uvicorn

        os.environ["COOK_MODEL"] = args.model
        os.environ["COOK_DB"] = args.db
        if args.no_llm:
            os.environ["COOK_NO_LLM"] = "1"
        ssl_args = {}
        if args.https or args.cert:
            if args.cert and args.key:
                cert, key = Path(args.cert), Path(args.key)
            else:
                from cooking_assistant_ai.api.tls import ensure_cert, local_addresses

                cert, key = ensure_cert()
                lan = [a for a in local_addresses() if a != "127.0.0.1"]
                print(f"TLS on (self-signed certificate at {cert}).")
                for a in lan:
                    print(f"  Tablet: https://{a}:{args.port}/")
                print("  The tablet must accept this certificate once. In Fully Kiosk, turn on the")
                print("  setting that ignores SSL/certificate errors, or install the .crt above on the device.")
            ssl_args = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

        src_dir = str(Path(__file__).resolve().parent)
        uvicorn.run("cooking_assistant_ai.main:app", host=args.host, port=args.port,
                    reload=args.reload, reload_dirs=[src_dir] if args.reload else None,
                    reload_includes=["*.py"] if args.reload else None, **ssl_args)
    else:
        if args.cmd is None:
            args = build_parser().parse_args(["repl"] + list(argv or sys.argv[1:]))
        asyncio.run(repl(args))
