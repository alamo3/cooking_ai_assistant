"""HTTP + WebSocket transport (spec section 5). The orchestrator never sees any of this."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cooking_assistant_ai.api import admin
from cooking_assistant_ai.core.fmt import fmt_dur, fmt_time, parse_clock_time
from cooking_assistant_ai.core.render import recipe_view
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.llm.client import DEFAULT_MODEL, LLM, OllamaLLM, ScriptedLLM
from cooking_assistant_ai.llm.factory import DEFAULT_BACKEND, build_llm
from cooking_assistant_ai.llm.openrouter import FallbackLLM, OpenRouterLLM
from cooking_assistant_ai.llm.llm_orchestrator import (
    Notice,
    Orchestrator,
    OutputEvent,
    Speech,
    SpeechEnd,
    SpeechStart,
    StateChanged,
    ToolCalled,
)
from cooking_assistant_ai.model.types import Recipe, Session
from cooking_assistant_ai.speech import STT, TTS, SentenceSplitter, build_stt, build_tts, build_vad
from cooking_assistant_ai.speech.stt import kitchen_prompt
from cooking_assistant_ai.speech.listener import Listener
from cooking_assistant_ai.storage import sessions as session_snapshots
from cooking_assistant_ai.storage.db import Store

log = logging.getLogger(__name__)


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    db_path: str = os.environ.get("COOK_DB", "cooking.db")
    no_llm: bool = os.environ.get("COOK_NO_LLM", "") == "1"
    idle_interval_s: float = float(os.environ.get("COOK_IDLE_INTERVAL", "60"))
    warm_model: bool = os.environ.get("COOK_WARM", "1") == "1"
    stt: str = os.environ.get("COOK_STT", "none")
    tts: str = os.environ.get("COOK_TTS", "none")
    backend: str = os.environ.get("COOK_LLM", DEFAULT_BACKEND)  # cloud | openrouter | ollama
    vad: str = os.environ.get("COOK_VAD", "silero")
    barge_in: str = os.environ.get("COOK_BARGE_IN", "voice")  # voice | transcript | off
    # Session snapshots: how often to write, and how stale a snapshot may be and still be
    # worth resuming. Twelve hours covers "the power went out during dinner"; anything older
    # is last week's cook and only gets in the way.
    autosave_s: float = float(os.environ.get("COOK_AUTOSAVE", "10"))
    session_max_age_s: float = float(os.environ.get("COOK_SESSION_MAX_AGE", str(12 * 3600)))


# --------------------------------------------------------------------------- live session

class LiveSession:
    def __init__(self, session: Session, llm: LLM, store: Store, settings: Settings, tts: TTS,
                 stt: Optional[STT] = None):
        self.session = session
        self.sockets: Set[WebSocket] = set()
        self.tts = tts
        self.stt = stt
        self.settings = settings
        self._splitter = SentenceSplitter()
        self.orchestrator = Orchestrator(session, llm, store, self.sink,
                                         idle_interval_s=settings.idle_interval_s)
        self.listener: Optional[Listener] = None
        self.playing = False  # client-reported playback state, for barge-in and echo guard
        self.last_snapshot: Optional[Dict[str, Any]] = None  # what is already in the database
        self.on_state_change = None  # set by the manager, to snapshot after every tool call

    # -- open microphone ------------------------------------------------------

    def tune_stt(self) -> None:
        """Bias whisper toward tonight's dishes and ingredients. Called whenever the loaded
        recipes change, since that is exactly what the cook is about to say out loud."""
        if self.stt is None or not getattr(self.stt, "available", False):
            return
        try:
            self.stt.prompt = kitchen_prompt(self.session, os.environ.get("COOK_WHISPER_PROMPT", ""))
        except Exception as e:  # a decoding hint is never worth failing a cook over
            log.warning("could not set the STT vocabulary hint: %s", e)

    def start_listening(self) -> Listener:
        self.tune_stt()
        if self.listener is None:
            if self.stt is None or not self.stt.available:
                raise RuntimeError("no STT backend configured")
            vad = build_vad(self.settings.vad)
            if vad is None:
                raise RuntimeError("VAD is disabled (COOK_VAD=none)")
            self.listener = Listener(
                self.stt, vad, on_utterance=self._heard, on_event=self._listener_event,
                is_playing=lambda: self.playing, on_barge_in=self.barge_in,
                barge_in_mode=self.settings.barge_in,
            )
        return self.listener

    def stop_listening(self) -> None:
        self.listener = None

    async def _heard(self, text: str) -> None:
        await self.broadcast({"type": "transcript", "text": text, "final": True})
        await self.orchestrator.submit(text)

    async def _listener_event(self, kind: str, data: Dict[str, Any]) -> None:
        await self.broadcast({"type": kind, **data})

    async def barge_in(self) -> None:
        cancelled = self.orchestrator.barge_in()
        self.playing = False
        await self.broadcast({"type": "stop_playback", "cancelled": cancelled})

    async def broadcast(self, msg: Dict[str, Any]) -> None:
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    async def _speak(self, sentences: List[str], proactive: bool) -> None:
        if not self.tts.available:
            return
        for s in sentences:
            pcm = await self.tts.synthesize(s)
            if pcm:
                await self.broadcast({
                    "type": "audio", "data": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": self.tts.sample_rate, "text": s, "proactive": proactive,
                })

    async def sink(self, ev: OutputEvent) -> None:
        if isinstance(ev, SpeechStart):
            self._splitter = SentenceSplitter()
            await self.broadcast({"type": "speech_start", "proactive": ev.proactive})
        elif isinstance(ev, Speech):
            if self.listener:
                self.listener.note_spoken(ev.text)
            await self.broadcast({"type": "text", "text": ev.text, "proactive": ev.proactive})
            await self._speak(self._splitter.feed(ev.text), ev.proactive)
        elif isinstance(ev, SpeechEnd):
            await self._speak(self._splitter.flush(), ev.proactive)
            await self.broadcast({"type": "speech_end", "proactive": ev.proactive, "text": ev.full_text})
        elif isinstance(ev, ToolCalled):
            await self.broadcast({
                "type": "tool_call", "name": ev.name, "args": ev.args,
                "ok": ev.result.get("ok"), "message": ev.result.get("message") or ev.result.get("reason"),
            })
        elif isinstance(ev, StateChanged):
            # A tool just changed the plan, a timer or the progress. These are exactly the
            # moments a power cut must not undo, so snapshot now rather than waiting for the
            # autosave tick.
            if self.on_state_change:
                self.on_state_change(self.session.id)
            self.tune_stt()
            await self.broadcast({"type": "state", **ev.state})
        elif isinstance(ev, Notice):
            await self.broadcast({"type": "notice", "text": ev.text, "level": ev.level})


class SessionManager:
    def __init__(self, store: Store, llm: LLM, settings: Settings, tts: TTS, stt: Optional[STT] = None):
        self.store = store
        self.llm = llm
        self.settings = settings
        self.tts = tts
        self.stt = stt
        self.sessions: Dict[str, LiveSession] = {}

    def create(self, recipe_ids: Optional[List[str]] = None, target_plating: Optional[str] = None,
               proactivity: Optional[float] = None) -> LiveSession:
        now = datetime.now()
        session = Session(id=uuid.uuid4().hex[:8], started_at=now)
        for ref in recipe_ids or []:
            r = self.store.find_recipe(ref)
            if r is None:
                raise HTTPException(404, f"recipe '{ref}' not found")
            session.add_recipe(r)
        if target_plating:
            parsed = parse_clock_time(target_plating, now)
            if parsed is None:
                raise HTTPException(400, f"bad target_plating '{target_plating}'")
            session.target_plating = parsed
        if proactivity is not None:
            session.proactivity = max(0.0, min(1.0, proactivity))
        live = LiveSession(session, self.llm, self.store, self.settings, self.tts, self.stt)
        live.on_state_change = self.persist
        live.orchestrator.start()
        self.sessions[session.id] = live
        return live

    def get(self, session_id: str) -> LiveSession:
        live = self.sessions.get(session_id)
        if live is None:
            raise HTTPException(404, "no such session")
        return live

    def restore(self, session: Session) -> LiveSession:
        """Bring a snapshot back to life. The orchestrator picks the timers up from there."""
        live = LiveSession(session, self.llm, self.store, self.settings, self.tts, self.stt)
        live.on_state_change = self.persist
        live.orchestrator.start()
        live.orchestrator.sync_timers()
        self.sessions[session.id] = live
        return live

    def persist(self, session_id: Optional[str] = None) -> int:
        """Snapshot sessions worth keeping. Returns how many were actually written."""
        written = 0
        for sid, live in list(self.sessions.items()):
            if session_id and sid != session_id:
                continue
            if not session_snapshots.is_worth_saving(live.session):
                continue
            body = session_snapshots.to_dict(live.session)
            if live.last_snapshot == body:
                continue  # nothing changed since the last write
            try:
                self.store.save_session(sid, body)
                live.last_snapshot = body
                written += 1
            except Exception as e:  # a failed snapshot must never take the cook down
                log.warning("could not snapshot session %s: %s", sid, e)
        return written

    async def delete(self, session_id: str, purge: bool = False) -> None:
        """purge=True for a cook that is genuinely over: drop the snapshot too, so the next
        startup does not helpfully restore a meal that was eaten yesterday."""
        live = self.sessions.pop(session_id, None)
        if live:
            await live.orchestrator.stop()
        if purge:
            self.store.delete_session(session_id)

    async def shutdown(self) -> None:
        for sid in list(self.sessions):
            await self.delete(sid)


# --------------------------------------------------------------------------- request models

class SessionCreate(BaseModel):
    recipe_ids: List[str] = []
    target_plating: Optional[str] = None
    proactivity: Optional[float] = None


class RecipeRef(BaseModel):
    recipe: str


class DietChoice(BaseModel):
    diet: str


class StartCooking(BaseModel):
    recipe_ids: List[str]
    target_plating: Optional[str] = None
    minutes_from_now: Optional[int] = None


class InventoryItem(BaseModel):
    name: str
    amount: float
    unit: Optional[str] = None


class InventoryPatch(BaseModel):
    items: List[InventoryItem] = []
    remove: List[str] = []


_RECIPE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "servings": {"type": "integer"},
        "ingredients": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "amount": {"type": "number"}, "unit": {"type": ["string", "null"]}},
            "required": ["name", "amount"]}},
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "duration_s": {"type": ["integer", "null"]},
            "appliance": {"type": ["string", "null"]}, "temp_f": {"type": ["integer", "null"]}},
            "required": ["text"]}},
    },
    "required": ["title", "servings", "ingredients", "steps"],
}


def _find_jsonld_recipe(html: str) -> Optional[Dict[str, Any]]:
    """Most recipe sites embed schema.org/Recipe as JSON-LD. Return it if present."""
    def walk(node: Any):
        if isinstance(node, dict):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if any(isinstance(x, str) and x.lower() == "recipe" for x in types):
                return node
            for key in ("@graph", "mainEntity", "itemListElement"):
                if key in node:
                    found = walk(node[key])
                    if found:
                        return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
        return None

    for m in re.finditer(r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", html, re.S | re.I):
        try:
            data = json.loads(m.group(1).strip())
        except ValueError:
            continue
        found = walk(data)
        if found:
            return found
    return None


def _jsonld_to_text(node: Dict[str, Any]) -> str:
    def text_of(x: Any) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("text") or x.get("name") or ""
        return ""

    steps: List[str] = []
    for item in node.get("recipeInstructions") or []:
        if isinstance(item, dict) and item.get("@type") == "HowToSection":
            steps.extend(text_of(s) for s in item.get("itemListElement") or [])
        else:
            steps.append(text_of(item))
    steps = [re.sub(r"\s+", " ", s).strip() for s in steps if text_of(s).strip()]
    lines = [f"Title: {node.get('name', '')}"]
    if node.get("recipeYield"):
        y = node["recipeYield"]
        lines.append(f"Yield: {y[0] if isinstance(y, list) else y}")
    for key in ("prepTime", "cookTime", "totalTime"):
        if node.get(key):
            lines.append(f"{key}: {node[key]}")
    lines.append("Ingredients:")
    lines.extend(f"- {re.sub(chr(10), ' ', i).strip()}" for i in node.get("recipeIngredient") or [])
    lines.append("Instructions:")
    lines.extend(f"{n}. {s}" for n, s in enumerate(steps, start=1))
    return re.sub(r"<[^>]+>", "", "\n".join(lines))


async def import_recipe_from_url(url: str, llm: LLM, store: Store) -> Recipe:
    import httpx

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=25) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            })
    except httpx.HTTPError as e:
        raise HTTPException(502, f"could not fetch {url}: {e}")
    if resp.status_code in (401, 403, 429, 503):
        raise HTTPException(422, f"{resp.status_code} from the site: it blocks automated access. "
                                 "Copy the recipe text from your browser and import it with {\"text\": ...} instead.")
    if resp.status_code >= 400:
        raise HTTPException(422, f"{resp.status_code} from the site for {url}")
    html = resp.text
    ld = _find_jsonld_recipe(html)
    if ld:
        source = "STRUCTURED RECIPE DATA:\n" + _jsonld_to_text(ld)[:14000]
    else:
        stripped = re.sub(r"<(script|style|nav|footer|header|aside|form)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", stripped)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200:
            raise HTTPException(422, "page has no readable recipe content (JavaScript-only page?)")
        source = "PAGE TEXT:\n" + text[:14000]
    return await extract_recipe(source, llm, store)


async def import_recipe_from_text(text: str, llm: LLM, store: Store) -> Recipe:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 40:
        raise HTTPException(422, "paste the full recipe text (ingredients and steps)")
    return await extract_recipe("PASTED RECIPE TEXT:\n" + text[:14000], llm, store)


async def extract_recipe(source: str, llm: LLM, store: Store) -> Recipe:
    if isinstance(llm, ScriptedLLM) and not llm.script:
        raise HTTPException(501, "recipe import needs a real LLM")
    prompt = (
        "Extract the recipe below as JSON. Rules: one ingredient entry per line, with amount as a number "
        "(convert fractions like 1/2 to 0.5; use 1 if no amount is given), unit as a short string (g, ml, cup, "
        "tbsp, tsp, oz, lb, clove) or null for countable items; steps in order with the original wording "
        "lightly tidied; appliance as 'oven', 'stovetop:1', 'air_fryer', 'rice_cooker', 'pressure_cooker', "
        "'bread_maker', 'grill', 'microwave' or null; temp_f in Fahrenheit (convert from C) when the step "
        "states a temperature; servings as an integer.\n"
        # Pages rarely time every step, but the planner interleaves dishes using these
        # numbers, so a null makes an imported recipe unschedulable. An estimate is far
        # better than nothing.
        "duration_s: ALWAYS give a number of seconds for EVERY step. Use the time the page states; where it "
        "gives none, estimate how long the step really takes (chopping an onion 120, bringing a pan up to "
        "heat 180, resting meat 600).\n\n" + source
    )
    raw = await llm.complete([{"role": "user", "content": prompt}], json_schema=_RECIPE_SCHEMA)
    try:
        data = json.loads(raw)
    except ValueError:
        raise HTTPException(502, "model did not return valid recipe JSON")
    if not data.get("steps") or not data.get("ingredients"):
        raise HTTPException(422, "could not find ingredients and steps on that page")
    data["id"] = store.next_recipe_id()
    recipe = Recipe.from_dict(data)
    store.put_recipe(recipe)
    return recipe


# --------------------------------------------------------------------------- app factory

def create_app(settings: Optional[Settings] = None, llm: Optional[LLM] = None,
               store: Optional[Store] = None, stt: Optional[STT] = None, tts: Optional[TTS] = None) -> FastAPI:
    settings = settings or Settings()
    store = store or Store(settings.db_path)
    if llm is None:
        llm = ScriptedLLM() if settings.no_llm else build_llm(settings.backend, settings.model)
    stt = stt or build_stt(settings.stt)
    tts = tts or build_tts(settings.tts)
    manager = SessionManager(store, llm, settings, tts, stt)

    def restore_sessions() -> List[str]:
        """Bring back any cook that was interrupted by a crash, a restart or a power cut."""
        restored: List[str] = []
        try:
            store.prune_sessions(settings.session_max_age_s)
            snapshots = store.load_sessions(max_age_s=settings.session_max_age_s)
        except Exception as e:
            log.warning("could not read session snapshots: %s", e)
            return restored
        now = datetime.now()
        for sid, saved_at, body in snapshots:
            if body.get("v") != session_snapshots.SCHEMA_VERSION:
                log.info("dropping session %s: snapshot format %s is no longer readable",
                         sid, body.get("v"))
                store.delete_session(sid)
                continue
            try:
                session = session_snapshots.from_dict(body)
            except Exception as e:
                log.warning("could not restore session %s: %s", sid, e)
                store.delete_session(sid)
                continue
            # Timers that ran out while we were down must not all fire at once on startup.
            late = session_snapshots.expire_timers(session, now)
            if late:
                labels = ", ".join(t.label for t in late)
                gap = fmt_dur(int((now - saved_at).total_seconds()))
                session.notes.append(
                    f"The server was down for about {gap}. These timers finished while it was "
                    f"off and were not announced: {labels}. Check them before carrying on.")
            manager.restore(session)
            restored.append(sid)
            titles = ", ".join(r.title for r in session.recipes.values()) or "no recipes"
            # Printed, not logged: after a power cut this is the first thing worth seeing.
            print(f"Recovered the cook from {fmt_time(saved_at)}: {titles}"
                  + (f" ({len(late)} timer(s) finished while it was down)" if late else ""))
        return restored

    async def autosave() -> None:
        while True:
            await asyncio.sleep(settings.autosave_s)
            try:
                manager.persist()
            except Exception as e:  # never let the snapshot loop die quietly
                log.warning("autosave failed: %s", e)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        app.state.restored_sessions = restore_sessions()
        if settings.warm_model:
            for name, thing in (("llm", llm), ("stt", stt), ("tts", tts)):
                if name == "llm" and not hasattr(llm, "warm"):
                    continue
                try:
                    await asyncio.wait_for(thing.warm(), timeout=180)  # type: ignore[union-attr]
                except Exception as e:  # server may not be up yet; first turn will retry
                    log.warning("%s warm-up failed: %s", name, e)
        saver = asyncio.create_task(autosave()) if settings.autosave_s > 0 else None
        yield
        if saver:
            saver.cancel()
        manager.persist()  # a clean shutdown still snapshots, so a restart loses nothing
        await manager.shutdown()
        await stt.aclose()

    app = FastAPI(title="Cooking Assistant", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.store = store
    app.state.llm = llm
    app.state.stt = stt

    # -- HTTP ---------------------------------------------------------------

    @app.get("/cert", include_in_schema=False)
    async def download_cert() -> FileResponse:
        """The self-signed certificate, so a tablet can trust this server once."""
        from cooking_assistant_ai.api.tls import CERT_NAME, DEFAULT_DIR

        path = DEFAULT_DIR / CERT_NAME
        if not path.exists():
            raise HTTPException(404, "no certificate generated yet; start the server with --https")
        return FileResponse(str(path), media_type="application/x-x509-ca-cert", filename="cooking-assistant.crt")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        info: Dict[str, Any] = {"ok": True, "model": settings.model, "llm": type(llm).__name__,
                                "backend": settings.backend,
                                "stt": stt.available, "tts": tts.available,
                                "vad": settings.vad if stt.available else "none",
                                "sessions": list(manager.sessions)}
        remote = llm.primary if isinstance(llm, FallbackLLM) else llm
        if isinstance(remote, OpenRouterLLM):
            info["model"] = remote.model
            info["usage"] = {"requests": remote.usage.requests,
                             "prompt_tokens": remote.usage.prompt_tokens,
                             "completion_tokens": remote.usage.completion_tokens,
                             "cost_usd": round(remote.usage.cost_usd, 5)}
            if isinstance(llm, FallbackLLM):
                info["fallbacks_to_local"] = llm.fallbacks
                # False means the local model has never been needed, so it holds no VRAM.
                info["local_loaded"] = llm.secondary_loaded
        return info

    @app.get("/admin/status")
    async def admin_status() -> Dict[str, Any]:
        """Enough for the tablet to decide whether a restart is worth offering."""
        changed = admin.changed_files()
        active = [sid for sid, s in manager.sessions.items() if s.session.recipes]
        return {
            "pid": os.getpid(),
            "uptime_s": round(admin.uptime_s(), 1),
            "code_changed": bool(changed),
            "changed_files": changed[:20],
            "changed_count": len(changed),
            "can_restart": admin.can_restart(),
            "sessions": list(manager.sessions),
            "active_sessions": active,
            "model": settings.model,
            "backend": settings.backend,
        }

    @app.post("/admin/restart", status_code=202)
    async def admin_restart(body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask the supervisor for a fresh process. The tablet's websocket reconnects on its own."""
        if not admin.can_restart():
            raise HTTPException(503, "this server was not started by the supervisor "
                                     "(use start.ps1), so it cannot restart itself")
        force = bool((body or {}).get("force"))
        # A restart throws away every cooking session: timers, tasks and progress are all
        # in memory. Never do that to a cook mid-recipe without being told twice.
        active = [sid for sid, s in manager.sessions.items() if s.session.recipes]
        if active and not force:
            raise HTTPException(409, f"{len(active)} cooking session(s) in progress; "
                                     f"restarting loses their timers and progress. "
                                     f"Send force=true to restart anyway.")

        async def _go() -> None:
            await asyncio.sleep(0.25)  # let this response reach the tablet first
            admin.request_restart()

        asyncio.create_task(_go())
        return {"ok": True, "restarting": True, "lost_sessions": active}

    @app.get("/recipes")
    async def list_recipes() -> List[Dict[str, Any]]:
        return [{"id": r.id, "title": r.title, "servings": r.servings, "steps": len(r.steps)}
                for r in store.list_recipes()]

    @app.get("/recipes/{recipe_id}")
    async def get_recipe(recipe_id: str, session: Optional[str] = None) -> Dict[str, Any]:
        r = store.get_recipe(recipe_id)
        if r is None:
            raise HTTPException(404, "no such recipe")
        # The library holds the original. If this cook has changed the recipe for the
        # session (a substitution, a scale), show what they will actually be cooking.
        live = manager.sessions.get(session) if session else None
        if live is not None and recipe_id in live.session.recipes:
            return recipe_view(live.session, live.session.recipes[recipe_id])
        return r.to_dict()

    @app.post("/recipes", status_code=201)
    async def add_recipe(body: Dict[str, Any]) -> Dict[str, Any]:
        if "steps" not in body and ("url" in body or "text" in body):
            if body.get("url"):
                r = await import_recipe_from_url(str(body["url"]), llm, store)
            else:
                r = await import_recipe_from_text(str(body.get("text", "")), llm, store)
            return r.to_dict()
        try:
            body.setdefault("id", store.next_recipe_id())
            r = Recipe.from_dict(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"bad recipe: {e}")
        store.put_recipe(r)
        return r.to_dict()

    @app.delete("/recipes/{recipe_id}", status_code=204)
    async def delete_recipe(recipe_id: str) -> None:
        if not store.delete_recipe(recipe_id):
            raise HTTPException(404, "no such recipe")

    @app.get("/diet")
    async def get_diet() -> Dict[str, Any]:
        from cooking_assistant_ai.core.diet import DIETS, describe

        return {"diet": store.diet, "options": list(DIETS), "description": describe(store.diet)}

    @app.put("/diet")
    async def put_diet(body: DietChoice) -> Dict[str, Any]:
        try:
            diet = store.set_diet(body.diet)
        except ValueError as e:
            raise HTTPException(422, str(e))
        from cooking_assistant_ai.core.diet import describe

        return {"diet": diet, "description": describe(diet)}

    @app.get("/meal-options")
    async def meal_options(meals: Optional[int] = None, scale: float = 1.0) -> Dict[str, Any]:
        """What the pantry supports, for the Recipes screen and for the model's tool."""
        from cooking_assistant_ai.core.mealplan import options_dict

        return options_dict(store, meals, scale)

    @app.get("/inventory")
    async def get_inventory() -> List[Dict[str, Any]]:
        return store.inventory()

    @app.patch("/inventory")
    async def patch_inventory(body: InventoryPatch) -> List[Dict[str, Any]]:
        for item in body.items:
            store.set_stock(item.name, item.amount, item.unit)
        for name in body.remove:
            store.remove_stock(name)
        return store.inventory()

    @app.post("/session", status_code=201)
    async def start_session(body: Optional[SessionCreate] = None) -> Dict[str, Any]:
        body = body or SessionCreate()
        live = manager.create(body.recipe_ids, body.target_plating, body.proactivity)
        return {"session_id": live.session.id, "state": live.orchestrator.state()}

    @app.delete("/session/{session_id}", status_code=204)
    async def end_session(session_id: str) -> None:
        manager.get(session_id)
        await manager.delete(session_id, purge=True)

    @app.get("/session/{session_id}/state")
    async def session_state(session_id: str) -> Dict[str, Any]:
        live = manager.get(session_id)
        return live.orchestrator.state()

    async def _session_tool(live: LiveSession, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a tool on behalf of the UI (same path the model uses) and push the new state."""
        result = dispatch(live.orchestrator.ctx, name, args)
        if not result.ok:
            raise HTTPException(400, result.reason or "rejected")
        live.orchestrator.sync_timers()
        state = live.orchestrator.state()
        await live.broadcast({"type": "tool_call", "name": name, "args": args, "ok": True,
                              "message": result.message, "source": "ui"})
        await live.broadcast({"type": "state", **state})
        return state

    @app.post("/session/{session_id}/recipes")
    async def session_add_recipe(session_id: str, body: RecipeRef) -> Dict[str, Any]:
        return await _session_tool(manager.get(session_id), "load_recipe", {"recipe": body.recipe})

    @app.delete("/session/{session_id}/recipes/{recipe_id}")
    async def session_remove_recipe(session_id: str, recipe_id: str) -> Dict[str, Any]:
        return await _session_tool(manager.get(session_id), "unload_recipe", {"recipe_id": recipe_id})

    @app.post("/session/{session_id}/steps/{step_id}/complete")
    async def session_complete_step(session_id: str, step_id: str) -> Dict[str, Any]:
        return await _session_tool(manager.get(session_id), "mark_complete", {"step_ids": [step_id]})

    @app.post("/session/{session_id}/start")
    async def session_start_cooking(session_id: str, body: StartCooking) -> Dict[str, Any]:
        """Load the chosen recipes, set plating, then have the model build the plan and brief the cook."""
        live = manager.get(session_id)
        ctx = live.orchestrator.ctx
        session = live.session
        wanted = []
        for ref in body.recipe_ids:
            r = store.find_recipe(ref)
            if r is None:
                raise HTTPException(404, f"recipe '{ref}' not found")
            wanted.append(r)
        if not wanted:
            raise HTTPException(400, "choose at least one recipe")
        wanted_ids = {r.id for r in wanted}
        for rid in list(session.recipes):
            if rid not in wanted_ids:
                res = dispatch(ctx, "unload_recipe", {"recipe_id": rid})
                if not res.ok:
                    raise HTTPException(400, res.reason or "cannot remove recipe")
        new_titles = []
        for r in wanted:
            if r.id not in session.recipes:
                dispatch(ctx, "load_recipe", {"recipe": r.id})
                new_titles.append(r.title)
        plating_args: Dict[str, Any] = {}
        if body.minutes_from_now is not None:
            plating_args = {"minutes_from_now": body.minutes_from_now}
        elif body.target_plating:
            plating_args = {"time": body.target_plating}
        if plating_args:
            res = dispatch(ctx, "set_target_plating", plating_args)
            if not res.ok:
                raise HTTPException(400, res.reason or "bad plating time")
        live.orchestrator.sync_timers()
        state = live.orchestrator.state()
        await live.broadcast({"type": "tool_call", "name": "start_cooking", "args": body.model_dump(exclude_none=True),
                              "ok": True, "message": "recipes loaded", "source": "ui"})
        await live.broadcast({"type": "state", **state})

        titles = ", ".join(r.title for r in wanted)
        # Plating is optional and normally unset: batch cooking has no serving deadline, and a
        # deadline forces every task as late as possible and invents conflicts.
        plating = (f"Target plating: {fmt_time(session.target_plating)}."
                   if session.target_plating else
                   "No serving deadline: schedule everything as early as it can run.")
        existing = [t.label for t in session.tasks.values() if t.is_open]
        prompt = (
            f"[SYSTEM] The cook chose these recipes on the tablet: {titles}. {plating}\n"
            + (f"Tasks already planned: {', '.join(existing)}. Extend the plan for the new recipes ({', '.join(new_titles) or 'none'}) without recreating those.\n" if existing else "")
            + "1. Build the complete plan now with add_task, following these rules exactly:\n"
              "   - One task per stretch on one appliance (sear, roast, boil, simmer, air fry). Never one task for a whole recipe.\n"
              "   - Every step_id belongs to at most ONE task. Never repeat a step in two tasks.\n"
              "   - Create no tasks for prep, seasoning, resting or serving steps: leave those out entirely, the cook plan places them for you.\n"
              "   - Chain each dish with after=\"<previous task label>\". Everything starts as soon as it can; do not set must_finish_by.\n"
              "   - appliance=\"stovetop\" picks a free burner; omit appliance and temp_f for anything off the heat.\n"
              "   Do not narrate the tool calls.\n"
              "2. Then brief the cook out loud in five or six short sentences from the PLAN SUMMARY the last tool "
              "result gives you: how long it takes and when to start, which appliances and how many burners, what "
              "to prep first, and the main ingredients to get out. End with the very first thing to do."
        )
        await live.orchestrator.submit_system(prompt)
        return state

    @app.post("/session/{session_id}/prep/{group_id}/complete")
    async def session_complete_prep(session_id: str, group_id: str) -> Dict[str, Any]:
        return await _session_tool(manager.get(session_id), "complete_prep", {"group": group_id})

    @app.delete("/session/{session_id}/timers/{timer_id}")
    async def session_cancel_timer(session_id: str, timer_id: str) -> Dict[str, Any]:
        return await _session_tool(manager.get(session_id), "cancel_timer", {"timer_id": timer_id})

    # -- static tablet app ----------------------------------------------------

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(web_dir), html=True), name="app")

        @app.middleware("http")
        async def _no_cache_app_files(request, call_next):
            # The tablet must always see the current app files; they are tiny, so skip caching.
            response = await call_next(request)
            if request.url.path.startswith("/app"):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/app/")

    # -- WebSocket ------------------------------------------------------------

    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket, session_id: Optional[str] = None) -> None:
        await ws.accept()
        try:
            live = manager.sessions.get(session_id) if session_id else None
            if live is None:
                live = manager.create()
        except HTTPException as e:
            await ws.send_json({"type": "error", "text": e.detail})
            await ws.close()
            return
        live.sockets.add(ws)
        await ws.send_json({"type": "session", "session_id": live.session.id})
        await ws.send_json({"type": "state", **live.orchestrator.state()})
        orch = live.orchestrator
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    await ws.send_json({"type": "error", "text": "messages must be JSON"})
                    continue
                kind = msg.get("type")
                if kind == "text":
                    text = str(msg.get("text", "")).strip()
                    if text:
                        await live.broadcast({"type": "transcript", "text": text, "final": True})
                        await orch.submit(text)
                elif kind == "audio":
                    if not stt.available:
                        await ws.send_json({"type": "error", "text": "no STT backend configured"})
                        continue
                    try:
                        pcm = base64.b64decode(msg.get("data", ""))
                        text = await stt.transcribe(pcm, int(msg.get("sample_rate", 16000)))
                    except Exception as e:
                        await ws.send_json({"type": "error", "text": f"stt failed: {e}"})
                        continue
                    if text:
                        await live.broadcast({"type": "transcript", "text": text, "final": True})
                        await orch.submit(text)
                elif kind == "listen":
                    want = bool(msg.get("on", True))
                    if want:
                        try:
                            live.start_listening()
                        except RuntimeError as e:
                            await ws.send_json({"type": "error", "text": str(e)})
                            await ws.send_json({"type": "listen", "on": False})
                            continue
                    else:
                        live.stop_listening()
                    await live.broadcast({"type": "listen", "on": want})
                elif kind == "audio_chunk":
                    if live.listener is None:
                        continue  # not listening (toggle off, or an old client); ignore quietly
                    try:
                        await live.listener.feed(base64.b64decode(msg.get("data", "")))
                    except Exception as e:
                        log.exception("listener failed")
                        await ws.send_json({"type": "error", "text": f"listening failed: {e}"})
                elif kind == "playback":
                    live.playing = bool(msg.get("playing"))
                elif kind == "barge_in":
                    await live.barge_in()
                elif kind == "set_proactivity":
                    orch.set_proactivity(float(msg.get("value", 0.5)))
                    await ws.send_json({"type": "state", **orch.state()})
                elif kind == "get_state":
                    await ws.send_json({"type": "state", **orch.state()})
                else:
                    await ws.send_json({"type": "error", "text": f"unknown message type '{kind}'"})
        except WebSocketDisconnect:
            pass
        finally:
            live.sockets.discard(ws)

    return app


def default_app() -> FastAPI:
    return create_app()
