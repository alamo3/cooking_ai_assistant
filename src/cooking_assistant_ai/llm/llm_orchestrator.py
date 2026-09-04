"""The turn loop (spec section 4). Transport-agnostic: consumes events from a queue,
emits OutputEvents through a callback. No FastAPI imports here.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from cooking_assistant_ai.core.claims import Claim, correction_prompt, unjustified_claims
from cooking_assistant_ai.core.clock import Clock
from cooking_assistant_ai.core.context import assemble_context
from cooking_assistant_ai.core.fmt import fmt_time
from cooking_assistant_ai.core.render import state_dict
from cooking_assistant_ai.core.scheduler import next_action, resolve
from cooking_assistant_ai.core.tools import ToolContext, dispatch, tool_schemas
from cooking_assistant_ai.llm.client import LLM, ToolCallRequest, extract_text_tool_calls, strip_control_markup
from cooking_assistant_ai.model.events import Event, IdleTick, SystemPrompt, TimerFired, UserUtterance
from cooking_assistant_ai.model.types import Session, Timer, Turn
from cooking_assistant_ai.speech.sentences import SentenceSplitter
from cooking_assistant_ai.storage.db import Store

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- output events

@dataclass
class SpeechStart:
    proactive: bool = False


@dataclass
class Speech:
    text: str
    proactive: bool = False


@dataclass
class SpeechEnd:
    proactive: bool = False
    full_text: str = ""


@dataclass
class ToolCalled:
    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class StateChanged:
    state: Dict[str, Any]


@dataclass
class Notice:
    """Diagnostics the transport may show (errors, suppressed idle turns)."""
    text: str
    level: str = "info"


OutputEvent = Union[SpeechStart, Speech, SpeechEnd, ToolCalled, StateChanged, Notice]
OutputSink = Callable[[OutputEvent], Awaitable[None]]


async def print_sink(ev: OutputEvent) -> None:
    """Phase 1/2 terminal output."""
    if isinstance(ev, SpeechStart):
        print("\n[assistant" + (" (proactive)" if ev.proactive else "") + "] ", end="", flush=True)
    elif isinstance(ev, Speech):
        print(ev.text, end="", flush=True)
    elif isinstance(ev, SpeechEnd):
        print("", flush=True)
    elif isinstance(ev, ToolCalled):
        status = "ok" if ev.result.get("ok") else "REJECTED: " + str(ev.result.get("reason"))
        print(f"\n  [tool] {ev.name}({ev.args}) -> {status}", flush=True)
    elif isinstance(ev, Notice):
        print(f"\n  [{ev.level}] {ev.text}", flush=True)


_STOP = object()


def order_tool_calls(calls: List[ToolCallRequest]) -> List[ToolCallRequest]:
    """Dispatch add_task calls that others reference (after=, before=, must_finish_by=) first.

    Models often emit a whole plan in one response with the references in speaking order
    rather than creation order; reordering saves a rejection-and-retry round. Stable for
    everything else.
    """
    labels: Dict[str, int] = {}
    for i, c in enumerate(calls):
        if c.name == "add_task" and isinstance(c.args.get("label"), str):
            labels[c.args["label"].strip().lower()] = i
    if not labels:
        return calls
    deps: Dict[int, set] = {}
    for i, c in enumerate(calls):
        if c.name != "add_task":
            continue
        for key in ("after", "before", "must_finish_by"):
            ref = c.args.get(key)
            if isinstance(ref, str) and ref.strip().lower() in labels and labels[ref.strip().lower()] != i:
                deps.setdefault(i, set()).add(labels[ref.strip().lower()])
    ordered: List[ToolCallRequest] = []
    placed: set = set()

    def place(i: int, stack: set) -> None:
        if i in placed or i in stack:
            return
        stack.add(i)
        for d in sorted(deps.get(i, ())):
            place(d, stack)
        stack.discard(i)
        placed.add(i)
        ordered.append(calls[i])

    for i in range(len(calls)):
        place(i, set())
    return ordered


class SpeechGate:
    """Streams finished sentences to the output, but holds everything from the first
    sentence that claims an action no tool has backed yet. Held sentences are released
    in order once a justifying tool call succeeds, or dropped/corrected at turn end.
    """

    def __init__(self, session: Session, output: OutputSink, stream: bool = True):
        self.session = session
        self.output = output
        self.stream = stream
        self.splitter = SentenceSplitter(min_chars=1)
        self.tools_ok: set = set()
        self.dispatched = 0  # tool calls attempted this turn, successful or not
        self.produced = False  # the model emitted some text, even if it was withheld
        self.held: List[str] = []
        self.spoken: List[str] = []
        self.started = False

    async def _emit(self, sentence: str) -> None:
        self.spoken.append(sentence)
        if not self.stream:
            return
        if not self.started:
            self.started = True
            await self.output(SpeechStart(proactive=False))
        await self.output(Speech(sentence + " ", proactive=False))

    async def _sentence(self, sentence: str) -> None:
        if self.held or unjustified_claims(sentence, self.tools_ok, self.session):
            self.held.append(sentence)
        else:
            await self._emit(sentence)

    async def feed(self, text: str) -> None:
        if text.strip():
            self.produced = True
        text = strip_control_markup(text)
        for s in self.splitter.feed(text):
            await self._sentence(s)

    async def end_round(self) -> None:
        for s in self.splitter.flush():
            await self._sentence(s)

    async def release(self) -> None:
        """After tool calls: emit held sentences in order until one is still unjustified."""
        while self.held and not unjustified_claims(self.held[0], self.tools_ok, self.session):
            await self._emit(self.held.pop(0))

    def drop_held(self) -> List[Claim]:
        claims: List[Claim] = []
        for s in self.held:
            claims.extend(unjustified_claims(s, self.tools_ok, self.session))
        self.held = []
        return claims

    def spoken_text(self) -> str:
        return " ".join(s.strip() for s in self.spoken if s.strip()).strip()


# --------------------------------------------------------------------------- orchestrator

class Orchestrator:
    def __init__(self, session: Session, llm: LLM, store: Store, output: OutputSink,
                 clock: Optional[Clock] = None, idle_interval_s: float = 60.0,
                 max_tool_rounds: int = 8, push_state: bool = True, batch_grace_s: float = 0.25,
                 max_corrections: int = 1, empty_retries: int = 1):
        self.session = session
        self.llm = llm
        self.output = output
        self.clock = clock or Clock()
        self.ctx = ToolContext(session, self.clock, store)
        self.queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self.tools = tool_schemas()
        self.idle_interval_s = idle_interval_s
        self.max_tool_rounds = max_tool_rounds
        self.push_state = push_state
        self.batch_grace_s = batch_grace_s
        self.max_corrections = max_corrections
        self.empty_retries = empty_retries
        self._timer_tasks: Dict[str, asyncio.Task] = {}
        self._run_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._current_turn: Optional[asyncio.Task] = None
        self._last_idle_at: Optional[datetime] = None
        self._busy = False
        self.turns_completed = 0
        self.session.last_turn_at = self.clock.now()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._run_task = asyncio.create_task(self.run())
        if self.idle_interval_s > 0:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def stop(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        for t in self._timer_tasks.values():
            t.cancel()
        self._timer_tasks.clear()
        await self.queue.put(_STOP)
        if self._run_task:
            try:
                await asyncio.wait_for(self._run_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._run_task.cancel()

    # -- producers ----------------------------------------------------------

    async def submit(self, text: str) -> None:
        await self.queue.put(UserUtterance(text))

    async def submit_system(self, text: str) -> None:
        """Queue an app-originated instruction (e.g. 'plan these recipes and brief the cook')."""
        await self.queue.put(SystemPrompt(text))

    def set_proactivity(self, value: float) -> None:
        self.session.proactivity = max(0.0, min(1.0, float(value)))

    def barge_in(self) -> bool:
        """Abort the in-flight generation. Returns True if something was cancelled."""
        if self._current_turn and not self._current_turn.done():
            self._current_turn.cancel()
            return True
        return False

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Test/REPL helper: block until the queue is drained and no turn is in flight."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self.queue.empty() and not self._busy:
                return
            await asyncio.sleep(0.01)
        raise asyncio.TimeoutError("orchestrator did not go idle")

    def state(self) -> Dict[str, Any]:
        return state_dict(self.session, self.clock.now())

    # -- main loop ----------------------------------------------------------

    async def run(self) -> None:
        self.sync_timers()
        while True:
            event = await self.queue.get()
            if event is _STOP:
                return
            self._busy = True
            try:
                if isinstance(event, TimerFired) and self.batch_grace_s > 0:
                    await asyncio.sleep(self.batch_grace_s)  # let timers that fire together arrive
                batch = [event]
                while not self.queue.empty():  # batch anything that arrived together
                    nxt = self.queue.get_nowait()
                    if nxt is _STOP:
                        await self.queue.put(_STOP)
                        break
                    batch.append(nxt)
                for events in self._group(batch):
                    synthesized = self.synthesize_turn(events)
                    if synthesized is None:
                        continue
                    prompt, proactive = synthesized
                    self._current_turn = asyncio.create_task(self._safe_turn(prompt, proactive))
                    try:
                        await self._current_turn
                    except asyncio.CancelledError:
                        pass
                    finally:
                        self._current_turn = None
            finally:
                self._busy = False

    @staticmethod
    def _group(batch: List[Event]) -> List[List[Event]]:
        """Consecutive TimerFired events merge into one turn; idle ticks are dropped when anything else is queued."""
        has_other = any(not isinstance(e, IdleTick) for e in batch)
        groups: List[List[Event]] = []
        for e in batch:
            if isinstance(e, IdleTick) and has_other:
                continue
            if isinstance(e, TimerFired) and groups and isinstance(groups[-1][-1], TimerFired):
                groups[-1].append(e)
            else:
                groups.append([e])
        return groups

    # -- turn synthesis (spec 4.3) -----------------------------------------

    def synthesize_turn(self, events: List[Event]) -> Optional[Tuple[str, bool]]:
        first = events[0]
        if isinstance(first, UserUtterance):
            return first.text, False
        if isinstance(first, SystemPrompt):
            text = first.text if first.text.startswith("[SYSTEM]") else "[SYSTEM] " + first.text
            return text, False
        if isinstance(first, TimerFired):
            lines = []
            for e in events:
                assert isinstance(e, TimerFired)
                t = e.timer
                hint = f' Hint set at creation: "{t.on_complete_hint}".' if t.on_complete_hint else ""
                lines.append(f'[SYSTEM] Timer "{t.label}" completed at {fmt_time(t.end_at)}.{hint}')
            lines.append("Tell the cook what to do now." if len(events) == 1
                         else "Tell the cook what to do now, covering all of these in one breath.")
            return "\n".join(lines), True
        if isinstance(first, IdleTick):
            if not self._idle_allowed():
                return None
            now = self.clock.now()
            resolve(self.session, now)
            nxt = next_action(self.session, now)
            nxt_txt = f"Next action is at {fmt_time(nxt[0])} - {nxt[1]}." if nxt else "Nothing is scheduled."
            mins = max(1, first.since_last_turn_s // 60)
            return (f"[SYSTEM] {mins} minute(s) since last exchange. {nxt_txt} "
                    "Say something useful if warranted, otherwise respond with exactly NOTHING."), True
        return None

    # -- proactivity gate ---------------------------------------------------

    def _idle_allowed(self) -> bool:
        p = self.session.proactivity
        if p <= 0.05:
            return False
        if not self.session.open_tasks() and not self.session.running_timers():
            return False
        now = self.clock.now()
        min_gap = 60 + (1.0 - p) * 540  # 1.0 -> 1 min, 0.5 -> ~5.5 min, 0.1 -> ~9.5 min
        last = self.session.last_turn_at or self.session.started_at
        if (now - last).total_seconds() < min_gap:
            return False
        if self._last_idle_at and (now - self._last_idle_at).total_seconds() < min_gap:
            return False
        self._last_idle_at = now
        return True

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(self.idle_interval_s)
            last = self.session.last_turn_at or self.session.started_at
            since = int((self.clock.now() - last).total_seconds())
            if self.queue.empty():
                await self.queue.put(IdleTick(since))

    # -- the turn -----------------------------------------------------------

    async def _safe_turn(self, prompt: str, proactive: bool) -> None:
        try:
            await self.handle_turn(prompt, proactive)
        except asyncio.CancelledError:
            await self.output(SpeechEnd(proactive=proactive))
            raise
        except Exception as e:  # keep the loop alive no matter what the model does
            log.exception("turn failed")
            await self.output(Notice(f"turn failed: {e}", level="error"))

    async def handle_turn(self, prompt: str, proactive: bool = False) -> str:
        session = self.session
        now = self.clock.now()
        gate = SpeechGate(session, self.output, stream=not proactive)
        for attempt in range(self.empty_retries + 1):
            gate = await self._generate(prompt, proactive, now)
            # Every model tested sometimes returns nothing at all: no text, no tool call,
            # in a second or two. One retry recovers it. Text that was generated and then
            # withheld as an unbacked claim is not "nothing", and a proactive turn is
            # allowed to be silent (that is how "NOTHING" is expressed), so neither retries.
            if proactive or gate.produced or gate.dispatched:
                break
            if attempt < self.empty_retries:
                await self.output(Notice("model returned nothing; retrying the turn", level="warning"))

        full = gate.spoken_text()
        if proactive:
            if not full or full.upper().rstrip(".!") == "NOTHING" or full.upper().startswith("NOTHING"):
                await self.output(Notice("idle turn suppressed", level="debug"))
                return ""
            await self.output(SpeechStart(proactive=True))
            await self.output(Speech(full, proactive=True))
        await self.output(SpeechEnd(proactive=proactive, full_text=full))

        # The transcript is written after generating, so assemble_context never sees this
        # turn's own prompt twice.
        session.transcript.append(Turn(role="user", text=prompt, at=now))
        if full:
            session.transcript.append(Turn(role="assistant", text=full, at=self.clock.now()))
        session.last_turn_at = self.clock.now()
        self.turns_completed += 1
        return full

    async def _generate(self, prompt: str, proactive: bool, now: datetime) -> "SpeechGate":
        """One full attempt: context, tool rounds, claim correction. Returns its SpeechGate."""
        session = self.session
        messages = assemble_context(session, prompt, now)
        gate = SpeechGate(session, self.output, stream=not proactive)
        corrections_left = self.max_corrections
        for _round in range(self.max_tool_rounds + 1):
            raw = ""
            fed = 0
            calls: List[ToolCallRequest] = []
            async for chunk in self.llm.stream(messages, self.tools):
                if chunk.tool_calls:
                    calls.extend(chunk.tool_calls)
                if chunk.text:
                    raw += chunk.text
                    pending = raw[fed:]
                    if "<tool_call" in raw or pending.rstrip().endswith("<"):
                        continue  # a text-embedded tool call may be forming; don't speak it
                    await gate.feed(pending)
                    fed = len(raw)
            text = raw
            if "<tool_call" in raw:
                text, embedded = extract_text_tool_calls(raw)
                calls.extend(embedded)
                await gate.feed(extract_text_tool_calls(raw[fed:])[0])
            elif fed < len(raw):
                await gate.feed(raw[fed:])
            await gate.end_round()

            if calls and _round < self.max_tool_rounds:
                calls = order_tool_calls(calls)
                messages.append({
                    "role": "assistant", "content": text,
                    "tool_calls": [{"function": {"name": c.name, "arguments": c.args}} for c in calls],
                })
                for c in calls:
                    result = dispatch(self.ctx, c.name, c.args)
                    env = result.envelope()
                    gate.dispatched += 1
                    if result.ok:
                        gate.tools_ok.add(c.name)
                    await self.output(ToolCalled(c.name, c.args, env))
                    messages.append({"role": "tool", "content": result.to_json(), "tool_name": c.name})
                self.sync_timers()
                if self.push_state:
                    await self.output(StateChanged(self.state()))
                await gate.release()
                continue

            # Final text for this turn: anything still held is an unbacked claim.
            await gate.release()
            if gate.held and corrections_left > 0 and _round < self.max_tool_rounds:
                corrections_left -= 1
                claims = gate.drop_held()
                await self.output(Notice("unbacked claim, asking the model to correct: " + "; ".join(c.describe() for c in claims), level="warning"))
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": correction_prompt(claims)})
                continue
            if gate.held:
                claims = gate.drop_held()
                await self.output(Notice("dropped unbacked claim: " + "; ".join(c.describe() for c in claims), level="warning"))
            break
        return gate

    # -- timers -------------------------------------------------------------

    def sync_timers(self) -> None:
        """Reconcile asyncio timer tasks with session.timers after any tool call."""
        for timer in list(self.session.timers.values()):
            running = timer.status == "running"
            if running and timer.id not in self._timer_tasks:
                self._timer_tasks[timer.id] = asyncio.create_task(self._run_timer(timer))
            elif not running and timer.id in self._timer_tasks:
                self._timer_tasks.pop(timer.id).cancel()

    async def _run_timer(self, timer: Timer) -> None:
        poll = 0.02 if self.clock.is_simulated else None
        while True:
            delay = (timer.end_at - self.clock.now()).total_seconds()
            if delay <= 0:
                break
            await asyncio.sleep(delay if poll is None else min(delay, poll))
        if timer.status == "running":
            timer.status = "fired"
            self._timer_tasks.pop(timer.id, None)
            await self.queue.put(TimerFired(timer))
