# Cooking Assistant — API Specification

Backend spec for a local, conversational cooking assistant. FastAPI orchestrator, Ollama-hosted LLM, deterministic state.

**Design invariants** (everything below follows from these):

1. All state mutation happens through tool calls. The model never holds authoritative state.
2. Tools accept *intent*, not computed times. Code resolves intent to wall-clock.
3. Every tool call returns the full recomputed state slice it touched.
4. Invalid operations are rejected with a reason, never accepted-and-flagged.
5. One renderer produces text blocks for both context assembly and tool results.
6. The orchestrator has no knowledge of transport. It consumes events from a queue.

---

## 1. Data models

### 1.1 Recipe

Immutable base. Loaded from storage, never mutated.

```python
@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    amount: float
    unit: str | None          # None for countable items

@dataclass(frozen=True)
class Step:
    id: str
    text: str
    duration_s: int | None     # None if not time-bound
    appliance: str | None      # "oven" | "stovetop:1" | "air_fryer" | None
    temp_f: int | None
    ingredient_ids: list[str]

@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    servings: int
    ingredients: list[Ingredient]
    steps: list[Step]
```

### 1.2 Overlay

Mutable diff applied over the base at render time. This is what makes "modify the recipe" real.

```python
@dataclass
class Substitution:
    ingredient_id: str
    replacement: str
    note: str | None
    at_step: str | None

@dataclass
class Overlay:
    recipe_id: str
    scale_factor: float = 1.0
    substitutions: list[Substitution] = field(default_factory=list)
    step_notes: dict[str, str] = field(default_factory=dict)   # step_id -> note
    skipped_steps: set[str] = field(default_factory=set)
    added_steps: list[Step] = field(default_factory=list)
```

**Rendering rule:** `render_recipe(base, overlay)` produces the text the model sees. The model never sees the base and overlay separately — it sees one coherent recipe with changes already applied, plus a short "changes made" list so it knows what was altered.

### 1.3 Task (scheduler unit)

```python
@dataclass
class Task:
    id: str
    label: str                 # "rice", "chicken roast"
    recipe_id: str
    step_ids: list[str]
    appliance: str | None
    temp_f: int | None
    duration_s: int
    start_at: datetime | None  # resolved by code, never set by model
    end_at: datetime | None
    status: Literal["pending", "active", "complete", "skipped"]
    depends_on: list[str] = field(default_factory=list)
```

### 1.4 Timer

```python
@dataclass
class Timer:
    id: str
    label: str                 # "chicken resting" — required, never generic
    task_id: str | None
    step_id: str | None
    end_at: datetime
    on_complete_hint: str | None   # "pull it out and tent with foil"
    status: Literal["running", "fired", "cancelled"]
```

`on_complete_hint` is what makes timer completion useful rather than a beep. The model sets it when creating the timer, and it's fed back in the synthesized turn when the timer fires.

### 1.5 Session

```python
@dataclass
class Session:
    id: str
    started_at: datetime
    recipes: dict[str, Recipe]
    overlays: dict[str, Overlay]
    tasks: dict[str, Task]
    timers: dict[str, Timer]
    completed_steps: set[str]
    notes: list[str]                  # from remember()
    target_plating: datetime | None
    transcript: list[Turn]
    proactivity: float = 0.5          # 0.0 silent, 1.0 chatty
```

In memory only. One session at a time, lives ~an hour. Recipes and inventory persist to SQLite; session state does not.

---

## 2. Tool specification

All tools are exposed to the model via Ollama's tool API. All return JSON.

### 2.1 Universal return envelope

```json
{
  "ok": true,
  "state": "<rendered text block for the affected domain>",
  "message": "optional human-readable note"
}
```

On rejection:

```json
{
  "ok": false,
  "reason": "oven is at 425°F for chicken until 7:15 PM; brussels need 375°F",
  "state": "<current state, unchanged>"
}
```

**Rejection reasons are prompts.** Write them with specifics — appliance, temperature, time window, conflicting task. A vague rejection produces a confused retry loop.

### 2.2 Planning tools

Used before cooking starts. Latency is not critical here; correctness is.

---

**`add_task`**

```json
{
  "label": "rice",
  "recipe_id": "r_002",
  "step_ids": ["s_1", "s_2"],
  "appliance": "stovetop:2",
  "temp_f": null,
  "duration_s": 900,
  "after": "chicken_sear",      // optional, task label or id
  "before": null,               // optional
  "must_finish_by": "plating"   // optional: "plating" | task ref | null
}
```

Note the absence of `start_at`. The model expresses ordering constraints; the resolver computes times.

Rejects on: appliance conflict in the resolved window, temperature conflict, unsatisfiable constraint, end time past `target_plating`.

Returns: full timeline block.

---

**`remove_task`** — `{"task_id": "t_003"}`. Reschedules dependents. Returns timeline.

**`get_plan`** — no args. Returns timeline block. Cheap; the model can call it freely.

**`set_target_plating`** — `{"time": "7:15 PM"}` or `{"minutes_from_now": 45}`.

**`replan`** — `{"reason": "chicken took 20 min longer than expected"}`. Clears all pending tasks and returns control to the model to rebuild via `add_task`. Escape hatch — rare.

### 2.3 Mid-cook tools

Latency matters here. Keep these fast and single-round-trip.

---

**`move_task`**

```json
{ "task_id": "t_003", "after": "t_001" }
```

or `{"task_id": "t_003", "delay_minutes": 10}`.

Shifts dependents automatically. Validates. Returns timeline.

---

**`mark_complete`** — `{"step_ids": ["s_4", "s_5"]}` or `{"task_id": "t_001"}`.

Advances state, recomputes "next action", cancels associated timers if the task is done early.

---

**`skip_step`** — `{"step_id": "s_7", "reason": "no thermometer"}`. Records in overlay.

### 2.4 Timer tools

**`set_timer`**

```json
{
  "label": "chicken resting",
  "duration_s": 600,
  "task_id": "t_001",
  "step_id": "s_9",
  "on_complete_hint": "slice against the grain, plate immediately"
}
```

`label` is required and must be descriptive. Reject generic labels ("timer", "timer 2") — they defeat the purpose when the timer fires.

**`cancel_timer`** — `{"timer_id": "tm_002"}`

**`get_timers`** — no args. Returns the timer block.

### 2.5 Recipe tools

**`substitute`**

```json
{
  "recipe_id": "r_001",
  "ingredient_id": "i_004",
  "replacement": "olive oil",
  "note": "1:1 by volume",
  "at_step": "s_3"
}
```

**`scale`** — `{"recipe_id": "r_001", "factor": 2.0}`. Recomputes ingredient amounts; does *not* scale durations (the model should reason about whether timing changes).

**`add_note`** — `{"recipe_id": "r_001", "step_id": "s_4", "note": "pan was smoking, reduced heat"}`

### 2.6 Memory

**`remember`** — `{"note": "prefers less salt than recipes call for"}`

The one piece of state that can't be derived. Called when something conversational is worth carrying forward. Kept in `session.notes`, rendered into every context.

### 2.7 Inventory

**`check_stock`** — `{"items": ["butter", "garlic"]}`
**`deduct`** — `{"items": [{"name": "butter", "amount": 50, "unit": "g"}]}`

---

## 3. Context assembly

Built fresh every turn. Fixed token budget. This is the highest-leverage code in the project.

### 3.1 Block order

```
[system prompt]
[session notes]
[recipes — rendered with overlay applied]
[changes made]
[timeline]
[timers]
[progress summary]
[recent transcript — last N turns verbatim]
[current turn]
```

### 3.2 Timeline block format

Fixed-width columns. Models parse aligned tables more reliably than prose.

```
Now: 6:42 PM                    Target plating: 7:15 PM

APPLIANCE          WINDOW        TASK              STATUS
oven 425°F         6:30–7:15     chicken roast     active, 33m left
stovetop:2         6:50–7:05     rice              pending
air fryer 375°F    7:00–7:12     brussels          pending

Next action: 6:50 PM — start rice
Conflicts: none
```

The `Next action` line is pre-computed deliberately. It's the single most frequently needed fact, and computing it in code means the model never has to scan and reason to find it.

### 3.3 Timer block format

```
TIMERS
chicken resting    ends 7:05 PM   (23m left)   → slice against grain, plate
```

### 3.4 Progress block

Derived from `completed_steps` and `overlay`. Never LLM-generated.

```
PROGRESS
Chicken (r_001): steps 1–6 done. Seared, in oven.
Rice (r_002): not started.
Changes: butter → olive oil (step 3); doubled garlic.
Notes: user prefers less salt. Pan was smoking at step 4.
```

### 3.5 Transcript compression

Keep the last **3–5 turns verbatim** for pronoun resolution ("how long for *that*?"). Everything older is represented by the progress block above.

Do **not** use the LLM to summarize history. It's slow, costs tokens, and can hallucinate. Everything needed is already in structured state — except conversational notes, which `remember()` captures explicitly.

Rough budget at 16K context:

| Block | Tokens |
|---|---|
| System prompt | 400 |
| Tool definitions | 600 |
| Recipes (2, rendered) | 2000 |
| Timeline + timers | 300 |
| Progress + notes | 200 |
| Recent transcript | 800 |
| **Total input** | **~4300** |

Leaves generous headroom. Three or more recipes is where you'd start trimming.

---

## 4. Orchestrator

### 4.1 Event queue

One `asyncio.Queue`, two producers.

```python
@dataclass
class UserUtterance:
    text: str

@dataclass
class TimerFired:
    timer: Timer

@dataclass
class IdleTick:
    since_last_turn_s: int

Event = UserUtterance | TimerFired | IdleTick
```

### 4.2 Turn loop

```python
async def run(self):
    while True:
        event = await self.queue.get()
        prompt = self.synthesize_turn(event)
        if prompt is None:          # IdleTick suppressed by proactivity
            continue
        ctx = self.assemble_context(prompt)
        async for chunk in self.llm.stream(ctx, tools=TOOLS):
            if chunk.is_tool_call:
                result = await self.dispatch_tool(chunk)
                # feed result back, continue generation
            else:
                await self.output(chunk.text)
```

### 4.3 Turn synthesis

The key to proactive speech: system events become synthetic user turns, so everything downstream is identical.

| Event | Synthesized prompt |
|---|---|
| `UserUtterance` | the transcript, verbatim |
| `TimerFired` | `[SYSTEM] Timer "chicken resting" completed at 7:05 PM. Hint set at creation: "slice against grain, plate immediately". Tell the user what to do now.` |
| `IdleTick` | `[SYSTEM] 4 minutes since last exchange. Next action is at 6:50 PM. Say something useful if warranted, otherwise respond with exactly NOTHING.` |

`IdleTick` fires on a schedule but is gated by `session.proactivity` before reaching the model — at low settings, most ticks are dropped without an inference call. When the model returns `NOTHING`, suppress output entirely.

### 4.4 Transport independence

The orchestrator takes an output callback. FastAPI passes a websocket writer; Phase 1 passes `print`; tests pass a list appender. No FastAPI imports in `orchestrator.py`.

---

## 5. HTTP + WebSocket

### 5.1 WebSocket `/ws/session`

Carries everything real-time. One connection per session.

**Client → server:**

```json
{"type": "audio", "data": "<base64 PCM 16kHz mono>"}
{"type": "text", "text": "how long left on the chicken"}
{"type": "barge_in"}
{"type": "set_proactivity", "value": 0.3}
```

**Server → client:**

```json
{"type": "transcript", "text": "how long left on the chicken", "final": true}
{"type": "audio", "data": "<base64 PCM 24kHz>"}
{"type": "speech_start", "proactive": true}
{"type": "speech_end"}
{"type": "state", "timeline": {...}, "timers": [...], "progress": {...}}
{"type": "tool_call", "name": "set_timer", "args": {...}}
```

The `state` message drives the tablet UI. Push it after every tool call so the display never lags the conversation.

`speech_start` with `proactive: true` lets the UI signal that the assistant spoke unprompted — worth distinguishing visually.

### 5.2 HTTP endpoints

Small surface. Everything real-time is on the websocket.

```
GET    /health
GET    /recipes
GET    /recipes/{id}
POST   /recipes                 # add, incl. URL import
DELETE /recipes/{id}
GET    /inventory
PATCH  /inventory
POST   /session                 # start; returns session_id
DELETE /session/{id}
GET    /session/{id}/state      # debug/recovery
```

---

## 6. Build order

Each phase gated by: use it for a real meal before moving on.

| Phase | Scope | Done when |
|---|---|---|
| 0 | Ollama + whisper.cpp + Kokoro running standalone on ROCm | each produces correct output from CLI |
| 1 | State models, tools, scheduler, validation, renderers — terminal driven, **no LLM** | you can cook a two-recipe meal by typing tool calls |
| 2 | LLM behind the existing tool interface | model drives the same loop you drove manually |
| 3 | Constrained decoding (GBNF); malformed calls eliminated | 50 turns with zero parse failures |
| 4 | STT in front, push-to-talk | works with a running extractor fan |
| 5 | TTS out, then sentence streaming | first audio under 1s |
| 6 | Proactive path — timers speak unprompted | timer fires, it tells you what to do |
| 7 | Tablet PWA, wake word, barge-in | hands-free through a full meal |

Phase 1 is the one that's tempting to skip and shouldn't be. Getting the state model right without model output in the way is much easier, and Phase 2 becomes a swap rather than a rewrite.
