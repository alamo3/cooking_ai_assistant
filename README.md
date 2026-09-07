# Cooking Assistant

A local, conversational, voice-ready kitchen assistant. FastAPI orchestrator, Ollama-hosted
LLM, deterministic state. The model never holds authoritative state: every change goes
through a tool call, code resolves intent to wall-clock time, and the context is rebuilt
from structured state on every turn.

Design spec: [cooking-assistant-api-spec(1).md](cooking-assistant-api-spec(1).md).

## Which model to use

Two configurations, both already the defaults. Everything else in this file is the evidence
behind them.

| | Command | Model | Planning | Messy | Speed | Cost |
|---|---|---|---|---|---|---|
| **Best experience** | `-Cloud`, `COOK_OPENROUTER_MODEL=google/gemini-3.8-flash` | Gemini 3.8 Flash | 5/5 | 8/8 | 6 s | ~25c a meal |
| **Best value** | `.\start.ps1 -Cloud` | `google/gemma-4-31b-it` | 6/6 | 8/8 | 9 s | ~3c a meal |
| **Offline or no spend** | `.\start.ps1` | `kitchen:gemma-q8` | 21/21 | — | 23 s | free, 13 GB VRAM |

Gemini 3.8 Flash needs `COOK_OPENROUTER_REASONING=low`; it refuses `off` entirely
("Reasoning is mandatory for this endpoint").

`-Cloud` runs the cloud model with the local one as automatic fallback, so a dropped
connection mid-meal degrades instead of ending dinner. Pick the local default if you would
rather not spend or need it to work with the internet down.

Two rules that fall out of the testing, and matter more than the choice above:

1. **Never run a reasoning model here without pinning the effort.** GLM and every
   qwen3.5/3.6/3.7 reason by default and spend minutes per turn, because reasoning tokens are
   re-spent on each of the 5 to 9 tool rounds. Set `COOK_OPENROUTER_REASONING=off`.
2. **Raise the quantization before reaching for a bigger model.** The same gemma4:12b went
   from 4/9 at q4 to 21/21 at q8. That single change beat every larger model tried.

## Layout

```
src/cooking_assistant_ai/
  model/      dataclasses: Recipe (immutable), Overlay, Task, Timer, Session, events
  core/
    scheduler.py   intent -> clock windows, appliance/temperature conflicts, next action
    render.py      one renderer for recipe / timeline / timers / progress blocks
    context.py     system prompt + per-turn context assembly
    tools.py       tool registry, universal {ok,state,message|reason} envelope
    clock.py       injectable clock (tests and the --sim REPL move time)
    mealplan.py    pantry matching, meal suggestions, shopping lists
    diet.py        halal / vegetarian / vegan ingredient rules
  storage/    SQLite for recipes + inventory, with seed data
  llm/
    client.py           OllamaLLM (streaming + tools) and ScriptedLLM (tests)
    llm_orchestrator.py event queue, turn synthesis, tool loop, timers, idle ticks
  speech/     STT/TTS interfaces, sentence splitter for streaming TTS (adapters optional)
  api/app.py  HTTP + WebSocket transport, serves the tablet app
  web/        tablet kiosk app (index.html, app.js, app.css)
  cli.py      terminal REPL (Phase 1 without a model, Phase 2 with one)
evals/        plan_bench.py (planning) and messy_bench.py (recovery), both need a live model
tests/        121 tests, no model needed (plus an opt-in speech round trip)
```

## Install (Windows)

```powershell
.\install.ps1              # or -Cloud for OpenRouter, -WhatIfOnly to see what it would do
```

One idempotent script: installs dependencies, checks Ollama and the speech builds, generates
the TLS certificate, opens the port on private networks, seeds `cooking.db`, and registers a
**logon task** so the assistant is already running when you walk into the kitchen. Re-run it
after pulling changes; `.\install.ps1 -Uninstall` removes the task and firewall rule and
leaves your code, venv and database alone.

It registers a logon task rather than a Windows service on purpose: Ollama runs as a tray app
in your user session and the GPU is only reachable from there, so a session-0 service could
talk to neither. The cost is that it starts at logon, not at boot — enable Windows'
automatic sign-in if you want it up before anyone touches the machine.

The address it prints is the one a tablet on your network can actually reach.
Sorting interfaces by metric picks the wrong one when a VPN is connected — Surfshark,
Tailscale and friends own the default route — so [lib.ps1](lib.ps1) drops tunnel and virtual
adapters by name and prefers ordinary home ranges, `192.168/16` first.

The firewall rule needs an elevated shell. Without one the install still succeeds and tells
you the single command to run as admin; until then the tablet cannot reach the server.

## Restarting from the tablet

The gear in the top right opens a server panel: uptime, model, backend, a **Restart server**
button, and an amber dot when source files have changed since the process started (it lists
which ones). That is the supported way to pick up a code change.

A process cannot restart itself — once it exits nothing is left to serve the page that would
bring it back — so the lifecycle belongs to the supervisor loop at the bottom of `start.ps1`.
[admin.py](src/cooking_assistant_ai/api/admin.py) only decides how the process ends:

| exit code | meaning | supervisor |
|---|---|---|
| 42 | restart requested | relaunch immediately |
| 0 | Ctrl+C, clean stop | stop |
| anything else | crash | relaunch with backoff, giving up after 5 crashes in 2 minutes |

The tablet's websocket reconnects on its own, so a restart shows up as a couple of seconds of
"restarting" and then "Server restarted".

**A restart throws away every cooking session** — timers, tasks and progress are all in
memory. `/admin/restart` returns 409 if a cook is in progress and the UI makes you confirm;
only `force: true` gets past it.

`--reload` (that is, `start.ps1 -Dev`) is separate: uvicorn's own reloader owns the process
tree there and picks up file changes by itself, so the restart button reports itself
unavailable.

## Surviving a crash or a power cut

The cooking session — the plan, the timers, what has been done, what was said — was the only
state not in the database, and losing it mid-cook is the worst failure this system has. It is
now snapshotted to a `sessions` table: after every state-changing tool call, on a timer
(`COOK_AUTOSAVE`, default 10 s), and on clean shutdown. Startup restores anything younger
than `COOK_SESSION_MAX_AGE` (default 12 h) under its original session id, so the tablet's
reconnect lands back in the same cook without doing anything.

This does not make the database a second source of truth (spec 1.5). Snapshots are write-only
while the server runs and read only at startup; the live `Session` always wins.

Two things that make restoring correct rather than merely possible, both in
[sessions.py](src/cooking_assistant_ai/storage/sessions.py):

- **Id counters are rebuilt from the ids in use.** They are iterators and cannot be
  serialized, and a fresh counter would hand out `t_001` again and overwrite the first task
  of the cook.
- **Timers that ran out while the power was off are retired, not fired.** Otherwise every one
  of them goes off at once on startup. They are marked fired and a session note says which
  finished unannounced and how long the server was down, so the assistant can mention it.

Recipes are stored in full in the snapshot rather than by id, so a session restores intact
even if the recipe was edited or deleted while the server was down. Ending a session with
`DELETE /session/{id}` drops its snapshot; a crash does not, which is the whole point.

```
Recovered the cook from 2:39 PM: Roast Chicken Thighs (1 timer finished while it was down)
```

## Run

Python 3.12 (pinned in `.python-version`; uv downloads it). The `speech` extra pulls
faster-whisper and Kokoro, including a CPU torch build.

```bash
uv sync --group dev --extra speech
uv run pytest
COOK_SPEECH_TESTS=1 uv run pytest tests/test_speech_roundtrip.py   # Kokoro -> Whisper round trip

# Phase 1: drive the state machine by hand, frozen clock
uv run cooking-assistant-ai repl --no-llm --sim --load r001 r002
#   /plating 7:30 PM
#   /tool add_task {"label":"chicken roast","recipe_id":"r001","step_ids":["r001-s5"],"must_finish_by":"plating"}
#   /tool set_timer {"label":"rice simmering","duration_s":900,"on_complete_hint":"take it off the heat"}
#   /now +16m        -> the timer fires (with a model it would speak)
#   /state

# Phase 2: the model drives the same tools (needs `ollama serve` and the model pulled)
uv run cooking-assistant-ai repl --model kitchen:gemma-q8 --load r001 r002

# Server (HTTPS + websocket on :8000), with voice in and out
COOK_STT=whisper.cpp COOK_TTS=kokoro uv run cooking-assistant-ai serve --https --model kitchen:gemma-q8

# or, on Windows, the start script (HTTPS by default; -Open launches the browser).
# This is the supervisor: it is what makes the tablet's restart button work.
.\start.ps1 -Open
```

Environment: `COOK_MODEL` (default `kitchen:gemma-q8`), `COOK_NUM_CTX` (default 16384, sent
with every request; the Modelfile carries the same value), `COOK_KEEP_ALIVE`
(default `30m` idle before Ollama frees the VRAM; `-1` pins it forever),
`COOK_DB` (default `cooking.db`), `COOK_AUTOSAVE` seconds between session
snapshots (default 10, `0` disables), `COOK_SESSION_MAX_AGE` seconds a snapshot stays
resumable (default 43200),
`COOK_STT` (`whisper.cpp` | `faster-whisper` | `none`), `COOK_TTS=kokoro`
(`COOK_TTS_VOICE`, default `af_heart`), `COOK_IDLE_INTERVAL` seconds, `COOK_WARM=0` to skip
warm-up of the model and speech engines at startup.

whisper.cpp settings: `COOK_WHISPER_CPP_BIN` (auto-detected under `whisper.cpp/build-vulkan`
then `whisper.cpp/build`), `COOK_WHISPER_CPP_MODEL` (auto: `ggml-base.en.bin`, falling back
to `ggml-small.en.bin`), `COOK_WHISPER_CPP_PORT` (8178), `COOK_WHISPER_CPP_THREADS` (8),
`COOK_WHISPER_CPP_URL` to use a server you started yourself, `COOK_WHISPER_PROMPT` for
vocabulary hints. faster-whisper settings: `COOK_WHISPER_MODEL`, `COOK_WHISPER_DEVICE`,
`COOK_WHISPER_COMPUTE`.

## Adding recipes

- **From a website**: on the tablet, tap "+ Add" and paste the page URL into "Paste a
  recipe URL", or `POST /recipes` with `{"url": "..."}`. The importer reads the site's
  embedded schema.org recipe data when present (most recipe sites) and falls back to the
  page text, then has the model structure it: numeric amounts, units, per-step durations,
  appliance and temperature. Takes 10 to 15 seconds. Check the result with `GET /recipes/{id}`.
- **From pasted text**: some sites (Allrecipes, for one) refuse non-browser requests.
  Copy the recipe from your browser and `POST /recipes` with `{"text": "..."}`; the same
  extraction runs on it.
- **Invented by the assistant**: ask for something new and it composes a dish from your pantry
  and saves it with `create_recipe`. See "Meal planning from the pantry" below.
- **By hand**: `POST /recipes` with the JSON shape returned by `GET /recipes/{id}`
  (`title`, `servings`, `ingredients[{name, amount, unit}]`, `steps[{text, duration_s, appliance, temp_f}]`).
- **Remove**: `DELETE /recipes/{id}`.

## Tablet app

`src/cooking_assistant_ai/web/` is served at `http://<server>:8000/` (plain HTML, CSS and
JS, no build step). Open it in a kiosk browser on the tablet.

- **Recipes screen** (where a fresh session starts): set the diet, tick every recipe you're
  cooking and tap "Start cooking". Recipes the diet forbids are dimmed and cannot be ticked.
  "Suggest meals for N portions" asks the model what to cook from what is in the pantry and
  shows the answer inline, so you never have to leave the screen to ask.
  There is no serving deadline unless you tick "serving at a set time". The
  server loads the whole selection and hands the model one planning turn: it builds all the tasks,
  then briefs you out loud from a code-computed summary (how long, when to start, which
  appliances and burners, what to prep first, the ingredients to get out) and names the first
  thing to do. Adding a recipe later re-runs the same flow and only extends the plan. Import
  from a URL and delete from the library live here too.
- **Cook screen**: the merged plan (NOW card, "Then:", full list) with recipe cards
  collapsed beneath; the chef avatar; the conversation with tool-call chips; timers
  with live countdowns and cancel; the timeline with next action and conflicts; a
  proactivity slider. Hold the big button (or the space bar) to talk, or type.

**The avatar** is an inline SVG chef (toque, face, steam) rather than an emoji, animated
purely in CSS off one attribute: the JS only ever sets `#avatar[data-state]`, so the drawing
can change freely without touching behaviour. Each state is visually distinct at a glance
from across a kitchen: *listening* breathes a blue halo and raises the brows, *thinking*
spins an amber arc and lifts steam, *speaking* and *heads up* animate the mouth (heads up
also pulses amber and nudges, so unprompted speech is obvious), *problem* frowns, *offline*
goes grey. The palette is warm charcoal and amber rather than the usual blue-grey dashboard,
since the thing lives in a kitchen.
- **Pantry screen**: inventory with +/- adjustments, inline amount edits, add and remove.

**The tablet needs HTTPS.** Browsers expose `navigator.mediaDevices` only in a secure
context, so over plain HTTP to a LAN address the microphone API is not merely blocked, it
is absent, and no browser or kiosk setting brings it back. `start.ps1` therefore serves
HTTPS by default with a self-signed certificate generated into `~/.cooking-assistant`
(covering localhost, this machine's hostname and its LAN addresses; `--https` on the serve
command does the same). Accept the certificate once on the tablet: in Fully Kiosk, turn on
the setting that ignores SSL/certificate errors, or download it from `https://<server>:8000/cert`
and install it. Then grant Fully Kiosk the Android microphone permission and enable its
microphone/WebView permission option. Pass `-Http` to `start.ps1` for plain HTTP, with no
tablet microphone; the page then shows a banner saying so rather than failing cryptically.
Audio comes back one sentence at a time and plays as it arrives.

**Always listening.** The "Always listening" toggle opens the microphone for good (the
setting survives reloads). The tablet streams 16 kHz audio in 100 ms chunks; the server
runs Silero voice activity detection to find where an utterance starts and stops (300 ms
pre-roll, 500 ms of silence ends it), transcribes it with whisper.cpp and submits it. Talk
over the assistant for 400 ms and it stops (barge-in). The browser's echo cancellation keeps
the assistant's own voice out of the mic; anything that leaks through is dropped when its
words match what was just spoken, and Whisper's silence hallucinations ("Thank you.") are
filtered. `COOK_VAD=energy` swaps in a simple level gate; `COOK_VAD=none` disables the mode.
"Hold to talk" remains available with the toggle off.

`COOK_BARGE_IN` picks how interruption works: `voice` (default) stops the assistant after
400 ms of speech while it is playing, which depends on the tablet's echo cancellation being
good; `transcript` waits until the interruption has been transcribed and cleared the echo
guard (about a second slower, never triggered by the assistant's own voice); `off` disables
voice interruption. If the assistant cuts itself off mid-sentence, switch to `transcript`.

UI-initiated actions go through the same tools as the model: `POST /session/{id}/start`
(`recipe_ids`, `minutes_from_now` or `target_plating`), `POST /session/{id}/recipes`,
`DELETE /session/{id}/recipes/{rid}`, `POST /session/{id}/steps/{sid}/complete`,
`DELETE /session/{id}/timers/{tid}`. Each pushes a `tool_call` and `state` to every
connected client.

## WebSocket protocol (`/ws/session?session_id=...`)

Client to server: `{"type":"text","text":"..."}`, `{"type":"audio","data":"<b64 pcm16 16k>"}`
(one complete utterance), `{"type":"listen","on":true}` then a stream of
`{"type":"audio_chunk","data":"<b64 pcm16 16k>"}` (open mic), `{"type":"playback","playing":true}`
(speaker state, gates barge-in and the echo guard), `{"type":"barge_in"}`,
`{"type":"set_proactivity","value":0.3}`, `{"type":"get_state"}`.

Server to client: `session`, `listen` (ack), `vad` (`speaking` true/false), `transcript`,
`speech_start` (with `proactive`), `text` (streamed chunks), `audio` (per sentence, when TTS
is configured), `speech_end`, `stop_playback` (barge-in: stop the speaker now), `tool_call`,
`state` (timeline, timers, progress, plus a rendered `text`), `dropped` (why an utterance was
discarded), `notice`, `error`.

## Cloud inference (the default)

`COOK_LLM=cloud` is the default: Gemini 3.8 Flash via OpenRouter, with the local Ollama model
as a safety net for when the internet drops mid-cook. The local model is **not built,
connected to or warmed** until a cloud request actually fails — constructing it is cheap but
warming it loads ~19 GB into VRAM, which would then sit there for the whole `keep_alive`
window for a backend that may never be used. `/health` reports `local_loaded` so you can see
whether it has ever been needed.

Without an `OPENROUTER_API_KEY` this degrades to the local model rather than refusing to
start. `start.ps1 -Local` (and `install.ps1 -Local`) forces local inference.

`COOK_LLM` picks the backend: `ollama` (default, local), `openrouter`, or `cloud` (OpenRouter
first, local Ollama automatically whenever it errors, so a dropped connection mid-meal does
not end the meal). The key comes from `OPENROUTER_API_KEY` and is never written to the repo:

```powershell
setx OPENROUTER_API_KEY "sk-or-..."   # once, then open a new terminal
.\start.ps1 -Cloud                    # OpenRouter with the local model as fallback
```

`COOK_OPENROUTER_MODEL` defaults to `openai/gpt-oss-120b`;
`COOK_OPENROUTER_REASONING` sets `low`/`medium`/`high` for models that take a reasoning effort.
`GET /health` reports `usage` (requests, tokens, `cost_usd`) and `fallbacks_to_local`, so a
meal's actual spend is measurable rather than estimated. Rate limits and 5xx are retried
twice with short backoff (about 1.5 s total) and then handed to the fallback, because a cook
at the stove is better served by a fast local answer than by patient retries.

**Free models are not viable for this app.** A single planning turn is 5 to 9 sequential
requests, and `:free` slugs are capped at 20 requests per minute and 50 per day without
purchased credits, on top of the upstream provider's own limiting. Tested with
`google/gemma-4-31b-it:free`: isolated calls work (1.4 s, tool call parsed correctly), but a
planning turn is throttled to failure. The paid slug has no such problem.

**Cloud models on the same benchmark.** Cost per meal is computed from published rates and
the measured 280K input / 10K output of a heavy meal, not from OpenRouter's usage counter,
which lags by minutes and under-reports right after a run:

| Model | Clean | Median turn | $/M in-out | Est. per meal |
|---|---|---|---|---|
| `google/gemma-4-31b-it` | 6/6 | 9 s | 0.09 / 0.34 | $0.029 |
| `qwen/qwen3-235b-a22b-2507` | 5/5 | 19 s | 0.087 / 0.35 | $0.028 |
| `moonshotai/kimi-k2-0905` | 5/5 | 9 s | 0.60 / 2.50 | $0.18 |
| `meta-llama/llama-4-maverick` | 4/5 | 10 s | 0.20 / 0.70 | $0.063 |
| `openai/gpt-4.1-mini` | 1/5 | 44 s | 0.40 / 1.60 | $0.128 |
| `qwen/qwen3.6-plus` | 5/5 | 14 s | 0.325 / 1.95 | $0.111 |
| `qwen/qwen3.6-27b` (same model as local) | 5/5 | 35 s | 0.30 / 2.00 | $0.104 |
| `qwen/qwen3.7-flash` | 2/5 | 8 s | 0.03 / 0.13 | $0.010 |
| `z-ai/glm-5.3-flash`, reasoning low | 3/4 | 13 s | 0.075 / 0.25 | $0.024 |
| `z-ai/glm-5.3-flash`, reasoning on | 2/2 | 313 s | 0.075 / 0.25 | ~$0.10 |

The qwen family is reliable at every size tested (`qwen3.6-27b`, `qwen3.6-plus` and
`qwen3-235b` all scored 5/5), which matches its 21/21 locally, but it is priced three to four
times higher per token than gemma and is slower. Two results are worth remembering: the
*same* model, `qwen3.6-27b`, is slower through the cloud (35 s) than on this machine (19 s),
so there is no speed argument for hosting it remotely, only a VRAM one. And the newest,
cheapest qwen, `qwen3.7-flash`, was among the least reliable at 2/5, so newer is not better.

Every qwen3.5/3.6/3.7 and GLM model reasons by default. Set `COOK_OPENROUTER_REASONING=off`
(sent as `reasoning: {enabled: false}`, the cloud equivalent of Ollama's `think=false`) or
they spend minutes per turn. On `qwen3.6-27b` that one setting is 19 s versus 1.9 s for a
single request.

`gemma-4-31b` wins on the combination: it ties kimi-k2 on quality and speed at a sixth of
the price, and ties qwen3-235b on price while being twice as fast. `qwen3-235b-a22b-2507` is
the pick if you would rather have a 235B mixture-of-experts and can accept 19 s.
`gpt-4.1-mini` was the surprise, scoring 1/5 and taking 44 s, and is not recommended here.

GLM 5.3 Flash plans well and is cheaper per token ($0.075/$0.25 against $0.09/$0.34), but its
reasoning is a liability here: left on it produced flawless plans at four to seven minutes a
turn, because reasoning tokens are re-spent on every one of the 5 to 9 tool rounds. Dialled
down it is fast but no more reliable than gemma. This is the same trap gpt-oss fell into, and
the rule generalises: **for this multi-round tool loop, prefer a non-reasoning model, or set
the effort explicitly.**

**Measured on the paid `google/gemma-4-31b-it`**, same benchmark as the local models:

| | cloud gemma-4-31b | local gemma q8 | local 27B |
|---|---|---|---|
| Clean trials | 6/6 | 21/21 | 21/21 |
| Median planning turn | 9 s | 23 s | 19 s |
| GPU VRAM for the LLM | 0 | 13 GB | 18.8 GB |
| Cost per planning turn | $0.0037 | 0 | 0 |

**Cost, measured on a deliberately heavy meal**: three recipes loaded, one planning turn and
25 conversational turns, 20 tool calls.

| | |
|---|---|
| Requests | 33 |
| Input tokens | 173,730 |
| Output tokens | 1,613 |
| Planning turn | $0.0018 |
| 25 conversational turns | $0.0281 ($0.0011 each) |
| **Whole meal** | **$0.030** |

So three cents for heavy use, and $10 of credits is about 330 such meals. Cost is dominated
by input, not output, at roughly 100 to 1: every turn resends the whole context, and every
tool round inside a turn resends it again. That means spend scales with how many recipes are
loaded and how many turns you take, not with how much the assistant says. Loading two
recipes instead of three is the single biggest lever.

Running `-Cloud` leaves the GPU holding only whisper (about 0.6 GB), which is the
configuration to use if you would rather your PC stayed free for other work.

[openrouter.py](src/cooking_assistant_ai/llm/openrouter.py) translates between the two
message shapes: OpenAI needs an `id` on each assistant tool call and a matching
`tool_call_id` on each tool result, and streams tool arguments as fragments that must be
concatenated per index. Tool *schemas* need no translation, since Ollama's and OpenAI's
function format already agree.

## GPU memory

The assistant shares one GPU with the Windows desktop, so it is tuned not to fill it:

- **Model size.** `kitchen:gemma-q8` (gemma4:12b at q8_0, ~13 GB resident) is the default.
  Scored on the same "Start cooking" turn (chicken plus rice, 90 minutes to plating). A
  trial is clean only if it creates 3+ tasks, plans both recipes, duplicates no step, orders
  each recipe's tasks correctly (no searing and roasting the same chicken at once), lands
  the meal within 20 minutes of plating, and delivers a briefing:

  | Model | Resident | Clean trials | Median |
  |---|---|---|---|
  | `kitchen:gemma-q8` (gemma4:12b q8_0) | 13 GB | 21/21 | 23 s |
  | qwen-kitchen (qwen3.6:27b q4_K_M) | 18.8 GB | 21/21 | 18 s |
  | `kitchen:gemma-qat` (gemma4:12b QAT) | 7.7 GB | 7/8 | 12 s |
  | gemma4:12b q4_K_M | 8.1 GB | 4/9 | 11 s |
  | gpt-oss:20b, reasoning low | 12 GB | 1/4* | 10 s |
  | qwen3:14b (`qwen-kitchen:14b`) | 10 GB | 1/3* | 16 s |
  | gpt-oss:20b, reasoning medium | 12 GB | 0/3* | 35 s |
  | mistral-small3.2:24b | 16 GB | 0/3* | 11 s |
  | qwen3.5:9b | 6.9 GB | 0/1* | 6 s |

  Rows marked * were scored before the ordering and timing checks existed, so their true
  rate is at best the number shown. The gemma rows include the empty-turn retry below.

  **Quantization mattered more than parameter count.** The same gemma4:12b scored 4/9 at
  q4_K_M, 7/8 quantization-aware-trained at the same size, and 21/21 at q8_0. Its q4 failure
  was specific: it stopped setting `must_finish_by`, so meals finished long before plating.
  Always try a higher quant of a small model before reaching for a bigger one.

  Over 21 trials each, gemma q8 and the 27B are indistinguishable on reliability, so memory
  is the tiebreaker: gemma leaves about 9 GB of the card free while cooking against the 27B's
  3 GB, and costs 5 s more per turn. A lower-quant 27B is not an option, since Ollama's
  `--quantize` supports only F32, F16, Q4_K_S, Q4_K_M and Q8_0, and qwen3.6:27b is already
  Q4_K_M.

  Use `--model kitchen:gemma-qat` to trade a little reliability for 5 GB and half the
  latency, or `--model qwen-kitchen` for the 27B. Gemma also invents an argument
  (`stovetop: true` rather than `appliance: "stovetop"`), harmless only because `add_task`
  falls back to the appliance recorded on the steps.
- **Empty turns.** Every model tested occasionally returns nothing at all, no text and no
  tool call, in a second or two. The orchestrator retries such a turn once
  (`empty_retries`), which recovered it every time it fired. Proactive turns are exempt,
  since silence is how they decline to speak.

  The smaller models produce syntactically valid calls and semantically wrong plans: rice
  chained after the chicken is served, a circular set of `after` references, a whole recipe
  given a 32-second duration, or a recipe left unplanned. Reproduce with
  `evals/plan_bench.py <model> <trials>` before switching. `--model qwen-kitchen:14b`
  remains available when headroom beats plan quality.

  Llama has nothing in the useful range: 3.1 is 8B (too weak, the 9B tier scored 0/1) or
  70B (43 GB), and Llama 4's smallest is a 109B mixture of experts.
- **Reasoning.** `COOK_THINK` sets the model's chain-of-thought: `false` (hybrids such as
  Qwen3, whose reasoning is slow and must not reach the cook), `low`/`medium`/`high` (models
  that require a level, such as gpt-oss), or `none` to omit the field. Unset, it is chosen
  per model. This matters more than it sounds: gpt-oss with reasoning forced off took 150 s
  and produced one task, and at `low` took 11 s and produced a complete plan.
- **Idle unload.** `COOK_KEEP_ALIVE` defaults to `30m`, so the weights are released between
  meals. The next question after that pays one reload.
- **Ollama flags.** `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` roughly halve
  the key-value cache and let 16K context stay affordable. `start.ps1` sets them as user
  environment variables; Ollama reads them at startup, so restart the Ollama app after a
  change. Check with `ollama ps` and `%LOCALAPPDATA%\Ollama\server.log`.

Measure what is actually resident with `ollama ps`, and per process with
`Get-Counter "\GPU Process Memory(*)\Local Usage"`.

## The two benchmarks

`evals/plan_bench.py <model> <trials>` scores building a plan. `evals/messy_bench.py <model>`
scores what happens when cooking goes wrong: eating 25 minutes earlier, the chicken needing
longer, running out of butter, skipping a step, "how much longer on that one?" with two timers
running, the cook claiming the rice is done when it never started, dropping the garlic, and a
guest turning out vegetarian. Each scenario starts from the same deterministically-built state
and is scored on an objective consequence: state that must change, or a reply that must ask
instead of guess.

**Both benchmarks now saturate.** Gemma and Gemini each score 8/8 on the messy set, so the
numbers can no longer separate good models, and the difference shows up only in *how* they
pass. On "I dropped half the garlic", gemma asked whether there was enough left; Gemini called
`check_stock`, found six cloves in the pantry and offered to smash two more. On the chicken
running long, gemma set a timer; Gemini cancelled the old timer, set a new one, moved the task
and pushed plating. Both pass. One is an assistant, the other is a prompt that answers.

If you extend the evals, add scenarios that a merely-adequate model would fail, not more of
the ones everything passes.

## Keeping state honest without taking over

Three guardrails sit between the model and the state. All of them *check or surface*; none
of them decides what happens in the kitchen, which stays the model's job.

- **Claim checking** catches the model saying it did something it did not do.
- **Omission nudging** is the mirror: the cook says the kitchen moved on ("it's going in the
  oven now", "I've rinsed the rice") and the turn ends with no state change. The orchestrator
  sends one `[SYSTEM]` nudge inviting the model to record it *or ask the cook what they meant*.
  It never calls the tool itself, and questions ("how long left?", "is the rice covered?") are
  never treated as progress reports.
- **Drift warnings** surface contradictions as questions in the timeline the model reads:
  a timer running for a task still marked pending, a task long past its window, a task
  complete with unmarked steps. Rendered as `CHECK:` lines, not applied automatically.

The division of labour: code owns arithmetic and bookkeeping (clock windows, which burner,
what has been recorded), the model owns judgement (what the cook meant, what to do when
something goes wrong, when to ask). A guardrail that silently corrected state would make the
assistant a state machine with a voice; one that asks keeps it a cook's assistant.

## Claim checking

State only changes through tools, so any first-person claim in the model's reply ("I've set
a timer", "I've marked that done", "I'll remember that") must be backed by a successful
tool call in the same turn. [claims.py](src/cooking_assistant_ai/core/claims.py) detects
such sentences; the orchestrator's `SpeechGate` streams ordinary sentences immediately,
holds everything from the first unbacked claim, releases held text in order once a
justifying tool call succeeds, and otherwise sends the model one corrective `[SYSTEM]`
message and speaks only the corrected reply. A claim still unbacked after that is dropped
with a `notice` on the websocket. Timer status remarks ("your rice timer is still running")
are accepted when such a timer exists.

## Dietary restrictions

`none`, `halal`, `vegetarian` or `vegan`, chosen on the Recipes screen, stored in SQLite so
it survives restarts, and changeable by voice with `set_diet`. It is not advisory: forbidden
recipes cannot be ticked on the tablet, are listed separately in the planner under "NOT
ALLOWED, do not recommend these", and the restriction is injected at the top of the system
prompt so it governs every substitution and suggestion, not only the meal planner.

[diet.py](src/cooking_assistant_ai/core/diet.py) separates two genuinely different questions:

- **excluded** — the ingredient itself breaks the rule (pork or alcohol for halal, chicken for
  vegetarian, dairy for vegan). Never recommended.
- **check** — allowed, but depends on sourcing. Meat can be halal; no ingredient list can tell
  you whether it was. The assistant says so instead of pretending to certify it.

Plant-based products are exempted from the dairy rules by qualifier, so "coconut cream",
"vegan butter" and "oat milk" are not mistaken for dairy, and word boundaries stop "beef
tomato" and "vegetable stock" being read as meat.

Verified live. On vegetarian it offered only the rice and sprouts, and when asked directly for
the chicken replied *"Your kitchen is set to vegetarian, so chicken thighs aren't an option...
or change your diet setting if you'd like to include poultry."* On halal it recommended the
chicken but added *"make sure your chicken thighs are halal-certified"*, and when asked about
deglazing with white wine — something no recipe mentions — answered *"Since this is a halal
kitchen, skip the wine"* and suggested broth with lemon or cider vinegar.

## Meal planning from the pantry

Ask "what should I cook this week?" and the model calls `suggest_meals(meals=N)`.
[mealplan.py](src/cooking_assistant_ai/core/mealplan.py) answers the factual half: for every
stored recipe, how many of its ingredients are in stock, which are missing and by how much,
and how many portions it makes, sorted so what you can cook today comes first. The model does
the choosing, because "what would make a good week of food" is judgement, not arithmetic.

Recipe names are matched to pantry names with the same normalizer the mise en place grouping
uses, so "Garlic cloves, smashed" finds "garlic". Salt, pepper, water, oil, sugar, flour and
butter are treated as staples: worth mentioning when short, never a reason to rule a recipe
out. Amounts are only compared when the units agree, so 200 g of rice against "1.5 cups"
counts as in stock rather than guessing at a conversion.

`shopping_list(recipe_ids, scale)` turns a chosen set into what to buy, subtracting stock and
merging duplicates across recipes. `GET /meal-options?meals=N` returns the same data as JSON.

Ask for suggestions three ways: the "Suggest meals for N portions" box on the Recipes screen,
the chat box or voice on the Cook screen, or `suggest_meals` directly.

**The assistant is not limited to the stored library: it can invent recipes.** `create_recipe`
composes a dish around what the pantry holds and saves it permanently, so it appears on the
Recipes screen and is cookable like any other. Ask for "two brand new dishes from what's in my
pantry" and it will propose them, and on your approval save them and tell you what to buy.

Three things are enforced in code rather than trusted to the model. The **diet** is checked at
creation, so a chicken dish in a vegan kitchen is refused with "chicken is not vegan; invent
something without chicken" and nothing is written. Every step is pushed to carry a **duration,
and an appliance and temperature when it is on the heat**, or the merged plan could not
schedule it. And a step's `uses` list is resolved to real ingredient ids, so an invented recipe
joins the shared **mise en place** grouping like a seed one. Recipes thinner than two
ingredients or two steps are rejected, and bare strings are accepted for both, since models
often send `["bread", "butter"]` rather than objects.

## Scheduling semantics the model is told about

- Everything starts as soon as it can: now, or when the tasks it comes `after` end.
- **There is normally no serving deadline.** Batch cooking has none, and setting one forces
  every task as late as possible and manufactures conflicts. The Recipes screen only asks for
  a time if you tick "serving at a set time", and the planning prompt tells the model not to
  set `must_finish_by`.
- If the cook does name a time ("we're eating at seven"), `set_target_plating` still works and
  `must_finish_by: "plating"` then schedules that task to end exactly at plating.
- `must_finish_by: "<task>"` schedules as late as possible so it ends when that task starts.
- Ovens are shared when the temperature matches; burners and the air fryer are exclusive.
- `appliance: "stovetop"` without a number gets the lowest burner free for its whole window
  (`COOK_BURNERS`, default 4); a task already on the heat keeps its burner. If every burner
  is busy the task is rejected with the window that's blocked.
- Rejections name the conflicting task, appliance, temperature and window.

## The merged cook plan

[plan.py](src/cooking_assistant_ai/core/plan.py) derives one step sequence across every
loaded recipe: steps covered by tasks take their task's clock (each step follows the one
before), prep steps a task doesn't cover are scheduled backwards from the recipe's first
task, steps after a task (rest, serve) follow it, and recipes with no tasks yet are listed
as unscheduled. The first open item is NOW. The model sees it as the COOK PLAN block and is
told "what's next" means that line; the tablet shows it as the big card at the top left,
with "Then:" beneath and the full list in an expander.

Mise en place: ingredients that two or more loaded recipes need prepped (chop, dice, mince,
trim, and so on, matched after normalizing names such as "garlic cloves, smashed" and
"garlic, minced") become a prep group at the top of the plan with the combined amount:
"Garlic: 4 (chicken) + 2 (beans) = 6 total". Salt, pepper, oil and water are never grouped.
Completing a group (`complete_prep` by voice, or the card's button) marks the matching step
in every recipe done.

## Changes to a recipe mid-cook

`substitute` is a **permanent edit**. It renames the ingredient and rewords every step that
named it, saves the result with `put_recipe`, and swaps the session's copy for the edited
one — so "butter -> olive oil" is what the recipe says next week too. Ids are preserved, so
running tasks, timers and completed steps still point at the same steps, and a follow-up
swap resolves the new name ("actually, ghee instead of the olive oil"). The model is told
the change is permanent and says so.

Matching is deliberately conservative: the ingredient's head name before any comma, on word
boundaries, singular or plural. "butter" -> "olive oil" rewrites "melt the butter" and
leaves "buttermilk" alone.

Pass `at_step` for a one-off: only that step is reworded, it lives in the session overlay
and the saved recipe is untouched. `scale`, `skip_step`, `add_step` and `add_note` stay
session-only the same way.

Whichever kind, the change shows up on every surface the cook uses — the COOK PLAN "NOW"
line, what the assistant reads aloud, the recipe block the model sees, and the tablet's step
list (badged "swapped", with "(instead of butter)" beside the ingredient for as long as the
session that made the swap lasts).

## What the first real cook changed

Cooking a meal with it surfaced things no benchmark did. Fixed so far:

- **Substitutions left the timeline stale.** Step text updated, but task and timer labels are
  free text the model wrote at `add_task` time ("butter sear") and nothing revisited them, so
  the plan said olive oil while "next action" still said butter. A whole-recipe swap now
  renames tasks, timer labels and completion hints too. A swap pinned with `at_step` does not:
  it is not a rename.
- **"Add the garlic" is useless with both hands full.** Every COOK PLAN line now carries the
  scaled amounts for that step (`[uses: 1 1/2 tsp kosher salt, ...]`) and the dish name, and
  the model is told to say both, so the cook never has to ask how much or for which dish.
- **Too much at once.** The prompt used to ask for "what to do now and, if useful, what comes
  next", which is two instructions. It now asks for exactly one, then silence.
- **Mishearing.** whisper.cpp now prefers `small.en` over `base.en` (0.59 s against 39 ms on
  Vulkan, worth it for ingredient names), and gets a decoding hint built from the session:
  the titles and ingredients of the dishes actually loaded, which are precisely the words the
  cook is about to say. `COOK_WHISPER_PROMPT` prepends your own.
- **Barge-in triggered on clatter.** Silero scores running water and pan lids as speech, and
  400 ms of it was enough to cut the assistant off. Now 900 ms sustained at a 0.65 threshold
  (`COOK_BARGE_IN_MS`, `COOK_VAD_THRESHOLD`).

## Speech

STT defaults to the whisper.cpp checkout in `whisper.cpp/`: the app spawns
`whisper-server` with the model resident and posts one 16 kHz WAV per utterance. TTS is
Kokoro-82M on CPU (it renders about 3 seconds of speech in 0.4 seconds).

STT latency for a 4 second utterance on this machine (RX 7900 XTX), after warm-up:

| Backend | Model | Median |
|---|---|---|
| whisper.cpp Vulkan | base.en | 39 ms |
| whisper.cpp Vulkan | small.en | 0.59 s |
| whisper.cpp CPU | base.en | 0.33 s |
| whisper.cpp CPU | small.en | 1.25 s |
| faster-whisper CPU int8 | small | 1 to 2 s |

The app prefers `whisper.cpp/build-vulkan/` when it exists. Kokoro weights download from
Hugging Face on first use (about 330 MB); ggml whisper models live in the whisper.cpp
checkout (`ggml-base.en.bin`, `ggml-small.en.bin`).

### Building whisper.cpp with Vulkan on Windows

Current ggml needs Vulkan headers newer than SDK 1.3.250 (for `VK_EXT_layer_settings`) and
the SPIRV-Headers CMake package. Both are header-only, so install them to a local prefix and
keep the SDK's loader library and glslc:

```bash
P=/tmp/vkprefix
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers /tmp/SPIRV-Headers
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers /tmp/Vulkan-Headers
for d in SPIRV-Headers Vulkan-Headers; do
  cmake -S /tmp/$d -B /tmp/$d/build -DCMAKE_INSTALL_PREFIX=$P
  cmake --build /tmp/$d/build --config Release --target install
done
cd whisper.cpp
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$P -DVulkan_INCLUDE_DIR=$P/include
cmake --build build-vulkan --config Release -j --target whisper-server whisper-cli
```

A newer Vulkan SDK (for its glslc) would let ggml enable the cooperative-matrix and
integer-dot shader paths, which is where `small.en` would gain the most.

The websocket sends each assistant sentence as its own `audio` message as soon as it is
complete, so playback starts while the model is still generating.
