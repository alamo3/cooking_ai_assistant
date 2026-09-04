# Cooking Assistant

A local, conversational, voice-ready kitchen assistant. FastAPI orchestrator, Ollama-hosted
LLM, deterministic state. The model never holds authoritative state: every change goes
through a tool call, code resolves intent to wall-clock time, and the context is rebuilt
from structured state on every turn.

Design spec: [cooking-assistant-api-spec(1).md](cooking-assistant-api-spec(1).md).

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
  storage/    SQLite for recipes + inventory, with seed data
  llm/
    client.py           OllamaLLM (streaming + tools) and ScriptedLLM (tests)
    llm_orchestrator.py event queue, turn synthesis, tool loop, timers, idle ticks
  speech/     STT/TTS interfaces, sentence splitter for streaming TTS (adapters optional)
  api/app.py  HTTP + WebSocket transport, serves the tablet app
  web/        tablet kiosk app (index.html, app.js, app.css)
  cli.py      terminal REPL (Phase 1 without a model, Phase 2 with one)
tests/        72 tests, no model needed (plus an opt-in speech round trip)
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

# or, on Windows, the start script (HTTPS by default; -Open launches the browser)
.\start.ps1 -Open
```

Environment: `COOK_MODEL` (default `kitchen:gemma-q8`), `COOK_NUM_CTX` (default 16384, sent
with every request; the Modelfile carries the same value), `COOK_KEEP_ALIVE`
(default `30m` idle before Ollama frees the VRAM; `-1` pins it forever),
`COOK_DB` (default `cooking.db`),
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
- **By hand**: `POST /recipes` with the JSON shape returned by `GET /recipes/{id}`
  (`title`, `servings`, `ingredients[{name, amount, unit}]`, `steps[{text, duration_s, appliance, temp_f}]`).
- **Remove**: `DELETE /recipes/{id}`.

## Tablet app

`src/cooking_assistant_ai/web/` is served at `http://<server>:8000/` (plain HTML, CSS and
JS, no build step). Open it in a kiosk browser on the tablet.

- **Recipes screen** (where a fresh session starts): tick every recipe you're cooking,
  set "Eat in N min" or a clock time, tap "Start cooking". The server loads the whole
  selection, sets plating, and hands the model one planning turn: it builds all the tasks,
  then briefs you out loud from a code-computed summary (how long, when to start, which
  appliances and burners, what to prep first, the ingredients to get out) and names the first
  thing to do. Adding a recipe later re-runs the same flow and only extends the plan. Import
  from a URL and delete from the library live here too.
- **Cook screen**: the merged plan (NOW card, "Then:", full list) with recipe cards
  collapsed beneath; the assistant avatar with its state (ready, listening, thinking,
  speaking, heads up for unprompted speech); the conversation with tool-call chips; timers
  with live countdowns and cancel; the timeline with next action and conflicts; a
  proactivity slider. Hold the big button (or the space bar) to talk, or type.
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

## Cloud inference (optional)

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
  `scratchpad/plan_bench.py <model> <trials>` before switching. `--model qwen-kitchen:14b`
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

## Scheduling semantics the model is told about

- Default is ASAP: start now, or when the tasks it comes `after` end.
- `must_finish_by: "plating"` schedules as late as possible so the task ends at plating.
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
