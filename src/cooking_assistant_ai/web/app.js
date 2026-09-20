/* Kitchen Assistant tablet app. Vanilla JS, no build step.
 * Talks to the FastAPI server on the same origin: /ws/session for everything
 * real-time, HTTP for recipes, pantry and UI-initiated tool calls. */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };

  const app = {
    sessionId: null,
    ws: null,
    state: null,
    stateAt: 0,
    health: { stt: false, tts: false },
    phase: "offline", // offline | idle | listening | thinking | speaking | proactive
    speaking: false,
    speakingProactive: false,
    bubble: null,
    reconnectDelay: 1000,
  };
  try { app.sessionId = localStorage.getItem("sessionId"); } catch (e) { /* ignore */ }

  // ------------------------------------------------------------ helpers
  function fmtTime(iso) {
    if (!iso) return "--:--";
    const d = new Date(iso);
    let h = d.getHours(), m = d.getMinutes();
    const ampm = h >= 12 ? "PM" : "AM";
    h = h % 12 || 12;
    return `${h}:${String(m).padStart(2, "0")} ${ampm}`;
  }
  function fmtClock(iso) {
    const d = iso ? new Date(iso) : new Date();
    let h = d.getHours(), m = d.getMinutes();
    h = h % 12 || 12;
    return `${h}:${String(m).padStart(2, "0")}`;
  }
  function fmtWindow(a, b) {
    if (!a || !b) return "";
    return `${fmtClock(a)}-${fmtClock(b)}`;
  }
  function fmtLeft(s) {
    s = Math.max(0, Math.round(s));
    if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  function fmtDur(s) {
    if (!s) return "";
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.round(s / 60)} min`;
    return `${Math.floor(s / 3600)}h${String(Math.round((s % 3600) / 60)).padStart(2, "0")}`;
  }
  let toastTimer = null;
  function toast(text, isError) {
    const t = $("#toast");
    t.textContent = text;
    t.className = "toast" + (isError ? " error" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 3000);
  }
  async function api(method, path, body) {
    const r = await fetch(path, {
      method, headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    return r.status === 204 ? null : r.json();
  }

  // ------------------------------------------------------------ phases / avatar
  const STATUS = {
    offline: "Offline", idle: "Ready", listening: "Listening", thinking: "Thinking",
    speaking: "Speaking", proactive: "Heads up", error: "Problem",
  };
  let thinkingGuard = null;
  function setPhase(phase, text) {
    app.phase = phase;
    $("#avatar").dataset.state = phase;
    $("#status").textContent = text || STATUS[phase] || phase;
    // Safety net: "thinking" must always end in speech or an error. If neither arrives
    // (server restarted mid-turn, dropped message), fall back to Ready.
    clearTimeout(thinkingGuard);
    if (phase === "thinking") {
      thinkingGuard = setTimeout(() => { if (app.phase === "thinking" && !app.speaking) setPhase("idle"); }, 90000);
    }
  }

  // ------------------------------------------------------------ audio playback
  const player = {
    ctx: null, nextTime: 0, sources: new Set(),
    ensure() {
      if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      if (this.ctx.state === "suspended") this.ctx.resume();
      return this.ctx;
    },
    enqueue(b64, rate) {
      const ctx = this.ensure();
      const bin = atob(b64);
      const n = bin.length / 2;
      const buf = ctx.createBuffer(1, n, rate || 24000);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < n; i++) {
        const lo = bin.charCodeAt(2 * i), hi = bin.charCodeAt(2 * i + 1);
        let v = (hi << 8) | lo;
        if (v >= 0x8000) v -= 0x10000;
        ch[i] = v / 32768;
      }
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      const at = Math.max(ctx.currentTime + 0.02, this.nextTime);
      src.start(at);
      this.nextTime = at + buf.duration;
      const wasBusy = this.sources.size > 0;
      this.sources.add(src);
      if (!wasBusy) this.onPlaying(true);
      src.onended = () => { this.sources.delete(src); if (!this.sources.size) { this.onPlaying(false); this.onDrain(); } };
    },
    stop() {
      const wasBusy = this.sources.size > 0;
      for (const s of this.sources) { try { s.stop(); } catch (e) { /* ignore */ } }
      this.sources.clear();
      this.nextTime = 0;
      if (wasBusy) this.onPlaying(false);
    },
    get busy() { return this.sources.size > 0; },
    onDrain() {},
    onPlaying() {},
  };
  player.onDrain = () => { if (!app.speaking && app.phase !== "listening") setPhase("idle"); };
  // The server needs to know when our speaker is live: it gates barge-in and the echo guard on it.
  player.onPlaying = (playing) => { if (app.ws && app.ws.readyState === WebSocket.OPEN) send({ type: "playback", playing }); };

  // ------------------------------------------------------------ audio capture (push to talk / open mic)
  const AUDIO_CONSTRAINTS = { audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } };

  // Browsers expose the microphone only in a secure context (HTTPS or localhost). On plain
  // HTTP to a LAN address navigator.mediaDevices is undefined entirely, so say so plainly
  // instead of failing with "cannot read properties of undefined".
  function micProblem() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) return null;
    const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (legacy) return null;
    if (!window.isSecureContext || location.protocol === "http:") {
      return `Microphone blocked: this page is on ${location.protocol}//${location.host}, and browsers only allow the mic over HTTPS. Restart the server with -Https and open https://${location.hostname}:${location.port || 443}/ instead.`;
    }
    return "This browser does not provide microphone access.";
  }
  function getMicStream() {
    const problem = micProblem();
    if (problem) return Promise.reject(new Error(problem));
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      return navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
    }
    const legacy = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia).bind(navigator);
    return new Promise((resolve, reject) => legacy(AUDIO_CONSTRAINTS, resolve, reject));
  }

  const recorder = {
    stream: null, ctx: null, source: null, proc: null, chunks: [], rate: 48000, active: false, onChunk: null,
    async start(onChunk) {
      if (this.active) return;
      this.stream = await getMicStream();
      this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
      if (this.ctx.state === "suspended") await this.ctx.resume();
      this.rate = this.ctx.sampleRate;
      this.source = this.ctx.createMediaStreamSource(this.stream);
      this.proc = this.ctx.createScriptProcessor(4096, 1, 1);
      this.chunks = [];
      this.onChunk = onChunk || null;
      this.proc.onaudioprocess = (e) => {
        const f32 = new Float32Array(e.inputBuffer.getChannelData(0));
        if (this.onChunk) this.onChunk(downsampleTo16k(f32, this.rate));  // streaming mode
        else this.chunks.push(f32);                                       // push-to-talk mode
      };
      this.source.connect(this.proc);
      this.proc.connect(this.ctx.destination);
      this.active = true;
    },
    stop() {
      if (!this.active) return null;
      this.active = false;
      try { this.source.disconnect(); this.proc.disconnect(); } catch (e) { /* ignore */ }
      this.stream.getTracks().forEach((t) => t.stop());
      let total = 0;
      for (const c of this.chunks) total += c.length;
      const merged = new Float32Array(total);
      let off = 0;
      for (const c of this.chunks) { merged.set(c, off); off += c.length; }
      this.chunks = [];
      return { pcm16: downsampleTo16k(merged, this.rate), seconds: total / this.rate };
    },
  };
  function downsampleTo16k(f32, rate) {
    const ratio = rate / 16000;
    const n = Math.floor(f32.length / ratio);
    const out = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      // average the source samples covering this output sample (cheap anti-aliasing)
      const a = Math.floor(i * ratio), b = Math.min(f32.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = a; j < b; j++) sum += f32[j];
      const v = Math.max(-1, Math.min(1, sum / Math.max(1, b - a)));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }
  function toBase64(int16) {
    const bytes = new Uint8Array(int16.buffer);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  // ------------------------------------------------------------ websocket
  function send(msg) {
    if (app.ws && app.ws.readyState === WebSocket.OPEN) app.ws.send(JSON.stringify(msg));
    else toast("Not connected", true);
  }
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/session${app.sessionId ? "?session_id=" + encodeURIComponent(app.sessionId) : ""}`;
    const ws = new WebSocket(url);
    app.ws = ws;
    $("#conn-dot").className = "dot busy";
    $("#conn-text").textContent = "connecting";
    ws.onopen = () => {
      app.reconnectDelay = 1000;
      $("#conn-dot").className = "dot ok";
      $("#conn-text").textContent = "connected";
      setPhase("idle");
      if (app.restarting) {                 // the supervisor gave us a fresh process
        app.restarting = false;
        toast("Server restarted");
        loadHealth();
      }
      loadServerStatus();
      document.dispatchEvent(new Event("ws-open"));
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handle(msg);
    };
    ws.onclose = () => {
      $("#conn-dot").className = "dot";
      $("#conn-text").textContent = app.restarting ? "restarting" : "reconnecting";
      setPhase("offline");
      setTimeout(connect, app.reconnectDelay);
      app.reconnectDelay = Math.min(10000, app.reconnectDelay * 1.6);
    };
    ws.onerror = () => { /* onclose follows */ };
  }
  function handle(msg) {
    switch (msg.type) {
      case "session":
        app.sessionId = msg.session_id;
        try { localStorage.setItem("sessionId", app.sessionId); } catch (e) { /* ignore */ }
        break;
      case "state":
        app.state = msg;
        app.stateAt = Date.now();
        renderState();
        // First load with nothing cooking: start on the recipe chooser.
        if (!autoScreened && !msg.progress.recipes.length) showScreen("recipes");
        autoScreened = true;
        break;
      case "transcript":
        addBubble("user", msg.text);
        if (app.phase !== "listening") setPhase("thinking");
        break;
      case "speech_start":
        app.speaking = true;
        app.speakingProactive = !!msg.proactive;
        app.bubble = addBubble("assistant" + (msg.proactive ? " proactive" : ""), "");
        setPhase(msg.proactive ? "proactive" : "speaking");
        break;
      case "text":
        if (!app.bubble) app.bubble = addBubble("assistant" + (msg.proactive ? " proactive" : ""), "");
        app.bubble.textContent += msg.text;
        scrollConversation();
        if (app.suggesting) {
          const out = $("#suggest-out");
          if (out.classList.contains("waiting")) { out.className = "suggest-out"; out.textContent = ""; }
          out.textContent += msg.text;
        }
        break;
      case "audio":
        try { player.enqueue(msg.data, msg.sample_rate); } catch (e) { console.warn("audio", e); }
        break;
      case "speech_end":
        if (app.suggesting) {
          const out = $("#suggest-out");
          if (msg.text) { out.className = "suggest-out"; out.textContent = msg.text; }
          app.suggesting = false;
          renderChoices();   // the model may have loaded recipes; refresh the ticks
        }
        app.speaking = false;
        if (app.bubble && !app.bubble.textContent && msg.text) app.bubble.textContent = msg.text;
        if (app.bubble && !app.bubble.textContent) app.bubble.remove();
        app.bubble = null;
        if (!player.busy) setPhase("idle");
        break;
      case "tool_call":
        addChip(msg);
        // Only the model's own tool calls mean a reply is coming; UI actions (step done,
        // cancel timer, add recipe) are complete once the state message lands.
        if (msg.source !== "ui" && app.phase === "idle") setPhase("thinking");
        break;
      case "preflight":
        renderPreflight(msg);
        break;
      case "notice":
        if (msg.level === "warning" || msg.level === "error") toast(msg.text, msg.level === "error");
        break;
      case "error":
        toast(msg.text, true);
        setPhase("error", "Problem");
        setTimeout(() => { if (app.phase === "error") setPhase("idle"); }, 2500);
        break;
      case "barge_in":
        break;
      case "listen":
        setListenUi(!!msg.on);
        break;
      case "vad":
        $("#listen-toggle").classList.toggle("hearing", !!msg.speaking);
        if (msg.speaking) {
          if (!app.speaking && !player.busy) setPhase("listening");
        } else if (app.phase === "listening") {
          setPhase("thinking");
        }
        break;
      case "stop_playback":
        player.stop();
        app.speaking = false;
        app.bubble = null;
        setPhase("listening");
        break;
      case "dropped":
        if (msg.reason === "echo") console.debug("echo dropped:", msg.text);
        break;
      default:
        break;
    }
  }

  // ------------------------------------------------------------ conversation
  function addBubble(cls, text) {
    const conv = $("#conversation");
    const b = el("div", "bubble " + cls, text);
    conv.appendChild(b);
    while (conv.children.length > 80) conv.removeChild(conv.firstChild);
    scrollConversation();
    return b;
  }
  // A tool call that worked does not belong in the conversation. Its effect is already on
  // screen - the plan, the timers, the step - and a run of them pushed the sentence the
  // assistant was actually saying up out of view. Rejections stay: they are rare, they are
  // the exception the cook may need to act on, and the UNRESOLVED panel only carries the
  // ones that are still outstanding.
  function addChip(msg) {
    if (msg.ok) return;
    const args = msg.args && Object.keys(msg.args).length
      ? " " + Object.entries(msg.args).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(", ")
      : "";
    const text = "✗ " + msg.name + args + " — " + (msg.message || "rejected");
    const chip = el("div", "chip rejected" + (msg.source === "ui" ? " ui" : ""), text);
    $("#conversation").appendChild(chip);
    scrollConversation();
  }
  function scrollConversation() {
    const conv = $("#conversation");
    conv.scrollTop = conv.scrollHeight;
  }

  // ------------------------------------------------------------ state rendering
  function renderState() {
    const s = app.state;
    if (!s) return;
    $("#plating").textContent = s.target_plating ? "plating " + fmtTime(s.target_plating) : "";
    const p = $("#proactivity");
    if (document.activeElement !== p) p.value = s.proactivity;
    renderPlan(s.plan);
    renderPlanSummary(s.plan.summary);
    renderAppliances(s.appliances);
    renderUnresolved(s.unresolved);
    renderStepPanel(s.progress.recipes);
    renderRecipes(s.progress.recipes);
    if (screen === "recipes") renderChoices();
    $("#recipes-count").textContent = s.progress.recipes.length ? `(${s.progress.recipes.length})` : "";
    renderTimers();
    renderTimeline(s.timeline);
    const nxt = s.timeline.next_action;
    $("#next-action").textContent = nxt ? `Next: ${nxt.text} at ${fmtTime(nxt.at)}` : "";
  }

  function itemMeta(it) {
    return [it.duration_s ? fmtDur(it.duration_s) : null,
      it.appliance ? it.appliance.replace("_", " ") + (it.temp_f ? ` ${it.temp_f}°F` : "") : null,
      it.status === "unscheduled" ? "not scheduled yet" : null,
      it.note ? "note: " + it.note : null].filter(Boolean).join(" · ");
  }
  function completeItem(it) {
    if (it.kind === "prep") return uiTool("POST", `/session/${app.sessionId}/prep/${it.id}/complete`);
    return uiTool("POST", `/session/${app.sessionId}/steps/${it.id}/complete`);
  }
  function renderPlanSummary(sum) {
    const box = $("#plan-summary");
    if (!sum || !sum.recipes.length) { box.textContent = ""; return; }
    const bits = [`${sum.steps} steps`];
    if (sum.planned && sum.span_s) bits.push(fmtDur(sum.span_s));
    if (sum.planned && sum.plating_at) bits.push("plate " + fmtTime(sum.plating_at));
    if (sum.appliances.length) bits.push(sum.appliances.map((a) => a.replace(/ \(.*\)/, "")).join(", "));
    box.textContent = "· " + bits.join(" · ");
  }
  function renderPlan(plan) {
    const nowBox = $("#plan-now"), nextBox = $("#plan-next"), list = $("#plan-list");
    nowBox.innerHTML = ""; nextBox.innerHTML = ""; list.innerHTML = "";
    if (!plan || !plan.items.length) {
      nowBox.className = "plan-now";
      nowBox.appendChild(el("div", "text", "Nothing planned yet."));
      nowBox.appendChild(el("div", "meta", "Pick recipes on the Recipes tab and tap Start cooking."));
      $("#plan-count").textContent = "";
      return;
    }
    const items = plan.items;
    const now = items.find((i) => i.id === plan.now_id);
    const remaining = items.filter((i) => i.status !== "done" && i.status !== "skipped");
    const next = remaining.find((i) => i !== now);
    if (now) {
      nowBox.className = "plan-now" + (now.kind === "prep" ? " prep" : "");
      const who = now.kind === "prep" ? "Prep for all recipes" : `${now.recipe_title} · step ${now.step_n}${now.at ? " · " + fmtTime(now.at) : ""}`;
      nowBox.appendChild(el("div", "who", who));
      nowBox.appendChild(el("div", "text", now.text));
      const meta = itemMeta(now);
      if (meta) nowBox.appendChild(el("div", "meta", meta));
      const actions = el("div", "actions");
      const done = el("button", "small", now.kind === "prep" ? "All prepped" : "Done");
      done.onclick = () => completeItem(now);
      actions.appendChild(done);
      nowBox.appendChild(actions);
    } else {
      nowBox.className = "plan-now";
      nowBox.appendChild(el("div", "text", "Everything is done."));
    }
    if (next) {
      nextBox.appendChild(document.createTextNode("Then: "));
      nextBox.appendChild(el("b", null, next.text));
      nextBox.appendChild(document.createTextNode(next.kind === "prep" ? "" : ` (${next.recipe_title}${next.at ? ", " + fmtTime(next.at) : ""})`));
    }
    $("#plan-count").textContent = `(${plan.done_count}/${plan.total} done)`;
    for (const it of items) {
      const row = el("div", "plan-item " + it.status + (it.kind === "prep" ? " prep" : ""));
      row.appendChild(el("div", "when", it.kind === "prep" ? "prep" : it.at ? fmtClock(it.at) : "--:--"));
      const body = el("div");
      body.appendChild(el("div", "text", it.text));
      const who = it.kind === "prep" ? "shared prep" : `${it.recipe_title} · step ${it.step_n}` + (itemMeta(it) ? " · " + itemMeta(it) : "");
      body.appendChild(el("div", "who", who));
      row.appendChild(body);
      if (it.status !== "done" && it.status !== "skipped") {
        row.style.cursor = "pointer";
        row.title = "Tap to mark done";
        row.onclick = () => completeItem(it);
      }
      list.appendChild(row);
    }
  }


  // ------------------------------------------------------------ the step you are on
  //
  // Voice alone was clunky for this: a spoken list of six things with six quantities is not
  // something anyone holds in their head over a hot pan, and the cook kept having to ask "how
  // much, and for which dish". The step and its ingredients now sit on screen, large, and
  // stay there until the step is actually done.
  //
  // The glyphs are keyed off the grocery key the model already assigns at import, matched
  // longest-first so "spring onion" does not land on "onion". They are emoji on purpose:
  // legible at a glance from across a kitchen, no assets to fetch, and nothing to go stale
  // when the tablet is offline.
  const FOOD_ICONS = [
    ["garlic powder", "\u{1F9C2}"], ["onion powder", "\u{1F9C2}"],
    ["chili powder", "\u{1F9C2}"], ["curry powder", "\u{1F9C2}"],
    ["spring onion", "\u{1F33F}"], ["green onion", "\u{1F33F}"], ["onion", "\u{1F9C5}"],
    ["garlic", "\u{1F9C4}"], ["ginger", "\u{1FAD5}"], ["chili", "\u{1F336}"],
    ["pepper", "\u{1FAD1}"], ["tomato", "\u{1F345}"], ["potato", "\u{1F954}"],
    ["carrot", "\u{1F955}"], ["cabbage", "\u{1F96C}"], ["spinach", "\u{1F96C}"],
    ["kale", "\u{1F96C}"], ["lettuce", "\u{1F96C}"], ["broccoli", "\u{1F966}"],
    ["mushroom", "\u{1F344}"], ["aubergine", "\u{1F346}"], ["eggplant", "\u{1F346}"],
    ["courgette", "\u{1F952}"], ["zucchini", "\u{1F952}"], ["cucumber", "\u{1F952}"],
    ["corn", "\u{1F33D}"], ["avocado", "\u{1F951}"], ["lemon", "\u{1F34B}"],
    ["lime", "\u{1F34B}"], ["coconut milk", "\u{1F965}"], ["coconut", "\u{1F965}"],
    ["tofu", "\u{1F9CA}"], ["tempeh", "\u{1F9CA}"], ["chickpea", "\u{1FAD8}"],
    ["lentil", "\u{1FAD8}"], ["bean", "\u{1FAD8}"], ["pea", "\u{1FAD8}"],
    ["rice", "\u{1F35A}"], ["noodle", "\u{1F35C}"], ["pasta", "\u{1F35D}"],
    ["bread", "\u{1F956}"], ["baguette", "\u{1F956}"], ["flour", "\u{1F33E}"],
    ["quinoa", "\u{1F33E}"], ["oat", "\u{1F33E}"], ["oil", "\u{1FAD2}"],
    ["butter", "\u{1F9C8}"], ["cheese", "\u{1F9C0}"], ["milk", "\u{1F95B}"],
    ["cream", "\u{1F95B}"], ["yogurt", "\u{1F95B}"], ["egg", "\u{1F95A}"],
    ["salt", "\u{1F9C2}"], ["sugar", "\u{1F36C}"], ["honey", "\u{1F36F}"],
    ["maple", "\u{1F36F}"], ["syrup", "\u{1F36F}"], ["vinegar", "\u{1F9F4}"],
    ["soy sauce", "\u{1F9F4}"], ["sauce", "\u{1F9F4}"], ["paste", "\u{1F962}"],
    ["stock", "\u{1F963}"], ["broth", "\u{1F963}"], ["water", "\u{1F4A7}"],
    ["wine", "\u{1F377}"], ["nut", "\u{1F95C}"], ["cashew", "\u{1F95C}"],
    ["peanut", "\u{1F95C}"], ["seed", "\u{1F33B}"], ["herb", "\u{1F33F}"],
    ["parsley", "\u{1F33F}"], ["coriander", "\u{1F33F}"], ["cilantro", "\u{1F33F}"],
    ["basil", "\u{1F33F}"], ["thyme", "\u{1F33F}"], ["yeast", "\u{1F9C2}"],
    ["turmeric", "\u{1F9C2}"], ["cumin", "\u{1F9C2}"], ["paprika", "\u{1F9C2}"],
    ["masala", "\u{1F9C2}"], ["curry", "\u{1F9C2}"], ["spice", "\u{1F9C2}"],
    ["powder", "\u{1F9C2}"],
  ].sort((a, b) => b[0].length - a[0].length);

  function foodIcon(need) {
    const hay = ((need.key || "") + " " + (need.name || "")).toLowerCase();
    for (const [word, glyph] of FOOD_ICONS) if (hay.includes(word)) return glyph;
    return "\u{1F372}";
  }

  // Which dishes to show, and in which order: the ones the model has moved the cook to,
  // most recent first. This used to pick the single dish furthest along, which is a rule
  // invented in the browser - with two dishes interleaved it showed whichever happened to
  // have more steps ticked off, so the cook watched one recipe run to the end while the
  // model was walking them between both. advance_step decides; this follows.
  function dishesOnTheGo(recipes) {
    const withStep = (recipes || [])
      .map((r) => ({ r, cur: (r.steps || []).find((st) => st.n === r.current_step) }))
      .filter((d) => d.cur);
    const led = withStep.filter((d) => d.r.focus >= 0)
      .sort((a, b) => a.r.focus - b.r.focus);
    if (led.length) return led.slice(0, 2);
    // Nothing advanced yet: fall back to whatever has work, so the first step is visible
    // before the model has moved anyone anywhere.
    return withStep.slice(0, 1);
  }

  function renderStepPanel(recipes) {
    const box = $("#step-now");
    const going = dishesOnTheGo(recipes);
    if (!going.length) { box.hidden = true; return; }
    const { r, cur } = going[0];

    $("#step-dish").textContent = r.title;
    $("#step-count").textContent = `Step ${cur.n} of ${r.steps.length}`;
    $("#step-text").textContent = cur.text;

    const needs = $("#step-needs");
    needs.innerHTML = "";
    for (const need of cur.needs || []) {
      const tile = el("div", "need");
      tile.appendChild(el("span", "need-icon", foodIcon(need)));
      tile.appendChild(el("span", "need-qty", need.qty || ""));
      tile.appendChild(el("span", "need-name", need.name));
      needs.appendChild(tile);
    }
    needs.hidden = !(cur.needs || []).length;

    const meta = [cur.duration_s ? fmtDur(cur.duration_s) : null,
      cur.appliance ? cur.appliance.replace("_", " ") + (cur.temp_f ? ` ${cur.temp_f}°F` : "") : null,
      cur.note ? "note: " + cur.note : null].filter(Boolean).join("  ·  ");
    $("#step-meta").textContent = meta;

    const done = $("#step-done");
    done.textContent = `Done with step ${cur.n}`;
    done.onclick = () => uiTool("POST", `/session/${app.sessionId}/steps/${cur.id}/complete`);

    const next = r.steps.find((st) => st.n > cur.n && st.status === "pending");
    $("#step-next").textContent = next ? `Then: ${next.text}` : "";

    // The other dish on the go, small but present, so interleaving is visible rather than
    // something the cook has to remember.
    const also = $("#step-also");
    also.innerHTML = "";
    for (const d of going.slice(1)) {
      const card = el("div", "also");
      const head = el("div", "also-head");
      head.appendChild(el("span", "also-dish", d.r.title));
      head.appendChild(el("span", "also-count", `Step ${d.cur.n} of ${d.r.steps.length}`));
      card.appendChild(head);
      card.appendChild(el("div", "also-text", d.cur.text));
      const tiles = el("div", "also-needs");
      for (const need of d.cur.needs || []) {
        const tile = el("span", "also-need");
        tile.appendChild(el("span", "also-icon", foodIcon(need)));
        tile.appendChild(el("span", "also-qty", `${need.qty || ""} ${need.name}`.trim()));
        tiles.appendChild(tile);
      }
      if ((d.cur.needs || []).length) card.appendChild(tiles);
      const done = el("button", "small", `Done with step ${d.cur.n}`);
      done.onclick = () => uiTool("POST", `/session/${app.sessionId}/steps/${d.cur.id}/complete`);
      card.appendChild(done);
      also.appendChild(card);
    }
    also.hidden = going.length < 2;
    box.hidden = false;
  }

  // Progress as a row of steps rather than "4/9" and a bar. Each step is a dot the cook can
  // tap: filled for done, ringed for where they are, hollow for what is left, struck for
  // skipped. It says the same thing as the fraction and also says which ones, and tapping is
  // the fastest way to correct the model when it has not kept up.
  function stepDots(r) {
    const row = el("div", "dots");
    row.style.setProperty("--hue", DISH_HUES[
      (app.state.progress.recipes.findIndex((x) => x.id === r.id) + DISH_HUES.length)
      % DISH_HUES.length]);
    for (const st of r.steps) {
      const dot = el("button", "dot " + st.status + (st.n === r.current_step ? " here" : ""));
      dot.title = `Step ${st.n}: ${st.text}`;
      dot.textContent = st.status === "done" ? "✓" : (st.status === "skipped" ? "–" : st.n);
      if (st.status !== "done") {
        dot.onclick = () => uiTool("POST", `/session/${app.sessionId}/steps/${st.id}/complete`);
      }
      row.appendChild(dot);
    }
    return row;
  }

  function renderRecipes(recipes) {
    const box = $("#recipe-cards");
    box.innerHTML = "";
    if (!recipes.length) {
      box.appendChild(el("div", "empty", "No recipes loaded. Tap + Add recipe, or just ask for one."));
      return;
    }
    for (const r of recipes) {
      const card = el("div", "card");
      const title = el("div", "title");
      title.appendChild(el("b", null, r.title));
      const done = r.steps.filter((st) => st.status === "done").length;
      title.appendChild(el("span", "progress", `serves ${r.servings}`));
      card.appendChild(title);
      card.appendChild(stepDots(r));

      // The step itself lives in the panel in the middle of the screen, not repeated here:
      // one place to look, at a size worth looking at.
      const cur = r.steps.find((st) => st.n === r.current_step);
      if (!cur) card.appendChild(el("div", "step-now", "All steps done"));
      if (r.changes) card.appendChild(el("div", "changes", r.changes));

      const actions = el("div", "actions");
      const rm = el("button", "small danger", "Remove");
      rm.onclick = () => uiTool("DELETE", `/session/${app.sessionId}/recipes/${r.id}`);
      actions.appendChild(rm);
      card.appendChild(actions);

      const det = el("details");
      det.appendChild(el("summary", null, "All steps and ingredients"));
      const ul = el("ul");
      for (const st of r.steps) {
        const li = el("li", st.status === "done" ? "done" : st.status === "skipped" ? "skipped" : st.n === r.current_step ? "current" : "", `${st.n}. ${st.text}`);
        if (st.substituted) li.appendChild(el("span", "sub", " (swapped)"));
        ul.appendChild(li);
      }
      det.appendChild(ul);
      const ing = el("ul");
      for (const i of r.ingredients) {
        const li = el("li", null, i.text);
        if (i.substituted_for) li.appendChild(el("span", "sub", ` (instead of ${i.substituted_for}${i.note ? ", " + i.note : ""})`));
        ing.appendChild(li);
      }
      det.appendChild(ing);
      card.appendChild(det);
      box.appendChild(card);
    }
  }

  function renderTimers() {
    const s = app.state;
    const box = $("#timers");
    box.innerHTML = "";
    if (!s || !s.timers.length) {
      box.appendChild(el("div", "empty", "No timers running"));
      return;
    }
    const elapsed = (Date.now() - app.stateAt) / 1000;
    for (const t of s.timers) {
      const left = t.seconds_left - elapsed;
      const card = el("div", "timer");
      card.appendChild(el("div", "label", t.label));
      card.appendChild(el("div", "left" + (left <= 0 ? " due" : left < 60 ? " soon" : ""), left <= 0 ? "due" : fmtLeft(left)));
      card.appendChild(el("div", "hint", t.hint ? `then: ${t.hint}` : `ends ${fmtTime(t.end_at)}`));
      const c = el("button", "small cancel", "Cancel");
      c.onclick = () => uiTool("DELETE", `/session/${app.sessionId}/timers/${t.id}`);
      card.appendChild(c);
      box.appendChild(card);
    }
  }

  // A timeline the way a timeline is drawn: one line, events pinned along it in the order
  // they happen, each leading to the next. The first attempt was a Gantt chart - lanes of
  // equipment with proportional bars - which is a project plan, not a cook's line, and in a
  // narrow column it truncated every label to "chi..." and "ud...".
  //
  // Laid out in pixels per minute rather than percentages, so an hour always looks like an
  // hour and nothing collapses into an unreadable smear; the track scrolls instead. Events
  // alternate above and below the line, which is how a historical timeline keeps labels apart.
  const DISH_HUES = [28, 190, 140, 330, 265, 95];
  const PX_PER_MIN = 9;
  const MIN_EVENT_GAP = 116;        // px, so two cards can never sit on top of each other

  function dishHue(recipeId, order) {
    const n = order.indexOf(recipeId);
    return DISH_HUES[(n < 0 ? 0 : n) % DISH_HUES.length];
  }

  function laneName(appliance) {
    if (!appliance) return "";
    const [family, ring] = appliance.split(":");
    if (family === "stovetop") return ring ? "Burner " + ring : "Hob";
    return family.replace("_", " ").replace(/^./, (c) => c.toUpperCase());
  }

  function renderTimeline(tl) {
    const box = $("#timeline");
    box.innerHTML = "";
    $("#conflicts").textContent = tl.conflicts.length ? "Conflicts: " + tl.conflicts.join("; ") : "";
    const placed = (tl.tasks || []).filter((t) => t.start_at && t.end_at)
      .sort((a, b) => (a.start_at < b.start_at ? -1 : 1));
    if (!placed.length) {
      box.appendChild(el("div", "empty", "No plan yet. Pick recipes on the Recipes tab."));
      return;
    }

    const now = app.state && app.state.now ? new Date(app.state.now) : new Date();
    const stamps = placed.flatMap((t) => [new Date(t.start_at), new Date(t.end_at)]).concat(now);
    const from = new Date(Math.min(...stamps));
    const to = new Date(Math.max(...stamps));
    const minutes = Math.max((to - from) / 60000, 10);
    const at = (d) => ((new Date(d) - from) / 60000) * PX_PER_MIN;

    // Nudge cards right only far enough that they do not overlap, keeping the order intact.
    // The dot stays on the true time; only the card slides, and the stalk leans to show it.
    // Crowding is counted per side: cards alternate above and below the line, so two events
    // a few minutes apart sit opposite each other rather than shoving one another along.
    // Counting them together made a cascade in which every card ended up evenly spaced and
    // none of them was anywhere near its own time.
    const lastOn = [-Infinity, -Infinity];
    const events = placed.map((t, i) => {
      const side = i % 2;
      const x = at(t.start_at);
      const slot = Math.max(x, lastOn[side] + MIN_EVENT_GAP);
      lastOn[side] = slot;
      return { t, x, slot, above: side === 0 };
    });

    const width = Math.max(minutes * PX_PER_MIN, ...lastOn.filter(Number.isFinite)) + 150;
    const order = [...new Set(placed.map((t) => t.recipe_id))];

    const scroller = el("div", "tl-scroll");
    const track = el("div", "tl-track");
    track.style.width = width + "px";
    track.appendChild(el("div", "tl-spine"));

    events.forEach((ev) => {
      const { t, x, slot, above } = ev;
      const hue = dishHue(t.recipe_id, order);

      const dot = el("div", "tl-dot " + t.status);
      dot.style.left = x + "px";
      dot.style.setProperty("--hue", hue);
      track.appendChild(dot);

      // How long it runs, drawn on the line itself between this event and its end.
      const run = el("div", "tl-run " + t.status);
      run.style.left = x + "px";
      run.style.width = Math.max(at(t.end_at) - x, 2) + "px";
      run.style.setProperty("--hue", hue);
      track.appendChild(run);

      const stalk = el("div", "tl-stalk " + (above ? "up" : "down"));
      stalk.style.left = Math.min(x, slot) + "px";
      stalk.style.width = Math.abs(slot - x) + "px";
      stalk.style.setProperty("--hue", hue);
      track.appendChild(stalk);

      const card = el("div", "tl-event " + (above ? "up" : "down") + " " + t.status);
      card.style.left = slot + "px";
      card.style.setProperty("--hue", hue);
      card.appendChild(el("div", "tl-when", fmtTime(t.start_at)));
      card.appendChild(el("div", "tl-what", t.label));
      const where = laneName(t.appliance);
      if (where || t.temp_f) {
        const line = el("div", "tl-where");
        line.appendChild(applianceIcon((t.appliance || "").split(":")[0]));
        line.appendChild(document.createTextNode(
          where + (t.temp_f ? ` ${t.temp_f}°` : "")));
        card.appendChild(line);
      }
      track.appendChild(card);
    });

    const marker = el("div", "tl-now");
    marker.style.left = at(now) + "px";
    marker.appendChild(el("span", "tl-now-pill", "now"));
    track.appendChild(marker);

    scroller.appendChild(track);
    box.appendChild(scroller);

    const key = el("div", "tl-key");
    for (const rid of order) {
      const dish = (app.state.progress.recipes.find((r) => r.id === rid) || {}).title || rid;
      const item = el("span", "tl-key-item");
      const dot = el("i");
      dot.style.setProperty("--hue", dishHue(rid, order));
      item.appendChild(dot);
      item.appendChild(document.createTextNode(dish));
      key.appendChild(item);
    }
    box.appendChild(key);

    // Open on what is happening rather than on the beginning of the evening.
    requestAnimationFrame(() => { scroller.scrollLeft = Math.max(0, at(now) - 90); });
  }


  async function renderChoices() {
    const box = $("#recipe-choices");
    box.innerHTML = "";
    try {
      choice.library = await api("GET", "/recipes");
      const d = await api("GET", "/meal-options");
      choice.diet = d.diet;
      $("#diet-select").value = d.diet;
      choice.blocked = new Map(d.recipes.filter((r) => !r.diet_ok).map((r) => [r.id, r.diet_note]));
    } catch (e) {
      box.appendChild(el("div", "empty", e.message));
      return;
    }
    const loaded = new Set((app.state ? app.state.progress.recipes : []).map((r) => r.id));
    if (!choice.selected.size) for (const id of loaded) choice.selected.add(id);
    if (!choice.library.length) box.appendChild(el("div", "empty", "No recipes in the library yet. Import one below."));
    for (const r of choice.library) {
      // A recipe the diet forbids cannot be selected at all: the restriction is not advisory.
      const blocked = choice.blocked && choice.blocked.get(r.id);
      if (blocked) choice.selected.delete(r.id);
      const card = el("label", "choice" + (choice.selected.has(r.id) ? " selected" : "")
        + (loaded.has(r.id) ? " loaded" : "") + (blocked ? " blocked" : ""));
      const cb = el("input");
      cb.type = "checkbox";
      cb.disabled = !!blocked;
      cb.checked = choice.selected.has(r.id);
      cb.onchange = () => {
        if (cb.checked) choice.selected.add(r.id); else choice.selected.delete(r.id);
        card.classList.toggle("selected", cb.checked);
        updateStartButton();
      };
      card.appendChild(cb);
      const body = el("div");
      body.appendChild(el("div", "title", r.title));
      body.appendChild(el("div", "sub", `serves ${r.servings} · ${r.steps} steps` + (loaded.has(r.id) ? " · in this session" : "")));
      if (blocked) body.appendChild(el("div", "diet-block", `${choice.diet}: ${blocked}`));
      card.appendChild(body);
      const del = el("button", "small danger del", "Delete");
      del.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!confirm(`Delete "${r.title}" from the library?`)) return;
        try { await api("DELETE", `/recipes/${r.id}`); choice.selected.delete(r.id); renderChoices(); }
        catch (err) { toast(err.message, true); }
      };
      card.appendChild(del);
      const det = el("details");
      det.appendChild(el("summary", null, "Ingredients and steps"));
      det.onclick = (e) => e.stopPropagation();
      det.appendChild(el("div", "muted", "loading…"));
      det.addEventListener("toggle", async () => {
        if (!det.open || det.dataset.loaded) return;
        det.dataset.loaded = "1";
        try {
          const full = await api("GET", `/recipes/${r.id}` + (app.sessionId ? `?session=${app.sessionId}` : ""));
          det.innerHTML = "";
          det.appendChild(el("summary", null, "Ingredients and steps"));
          const ul = el("ul");
          for (const i of full.ingredients) {
            const li = el("li", null, i.text || `${i.amount % 1 ? i.amount.toFixed(2) : i.amount}${i.unit ? " " + i.unit : ""} ${i.name}`);
            if (i.substituted_for) li.appendChild(el("span", "sub", ` (instead of ${i.substituted_for})`));
            ul.appendChild(li);
          }
          det.appendChild(ul);
          const ol = el("ol");
          for (const s of full.steps) ol.appendChild(el("li", null, s.text));
          det.appendChild(ol);
        } catch (err) { det.appendChild(el("div", "muted", err.message)); }
      });
      card.appendChild(det);
      box.appendChild(card);
    }
    updateStartButton();
  }
  function updateStartButton() {
    const btn = $("#btn-start-cooking");
    const n = choice.selected.size;
    btn.disabled = n === 0;
    btn.textContent = n ? `Start cooking (${n})` : "Start cooking";
  }
  async function startCooking() {
    const ids = [...choice.selected];
    if (!ids.length) return;
    // No serving deadline by default: batch cooking has none, and a deadline forces every
    // task as late as possible. Tick the box only when eating at a set time.
    const body = { recipe_ids: ids };
    const at = $("#plating-time").value;
    if ($("#plating-on").checked && at) body.target_plating = at;
    const btn = $("#btn-start-cooking");
    btn.disabled = true;
    btn.textContent = "Planning…";
    try {
      player.ensure();
      await api("POST", `/session/${app.sessionId}/start`, body);
      showScreen("cook");
      setPhase("thinking", "Planning your meal");
    } catch (e) {
      toast(e.message, true);
    } finally {
      updateStartButton();
    }
  }
  // Ask for suggestions without leaving the Recipes screen: the answer streams in here as
  // well as into the Cook conversation, so picking recipes and asking about them are one flow.
  function askForSuggestions(e) {
    if (e) e.preventDefault();
    const n = Number($("#suggest-meals").value) || 8;
    const out = $("#suggest-out");
    out.hidden = false;
    out.className = "suggest-out waiting";
    out.textContent = "Thinking about what you can make…";
    app.suggesting = true;
    player.ensure();
    send({ type: "text", text: `I want to batch cook about ${n} portions this week. Looking at what's in my pantry, what should I make?` });
    setPhase("thinking");
  }

  function bindRecipes() {
    $("#btn-start-cooking").addEventListener("click", startCooking);
    $("#suggest-form").addEventListener("submit", askForSuggestions);
    $("#plating-on").addEventListener("change", (e) => { $("#plating-time").disabled = !e.target.checked; });
    $("#diet-select").addEventListener("change", async (e) => {
      try {
        await api("PUT", "/diet", { diet: e.target.value });
        toast(e.target.value === "none" ? "No dietary restriction" : `Diet set to ${e.target.value}`);
        renderChoices();
      } catch (err) { toast(err.message, true); }
    });
    $("#import-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = $("#import-url");
      const url = input.value.trim();
      if (!url) return;
      const btn = e.target.querySelector("button");
      btn.disabled = true;
      btn.textContent = "Importing…";
      try {
        const r = await api("POST", "/recipes", { url });
        toast(`Imported "${r.title}" (${r.steps.length} steps)`);
        choice.selected.add(r.id);
        input.value = "";
        renderChoices();
      } catch (err) {
        toast("Import failed: " + err.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = "Import";
      }
    });
  }
  // ------------------------------------------------------------ push to talk
  let pttDown = false;
  async function pttStart(ev) {
    if (ev) ev.preventDefault();
    const problem = micProblem();
    if (problem) { toast(problem, true); return; }
    if (pttDown || !app.health.stt) return;
    pttDown = true;
    if (app.speaking || player.busy) { send({ type: "barge_in" }); player.stop(); app.speaking = false; app.bubble = null; }
    player.ensure();
    try {
      await recorder.start();
    } catch (e) {
      pttDown = false;
      toast("Microphone unavailable: " + e.message, true);
      return;
    }
    $("#ptt").classList.add("recording");
    $("#ptt").textContent = "Listening… release to send";
    setPhase("listening");
  }
  function pttStop(ev) {
    if (ev) ev.preventDefault();
    if (!pttDown) return;
    pttDown = false;
    const btn = $("#ptt");
    btn.classList.remove("recording");
    btn.textContent = "Hold to talk";
    const rec = recorder.stop();
    if (!rec || rec.seconds < 0.4) { setPhase("idle"); return; }
    send({ type: "audio", data: toBase64(rec.pcm16), sample_rate: 16000 });
    setPhase("thinking", "Listening done, thinking");
  }
  // ------------------------------------------------------------ always listening (open mic)
  const listen = { on: false, wanted: false };
  try { listen.wanted = localStorage.getItem("listen") === "1"; } catch (e) { /* ignore */ }

  function setListenUi(on) {
    listen.on = on;
    const btn = $("#listen-toggle");
    btn.classList.toggle("on", on);
    btn.classList.remove("hearing");
    // The button carries a lit dot now, so setting textContent would wipe it out.
    $("#listen-label").textContent = on ? "Listening" : "Listen";
    btn.title = on ? "Always listening: on" : "Always listening: off";
    $("#ptt").hidden = on;
    if (on && app.phase === "idle") setPhase("idle", "Listening for you");
    if (!on && app.phase === "idle") setPhase("idle");
  }
  async function listenStart() {
    if (!app.health.stt) { toast("Server has no speech recognition", true); return; }
    const problem = micProblem();
    if (problem) { toast(problem, true); return; }
    if (pttDown) pttStop();
    player.ensure();
    try {
      await recorder.start((pcm16) => send({ type: "audio_chunk", data: toBase64(pcm16) }));
    } catch (e) {
      toast("Microphone unavailable: " + e.message, true);
      return;
    }
    send({ type: "listen", on: true });
    try { localStorage.setItem("listen", "1"); } catch (e) { /* ignore */ }
  }
  function listenStop() {
    recorder.stop();
    send({ type: "listen", on: false });
    setListenUi(false);
    try { localStorage.setItem("listen", "0"); } catch (e) { /* ignore */ }
  }
  function bindListen() {
    $("#listen-toggle").addEventListener("click", () => { if (listen.on) listenStop(); else listenStart(); });
    // Resume open-mic after a reconnect (the server side of a listener dies with its session).
    document.addEventListener("ws-open", () => { if (listen.on || listen.wanted) listenStart().catch(() => {}); });
  }

  function bindPtt() {
    const btn = $("#ptt");
    btn.addEventListener("pointerdown", pttStart);
    btn.addEventListener("pointerup", pttStop);
    btn.addEventListener("pointercancel", pttStop);
    btn.addEventListener("pointerleave", (e) => { if (pttDown) pttStop(e); });
    btn.addEventListener("contextmenu", (e) => e.preventDefault());
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !e.repeat && document.activeElement.tagName !== "INPUT") { pttStart(e); }
    });
    document.addEventListener("keyup", (e) => {
      if (e.code === "Space" && document.activeElement.tagName !== "INPUT") { pttStop(e); }
    });
  }

  // ------------------------------------------------------------ pantry
  async function loadPantry() {
    const tbody = $("#pantry-table tbody");
    tbody.innerHTML = "";
    try {
      const items = await api("GET", "/inventory");
      for (const it of items) {
        const tr = el("tr", it.amount <= 0 ? "low" : "");
        tr.appendChild(el("td", null, it.name));
        const td = el("td", "amount");
        const minus = el("button", "small", "−");
        const input = el("input");
        input.type = "number"; input.step = "any"; input.min = "0"; input.value = it.amount;
        const plus = el("button", "small", "+");
        const stepBy = it.unit === "g" || it.unit === "ml" ? 50 : 1;
        const setAmount = async (v) => {
          v = Math.max(0, Number(v));
          input.value = v;
          try { await api("PATCH", "/inventory", { items: [{ name: it.name, amount: v, unit: it.unit }] }); tr.className = v <= 0 ? "low" : ""; }
          catch (e) { toast(e.message, true); }
        };
        minus.onclick = () => setAmount(Number(input.value) - stepBy);
        plus.onclick = () => setAmount(Number(input.value) + stepBy);
        input.onchange = () => setAmount(input.value);
        td.append(minus, input, plus);
        tr.appendChild(td);
        tr.appendChild(el("td", null, it.unit || "count"));
        const rm = el("td");
        const b = el("button", "small danger", "Remove");
        b.onclick = async () => {
          try { await api("PATCH", "/inventory", { remove: [it.name] }); tr.remove(); } catch (e) { toast(e.message, true); }
        };
        rm.appendChild(b);
        tr.appendChild(rm);
        tbody.appendChild(tr);
      }
      if (!items.length) {
        const tr = el("tr");
        tr.appendChild(el("td", "empty", "Pantry is empty"));
        tbody.appendChild(tr);
      }
    } catch (e) {
      toast(e.message, true);
    }
  }
  function bindPantry() {
    $("#pantry-add").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = e.target;
      const name = f.name.value.trim(), amount = Number(f.amount.value), unit = f.unit.value.trim() || null;
      if (!name) return;
      try {
        await api("PATCH", "/inventory", { items: [{ name, amount, unit }] });
        f.reset();
        loadPantry();
      } catch (err) { toast(err.message, true); }
    });
  }

  // ------------------------------------------------------------ screens & misc
  let screen = "cook";
  let autoScreened = false;
  function showScreen(name) {
    screen = name;
    autoScreened = true;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.screen === name));
    document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("active", s.id === "screen-" + name));
    if (name === "pantry") { loadPantry(); loadKitchen(); }
    if (name === "recipes") renderChoices();
  }
  function bindUi() {
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showScreen(t.dataset.screen)));
    $("#btn-add-recipe").addEventListener("click", () => showScreen("recipes"));
    bindRecipes();
    $("#text-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const input = $("#text-input");
      const text = input.value.trim();
      if (!text) return;
      if (app.speaking || player.busy) { send({ type: "barge_in" }); player.stop(); app.speaking = false; app.bubble = null; }
      player.ensure();
      send({ type: "text", text });
      input.value = "";
      setPhase("thinking");
    });
    $("#proactivity").addEventListener("change", (e) => send({ type: "set_proactivity", value: Number(e.target.value) }));
    bindPtt();
    bindListen();
    bindPantry();
    // keep the screen awake in kiosk use where the API exists
    if ("wakeLock" in navigator) {
      const lock = () => navigator.wakeLock.request("screen").catch(() => {});
      lock();
      document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") lock(); });
    }
  }
  async function loadHealth() {
    try {
      const h = await api("GET", "/health");
      app.health = h;
      const btn = $("#ptt");
      const mic = micProblem();
      btn.disabled = !h.stt || !!mic;
      if (!h.stt) btn.textContent = "Voice input off (server has no STT)";
      else if (mic) btn.textContent = "Microphone unavailable (tap for why)";
      $("#listen-toggle").disabled = !h.stt || h.vad === "none" || !!mic;
      showMicBanner(mic, h);
    } catch (e) { /* server not up yet; reconnect loop will retry */ }
  }

  let micBannerShown = false;
  function showMicBanner(problem, health) {
    const bar = $("#mic-banner");
    if (!problem || !health || !health.stt) { bar.hidden = true; return; }
    if (!micBannerShown) {
      bar.innerHTML = "";
      bar.appendChild(el("span", null, problem));
      if (location.protocol === "http:") {
        const a = el("a", null, " Open the HTTPS page ");
        a.href = `https://${location.hostname}:${location.port || 8000}/app/`;
        bar.appendChild(a);
        const c = el("a", null, " · install certificate");
        c.href = "/cert";
        bar.appendChild(c);
      }
      micBannerShown = true;
    }
    bar.hidden = false;
  }
  function tick() {
    $("#now").textContent = fmtClock();
    if (app.state && app.state.timers.length) renderTimers();
  }



  // ------------------------------------------------------------ appliance board
  // A picture of the kitchen rather than a list: each appliance is drawn, and lights up when
  // it is actually running. One stroke colour plus a few named parts the CSS lights by state,
  // so the state lives in the stylesheet and this stays declarative.
  const APPLIANCE_ICONS = {
    // a burner ring seen from above, with the flame crown that lights when it is on
    stovetop: `
      <circle class="ring" cx="32" cy="32" r="20"/>
      <circle class="ring" cx="32" cy="32" r="11"/>
      <g class="flame">
        <circle cx="32" cy="32" r="15.5"/>
        <path d="M32 12v5M32 47v5M12 32h5M47 32h5M18 18l3.5 3.5M42.5 42.5L46 46M46 18l-3.5 3.5M21.5 42.5L18 46"/>
      </g>`,
    oven: `
      <rect class="body" x="9" y="11" width="46" height="42" rx="5"/>
      <path d="M9 22h46"/>
      <circle cx="16" cy="16.5" r="1.8"/><circle cx="23" cy="16.5" r="1.8"/>
      <rect class="glow" x="15" y="28" width="34" height="18" rx="3"/>`,
    air_fryer: `
      <path class="body" d="M15 14h34a4 4 0 0 1 4 4l-3 32a5 5 0 0 1-5 4.5H19a5 5 0 0 1-5-4.5l-3-32a4 4 0 0 1 4-4z"/>
      <path d="M13 30h38"/>
      <rect class="glow" x="22" y="18" width="20" height="7" rx="3"/>
      <path d="M26 44h12"/>`,
    rice_cooker: `
      <path class="body" d="M11 30h42v13a8 8 0 0 1-8 8H19a8 8 0 0 1-8-8z"/>
      <path d="M9 30c0-8 10-13 23-13s23 5 23 13"/>
      <circle cx="32" cy="21" r="2.2"/>
      <rect class="glow" x="25" y="38" width="14" height="5" rx="2.5"/>
      <g class="steam">
        <path d="M25 13c2-3-2-5 0-8"/><path d="M32 10c2-3-2-5 0-8"/><path d="M39 13c2-3-2-5 0-8"/>
      </g>`,
    microwave: `
      <rect class="body" x="6" y="15" width="52" height="34" rx="4"/>
      <rect class="glow" x="11" y="20" width="30" height="24" rx="3"/>
      <path d="M46 21v8M46 33h6M46 38h6"/>
      <path d="M41 20v24"/>`,
    pressure_cooker: `
      <path class="body" d="M12 28h40v14a9 9 0 0 1-9 9H21a9 9 0 0 1-9-9z"/>
      <path d="M10 28h44"/>
      <path d="M32 28v-6"/><circle class="glow" cx="32" cy="19" r="3.5"/>
      <path d="M12 34H7M52 34h5"/>`,
    bread_maker: `
      <rect class="body" x="10" y="16" width="44" height="36" rx="5"/>
      <rect class="glow" x="18" y="23" width="20" height="13" rx="3"/>
      <path d="M44 24v6M44 34h5"/>
      <path d="M10 44h44"/>`,
    grill: `
      <path class="body" d="M8 34a24 24 0 0 1 48 0z"/>
      <path class="glow" d="M12 40h40"/>
      <path d="M8 34h48"/><path d="M18 40v10M46 40v10"/>`,
    default: `
      <rect class="body" x="10" y="16" width="44" height="34" rx="5"/>
      <rect class="glow" x="18" y="24" width="28" height="14" rx="3"/>`,
  };

  function applianceIcon(family) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 64 64");
    svg.setAttribute("class", "appl-icon");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = APPLIANCE_ICONS[family] || APPLIANCE_ICONS.default;
    return svg;
  }

  // Coarser than fmtLeft, which is the countdown format for timers: a hob does not need
  // seconds. Named apart so it cannot shadow that one.
  function fmtCoarse(s) {
    if (s == null) return "";
    if (s < 60) return Math.max(0, Math.round(s)) + "s";
    const m = Math.round(s / 60);
    return m < 60 ? m + "m" : Math.floor(m / 60) + "h " + (m % 60) + "m";
  }

  // Only what is actually in use gets a tile. A fully drawn kitchen is eight to eleven of
  // them, nearly all idle, and on the tablet that pushed the timers off the bottom of the
  // right-hand column - so the board was crowding out the one thing that is genuinely
  // time-critical. The server still reports every station, because the model reasons about
  // free rings; this is the cook's view, not the model's.
  const IN_USE = ["active", "due", "reserved", "needed"];

  function renderAppliances(rows) {
    const box = $("#appliances");
    // Not named "busy": each card below already has its own busy, and a shadowed name in
    // this file once made timers render as "12m" instead of "12:30" for a whole cook.
    const inUse = (rows || []).filter((a) => IN_USE.includes(a.status));
    if (!inUse.length) { box.hidden = true; box.innerHTML = ""; return; }
    box.hidden = false;
    box.innerHTML = "";
    for (const a of inUse) {
      const card = el("div", "appl " + a.status + (a.untimed ? " untimed" : "")
                              + (a.owned === false ? " unowned" : ""));
      card.appendChild(applianceIcon(a.family));

      const name = el("div", "appl-name", a.label);
      if (a.temp_f) name.appendChild(el("span", "appl-temp", a.temp_f + "°"));
      card.appendChild(name);

      const busy = a.current || a.next;
      const what = busy ? busy.label
                        : (a.status === "needed" ? (a.needed_by || []).join(", ") || "needed" : "free");
      card.appendChild(el("div", "appl-what", what));

      let when = "";
      if (a.owned === false) when = "you don't have this";
      else if (!busy && a.status === "needed") when = "not scheduled yet";
      else if (a.current && a.current.awaits_cook) when = "tell me when it's done";
      else if (a.current && a.current.remaining_s != null) when = fmtCoarse(a.current.remaining_s) + " left";
      else if (a.next && a.next.starts_in_s != null) {
        when = a.next.starts_in_s <= 0 ? "start now" : "in " + fmtCoarse(a.next.starts_in_s);
      }
      if (when) card.appendChild(el("div", "appl-when", when));
      box.appendChild(card);
    }

    // What is free, as one line rather than a row of empty tiles, so the cook can still
    // answer "is there a ring going spare?" without reading a grid.
    const idle = (rows || []).filter((a) => !IN_USE.includes(a.status) && a.owned !== false);
    if (idle.length) {
      const rings = idle.filter((a) => a.family === "stovetop").length;
      const others = idle.filter((a) => a.family !== "stovetop").map((a) => a.label.toLowerCase());
      const bits = [];
      if (rings) bits.push(rings === 1 ? "1 ring" : rings + " rings");
      bits.push(...others);
      // Capped: a well equipped kitchen idles six or seven things and the summary would
      // wrap to the height of the tiles it replaced.
      const shown = bits.slice(0, 4);
      if (bits.length > shown.length) shown.push("+" + (bits.length - shown.length) + " more");
      box.appendChild(el("div", "appl-idle", "Free: " + shown.join(", ")));
    }
  }


  // Rejections the assistant has not dealt with. Previously these scrolled past in the
  // conversation and were gone; the cook could see them and the model could not.
  function renderUnresolved(rows) {
    const box = $("#unresolved");
    if (!rows || !rows.length) { box.hidden = true; box.innerHTML = ""; return; }
    box.innerHTML = "";
    box.appendChild(el("div", "unresolved-head",
                       rows.length === 1 ? "1 unresolved problem" : rows.length + " unresolved problems"));
    for (const r of rows) {
      const item = el("div", "unresolved-item");
      item.appendChild(el("b", null, r.tool));
      item.appendChild(document.createTextNode(" " + r.reason));
      box.appendChild(item);
    }
    box.hidden = false;
  }

  // ------------------------------------------------------------ missing ingredients
  // Shown as a banner as well as spoken: if the model fumbles the sentence, the cook still
  // finds out before the first pan goes on.
  function renderPreflight(report) {
    const bar = $("#preflight-banner");
    const missing = (report.missing || []).filter((m) => !m.staple);
    if (!missing.length) { bar.hidden = true; bar.innerHTML = ""; return; }
    bar.innerHTML = "";
    bar.appendChild(el("b", null, missing.length === 1 ? "Missing an ingredient"
                                                       : `Missing ${missing.length} ingredients`));
    // The same tiles as the step panel, so an ingredient looks the same wherever it appears.
    const tiles = el("div", "gap-tiles");
    for (const m of missing) {
      const tile = el("div", "gap");
      tile.appendChild(el("span", "gap-icon", foodIcon(m)));
      const body = el("span", "gap-body");
      body.appendChild(el("span", "gap-name", m.text || m.name));
      if (m.swaps && m.swaps.length) body.appendChild(el("span", "gap-swap", "try " + m.swaps[0]));
      tile.appendChild(body);
      tiles.appendChild(tile);
    }
    bar.appendChild(tiles);
    const dismiss = el("button", "small ghost", "Dismiss");
    dismiss.onclick = () => { bar.hidden = true; };
    bar.appendChild(dismiss);
    bar.hidden = false;
  }

  // ------------------------------------------------------------ my kitchen
  // Which appliances the cook actually owns. This is a constraint, not decoration: the board
  // only draws these, and add_task refuses anything else.
  let kitchen = null;

  async function loadKitchen() {
    try { kitchen = await api("GET", "/kitchen"); } catch (e) { return; }
    $("#burner-count").value = kitchen.burners;
    $("#burner-cap").value = kitchen.burner_cap;
    renderBurnerSizes();
    const box = $("#appliance-picker");
    box.innerHTML = "";
    for (const opt of kitchen.options) {
      const card = el("div", "appl-pick " + (opt.owned ? "on" : "off"));
      card.appendChild(applianceIcon(opt.id));
      card.appendChild(el("div", "appl-name", opt.label));
      card.appendChild(el("div", "appl-when", opt.owned ? "have it" : "don't have it"));
      card.onclick = () => saveKitchen(opt.id);
      box.appendChild(card);
    }
  }

  // How big each ring is, and how many pans genuinely fit. Four rings is not four pans.
  function renderBurnerSizes() {
    const box = $("#burner-sizes");
    box.innerHTML = "";
    if (!kitchen || !kitchen.appliances.includes("stovetop")) return;
    for (let i = 0; i < kitchen.burners; i++) {
      const row = el("label", "burner-size");
      row.appendChild(el("span", null, "Burner " + (i + 1)));
      const sel = el("select");
      for (const size of kitchen.pan_sizes) {
        const o = el("option", null, size);
        o.value = size;
        if ((kitchen.burner_sizes[i] || "large") === size) o.selected = true;
        sel.appendChild(o);
      }
      sel.onchange = () => {
        for (let j = 0; j < kitchen.burners; j++) {
          if (!kitchen.burner_sizes[j]) kitchen.burner_sizes[j] = "large";
        }
        kitchen.burner_sizes[i] = sel.value;
        saveKitchen(null);
      };
      row.appendChild(sel);
      box.appendChild(row);
    }
  }

  async function saveKitchen(toggleId) {
    if (!kitchen) return;
    let chosen = kitchen.options.filter((o) => o.owned).map((o) => o.id);
    if (toggleId) {
      chosen = chosen.includes(toggleId) ? chosen.filter((x) => x !== toggleId)
                                         : chosen.concat([toggleId]);
    }
    const burners = Math.max(1, Math.min(8, parseInt($("#burner-count").value, 10) || 4));
    try {
      const cap = Math.max(1, Math.min(8, parseInt($("#burner-cap").value, 10) || burners));
      kitchen = await api("PUT", "/kitchen", {
        appliances: chosen, burners, burner_cap: cap,
        burner_sizes: (kitchen.burner_sizes || []).slice(0, burners),
      });
      await loadKitchen();
      toast("Kitchen updated");
    } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ server panel
  // The server cannot restart itself, so this asks the supervisor (start.ps1) for a fresh
  // process. The websocket below reconnects on its own once it comes back.
  let serverStatus = null;

  function fmtUptime(s) {
    if (s == null) return "-";
    if (s < 90) return Math.round(s) + "s";
    if (s < 5400) return Math.round(s / 60) + "m";
    const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
    return h + "h " + m + "m";
  }

  async function loadServerStatus() {
    try {
      serverStatus = await api("GET", "/admin/status");
    } catch (e) { return; }  // server down; the reconnect loop is already on it
    $("#server-badge").hidden = !serverStatus.code_changed;
    if (!$("#server-panel").hidden) renderServerPanel();
  }

  function renderServerPanel() {
    const s = serverStatus;
    if (!s) return;
    $("#srv-uptime").textContent = fmtUptime(s.uptime_s);
    $("#srv-model").textContent = s.model || "-";
    $("#srv-backend").textContent = s.backend || "-";

    const changed = $("#srv-changed");
    changed.hidden = !s.code_changed;
    if (s.code_changed) {
      changed.innerHTML = "";
      const n = s.changed_count;
      changed.appendChild(el("div", null, `Code changed since this server started (${n} file${n === 1 ? "" : "s"}). Restart to load it.`));
      const ul = el("ul");
      for (const f of s.changed_files.slice(0, 6)) ul.appendChild(el("li", null, f));
      if (n > 6) ul.appendChild(el("li", null, `and ${n - 6} more`));
      changed.appendChild(ul);
    }

    const warn = $("#srv-warn");
    const active = (s.active_sessions || []).length;
    if (!s.can_restart) {
      warn.hidden = false;
      warn.textContent = "This server was started without the supervisor, so it cannot restart itself. Use start.ps1.";
    } else if (active) {
      warn.hidden = false;
      warn.textContent = `${active} cooking session${active === 1 ? "" : "s"} in progress. Restarting loses their timers and progress.`;
    } else {
      warn.hidden = true;
    }
    $("#srv-restart").disabled = !s.can_restart;
  }

  function toggleServerPanel(show) {
    const panel = $("#server-panel");
    const open = show === undefined ? panel.hidden : show;
    panel.hidden = !open;
    if (open) { loadServerStatus(); renderServerPanel(); }
  }

  async function restartServer() {
    const active = ((serverStatus && serverStatus.active_sessions) || []).length;
    if (active && !confirm(`${active} cooking session${active === 1 ? "" : "s"} in progress.\nRestarting loses their timers, tasks and progress. Restart anyway?`)) return;
    const btn = $("#srv-restart");
    btn.disabled = true;
    btn.textContent = "Restarting...";
    try {
      await api("POST", "/admin/restart", { force: active > 0 });
      toggleServerPanel(false);
      toast("Restarting the server...");
      app.restarting = true;
      $("#conn-text").textContent = "restarting";
    } catch (e) {
      toast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Restart server";
    }
  }

  // ------------------------------------------------------------ boot
  bindUi();
  $("#burner-count").onchange = () => saveKitchen(null);
  $("#burner-cap").onchange = () => saveKitchen(null);
  $("#server-chip").onclick = () => toggleServerPanel();
  $("#srv-close").onclick = () => toggleServerPanel(false);
  $("#srv-restart").onclick = restartServer;
  loadHealth();
  loadServerStatus();
  connect();
  tick();
  setInterval(tick, 1000);
  setInterval(loadHealth, 30000);
  setInterval(loadServerStatus, 15000);
})();
