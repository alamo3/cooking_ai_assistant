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
  function addChip(msg) {
    const args = msg.args && Object.keys(msg.args).length
      ? " " + Object.entries(msg.args).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(", ")
      : "";
    const text = (msg.ok ? "✓ " : "✗ ") + msg.name + args + (msg.ok ? "" : " — " + (msg.message || "rejected"));
    const chip = el("div", "chip" + (msg.ok ? "" : " rejected") + (msg.source === "ui" ? " ui" : ""), text);
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
      title.appendChild(el("span", "progress", `${done}/${r.steps.length} · serves ${r.servings}`));
      card.appendChild(title);
      const bar = el("div", "bar");
      const fill = el("i");
      fill.style.width = `${r.steps.length ? (100 * done) / r.steps.length : 0}%`;
      bar.appendChild(fill);
      card.appendChild(bar);

      const cur = r.steps.find((st) => st.n === r.current_step);
      if (cur) {
        const now = el("div", "step-now");
        now.appendChild(el("span", "n", `Step ${cur.n}`));
        now.appendChild(document.createTextNode(cur.text));
        card.appendChild(now);
        const meta = [cur.duration_s ? fmtDur(cur.duration_s) : null,
          cur.appliance ? cur.appliance.replace("_", " ") + (cur.temp_f ? ` ${cur.temp_f}°F` : "") : null,
          cur.note ? "note: " + cur.note : null].filter(Boolean).join(" · ");
        if (meta) card.appendChild(el("div", "step-meta", meta));
        const next = r.steps.find((st) => st.n > cur.n && st.status === "pending");
        if (next) card.appendChild(el("div", "step-next", `Then: ${next.text}`));
      } else {
        card.appendChild(el("div", "step-now", "All steps done"));
      }
      if (r.changes) card.appendChild(el("div", "changes", r.changes));

      const actions = el("div", "actions");
      if (cur) {
        const b = el("button", "small", `Done with step ${cur.n}`);
        b.onclick = () => uiTool("POST", `/session/${app.sessionId}/steps/${cur.id}/complete`);
        actions.appendChild(b);
      }
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

  function renderTimeline(tl) {
    const box = $("#timeline");
    box.innerHTML = "";
    if (!tl.tasks.length) {
      box.appendChild(el("div", "empty", "No plan yet. Pick recipes on the Recipes tab."));
    } else {
      const table = el("table");
      const head = el("tr");
      for (const h of ["Task", "When", "Where", "Status"]) head.appendChild(el("th", null, h));
      table.appendChild(head);
      const order = { active: 0, pending: 1, complete: 2, skipped: 3 };
      const tasks = [...tl.tasks].sort((a, b) => (order[a.status] - order[b.status]) || ((a.start_at || "") < (b.start_at || "") ? -1 : 1));
      for (const t of tasks) {
        const tr = el("tr", t.status);
        tr.appendChild(el("td", null, t.label));
        tr.appendChild(el("td", "win", fmtWindow(t.start_at, t.end_at)));
        tr.appendChild(el("td", null, t.appliance ? t.appliance.replace("_", " ") + (t.temp_f ? ` ${t.temp_f}°` : "") : ""));
        tr.appendChild(el("td", null, t.status));
        table.appendChild(tr);
      }
      box.appendChild(table);
    }
    $("#conflicts").textContent = tl.conflicts.length ? "Conflicts: " + tl.conflicts.join("; ") : "";
  }

  async function uiTool(method, path, body) {
    try { await api(method, path, body); } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ recipe selection screen
  const choice = { selected: new Set(), library: [] };

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
    btn.textContent = on ? "Always listening: on" : "Always listening: off";
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
    if (name === "pantry") loadPantry();
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
