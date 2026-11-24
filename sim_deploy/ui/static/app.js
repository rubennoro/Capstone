// sim_deploy ui: the page (ui/DESIGN.md section 4). Plain JS, no build, no external files (offline lab laptop).
//
// Control is REST only (POST /api/run, POST /api/stop); everything live arrives on one WebSocket /ws: JSON `status`
// (5 Hz), `telemetry` (10 Hz while the sim runs), `event` messages, and binary frames (1 byte channel id + JPEG).
// Frames are decoded newest-first: while one decodes, later ones overwrite the single pending slot, so a slow
// laptop draws fewer frames instead of falling behind.
"use strict";

const RECONNECT_MS = 1000;
const STALE_MS = 3000;            // a camera channel or telemetry older than this shows "no data"
const TICK_MS = 500;              // fps captions and staleness
const BUSY = new Set(["starting", "running", "stopping"]);
const CHANNEL_IDS = [1, 2, 3, 4]; // 1 external, 2 main (IMX519), 3 ultrawide (IMX219), 4 policy 2x2
const STORE_KEY = "sim_deploy_ui.v1";
const HAS_BITMAP = typeof createImageBitmap === "function";

// ------------------------------------------------------------------------------------------------------- state
const state = {
  ws: null,
  connected: false,
  policies: [],
  replays: [],
  builtins: [],
  builtinLastEval: null,      // /api/policies "builtin_last_eval" (hold_still's last sim result)
  openSmoke: new Set(),       // tasks whose "smoke / old" fold is open
  showAllReplays: false,      // also list replays whose placement misses the grasp
  policiesLoaded: false,
  selected: null,             // a Policy.id, "hold_still" or a Replay.id
  status: null,               // the latest Status
  telemetry: null,
  telemetryAt: 0,
  evals: [],
  inFlight: false,            // a POST /api/run is on its way
  expectStart: 0,             // performance.now() of an accepted run until the status shows it busy (or 4 s)
  stopPresses: 0,             // Stop presses in this run: the second one is the force
  runId: null,
  rolloutRecords: new Map(),  // rollout index (0-based) -> the outcome dict of its rollout_end event
  runEnd: null,               // the run_end event of this run
  evalsBefore: null,          // eval_dir of every index.csv row seen before this run started (null: unknown)
  shown: { log: "", chips: "", picker: "", results: "", runResult: "", steps: "", header: "", selected: "" },
  frames: null,               // {rel, busy, urls, error}: the frames button of this run's result row
  // U6
  builtinItems: [],           // /api/policies "builtin_items"
  startSetKnown: false,       // the catalog has a "start_set" key (a server that knows RunRequest.starts)
  startSet: null,             // /api/policies "start_set" ({name, file, n_starts, needs_home}) or null (no file)
  startsPref: "random",       // the viewer's choice in "Eraser starts" (shown as random while it is not allowed)
  stepMemo: null,             // per run: what the step panel remembers (newMemo())
  stepNodes: [],              // the detail <span> of every rendered step (updated in place)
  bringupOpen: false,         // the folded "Sim and policy ready" line is open
  logOpenedFor: null,         // run_id whose failure opened the raw log once
  // visual restyle
  catalogVersion: 0,          // +1 per /api/policies load (the play card's header key)
  catalogError: null,         // the last /api/policies error (Play stays off until the catalog loaded)
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------------- words (help.js)
const H = typeof HELP !== "undefined" ? HELP : { FIELDS: {}, STEPS: {}, OUTCOMES: {}, KINDS: {}, TYPES: {}, PAGE: {}, GLOSSARY: [] };
const F = H.FIELDS || {}, P = H.PAGE || {};
const SHORT = P.short || {};
const fillText = typeof fill === "function" ? fill : (t) => String(t || "");

function setName() { return (state.startSet && state.startSet.name) || "w3"; }
function setVars(extra) {
  return Object.assign({ SET: setName().toUpperCase(), set: setName(),
                         n: state.startSet && state.startSet.n_starts !== undefined ? state.startSet.n_starts : "?" },
                       extra || {});
}

// The words on the page from HELP: [data-label] text, [data-help] help lines, [data-tip] tooltips, [data-opt]
// select options, the glossary and the results table's header tooltips.
function applyHelp() {
  for (const n of document.querySelectorAll("[data-label]")) {
    const k = n.dataset.label;
    const t = (F[k] && F[k].label) || (typeof P[k] === "string" ? P[k] : null);
    if (t) n.textContent = t;
  }
  for (const n of document.querySelectorAll("[data-help]")) {
    const f = F[n.dataset.help];
    if (f && f.help !== undefined) { n.textContent = f.help; n.dataset.default = f.help; }
  }
  for (const n of document.querySelectorAll("[data-tip]")) {
    const f = F[n.dataset.tip];
    if (f && f.tip) n.title = f.tip;
  }
  for (const n of document.querySelectorAll("[data-opt]")) {
    const [k, v] = n.dataset.opt.split(".");
    const t = F[k] && F[k].options && F[k].options[v];
    if (t) n.textContent = fillText(t, setVars());
  }
  for (const n of document.querySelectorAll("[data-th]")) {
    const t = P.th && P.th[n.dataset.th];
    if (t) n.title = t;
  }
  const ps = F.policy_settings || {};
  if (ps.not_arm) $("not-arm").textContent = ps.not_arm;
  if (ps.not_arm_tip) $("not-arm").title = ps.not_arm_tip;
  if (ps.reset) $("btn-arm-reset").textContent = ps.reset;
  if (P.picker_tip) $("picker-title").title = P.picker_tip;
  for (const n of document.querySelectorAll("[data-cam]")) {
    const t = P.cams && P.cams[n.dataset.cam];
    if (t) n.textContent = t;
  }
  $("glossary-list").replaceChildren(...(H.GLOSSARY || []).flatMap(([term, text]) =>
    [el("dt", { text: term }), el("dd", { text })]));
}

// "behavior prompting" + raw "prompting"; other = the train_config verbatim; anything unknown = "type unknown".
function kindWords(kind, trainConfig) {
  const K = H.KINDS || {};
  if (kind === "other") return { word: trainConfig || "other", raw: "other", short: "other" };
  const k = K[kind];
  if (k) return { word: k.word || kind, raw: kind, short: k.short || kind };
  return { word: (K.unknown && K.unknown.word) || "type unknown", raw: "unknown", short: "unknown" };
}

function typeOfKind(kind) {
  return kind === "hold_still" ? "builtin" : kind === "replay" ? "replay" : "policy";
}

// "62 demos (PICK_TEST1 30 + PICK_TEST2 32)" (the breakdown only with more than one recording)
function demosText(p) {
  const recs = Array.isArray(p.recordings) ? p.recordings : null;
  let n = p.episodes;
  if ((n === null || n === undefined) && recs) n = recs.reduce((a, r) => a + (Number(r.episodes) || 0), 0);
  if (n === null || n === undefined) return null;
  const base = fillText(P.demos || "{n} demos", { n });
  return recs && recs.length > 1 ? `${base} (${recs.map((r) => `${r.label} ${r.episodes ?? "?"}`).join(" + ")})` : base;
}

// {type: policy|replay|builtin, id, kind, item, index} for a picker id / RunRequest.policy, from the catalog.
function catalogItem(id) {
  if (!id) return null;
  if (id === "hold_still" || state.builtins.includes(id)) {
    const b = state.builtinItems.find((x) => x && x.id === id) || null;
    return { type: "builtin", id, kind: (b && b.kind) || id, item: b };
  }
  const p = state.policies.find((x) => x.id === id);
  if (p) return { type: p.type || "policy", id, kind: p.kind || "unknown", item: p };
  const r = state.replays.find((x) => x.id === id);
  if (r || String(id).startsWith("replay:")) {
    const m = String(id).match(/#(\d+)$/);
    return { type: "replay", id, kind: "replay", item: r || null,
             index: r && typeof r.index === "number" ? r.index : (m ? Number(m[1]) : null) };
  }
  return { type: "policy", id, kind: "unknown", item: null };
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}

// ------------------------------------------------------------------------------------------------- small format
const fmt = {
  mm: (m) => (m === null || m === undefined ? null : `${(m * 1000).toFixed(1)} mm`),
  ms: (v) => (v === null || v === undefined ? null : `${Number(v).toFixed(1)} ms`),
  mse: (v) => (v === null || v === undefined || v === "" ? null : Number(v).toExponential(1)),
  rate: (v) => (v === null || v === undefined || v === "" ? "" : Number(v).toFixed(2)),
  elapsed(s) {
    if (s === null || s === undefined) return null;
    if (s < 60) return `${Number(s).toFixed(1)} s`;
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, "0")}`;
  },
};

// ------------------------------------------------------------------------------------------------------ REST
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error(`${method} ${path}: server not reachable`);
  }
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (data === null) throw new Error(`${method} ${path}: HTTP ${res.status}`);
  if (!res.ok && data.ok === undefined) data = { ok: false, error: data.error || `HTTP ${res.status}` };
  if (!res.ok && data.ok !== false) data.ok = false;
  if (data.ok === false && !data.error) data.error = `HTTP ${res.status}`;
  return data;
}

async function loadPolicies() {
  const err = $("picker-error");
  try {
    const data = await api("GET", "/api/policies");
    if (data.ok === false) throw new Error(data.error);
    state.policies = Array.isArray(data.policies) ? data.policies : [];
    state.replays = Array.isArray(data.replays) ? data.replays : [];
    state.builtins = Array.isArray(data.builtins) ? data.builtins : [];
    state.builtinLastEval = data.builtin_last_eval || null;
    state.builtinItems = Array.isArray(data.builtin_items) ? data.builtin_items : [];
    state.startSetKnown = "start_set" in data;
    state.startSet = data.start_set && typeof data.start_set === "object" ? data.start_set : null;
    state.policiesLoaded = true;
    state.catalogError = null;
    applyHelp();                                   // the start-set option text names the set
    err.classList.add("hidden");
  } catch (e) {
    state.catalogError = e.message;
    err.replaceChildren(`policies: ${e.message}`,
      el("button", { type: "button", class: "quiet", onclick: loadPolicies, text: "retry" }));
    err.classList.remove("hidden");
  }
  state.catalogVersion += 1;                       // the play card's header follows a reloaded catalog
  state.shown.picker = "";
  state.shown.header = "";
  renderPicker();
  applySettingsRules();
  renderSelected();
  renderButtons();
  renderSteps();
  renderRunHeader();
}

async function loadEvals() {
  const err = $("results-error");
  try {
    const data = await api("GET", "/api/evals");
    if (data.ok === false) throw new Error(data.error);
    state.evals = Array.isArray(data.evals) ? data.evals : [];
    err.classList.add("hidden");
  } catch (e) {
    err.replaceChildren(`results: ${e.message}`,
      el("button", { type: "button", class: "quiet", onclick: loadEvals, text: "retry" }));
    err.classList.remove("hidden");
  }
  state.shown.results = "";
  renderResults();
  renderRunResult();
}

async function loadStatus() {
  try {
    const s = await api("GET", "/api/status");
    if (s && s.state) onStatus(s);
  } catch (e) { /* the socket brings it too */ }
}

// ------------------------------------------------------------------------------------------------- the socket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    state.connected = true;
    ws.send(JSON.stringify({ type: "subscribe", channels: CHANNEL_IDS }));
    renderConnection();
    renderButtons();
    loadStatus();
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") onText(ev.data);
    else onBinary(ev.data);
  };
  ws.onclose = () => {
    const was = state.connected;
    state.connected = false;
    state.ws = null;
    renderConnection();
    renderButtons();
    setTimeout(connect, was ? 200 : RECONNECT_MS);
  };
  ws.onerror = () => { /* onclose follows */ };
}

function onText(text) {
  let msg;
  try { msg = JSON.parse(text); } catch (e) { return; }
  // The payload may sit next to "type" or under a key of the same name; accept both.
  switch (msg.type) {
    case "status": onStatus(msg.status && typeof msg.status === "object" ? msg.status : msg); break;
    case "telemetry": onTelemetry(msg.telemetry && typeof msg.telemetry === "object" ? msg.telemetry : msg); break;
    case "event": onEvent(msg.event && typeof msg.event === "object" ? msg.event : msg); break;
    case "error": showRunMsg(`server: ${msg.text || msg.error || "error"}`, "error"); break;
    default: break;
  }
}

// ------------------------------------------------------------------------------------------------------ status
function isBusy() {
  return !!(state.status && BUSY.has(state.status.state));
}

function onStatus(s) {
  const prevState = state.status ? state.status.state : null;
  if (s.run_id && s.run_id !== state.runId) {
    if (state.runId !== null || BUSY.has(s.state)) {
      state.rolloutRecords.clear();
      state.runEnd = null;
      state.frames = null;
      state.stopPresses = 0;
      state.evalsBefore = BUSY.has(s.state) ? new Set(state.evals.map((r) => r.eval_dir)) : null;
    }
    state.runId = s.run_id;
  }
  state.status = s;
  updateMemo();
  if (BUSY.has(s.state)) state.expectStart = 0;
  if (!BUSY.has(s.state)) state.stopPresses = 0;
  if (prevState && BUSY.has(prevState) && (s.state === "done" || s.state === "failed")) {
    showRunMsg(s.state === "done" ? "run finished" : "run failed", s.state === "done" ? "ok" : "error");
    // the run ended: new rows in index.csv, new last_eval on the policies
    setTimeout(() => { loadEvals(); loadPolicies(); }, 300);
  }
  renderStatus();
  renderButtons();
}

function onTelemetry(t) {
  state.telemetry = t;
  state.telemetryAt = performance.now();
  renderTelemetry();
  if (isBusy()) { updateMemo(); renderSteps(); }
}

// ------------------------------------------------------------------------------------- step panel memory
// Per run: the last step seen (the finished view and the "stopping" phase keep it), the last one before the
// finish (where a failed run failed), every key seen, when bring-up ended, the eraser height when the policy took
// control (for "eraser up x cm") and the place step's detail (shown on the done step).
function newMemo(runId) {
  return { runId, lastPos: null, lastWork: null, seen: new Set(), bringupS: null, eraserBase: null,
           eraserKey: null, placeDetail: null, setName: setName(),
           times: new Map(), order: [] };            // step id -> since_s, ids in the order they began
}

function stepId(key, rollout) { return `${key}:${rollout ?? ""}`; }

function notePos(pos, elapsed) {
  const m = state.stepMemo;
  if (!m || !pos) return;
  m.lastPos = pos;
  m.seen.add(pos.key);
  if (pos.stage !== "finish") m.lastWork = pos;
  if (pos.stage === "rollout" && m.bringupS === null) m.bringupS = pos.since_s ?? elapsed ?? null;
  if (pos.key === "place" && pos.detail) m.placeDetail = { rollout: pos.rollout, text: pos.detail };
  if (typeof pos.since_s === "number") {           // the timeline's per-step times
    const id = stepId(pos.key, pos.rollout);
    if (m.times.get(id) !== pos.since_s) {         // new, or entered again (a sim restart): it began now
      const i = m.order.indexOf(id);
      if (i >= 0) m.order.splice(i, 1);
      m.order.push(id);
      m.times.set(id, pos.since_s);
    }
  }
}

function updateMemo() {
  const s = state.status;
  if (!s) return;
  const rid = s.run_id || null;
  if (!state.stepMemo || state.stepMemo.runId !== rid) state.stepMemo = newMemo(rid);
  const m = state.stepMemo;
  m.setName = setName();
  if (!BUSY.has(s.state)) return;                 // finished: keep what the live run showed
  notePos(typeof stepPosition === "function" ? stepPosition(s) : null, s.elapsed_s);
  const pos = m.lastPos;
  if (pos && pos.key === "control") {
    const key = `${pos.rollout}`;
    const h = sceneTelemetry() && state.telemetry.eraser ? state.telemetry.eraser.height_m : null;
    if (m.eraserKey !== key && typeof h === "number") { m.eraserKey = key; m.eraserBase = h; }
  } else {
    m.eraserKey = null;
    m.eraserBase = null;
  }
}

function sceneTelemetry() {
  return sourceOk("scene") ? state.telemetry : null;
}

function onEvent(ev) {
  const kind = ev.kind;
  if (ev.run_id && state.runId && String(ev.run_id) !== String(state.runId)) return;   // an older run
  if (kind === "step") {                           // {kind: "step", key, rollout, since_s, detail} (runner)
    if (!state.stepMemo && state.status) updateMemo();
    const stage = typeof STEP_STAGE_OF !== "undefined" ? STEP_STAGE_OF[ev.key] : null;
    if (stage) {
      notePos({ stage, key: ev.key, rollout: stage === "rollout" ? (ev.rollout ?? null) : null,
                since_s: ev.since_s ?? null, detail: ev.detail ?? null, fallback: false },
              state.status ? state.status.elapsed_s : null);
      renderSteps();
    }
  } else if (kind === "rollout_end") {
    const rec = rolloutRecord(ev);
    const n = firstNumber(rec.n, ev.n, ev.index, ev.rollout_index);
    if (n !== null) state.rolloutRecords.set(n, rec);
    state.shown.chips = "";
    renderChips();
  } else if (kind === "run_end") {
    state.runEnd = ev;
    loadEvals();
    loadPolicies();
  } else if (kind === "error") {
    showRunMsg(`error: ${ev.error || ev.text || ev.message || JSON.stringify(ev)}`, "error");
  }
}

// rollout_end carries the outcome dict of rollouts.jsonl: directly, or under "outcome"/"record"/"rollout".
function rolloutRecord(ev) {
  for (const k of ["record", "outcome", "rollout"]) {
    if (ev[k] && typeof ev[k] === "object") return ev[k];
  }
  return ev;
}

function firstNumber(...values) {
  for (const v of values) if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

// ----------------------------------------------------------------------------------------------- binary frames
const channels = {};
function channelNodes(frame) {
  return { text: frame.querySelector(".ts-text"), nodata: frame.querySelector(".nodata") };
}
for (const id of CHANNEL_IDS) {
  const canvas = $(`canvas-${id}`);
  channels[id] = {
    id, canvas, ctx: canvas.getContext("2d"), frame: canvas.parentElement, fps: $(`fps-${id}`),
    decoding: false, next: null, lastAt: 0, arrivals: [], ever: false, w: 0, h: 0,
    state: channelNodes(canvas.parentElement), shown: "",
  };
  channels[id].frame.classList.add("never");
}

function onBinary(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < 2) return;
  const ch = channels[new Uint8Array(buf, 0, 1)[0]];
  if (!ch) return;
  ch.next = new Blob([new Uint8Array(buf, 1)], { type: "image/jpeg" });   // newest only
  ch.arrivals.push(performance.now());
  if (!ch.decoding) decodeLoop(ch);
}

async function decodeLoop(ch) {
  ch.decoding = true;
  while (ch.next) {
    const blob = ch.next;
    ch.next = null;
    try {
      if (HAS_BITMAP) {
        const bmp = await createImageBitmap(blob);
        draw(ch, bmp, bmp.width, bmp.height);
        bmp.close();
      } else {
        await drawViaImage(ch, blob);
      }
    } catch (e) { /* a broken JPEG: skip it */ }
  }
  ch.decoding = false;
}

function drawViaImage(ch, blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { draw(ch, img, img.naturalWidth, img.naturalHeight); URL.revokeObjectURL(url); resolve(); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("bad jpeg")); };
    img.src = url;
  });
}

function draw(ch, src, w, h) {
  if (ch.canvas.width !== w || ch.canvas.height !== h) { ch.canvas.width = w; ch.canvas.height = h; }
  ch.ctx.drawImage(src, 0, 0);
  ch.w = w;
  ch.h = h;
  ch.lastAt = performance.now();
  if (!ch.ever) { ch.ever = true; ch.frame.classList.remove("never"); }
  ch.frame.classList.add("live");
  ch.frame.classList.remove("stale");
}

const CHANNEL_SOURCE = { 1: "external", 2: "c0", 3: "c0", 4: "policy_view" };

function tickChannels() {
  const now = performance.now();
  const sources = state.telemetry && performance.now() - state.telemetryAt < STALE_MS ? state.telemetry.sources : null;
  for (const ch of Object.values(channels)) {
    ch.arrivals = ch.arrivals.filter((t) => now - t < 2000);
    const stale = !ch.ever || now - ch.lastAt > STALE_MS;
    ch.frame.classList.toggle("live", !stale);
    ch.frame.classList.toggle("stale", ch.ever && stale);
    ch.frame.classList.toggle("never", !ch.ever);
    const r = sources ? sources[CHANNEL_SOURCE[ch.id]] : null;
    const why = r && !r.ok ? (r.reason || r.error || "") : "";
    // the tile: LIVE / "Stalled 3.2 s" (last frame dimmed) / "Waiting for camera" + the source's reason
    const age = ch.ever ? fmt.elapsed((now - ch.lastAt) / 1000) : null;
    const tsText = !stale ? (P.cam_live || "Live") : ch.ever ? fillText(P.cam_stalled || "Stalled {s}", { s: age }) : "";
    const waiting = (P.cam_waiting || "Waiting for camera") + (why ? `\n${why}` : "");
    const tip = ch.ever && stale ? [fillText(P.cam_stalled_tip || "no new frame for {s}", { s: age }), why].filter(Boolean).join("\n")
      : why;
    const size = ch.w ? `${ch.w}×${ch.h}` : "";
    const meta = !stale ? fillText(P.cam_meta || "{w}×{h} · {fps} fps",
                                   { w: ch.w, h: ch.h, fps: (ch.arrivals.length / 2).toFixed(1) }) : size || "-";
    const key = `${tsText}|${waiting}|${tip}|${meta}`;
    if (key === ch.shown) continue;
    ch.shown = key;
    ch.state.text.textContent = tsText;
    ch.state.nodata.textContent = waiting;
    ch.frame.title = tip;
    ch.fps.textContent = meta;
  }
}

// --------------------------------------------------------------------------------------------- policy picker
function runName(id) {
  const i = id.indexOf("/");
  return i >= 0 ? id.slice(i + 1) : id;
}

function lastEvalText(row) {
  if (!row) return null;
  const k = row.successes, n = row.rollouts;
  if (k === undefined || k === null || k === "" || n === undefined || n === "") return "sim: ?";
  return `sim ${k}/${n}`;
}

// Every selectable entry, in display order. `hide` marks the ones folded away by default: "smoke" (smoke /
// duplicate runs, under a "smoke / old" disclosure per task) and "nograsp" (replays whose placement misses the
// grasp, behind "show all"). A filter shows matches of both. `section` (policies | builtin | replays) puts a
// subtitle line before its first group; `badges` = the kind badge (in its type's colour) and state/data chips.
function entries() {
  const out = [];
  const byTask = new Map();
  for (const p of state.policies) {
    const task = p.task || "(no task)";
    if (!byTask.has(task)) byTask.set(task, []);
    byTask.get(task).push(p);
  }
  const isRunnable = (p) => (typeof p.runnable === "boolean" ? p.runnable : !!p.checkpoint);
  for (const task of [...byTask.keys()].sort()) {
    const list = byTask.get(task).slice().sort((a, b) =>
      (isRunnable(b) - isRunnable(a)) || String(a.id).localeCompare(String(b.id)));
    for (const p of list) {
      const runnable = isRunnable(p);
      const kind = p.kind || "unknown";
      const kw = kindWords(kind, p.train_config);
      const name = p.label || runName(p.id);
      const notFetched = !runnable && (p.status === "not_fetched" || !p.checkpoint);
      const meta = [];
      if (p.label) meta.push(runName(p.id));
      const demos = demosText(Object.assign({}, p, { recordings: null }));
      if (demos) meta.push(demos);
      const mse = fmt.mse(p.val_mse);
      if (mse) meta.push(`mse ${mse}`);
      const last = lastEvalText(p.last_eval);
      if (last) meta.push(last);
      const tip = [p.label ? `${p.label}\n${p.id}` : p.id, `${kw.word} (${kw.raw})`,
                   p.dataset ? `dataset: ${p.dataset}${demosText(p) ? ` · ${demosText(p)}` : ""}` : null,
                   p.train_config ? `config: ${p.train_config}` : null,
                   p.data_source ? `data: ${p.data_source}` : null,
                   p.checkpoint ? `checkpoint: ${p.checkpoint}` : null,
                   !runnable ? (p.not_runnable_reason || "no checkpoint here: fetch it first (train/fetch.sh)") : null,
                   p.note ? `note: ${p.note}` : null,
                   p.last_eval ? `last sim: ${last} (${p.last_eval.date || ""} ${p.last_eval.tag || ""})` : null,
                   P.picker_tip || null]
        .filter(Boolean).join("\n");
      const badges = [{ text: kw.short, cls: "t-policy" }];
      if (!runnable) badges.push({ text: notFetched ? (P.not_fetched || "not fetched") : (P.not_runnable || "not runnable"),
                                   cls: "kind-state" });
      if (p.data_source && p.data_source !== "dataproc") badges.push({ text: p.data_source, cls: "kind-data" });
      out.push({ group: task, section: "policies", id: p.id, name, kind, type: "policy", runnable, tip,
                 meta: meta.join(" · "), badges, hide: p.smoke ? "smoke" : null,
                 search: `${p.id} ${name} ${kind} ${kw.word} ${p.dataset || ""} ${p.data_source || ""}`.toLowerCase() });
    }
  }
  for (const b of state.builtins) {
    const last = b === "hold_still" ? lastEvalText(state.builtinLastEval) : null;
    const item = state.builtinItems.find((x) => x && x.id === b) || {};
    const kw = kindWords(item.kind || b, null);
    out.push({ group: "built-in", section: "builtin", id: b, name: b, type: "builtin",
               badges: [{ text: kw.short, cls: "t-builtin" }],
               meta: [b === "hold_still" ? (P.no_policy || "no policy: the arm holds still") : "", last].filter(Boolean).join(" · "),
               tip: [b, (H.TYPES && H.TYPES.builtin && H.TYPES.builtin.tip) || ""].filter(Boolean).join("\n"),
               runnable: true, hide: null, search: `${b} ${kw.word}`.toLowerCase(), kind: item.kind || b });
  }
  const replays = state.replays.slice().sort((a, b) =>
    String(a.group || "").localeCompare(String(b.group || ""))
    || String(a.dataset || "").localeCompare(String(b.dataset || ""))
    || (!!b.grasp_ok - !!a.grasp_ok) || ((a.index ?? 0) - (b.index ?? 0)));
  for (const r of replays) {
    const ok = !!r.grasp_ok;
    const err = typeof r.height_error_m === "number" ? `height error ${fmt.mm(r.height_error_m)}` : null;
    const last = lastEvalText(r.last_eval);
    const multi = Array.isArray(r.dataset_recordings) && r.dataset_recordings.length > 1;
    const rec = multi && r.recording ? r.recording : null;
    const where = [r.group, r.dataset].filter(Boolean).join("/") || "?";
    out.push({ group: `replay · ${where}`, section: "replays", id: r.id, name: r.name || `#${r.index}`,
               badges: [{ text: "replay", cls: "t-replay" },
                        { text: ok ? "grasp ok" : "grasp ✗", cls: ok ? "kind-grasp-ok" : "kind-grasp-bad" }],
               meta: [`#${r.index}`, rec, err, last].filter(Boolean).join(" · "), runnable: true, kind: "replay",
               type: "replay",
               tip: [r.id, ok ? "the replay placement grasps (|height error| < 2 cm)" : "the replay placement misses the grasp",
                     err, r.recording ? `recording: ${r.recording}` : null, r.error ? `error: ${r.error}` : null,
                     (H.TYPES && H.TYPES.replay && H.TYPES.replay.tip) || null].filter(Boolean).join("\n"),
               hide: ok ? null : "nograsp",
               search: `${r.id} ${r.name || ""} replay ${r.dataset || ""} ${r.recording || ""}`.toLowerCase() });
  }
  return out;
}

function findEntry(id) {
  return entries().find((e) => e.id === id) || null;
}

function toggleButton(text, expanded, onclick) {
  return el("button", { type: "button", class: "fold", "aria-expanded": expanded ? "true" : "false", onclick, text });
}

function pickerRow(e) {
  const input = el("input", { type: "radio", name: "policy", value: e.id, disabled: !e.runnable,
                              checked: e.id === state.selected });
  input.addEventListener("change", () => select(e.id));
  return el("label", { class: `prow${e.runnable ? "" : " disabled"}${e.id === state.selected ? " selected" : ""}`,
                       title: e.tip },
    input,
    el("span", { class: "name", text: e.name }),
    el("span", { class: "badges" }, ...e.badges.map((b) => el("span", { class: `kind ${b.cls}`, text: b.text }))),
    el("span", { class: "meta", text: e.meta || " " }));
}

function renderPicker() {
  const box = $("picker");
  const filter = $("picker-filter").value.trim().toLowerCase();
  const all = entries();
  const key = JSON.stringify([filter, state.selected, [...state.openSmoke], state.showAllReplays,
                              all.map((e) => [e.id, e.meta, e.badges.map((b) => b.text)])]);
  if (key === state.shown.picker) return;
  state.shown.picker = key;
  const scroll = box.scrollTop;
  box.replaceChildren();
  if (!state.policiesLoaded) { box.append(el("p", { class: "muted pad", text: "loading…" })); return; }
  const list = filter ? all.filter((e) => e.search.includes(filter)) : all;
  if (!list.length) {
    box.append(el("p", { class: "muted pad", text: all.length ? "nothing matches the filter" : "no policies found" }));
    return;
  }
  const groups = new Map();
  for (const e of list) {
    if (!groups.has(e.group)) groups.set(e.group, []);
    groups.get(e.group).push(e);
  }
  let replayToggleShown = false;
  let section = null;
  const SECTION_TITLES = { policies: P.group_policies, builtin: P.group_builtin, replays: P.group_replays };
  for (const [group, items] of groups) {
    if (items[0].section !== section) {
      section = items[0].section;
      if (SECTION_TITLES[section]) box.append(el("div", { class: "section-title", text: SECTION_TITLES[section] }));
    }
    const isReplay = group.startsWith("replay");
    if (isReplay && !replayToggleShown && !filter) {
      replayToggleShown = true;
      const hidden = all.filter((e) => e.hide === "nograsp").length;
      if (hidden) {
        box.append(el("div", { class: "fold-row" }, toggleButton(
          state.showAllReplays ? `replays: hide the ${hidden} without a grasp` : `replays: show all (${hidden} more without a grasp)`,
          state.showAllReplays, () => { state.showAllReplays = !state.showAllReplays; save(); renderPicker(); })));
      }
    }
    const shown = filter ? items : items.filter((e) => e.hide !== "nograsp" || state.showAllReplays || e.id === state.selected);
    const main = shown.filter((e) => e.hide !== "smoke" || filter);
    const smoke = filter ? [] : shown.filter((e) => e.hide === "smoke");
    if (!main.length && !smoke.length) continue;
    box.append(el("div", { class: "group-title", text: group }));
    for (const e of main) box.append(pickerRow(e));
    if (smoke.length) {
      const open = state.openSmoke.has(group) || smoke.some((e) => e.id === state.selected);
      box.append(el("div", { class: "fold-row" }, toggleButton(`smoke / old (${smoke.length})`, open, () => {
        if (state.openSmoke.has(group)) state.openSmoke.delete(group); else state.openSmoke.add(group);
        save();
        renderPicker();
      })));
      if (open) for (const e of smoke) box.append(pickerRow(e));
    }
  }
  box.scrollTop = scroll;
}

function select(id) {
  state.selected = id;
  save();
  renderPicker();
  applySettingsRules();
  renderSelected();
  renderButtons();
  renderSteps();
  if ($("results-only-selected").checked) { state.shown.results = ""; }
  renderResults();
}

// --------------------------------------------------------------------------------------------------- settings
const SETTINGS = ["set-rollouts", "set-seed", "set-home", "set-look", "set-timeout", "set-tap", "set-precision",
                  "set-steps", "set-exech", "set-tag", "set-starts"];
// 2026-11-21: the defaults became look robot and the arm's bf16 / 8 steps / exec_h 12. Settings saved before that
// (no settingsVersion 2) do not restore these, so an old "station default" / empty field does not hide them.
const SETTINGS_V2_ONLY = ["set-look", "set-precision", "set-steps", "set-exech"];
const ARM = { precision: "bf16", diffusion_steps: 8, exec_h: 12 };   // the real arm's policy settings
// 2026-11-24: default placement seed 523: rollout 0 puts the eraser at ~(0.534, -0.068), the in-range spot nearest the
// teleop_marker demos' objects (x 0.53, y -0.16..-0.03). Settings saved before (settingsVersion < 3) keep their seed
// out, so an old saved seed does not hide the new default.
const DEFAULT_SEED = 523;
const SETTINGS_V3_ONLY = ["set-seed"];

function save() {
  try {
    const s = { settingsVersion: 3, selected: state.selected, advanced: $("advanced").open,
                more: $("more").open, hideHelp: $("hide-help").checked,
                showAllReplays: state.showAllReplays,
                openSmoke: [...state.openSmoke],
                onlySelected: $("results-only-selected").checked };
    for (const id of SETTINGS) { const n = $(id); s[id] = n.type === "checkbox" ? n.checked : n.value; }
    s["set-starts"] = state.startsPref;            // the choice, also while it shows random (not allowed)
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch (e) { /* no storage: fine */ }
}

function restore() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (!s) return;
    state.selected = s.selected || null;
    state.showAllReplays = !!s.showAllReplays;
    if (Array.isArray(s.openSmoke)) state.openSmoke = new Set(s.openSmoke);
    $("advanced").open = !!s.advanced;
    $("more").open = !!s.more;
    $("hide-help").checked = !!s.hideHelp;
    $("results-only-selected").checked = !!s.onlySelected;
    for (const id of SETTINGS) {
      if (!(id in s)) continue;
      const ver = Number(s.settingsVersion) || 1;
      if (ver < 2 && SETTINGS_V2_ONLY.includes(id)) continue;
      if (ver < 3 && SETTINGS_V3_ONLY.includes(id)) continue;
      const n = $(id);
      if (n.type === "checkbox") n.checked = !!s[id]; else n.value = s[id];
    }
    // P2: an old saved "trainstart" (or anything not offered any more) means the start pose
    if (![...$("set-home").options].some((o) => o.value === $("set-home").value)) $("set-home").value = "start";
    if (![...$("set-look").options].some((o) => o.value === $("set-look").value)) $("set-look").value = "robot";
    if (![...$("set-precision").options].some((o) => o.value === $("set-precision").value)) $("set-precision").value = "bf16";
    state.startsPref = s["set-starts"] === "set" ? "set" : "random";
  } catch (e) { /* ignore */ }
  applyHideHelp();
}

function applyHideHelp() {
  document.body.classList.toggle("hide-help", $("hide-help").checked);
}

function intOrNull(id, min, max, label) {
  const v = $(id).value.trim();
  if (v === "") return { value: null };
  const n = Number(v);
  if (!Number.isInteger(n) || n < min || (max !== undefined && n > max)) {
    return { error: `${label}: a whole number${max !== undefined ? ` ${min}..${max}` : ` >= ${min}`}` };
  }
  return { value: n };
}

// May "Eraser starts" be the start set for the picked entry and these settings? {show, allowed, reason}.
// show = false: no start set (an older server, or start_set null = the file is missing): the option is hidden.
// The driver's / runner's rules: not for a replay; needs Arm start = start pose; a trained policy only if its
// run folder is in the set's checkpoints (catalog starts_set); at most n_starts rollouts.
function startsRule() {
  const W = F.starts || {};
  if (!state.startSetKnown || !state.startSet) return { show: false, allowed: false, reason: "" };
  const info = catalogItem(state.selected);
  const n = state.startSet.n_starts;
  const rollouts = Number($("set-rollouts").value) || 1;
  const v = setVars();
  if (info && info.type === "replay") return { show: true, allowed: false, reason: fillText(W.no_replay, v) };
  if ($("set-home").value !== (state.startSet.needs_home || "start")) {
    return { show: true, allowed: false, reason: fillText(W.no_home, v) };
  }
  if (info && info.type === "policy" && info.item && info.item.starts_set === false) {
    return { show: true, allowed: false, reason: fillText(W.no_split, v) };
  }
  if (typeof n === "number" && rollouts > n) return { show: true, allowed: false, reason: fillText(W.too_many, v) };
  return { show: true, allowed: true, reason: "" };
}

function effectiveStarts() {
  const r = startsRule();
  return r.show && r.allowed && state.startsPref === "set" ? "set" : "random";
}

// Greys / notes that follow the picked entry and the settings: the start-set select, the seed, the policy
// settings' note and the "not the arm's settings" mark.
function applySettingsRules() {
  const sel = $("set-starts");
  const rule = startsRule();
  const setOpt = sel.querySelector('option[value="set"]');
  setOpt.hidden = !rule.show;
  setOpt.disabled = !rule.show;
  sel.disabled = rule.show && !rule.allowed;
  sel.value = effectiveStarts();
  sel.title = rule.reason || "";
  const hs = $("help-starts");
  hs.textContent = rule.reason || hs.dataset.default || "";
  hs.classList.toggle("warn", !!rule.reason && state.startsPref === "set");
  $("set-starts").closest(".frow").classList.toggle("field-off", sel.disabled);

  const info = catalogItem(state.selected);
  const type = info ? info.type : "policy";
  const seedOff = type === "replay" ? F.seed && F.seed.off_replay : effectiveStarts() === "set" ? F.seed && F.seed.off_set : null;
  $("set-seed").disabled = !!seedOff;
  $("btn-dice").disabled = !!seedOff;
  $("set-seed").title = seedOff || "";
  const hseed = $("help-seed");
  hseed.textContent = seedOff || hseed.dataset.default || "";
  $("set-seed").closest(".frow").classList.toggle("field-off", !!seedOff);

  const note = $("policy-note");
  const noNet = type === "replay" || type === "builtin";
  note.textContent = noNet ? ((F.policy_settings && F.policy_settings.not_used) || "not used") : "";
  note.classList.toggle("hidden", !noNet);
  $("not-arm").classList.toggle("hidden", !notArm());
}

function notArm() {
  const steps = $("set-steps").value.trim(), exech = $("set-exech").value.trim();
  return ($("set-precision").value || ARM.precision) !== ARM.precision
    || (steps !== "" && Number(steps) !== ARM.diffusion_steps) || (exech !== "" && Number(exech) !== ARM.exec_h);
}

function resetToArm() {
  $("set-precision").value = ARM.precision;
  $("set-steps").value = String(ARM.diffusion_steps);
  $("set-exech").value = String(ARM.exec_h);
  onSettingsChange();
}

// The settings as a RunRequest without the policy (also for the step panel's preview), or {error}.
function readSettings() {
  const rollouts = intOrNull("set-rollouts", 1, 100, "rollouts");
  const seed = intOrNull("set-seed", 0, undefined, "seed");
  const steps = intOrNull("set-steps", 1, undefined, "diffusion steps");
  const exech = intOrNull("set-exech", 1, 16, "exec_h");
  for (const r of [rollouts, seed, steps, exech]) if (r.error) return { error: r.error };
  const timeout = Number($("set-timeout").value);
  if (!Number.isFinite(timeout) || timeout <= 0) return { error: "timeout: seconds > 0" };
  const tag = $("set-tag").value.trim() || "ui";
  if (!/^[A-Za-z0-9_.-]+$/.test(tag)) return { error: "tag: letters, digits, _ . - only" };
  const request = {
    policy: state.selected,
    rollouts: rollouts.value === null ? 1 : rollouts.value,
    seed: seed.value === null ? DEFAULT_SEED : seed.value,
    home: $("set-home").value,
    look: $("set-look").value,
    timeout_s: timeout,
    tap: $("set-tap").checked,
    tag,
    precision: $("set-precision").value || "bf16",
    diffusion_steps: steps.value === null ? 8 : steps.value,
    exec_h: exech.value === null ? 12 : exech.value,
  };
  // U6: only to a server whose catalog has "start_set" (an older one refuses unknown keys)
  if (state.startSetKnown) request.starts = effectiveStarts();
  return { request };
}

// The RunRequest (DESIGN.md section 2), or {error}.
function buildRequest() {
  if (!state.selected) return { error: "pick a policy first" };
  const entry = findEntry(state.selected);
  if (state.policiesLoaded && !entry) return { error: "the selected policy is not in the list any more" };
  if (entry && !entry.runnable) return { error: "this run has no checkpoint here (not fetched)" };
  return readSettings();
}

// ------------------------------------------------------------------------------------------ start pose (P2)
async function loadStartPose() {
  try {
    const data = await api("GET", "/api/start_pose");
    state.startPose = data.ok === false ? { error: data.error } : data;
  } catch (e) {
    state.startPose = { error: String(e.message || e) };
  }
  renderStartPose();
}

// One line (ellipsis); the whole text and the details in the tooltip; click reloads.
function renderStartPose() {
  const box = $("start-pose");
  const sp = state.startPose;
  const W = F.home || {};
  const show = (text, tip, bad) => {
    box.textContent = text;
    box.title = [text, tip, "(click to reload)"].filter(Boolean).join("\n");
    box.classList.toggle("bad", !!bad);
  };
  if ($("set-home").value === "deploy") {
    show(W.deploy_line || "deploy home", W.deploy_tip, false);
    return;
  }
  if (!sp) { show("start pose: loading… (the first load takes a few seconds)", "", false); return; }
  if (sp.error) { show(`start pose: ${sp.error}`, "", true); return; }
  if (!sp.set) { show(sp.text || "start pose: not set", sp.file || "", true); return; }
  const bad = Object.entries(sp.checks || {}).filter(([, c]) => !c.ok).map(([k]) => k);
  const git = sp.vs_git || {};
  const text = `start pose: ${sp.text}` + (bad.length ? ` · FAILS ${bad.join(", ")}` : "")
    + (sp.region && !sp.region.ok ? ` · eraser region: ${sp.region.detail}` : "")
    + (git.same === true ? ` · = the real arm's (${git.ref || "upstream"})` : git.text ? ` · ${git.text}` : "");
  const tip = `${sp.file}\nsource: ${sp.source || "?"}\nq_start_rad: [${(sp.q_start_rad || []).map((v) => v.toFixed(4)).join(", ")}]`
    + Object.entries(sp.vs_demos || {}).map(([k, v]) => `\n${k}: ${v.text}`).join("");
  show(text, tip, bad.length > 0 || git.same === false);
}

// ----------------------------------------------------------------------------------- run header (U6)
// Line 1: type badge + title; line 2: task · kind · dataset · chips; line 3: the settings. For the play card
// ("Ready to run", from the page's settings) and the banner above the external view (from status.request).
function headerLines(id, req) {
  const info = catalogItem(id) || { type: "policy", item: null };
  const it = info.item || {};
  const T = (H.TYPES && H.TYPES[info.type]) || {};
  const badge = el("span", { class: `tbadge t-${info.type}`, title: T.tip || null },
                   T.badge || info.type.toUpperCase(), T.sub ? el("span", { class: "tsub", text: ` · ${T.sub}` }) : null);
  const title = info.type === "policy" ? (it.label || runName(String(id)))
    : info.type === "replay" ? (it.name || `#${info.index ?? "?"}`) : String(id);
  const line1 = el("div", { class: "rh-line1" }, badge, el("span", { class: "rh-title", text: title, title: String(id) }));

  const l2 = [];
  const sep = () => (l2.length ? " · " : null);
  const add = (...nodes) => { const s = sep(); if (s) l2.push(s); l2.push(...nodes); };
  const chips = [];
  if (info.type === "policy") {
    if (it.task) add(it.task);
    const kw = kindWords(info.kind, it.train_config);
    if (kw.raw === "unknown" || kw.raw === "other") add(kw.word);
    else add(kw.word, " ", el("span", { class: "raw", text: `(${kw.raw})` }));
    if (it.dataset) { const d = demosText(it); add(d ? `${it.dataset} · ${d}` : it.dataset); }
    if (it.data_source && it.data_source !== "dataproc") chips.push({ text: it.data_source, cls: "kind-data" });
    if (it.smoke) chips.push({ text: P.smoke || "smoke", cls: "kind-state" });
  } else if (info.type === "replay") {
    const where = [it.group, it.dataset].filter(Boolean).join("/");
    if (where) add(where);
    add(fillText(P.demo || "demo #{i}", { i: info.index ?? "?" }));
    if (Array.isArray(it.dataset_recordings) && it.dataset_recordings.length > 1 && it.recording) add(it.recording);
    if (typeof it.grasp_ok === "boolean") {
      chips.push({ text: it.grasp_ok ? "grasp ok" : "grasp ✗", cls: it.grasp_ok ? "kind-grasp-ok" : "kind-grasp-bad" });
    }
  } else {
    add(P.no_policy || "no policy: the arm holds still");
  }
  for (const c of chips) l2.push(" ", el("span", { class: `kind ${c.cls}`, text: c.text }));
  const lines = [line1, el("div", { class: "rh-line2" }, ...l2)];
  if (req) lines.push(el("div", { class: "rh-line3", text: settingsLine(req, info.type) }));
  return lines;
}

function settingsLine(req, type) {
  const n = req.rollouts ?? 1;
  const parts = [`${n} ${n === 1 ? (SHORT.rollout || "rollout") : (SHORT.rollouts || "rollouts")}`];
  parts.push((SHORT.look && SHORT.look[req.look]) || req.look || "robot cell");
  parts.push((SHORT.home && SHORT.home[req.home]) || req.home || "start pose");
  const set = req.starts === "set" && type !== "replay";
  if (type !== "replay") parts.push(fillText((SHORT.starts && SHORT.starts[set ? "set" : "random"]) || (set ? "start set" : "random"), setVars()));
  parts.push(fillText(SHORT.limit || "{T} s limit", { T: req.timeout_s ?? 60 }));
  if (type !== "replay" && !set) parts.push(fillText(SHORT.seed || "seed {seed}", { seed: req.seed ?? DEFAULT_SEED }));
  if (req.tap) parts.push(SHORT.tap || "tap");
  if (req.tag && req.tag !== "ui") parts.push(fillText(SHORT.tag || "tag {tag}", { tag: req.tag }));
  if (type === "policy" && (req.precision !== ARM.precision || req.diffusion_steps !== ARM.diffusion_steps
                            || req.exec_h !== ARM.exec_h) && req.precision !== undefined) {
    parts.push(fillText(SHORT.not_arm || "{precision} · {steps} steps · exec_h {exech}",
                        { precision: req.precision ?? "bf16", steps: req.diffusion_steps ?? 8, exech: req.exec_h ?? 12 }));
  }
  return parts.join(" · ");
}

// "5 rollouts in the robot cell. Each: home → start pose → eraser at a seeded spot (seed 0) → policy for up to 60 s."
function planSentence(req, type, index) {
  const n = req.rollouts ?? 1;
  const look = (F.look && F.look.words && F.look.words[req.look]) || req.look;
  const v = setVars({ N: n, rollouts: n === 1 ? (SHORT.rollout || "rollout") : (SHORT.rollouts || "rollouts"), look,
                      seed: req.seed ?? DEFAULT_SEED, T: req.timeout_s ?? 60, i: index ?? "?" });
  const chain = [P.plan_home || "home"];
  if (req.home !== "deploy") chain.push(P.plan_start || "start pose");
  chain.push(type === "replay" ? P.plan_replay : req.starts === "set" ? P.plan_set : P.plan_random);
  chain.push(type === "builtin" ? P.plan_hold : type === "replay" ? P.plan_replay_ctl : P.plan_policy);
  return `${fillText(P.plan_head || "{N} {rollouts} in {look}.", v)} ${P.plan_each || "Each:"} `
    + `${chain.filter(Boolean).map((c) => fillText(c, v)).join(" → ")}.`;
}

function renderSelected() {
  const box = $("selected");
  const b = state.selected ? buildRequest() : null;
  const key = JSON.stringify([state.selected, isBusy(), b, state.policiesLoaded, state.startSetKnown,
                              state.catalogVersion]);
  if (key === state.shown.selected) return;
  state.shown.selected = key;
  const plan = $("plan");
  if (!state.selected) {
    box.textContent = P.nothing || "nothing selected: pick a policy above";
    plan.textContent = "";
    return;
  }
  const info = catalogItem(state.selected);
  box.title = state.selected;
  const head = el("div", { class: "rh-ready", text: isBusy() ? (P.next || "Next run") : (P.ready || "Ready to run") });
  const lines = headerLines(state.selected, b.request || null);
  if (b.error) lines.push(el("div", { class: "rh-line3 run-msg error", text: b.error }));
  box.replaceChildren(head, ...lines);
  plan.textContent = b.request ? planSentence(b.request, info ? info.type : "policy", info ? info.index : null) : "";
}

// The banner above the external view: what this run is (status.request.policy, looked up in the catalog) and a
// state chip; with no run, one muted sentence. Always the same height (the video under it never moves).
function renderRunHeader() {
  const box = $("run-header");
  const s = state.status;
  const req = s && s.request;
  const show = !!req && (isBusy() || s.state === "done" || s.state === "failed");
  const key = JSON.stringify(show ? [s.run_id, s.state, req, state.policiesLoaded, state.catalogVersion] : null);
  if (key === state.shown.header) return;
  state.shown.header = key;
  if (!show) {
    box.className = "run-header idle";
    box.title = "";
    box.replaceChildren(el("div", { text: P.banner_idle || "No run yet." }));
    return;
  }
  const lines = headerLines(req.policy, req);
  const word = s.state === "stopping" ? P.banner_stopping : s.state === "done" ? P.banner_done
    : s.state === "failed" ? P.banner_failed : P.banner_now;
  lines[0].append(el("span", { class: `state rh-state state-${s.state}`, text: word || s.state }));
  box.className = "run-header";
  box.replaceChildren(...lines);
  box.title = JSON.stringify(req, null, 1);
}

// ---------------------------------------------------------------------------------------------- play and stop
// Play waits for /api/policies too: before it, the request would miss `starts` (a saved "set" would be dropped).
function canPlay() {
  return state.connected && state.policiesLoaded && !state.inFlight && !isBusy() && !state.expectStart
    && !!state.selected;
}

async function play() {
  if (!canPlay()) return;
  const b = buildRequest();
  if (b.error) { showRunMsg(b.error, "error"); return; }
  state.inFlight = true;
  showRunMsg("starting…", "");
  renderButtons();
  try {
    const r = await api("POST", "/api/run", b.request);
    if (r.ok) {
      state.expectStart = performance.now();
      if (r.run_id) state.runId = String(r.run_id);
      state.evalsBefore = new Set(state.evals.map((r) => r.eval_dir));
      state.rolloutRecords.clear();
      state.runEnd = null;
      state.frames = null;
      state.stopPresses = 0;
      state.shown.chips = "";
      showRunMsg(`started ${r.run_id || ""}`.trim(), "ok");
      setTimeout(() => { if (state.expectStart) { state.expectStart = 0; renderButtons(); } }, 4000);
    } else {
      if (Array.isArray(r.problems) && r.problems.length) {
        showRunMsg(null, "error", ["not started:", ...r.problems]);
      } else {
        showRunMsg(`not started: ${r.error}`, "error");
      }
    }
  } catch (e) {
    showRunMsg(e.message, "error");
  } finally {
    state.inFlight = false;
    renderButtons();
  }
}

async function stop() {
  if (!isBusy() && !state.expectStart) return;
  const force = state.stopPresses >= 1 || (state.status && state.status.state === "stopping");
  state.stopPresses += 1;
  showRunMsg(force ? "force stop sent: killing the run…" : "stopping… (the rollout ends cleanly; press again to force)", "");
  renderButtons();
  try {
    const r = await api("POST", "/api/stop", {});
    if (!r.ok) showRunMsg(`stop: ${r.error}`, "error");
  } catch (e) {
    showRunMsg(e.message, "error");
  }
}

function showRunMsg(text, kind, lines) {
  const m = $("run-msg");
  if (lines) {
    m.replaceChildren(el("div", { text: lines[0] }),
                      el("ul", {}, ...lines.slice(1).map((l) => el("li", { text: l }))));
  } else {
    m.textContent = text;
  }
  m.className = `run-msg${kind ? ` ${kind}` : ""}`;
}

const STOP_ICON = $("btn-stop").querySelector("svg").cloneNode(true);

function renderButtons() {
  const B = P.buttons || {};
  const playBtn = $("btn-play"), stopBtn = $("btn-stop");
  const busy = isBusy() || !!state.expectStart;
  playBtn.disabled = !canPlay();
  // why Play is off: the tooltip, and the line under the buttons (a disabled button's tooltip is easy to miss)
  const why = !state.connected ? B.tip_offline || "the UI server is not connected"
    : !state.policiesLoaded ? (state.catalogError ? B.tip_catalog || "the policy list did not load"
                                                  : B.tip_loading || "the policy list is still loading")
    : isBusy() ? B.tip_busy || "a run is going: stop it first"
    : !state.selected ? B.tip_nothing || "pick a policy first" : "";
  playBtn.title = playBtn.disabled ? why : B.tip_play || "start the run (Space)";
  $("play-why").textContent = playBtn.disabled && !busy ? why : "";
  const playLabel = playBtn.querySelector(".btn-label");
  if (playLabel.textContent !== (B.play || "Play")) playLabel.textContent = B.play || "Play";

  stopBtn.disabled = !busy;
  const stopping = state.stopPresses >= 1 || (state.status && state.status.state === "stopping");
  const label = state.stopPresses >= 2 ? B.killing || "killing…" : stopping ? B.force || "Force stop" : B.stop || "Stop";
  const sub = stopping && state.stopPresses < 2 ? B.stopping || "stopping…" : "";
  if (stopBtn.dataset.key !== `${label}|${sub}`) {
    stopBtn.dataset.key = `${label}|${sub}`;
    stopBtn.replaceChildren(STOP_ICON.cloneNode(true), el("span", { class: "btn-label", text: label }),
                            sub ? el("span", { class: "sub", text: sub }) : null);
  }
  stopBtn.classList.toggle("force", stopping);
  stopBtn.title = stopping ? B.tip_force || "press again to kill the run now" : B.tip_stop || "stop after the current step (Space)";
}

// ---------------------------------------------------------------------------------------------- status render
function setDD(id, text, cls) {
  const d = $(id);
  if (text === null || text === undefined || text === "") { d.textContent = "no data"; d.className = "nodata"; }
  else { d.textContent = text; d.className = cls || ""; }
}

function renderConnection() {
  const link = $("link"), banner = $("banner");
  if (state.connected) {
    link.textContent = "connected"; link.className = "badge ok";
    banner.classList.add("hidden");
  } else {
    link.textContent = "disconnected"; link.className = "badge bad";
    banner.textContent = "The UI server is not reachable: retrying… (python -m sim_deploy.ui)";
    banner.className = "banner warn";
  }
  renderStatus();
}

function renderStatus() {
  const s = state.status;
  const st = !state.connected && !s ? "offline" : (s ? s.state : "idle");
  const badge = $("state-badge");
  badge.textContent = st === "stopping" ? "stopping…" : st;
  badge.className = `state state-${st}`;
  setDD("st-state", st);
  const busy = isBusy();
  if (s && s.rollouts) setDD("st-rollout", `${s.rollout ?? "-"}/${s.rollouts}`);
  else setDD("st-rollout", busy ? "-" : "-");
  setDD("st-phase", s && s.phase ? s.phase : "-");
  setDD("st-elapsed", s && s.run_id ? fmt.elapsed(s.elapsed_s) || "-" : "-");

  renderRunHeader();

  const notes = s && Array.isArray(s.notes) ? s.notes : [];
  const notesBox = $("notes");
  const notesKey = notes.join("\n");
  if (notesBox.dataset.key !== notesKey) {
    notesBox.dataset.key = notesKey;
    notesBox.replaceChildren(...notes.map((n) => el("li", { text: n })));
  }
  notesBox.classList.toggle("hidden", !notes.length);

  const errBox = $("last-error");
  if (s && s.last_error) { errBox.textContent = s.last_error; errBox.classList.remove("hidden"); }
  else errBox.classList.add("hidden");

  // a failed run opens the raw log once (the viewer may close it again)
  if (s && s.state === "failed" && s.run_id && state.logOpenedFor !== s.run_id) {
    state.logOpenedFor = s.run_id;
    $("log-details").open = true;
  }

  if (state.lastBusy !== busy) { state.lastBusy = busy; renderSelected(); }   // "Ready to run" <-> "Next run"
  renderSteps();
  renderChips();
  renderLog();
  renderRunResult();
}

// ------------------------------------------------------------------------------------- the step panel (U6)
// computeSteps() (steps.js) gives the steps; this draws them as a timeline (10 px nodes on a connector, an elapsed
// column). Recorded rollouts that computeSteps does not show in detail fold to one line each with their outcome chip
// (the last 3; earlier ones in one line). Rebuilt only when a row's state/title/help or the fold changes; the live
// details ("12.3 s of 60 s", "eraser up ...") and the running step's time are updated in place, so the 5-10 Hz
// status and telemetry do not rebuild the list (a fold button keeps its focus).
const RLINES_SHOWN = 3;

// success -> ok, fail/timeout -> warn, a fault or crash -> fail, the rest (stopped, not run...) neutral
function outcomeCat(o) {
  if (o === "success") return "ok";
  if (o === "fail" || o === "timeout") return "warn";
  if (["fault", "crashed", "sim_error", "home_failed", "start_pose_failed", "start_failed"].includes(o)) return "fail";
  return "neutral";
}

// "12.3 s" for a step: done = until the next step began, now = until the status' elapsed; "" when unknown
function stepTime(m, id, stepState) {
  if (!m || !m.times || !m.times.has(id)) return "";
  const t0 = m.times.get(id);
  if (stepState === "now") {
    const el_ = state.status && typeof state.status.elapsed_s === "number" ? state.status.elapsed_s : null;
    return el_ === null ? "" : fmt.elapsed(Math.max(0, el_ - t0)) || "";
  }
  if (stepState !== "done") return "";
  const i = m.order.indexOf(id);
  const next = i >= 0 && i + 1 < m.order.length ? m.times.get(m.order[i + 1]) : null;
  return typeof next === "number" ? fmt.elapsed(Math.max(0, next - t0)) || "" : "";
}

// a recorded rollout k (1-based): from its first step to the first step after its last one
function rolloutTime(m, k) {
  if (!m || !m.order) return "";
  const idx = m.order.map((id, i) => (id.endsWith(`:${k}`) ? i : -1)).filter((i) => i >= 0);
  if (!idx.length) return "";
  const t0 = m.times.get(m.order[idx[0]]);
  const after = idx[idx.length - 1] + 1;
  const t1 = after < m.order.length ? m.times.get(m.order[after]) : null;
  return typeof t1 === "number" ? fmt.elapsed(Math.max(0, t1 - t0)) || "" : "";
}

function outcomeChip(o, k) {
  const word = H.OUTCOMES && H.OUTCOMES[o];
  return el("span", { class: `chip o-${String(o).replace(/[^a-z_]/gi, "")}`, title: word ? `${o}: ${word}` : String(o),
                      "aria-label": `rollout ${k}: ${o}` },
    el("span", { class: "g", "aria-hidden": "true", text: shortOutcome(o) }), String(o).replace(/_/g, " "));
}

function renderSteps() {
  const s = state.status;
  const finished = !!s && (s.state === "done" || s.state === "failed") && !!s.request;
  const mode = isBusy() && s.request ? "live" : finished ? "finished" : "preview";
  let items = [];
  try {
    if (typeof computeSteps !== "function") throw new Error("steps.js did not load");
    if (mode === "preview") {
      const r = readSettings();
      const req = r.request || { rollouts: 1, home: $("set-home").value, look: $("set-look").value, timeout_s: 60 };
      const info = catalogItem(state.selected);
      items = computeSteps(null, null, Object.assign({}, req, { policy: state.selected }), info,
                           { setName: setName() });
    } else {
      items = computeSteps(s, sceneTelemetry(), s.request, catalogItem(s.request.policy),
                           state.stepMemo || newMemo(s.run_id || null));
    }
  } catch (e) {
    items = [{ group: "", key: "error", title: `step panel: ${e.message}`, help: "", tip: "", state: "fail", detail: "" }];
  }
  const m = mode === "preview" ? null : state.stepMemo;
  const bring = items.filter((i) => STEP_STAGE_OF_SAFE(i.key) === "bringup");
  const fold = mode !== "preview" && bring.length > 0 && bring.every((i) => i.state === "done");
  const open = fold && state.bringupOpen;
  const modeChip = $("steps-mode");
  modeChip.textContent = mode === "live" ? (P.steps_live || "live")
    : mode === "finished" ? (P.steps_done || "finished") : (P.steps_preview || "plan");
  modeChip.className = `mode-chip m-${mode}`;

  const stopLine = $("stopping-line");
  stopLine.textContent = P.stopping || "Stopping…";
  stopLine.classList.toggle("hidden", !(s && s.state === "stopping"));

  // the rows: steps, the folded bring-up line, recorded rollouts (one line each, the earlier ones in one)
  const rows = [];
  const gBring = bring.length ? bring[0].group : ((P.groups && P.groups.bringup) || "Bring-up");
  const gRoll = (P.groups && P.groups.rollouts) || "Rollouts";
  const rest = items.filter((i) => !bring.includes(i));
  if (fold) {
    rows.push({ kind: "fold", group: gBring, open });
    if (open) for (const it of bring) rows.push({ kind: "step", group: it.group, it });
  } else {
    for (const it of bring) rows.push({ kind: "step", group: it.group, it });
  }
  if (mode !== "preview") {
    const outcomes = Array.isArray(s.outcomes) ? s.outcomes : [];
    const detailed = rest.find((i) => typeof i.rollout === "number");
    const upTo = detailed ? detailed.rollout - 1 : outcomes.length;
    const rec = [];
    for (let i = 0; i < Math.min(upTo, outcomes.length); i++) {
      const o = outcomes[i];
      if (o !== undefined && !(mode === "live" && o === "not_run")) rec.push({ k: i + 1, o });
    }
    const earlier = rec.length > RLINES_SHOWN ? rec.slice(0, rec.length - RLINES_SHOWN) : [];
    if (earlier.length) {
      const counts = new Map();
      for (const r of earlier) counts.set(r.o, (counts.get(r.o) || 0) + 1);
      const cats = new Set(earlier.map((r) => outcomeCat(r.o)));
      rows.push({ kind: "earlier", group: gRoll, k: earlier[earlier.length - 1].k,
                  cat: cats.size === 1 ? [...cats][0] : "neutral",
                  detail: [...counts].map(([o, n]) => `${n} ${String(o).replace(/_/g, " ")}`).join(" · ") });
    }
    for (const r of rec.slice(earlier.length)) rows.push({ kind: "rline", group: gRoll, k: r.k, o: r.o });
  }
  for (const it of rest) rows.push({ kind: "step", group: it.group, it });
  rows.forEach((r, i) => {
    r.first = i === 0 || rows[i - 1].group !== r.group;
    r.last = i === rows.length - 1 || rows[i + 1].group !== r.group;
  });

  const sig = rows.map((r) => r.kind === "step" ? [r.group, r.it.key, r.it.rollout, r.it.title, r.it.help, r.it.state]
    : [r.kind, r.group, r.k ?? null, r.o ?? r.detail ?? null, r.open ?? null]);
  const key = JSON.stringify([mode, m && m.bringupS, m ? m.order.length : 0, sig]);
  const SS = P.step_states || {};
  if (key !== state.shown.steps) {
    state.shown.steps = key;
    const ol = $("steps");
    ol.replaceChildren();
    state.stepNodes = [];
    let group = null;
    for (const r of rows) {
      if (r.group !== group) {
        group = r.group;
        if (group) ol.append(el("li", { class: "s-group overline", text: group }));
      }
      const pos = `${r.first ? " first" : ""}${r.last ? " last" : ""}`;
      const node = el("span", { class: "s-node", "aria-hidden": "true" });
      if (r.kind === "fold") {
        const secs = m && typeof m.bringupS === "number" ? fmt.elapsed(m.bringupS) : "";
        ol.append(el("li", { class: `step s-done step-fold${pos}` }, node,
          el("span", { class: "s-title" }, el("button", {
            type: "button", class: "fold", "aria-expanded": r.open ? "true" : "false",
            text: P.bringup_done || "Sim and policy ready",
            onclick: () => { state.bringupOpen = !state.bringupOpen; renderSteps(); } })),
          el("span", { class: "s-time", text: secs })));
      } else if (r.kind === "earlier") {
        ol.append(el("li", { class: `step s-rline rl-${r.cat}${pos}` }, node,
          el("span", { class: "s-title", text: fillText(P.rollouts_earlier || "Rollouts 1 to {k}", { k: r.k }) }),
          el("span", { class: "s-time" }),
          el("span", { class: "s-detail", text: r.detail })));
      } else if (r.kind === "rline") {
        ol.append(el("li", { class: `step s-rline rl-${outcomeCat(r.o)}${pos}` }, node,
          el("span", { class: "s-title" }, fillText(P.rollout_line || "Rollout {k}", { k: r.k }), outcomeChip(r.o, r.k)),
          el("span", { class: "s-time", text: rolloutTime(m, r.k) })));
      } else {
        const it = r.it;
        const showHelp = mode === "preview" || it.state === "now" || it.state === "fail";
        const id = stepId(it.key, it.rollout);
        const detail = el("span", { class: "s-detail", text: it.detail || "", title: it.detail || null });
        const time = el("span", { class: "s-time", text: stepTime(m, id, it.state) });
        state.stepNodes.push({ detail, time, id, state: it.state });
        ol.append(el("li", { class: `step s-${it.state}${pos}`, title: [it.title, it.help, it.tip].filter(Boolean).join("\n") },
          node,
          el("span", { class: "s-title" }, el("span", { class: "sr", text: `${SS[it.state] || it.state}: ` }), it.title),
          time,
          showHelp && it.help ? el("span", { class: "s-help", text: it.help }) : null,
          detail));
      }
    }
  } else {
    const steps = rows.filter((r) => r.kind === "step");
    steps.forEach((r, i) => {
      const n = state.stepNodes[i];
      if (!n) return;
      const d = r.it.detail || "";
      if (n.detail.textContent !== d) { n.detail.textContent = d; n.detail.title = d; }
      const t = stepTime(m, n.id, n.state);
      if (n.time.textContent !== t) n.time.textContent = t;
    });
  }
}

function STEP_STAGE_OF_SAFE(key) {
  return typeof STEP_STAGE_OF !== "undefined" ? STEP_STAGE_OF[key] : null;
}

// telemetry.sources.<name>.ok says per source whether it is live (the server nulls the fields of a stale source);
// without "sources" (an older server) the age of the last message decides.
const SOURCE_NAMES = ["c0", "state", "external", "policy_view", "c2", "scene", "steps"];

function sourceOk(name) {
  const t = state.telemetry;
  if (!t || performance.now() - state.telemetryAt > STALE_MS) return false;
  if (!t.sources) return true;
  return !!(t.sources[name] && t.sources[name].ok);
}

function renderTelemetry() {
  const t = state.telemetry;
  const fresh = !!t && performance.now() - state.telemetryAt < STALE_MS;
  const src = fresh && t.sources ? t.sources : null;
  const anyLive = fresh && (!src || SOURCE_NAMES.some((k) => src[k] && src[k].ok));
  const sim = $("sim-badge");
  sim.textContent = anyLive ? "sim: live" : "sim: no data";
  sim.className = anyLive ? "badge ok" : "badge";
  sim.title = src ? SOURCE_NAMES.filter((k) => src[k]).map((k) => {
    const r = src[k];
    return `${k}: ${r.ok ? "ok" : "no data"}${r.rate_hz ? ` ${r.rate_hz} Hz` : ""}${r.error ? ` (${r.error})` : ""}`
      + (r.reason ? ` [${r.reason}]` : "");
  }).join("\n") : "telemetry from the running sim";
  if (!fresh) {
    for (const id of ["st-mode", "st-width", "st-eraser", "st-contacts", "st-infer", "st-margin", "st-steps",
                      "st-imgage"]) setDD(id, null);
    return;
  }
  const c2 = sourceOk("c2"), scene = sourceOk("scene"), steps = sourceOk("steps");
  const mode = c2 ? t.mode : null;
  setDD("st-mode", mode ? mode + (t.c2_error ? `: ${t.c2_error}` : "") : null,
        mode === "fault" || mode === "stopped" || t.c2_error ? "bad" : "");
  const width = c2 ? fmt.mm(t.width_m) : null;
  const gap = scene && t.finger_gap_m !== null && t.finger_gap_m !== undefined ? fmt.mm(t.finger_gap_m) : null;
  setDD("st-width", width ? width + (gap ? ` (gap ${gap})` : "") : gap ? `gap ${gap}` : null);
  setDD("st-eraser", scene && t.eraser ? fmt.mm(t.eraser.height_m) : null);
  const ct = scene ? t.contacts : null;
  if (ct) {
    const c = [];
    if (ct.both_fingers) c.push("both fingers");
    else if (ct.finger_eraser) c.push("finger-eraser");
    if (ct.table) c.push("table");
    setDD("st-contacts", c.length ? c.join(", ") : "none",
          ct.table ? "bad" : ct.both_fingers || ct.finger_eraser ? "on" : "");
  } else setDD("st-contacts", null);
  const p = steps ? t.policy : null;
  if (p) {
    const inf = fmt.ms(p.infer_ms);
    setDD("st-infer", inf ? inf + (p.cycle_ms !== null && p.cycle_ms !== undefined ? ` · cycle ${fmt.ms(p.cycle_ms)}` : "") : null);
    setDD("st-margin", fmt.ms(p.margin_ms), p.margin_ms !== null && p.margin_ms !== undefined && p.margin_ms < 0 ? "neg" : "");
    const n = p.steps ?? null;
    setDD("st-steps", n === null ? null
      : `${n}${p.skipped ? ` · ${p.skipped} skipped` : ""}${p.last_skip_reason ? ` (${p.last_skip_reason})` : ""}`,
      p.skipped ? "neg" : "");
    setDD("st-imgage", fmt.ms(p.image_age_ms));
  } else {
    for (const id of ["st-infer", "st-margin", "st-steps", "st-imgage"]) setDD(id, null);
  }
}


// One chip per rollout: done ones by outcome, the current one pulsing, the rest dashed.
function renderChips() {
  const s = state.status;
  const outcomes = s && Array.isArray(s.outcomes) ? s.outcomes : [];
  const N = s && s.rollouts ? s.rollouts : outcomes.length;
  const busy = isBusy();
  const key = JSON.stringify([s ? s.run_id : null, outcomes, N, busy, state.rolloutRecords.size]);
  if (key === state.shown.chips) return;
  state.shown.chips = key;
  const ul = $("chips");
  ul.replaceChildren();
  if (!N) { ul.append(el("li", { class: "muted", text: P.no_run || "no run yet" })); return; }
  // while busy, a not_run (a sim restart left it for the resumed driver) is still to come; the current rollout is
  // the first one not done (rollouts run in order, reruns too)
  const notDone = (o) => o === undefined || (busy && o === "not_run");
  let current = -1;
  if (busy) for (let i = 0; i < N; i++) if (notDone(outcomes[i])) { current = i; break; }
  for (let i = 0; i < N; i++) {
    const o = outcomes[i];
    if (!notDone(o)) {
      const rec = state.rolloutRecords.get(i);
      const word = H.OUTCOMES && H.OUTCOMES[o];
      const tip = [`rollout ${i + 1}: ${o}${word ? ` (${word})` : ""}`];
      if (rec) {
        if (rec.grasped !== undefined) tip.push(`grasped: ${rec.grasped ? "yes" : "no"}`);
        if (rec.lifted !== undefined) tip.push(`lifted: ${rec.lifted ? "yes" : "no"}`);
        if (rec.max_lift_m !== undefined && rec.max_lift_m !== null) tip.push(`max lift: ${fmt.mm(rec.max_lift_m)}`);
        if (rec.note) tip.push(`note: ${rec.note}`);
        if (rec.fault) tip.push(`fault: ${rec.fault}`);
      }
      ul.append(el("li", { class: `chip o-${String(o).replace(/[^a-z_]/gi, "")}`, title: tip.join("\n"),
                           "aria-label": tip[0] },
        String(i + 1), el("span", { class: "g", "aria-hidden": "true", text: shortOutcome(o) })));
    } else if (i === current) {
      ul.append(el("li", { class: "chip current", title: `rollout ${i + 1}: running`,
                           "aria-label": `rollout ${i + 1} running`, text: `${i + 1}` }));
    } else {
      ul.append(el("li", { class: "chip pending", title: `rollout ${i + 1}: ${busy ? "to come" : "not run"}`,
                           text: `${i + 1}` }));
    }
  }
}

// a text glyph per outcome next to the colour (no emoji): ✓ ✕ ◷ ■ ! –
function shortOutcome(o) {
  return { success: "✓", fail: "✕", timeout: "◷", stopped: "■", fault: "!", crashed: "!",
           sim_error: "!", not_run: "-", interrupted: "\u25A0", home_failed: "!", start_pose_failed: "!",
           start_failed: "!", ended_by_policy_deploy: "■" }[o] || String(o);
}

function renderLog() {
  const s = state.status;
  const lines = s && Array.isArray(s.log_tail) ? s.log_tail : [];
  const text = lines.join("\n");
  if (text === state.shown.log) return;
  state.shown.log = text;
  const pre = $("log");
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 20;
  pre.textContent = text || "(no output yet)";
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

// After a run: k/N successes, and the link to its summary.html once index.csv has the row; then the runner's
// `evaltool score` line (Status.score) and a frames button (POST /api/frames -> evaltool frames).
function renderRunResult() {
  const box = $("run-result");
  const s = state.status;
  if (!s || (s.state !== "done" && s.state !== "failed") || !Array.isArray(s.outcomes) || !s.request) {
    box.classList.add("hidden");
    state.shown.runResult = "";
    return;
  }
  const k = s.outcomes.filter((o) => o === "success").length;
  const n = s.outcomes.length;
  const rel = (state.runEnd && state.runEnd.eval_rel) || null;
  const url = (state.runEnd && state.runEnd.summary_url) || (rel ? `/evals/${rel}/summary.html` : null)
    || (s.state === "done" ? newestSummaryFor(s.request.policy, s.request.tag) : null);
  const key = JSON.stringify([s.run_id, s.state, k, n, url, s.score, state.frames]);
  if (key === state.shown.runResult) return;       // rebuilt only on change (a button must survive a click)
  state.shown.runResult = key;
  const parts = n ? [el("strong", { text: `${k}/${n} success` }), ` · ${s.state}`]
    : [el("strong", { text: s.state === "failed" ? "failed before the first rollout" : "no rollouts" })];
  if (url) parts.push(" · ", el("a", { href: url, target: "_blank", rel: "noopener", text: "summary.html" }));
  if (s.score) parts.push(el("div", { class: "score-line" }, ...scoreParts(s.score)));
  box.replaceChildren(...parts);
  box.classList.remove("hidden");
}

function scoreParts(sc) {
  if (sc.state === "scoring") return [el("span", { class: "muted", text: "scoring (evaltool score)…" })];
  if (sc.state !== "done") return [el("span", { class: "inline-error", text: sc.error || "evaltool score failed" })];
  const kn = (v) => (Array.isArray(v) && v[1] ? `${v[0]}/${v[1]}` : "-");
  const hasZ = sc.min_z_m !== null && sc.min_z_m !== undefined;
  const out = [`score: success ${kn(sc.success)} · grasp ${kn(sc.grasp)} · max lift ${fmt.mm(sc.max_lift_m) || "-"}`
    + ` · min z ${fmt.mm(sc.min_z_m) || "-"}${hasZ ? ` (${sc.min_z_of})` : ""}`
    + (sc.invalid ? ` · ${sc.invalid} invalid` : "")];
  if (sc.eval_rel) {
    out.push(" · ", el("a", { href: `/evals/${encodeURI(sc.eval_rel)}/score.json`, target: "_blank", rel: "noopener",
      text: "score.json" }));
    const f = state.frames && state.frames.rel === sc.eval_rel ? state.frames : null;
    out.push(" · ", el("button", { type: "button", class: "small", disabled: !!(f && f.busy),
      text: f && f.busy ? "frames…" : "frames", onclick: () => makeFrames(sc.eval_rel) }));
    if (f && f.error) out.push(" ", el("span", { class: "inline-error", text: f.error }));
    if (f && f.urls) {
      if (!f.urls.length) out.push(" no video");
      for (const u of f.urls) {
        const m = u.match(/rollouts\/(\d+)\/frames\.jpg$/);
        out.push(" ", el("a", { href: u, target: "_blank", rel: "noopener", text: m ? `ep ${Number(m[1])}` : "frames" }));
      }
    }
  }
  return out;
}

async function makeFrames(rel) {
  state.frames = { rel, busy: true };
  renderRunResult();
  try {
    const r = await api("POST", "/api/frames", { eval_rel: rel });
    state.frames = r.ok ? { rel, urls: r.frames || [] } : { rel, error: `frames: ${r.error}` };
  } catch (e) {
    state.frames = { rel, error: `frames: ${e.message}` };
  }
  renderRunResult();
}

function newestSummaryFor(policy, tag) {
  for (let i = state.evals.length - 1; i >= 0; i--) {
    const r = state.evals[i];
    if (state.evalsBefore && state.evalsBefore.has(r.eval_dir)) break;   // older than this run
    if (rowPolicyId(r) === policy && (!tag || r.tag === tag) && r.summary_url) return r.summary_url;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------- results table
function rowPolicyId(r) {
  const p = String(r.policy || "");
  if (p === "hold_still" || p.startsWith("replay:")) return p;
  if (r.task && r.run_folder) return `${r.task}/${r.run_folder}`;
  return p;
}

function renderResults() {
  const only = $("results-only-selected").checked;
  const rows = state.evals.slice().reverse().filter((r) => !only || rowPolicyId(r) === state.selected);
  const key = JSON.stringify([only, state.selected, state.evals.length, state.evals[state.evals.length - 1]]);
  if (key === state.shown.results) return;
  state.shown.results = key;
  const body = $("results-body");
  body.replaceChildren();
  if (!rows.length) {
    body.append(el("tr", {}, el("td", { colspan: 13, class: "muted",
      text: state.evals.length ? "no results for the selected policy" : "no sim results yet" })));
    return;
  }
  for (const r of rows) {
    const rate = r.success_rate === "" || r.success_rate === undefined ? null : Number(r.success_rate);
    const ci = r.ci95_lo !== undefined && r.ci95_lo !== "" ? ` (${fmt.rate(r.ci95_lo)}–${fmt.rate(r.ci95_hi)})` : "";
    const bar = el("span", { class: "rate-bar", "aria-hidden": "true" });
    if (rate !== null) { const fill = el("span"); fill.style.width = `${Math.round(rate * 100)}%`; bar.append(fill); }
    const links = el("td");
    if (r.summary_url) links.append(el("a", { href: r.summary_url, target: "_blank", rel: "noopener", text: "summary" }));
    if (r.folder_url) {
      if (r.summary_url) links.append(" · ");
      links.append(el("a", { href: r.folder_url, target: "_blank", rel: "noopener", text: "folder" }));
    }
    const pid = rowPolicyId(r);
    const kind = r.kind || "";
    body.append(el("tr", { class: pid === state.selected ? "mine" : null, title: r.eval_dir || "" },
      el("td", { text: r.date || "" }),
      el("td", { class: "mono", text: pid }),
      el("td", {}, kind ? el("span", { class: `kind t-${typeOfKind(kind)}`, title: `${kindWords(kind, null).word} (${kind})`,
                                       text: kindWords(kind, null).short }) : ""),
      el("td", { text: r.dataset || "" }),
      el("td", { text: r.tag || "" }),
      el("td", { text: r.look || "" }),
      el("td", { class: "num", text: r.seed ?? "" }),
      el("td", { class: "num", text: `${r.successes ?? "?"}/${r.rollouts ?? "?"}` }),
      el("td", { class: "num" }, bar, rate === null ? "-" : `${fmt.rate(rate)}${ci}`),
      el("td", { class: "num", text: fmt.rate(r.grasp_rate) || "-" }),
      el("td", { class: "num", text: fmt.rate(r.lift_rate) || "-" }),
      el("td", { class: "num", text: `${r.faults ?? 0} / ${r.stops ?? 0} / ${r.timeouts ?? 0}` }),
      links));
  }
}

// --------------------------------------------------------------------------------------------------- keyboard
// Space = Play, or Stop while a run is going (a second Space = force). Not while typing in a field or on a
// radio/checkbox/select/summary or a fold button (aria-expanded: Space toggles it); on any other focused button the
// native click is suppressed so one press is one action.
function spaceIsOurs(target) {
  if (!target || target === document.body) return true;
  if (target.isContentEditable) return false;
  const tag = target.tagName;
  if (tag === "BUTTON" && target.hasAttribute("aria-expanded")) return false;   // a fold button: Space folds
  return !(tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "SUMMARY");
}

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.ctrlKey || e.altKey || e.metaKey) return;
  if (!spaceIsOurs(e.target)) return;
  e.preventDefault();
  if (e.repeat) return;
  if (isBusy() || state.expectStart) stop(); else play();
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && spaceIsOurs(e.target) && e.target && e.target.tagName === "BUTTON") e.preventDefault();
});

// ------------------------------------------------------------------------------------------------------ wiring
function onSettingsChange(e) {
  if (e && e.target && e.target.id === "set-starts") state.startsPref = e.target.value === "set" ? "set" : "random";
  applySettingsRules();
  save();
  renderSelected();
  renderStartPose();
  renderSteps();
}

function init() {
  applyHelp();
  restore();
  for (const id of SETTINGS) {
    $(id).addEventListener("input", onSettingsChange);
    $(id).addEventListener("change", onSettingsChange);
  }
  $("advanced").addEventListener("toggle", save);
  $("more").addEventListener("toggle", save);
  $("hide-help").addEventListener("change", () => { applyHideHelp(); save(); });
  $("btn-arm-reset").addEventListener("click", resetToArm);
  $("log-details").addEventListener("toggle", () => {
    const pre = $("log");
    if ($("log-details").open) pre.scrollTop = pre.scrollHeight;
  });
  applySettingsRules();
  $("btn-dice").addEventListener("click", () => {
    $("set-seed").value = String(Math.floor(Math.random() * 10000));
    onSettingsChange();
  });
  $("picker-filter").addEventListener("input", renderPicker);
  $("btn-reload-policies").addEventListener("click", loadPolicies);
  $("btn-reload-evals").addEventListener("click", loadEvals);
  $("results-only-selected").addEventListener("change", () => { save(); renderResults(); });
  $("btn-play").addEventListener("click", play);
  $("btn-stop").addEventListener("click", stop);
  setInterval(() => { tickChannels(); renderTelemetry(); }, TICK_MS);

  renderConnection();
  renderSelected();
  renderButtons();
  renderTelemetry();
  tickChannels();
  loadPolicies();
  loadEvals();
  loadStatus();
  loadStartPose();
  $("start-pose").addEventListener("click", loadStartPose);
  connect();
}

init();
