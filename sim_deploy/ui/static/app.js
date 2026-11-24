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
  shown: { log: "", chips: "", picker: "", results: "", runResult: "" },
  frames: null,               // {rel, busy, urls, error}: the frames button of this run's result row
};

const $ = (id) => document.getElementById(id);

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
    state.policiesLoaded = true;
    err.classList.add("hidden");
  } catch (e) {
    err.replaceChildren(`policies: ${e.message}`,
      el("button", { type: "button", class: "quiet", onclick: loadPolicies, text: "retry" }));
    err.classList.remove("hidden");
  }
  state.shown.picker = "";
  renderPicker();
  renderSelected();
  renderButtons();
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
}

function onEvent(ev) {
  const kind = ev.kind;
  if (ev.run_id && state.runId && String(ev.run_id) !== String(state.runId)) return;   // an older run
  if (kind === "rollout_end") {
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
for (const id of CHANNEL_IDS) {
  const canvas = $(`canvas-${id}`);
  channels[id] = {
    id, canvas, ctx: canvas.getContext("2d"), frame: canvas.parentElement, fps: $(`fps-${id}`),
    decoding: false, next: null, lastAt: 0, arrivals: [], ever: false,
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
    const r = sources ? sources[CHANNEL_SOURCE[ch.id]] : null;
    const why = r && !r.ok ? (r.reason || r.error || "") : "";
    ch.frame.querySelector(".nodata").textContent =
      (ch.ever && stale ? "no data (last frame shown)" : "no data") + (why ? `\n${why}` : "");
    ch.fps.textContent = stale ? "-" : `${(ch.arrivals.length / 2).toFixed(1)} fps`;
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
// grasp, behind "show all"). A filter shows matches of both.
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
      const name = p.label || runName(p.id);
      const meta = [];
      if (!runnable) meta.push(p.status === "not_fetched" || !p.checkpoint ? "not fetched" : "not runnable");
      if (p.label) meta.push(runName(p.id));
      if (p.episodes !== null && p.episodes !== undefined) meta.push(`${p.episodes} eps`);
      const mse = fmt.mse(p.val_mse);
      if (mse) meta.push(`mse ${mse}`);
      const last = lastEvalText(p.last_eval);
      if (last) meta.push(last);
      const tip = [p.label ? `${p.label}\n${p.id}` : p.id, p.dataset ? `dataset: ${p.dataset}` : null, p.train_config ? `config: ${p.train_config}` : null,
                   p.checkpoint ? `checkpoint: ${p.checkpoint}` : null,
                   !runnable ? (p.not_runnable_reason || "no checkpoint here: fetch it first (train/fetch.sh)") : null,
                   p.note ? `note: ${p.note}` : null,
                   p.last_eval ? `last sim: ${last} (${p.last_eval.date || ""} ${p.last_eval.tag || ""})` : null]
        .filter(Boolean).join("\n");
      const notFetched = !runnable && (p.status === "not_fetched" || !p.checkpoint);
      out.push({ group: task, id: p.id, name, kind, runnable, tip, meta: meta.join(" · "),
                 badge: runnable ? kind : (notFetched ? "not fetched" : "not runnable"),
                 badgeClass: runnable ? `kind-${kind}` : "kind-nofetch", hide: p.smoke ? "smoke" : null,
                 search: `${p.id} ${name} ${kind} ${p.dataset || ""}`.toLowerCase() });
    }
  }
  for (const b of state.builtins) {
    const last = b === "hold_still" ? lastEvalText(state.builtinLastEval) : null;
    out.push({ group: "built-in", id: b, name: b, badge: "baseline", badgeClass: "kind-builtin",
               meta: [b === "hold_still" ? "no policy: the arm holds still" : "", last].filter(Boolean).join(" · "),
               tip: b, runnable: true, hide: null, search: b.toLowerCase(), kind: "builtin" });
  }
  const replays = state.replays.slice().sort((a, b) =>
    String(a.group || a.dataset || "").localeCompare(String(b.group || b.dataset || ""))
    || (!!b.grasp_ok - !!a.grasp_ok) || ((a.index ?? 0) - (b.index ?? 0)));
  for (const r of replays) {
    const ok = !!r.grasp_ok;
    const err = typeof r.height_error_m === "number" ? `height error ${fmt.mm(r.height_error_m)}` : null;
    const last = lastEvalText(r.last_eval);
    out.push({ group: `replay · ${r.group || r.dataset || "?"}`, id: r.id, name: r.name || `#${r.index}`,
               badge: ok ? "grasp ok" : "grasp ✗", badgeClass: ok ? "kind-grasp-ok" : "kind-grasp-bad",
               meta: [`#${r.index}`, err, last].filter(Boolean).join(" · "), runnable: true, kind: "replay",
               tip: [r.id, ok ? "the replay placement grasps (|height error| < 2 cm)" : "the replay placement misses the grasp",
                     err].filter(Boolean).join("\n"),
               hide: ok ? null : "nograsp", search: `${r.id} ${r.name || ""} replay ${r.dataset || ""}`.toLowerCase() });
  }
  return out;
}

function findEntry(id) {
  return entries().find((e) => e.id === id) || null;
}

function toggleButton(text, expanded, onclick) {
  return el("button", { type: "button", class: "fold", "aria-expanded": expanded ? "true" : "false", onclick,
                        text: `${expanded ? "▾" : "▸"} ${text}` });
}

function pickerRow(e) {
  const input = el("input", { type: "radio", name: "policy", value: e.id, disabled: !e.runnable,
                              checked: e.id === state.selected });
  input.addEventListener("change", () => select(e.id));
  return el("label", { class: `prow${e.runnable ? "" : " disabled"}${e.id === state.selected ? " selected" : ""}`,
                       title: e.tip },
    input,
    el("span", { class: "name", text: e.name }),
    el("span", { class: `kind ${e.badgeClass}`, text: e.badge }),
    el("span", { class: "meta", text: e.meta || " " }));
}

function renderPicker() {
  const box = $("picker");
  const filter = $("picker-filter").value.trim().toLowerCase();
  const all = entries();
  const key = JSON.stringify([filter, state.selected, [...state.openSmoke], state.showAllReplays,
                              all.map((e) => [e.id, e.meta, e.badge])]);
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
  for (const [group, items] of groups) {
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
  renderSelected();
  renderButtons();
  if ($("results-only-selected").checked) { state.shown.results = ""; }
  renderResults();
}

// --------------------------------------------------------------------------------------------------- settings
const SETTINGS = ["set-rollouts", "set-seed", "set-home", "set-look", "set-timeout", "set-tap", "set-precision",
                  "set-steps", "set-exech", "set-tag"];
// 2026-11-21: the defaults became look robot and the arm's bf16 / 8 steps / exec_h 12. Settings saved before that
// (no settingsVersion 2) do not restore these, so an old "station default" / empty field does not hide them.
const SETTINGS_V2_ONLY = ["set-look", "set-precision", "set-steps", "set-exech"];

function save() {
  try {
    const s = { settingsVersion: 2, selected: state.selected, advanced: $("advanced").open,
                showAllReplays: state.showAllReplays,
                openSmoke: [...state.openSmoke],
                onlySelected: $("results-only-selected").checked };
    for (const id of SETTINGS) { const n = $(id); s[id] = n.type === "checkbox" ? n.checked : n.value; }
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
    $("results-only-selected").checked = !!s.onlySelected;
    for (const id of SETTINGS) {
      if (!(id in s)) continue;
      if (s.settingsVersion !== 2 && SETTINGS_V2_ONLY.includes(id)) continue;
      const n = $(id);
      if (n.type === "checkbox") n.checked = !!s[id]; else n.value = s[id];
    }
    // P2: an old saved "trainstart" (or anything not offered any more) means the start pose
    if (![...$("set-home").options].some((o) => o.value === $("set-home").value)) $("set-home").value = "start";
    if (![...$("set-look").options].some((o) => o.value === $("set-look").value)) $("set-look").value = "robot";
    if (![...$("set-precision").options].some((o) => o.value === $("set-precision").value)) $("set-precision").value = "bf16";
  } catch (e) { /* ignore */ }
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

// The RunRequest (DESIGN.md section 2), or {error}.
function buildRequest() {
  if (!state.selected) return { error: "pick a policy first" };
  const entry = findEntry(state.selected);
  if (state.policiesLoaded && !entry) return { error: "the selected policy is not in the list any more" };
  if (entry && !entry.runnable) return { error: "this run has no checkpoint here (not fetched)" };
  const rollouts = intOrNull("set-rollouts", 1, 100, "rollouts");
  const seed = intOrNull("set-seed", 0, undefined, "seed");
  const steps = intOrNull("set-steps", 1, undefined, "diffusion steps");
  const exech = intOrNull("set-exech", 1, 16, "exec_h");
  for (const r of [rollouts, seed, steps, exech]) if (r.error) return { error: r.error };
  const timeout = Number($("set-timeout").value);
  if (!Number.isFinite(timeout) || timeout <= 0) return { error: "timeout: seconds > 0" };
  const tag = $("set-tag").value.trim() || "ui";
  if (!/^[A-Za-z0-9_.-]+$/.test(tag)) return { error: "tag: letters, digits, _ . - only" };
  return {
    request: {
      policy: state.selected,
      rollouts: rollouts.value === null ? 1 : rollouts.value,
      seed: seed.value === null ? 0 : seed.value,
      home: $("set-home").value,
      look: $("set-look").value,
      timeout_s: timeout,
      tap: $("set-tap").checked,
      tag,
      precision: $("set-precision").value || "bf16",
      diffusion_steps: steps.value === null ? 8 : steps.value,
      exec_h: exech.value === null ? 12 : exech.value,
    },
  };
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

function renderStartPose() {
  const box = $("start-pose");
  const sp = state.startPose;
  if ($("set-home").value === "deploy") {
    box.textContent = "deploy home: fingertips straight down, ~30 cm above the table (sim PC with no start pose)";
    box.classList.remove("bad");
    return;
  }
  if (!sp) { box.textContent = "start pose: loading… (the first load takes a few seconds)"; return; }
  if (sp.error) { box.textContent = `start pose: ${sp.error}`; box.classList.add("bad"); return; }
  if (!sp.set) { box.textContent = sp.text || "start pose: not set"; box.classList.add("bad"); return; }
  const bad = Object.entries(sp.checks || {}).filter(([, c]) => !c.ok).map(([k]) => k);
  const git = sp.vs_git || {};
  box.textContent = `start pose: ${sp.text}` + (bad.length ? ` · FAILS ${bad.join(", ")}` : "")
    + (sp.region && !sp.region.ok ? ` · eraser region: ${sp.region.detail}` : "")
    + (git.same === true ? ` · = the real arm's (${git.ref || "upstream"})` : git.text ? ` · ${git.text}` : "");
  box.title = `${sp.file}\nsource: ${sp.source || "?"}\nq_start_rad: [${(sp.q_start_rad || []).map((v) => v.toFixed(4)).join(", ")}]`
    + Object.entries(sp.vs_demos || {}).map(([k, v]) => `\n${k}: ${v.text}`).join("");
  box.classList.toggle("bad", bad.length > 0 || git.same === false);
}

function renderSelected() {
  const box = $("selected");
  const entry = state.selected ? findEntry(state.selected) : null;
  if (!state.selected) { box.textContent = "nothing selected: pick a policy above"; return; }
  const b = buildRequest();
  const r = b.request;
  const line = r ? `${r.rollouts} rollouts · seed ${r.seed} · home ${r.home} · look ${r.look} · ${r.timeout_s} s${r.tap ? " · tap" : ""}`
    + (r.precision ? ` · ${r.precision}` : "") + (r.diffusion_steps ? ` · ${r.diffusion_steps} steps` : "")
    + (r.exec_h ? ` · exec_h ${r.exec_h}` : "") + ` · tag ${r.tag}` : b.error;
  box.title = state.selected;
  box.replaceChildren(el("strong", { text: entry ? entry.name : state.selected }), el("br"), el("span", { text: line }));
}

// ---------------------------------------------------------------------------------------------- play and stop
function canPlay() {
  return state.connected && !state.inFlight && !isBusy() && !state.expectStart && !!state.selected;
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

function renderButtons() {
  const playBtn = $("btn-play"), stopBtn = $("btn-stop");
  playBtn.disabled = !canPlay();
  playBtn.title = !state.connected ? "the UI server is not connected"
    : isBusy() ? "a run is going: stop it first"
    : !state.selected ? "pick a policy first" : "start the run (Space)";
  const busy = isBusy() || !!state.expectStart;
  stopBtn.disabled = !busy;
  const stopping = state.stopPresses >= 1 || (state.status && state.status.state === "stopping");
  if (state.stopPresses >= 2) {
    stopBtn.textContent = "killing…";
  } else if (stopping) {
    stopBtn.replaceChildren(el("span", { class: "sub", text: "stopping\u2026" }), "\u25A0 Force");
  } else {
    stopBtn.textContent = "■ Stop";
  }
  stopBtn.classList.toggle("force", stopping);
  stopBtn.title = stopping ? "press again to kill the run now" : "stop after the current step (Space)";
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

  const what = $("running-what");
  const req = s && s.request;
  if (req && (busy || st === "done" || st === "failed")) {
    what.textContent = `${req.policy} · ${req.rollouts} rollouts · seed ${req.seed}${req.tag ? ` · ${req.tag}` : ""}`;
    what.title = JSON.stringify(req, null, 1);
  } else { what.textContent = ""; what.title = ""; }

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

  renderChips();
  renderLog();
  renderRunResult();
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
  if (!N) { ul.append(el("li", { class: "muted", text: "no run yet" })); return; }
  for (let i = 0; i < N; i++) {
    const o = outcomes[i];
    if (o !== undefined) {
      const rec = state.rolloutRecords.get(i);
      const tip = [`rollout ${i + 1}: ${o}`];
      if (rec) {
        if (rec.grasped !== undefined) tip.push(`grasped: ${rec.grasped ? "yes" : "no"}`);
        if (rec.lifted !== undefined) tip.push(`lifted: ${rec.lifted ? "yes" : "no"}`);
        if (rec.max_lift_m !== undefined && rec.max_lift_m !== null) tip.push(`max lift: ${fmt.mm(rec.max_lift_m)}`);
        if (rec.note) tip.push(`note: ${rec.note}`);
        if (rec.fault) tip.push(`fault: ${rec.fault}`);
      }
      ul.append(el("li", { class: `chip o-${String(o).replace(/[^a-z_]/gi, "")}`, title: tip.join("\n"),
                           "aria-label": tip[0], text: `${i + 1} ${shortOutcome(o)}` }));
    } else if (busy && i === outcomes.length) {
      ul.append(el("li", { class: "chip current", title: `rollout ${i + 1}: running`,
                           "aria-label": `rollout ${i + 1} running`, text: `${i + 1} …` }));
    } else {
      ul.append(el("li", { class: "chip pending", title: `rollout ${i + 1}: ${busy ? "to come" : "not run"}`,
                           text: `${i + 1}` }));
    }
  }
}

function shortOutcome(o) {
  return { success: "✓", fail: "✗", timeout: "⌛", stopped: "■", fault: "!", crashed: "!",
           sim_error: "!", not_run: "-", interrupted: "\u25A0" }[o] || String(o);
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
      el("td", { class: "mono", text: r.date || "" }),
      el("td", { class: "mono", text: pid }),
      el("td", {}, kind ? el("span", { class: `kind kind-${kind}`, text: kind }) : ""),
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
// radio/checkbox/select (Space belongs to them there); on a focused button the native click is suppressed so one
// press is one action.
function spaceIsOurs(target) {
  if (!target || target === document.body) return true;
  if (target.isContentEditable) return false;
  const tag = target.tagName;
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
function onSettingsChange() { save(); renderSelected(); renderStartPose(); }

function init() {
  restore();
  for (const id of SETTINGS) {
    $(id).addEventListener("input", onSettingsChange);
    $(id).addEventListener("change", onSettingsChange);
  }
  $("advanced").addEventListener("toggle", save);
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
