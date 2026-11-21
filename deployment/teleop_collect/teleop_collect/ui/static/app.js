// teleop_collect ui: the operator's page (STRUCTURE.md 9.2, 9.3; PLAN.md 4.3, 5.4). Plain JS, no build, offline.
//
// It shows the status, the previews and the episodes, sends the buttons as commands (each answered by an ack) and
// sends the key state: on every change and every keys_resend_ms while the page has focus. Motion and gripper keys
// count only while the page has focus and the phase is "recording"; losing focus sends an empty held set with
// focus false. The server releases the keys by itself when the page goes silent (the deadman).
"use strict";

// ------------------------------------------------------------------------------------------------ the keyboard map
// STRUCTURE.md 4.1. KeyboardEvent.code (the physical key), so the map does not depend on the keyboard layout.
const MOTION_KEY_MAP = Object.freeze({
  KeyW: "x+", KeyS: "x-",
  KeyA: "y+", KeyD: "y-",
  KeyR: "z+", KeyF: "z-",
  KeyU: "twist+", KeyO: "twist-",
  KeyI: "tilt_fwd", KeyK: "tilt_back",
  KeyJ: "tilt_left", KeyL: "tilt_right",
});
const GRIPPER_KEY_MAP = Object.freeze({ KeyN: "close", KeyM: "open" });
const SLOW_KEYS = Object.freeze(["ShiftLeft", "ShiftRight"]);   // Shift held: slow
const STOP_ARM_KEY = "Escape";                                  // Esc: stop_arm
// The order held keys are sent in (teleop's MOTION_KEYS).
const MOTION_KEY_ORDER = ["x+", "x-", "y+", "y-", "z+", "z-", "twist+", "twist-",
                          "tilt_fwd", "tilt_back", "tilt_left", "tilt_right"];

// Display only: how the held motion names read in the page (the names sent stay as above).
const MOTION_KEY_WORDS = Object.freeze({
  "x+": "forward", "x-": "back", "y+": "left", "y-": "right", "z+": "up", "z-": "down",
  "twist+": "twist", "twist-": "twist back",
  tilt_fwd: "tilt forward", tilt_back: "tilt back", tilt_left: "tilt left", tilt_right: "tilt right",
});

const RECONNECT_MS = 1000;
const MESSAGES_SHOWN = 30;

// ------------------------------------------------------------------------------------------------------- state
const state = {
  ws: null,
  connected: false,
  hello: { keys_resend_ms: 50 },
  status: null,
  held: new Set(),           // motion key names held right now (only while keys are enabled)
  shift: false,
  choosing: false,           // Stop episode pressed: Save success / Save fail / Discard shown, keys off
  lastKeys: "",              // the last key state sent (JSON)
  resendTimer: null,
  nextId: 1,
  pending: new Map(),        // command id -> its name
  previewHeader: null,
  previewUrls: {},
  tasksShown: "",
  episodesShown: "",
  messages: new Map(),       // id -> message
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------------------------- the socket
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "blob";
  state.ws = ws;
  ws.onopen = () => {
    state.connected = true;
    state.lastKeys = "";
    sendKeys(true);
    render();
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") onText(event.data);
    else onBinary(event.data);
  };
  ws.onclose = () => {
    state.connected = false;
    state.ws = null;
    state.held.clear();
    state.pending.clear();
    if (state.resendTimer !== null) { clearInterval(state.resendTimer); state.resendTimer = null; }
    render();
    setTimeout(connect, RECONNECT_MS);
  };
}

function send(message) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(message));
    return true;
  }
  return false;
}

function onText(text) {
  const msg = JSON.parse(text);
  switch (msg.type) {
    case "hello":
      state.hello = msg;
      if (state.resendTimer !== null) clearInterval(state.resendTimer);
      state.resendTimer = setInterval(() => { if (pageFocused()) sendKeys(true); }, msg.keys_resend_ms);
      break;
    case "status":
      state.status = msg;
      for (const m of msg.messages || []) state.messages.set(m.id, m);
      sendKeys(false);          // the phase may have changed: keys on or off
      render();
      break;
    case "preview":
      state.previewHeader = msg;
      break;
    case "ack":
      onAck(msg);
      break;
    case "error":
      console.warn("ui server:", msg.text);
      break;
  }
}

function onBinary(blob) {
  const header = state.previewHeader;
  state.previewHeader = null;
  if (!header) return;
  const img = $(`preview-${header.camera}`);
  if (!img) return;
  const url = URL.createObjectURL(new Blob([blob], { type: "image/jpeg" }));
  const old = state.previewUrls[header.camera];
  state.previewUrls[header.camera] = url;
  img.src = url;
  if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
}

// --------------------------------------------------------------------------------------------------- commands
function command(name, args = {}) {
  const id = state.nextId++;
  if (!send({ type: "command", name, args, id })) {
    showAction(`${name}: not sent (disconnected)`, true);
    return;
  }
  state.pending.set(id, name);
  showAction(`${name} ...`, false);
}

function onAck(msg) {
  const name = state.pending.get(msg.id) || "command";
  state.pending.delete(msg.id);
  if (msg.ok) {
    showAction(`${name}: ok`, false);
    if (name === "create_task") hideNewTask();
  } else {
    showAction(`${name} refused: ${msg.reason}`, true);
  }
}

function showAction(text, refused) {
  const el = $("last-action");
  el.textContent = text;
  el.className = refused ? "hint refused" : "muted";
}

// -------------------------------------------------------------------------------------------------------- keys
function pageFocused() {
  return document.hasFocus() && document.visibilityState === "visible";
}

function teleopUp() {
  return state.connected && state.status !== null && state.status.teleop_answering === true;
}

function keysEnabled() {
  return pageFocused() && teleopUp() && state.status.phase === "recording" && !state.choosing;
}

// Send the key state: on a change, or always with force (the resend). Keys not enabled = an empty held set.
function sendKeys(force) {
  if (!keysEnabled()) { state.held.clear(); state.shift = false; }
  const held = MOTION_KEY_ORDER.filter((k) => state.held.has(k));
  const msg = { type: "keys", held, slow: keysEnabled() && state.shift, focus: pageFocused() };
  const text = JSON.stringify(msg);
  if (!force && text === state.lastKeys) return;
  if (send(msg)) state.lastKeys = text;
  renderKeys(msg);
}

// Losing the keys (blur, hidden tab): an empty held set with focus false, sent at once.
function loseKeys() {
  state.held.clear();
  state.shift = false;
  const msg = { type: "keys", held: [], slow: false, focus: false };
  if (send(msg)) state.lastKeys = JSON.stringify(msg);
  renderKeys(msg);
}

function isTyping(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ||
         target instanceof HTMLSelectElement;
}

function onKeyDown(e) {
  if (e.code === STOP_ARM_KEY) {                  // Esc always stops the arm, even while typing
    e.preventDefault();
    if (!e.repeat) stopArm();
    return;
  }
  if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
  state.shift = e.shiftKey;
  if (SLOW_KEYS.includes(e.code)) { sendKeys(false); return; }
  const motion = MOTION_KEY_MAP[e.code];
  if (motion !== undefined) {
    e.preventDefault();
    if (e.repeat) return;                         // auto-repeat: the key is already held
    if (keysEnabled()) state.held.add(motion);
    sendKeys(false);
    return;
  }
  const grip = GRIPPER_KEY_MAP[e.code];
  if (grip !== undefined) {
    e.preventDefault();
    if (e.repeat) return;                         // key down only, no repeat
    if (keysEnabled()) send({ type: "gripper", command: grip });
  }
}

function onKeyUp(e) {
  state.shift = e.shiftKey;
  const motion = MOTION_KEY_MAP[e.code];
  if (motion !== undefined) state.held.delete(motion);
  sendKeys(false);
}

function stopArm() {
  state.choosing = false;
  state.held.clear();
  sendKeys(false);
  command("stop_arm");
  render();
}

// ------------------------------------------------------------------------------------------------------ render
function fmt(x, digits) { return (x === null || x === undefined) ? "-" : Number(x).toFixed(digits); }

function mmss(seconds) {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function camText(h) {
  return h ? `${h.status} ${Math.round(h.fps)}` : "-";
}

function renderKeys(msg) {
  const el = $("keys-state");
  if (!el) return;
  if (!msg.focus) { el.textContent = "keys off (page not focused)"; return; }
  if (!keysEnabled()) { el.textContent = "keys off (not recording)"; return; }
  const words = msg.held.map((k) => MOTION_KEY_WORDS[k] || k);
  el.textContent = words.length ? `held: ${words.join(", ")}${msg.slow ? " (slow)" : ""}` : "keys live";
}

function renderBanner(s) {
  const banner = $("banner");
  let text = "";
  let warn = false;
  if (!state.connected) text = "disconnected: the ui is not running (teleop_collect stopped?)";
  else if (s && s.phase === "done") text = `stopped: ${s.final_line || "teleop ended"}`;
  else if (!s || !s.teleop_answering) text = (s && s.teleop_error) || "teleop not answering";
  else if (!s.frames_answering) { text = s.frames_error || "frames not answering"; warn = true; }
  banner.textContent = text;
  banner.classList.toggle("hidden", text === "");
  banner.classList.toggle("warn", warn);
}

function renderTasks(s) {
  const tasks = (s && s.tasks) || [];
  const key = JSON.stringify([tasks, s ? s.task : ""]);
  const select = $("task-select");
  if (key !== state.tasksShown && document.activeElement !== select) {
    state.tasksShown = key;
    select.replaceChildren();
    const none = new Option(tasks.length ? "pick a task" : "no task yet: + new task", "");
    none.disabled = true;
    select.add(none);
    for (const t of tasks) select.add(new Option(t, t));
    select.value = (s && s.task) || "";
  }
  $("task-description").textContent = (s && s.task_description) || "";
}

function renderEpisodes(s) {
  const episodes = (s && s.episodes) || [];
  const recording = s && s.phase === "recording";
  const key = JSON.stringify([episodes, s ? s.task : "", recording]);
  if (key === state.episodesShown) return;
  state.episodesShown = key;
  const list = $("episodes");
  list.replaceChildren();
  if (!episodes.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = s && s.task ? "none yet" : "pick a task";
    list.append(li);
    return;
  }
  for (const ep of episodes) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = ep.name;
    const outcome = document.createElement("span");
    outcome.className = `outcome-${ep.outcome}`;
    outcome.textContent = ep.outcome;
    const dur = document.createElement("span");
    dur.textContent = `${ep.duration_s.toFixed(1)} s`;
    const frames = document.createElement("span");
    frames.className = "muted";
    frames.textContent = `${ep.frames_main} frames`;
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "delete";
    del.disabled = recording;
    del.onclick = () => {
      if (confirm(`Delete ${ep.name} of ${s.task} (${ep.outcome}, ${ep.duration_s.toFixed(1)} s)? ` +
                  "Its folder is removed from disk.")) {
        command("delete_episode", { episode: ep.name });
      }
    };
    li.append(name, outcome, dur, frames);
    if (ep.outcome === "unsaved") {             // ended without a choice: label it here (or delete it)
      for (const [label, value] of [["save success", "success"], ["save fail", "fail"]]) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.className = `label-${value}`;
        b.disabled = recording;
        b.onclick = () => command("label_episode", { episode: ep.name, outcome: value });
        li.append(b);
      }
    }
    li.append(del);
    list.append(li);
  }
}

function renderMessages() {
  const ids = [...state.messages.keys()].sort((a, b) => b - a);
  for (const id of ids.slice(MESSAGES_SHOWN)) state.messages.delete(id);
  const list = $("messages");
  list.replaceChildren();
  for (const id of ids.slice(0, MESSAGES_SHOWN)) {
    const m = state.messages.get(id);
    const li = document.createElement("li");
    li.className = m.level;
    li.textContent = m.text;
    list.append(li);
  }
}

function render() {
  const s = state.status;
  const up = teleopUp();
  const has = up && s.phase !== undefined;
  renderBanner(s);
  const link = $("link");
  link.textContent = state.connected ? (up ? "connected" : "teleop not answering") : "disconnected";
  link.className = `badge ${state.connected && up ? "ok" : "bad"}`;
  $("mock-badge").classList.toggle("hidden", !(s && s.pc_mock));
  renderTasks(s);
  renderEpisodes(s);
  renderMessages();

  const recording = has && s.phase === "recording";
  if (!recording) state.choosing = false;
  $("rec").classList.toggle("hidden", !recording);
  if (recording) $("rec").textContent = `REC ${s.recording_episode} ${mmss(s.episode_s)}`;
  $("phase").textContent = has ? s.phase + (s.ready ? "" : " (starting up)") : "-";
  $("arm").textContent = has ? (s.arm_mode || "-") + (s.moving ? " · moving" : "") : "-";
  $("tcp").textContent = has && s.tcp_m ? `${s.tcp_m.map((v) => v.toFixed(3)).join(" ")} m` : "-";
  $("grip").textContent = has
    ? `${fmt(s.gripper_width_m, 3)} m · cmd ${s.commanded_width_m > 0 ? "open" : "close"}` : "-";
  const cams = (s && s.preview_health) || (has ? s.cameras : null);
  $("cams").textContent = cams ? `${camText(cams.main)} / ${camText(cams.ultrawide)} fps` : "-";
  $("cam-main").textContent = cams ? camText(cams.main) : "-";
  $("cam-ultrawide").textContent = cams ? camText(cams.ultrawide) : "-";
  $("c2").textContent = has ? `${fmt(s.c2_hz, 0)} Hz` : "-";
  $("rtt").textContent = has && s.rtt_ms !== null ? `${fmt(s.rtt_ms, 1)} ms` : "-";
  $("clock").textContent = has ? (s.clock_ok ? "ok" : "not ready") : "-";
  $("speed").textContent = has ? (s.slow ? "slow" : "normal") : "-";
  $("limit").classList.toggle("hidden", !(has && s.at_limit));

  const hint = $("hint");
  if (!has) { hint.textContent = ""; hint.className = "hint"; }
  else if (recording) {
    hint.textContent = state.choosing ? "motion off: save or discard the episode" : "recording: hold keys to move";
    hint.className = "hint";
  } else if (s.start_allowed) {
    hint.textContent = s.start_pose_confirmed ? "at the start pose ✓ ready to start"
      : "at the start-up pose ✓ ready to start (the PC checks it at Start)";
    hint.className = "hint ok";
  }
  else { hint.textContent = s.start_refusal; hint.className = "hint refused"; }

  const busy = !has || s.phase === "stopping" || s.phase === "done" || !s.ready;
  $("btn-start").disabled = !has || !s.start_allowed || recording;
  $("btn-start").classList.toggle("hidden", recording);
  $("btn-stop-episode").classList.toggle("hidden", !recording || state.choosing);
  $("save-choice").classList.toggle("hidden", !recording || !state.choosing);
  $("btn-stop-arm").disabled = !state.connected;
  $("task-select").disabled = busy || recording;
  $("new-task-toggle").disabled = busy || recording;
  renderKeys({ held: [...state.held], slow: state.shift, focus: pageFocused() });
}

// ------------------------------------------------------------------------------------------------- the buttons
function hideNewTask() {
  $("new-task-form").classList.add("hidden");
  $("new-task-toggle").classList.remove("hidden");
}

function setup() {
  $("btn-start").onclick = () => command("start_episode");
  $("btn-stop-episode").onclick = () => { state.choosing = true; sendKeys(false); render(); };
  $("btn-keep").onclick = () => { state.choosing = false; render(); };
  const stopAs = (outcome) => () => {
    state.choosing = false;
    command("stop_episode", { outcome });
    render();
  };
  $("btn-save-success").onclick = stopAs("success");
  $("btn-save-fail").onclick = stopAs("fail");
  $("btn-discard").onclick = stopAs("discard");
  $("btn-stop-arm").onclick = stopArm;
  $("task-select").onchange = (e) => { if (e.target.value) command("select_task", { name: e.target.value }); };
  $("new-task-toggle").onclick = () => {
    $("new-task-form").classList.remove("hidden");
    $("new-task-toggle").classList.add("hidden");
    $("new-task-name").focus();
  };
  $("new-task-cancel").onclick = hideNewTask;
  $("new-task-form").onsubmit = (e) => {
    e.preventDefault();
    command("create_task", { name: $("new-task-name").value.trim(),
                             description: $("new-task-description").value.trim() });
  };
  // Buttons must not keep the keyboard focus (Space / Enter would press them again): give it back to the page.
  for (const b of document.querySelectorAll("button")) b.addEventListener("mouseup", () => b.blur());

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", loseKeys);
  window.addEventListener("focus", () => sendKeys(true));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") loseKeys(); else sendKeys(true);
  });
  render();
  connect();
}

setup();
