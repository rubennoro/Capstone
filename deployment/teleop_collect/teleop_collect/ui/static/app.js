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
  "twist+": "twist ↺", "twist-": "twist ↻",
  tilt_fwd: "tilt forward", tilt_back: "tilt back", tilt_left: "tilt left", tilt_right: "tilt right",
});

// Display only: how the commands read in the line under Stop arm (the names sent stay as they are).
const COMMAND_WORDS = Object.freeze({
  start_episode: "Start episode", stop_episode: "Stop episode", stop_arm: "Stop arm", select_task: "Select task",
  create_task: "Create task", delete_task: "Delete task", delete_episode: "Delete episode",
  label_episode: "Save episode", set_data_root: "Change storage",
});

const THEME_KEY = "teleop_collect.theme";           // localStorage; theme.js reads it before the first paint
const RECONNECT_MS = 1000;
const REFUSAL_SHOWN_MS = 6000;
const MESSAGES_SHOWN = 30;
const CAMERA_WORDS = Object.freeze({ main: "Wrist camera", ultrawide: "Wide view" });
const TRASH_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.1' +
  'a1 1 0 0 0 1 .9h3.8a1 1 0 0 0 1-.9l.6-8.1" fill="none" stroke="currentColor" stroke-width="1.3" ' +
  'stroke-linecap="round" stroke-linejoin="round"/></svg>';

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
  // display only
  focusCam: null,            // the enlarged camera ("main", "ultrawide") or null: both side by side
  confirmEpisode: null,      // the episode whose inline "Delete ...? Yes / No" is open
  deleteTaskName: null,      // the task the inline delete-task confirmation is about
  logSeen: 0,                // the newest message id seen with the activity log open
  messagesShown: "",
  confirmDiscard: false,     // the save choice's "Discard episode_NNN?" is open
  wasRecording: false,       // the phase at the last render was recording (forms close when recording starts)
  actionTimer: null,         // clears the refusal line
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
    showAction(`${commandWord(name)}: not sent (disconnected)`, true);
    return;
  }
  state.pending.set(id, name);
  showAction("", false);                          // a new command clears an old refusal
}

function onAck(msg) {
  const name = state.pending.get(msg.id) || "command";
  state.pending.delete(msg.id);
  if (msg.ok) {                                    // no message: the new state is the feedback
    if (name === "create_task") hideNewTask();
    if (name === "set_data_root") hideStorageForm();
    if (name === "delete_task") hideDeleteTask();
  } else {
    // A form's refusal shows inside the form, where the operator is looking; the others in the state card.
    const errorBox = { create_task: "new-task-error", set_data_root: "storage-error",
                       delete_task: "delete-task-error" }[name];
    if (errorBox && !$(errorBox).closest(".hidden")) $(errorBox).textContent = cap(msg.reason);
    else showAction(`${commandWord(name)} refused: ${msg.reason}`, true);
  }
}

// A refusal, shown in the state card for REFUSAL_SHOWN_MS; anything else clears it.
function showAction(text, refused) {
  const el = $("last-action");
  if (state.actionTimer !== null) { clearTimeout(state.actionTimer); state.actionTimer = null; }
  el.textContent = refused ? text : "";
  if (!refused) return;
  state.actionTimer = setTimeout(() => { el.textContent = ""; state.actionTimer = null; }, REFUSAL_SHOWN_MS);
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
  closeForms();
  render();
}

// ------------------------------------------------------------------------------------------------------ render
function fmt(x, digits) { return (x === null || x === undefined) ? "-" : Number(x).toFixed(digits); }

function mmss(seconds) {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// The first letter upper case (teleop's texts are lower case sentences; the text itself is kept).
function cap(text) { return text ? text.charAt(0).toUpperCase() + text.slice(1) : ""; }

function commandWord(name) { return COMMAND_WORDS[name] || name; }

function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

// A long path cut in the middle (the full path is in the title).
function shortPath(path) {
  return path.length <= 40 ? path : `${path.slice(0, 16)}…${path.slice(-23)}`;
}

function setDot(el, level, title) {
  el.querySelector(".dot").className = `dot ${level}`;
  if (title === undefined) return;
  el.title = title;
  const hidden = el.querySelector(".sr-only");        // the same details for screen readers
  if (hidden) hidden.textContent = `: ${title.replace(/^[A-Za-z]+: /, "")}`;
}

function keysOffReason() {
  if (!state.connected) return "Not connected";
  if (!teleopUp()) return "Teleop not answering";
  if (state.status.phase !== "recording") return "Start an episode to drive";
  if (state.choosing) return "Choose save or discard first";
  if (!pageFocused()) return "Click the page to give it the keyboard";
  if (isTyping(document.activeElement)) return "Typing in a field · click the page to drive";
  return "Keys off";
}

function renderKeys(msg) {
  const el = $("keys-state");
  if (!el) return;
  // Keys typed into a field never reach the arm (onKeyDown), so the map shows them off then too.
  const live = msg.focus && keysEnabled() && !isTyping(document.activeElement);
  const words = msg.held.map((k) => MOTION_KEY_WORDS[k] || k);
  let text = keysOffReason();
  if (live) text = words.length ? `${cap(words.join(" + "))}${msg.slow ? " · slow" : ""}` : "Keys live";
  $("keys-state-text").textContent = text;
  el.className = `keys-state${live ? " live" : ""}`;
  el.querySelector(".dot").className = `dot ${live ? "ok" : ""}`;
  $("controls").classList.toggle("off", !live);
  for (const capEl of document.querySelectorAll(".cap[data-key]")) {
    const key = capEl.dataset.key;
    capEl.classList.toggle("held", live && (key === "slow" ? msg.slow : msg.held.includes(key)));
  }
}

function renderBanner(s) {
  const banner = $("banner");
  let text = "";
  let warn = false;
  if (!state.connected) text = "disconnected: the ui is not running (teleop_collect stopped?) · retrying";
  else if (s && s.phase === "done") text = `stopped: ${s.final_line || "teleop ended"}`;
  else if (!s || !s.teleop_answering) text = (s && s.teleop_error) || "teleop not answering";
  else if (!s.frames_answering) { text = s.frames_error || "frames not answering"; warn = true; }
  banner.textContent = cap(text);
  banner.classList.toggle("hidden", text === "");
  banner.classList.toggle("warn", warn);
}

function cameraHealth(s, has) {
  return (s && s.preview_health) || (has ? s.cameras : null);
}

function renderHealth(s, up, has) {
  // Arm: teleop answering + the PC's mode.
  if (!state.connected) setDot($("h-arm"), "bad", "Arm: disconnected");
  else if (!up) setDot($("h-arm"), "bad", "Arm: teleop not answering");
  else {
    const mode = has ? s.arm_mode : "";
    const level = mode === "fault" ? "bad" : ["idle", "running", "holding", "stopped"].includes(mode) ? "ok"
      : "warn";                                   // no state yet, or the PC is busy (homing)
    setDot($("h-arm"), level, `Arm: ${mode || "no state from the PC yet"}${has && s.moving ? " · moving" : ""}`);
  }
  // Cameras: both cameras' health.
  const cams = cameraHealth(s, has);
  if (!state.connected || (s && s.frames_answering === false)) {
    setDot($("h-cams"), "bad", `Cameras: ${(s && s.frames_error) || "frames not answering"}`);
  } else if (!cams) setDot($("h-cams"), "", "Cameras: no data yet");
  else {
    const st = [cams.main.status, cams.ultrawide.status];
    const level = st.includes("down") ? "bad" : st.includes("stale") ? "warn" : "ok";
    setDot($("h-cams"), level,
           `Cameras: wrist ${cams.main.status} ${Math.round(cams.main.fps)} fps · ` +
           `wide ${cams.ultrawide.status} ${Math.round(cams.ultrawide.fps)} fps`);
  }
  // Link: the page's socket, the PC's round trip and clock.
  if (!state.connected) setDot($("h-link"), "bad", "Link: disconnected");
  else if (!has) setDot($("h-link"), "warn", "Link: connected, teleop not answering");
  else {
    const rtt = s.rtt_ms !== null ? `RTT ${fmt(s.rtt_ms, 1)} ms` : "no RTT yet";
    setDot($("h-link"), s.clock_ok ? "ok" : "warn",
           `Link: connected · ${rtt} · clock ${s.clock_ok ? "ok" : "not ready"} · ${fmt(s.c2_hz, 0)} Hz`);
  }
}

function renderStorage(s, locked) {
  const root = s && typeof s.data_root === "string" && s.data_root ? s.data_root : "";
  const el = $("data-root");
  el.textContent = root ? shortPath(root) : "-";
  el.title = root || "storage folder unknown";
  $("storage-change").disabled = locked;
  $("storage-form").querySelector("[type=submit]").disabled = locked;
}

function renderTasks(s) {
  const tasks = (s && s.tasks) || [];
  const key = JSON.stringify([tasks, s ? s.task : ""]);
  const select = $("task-select");
  if (key !== state.tasksShown && document.activeElement !== select) {
    state.tasksShown = key;
    select.replaceChildren();
    const none = new Option(tasks.length ? "Choose a task" : "No tasks yet", "");
    none.disabled = true;
    select.add(none);
    for (const t of tasks) select.add(new Option(t, t));
    select.value = (s && s.task) || "";
  }
  $("task-description").textContent = (s && s.task_description) || "";
}

function renderDeleteTask(s, locked) {
  const task = (s && s.task) || "";
  $("delete-task-toggle").disabled = locked || !task;
  $("delete-task-toggle").classList.toggle("hidden", !task || !$("delete-task-form").classList.contains("hidden"));
  const form = $("delete-task-form");
  if (!form.classList.contains("hidden") && state.deleteTaskName !== task) hideDeleteTask();
  if (form.classList.contains("hidden")) return;
  const n = (s && s.episodes ? s.episodes.length : 0);
  const text = $("delete-task-text");
  const strong = document.createElement("strong");
  strong.textContent = task;
  text.replaceChildren("Delete task ", strong, ` and its ${plural(n, "episode")}? This removes the folder from disk. ` +
                       "Type the task name to confirm.");
  $("delete-task-confirm").disabled = locked || $("delete-task-input").value !== task;
}

function episodeCounts(episodes) {
  const c = { success: 0, fail: 0, unsaved: 0 };
  for (const ep of episodes) if (ep.outcome in c) c[ep.outcome] += 1;
  return c;
}

function renderEpisodes(s) {
  const episodes = (s && s.episodes) || [];
  const recording = s && s.phase === "recording";
  if (state.confirmEpisode && !episodes.some((ep) => ep.name === state.confirmEpisode)) state.confirmEpisode = null;
  const key = JSON.stringify([episodes, s ? s.task : "", recording, state.confirmEpisode]);
  if (key === state.episodesShown) return;
  state.episodesShown = key;
  const c = episodeCounts(episodes);
  const counts = $("episode-counts");
  counts.replaceChildren();
  if (episodes.length) {
    const parts = [`${episodes.length} total`, `${c.success} success`, `${c.fail} fail`];
    if (c.unsaved) parts.push(`${c.unsaved} unsaved`);
    counts.textContent = parts.join(" · ");
  }
  const list = $("episodes");
  list.replaceChildren();
  if (!episodes.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = s && s.task ? "No episodes yet. Press Start episode to record one."
      : "Choose a task, or add a new task";
    list.append(li);
    return;
  }
  const button = (text, cls, onclick) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = text;
    b.disabled = recording;
    b.onclick = onclick;
    return b;
  };
  for (const ep of [...episodes].reverse()) {             // newest first
    const li = document.createElement("li");
    if (state.confirmEpisode === ep.name) {               // asked inline, not with a browser dialog
      li.className = "confirming";
      const q = document.createElement("span");
      q.className = "ep-confirm";
      const nm = document.createElement("span");
      nm.className = "ep-name";
      nm.textContent = ep.name;
      q.append("Delete ", nm, " from disk?");
      const actions = document.createElement("span");
      actions.className = "ep-actions";
      actions.append(
        button("No", "btn btn-plain btn-sm", () => { state.confirmEpisode = null; render(); }),
        button("Yes, delete", "btn btn-danger btn-sm", () => {
          state.confirmEpisode = null;
          command("delete_episode", { episode: ep.name });
          render();
        }));
      li.append(q, actions);
      list.append(li);
      continue;
    }
    li.className = ep.outcome;
    const name = document.createElement("span");
    name.className = "ep-name";
    name.textContent = ep.name;
    const outcome = document.createElement("span");
    outcome.className = `pill pill-${ep.outcome}`;
    outcome.textContent = cap(ep.outcome);
    const dur = document.createElement("span");
    dur.className = "ep-dur";
    dur.textContent = `${ep.duration_s.toFixed(1)} s`;
    const frames = document.createElement("span");
    frames.className = "ep-frames";
    frames.textContent = `${ep.frames_main} frames`;
    const actions = document.createElement("span");
    actions.className = "ep-actions";
    if (ep.outcome === "unsaved") {             // ended without a choice: label it here (or delete it)
      for (const [label, value] of [["Save ✓", "success"], ["Save ✗", "fail"]]) {
        const b = button(label, value === "success" ? "btn btn-label-success" : "btn btn-secondary",
                         () => command("label_episode", { episode: ep.name, outcome: value }));
        b.title = `Save as ${value}`;
        b.setAttribute("aria-label", `Save ${ep.name} as ${value}`);
        actions.append(b);
      }
    }
    const del = button("", "icon-btn", () => { state.confirmEpisode = ep.name; render(); });
    del.innerHTML = TRASH_ICON;
    del.title = `Delete ${ep.name}`;
    del.setAttribute("aria-label", `Delete ${ep.name}`);
    actions.append(del);
    li.title = `${ep.name}: ${ep.outcome}, ${ep.duration_s.toFixed(1)} s, ${ep.frames_main} frames`;
    li.append(name, outcome, dur);
    if (ep.outcome !== "unsaved") li.append(frames);      // no room next to the save buttons (it is in the title)
    li.append(actions);
    list.append(li);
  }
}

function renderMessages() {
  const ids = [...state.messages.keys()].sort((a, b) => b - a);
  for (const id of ids.slice(MESSAGES_SHOWN)) state.messages.delete(id);
  const shown = ids.slice(0, MESSAGES_SHOWN);
  const open = $("log").open;
  if (open && shown.length) state.logSeen = Math.max(state.logSeen, shown[0]);
  const alert = shown.some((id) => id > state.logSeen && state.messages.get(id).level !== "info");
  $("log-alert").classList.toggle("hidden", open || !alert);
  $("log-count").textContent = String(shown.length);
  const newest = shown.length ? state.messages.get(shown[0]).text : "";
  $("log-latest").textContent = open ? "" : newest;
  const key = shown.join(",");
  if (key === state.messagesShown) return;
  state.messagesShown = key;
  const list = $("messages");
  list.replaceChildren();
  if (!shown.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nothing yet";
    list.append(li);
  }
  for (const id of shown) {
    const m = state.messages.get(id);
    const li = document.createElement("li");
    li.className = m.level;
    li.textContent = m.text;
    list.append(li);
  }
}

function renderCameras(s, has, recording) {
  const cams = cameraHealth(s, has);
  const framesGone = !state.connected || (s && s.frames_answering === false);
  for (const cam of ["main", "ultrawide"]) {
    const h = cams ? cams[cam] : null;
    const fig = $(`fig-${cam}`);
    fig.classList.toggle("down", framesGone || !h || h.status === "down");
    fig.classList.toggle("big", state.focusCam === cam);
    fig.classList.toggle("small", state.focusCam !== null && state.focusCam !== cam);
    fig.setAttribute("aria-label", `${CAMERA_WORDS[cam]}: ${state.focusCam === cam ? "click to show both" :
                                    "click to enlarge"}`);
    const el = $(`cam-${cam}`);
    let level = "bad";
    let text = "No signal";
    if (h && !framesGone) {
      level = h.status === "ok" ? "ok" : h.status === "stale" ? "warn" : "bad";
      text = h.status === "ok" ? `${Math.round(h.fps)} fps` : h.status === "stale" ? `Stale · ${Math.round(h.fps)} fps`
        : "Down";
    }
    setDot(el, level, h ? `${h.status}, ${fmt(h.fps, 1)} fps, ${h.dropped} dropped, newest ` +
                         `${fmt(h.last_frame_age_ms, 0)} ms old` : "no health yet");
    el.querySelector(".cam-fps").textContent = text;
  }
  $("cameras").classList.toggle("focused", state.focusCam !== null);
  $("cam-hint").textContent = state.focusCam ? "Click the large camera to show both side by side"
    : "Click a camera to enlarge it";
  $("rec-badge").classList.toggle("hidden", !recording);
  if (recording) $("rec-text").textContent = "REC " + mmss(s.episode_s);
  $("limit").classList.toggle("hidden", !(has && s.at_limit));
}

function renderDiagnostics(s, has) {
  $("phase").textContent = has ? s.phase + (s.ready ? "" : " (starting up)") : "-";
  $("arm").textContent = has ? (s.arm_mode || "-") + (s.moving ? " · moving" : "") : "-";
  $("tcp").textContent = has && s.tcp_m
    ? s.tcp_m.map((v, i) => `${"xyz"[i]} ${(v * 1000).toFixed(1)}`).join("  ") + " mm" : "-";
  $("grip").textContent = has
    ? `${has && s.gripper_width_m !== null ? fmt(s.gripper_width_m * 1000, 1) : "-"} mm · commanded ` +
      `${s.commanded_width_m > 0 ? "open" : "close"}` : "-";
  const cams = cameraHealth(s, has);
  const camLine = (h) => `${h.status} ${Math.round(h.fps)} fps, ${h.dropped} dropped`;
  $("cams").textContent = cams ? `wrist ${camLine(cams.main)} · wide ${camLine(cams.ultrawide)}` : "-";
  $("c2").textContent = has ? `${fmt(s.c2_hz, 0)} Hz` : "-";
  $("rtt").textContent = has && s.rtt_ms !== null ? `${fmt(s.rtt_ms, 1)} ms` : "-";
  $("clock").textContent = has ? (s.clock_ok ? "ok" : "not ready") : "-";
  $("speed").textContent = has ? (s.slow ? "slow (¼)" : "normal") : "-";
  $("diag-summary").textContent = has
    ? `${s.arm_mode || "no arm state"} · ${fmt(s.c2_hz, 0)} Hz · ` +
      `${s.rtt_ms !== null ? fmt(s.rtt_ms, 1) + " ms" : "-"} round trip · clock ${s.clock_ok ? "ok" : "not ready"}`
    : "";
  // The gripper keys show the newest command (N: closed, M: open).
  for (const capEl of document.querySelectorAll(".cap[data-grip]")) {
    const current = has && (s.commanded_width_m > 0 ? "open" : "close") === capEl.dataset.grip;
    capEl.classList.toggle("current", current);
    capEl.title = current ? "The last gripper command" : "";
  }
}

// The one place that says what to do next.
function renderState(s, has, recording) {
  let dot = "";
  let title;
  let detail = "";
  let detailClass = "state-detail";
  if (!state.connected) { title = "Not connected"; detail = "Waiting for teleop_collect…"; dot = "warn"; }
  else if (!has) {
    title = "Waiting for teleop"; detail = cap((s && s.teleop_error) || "teleop not answering"); dot = "warn";
  }
  else if (s.phase === "done") { title = "Stopped"; detail = cap(s.final_line || "teleop ended"); }
  else if (recording && state.choosing) {
    dot = "rec"; title = "Save this episode?";
    detail = `${s.recording_episode} is still recording. Keys are off until you choose.`;
  } else if (recording) {
    dot = "rec"; title = `Recording ${s.recording_episode}`;
    detail = "Hold keys to move · wait ~1 s after Start";
  } else if (s.phase === "stopping") { title = "Finishing the episode…"; }
  else if (s.phase === "starting" || !s.ready) {
    title = "Starting up…";
    detail = s.start_refusal !== "starting up" ? cap(s.start_refusal) : "";
  }
  else if (s.start_allowed) {
    dot = "ok"; title = "Ready"; detailClass += " ok";
    detail = cap(s.start_pose_confirmed ? "at the start pose ✓ ready to start"
      : "at the start-up pose ✓ ready to start (the PC checks it at Start)");
  } else { dot = "warn"; title = "Not ready"; detailClass += " refused"; detail = cap(s.start_refusal); }
  $("state-dot").className = `state-dot ${dot}`;
  $("state-title").textContent = title;
  const hint = $("hint");
  hint.textContent = detail;
  hint.className = detailClass;
  const timer = $("state-timer");
  timer.classList.toggle("hidden", !recording);
  if (recording) timer.textContent = mmss(s.episode_s);
}

function render() {
  const s = state.status;
  const up = teleopUp();
  const has = up && s.phase !== undefined;
  renderBanner(s);
  renderHealth(s, up, has);
  $("mock-badge").classList.toggle("hidden", !(s && s.pc_mock));
  renderTasks(s);
  renderEpisodes(s);
  renderMessages();

  const recording = has && s.phase === "recording";
  if (!recording) state.choosing = false;
  if (!state.choosing) state.confirmDiscard = false;
  if (recording && !state.wasRecording) closeForms();
  state.wasRecording = recording;
  renderCameras(s, has, recording);
  renderDiagnostics(s, has);
  renderState(s, has, recording);

  const busy = !has || s.phase === "stopping" || s.phase === "done" || !s.ready;
  $("btn-start").disabled = !has || !s.start_allowed || recording;
  $("btn-start").classList.toggle("hidden", recording);
  $("btn-stop-episode").classList.toggle("hidden", !recording || state.choosing);
  $("save-choice").classList.toggle("hidden", !recording || !state.choosing || state.confirmDiscard);
  $("discard-confirm").classList.toggle("hidden", !recording || !state.confirmDiscard);
  if (recording && state.confirmDiscard) {
    $("discard-text").textContent = `Discard ${s.recording_episode}? It is deleted from disk.`;
  }
  $("btn-stop-arm").disabled = !state.connected;
  $("task-select").disabled = busy || recording;
  $("new-task-toggle").disabled = busy || recording;
  $("new-task-form").querySelector("[type=submit]").disabled = busy || recording;
  renderStorage(s, busy || recording);
  renderDeleteTask(s, busy || recording);
  renderKeys({ held: [...state.held], slow: state.shift, focus: pageFocused() });
}

// ------------------------------------------------------------------------------------------------- the buttons
function hideNewTask() {
  $("new-task-form").classList.add("hidden");
  $("task-card").classList.remove("creating");
  $("new-task-error").textContent = "";
}

function hideStorageForm() {
  $("storage-form").classList.add("hidden");
  $("storage-view").classList.remove("hidden");
  $("storage-error").textContent = "";
}

function hideDeleteTask() {
  state.deleteTaskName = null;
  $("delete-task-form").classList.add("hidden");
  $("delete-task-toggle").classList.remove("hidden");
  $("delete-task-input").value = "";
  $("delete-task-error").textContent = "";
}

// The header's theme toggle: dark (default) or light, remembered per browser.
function renderThemeToggle() {
  const light = document.documentElement.dataset.theme === "light";
  const label = light ? "Dark theme" : "Light theme";       // what pressing it switches to
  $("theme-toggle").setAttribute("aria-label", label);
  $("theme-toggle").title = label;
  $("theme-sun").classList.toggle("hidden", light);
  $("theme-moon").classList.toggle("hidden", !light);
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage blocked: not remembered */ }
  renderThemeToggle();
}

// Every inline form closed and its field blurred, so the keys reach the page again (recording starts, Stop arm).
function closeForms() {
  hideNewTask();
  hideStorageForm();
  hideDeleteTask();
  state.confirmEpisode = null;
  if (isTyping(document.activeElement)) document.activeElement.blur();
}

function toggleCamera(cam) {
  state.focusCam = state.focusCam === cam ? null : cam;
  render();
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
  $("btn-discard").onclick = () => { state.confirmDiscard = true; render(); };
  $("btn-discard-back").onclick = () => { state.confirmDiscard = false; render(); };
  $("btn-discard-yes").onclick = stopAs("discard");
  $("btn-stop-arm").onclick = stopArm;
  $("theme-toggle").onclick = toggleTheme;
  renderThemeToggle();
  $("task-select").onchange = (e) => { if (e.target.value) command("select_task", { name: e.target.value }); };
  $("new-task-toggle").onclick = () => {
    hideDeleteTask();
    $("new-task-form").classList.remove("hidden");
    $("task-card").classList.add("creating");
    $("new-task-name").focus();
  };
  $("new-task-cancel").onclick = hideNewTask;
  $("new-task-form").onsubmit = (e) => {
    e.preventDefault();
    $("new-task-error").textContent = "";
    command("create_task", { name: $("new-task-name").value.trim(),
                             description: $("new-task-description").value.trim() });
  };

  // The storage folder: an inline field instead of the path.
  $("storage-change").onclick = () => {
    const s = state.status;
    $("storage-path").value = (s && s.data_root) || "";
    $("storage-error").textContent = "";
    $("storage-view").classList.add("hidden");
    $("storage-form").classList.remove("hidden");
    $("storage-path").focus();
    $("storage-path").select();
  };
  $("storage-cancel").onclick = hideStorageForm;
  $("storage-form").onsubmit = (e) => {
    e.preventDefault();
    const path = $("storage-path").value.trim();
    if (!path) { $("storage-error").textContent = "Type a folder, e.g. ~/CapstoneTeleop"; return; }
    $("storage-error").textContent = "";
    command("set_data_root", { path });
  };

  // Delete task: typed confirmation, inline.
  $("delete-task-toggle").onclick = () => {
    const s = state.status;
    if (!s || !s.task) return;
    hideNewTask();
    state.deleteTaskName = s.task;
    $("delete-task-input").value = "";
    $("delete-task-error").textContent = "";
    $("delete-task-form").classList.remove("hidden");
    $("delete-task-toggle").classList.add("hidden");
    render();
    $("delete-task-input").focus();
  };
  $("delete-task-cancel").onclick = hideDeleteTask;
  $("delete-task-input").oninput = () => render();
  $("delete-task-form").onsubmit = (e) => {
    e.preventDefault();
    const name = state.deleteTaskName;
    if (!name || $("delete-task-input").value !== name) return;
    $("delete-task-error").textContent = "";
    command("delete_task", { name });
  };

  // The cameras: click (or Enter / Space) to enlarge one; again to show both.
  for (const cam of ["main", "ultrawide"]) {
    const fig = $(`fig-${cam}`);
    fig.onclick = () => toggleCamera(cam);
    fig.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleCamera(cam); }
    };
  }
  $("log").addEventListener("toggle", () => render());

  // Buttons must not keep the keyboard focus (Space / Enter would press them again): give it back to the page.
  for (const b of document.querySelectorAll("button")) b.addEventListener("mouseup", () => b.blur());
  // The same for the buttons drawn later (episode rows) and the cameras.
  document.addEventListener("mouseup", (e) => {
    const el = e.target instanceof Element ? e.target.closest(".episodes button, .camera") : null;
    if (el) el.blur();
  });

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", loseKeys);
  window.addEventListener("focus", () => sendKeys(true));
  // A field taking or losing the focus turns the key map off or on (after the focus has moved).
  document.addEventListener("focusin", () => setTimeout(render, 0));
  document.addEventListener("focusout", () => setTimeout(render, 0));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") loseKeys(); else sendKeys(true);
  });
  render();
  connect();
}

setup();
