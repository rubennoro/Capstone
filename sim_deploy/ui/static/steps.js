// sim_deploy ui: the step panel's model. Pure functions, no DOM: app.js renders what computeSteps() returns.
// Words come from HELP.STEPS / HELP.PAGE / HELP.OUTCOMES (help.js).
//
// computeSteps(status, telemetry, request, entry, memo) -> [{group, key, title, help, tip, state, detail}]
//   status    the runner's Status, or null for the preview (nothing running: the planned steps of `request`)
//   telemetry the latest telemetry, or null when it is stale / the scene source is down
//   request   the RunRequest (preview: the page's settings; live: status.request)
//   entry     {type: "policy"|"replay"|"builtin", index?} of the picked entry, or null (derived from request.policy)
//   memo      optional, kept by the page per run: {lastPos, lastWork, seen: Set, eraserBase}; without lastWork (a
//             page opened after the run) status.last_work_step (runner.py) gives where a failed run failed
//   state     todo | now | done | skip | fail
//   rollout   the 1-based rollout of a rollout-stage item, else null (the page's per-step timing)
//
// Where a run is: status.step ({key, rollout, since_s, detail}, sim_deploy/ui/runner.py) when the server sends it,
// else the phase string mapped by PHASE_STEPS (the fallback; then a rollout shows one "reset" step for h, p and the
// eraser placement, and the current rollout = outcomes.length + 1).
"use strict";

const STEP_STAGES = {
  bringup: ["prepare", "sim_pc", "sim_pi", "policy"],
  rollout: ["home", "start_pose", "place", "control", "record"],
  finish: ["wrapup", "sim_stop", "score"],
};
const STEP_STAGE_OF = {};
for (const [stage, keys] of Object.entries(STEP_STAGES)) for (const k of keys) STEP_STAGE_OF[k] = stage;
STEP_STAGE_OF.reset = "rollout";
const STAGE_RANK = { bringup: 0, rollout: 1, finish: 2 };
// "reset" (the fallback) stands for home + start_pose + place
const KEY_RANK = { prepare: 0, sim_pc: 1, sim_pi: 2, policy: 3, reset: 0, home: 0, start_pose: 1, place: 2,
                   control: 3, record: 4, wrapup: 0, sim_stop: 1, score: 2 };
const STEP_BUSY = ["starting", "running", "stopping"];

// Fallback when status.step is absent: Status.phase -> step key. The phases are the runner's: RUNSH_PHASES in
// sim_deploy/ui/runner.py (run.sh / driver lines), "starting run.sh" (the runner's first phase), and policy_deploy's
// status heads without brackets ("starting", "idle", "ep N starting", "ep N running X s", "done"). Keep in step with
// RUNSH_PHASES. Phases that say nothing about the step ("stopping", "killing", "stopped", "finished", "failed") map
// to null: the page keeps the last known step.
const PHASE_STEPS = [
  [/^starting run\.sh/, "prepare"],
  [/^building the model/, "prepare"],
  [/^starting the sim PC/, "sim_pc"],
  [/^sim PC up/, "sim_pc"],
  [/^restarting the sim/, "sim_pc"],
  [/^retrying the sim start/, "sim_pc"],
  [/^starting the sim Pi/, "sim_pi"],
  [/^starting the eval driver/, "policy"],
  [/^preparing/, "policy"],
  [/^starting$/, "policy"],
  [/^idle$/, "reset"],
  [/^homing/, "reset"],
  [/^ep \d+ (starting|running)/, "control"],
  [/^done$/, "record"],
  [/^stopping policy_deploy/, "wrapup"],
  [/^results written/, "sim_stop"],
  [/^sim stopped/, "score"],
];

// "Hello {name}" -> "Hello Ada"; unknown names stay as they are.
function fill(text, vars) {
  if (!text) return "";
  return String(text).replace(/\{(\w+)\}/g, (m, k) => (vars && vars[k] !== undefined && vars[k] !== null ? String(vars[k]) : m));
}

function phaseStep(phase) {
  const p = String(phase || "").trim();
  for (const [rx, key] of PHASE_STEPS) if (rx.test(p)) return key;
  return null;
}

// Where the run is now: {stage, key, rollout (1-based), since_s, detail, fallback} or null.
function stepPosition(status) {
  if (!status) return null;
  const outcomes = Array.isArray(status.outcomes) ? status.outcomes : [];
  const N = status.rollouts || 0;
  const doneN = outcomes.filter((o) => o !== "not_run").length;   // a restart fills not_run, then replaces it
  const next = N ? Math.min(doneN + 1, N) : doneN + 1;
  const s = status.step;
  if (s && typeof s === "object" && STEP_STAGE_OF[s.key]) {
    const stage = STEP_STAGE_OF[s.key];
    const r = typeof s.rollout === "number" ? s.rollout : (typeof status.rollout === "number" && status.rollout > 0
      ? status.rollout : next);
    return { stage, key: s.key, rollout: stage === "rollout" ? r : null, since_s: s.since_s ?? null,
             detail: s.detail ?? null, fallback: false };
  }
  if (s && typeof s === "object") return null;          // a key this page does not know
  const key = phaseStep(status.phase);
  if (!key) return null;
  const stage = STEP_STAGE_OF[key];
  if (stage === "rollout" && N && doneN >= N) {     // every rollout recorded: the driver wraps up
    return { stage: "finish", key: "wrapup", rollout: null, since_s: null, detail: null, fallback: true };
  }
  return { stage, key, rollout: stage === "rollout" ? next : null, since_s: null, detail: null, fallback: true };
}

// status.last_work_step ({key, rollout, since_s, detail}: the last step that was not wrapup / sim_stop) as a position
function workPosition(status) {
  const w = status && status.last_work_step;
  if (!w || typeof w !== "object" || !STEP_STAGE_OF[w.key]) return null;
  const stage = STEP_STAGE_OF[w.key];
  return { stage, key: w.key, rollout: stage === "rollout" ? (typeof w.rollout === "number" ? w.rollout : 1) : null,
           since_s: w.since_s ?? null, detail: w.detail ?? null, fallback: false };
}

function targetType(entry, policy) {
  if (entry && entry.type) return entry.type;
  const p = String(policy || "");
  if (p === "hold_still") return "builtin";
  if (p.startsWith("replay:")) return "replay";
  return "policy";
}

function replayIndex(entry, policy) {
  if (entry && typeof entry.index === "number") return entry.index;
  const m = String(policy || "").match(/#(\d+)$/);
  return m ? Number(m[1]) : null;
}

function posRank(stage, rollout, key) {
  return [STAGE_RANK[stage] ?? 0, stage === "rollout" ? (rollout || 1) : 0, KEY_RANK[key] ?? 0];
}

function cmpRank(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

function computeSteps(status, telemetry, request, entry, memo) {
  const H = typeof HELP !== "undefined" ? HELP : {};
  const S = H.STEPS || {}, P = H.PAGE || {}, O = H.OUTCOMES || {};
  const m = memo || {};
  const st = status ? status.state : null;
  const live = !!st && STEP_BUSY.includes(st);
  const finished = !!st && (st === "done" || st === "failed") && !!status.request;
  const req = request || (status && status.request) || {};
  const type = targetType(entry, req.policy);
  const N = (status && status.rollouts) || req.rollouts || 1;
  const outcomes = status && Array.isArray(status.outcomes) ? status.outcomes : [];
  const doneN = outcomes.filter((o) => o !== "not_run").length;
  const homeStart = (req.home || "start") === "start";
  const startsSet = req.starts === "set" && type !== "replay";
  // the reset steps one by one, unless this server sends no status.step at all (then one "reset" step)
  const fallback = (live || finished) && !("step" in status);
  const vars = { seed: req.seed ?? 0, N, i: replayIndex(entry, req.policy) ?? "?", T: req.timeout_s ?? 60,
                 SET: String((m.setName || "w3")).toUpperCase(), set: m.setName || "w3" };

  let pos = live || finished ? (stepPosition(status) || m.lastPos || null) : null;
  let failPos = null;
  if (finished && st === "failed") {
    const lastWork = m.lastWork || workPosition(status);   // the memo, else the runner's (a page opened later)
    failPos = doneN < N && lastWork ? lastWork : (pos || lastWork || null);
    if (!failPos) failPos = { stage: "bringup", key: "prepare", rollout: null };
  }
  const ref = failPos || pos;
  const refRank = ref ? posRank(ref.stage, ref.rollout, ref.key) : null;
  const items = [];
  const t = (key) => S[key] || {};

  // the state of the step at (stage, rollout, key)
  function stateOf(stage, rollout, key) {
    if (!live && !finished) return "todo";
    if (finished && st === "done") return "done";
    if (!refRank) return live ? "todo" : "skip";
    const c = cmpRank(posRank(stage, rollout, key), refRank);
    if (failPos) return c < 0 ? "done" : c === 0 ? "fail" : "skip";
    return c < 0 ? "done" : c === 0 ? "now" : "todo";
  }

  function push(group, key, title, help, tip, state, detail, rollout) {
    items.push({ group, key, title, help: help || "", tip: tip || "", state, detail: detail || "",
                 rollout: rollout ?? null });
  }

  function failDetail(state) {
    return state === "fail" ? (status && status.last_error) || "" : "";
  }

  // ---------------------------------------------------------------------------------------------- bring-up
  const gBring = (P.groups && P.groups.bringup) || "Bring-up";
  for (const key of STEP_STAGES.bringup) {
    const state = stateOf("bringup", null, key);
    let detail = failDetail(state);
    if (state === "now" && pos) {
      detail = pos.detail || (/restart|retry/.test(String(status.phase || "")) ? status.phase : "");
    }
    const help = key === "policy" && type !== "policy" ? t(key).help_none : t(key).help;
    push(gBring, key, t(key).title || key, help, t(key).tip, state, detail);
  }

  // ---------------------------------------------------------------------------------------------- rollouts
  const groupOf = (k) => fill((P.groups && P.groups.rollout) || "Rollout {k} of {N}", { k, N });
  const rolloutKeys = fallback ? ["reset", "control", "record"]
    : STEP_STAGES.rollout.filter((k) => k !== "start_pose" || homeStart);

  function rolloutSteps(k) {
    const g = groupOf(k);
    const v = Object.assign({}, vars, { k, start: k - 1 });   // start: the driver's 0-based start-set index
    for (const key of rolloutKeys) {
      const state = stateOf("rollout", k, key);
      let help = t(key).help, tip = t(key).tip, detail = failDetail(state);
      const here = pos && pos.stage === "rollout" && pos.rollout === k;
      if (key === "place") {
        help = type === "replay" ? t(key).help_replay : startsSet ? t(key).help_set : t(key).help_random;
        if (here && pos.key === "place" && pos.detail) detail = pos.detail;
        else if (here && cmpRank(posRank("rollout", k, "place"), posRank(pos.stage, pos.rollout, pos.key)) < 0
                 && m.placeDetail && m.placeDetail.rollout === k) detail = m.placeDetail.text;
      } else if (key === "control") {
        if (type === "builtin") help = t(key).help_hold;
        else if (type === "replay") { help = t(key).help_replay; tip = t(key).tip_replay || tip; }
        if (state === "now") detail = controlDetail(status, telemetry, m, v, t(key));
      } else if (key === "record") {
        const word = state === "now" && pos && pos.detail ? String(pos.detail).trim() : null;
        if (word) detail = O[word] ? `${word}: ${O[word]}` : word;
      } else if (state === "now" && pos && pos.detail) {
        detail = pos.detail;
      }
      push(g, key, t(key).title || key, fill(help, v), fill(tip, v), state, detail, k);
    }
    if (k < N) {
      const more = t("more");
      const title = k + 1 === N ? fill(more.title_one || "Rollout {N}", { N }) : fill(more.title, { a: k + 1, N });
      push(g, "more", title, more.help, "", stateOf("rollout", k + 1, "home") === "skip" ? "skip" : "todo", "");
    }
  }

  const gRoll = (P.groups && P.groups.rollouts) || "Rollouts";
  function summary(state) {
    const d = t("rollouts_done");
    push(gRoll, "rollouts_done", fill(d.title || "{k} of {N} rollouts recorded", { k: outcomes.length, N }),
         d.help, "", state, "");
  }

  if (!live && !finished) rolloutSteps(1);
  else if (failPos && failPos.stage === "rollout") rolloutSteps(failPos.rollout || 1);
  else if (finished) summary(outcomes.length ? "done" : "skip");
  else if (!pos || pos.stage === "bringup") rolloutSteps(Math.min(status.rollout > 0 ? status.rollout : doneN + 1, N));
  else if (pos.stage === "rollout") rolloutSteps(Math.min(pos.rollout || doneN + 1, N));
  else summary("done");

  // ---------------------------------------------------------------------------------------------- finish
  const gFin = (P.groups && P.groups.finish) || "Finish";
  for (const key of STEP_STAGES.finish) {
    let state, detail = "";
    if (key === "score") {
      const sc = status && status.score;
      if (sc && sc.state === "scoring") state = "now";
      else if (sc && sc.state === "done") state = "done";
      else if (sc && sc.state) { state = "fail"; detail = sc.error || "evaltool score failed"; }
      else state = finished ? (outcomes.length ? "todo" : "skip") : "todo";
    } else if (failPos && failPos.stage !== "finish") {
      const simStopped = (m.seen && m.seen.has && m.seen.has("sim_stop"))
        || !!(status.step && status.step.key === "sim_stop");
      state = key === "wrapup" ? (status.metrics ? "done" : "skip") : (simStopped ? "done" : "skip");
    } else {
      state = stateOf("finish", null, key);
      detail = failDetail(state);
      if (state === "now" && pos && pos.detail) detail = pos.detail;
    }
    push(gFin, key, t(key).title || key, t(key).help, t(key).tip, state, detail);
  }
  return items;
}

// "12.3 s of 60 s · ≈ eraser up 3.1 cm (success at 5 cm held 1 s) · gripping"
function controlDetail(status, telemetry, memo, vars, words) {
  const out = [];
  const mt = String((status && status.phase) || "").match(/^ep \d+ running ([\d.]+)\s*s/);
  if (mt) out.push(fill(words.time || "{t} s of {T} s", { t: Number(mt[1]).toFixed(1), T: vars.T }));
  const h = telemetry && telemetry.eraser ? telemetry.eraser.height_m : null;
  if (typeof h === "number" && typeof memo.eraserBase === "number") {
    const cm = Math.max(0, (h - memo.eraserBase) * 100);
    out.push(fill(words.lift || "≈ eraser up {cm} cm", { cm: cm.toFixed(1) }));
  }
  if (telemetry && telemetry.contacts && telemetry.contacts.both_fingers) out.push(words.gripping || "gripping");
  return out.join(" · ");
}
