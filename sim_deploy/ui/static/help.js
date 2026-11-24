// sim_deploy ui: every word the page shows, in one place. Edit freely; app.js and steps.js only read this.
//
// - `label` = the field's name, `help` = the short line under it (keep it under ~60 characters: the side columns
//   are ~300 px wide), `tip` = the longer text in the hover tooltip.
// - {name} placeholders are filled by the page (fill() in steps.js): {seed}, {k}, {n}, {i}, {set}, {t}, {T}, {cm}...
// - The text in index.html is only a fallback for when this file does not load; this file wins.
"use strict";

const HELP = {
  // ------------------------------------------------------------------------------------------- settings fields
  FIELDS: {
    sec_rollouts: { label: "Rollouts" },
    sec_scene: { label: "Scene" },
    rollouts: {
      label: "Rollouts",
      help: "How many trials. Each: arm resets, eraser placed, policy tries to pick it up.",
      tip: "One rollout = one closed-loop trial from one eraser placement (1 to 100).",
    },
    starts: {
      label: "Eraser starts",
      help: "Where the eraser goes in each rollout.",
      tip: "random: a spot from the placement seed.\nstart set: the fixed starts of the eval batch "
        + "(sim_deploy/eval/start_sets/w3.yaml), numbered from 0 as the driver does: rollout 1 uses start #0, "
        + "rollout 2 start #1, ...",
      options: { random: "Random from the seed", set: "{SET} start set ({n} fixed starts, same as the batch)" },
      no_replay: "not used: a replay puts the eraser where the demo had it",
      no_home: "start set needs Arm start = start pose (built there)",
      no_split: "the start set was built without this run's training split",
      too_many: "the {set} start set has {n} starts: at most {n} rollouts",
    },
    seed: {
      label: "Placement seed",
      help: "Chooses where the eraser goes; same seed = same spots for every policy.",
      tip: "Rollout k of seed S always gets the same eraser spot. The dice picks a random seed.\nDefault 523: "
        + "rollout 1 starts the eraser at about (0.534, -0.068), the spot nearest the teleop demos' objects.",
      off_replay: "not used: a replay puts the eraser where the demo had it",
      off_set: "not used: the start set fixes the spots",
    },
    timeout: {
      label: "Time limit per rollout (s)",
      help: "Most time the policy gets; no lift by then = timeout.",
      tip: "Seconds the policy has in each rollout before it counts as a timeout.",
    },
    home: {
      label: "Arm start",
      help: "",
      tip: "Where each rollout starts. start pose: the robot PC's h (home), then p (start pose), as on the real arm. "
        + "deploy home: h only.",
      options: { start: "start pose, like the real arm (h, then p)", deploy: "deploy home only (h)" },
      deploy_line: "deploy home: fingertips down, ~30 cm above the table",
      deploy_tip: "the sim PC runs without a start pose; every rollout starts at the deploy home (h)",
    },
    look: {
      label: "Scene look",
      help: "All use the 30 mm eraser.",
      tip: "What both sims load (run.sh --look). robot cell = looks like the real lab (default); "
        + "training scene = where the demos were recorded; current = the older sim look.",
      options: { robot: "robot cell (like the real lab)", train: "training scene (where the demos were recorded)",
                 current: "current (older sim look)" },
      words: { robot: "the robot cell", train: "the training scene", current: "the older sim look" },
    },
    more: { label: "More options" },
    tag: {
      label: "Run name (tag)",
      help: "blank = ui",
      tip: "Goes into the eval folder name and the results table (letters, digits, _ . -).",
    },
    tap: {
      label: "Save per-step policy data (tap)",
      help: "inputs, planned path, images every 10th step; for debugging, bigger folder",
      tip: "policy_deploy's manager.save_tap: observation, action chunk and waypoints per step.",
    },
    policy_settings: {
      label: "Policy settings: leave as the arm's",
      not_arm: "not the arm's settings",
      not_arm_tip: "The real arm runs bf16, 8 denoising steps, 12 actions per chunk.",
      reset: "Reset to the arm's",
      not_used: "no network runs; not used",
    },
    precision: { label: "Precision", help: "bf16 = arm", tip: "Number format of the network on the GPU." },
    steps: { label: "Denoising steps", help: "8 = arm", tip: "Diffusion denoising steps per action chunk." },
    exech: {
      label: "Actions used per chunk (exec_h)",
      help: "12 = arm",
      tip: "How many actions of each predicted chunk run before the next inference (1 to 16).",
    },
    hide_help: { label: "hide explanations", tip: "Hide the grey help lines (tooltips stay)." },
  },

  // ---------------------------------------------------------------------------------------------- step panel
  // One entry per step key (status.step.key; sim_deploy/ui/runner.py). `reset` is the page's own fallback step.
  STEPS: {
    prepare: { title: "Prepare the scene", help: "run.sh checks the setup, builds the model if needed",
               tip: "run.sh: checks, then builds sim_deploy/build/<look>/ if it is missing or older than the station." },
    sim_pc: { title: "Start the simulated robot PC", help: "stands in for the arm's PC (franka_control)",
              tip: "The sim PC: MuJoCo arm + eraser behind the real PC's sockets." },
    sim_pi: { title: "Start the simulated cameras", help: "renders the rig's two cameras, sent like the Pi",
              tip: "The sim Pi renders the IMX519 and IMX219 views and streams them as the Pi does." },
    policy: { title: "Load the policy", help: "policy_deploy loads the checkpoint",
              help_none: "policy_deploy starts; no network runs",
              tip: "The same policy_deploy as on the laptop next to the real arm, with this run's settings." },
    home: { title: "Home (h)", help: "the robot PC moves the arm to its home",
            tip: "The driver presses h: the sim PC moves the arm to the deploy home." },
    start_pose: { title: "Go to the start pose (p)", help: "then to the start pose, like the real arm",
                  tip: "The driver presses p: the sim PC moves the arm to stations/start_pose.yaml." },
    place: { title: "Place the eraser",
             help_random: "at the spot seed {seed} gives rollout {k}, then 0.3 s to settle",
             help_set: "{SET} start #{start} (the driver's start index, from 0)",
             help_replay: "where it was in demo #{i}",
             tip: "The driver puts the eraser flat on the table and lets it settle before the policy starts." },
    control: { title: "Policy in control (c)", help: "ends at success, the time limit, a fault or a stop",
               tip: "Ends at success (lifted ≥ 5 cm, held 1 s, clear of the table), the time limit, a robot-PC "
                 + "fault or stop, or policy_deploy ending it.",
               help_hold: "no policy, the arm holds still (baseline)",
               help_replay: "plays back demo #{i}'s motion open loop, no cameras",
               tip_replay: "Plays back demo #{i}'s recorded motion open loop; ignores the cameras.",
               time: "{t} s of {T} s",
               lift: "≈ eraser up {cm} cm (success at 5 cm held 1 s)",
               gripping: "gripping" },
    record: { title: "Record the result", help: "the outcome goes into rollouts.jsonl",
              tip: "The driver writes this rollout's record (eval folder rollouts.jsonl)." },
    reset: { title: "Reset (home, start pose, place the eraser)", help: "h, p, then the eraser at its spot",
             tip: "This server does not report the single reset steps; they run in this order." },
    more: { title: "Rollouts {a} to {N}", help: "the same steps for each", title_one: "Rollout {N}" },
    rollouts_done: { title: "{k} of {N} rollouts recorded", help: "one chip per rollout under Rollouts" },
    wrapup: { title: "Write the results", help: "metrics.json, summary.html, a row in index.csv",
              tip: "The driver quits policy_deploy and writes the eval folder's results." },
    sim_stop: { title: "Stop the sim", help: "run.sh stops the sim PC and the sim Pi" },
    score: { title: "Score with evaltool", help: "success, grasp and lift per rollout",
             tip: "python -m sim_deploy.evaltool score <eval>: the scores the results table uses." },
  },

  // -------------------------------------------------------------------------------------- rollout outcomes
  OUTCOMES: {
    success: "lifted and held",
    fail: "not lifted",
    timeout: "time limit reached",
    stopped: "stopped by the robot PC or policy_deploy",
    fault: "robot-PC fault",
    crashed: "policy_deploy exited",
    home_failed: "could not home",
    start_pose_failed: "could not reach the start pose",
    start_failed: "could not start the policy",
    not_run: "not run",
    interrupted: "stopped from the page",
    ended_by_policy_deploy: "ended by policy_deploy",
    sim_error: "sim error",
  },

  // ------------------------------------------------------------------------- policy kinds (catalog "kind")
  // `word` = plain words, shown with the raw kind in small type. "other" shows the train_config verbatim.
  KINDS: {
    prompting: { word: "behavior prompting", short: "prompting" },
    plain: { word: "plain diffusion", short: "plain" },
    diffusion: { word: "plain diffusion", short: "plain" },     // the driver's word for plain (index.csv)
    other: { word: "", short: "other" },
    unknown: { word: "type unknown", short: "unknown" },
    replay: { word: "replay", short: "replay" },
    hold_still: { word: "hold still", short: "hold still" },
  },

  // --------------------------------------------------------------------------- entry types (badge colours)
  TYPES: {
    policy: { badge: "POLICY", sub: "", tip: "A trained checkpoint, closed loop on the cameras." },
    replay: { badge: "REPLAY", sub: "open loop", tip: "Plays back one recorded demo's motion; no cameras." },
    builtin: { badge: "BASELINE", sub: "hold still", tip: "No policy: the arm holds still." },
  },

  // --------------------------------------------------------------------------------------- other page text
  PAGE: {
    picker_tip: "The sim has one eraser; every entry runs in that scene.",
    group_policies: "Policies: trained checkpoints, closed loop",
    group_replays: "Replays: play back one recorded demo, open loop",
    group_builtin: "Baseline: no policy",
    not_fetched: "not fetched",
    not_runnable: "not runnable",
    ready: "Ready to run",
    nothing: "nothing selected: pick a policy above",
    steps_preview: "plan",
    steps_live: "live",
    steps_done: "finished",
    groups: { bringup: "Bring-up", rollout: "Rollout {k} of {N}", rollouts: "Rollouts", finish: "Finish" },
    bringup_done: "Sim and policy ready",          // the folded bring-up line; its time sits in the time column
    stopping: "Stopping: the current rollout ends cleanly, then results are written. Press Stop again to force.",
    no_run: "no run yet",
    raw_log: "Raw log",
    live_numbers: "Live numbers",
    // panel titles and the header (visual restyle 2026-11-23)
    subtitle: "sim eval",
    panel_policy: "Policy",
    panel_settings: "Run settings",
    panel_progress: "Progress",
    panel_results: "Sim results",
    chips_title: "Rollouts",
    only_selected: "only the selected policy",
    reload: "Reload",
    // the camera tiles: name (shown in small capitals), the caption's right side, the state inside the tile
    cams: { 1: "External", 2: "Wrist · main (IMX519)", 3: "Wrist · ultrawide (IMX219)", 4: "Policy input · 2×2" },
    cam_meta: "{w}×{h} · {fps} fps",
    cam_live: "Live",
    cam_stalled: "Stalled {s}",
    cam_stalled_tip: "no new frame for {s}; the last frame is shown dimmed",
    cam_waiting: "Waiting for camera",
    // the run banner above the external view
    banner_idle: "No run yet. Pick an entry on the left and press Play.",
    banner_now: "Running",
    banner_stopping: "Stopping",
    banner_done: "Last run · finished",
    banner_failed: "Last run · failed",
    // Play / Stop: labels and why a button is off (shown as the tooltip and the line under the buttons)
    buttons: {
      play: "Play", stop: "Stop", force: "Force stop", stopping: "stopping…", killing: "killing…",
      tip_play: "start the run (Space)",
      tip_offline: "the UI server is not connected",
      tip_busy: "a run is going: stop it first",
      tip_nothing: "pick a policy first",
      tip_loading: "the policy list is still loading",
      tip_catalog: "the policy list did not load: press retry above",
      tip_stop: "stop after the current step (Space)",
      tip_force: "press again to kill the run now",
    },
    // the step timeline: the state word for screen readers, finished rollouts folded to one line each
    step_states: { todo: "to do", now: "running", done: "done", skip: "skipped", fail: "failed" },
    rollout_line: "Rollout {k}",
    rollouts_earlier: "Rollouts 1 to {k}",
    // the sentence under Play
    plan_head: "{N} {rollouts} in {look}.",
    plan_each: "Each:",
    plan_home: "home",
    plan_start: "start pose",
    plan_random: "eraser at a seeded spot (seed {seed})",
    plan_set: "eraser at the next {SET} start (#0, #1, ... in order)",
    plan_replay: "eraser where demo #{i} had it",
    plan_policy: "policy for up to {T} s",
    plan_hold: "arm holds still for {T} s (baseline)",
    plan_replay_ctl: "demo #{i}'s motion, open loop, up to {T} s",
    next: "Next run",
    // run header, line 3 (and the play card): short words for the settings
    short: {
      rollout: "rollout", rollouts: "rollouts",
      look: { robot: "robot cell", train: "training scene", current: "current look" },
      home: { start: "start pose", deploy: "deploy home" },
      starts: { random: "random spots", set: "{SET} start set" },
      limit: "{T} s limit", seed: "seed {seed}", tap: "tap", tag: "tag {tag}",
      not_arm: "{precision} · {steps} steps · exec_h {exech}",
    },
    // run header, line 2
    no_policy: "no policy: the arm holds still",
    demos: "{n} demos",
    demo: "demo #{i}",
    smoke: "smoke",
    // results table header tooltips
    th: {
      success: "rollouts with a success: lifted ≥ 5 cm, held 1 s, clear of the table",
      rate: "success rate with its 95% Wilson interval",
      grasp: "share of rollouts where both fingers touched the eraser",
      lift: "share of rollouts where the eraser went up",
      fst: "faults (robot PC) / stops (robot PC or policy_deploy) / timeouts",
    },
  },

  // --------------------------------------------------------------------------------- the "?" glossary
  GLOSSARY: [
    ["rollout vs episode", "A rollout is one eval trial. An episode is a training demo (policy_deploy's logs "
      + "still say episode)."],
    ["one eraser", "The sim has one eraser; every entry (also PICK_DOUBLE) runs in that one-eraser scene."],
    ["placement seed", "Picks the eraser spot of every rollout. Same seed = same spots for every policy."],
    ["arm start", "start pose: h then p, as on the real arm. deploy home: h only."],
    ["scene look", "robot cell = like the real lab; training scene = where the demos were recorded; "
      + "current = the older sim look. All with the 30 mm eraser."],
    ["eraser starts", "random = from the seed; start set = the batch's 18 fixed starts #0 to #17 (shared "
      + "demos 0-9 + 8 real-trial poses); rollout 1 uses #0."],
    ["success", "Eraser centre ≥ 5 cm above where it settled, held 1 s, lowest corner clear of the table."],
    ["grasp", "Both fingers touch the eraser. Reported, not required for success."],
    ["lift", "The eraser went up (max lift). lift_without_grasp is flagged by evaltool."],
    ["timeout", "The time limit per rollout ran out without a success."],
    ["fault", "The robot PC stopped the arm (e.g. the tilt gate, a reflex). The sim is restarted if it latched."],
    ["tap", "policy_deploy saves its inputs, planned path and images per step (debugging)."],
    ["replay", "Plays back one recorded demo's motion open loop; the eraser goes where the demo had it."],
    ["hold_still", "Baseline: no policy, the arm holds still."],
    ["prompting", "Behavior-prompting diffusion policy (train config umi_policy_dunetp)."],
    ["plain", "Plain diffusion policy (umi_policy_dunet); the results table's 'diffusion'."],
  ],
};
