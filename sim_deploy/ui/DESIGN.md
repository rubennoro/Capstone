# sim_deploy/ui: design

A local web page to **pick a policy and run sim rollouts on it with a Play button**, watching it live. Same stack as
`teleop_collect/ui` on rnoro/teleop-collect-deploy (aiohttp, one WebSocket, a static page; 127.0.0.1 only).
`python -m sim_deploy.ui` (env `~/envs/sim_deploy`) → open http://127.0.0.1:8770.

The UI **never replaces the sim chain**: it starts `sim_deploy/run.sh` (sim PC + sim Pi + eval driver + the copied
policy_deploy) as a subprocess, and only *watches* the running processes through their existing outputs. It sends no
C3 and is not a C1/C4 client. So what runs from the UI is exactly what runs from the command line.

## 1. Parts (one agent each)

| Part | File(s) | Job |
|---|---|---|
| U1 server | `ui/server.py`, `ui/__main__.py`, `ui/settings.py` | aiohttp app: static page, REST (§2), WebSocket (§3), wires U2/U3/U5 |
| U2 runner | `ui/runner.py` | start/stop one eval run (`run.sh ...`), parse its output into events, follow the eval folder |
| U3 live | `ui/live.py` | live data: camera JPEGs from C0, the policy's 224 views, external-camera render, C2 + ground truth |
| U4 page | `ui/static/index.html`, `app.js`, `style.css` | the page (§4) |
| U5 catalog | `ui/catalog.py` | the list of policies and replay episodes (training demonstrations), and past results |

Shared: `ui/settings.py` (U1 owns; ports/endpoints/paths below as defaults, overridable by `ui/config.yaml`).

## 2. REST (JSON)

| Method, path | Body / reply |
|---|---|
| `GET /api/policies` | `{"policies": [Policy], "replays": [Replay], "builtins": ["hold_still"]}` (U5) |
| `GET /api/evals` | `{"evals": [EvalRow]}`: rows of `sim_evals/index.csv` + links (U5) |
| `POST /api/run` | `RunRequest` → `{"ok": true, "run_id": str}` or `{"ok": false, "error": str}` (409 if one is running) |
| `POST /api/stop` | `{}` → `{"ok": true}`: stops after pressing `n`/`q` cleanly; a second stop kills (U2) |
| `GET /api/status` | the latest `Status` (§3) |
| `POST /api/frames` | `{"eval_rel": <eval folder under sim_evals/>}` → `evaltool frames` → `{"ok": true, "frames": [url]}` |
| `GET /evals/<path>` | static files under `sim_evals/` (summary.html, mp4, images) |

- `Policy`: `{"id": "PICK_TEST1/run_006_2026-11-18", "task": "PICK_TEST1", "run_folder": abs path, "kind":
  "prompting"|"plain"|"unknown", "train_config": str|null, "dataset": str|null, "episodes": int|null, "checkpoint": abs
  path|null, "val_mse": float|null, "label": str (from RUNS.md if present), "last_eval": EvalRow|null}`. Only run folders
  with a checkpoint are runnable (`checkpoint` not null); others are listed greyed out ("not fetched").
- `Replay`: `{"id": "replay:<zarr>#<idx>", "dataset": str, "index": int, "grasp_ok": bool}` (`grasp_ok` from
  `sim_deploy.scene` replay placement: `|height_error_m| < 0.02`).
- `RunRequest`: `{"policy": Policy.id | "hold_still" | Replay.id, "rollouts": int (1..100, default 1), "seed": int,
  "home": "deploy"|"trainstart", "timeout_s": float, "tap": bool, "tag": str, "precision": "fp32"|"bf16"|null,
  "diffusion_steps": int|null, "exec_h": int|null}` (null = the eval driver's / station default). U2 maps it to the
  eval driver's CLI flags (read `sim_deploy/eval/README.md`; flags missing there are passed only if the driver has them).
- `EvalRow`: the `index.csv` columns + `"summary_url"`, `"folder_url"`.

## 3. WebSocket `/ws`

Server → page, JSON text messages (`{"type": ...}`) and binary frames:

| Message | When | Content |
|---|---|---|
| `status` | 5 Hz | `Status`: `{"state": "idle"|"starting"|"running"|"stopping"|"done"|"failed", "run_id", "request": RunRequest, "rollout": n, "rollouts": N, "phase": str (driver/manager phase), "elapsed_s", "outcomes": [str], "last_error": str|null, "log_tail": [str] (last 30 lines)}` (U2) |
| `telemetry` | 10 Hz while the sim runs | `{"t_ns", "mode": C2 mode, "tcp": [x,y,z], "width_m", "eraser": {"pose": [7], "height_m"}, "contacts": {"finger_eraser": bool, "table": bool}, "policy": {"infer_ms", "margin_ms", "skipped"} | null}` (U3) |
| `event` | as they happen | `{"kind": "rollout_start"|"rollout_end"|"run_end"|"error", ...}` (U2), e.g. `rollout_end` carries the outcome dict from `rollouts.jsonl` |
| binary frame | ≤ 10 fps per channel | 1 byte channel id + JPEG bytes. Channels: 1 `external` (640×480 render), 2 `main` (IMX519 as streamed), 3 `ultrawide` (IMX219 fisheye as streamed), 4 `policy` (one 2×2 mosaic, 448×448, of the four 224×224 images the policy gets: main T−50/T, ultrawide T−100/T) |

Page → server: `{"type": "subscribe", "channels": [1,2,3,4]}` (default all). Nothing else (control is REST).

## 4. The page

- Left: **policy picker** grouped by task (PICK_TEST1, PICK_DOUBLE, ...), each row: label, kind badge (prompting /
  plain), dataset episodes, val MSE, last sim result (success k/N); then `hold_still` and replay episodes. Settings:
  rollouts, seed (with dice), home (deploy / trainstart), timeout, tap, advanced (precision, diffusion steps, exec_h,
  tag). Big **▶ Play** and **■ Stop** buttons.
- Centre: the external view large; under it IMX519 and IMX219 as streamed; the policy's 2×2 view.
- Right/bottom: status strip (state, rollout k/N, phase, elapsed, C2 mode, gripper width, eraser height, inference
  ms, margin), outcome chips per rollout, log tail, and the results table (`/api/evals`) with links to summary.html.
- Works at 1280×800 and wider; dark/light from `prefers-color-scheme`. No external CDNs (offline lab laptop).

## 5. Live data sources (U3), all read-only

| Data | Source | Notes |
|---|---|---|
| raw camera JPEGs | C0 SUB `tcp://127.0.0.1:5555` | PUB allows extra subscribers; pass the JPEG bytes through (no re-encode), decimate to 10 fps |
| policy view | the same C0 packets → the copy's `ImagePrep` (`policy_deploy.frames._dataproc.imageprep`, rig `deployment/contracts/rigs/rig_pi_v3_640.yaml`) with the same frame offsets (main T−50 ms/T, UW T−100/T) | the exact code path the policy uses; `policy_deploy` (copy) must be importable in `~/envs/sim_deploy` (U3 installs it editable, no deps change) |
| external view | state SUB `tcp://127.0.0.1:5650` → MuJoCo mirror of `sim_deploy/build/sim_rig/scene.xml`, camera `cam_external`, offscreen EGL | reuse the eval driver's mirror code if it is importable (`sim_deploy/eval/sim_link.py`), else a small copy |
| C2 | SUB `tcp://127.0.0.1:5620`, decode with `deploy_contracts` | mode, O_T_TCP, width |
| ground truth | REQ `tcp://127.0.0.1:5651` `ground_truth` at ≤ 5 Hz | eraser pose/height, contacts |
| policy timing | the running policy_deploy session's `manager/steps.jsonl` (path from U2) | infer ms, margin, skipped |

Every source reconnects by itself and shows "no data" when the sim isn't running.

## 6. Rules

- Code only in `sim_deploy/ui/` (U3 may add `policy_deploy` editable to `~/envs/sim_deploy`; `aiohttp` may be added to
  `~/envs/sim_deploy` and to `sim_deploy/setup_envs.sh`'s `sim` env line).
- Another agent (the integrator) is running the full sim chain on ports 5555/5610/5620/5630/5640/5650/5651 right now:
  **don't start the sim chain or bind those ports**. Test against stubs on other ports (make every endpoint
  configurable). A final live test happens after integration.
- Never write in `~/Capstone-final-deployment`, `~/Capstone`, `~/Capstone-data-work` (read only). No commit/push. No
  test suites; smoke-run what you build.

## U5: as built (`ui/catalog.py`)

- `Catalog(settings)` reads `policy_data_root`, `datasets_root`, `sim_evals`, `model_json` from U1's `Settings`
  (a plain dict or `None` works too; `settings.extra["catalog_cache_dir"]` moves the cache). Methods:
  `policies(replays=True)`, `evals()`, `replays()`, `warm()` (fills every cache; the first replay pass takes ~10 s
  for the 9 packages, then ~5 ms per call). Thread-safe (one lock). Never imports torch (numpy/scipy/yaml only).
- Caching: every call stats the run folders, `checkpoints/`, `job.json`, `RUNS.md`, the zarr files and `index.csv`;
  it rebuilds only what changed. Replay placements and episode counts persist in `sim_deploy/ui/.cache/replays.json`
  (gitignored), keyed per zarr by size + mtime and by a scene key (model.json sha + home TCP + station file +
  `scene/placement.py` + `scene/zarr_lowdim.py`).
- `Policy`: the §2 fields, plus `runnable`, `not_runnable_reason`, `smoke`, `note` (job.json), `train_config_source`,
  `dataset_path`, `status` (the run's STATUS file). `checkpoint` comes from `sim_deploy.eval.target.resolve` itself
  (imported lazily; a copy of its rule, last of `sorted(checkpoints/*.ckpt)`, only if the import fails), so a
  runnable row is one the driver accepts. `train_config`: `run_config/job_info.json` `config_name` >
  `run_config/submit_config.env` last `TRAIN_CONFIG=` (none = prompting, per RUNS.md) > README "Policy config" >
  job.json/train_meta.json keys; not fetched runs hold only job.json, so they show `kind: "unknown"` (their label
  says the type). Note: the driver's `Target.kind` says `diffusion` where the catalog says `plain`, and falls back
  to `diffusion` where the catalog says `unknown`; `index.csv` rows carry the driver's word.
- `label`: from `<task>/RUNS.md` (tables: run names in columns headed run/folder, including `run_A to run_B`
  ranges; prose: the paragraph's lead-in and the parenthesis after the name), e.g. "Behavior prompting · pick1-v1 ·
  30 eps", "Plain diffusion · pick1-ep15-v1 · 15 eps · smoke", "Ignore these folders · ..."; else
  "<dataset> · <job.json note>"; else the dataset. Since 2026-11-22 there is no per-task RUNS.md: the notes moved to
  one `~/Capstone/data/RUNS.md` for all tasks, which the catalog does not read, so every label is the
  "<dataset> · <note>" fallback today (kept as is, user decision 2026-11-23). `episodes` = the dataset package's
  episode count (else "N eps" in the note). `last_eval` = the newest `index.csv` row with `task/run_folder` = the id.
- `Replay`: the §2 fields, plus `group` (the zarr's grandparent folder: PICK_TEST1, multi), `dataset_path`, `name`,
  `height_error_m`, `error` (e.g. no grasp found), `last_eval` (row `_replay/<group>/<dataset>_epNNN`). The id uses
  the zarr's absolute path. `/api/policies` also has `builtin_last_eval: {"hold_still": row|null}`.
- `EvalRow`: the `index.csv` columns with numbers parsed (empty = null), newest first (by `date`, then file order),
  `summary_url` = `/evals/<eval_dir>/summary.html`, `folder_url` = `/evals/<eval_dir>/` (URL-quoted).
- CLI: `python -m sim_deploy.ui.catalog [--json] [--no-replays] [--data-root D] [--datasets-root D] [--sim-evals D]`.
- Smoke 2026-11-20 against the real folders: 36 run folders, 5 runnable (PICK_TEST1 run_006/012/014/016/020, all
  prompting, val MSE 0.00020 to 0.00084); the rest not fetched (PICK_TEST1 plain-diffusion runs 021 to 031, all
  PICK_DOUBLE runs, old duplicates). Replays: 282 episodes in 9 packages (subsets repeat episodes), 40 grasp_ok
  (pick1-v1: 2, 3, 8; pick1-double-v2: 2, 3, 8, 44, 45, 48, 49). `index.csv` does not exist yet (0 eval rows).

## U2: as built (2026-11-20)

`ui/runner.py`: `Runner(settings, resolve_policy=None)` (a `Settings` or a dict; uses `repo`, `run_sh`, `sim_evals`,
`policy_data_root`, `log_tail`, and from `extra`: `stop_kill_after_s` (150), `driver_match` (`sim_deploy.eval`),
`runner_poll_s` (0.25)). `start(req) -> run_id` (`RunnerBusy` (RuntimeError) if busy, `InvalidRequest` (ValueError)
if malformed; uses `req["target"]` when the server sends it, else resolves `policy`: `hold_still`, `replay:<zarr>#<i>`,
an absolute run folder or `<task>/<run>` under `policy_data_root`; a run folder needs `checkpoints/*.ckpt`),
`stop() -> {"ok", "action": none|graceful|kill, "detail"}`, `status()`, `events()` (one `queue.Queue`),
`drain_events()`, `session_dir()`, `close()` (kill if still running).

| RunRequest | run.sh / driver flag |
|---|---|
| `target` / `policy` | `--policy <run folder \| hold_still \| replay:…>` |
| `rollouts` (1..100, default 1) | `--rollouts N` (always passed; the driver's default is 10) |
| `seed` (default 0) | `--seed S` |
| `tag` (default `ui`; `[A-Za-z0-9._-]{1,40}`) | `--tag T` |
| `timeout_s` (default 60) | `--timeout` |
| `tap` | `--tap` |
| `precision` / `diffusion_steps` / `exec_h` (null = station) | `--precision` / `--diffusion-steps` / `--exec-h` |
| `home` | **none**: the driver has no home option (every rollout starts with `h`); `trainstart` adds a line to `Status.notes` |
| `dry_run` (extra, optional) | `--dry-run` |

**Progress.** Eval folder from "[eval] eval folder <path>" (else the newest `sim_evals/**/eval.yaml` written after the
start whose folder name ends in the tag). `rollouts.jsonl` complete lines → `rollout_end` (`n`, `rollout` = n+1,
`outcome`, `success`, `record`, `source`); the "[eval] rollout N: …" stdout line only if the file lacks N 2 s later.
`<eval>/policy_deploy.log` (the manager's raw pty output; patterns imported from `eval/pd_session.py`): status heads →
`phase` (`idle`, `homing 0.4 s`, `ep 2 running 1.0 s`), "episode <id> started" → `rollout_start` (`n`,
`pd_rollout_id`, `session_dir`, `seed_key`). `metrics.json` → `Status.metrics` and `run_end` (`state`, `exit_code`,
`stopped_by_user`, `killed`, `eval_dir`, `eval_rel` (under sim_evals, for /evals/ links), `metrics`, `outcomes`,
`log_file`, `logs_dir`, `last_error`). "[eval] error: …" / "policy_deploy did not reach idle" → `error`. run.sh lines
→ `phase` before the driver. Session folder from "[eval] policy_deploy ready (session …)" (updated when the driver
restarts policy_deploy). Extra Status keys: `eval_dir`, `logs_dir` (run.sh's), `log_file`
(`sim_evals/_logs/ui/<run_id>.log`: all of run.sh's stdout+stderr, the events, the stops), `session_dir`, `metrics`,
`exit_code`, `stopped_by_user`, `notes`, `pid`, `command`, `pd_status`. States: exit 0 → `done`; a ui stop → `done` +
`stopped_by_user`; a kill → `failed`; other non-zero → `failed`, `last_error` = the last bad-looking tail line + exit.

**Stop.** 1st: SIGINT to the driver process only (found under run.sh's group by its argv `-m sim_deploy.eval`): it
stops the rollout (`s`), quits policy_deploy, writes rollouts.jsonl / metrics.json / summary.html / the index row,
exits 1, and run.sh's EXIT trap stops the sim PC and sim Pi. Before the driver runs: SIGINT to the group (as Ctrl-C).
The rollout in progress gets a synthetic `rollout_end` `interrupted` (`synthetic: true`). 2nd stop, or
`stop_kill_after_s` after the 1st: SIGKILL to every descendant of run.sh (by ppid, so also policy_deploy in pexpect's
own session) and the group.

**Smoke** (fake `run.sh` + fake driver with the real output formats, in the scratchpad, no sim, no ports): validation
rejects; 3 rollouts → 3 starts/3 ends/run_end, metrics, `trainstart` note, flags as above; a stop in rollout 2 of 5 →
`done`, `['success', 'interrupted']`, metrics written by the driver, no processes left; driver error → `failed` with its
line; a driver ignoring SIGINT → 2nd stop kills it and its own-session child; missing "eval folder" line → found by
watching; port busy → `failed` with run.sh's line; stop during sim bring-up → `done` in 0.2 s; auto-kill after 3 s.
Not yet run against the real chain.

**For the integrator** (not changed, eval/ and run.sh are yours): (1) on SIGINT the driver drops the rollout in
progress (it is not in rollouts.jsonl / metrics; the ui shows it as `interrupted`); recording it as a failure would
need the driver. (2) A terminal Ctrl-C (group SIGINT) also kills run.sh's `tee`, so the driver's next `log()` raises
BrokenPipeError in the KeyboardInterrupt handler and `_collect` / `_write_results` are skipped (`tee -i` would avoid
it); the ui does not hit this (driver-only SIGINT). (3) No home choice in the driver (`RunRequest.home`). (4) A clean
stop can take ~30 s (`pd.quit`) + collection + up to 5 s per sim process; U1's `shutdown_clean_s` 30 may cut it short.

## U3: as built (2026-11-20, `ui/live.py`)

**API.** `Live(settings)`: `start()`, `stop()` (joins the threads, < 0.2 s), `latest_frames() -> {1..4: jpeg}`
(channels with no frame yet absent; the bytes object changes only with a new frame), `frame_seq() -> {1..4: int}`
(frames produced per channel), `telemetry() -> dict`, `set_session_dir(path | None)` (the session folder, its
`manager/` folder or `steps.jsonl` itself). Getters only copy under a lock; nothing blocks the caller.
Settings: U1's `Settings` (attributes), then its `extra` (config.yaml keys U1 does not know), then a plain mapping
also works. Read: `c0_endpoint`, `state_endpoint`, `c2_endpoint`, `scene_endpoint`, `scene_xml`, `rig_yaml`,
`external_camera`, `external_width/height`, `jpeg_quality`, `ground_truth_hz`, `frame_fps` (caps every channel); via
`extra`: `camera_fps` 10, `policy_fps` 5, `external_fps` 10, `external_shadows` true, `external_shadow_size` 2048,
`scene_timeout_s` 0.25, `scene_retry_s` 1.0, `c2_hz` 20, `steps_poll_s` 0.2, `stale_s` 1.0, `table_top_z_m` 0.0 (P3/P4; was 0.050),
`frames_config` (the copy's `config/frames_process.yaml`). All defaults in `live.DEFAULTS`.

**Telemetry** (§3 plus extras): `t_ns`, `mode`, `tcp` [x, y, z] (C2 `O_T_TCP` translation), `width_m` (C2, training
units), `c2_error`, `eraser` {`pose` [7], `height_m` = centre z − `table_top_z_m` (≈ 0.006 resting flat)},
`contacts` {`finger_eraser` (a `pad_*`/`finger_holder_*` geom on the eraser, the eval driver's rule), `both_fingers`
(l and r: the driver's "grasp"), `table` (anything but the eraser on the table)}, `finger_gap_m`, `policy`
{`infer_ms` (= `policy_ms`), `margin_ms`, `skipped` (steps not sent in this rollout), `steps`, `last_skip_reason`,
`cycle_ms`, `image_age_ms`, `frames_ms`, `episode_id`, `chunk_id`} or null, `sources` {c0, state, external,
policy_view, c2, scene, steps: {`ok` (a message within `stale_s`), `age_s`, `rate_hz`, `n`, `error`}; policy_view
also `reason` (why no mosaic: the selector's message)}. C2 and ground-truth fields are null when their source is
stale ("no data"); `policy` keeps the last step (steps.jsonl is quiet between rollouts; `sources.steps.age_s` tells).
Telemetry is never empty, so U1 sends it even with no sim: the page should key "no data" on `sources.*.ok`.

**Sources** (one daemon thread each; a thread that raises is restarted after 1 s with the error in `sources`):
- C0 SUB (RCVHWM 60, drained): `PiStreamPacket.from_parts` (the copy); channels 2/3 = the packet's JPEGs as sent,
  decimated to `camera_fps`. Every packet also feeds the policy view.
- Policy view (channel 4, 448×448: top main T−50 ms | T, bottom ultrawide T−100 ms | T): the copy's `PairRule`,
  `BufferedPacket`/`PacketBuffer`, `PacketSelector` with `frames_process.yaml`'s spacings 50/100 ms, tolerance 2
  frames, pair window 60/25 ms, reuse 2, keep 1 s, anchored on the newest packet; then `cv2.imdecode` + BGR→RGB (as
  `JpegDecoder`) and `ImagePrep.prep_main` / `prep_ultrawide` on rig `rig_pi_v3_640.yaml`. Simplified: packet times
  are the Pi-clock main mid-exposures (no clock map; the pick uses only differences), no camera-health gate, and only
  the picked packets are decoded (buffered packets carry a 1×1 marker image; 224 images cached per packet).
- State SUB (CONFLATE) → a MuJoCo mirror of `scene_xml` (newest qpos, no interpolation), `cam_external`, groups 0-3,
  at `external_fps`. The model is loaded when the first state arrives, its `model.json` sha compared with the
  message's (mismatch: error in `sources.external`, retried every 5 s; a sha change mid-run reloads). EGL on the NVIDIA
  GPU (E5's fix, applied at import: **import ui.live before anything imports mujoco**); offscreen buffer cut to
  640×480, shadow map 2048. `sources.state.rate_hz` is the sampled rate (≈ 20 Hz), not the 200 Hz export.
- C2 SUB (CONFLATE) decoded with `deploy_contracts.decode(..., ArmState)` at ≤ `c2_hz`.
- Scene REQ `ground_truth` at `ground_truth_hz`, 250 ms timeout; on a timeout the REQ socket is closed and reopened
  and the next request waits `scene_retry_s` (so with no sim PC it is 1 connect per s).
- steps.jsonl: re-read from the start on a new session folder, then followed (truncation / replacement reopens it).

**Env.** The copy's `deployment/policy_deploy` installed editable in `~/envs/sim_deploy` with `--no-deps` (its
imports work there: numpy 2.2.6, opencv-headless 4.12.0.88, scipy 1.15.3, PyYAML 6.0.3 = the light env's pins);
added to `setup_envs.sh`'s `sim()` with an import check. Side effect: `~/envs/sim_deploy/bin/policy_deploy` now exists
(the eval driver calls `~/envs/sim_policy_deploy/bin/policy_deploy` by full path, so it is unaffected).

**Smoke (2026-11-20, stubs on 5755 C0 / 5750 state / 5720 C2 / 5751 scene, no sim chain running; scripts in the
session scratchpad `u3/`).** C0 stub: 18 real sim 640×480 frame pairs (E5 `raw/`) as `pi_stream/v1` at 30 Hz with the
sim Pi's timing; state: E5's `state_stub`; C2: ArmState 100 Hz; scene: a REP answering `ground_truth`; a steps.jsonl
writer at 2.5 Hz. Through U1's `Settings.from_mapping`:
- no sim → every source `ok: false`, telemetry fields null, no frames, scene error "no reply within 250 ms";
- data → channels 1/2/3/4 at 10.0 / 10.1 / 10.1 / 5.0 fps, sizes 640×480 / 640×480 / 640×480 / 448×448; C2 mode, tcp,
  width; eraser pose/height; contacts; policy timing and per-rollout skip count; scene 5.0 Hz;
- stubs killed → all stale within 1 s; restarted → every source back by itself in < 6 s (no Live restart);
- `stop()` 0.05-0.13 s.
- CPU (whole Live process, 20-core laptop): 38 % of one core with the defaults (policy view 5 fps); policy view at
  10 fps → 77 % (one mosaic ≈ 50 ms CPU: 4 JPEG decodes + 2 letterboxes + 2 fisheye remaps); external render ≈ 10 %
  at 10 fps with shadows; C0 pass-through + C2 + scene + steps ≈ 7 %. VRAM 70 MiB (nvidia-smi, type G).
- Not yet run against the real sim chain (it was not running; the 5651 REQ must not be tried against the integrator's
  chain anyway). Samples: `u3/ch1_external.jpg`, `ch2_main.jpg`, `ch3_ultrawide.jpg`, `ch4_policy.jpg`.

## U1: as built (2026-11-20, `ui/server.py`, `__main__.py`, `settings.py`, `config.yaml`, `_stubs.py`)

- **Run:** `cd ~/Capstone-main-sim && unset PYTHONPATH && ~/envs/sim_deploy/bin/python -m sim_deploy.ui [--port 8770]
  [--stub [runner,live,catalog]] [--config FILE] [-v]` → http://127.0.0.1:8770. `--stub` alone fakes all three parts
  (`_stubs.py`: fake runs with phases/outcomes/events, 4 moving-dot JPEG channels at 10 fps, telemetry while a fake
  session runs, fake policies/replays/evals); `--stub live` etc. fakes only those. A missing part without `--stub`:
  `FATAL sim_deploy.ui: ui/<part>.py is missing (run with --stub <part> ...)`, exit 1.
- **Env:** `aiohttp==3.10.10` (+ aiohappyeyeballs 2.7.1, aiosignal 1.4.0, async-timeout 4.0.3, attrs 26.1.0,
  frozenlist 1.8.0, idna 3.20, multidict 6.9.1, propcache 0.5.4, yarl 1.25.1: the versions of
  `~/envs/teleop_collect`) in `~/envs/sim_deploy` and in `setup_envs.sh`'s `SIM_PINS`.
- **Settings** (`settings.py`, frozen dataclass; `config.yaml` has the same flat keys; unknown keys →
  `settings.extra` with a warning): `http_port` 8770, `status_hz` 5, `telemetry_hz` 10, `frame_fps` 10, `log_tail` 30,
  `shutdown_clean_s` 180 (the runner auto-kills 150 s after a first stop), `shutdown_kill_s` 5; paths `repo`,
  `run_sh`, `python`, `sim_evals`, `policy_data_root`, `datasets_root`, `policy_station`, `scene_xml`, `model_json`,
  `rig_yaml`; live `c0_endpoint`, `c2_endpoint`, `state_endpoint`, `scene_endpoint`, `ground_truth_hz`,
  `external_camera`, `external_width/height`, `jpeg_quality` 80; `stub` (bool). `http_port` may not be a sim port.
  Every part gets this object: `Runner(settings)`, `Live(settings)`, `Catalog(settings)`.
- **Adapters** (server.py docstring has the full expected API): sync methods run on worker threads
  (`asyncio.to_thread`), `async def` / awaitables are awaited. `runner.events()` is called once and may be a
  `queue.Queue` (used: U2), an `asyncio.Queue`, an (async) iterator, or a list (then polled with the status).
  `live.latest_frames()` → `{1..4 or name: jpeg | (seq, jpeg)}` polled at `frame_fps` (a frame is sent once: same
  bytes object or same seq is skipped; equal bytes in a new object are sent since U6, see "External view (X)"), or `live.subscribe(callback)`. Catalog functions get the settings only
  if they have a required parameter (`policies(replays=True)` is called bare). `runner.session_dir()` →
  `live.set_session_dir()` on every change (checked at status_hz). `runner.close()` (or `shutdown()`) last on exit.
- **REST:** as §2. `POST /api/run`: the body is validated in full (all problems in one `error`, joined by `; `, and as
  a `problems` list): unknown keys are refused; `policy` must be a catalog Policy id with a checkpoint (else "not
  fetched"), `hold_still`, a catalog replay id or `replay:<existing zarr>#<i>`; `rollouts` 1..100 (default 1),
  `seed` 0..2³²−1 (0), `home` deploy|trainstart (deploy), `timeout_s` 1..3600 (60), `tap` bool (false), `tag`
  `[A-Za-z0-9][A-Za-z0-9_.-]{0,59}` (`ui`), `precision` fp32|bf16|null, `diffusion_steps` / `exec_h` 1..1000|null.
  The runner gets the request with every key filled plus `"target"` (the driver's `--policy`: run folder |
  hold_still | replay:...). 400 on a bad request (also the runner's ValueError), 409 when a run is
  starting/running/stopping (checked under a lock, and again if `start` raises while the runner reports busy), 500 when
  a part fails. `POST /api/stop` → `{"ok": true, "state": ...}` (always calls `runner.stop()`; a second stop kills).
  `GET /api/status` asks the runner fresh. JSON NaN/inf → null; Paths → strings.
- **`/evals/<path>`:** files under `settings.sim_evals` (resolved; anything resolving outside, including `..`,
  absolute and %-encoded paths and symlinks pointing out → 403); folders → a small HTML listing (`/evals/x` →
  302 `/evals/x/`); `.jsonl .yaml .log .csv .txt .md` served as text/plain, `.json` as JSON so the browser shows them;
  Range requests work (mp4 seeking). `/static/*` = `ui/static/`, `/` = `static/index.html` (no-store).
- **Security:** binds 127.0.0.1 only; teleop_collect's Host check (DNS rebinding) and Origin check (must be
  `http://127.0.0.1:<port>` or `http://localhost:<port>`; none at all, e.g. curl, is allowed).
- **WebSocket:** on connect the page gets the current status and the newest frame of each channel. Per page: one slot
  each for status, telemetry and each channel (newest wins; overwriting an unsent frame counts as a drop) and a
  queue for events (max 1000). The writer sends events first, then status, telemetry, frames. The socket's kernel
  send buffer is fixed at 256 KiB so a page that stops reading makes the writer wait and its frames are dropped
  rather than queued (smoke: a page idle 6 s got ~3.5 s of frames on resume, 132 dropped; before the fix the kernel
  had buffered all 6 s). Telemetry is sent whenever `live.telemetry()` returns a non-empty dict (U3 returns one
  with nulls and `sources` when there is no data). Page → server: `{"type": "subscribe", "channels": [...]}` (1..4 or
  names); anything else → `{"type": "error", "text": ...}` to that page. Events go out as `{"type": "event", ...}`.
- **Shutdown:** Ctrl-C / SIGTERM: if a run is busy, one clean `stop()` (none if it is already stopping, since a
  second would kill), wait up to `shutdown_clean_s` for a non-busy state, then a kill `stop()`; a second Ctrl-C kills
  at once. Then `live.stop()`, pages closed (1001), `runner.close()`, server cleanup; exit 0.
- **Smoke (2026-11-20, port 8791, nothing on the sim ports):** `--stub`: every endpoint with curl (all bad-request
  cases, not-fetched / unknown policy / missing replay, run → 200, second run → 409, stop, stop-stop → killed,
  /evals files, listing, redirect, Range 206, 8 path-escape attempts → 403, Host/Origin → 403); a Python WS client:
  ~8 fps per channel, status ~5 Hz, subscribe [1] → only channel 1, bad messages → errors, events
  rollout_start/rollout_end/run_end, telemetry 10 Hz while running; shutdown idle, during a run, after a page stop,
  and double Ctrl-C. Real U2+U3+U5 with live endpoints moved to unused ports (a config with ports 15555/15620/15650/15651): policies (36, 5
  runnable, 282 replays), evals, status, an unknown-policy run → 400, WS status + telemetry ("no data"). Not run
  against the sim chain or with a real run.

## U4: as built

`ui/static/index.html`, `app.js`, `style.css`: vanilla JS, no build, no external files. The page loads them from
`/static/...` (absolute paths) and opens `ws(s)://<host>/ws`.

- **Layout** (1280×800 up, `html` min-width 1024): left = picker, settings, Play/Stop; centre = external view large,
  then main / ultrawide / policy 2×2 (with the quadrant legend main T−50, main T / UW T−100, UW T); right = status
  strip, notes, `last_error`, rollout chips, log tail; bottom = results table. Light/dark from
  `prefers-color-scheme` (CSS variables, same palette as teleop_collect/ui).
- **Picker**: radio rows grouped by task (sorted; runnable first), kind badge, label (else run name), run name, dataset
  episodes, val MSE, last sim k/N; tooltips carry id, dataset, config, checkpoint or `not_runnable_reason`, note.
  `runnable` is used when present (else `checkpoint != null`); non-runnable rows are greyed and can't be picked.
  `smoke: true` rows fold under "smoke / old (n)" per task. Then `built-in` (hold_still + `builtin_last_eval`), then
  replays grouped by `group`/dataset, grasp_ok first; non-grasp_ok hidden behind one "show all" toggle. A filter box
  searches everything (folds ignored while filtering). Selection, settings and fold state persist in localStorage.
- **Settings → RunRequest**: rollouts (1..100), seed (+ dice 0..9999), home, timeout_s, tap; advanced: precision
  (""→null), diffusion steps / exec_h (blank→null), tag (blank→"ui"). Checked client-side, then the server's
  `error` / `problems` list is shown inline under the buttons.
- **Play/Stop**: Play disabled while disconnected, busy (starting/running/stopping), a POST in flight, or nothing
  picked. Stop: first press POSTs /api/stop and the button turns "stopping… ■ Force"; the next press POSTs again
  (the server's kill). Space = Play / Stop unless focus is in an input, select, textarea or summary (on a focused
  button the native click is suppressed so one press = one action).
- **WebSocket**: sends `subscribe [1,2,3,4]` on open, reconnects every 1 s (banner while down). Accepts payloads at
  the top level next to `type` (as the server sends) or nested under the same key; also `{"type":"error","text"}`.
  Binary: byte 0 = channel, rest JPEG → `createImageBitmap` (object URL + `<img>` fallback, URL revoked after
  drawing) → canvas; one decode at a time per channel, later frames overwrite a single pending slot (newest only).
  A panel with no frame for 3 s shows "no data" (dimmed last frame) plus the source's `reason`/`error`.
- **Telemetry**: "no data" per field from `telemetry.sources.<name>.ok` (c2 → mode/width/c2_error; scene → eraser
  height, contacts incl. `both_fingers`, `finger_gap_m`; steps → infer/cycle ms, margin (red when < 0), steps /
  skipped / `last_skip_reason`, image age); without `sources` the message age decides. The header "sim" badge is
  live when any source is ok (tooltip lists every source).
- **Status**: `state` badge, rollout as `rollout/rollouts` (the runner's 1-based `rollout`), phase, elapsed,
  `notes` lines, `last_error` box, `log_tail`. Chips: one per rollout from `status.outcomes` (current one pulses,
  rest dashed); tooltips add grasped/lifted/max lift/note/fault from `rollout_end.record`. Events of another run_id
  are ignored. After a run: "k/N success" + summary.html link from `run_end.eval_rel` (else the newest new
  index.csv row for that policy+tag); `/api/evals` and `/api/policies` reload on `run_end` and on busy→done/failed.
- **Results table**: `/api/evals` newest first, rows of the selected policy highlighted, an "only the selected
  policy" filter, rate bar + Wilson CI, grasp/lift, f/s/t, summary / folder links (new tab).
- Smoke-tested in headless Chromium (playwright-core) against U4's mock (port 8771) and against
  `python -m sim_deploy.ui --stub` (port 8793): idle, starting, running, stopping, force-stop, done, failed, REST
  error with problems list, folds, Space in a field, dark mode. Not run against the real sim chain.

## P2: start pose (2026-11-20)

Supersedes the `home` rows above (§2 RunRequest, U1's validation, U2's table): `RunRequest.home` = `"start"` (default;
the page's "start pose (p)": every rollout `h`, then the PC's `p`, then `c`) | `"deploy"` ("deploy home (h only)").
An old request or a saved page setting with `"trainstart"` is taken as `"start"` (server `HOME_ALIASES`, runner
`HOME_ALIASES`, app.js `restore()` falls back to `start` for any value the select no longer offers). The runner passes
`--home start|deploy` to `run.sh` (which starts the sim PC to match) and no longer adds the "no home option" note.

`GET /api/start_pose` (U1, the one server addition): `sim_deploy.scene.start_pose.show()` trimmed to `set`, `file`,
`source`, `text` (one line: TCP, height above the table, tilt, heading), `q_start_rad`, `tcp_O_m`, `tilt_deg`,
`heading_words`, `height_above_table_m`, `ok`, `checks` {name: {ok, detail}}, `vs_demos`, `region`, `move_time_s`; run
on a worker thread and cached until `stations/start_pose.yaml` or `model.json` changes (the first call loads the
MuJoCo model: ~10 s). The page (U4) loads it at start and shows it under the settings for `start` (red when unset or a
check fails; the tooltip has the file, source, q and the demos comparison; click reloads); for `deploy` a fixed line.
U5: replay `grasp_ok` is carried from the start pose when one is set (cache key includes `start_pose.yaml`,
`control_sim.yaml`, `scene/start_pose.py`). Smoke: `--stub` on :8794 (DESIGN.md "P2").

## W4: look and the arm's policy settings (2026-11-21)

Supersedes the `precision` / `diffusion_steps` / `exec_h` "null = station" rows above (§2, U1, U2) and adds `look`.

- `RunRequest.look` = `"robot"` (default: the real robot cell, `stations/sim_rig_look_robot.yaml`) | `"train"` (the
  training scene, `sim_rig_look_train.yaml`) | `"current"` (E7's look, `sim_rig.yaml`); every look has W0's 30 mm wide
  eraser. The runner passes `--look <look>` to `run.sh`, which picks the station (`sim_deploy/looks.py`), builds
  `build/<station id>/` if missing or older than the station, starts the sim PC on that build (an overlay
  `build/look_<look>.sim_pc.yaml` with `sim.model` / `sim.model_json`) and the sim Pi on that station, and passes
  `--look` and `--model` to the driver. The driver writes `look: {name, station, model_dir}` into `eval.yaml`, a `look`
  column into `sim_evals/index.csv` (older rows empty) and a "look" row into `summary.html`.
- `precision` / `diffusion_steps` / `exec_h` default to **bf16 / 8 / 12** (the arm's settings) in the server, the
  runner and the driver (`--precision bf16 --diffusion-steps 8 --exec-h 12` are the driver's defaults, so `run.sh`
  without flags runs them too); null in a request = the default. `eval.yaml` `station_policy` / `station_manager` and
  the summary's "policy settings" row show what ran.
- The page: a "look" select under "home" (default robot cell); advanced shows precision bf16, steps 8, exec_h 12
  prefilled; the selected-box line shows `look ...` and the settings; the results table has a `look` column. Saved
  page settings from before this change (no `settingsVersion: 2`) do not restore look / precision / steps / exec_h.
- Live (U3): the external camera renders the build whose `model.json` sha matches the state export (`build/*/`), so
  it follows the run's look.
- `GET /api/start_pose` adds `vs_git` (`start_pose.compare_git()`: `stations/start_pose.yaml` vs
  `rnoro/final-deployment`'s `station.yaml` q_start_rad, read only, not cached); the page appends "= the real arm's"
  or the warning and turns red when they differ. `run.sh` prints the same line at each start (a framed WARNING when
  they differ; nothing is changed: `python -m sim_deploy.scene start-pose --from-git` adopts it).
- Eval tool link (2026-11-22): the runner starts `run.sh` with env `SIM_DEPLOY_LAUNCHER=ui` (eval.yaml `launcher: ui`,
  `evaltool/SCHEMA.md`). After a run with a `rollouts.jsonl`, it runs `python -m sim_deploy.evaltool score <eval>` in
  a thread (the UI stays usable; errors are shown, never fatal) and puts `Status.score` = `{state: scoring|done|error,
  eval_rel, success [k, n], grasp [k, n], max_lift_m (max), min_z_m (min, fingers/rig else TCP), min_z_of, invalid,
  error}` (+ a `score` event). The run result line shows it, a score.json link and a "frames" button
  (`POST /api/frames` → `evaltool frames`, then one link per rollout's frames.jpg). Scoring stays the separate tool.

## U6: redesign (2026-11-23)

### External view (X)

- Cause of "no data": with the robot / training looks the scene has a visual-only backdrop geom `background`
  (contype = conaffinity = 0) around the rig camera's view; `cam_external` sits outside it, so every render was the
  same flat grey frame, and the server's content dedupe sent it once, then the page marked it stale after 3 s.
- `live.py` `_run_external` hides it right after loading the model with `eval/sim_link.hide_backdrop(m)` (alpha 0,
  material -1, only when the geom exists and is visual-only), the same helper the driver's video now uses (behaviour
  unchanged there). While the model compiles, the external source's error reads
  `loading <look folder> (MuJoCo compile, ~5-10 s)`; it is cleared on success as before.
- `server.py` `_offer_frame` (no-seq frames): skips a frame only when it is the same bytes object as the last one
  sent (Live hands back the same object until a new frame replaces it, on all four channels); equal bytes in a new
  object are sent, so a static scene keeps updating. Content is still compared for bytearray / memoryview values
  (copied to bytes there, so identity cannot hold).

### Catalog fields (C)

`ui/catalog.py` (+ `StubCatalog` in `_stubs.py` with the same fields). Existing fields and values are unchanged except
`kind` for a non-empty unrecognised train_config (now `other`, was `unknown`).

- `kind_of`: `umi_policy_dunetp` → prompting, `umi_policy_dunet` → plain, any other non-empty config → `other`, none →
  `unknown`. Precedence of the config unchanged (job_info.json > submit_config.env > README > job.json/train_meta);
  new last fallback when none is found: `run_config/config.yaml` `use_prompting: true|false` (an indented key; one
  value only, else ignored) gives the kind alone (`train_config` stays null, `train_config_source` and `kind_source`
  = "run_config/config.yaml use_prompting"). Today it never fires: every run with a config.yaml also has job_info.json.
  Names are never parsed for meaning.
- package.json: `<data_root>/<task>/package/<dataset>/package.json` (a symlinked folder for the umi packages, a real
  folder for teleop), else the zarr's folder; a replay reads its zarr's folder. `_package_info` (type-checked, never
  raises, any failure → null): `recordings` [{label, episodes}] from `totals.per_label`, else counted from
  `episodes[].label`, else `labels` with null counts; `data_source` = `source.kind` verbatim (teleop:
  "teleop_collect"), else "dataproc" when `dataproc_version` is present, else null; `rig`; `label_dir`; `episodes`
  (`totals.episodes`); `names` {pi_name: label}.
- Policy adds: `type: "policy"`, `kind_source`, `recordings`, `data_source`, `rig`, `cluster_label` (job.json `label`,
  else summary.json `task`: "multi" for PICK_DOUBLE), `package_json`, `starts_set` (the resolved run folder is in
  `eval/start_sets/w3.yaml` `checkpoints[].run_folder`, read with `eval/start_set.load`; missing file → false: 14 runs
  today). `episodes` falls back to the package's `totals.episodes` before the note, so the teleop runs now show 8 / 16
  / 20 / 24 (their packages are outside `datasets_root`).
- Replay adds: `type: "replay"`, `kind: "replay"`, `recording` (its `name` → package `episodes[].pi_name` → label, null
  if not found; all 282 found today), `dataset_recordings`, `data_source`. Not stored in `.cache/replays.json` (read
  per rebuild).
- Top level adds `builtin_items: [{"id": "hold_still", "type": "builtin", "kind": "hold_still"}]` (`builtins` kept) and
  `start_set: {"name": "w3", "file": "sim_deploy/eval/start_sets/w3.yaml", "n_starts": 18, "needs_home": "start"}`
  (null when the file is missing or unreadable).
- Caching: the policy signature adds each task's `package/` mtime, every `package/<dataset>/package.json` mtime and the
  start-set file's mtime; the replay signature adds each zarr's `package.json` mtime.
- CLI table: a `data` column (the data source when not dataproc, e.g. `teleop_collect`, else `PICK_TEST1 30+PICK_TEST2
  32`), the same on each replay dataset line, and a `start set:` line.

### Step signals + starts (B)

- Driver (`eval/driver.py` `_rollout()`, `log()`; no behaviour change): `[eval] step n home` (before h),
  `step n start_pose` (home start, before p), `step n place: x X y Y (replay|set|random)` (before set_object),
  `step n control` (before c), `step n record: <outcome>` (watch loop ended, before y/n; `ended_by_policy_deploy`
  already mapped to stopped / fail). n = the 0-based index. None start `[eval] rollout` (batch.py, start_sweep.py).
- `Status.step` = `{"key", "rollout" (1-based or null), "since_s" (status elapsed at entry, kept when only the detail
  changes), "detail"}` or null; event `{"kind": "step", "run_id", "t", "key", "rollout", "since_s", "detail"}` on
  every change (the server sends it as `{"type": "event", "kind": "step", ...}`, like every runner event). Keys:
  `prepare` (Play), `sim_pc` / `sim_pi` / `policy` (RUNSH_PHASES, now `(regex, phase, step key)`; policy_deploy heads
  while still in bring-up also give `policy`), `home` .. `record` (RE_STEP), `wrapup` (`[eval] interrupted`,
  `k/n successes`), `sim_stop` (`[run.sh] stopped`). While stopping only record / wrapup / sim_stop move it.
- `status.rollout` = n+1 at `step n home` (set, so a rerun after a restart moves it back); a `not_run` outcome no
  longer advances it. New phases `restarting the sim (fault)` / `retrying the sim start` (run.sh's
  `the sim PC latched a fault: restarting the sim (k) ...` / `sim start failed: retrying in W s (i)`), step `sim_pc`
  with detail `restart after a fault (k)` / `retry i in W s`, and a `notes` line each ("the sim PC faulted; restarted
  the sim and continued (k)", "the sim did not start; retrying the start in W s (i)"). Old phase strings unchanged.
- Restart (exit 3 → `--resume`): the first driver call writes `not_run` for the rest; the runner now replaces a
  `not_run` outcome when the resumed driver's line for that n arrives (it used to keep `not_run` forever), the
  `rollout_start` index is the current step's rollout (fallback: the first n neither started nor done, `not_run`
  counting as not done), and `_finish` marks started rollouts still `not_run` as `interrupted`.
- `RunRequest.starts`: `random` (default, nothing passed) | `set` (`--starts set`). server.py `validate_request`
  takes the key (choice); the runner's `_check_start_set` refuses (InvalidRequest → 400) a replay, home ≠ start,
  rollouts > the file's starts (18), or a run folder not in `eval/start_sets/w3.yaml` `checkpoints[].run_folder`
  (path or resolved path equal; `eval/start_set.load`); hold_still is allowed. Normalized request carries `starts`.
- `StubRunner`: the real phase strings (`starting run.sh` .. `ep N running X s` .. `sim stopped`), the same step
  keys / events (deduplicated), outcomes success / timeout / fail / fault (no `dropped`), rollout_start /
  rollout_end with 0-based `n` and the real shape (`outcome` str + `record` dict), `starts` checked (no file read).

### Page (A, D–F)

`ui/static/`: `help.js` (one global `HELP`: FIELDS, STEPS, OUTCOMES, KINDS, TYPES, PAGE, GLOSSARY; every word the
page shows, `{name}` placeholders; it wins over the fallback text in index.html), `steps.js` (pure
`computeSteps(status, telemetry, request, entry, memo)`, `stepPosition(status)`, `fill()`), `app.js`, `style.css`,
`index.html` (plain `<script>` tags: help.js, steps.js, app.js), `fonts/` (IBM Plex Sans + LICENSE.txt, copied from
teleop_collect/ui). Unchanged: the WebSocket protocol, Play/Stop + Space, `#chips`/`renderChips`, the results table,
`localStorage` key `sim_deploy_ui.v1` (new fields `more`, `hideHelp`, `set-starts` in the same object), the
RunRequest ids/values.

- **Settings**: sections Rollouts (Rollouts, Eraser starts `#set-starts` random|set, Placement seed + dice, Time
  limit), Scene (Arm start, the start-pose line on one line with the rest in the tooltip, Scene look), folded More
  options (`#more`: Run name (tag), tap) and Policy settings (`#advanced`: precision / steps / exec_h, "Reset to the
  arm's", a warn-tint "not the arm's settings" chip on the summary, "no network runs; not used" for replay /
  hold_still). Help lines under each field (hidden by "hide explanations"), longer text in tooltips.
- **starts** is sent only when `/api/policies` has a `start_set` key (an older server refuses unknown keys). `start_set`
  null hides the option. Not allowed (select disabled, shown as random, reason as its help line + tooltip): a
  replay, Arm start ≠ `start_set.needs_home`, a policy with `starts_set: false`, rollouts > `n_starts`. The viewer's
  choice is kept and comes back when allowed. Seed greyed for a replay or starts=set (with the reason).
- **Run header**: "Ready to run" ("Next run" while busy) in the play block from the page's settings, and a banner
  above the external view during/after a run from `status.request` (replaces `#running-what`; since the visual
  restyle always shown: an idle sentence with no run, a state chip Running / Stopping / Last run · finished|failed
  at the right of line 1): type badge
  (POLICY accent / REPLAY · open loop amber / BASELINE · hold still grey) + title; task · kind words + raw kind ·
  dataset + demos (recording breakdown when > 1) · chips (`data_source` ≠ dataproc, smoke); the settings line.
  Under Play: the plan sentence. Picker rows: kind badge in the type colour + separate state badge (not fetched /
  not runnable) + data chip; replays grouped per `group/dataset`, `recording` in the row when the dataset has > 1;
  section subtitles for policies / baseline / replays; results-table kind cells in the same colours (`diffusion` →
  plain) and header tooltips.
- **Step panel** ("Progress", under the cameras on the right since the visual restyle; `#chips` and the run result
  in its side column with "Live numbers"; log in a closed `<details>` "Raw log", opened once when a run fails).
  Modes (chip next to the title): plan (idle: the steps of the current settings, greyed, with help), live (current
  step = accent ring node, pulsing unless prefers-reduced-motion; done = filled check; skipped = dashed; the
  stopping line), finished (cross node on the failing step + `last_error`). Bring-up folds to "Sim and policy
  ready" + its time once a rollout step is reached (a fold button opens it again; it unfolds by itself when a sim
  restart brings `sim_pc` back). Where a failed run failed: the memo's last step before the finish, else (a page
  opened after the run) `status.last_work_step` (runner.py / StubRunner: the last `status.step` that was not
  `wrapup` / `sim_stop`, null per run), so run.sh's trap (`sim_stop`) no longer takes the cross; `sim_stop` counts
  as done when `status.step` is `sim_stop`. Place help with starts=set: "W3 start #n" with n = the driver's 0-based
  start index (rollout 1 = #0; also the field tooltip, plan sentence and glossary). The list is rebuilt only when a step's state/title/help changes; live details (place
  x/y, "12.3 s of 60 s" from the phase `ep N running X s`, "≈ eraser up 3.1 cm" = telemetry height minus its first
  value in the control step, "gripping" = `contacts.both_fingers`, the record outcome) are updated in place.
- **Position**: `status.step` and `step` events (`ev.kind === "step"`); without `status.step` in the status
  (an older server) the phase is mapped by `PHASE_STEPS` in steps.js (mirrors `RUNSH_PHASES` + policy_deploy heads),
  a rollout shows one "Reset (home, start pose, place the eraser)" step, and the rollout = outcomes (not `not_run`)
  + 1. The page keeps a per-run memo (last step, last step before the finish = where a failed run failed, keys
  seen, bring-up time, eraser baseline, place detail).
- **Play** stays disabled until `/api/policies` has loaded (else the request would miss `starts` and drop a saved
  "set"); the reason is the tooltip and the line under the buttons. Space on a fold button (`aria-expanded`)
  toggles the fold instead of Play/Stop. The play block's header key includes a catalog version (+1 per load).
- Checked: `node --check` on the three scripts only (no browser, no server).

### Visual restyle (2026-11-23)

The page looked prototyped; restyled with the user's layout "cameras top-right". Supersedes the layouts in section 4
and U4 (three columns, cameras in the centre). Element ids, `data-*` hooks, the RunRequest, the socket, Play/Stop +
Space, `localStorage` key and every word in help.js kept; new words are in help.js (`PAGE`: panel titles, `cams`,
`cam_*`, `banner_*`, `buttons`, `step_states`, `rollout_line`, `rollouts_earlier`).

- **Grid** (`main.layout`, one CSS grid, gap 16, `max-width` 2200): columns `minmax(340px, 400px) | 1fr` (380–440 from
  1600 px), areas `run views` / `picker views` / `settings progress` / `results results`, rows `auto minmax(0, 1fr)
  auto auto`. The run panel (`section.area-run`, user 2026-11-23: Play matters most, so top left beside the cameras)
  holds the play block: "Ready to run" summary, Play / Stop, the reason line `#play-why`, plan sentence, messages.
  The cameras span the run and picker rows; the picker list has `height: 0; flex: 1` so it takes the rest of that
  height and scrolls in it (min 300 px). Run settings are their own panel under the picker.
  Progress: steps left, a side column (Rollouts chips, run result, Live numbers, notes, `last_error`) right of one
  rule; Raw log folded under both. Below 1100 px one column: run, picker, settings, cameras, progress, results.
- **Tokens** (`style.css :root`, light; dark under `prefers-color-scheme: dark` only, no toggle): spacing 4–48,
  radii 3/6/8/12, `--bg --surface --sunken --text --text-2 --muted --line --line-strong --accent --ok --warn --fail
  --frame`, `--accent-on` (white on #3b4fd0, #0e1012 on #8f9cff), status tints `color-mix(<status> 12%,
  transparent)`. Type: IBM Plex Sans; title 20/28 600, panel titles 15/22 600, body 13.5/20, captions 12/16,
  overlines 11/16 600 caps +0.06em, hero number (run result) 22/28 500; tabular + lining figures everywhere; mono only
  for ids (policy column), the log and error text. Panels: surface, 1 px `--line`, radius 12, padding 16, no
  shadow (only the glossary menu has one); no boxes inside panels.
- **Chips** (badges, state, entry types, outcomes, kind/data/grasp, "not the arm's settings"): 20 px, radius 3,
  tint + full-strength text (no white on a saturated fill; AA in both themes), a leading dot on status chips,
  outcome chips carry a text glyph (✓ ✕ ◷ ■ ! –) besides the colour. Play = the accent button; Stop = a neutral
  button with fail text, fail tint when it has become "Force stop". Icons are inline SVG (reload, dice, play,
  stop) or CSS (fold chevrons, step nodes); no emoji.
- **Video tiles**: `--frame` background, radius 8, no border; sizes reserved before any frame: external
  `width 100%; height min(75cqi, 50vh)` (4:3 = 640×480 when it fits, letterboxed when capped), the small row
  `4fr 4fr 3fr` with one height `min((100cqi − 24px)·3/11, 24vh)` = equal heights for 4:3, 4:3, 1:1. Caption
  under the tile: overline name left (`PAGE.cams`), muted "640×480 · 10.0 fps" right (`#fps-N`, the size of the
  last frame). States from the existing 3 s stale logic (`.frame.live|stale|never`): LIVE chip top-left inside
  (dark glass, pulsing dot); stale = "Stalled 4.5 s" in warn, last frame dimmed, tooltip with the source's
  reason; never = centred "Waiting for camera" + the source's reason. The policy mosaic's quadrant names sit inside
  the image (bottom-left of each quadrant).
- **Run banner** (`#run-header`, top of the cameras panel): fixed min-height 72 px, idle sentence when no run, else
  the header lines + state chip; the video under it never moves.
- **Step timeline**: 10 px nodes on a 1 px `--line-strong` connector (cut at each group's first/last row), 24 px
  pitch; todo hollow muted, now accent ring (pulse), done filled ok + check, fail filled fail + cross, skip dashed;
  a visually hidden state word per row (`PAGE.step_states`). Elapsed column right: the running step = status
  elapsed − its `since_s`; a done step = until the next step began (per-run memo `times` / `order` from `since_s`;
  empty when unknown, e.g. a page opened mid-run or an older server). Group headers are overlines. Recorded rollouts
  that are not shown in detail fold to one line each ("Rollout 2" + outcome chip + its time; node coloured by
  outcome), the last 3; earlier ones are one line "Rollouts 1 to k" with counts.
- **Results table**: 32 px rows, no vertical lines, `--line` separators, sticky overline header, numbers right
  (units in the header, e.g. "success (k/N)"), hover `--sunken`, the selected policy's rows with a 2 px accent bar
  left, no zebra; the table area reserves 228 px (max 420, scrolls).
- **Motion**: 150 ms `cubic-bezier(.16,1,.3,1)` on state changes (colours, opacity, chevrons) only; pulses 2 s
  (LIVE dot, running step, running state chip, current rollout chip); everything off under
  `prefers-reduced-motion`. Focus ring 2 px accent, 2 px offset.
