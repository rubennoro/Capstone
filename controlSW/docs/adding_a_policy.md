# Adding a policy

A policy is a robotmq server. The launcher starts it from `configs/run/policy/<id>.yaml`, the task talks to it over five
topics, and `policy-contract` proves that it speaks the protocol before any robot or simulator is involved. Nothing else in
the repo needs to change to try a new policy, for example a plain UMI diffusion policy.

## The five topics

Every request and reply is `robotmq.serialize(...)`d. A reply that is a string starting with `ERROR` is an error. The
client sends each request once (no resend) and waits for a per-topic timeout.

| Topic | Request | Reply | Limit |
|---|---|---|---|
| `policy_config` | `True` | dict: `workspace.{proprio_length, img_obs_horizon, obs_down_sample_steps, ultrawide_down_sample_steps}` (ints), `use_prompting` (bool); extra keys are fine | 5 s |
| `policy_reset` | `"RESET"` | `"OK"`; clears any prompt and the policy's history | **< 0.5 s** (checked 5 times) |
| `policy_prompt` | `{"task_name": str, "prompt_index": int}` | `"OK"` | 60 s |
| `policy_inference` | obs dict (below) | dict (below) | < `inference_timeout_s` (2.0 s in deploy.yaml) |
| `export_recorded_data` | a time string | anything (`""` is fine); a missing topic is only a warning | 5 s |

Observation, per inference call (shapes are `(T, ...)` with T = 2 steps; see `CONTRACT_OBS` in `tools/analyze_tap.py`):

| Key | Shape | dtype |
|---|---|---|
| `robot0_eef_pos` | (2, 3) | float32 |
| `robot0_eef_rot_axis_angle` | (2, 6) | float32 (rot6d, despite the name) |
| `robot0_gripper_width` | (2, 1) | float32 |
| `robot0_eef_rot_axis_angle_wrt_start` | (2, 6) | float32 |
| `robot0_main_camera`, `robot0_ultrawide_camera` | (2, 720, 960, 3) | uint8 |
| `episode_idx` | () | int64 |
| `task_name` | () | `<U11` |

The pose and rotation entries are relative to the newest observed pose, so the last step is always position 0 and rot6d
`[1,0,0,0,1,0]`. A policy that does not need an input just ignores the key.

Reply: `action0_eef_pos` (K, 3), `action0_eef_rot_axis_angle` (K, 6), `action0_gripper_width` (K, 1), all float32.
Extra keys are ignored.

## Pose convention

- K = 16 steps at 50 ms (the task runs at 20 Hz and executes the first `exec_h` = 8 of them).
- Position and rotation of step k are the pose at step k expressed in the frame of the last observed pose, so step 0 is
  the identity (the contract tool warns when action 0 is more than 2 cm or 5 degrees from it).
- Rotation is rot6d = the first two **rows** of the rotation matrix, row 1 then row 2. Rows must be unit length (norm in
  [0.9, 1.1]) and perpendicular (|cos| < 0.1).
- Gripper width is absolute, in training units (metres between the fingertips), in [0, 0.12].
- Replies must be finite.

## Prompting

`use_prompting: true` means the server follows the behavior-prompting rules, and the task sends a prompt after each reset:
inference before a prompt returns an `ERROR...` string, and a second prompt without a reset returns an `ERROR...` string.
A plain diffusion policy sets `use_prompting: false`. The task then does not need a prompt, and a prompt that does arrive
should simply be answered `"OK"` and ignored. `mocks/fake_policy_server.py --no-prompt-rules` behaves like that, and is the
model to copy.

## `run/policy/<id>.yaml`

Copy `configs/run/policy/dp_template.yaml` (`kind: command`) and edit it. The kinds the launcher knows:

| `kind` | Starts | Needs |
|---|---|---|
| `bp_server` | `franka_deploy.inference.server` (behavior-prompting model) | `checkpoint`, `prompt_zarr`, `python` |
| `replay` | `tools.replay_server` (the demo's own actions) | `episode_index`, `speed_scale`, `start_frame_index` |
| `fake` | `mocks.fake_policy_server` (hold still; no GPU, no checkpoint) | `delay_s` |
| `command` | your command line | `cmd` |

Command format: `cmd` is a list (or a string, split like a shell would). Each element may contain these placeholders:
`{python}` (the policy's `python`: a key of `python:` in deploy.yaml such as `bp`, or a path; the host group gives
`${host.python.bp}`), `{endpoint}` (where to listen, e.g. `tcp://0.0.0.0:18765`), `{ready_file}`, `{work_dir}` (a fresh
`<session>/server`), `{checkpoint}` and `{prompt_zarr}` (the policy's own keys, else deploy.yaml's). `check-config` rejects
a command without `{endpoint}` or `{ready_file}`, or with an unknown placeholder.

The server must (1) serve the five topics on `{endpoint}` and (2) write `{ready_file}` once it can answer an inference
call (after any warm-up), atomically: write `<file>.tmp` and rename. The launcher waits `ready_timeout_s` for it. Content is
free; `inference/server.py` writes `{endpoint, pid, ready_monotonic, ...}`.

Example, a UMI diffusion-policy server (`configs/run/policy/umi_dp.yaml`):

```yaml
schema: run.policy/v1
id: umi_dp
kind: command
cmd: ["{python}", "-u", "/home/me/umi_dp/serve.py", "--ckpt", "{checkpoint}", "--listen", "{endpoint}",
      "--ready-file", "{ready_file}", "--work-dir", "{work_dir}"]
python: ${host.python.bp}
checkpoint: /home/me/models/umi_dp/best.ckpt
ready_timeout_s: 300
contract: {K: 16, exec_h: 8, uses_prompt: false, reset_max_s: 0.5}
task: {agent_hz: 20, exec_h: "${policy.contract.exec_h}", inference_timeout_s: 2.0, lookahead_s: 0.1, policy_latency_s: 0.1}
provenance:      # every key needs a status (measured|placeholder|hw|datasheet|derived|decision|fitted) and a source
  kind: {status: decision, source: "adding_a_policy.md"}
  cmd: {status: placeholder, source: "replace with the real command line of the server"}
  python: {status: derived, source: "host group"}
  checkpoint: {status: placeholder, source: "replace with the real checkpoint path"}
  ready_timeout_s: {status: decision, source: "slow model load"}
  contract: {status: decision, source: "the new policy's chunk length and prompting"}
  task: {status: decision, source: "deploy.yaml:21-29; run_task.py:57-61 literals"}
  task.exec_h: {status: derived, source: "from contract.exec_h"}
critical: [cmd]
```

## Check it

1. Start the server by hand (or use the launcher) and run the contract tool:

   ```
   python -m franka_deploy policy-contract --endpoint tcp://localhost:18765 --expect run/policy/umi_dp.yaml --out contract.json
   python -m franka_deploy policy-contract --help
   ```

   It prints one PASS/WARN/FAIL/SKIP line per check and exits 1 when anything failed. `--expect` also compares the yaml's
   `contract` (K, uses_prompt, reset_max_s) with what the server reports. `--obs tap:DIR` sends the first real observation
   of a tap directory instead of the dummy one. It leaves the server reset, so it is safe to run against a live server.
2. Validate the config without starting anything:
   `python -m franka_deploy check-config --config configs/deploy.yaml --profile station-mock` (set `run: {policy: umi_dp, ...}`).
3. Run it in the launcher: put `run: {policy: umi_dp, ...}` in the profile (the `station-mock` profile of deploy.yaml is the
   hardware-free one) and smoke-test with
   `python -m franka_deploy run --config configs/deploy.yaml --profile station-mock --no-tty --script "2:q"`, which starts
   everything, reaches READY and quits. The sim evaluator (`franka_sim eval ... --policy umi_dp`) picks the policy the
   same way. The launcher records the policy id and kind in `launcher_summary.json`.

Known limits: the contract tool checks the protocol and the reply geometry, not whether the actions are good; the
dummy observation is black images, so a vision policy may reply with arbitrary but valid-looking chunks.
