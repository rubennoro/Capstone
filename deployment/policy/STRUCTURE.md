# policy/

The design of the policy process. Written from PLAN.md sections 4 (C4), 5, 6.2 and 9, following
`deploy/STRUCTURE.md` (the `launch.py` interface, the common arguments, `station.yaml`/`config.yaml`, and section 7,
"Conventions for all folders") and `contracts/STRUCTURE.md` (the C4 message classes, limits, constants and the
transport helpers). PLAN.md is the source of truth: where this file and PLAN.md disagree, PLAN.md wins and this file is
fixed.

## 1. Role

A process that runs the trained model: it takes an input that is ready for the model and returns the model's raw
output.

**Owns**

- Loading the trained model (checkpoint) onto the GPU (or the CPU, PLAN §9) and warming it up.
- Answering C4 requests: images plus relative robot state in, the model's chunk of 16 relative steps out.
- Resetting between episodes.
- The "hold still" stand-in. Until a policy trained on rig data exists, `HoldStillPolicy` is the only policy used on
  the real arm (`run_001` was trained on iPhone images and isn't expected to work on the rig).
- Mapping the contract's field names to the model's own input/output names (`KeyMap`).
- Clipping the chunk's gripper width into `[0, max opening]` and counting the clips (PLAN §5, normal operation).

**Doesn't own**

- Converting poses between absolute and relative, or timing the steps (that is `message_comms/`).
- Talking to the arm or the Pi, or deciding when to run.
- Deciding whether an episode concludes. It answers errors; `message_comms` decides (PLAN §5).
- Image preparation. Images arrive already 224×224, exactly as in training (C1/C4); the policy never resizes.

**Talks to**

`message_comms/` only, over C4 (request/reply). It receives inputs that are already relative and returns the raw
relative chunk. `deploy/` starts it and runs its checks (section 8).

Terms used below:

- **checkpoint**: the `.ckpt` file training writes. It holds the training config (`cfg`), the network weights
  (`state_dicts`) and a few counters (`pickles`, e.g. the epoch).
- **shape_meta**: the part of the training config that lists the model's inputs and outputs: their names, shapes, how
  many past steps (`horizon`) and whether the model actually uses them (`ignore_by_policy`).
- **normalizer**: the model's own scaling of inputs and outputs (offset and scale per field), saved in the checkpoint.
  The policy process never normalizes anything itself.
- **DDIM steps** (`num_inference_steps`): how many denoising steps the diffusion model takes to produce one chunk.
  Fewer steps are faster but rougher (PLAN §9 backup).
- **warm-up**: running a few inferences on a made-up observation before declaring ready, so the first real request
  isn't slowed by one-time GPU set-up.
- **stand-in**: a policy that speaks C4 exactly like the real one but needs no model (`HoldStillPolicy`).

## 2. File tree

```
policy/
  STRUCTURE.md
  launch.py                 PolicyProcess (ManagedProcess subclass), section 8. Not part of the package.
  config.yaml               every setting of this process with its default, section 7
  pyproject.toml            package "policy"; dependencies, section 11
  policy/                   the Python package
    __init__.py             empty apart from a docstring: importing the package imports nothing heavy
    __main__.py             `python -m policy`: parses the common arguments, runs PolicyApp (section 9)
    errors.py               SettingsError, PolicyLoadError, BadRequest, PolicyFatal
    settings.py             Settings: config.yaml + the station's `policy` section, validated
    app.py                  PolicyApp: start-up order, warm-up, ready file, serving, FATAL line
    policy_server.py        PolicyServer: answers C4; enforces reset / prompt order
    policy_factory.py       build_policy(): kind name -> Policy object; imports the chosen class lazily
    base_policy.py          Policy (abstract), incl. the shared chunk builder that clips widths
    promptable_policy.py    PromptablePolicy (abstract, extends Policy)
    hold_still_policy.py    HoldStillPolicy                         (no torch)
    key_map.py              KeyMap: contract names <-> model names  (no torch; numpy only)
    neutral_observation.py  neutral_observation(): a valid made-up Observation, for warm-up and the probe
    diffusion_policy.py     DiffusionPolicy                         (torch)
    model_runner.py         ModelRunner: Observation -> tensors -> predict_action -> chunk arrays   (torch)
    checkpoint_loader.py    CheckpointLoader, LoadedModel: checkpoint -> model on the device, with checks (torch)
    light_workspace.py      inference-only stand-in for behavior_prompting's TrainPolicyWorkspace (torch)
    self_check.py           `python -m policy.self_check`: the package-level checks deploy check runs (section 8)
    probe.py                `python -m policy.probe`: sends config / reset / infer to a running policy (troubleshooting)
  tests/
    conftest.py             fixtures: shape_meta dicts (run_001's, a plain-diffusion one), clocks
    fakes.py                FakePromptablePolicy, RaisingPolicy, BadChunkPolicy, OddChunkPolicy (no torch)
    fakes_torch.py          TinyDiffusionModel (torch; imported only by torch tests)
    test_*.py               section 10
```

**Torch rule** (deploy/STRUCTURE.md §7.3 and §8): only the four files marked "(torch)" import torch or
`behavior_prompting`, and nothing imports them except `policy_factory.build_policy` when `kind: diffusion` is
selected. So `HoldStillPolicy` runs in the light environment without torch, which is how `--mock` runs.

**Where each file comes from.** Copied or adapted files start with the provenance header of deploy/STRUCTURE.md §7.6.

| File | Source | Changes |
|---|---|---|
| `light_workspace.py` | `rnoro/main-sim:controlSW/franka_deploy/franka_deploy/inference/light_workspace.py @ 4be9982` | the vision-normalisation check accepts both `TimmObsEncoder` (the encoder of a plain diffusion policy) and `TransformerObsEncoder` (the prompting encoder, the only one main-sim knew) |
| `checkpoint_loader.py` | `rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd` (`PolicyInferenceNodeRealEnv.__init__`: `torch.load` with `dill`, the workspace, `load_payload` without the optimizer, `eval().to(device)`, `reset()`), plus the start-up checks of main-sim's `inference/server.py @ 4be9982` (`build_node`) | no `<ckpt>.yaml` written next to the checkpoint, no robotmq server, no prompt dataset, no language tokenizer; adds the C4 compatibility checks (section 5.9) and the DDIM-steps override |
| `model_runner.py` | `detached_inference.py @ edfe8bd`: `_process_obs` (image layout), the `policy_inference` branch of `run_node`, `_process_action`; main-sim `server.py` `node_infer` | the resize (`get_image_transform_with_border`) is removed (images arrive 224×224); keys come from `KeyMap` instead of the `robot{i}_*` names |
| `hold_still_policy.py` | new; behaves like `mocks/fake_policy_server.py --no-prompt-rules @ 935c0b3` (identity chunk, no prompt needed) | width = the current width instead of a fixed 0.07 m |
| `probe.py` | new; replaces `tools/policy_contract.py @ 935c0b3` | most of that tool's checks are now the `contracts` validators, run on every message |
| everything else | new | |

The upstream `detached_inference.py` itself is **not imported**: its constructor writes `<ckpt>.yaml` beside the
checkpoint, starts a robotmq server and imports `transformers`' tokenizer. Only the pieces above are adapted.

**C4 sizes.** The fixed C4 numbers are imported from `contracts` (`contracts/contracts/sizes.py`): `OBS_STEPS = 2`,
`ACTION_STEPS = 16`, `STEP_MS = 50`, `IMAGE_SIZE = 224`. `policy/` has no copy of them: `Policy.config()` reports
them, and the shape checks of section 5.9 compare the checkpoint against them.

## 3. How the process runs

### 3.1 Start-up, in order

`PolicyApp.run()`:

1. Set up logging (deploy §7.6 format, to stderr).
2. `Settings.load(station_file, session, mock)`: read `config.yaml`, merge the station's `policy` section, read the
   endpoint (and, without `--session`, `session_root`) from the station file, validate everything. Problems →
   `SettingsError` → FATAL.
3. Make the session folder (`--session DIR` → `DIR/policy/`; alone → `<session_root>/<timestamp>_policy_alone/policy/`).
4. `build_policy(settings)`: the policy for the active kind (`hold_still` with `--mock`, else `kind`). For `diffusion`
   this loads the checkpoint onto the device and runs the load checks (section 5.9). Problems → `PolicyLoadError` →
   FATAL.
5. **Warm-up** (section 3.3).
6. `PolicyServer(policy, endpoint)`: **bind** the C4 socket (`endpoints.policy`). Nothing answers before this point,
   so nothing can be answered "as if ready" (deploy §4.4).
7. Write `startup.json` into the session folder (section 3.6).
8. Create the ready file (empty; written as `<file>.tmp` then renamed, so the launcher never sees half a file).
9. Install the SIGINT/SIGTERM handler, then serve until stopped or a fatal error (section 3.2).

Loading `run_001`-sized checkpoints takes tens of seconds; `launcher.ready_timeout_s.policy` is 300 s
(deploy §6.1). A Ctrl-C during loading (before the signal handler of step 9 is installed) raises Python's normal
`KeyboardInterrupt`; `PolicyApp.run` catches it, so the process exits 0 with no FATAL line.

### 3.2 Thread model

**Single-threaded.** One thread does everything: load, warm-up, then a loop of
`RpcServer.serve_once(timeout_s=poll_s)`. C4 is request/reply with one client (`message_comms`), so requests never
overlap and there is nothing to lock. Torch runs on this thread. The signal handler only sets a `threading.Event`;
the loop sees it within `poll_s` (or right after the current inference finishes), closes the socket and returns
exit code 0.

### 3.3 Warm-up

- `PolicyApp` builds a made-up observation with `neutral_observation(warmup_gripper_width_m)`: black images, both
  pose rows at the anchor (position 0, rot6d `[1,0,0,0,1,0]`), `wrt_start` identity, the given width. It is a real
  `Observation`, so it passes the C4 validator.
- It calls `policy.infer()` `warmup_inferences` times (default 2: the first one pays for CUDA/cuDNN set-up, the
  second gives a representative time), timing each one, then calls `policy.reset()`.
- The times are logged (`warm-up inference 1/2: 1.84 s, 2/2: 0.09 s on cuda`) and written to `startup.json`.
- The synthetic input can make a model produce a chunk that fails a C4 **value** check (e.g. rot6d rows that are not
  unit length on black images). During warm-up only, a `ContractError` with `problem == "value"` is logged as a
  **warning** and warm-up continues. Any other warm-up failure (shape, dtype, CUDA, any exception) is fatal. In normal
  serving every contract failure of its own reply is fatal (section 3.5).
- `HoldStillPolicy` runs the same warm-up; it costs microseconds and proves its reply path before ready.

### 3.4 `reset` stays under 0.5 s

`reset` never touches the GPU, the disk or the network:

- `DiffusionPolicy.reset()` calls the model's own `reset()` (which clears the observation encoder's state; trivial for
  a plain diffusion policy). No re-warm-up, no `torch.cuda.empty_cache()`.
- `HoldStillPolicy.reset()` does nothing.

The policy keeps **no observation history**: every `infer` request carries both observation steps (C4), so there
is nothing else to clear. `PolicyServer` times every reset and logs a warning if one takes longer than `reset_max_s`
(0.5 s, PLAN §4); the unit tests assert it for both policies.

### 3.5 Failure rule (PLAN §5, as it applies here)

Two kinds of error, and nothing else:

| What | Example | Reaction |
|---|---|---|
| **Bad request** (the request is the problem) | broken C4 message (wrong shape, schema version); unknown request type; `infer` before the first `reset`; `prompt` to a policy without prompting; a second `prompt` without a `reset` | `ErrorReply` naming the problem; the process **keeps serving**. `message_comms` concludes on it (PLAN §5), but that is its decision. |
| **Own unrecoverable problem** | settings invalid; checkpoint missing or fails to load or fails its checks; CUDA not available or a CUDA error during inference; its own reply fails the C4 validator (wrong shape or dtype, NaN, non-unit or non-perpendicular rot6d); any unexpected exception inside the policy | at start-up: FATAL line, exit 1. While serving: the current request still gets an `ErrorReply` (so `message_comms` learns the reason at once instead of timing out), **then** the process writes its FATAL line and exits 1. |

**Not failures of this process:**

- A model width outside `[0, max opening]`. It is clipped into range and counted (section 4.1), as PLAN §5 lists under
  normal operation; `franka_control` checks the width again.
- An **odd chunk**: step 0 more than 2 cm / 5° from the anchor. The C4 validator doesn't check this (PLAN §4); the
  policy doesn't check it either and returns the chunk as the model produced it. `message_comms` checks it, skips that
  step and counts it (a "slow" case, PLAN §5). So the policy never exits FATAL for it.

Request-level contract errors are handled by `contracts.RpcServer` itself (it decodes and validates before any
handler runs). Policies raise `BadRequest` for problems caused by the request (e.g. a future prompting policy given an
unknown `task_name`); anything else a policy raises is treated as its own failure.

A **slow** reply is not this process's concern: `message_comms` measures it and decides (PLAN §5).

FATAL lines (deploy §7.6, `FATAL policy: <what failed>: <what to check>`), for example:

```
FATAL policy: policy.kind is not set: choose diffusion or hold_still in station.yaml (policy.kind)
FATAL policy: checkpoint not found: /home/me/ckpt/best.ckpt: check policy.checkpoint in station.yaml
FATAL policy: CUDA not available: check the GPU driver, or set policy.device: cpu (PLAN §9 backup)
FATAL policy: checkpoint uses prompting (shape_meta.use_prompting: true): kind diffusion needs a plain diffusion checkpoint
FATAL policy: model action horizon 12 < 16 (C4 action_steps): use a checkpoint trained with action_horizon >= 16
FATAL policy: CUDA error during infer: CUDA out of memory ...: check GPU memory, or set policy.device: cpu
FATAL policy: own reply broke C4: c4.chunk/v1: field 'eef_rot6d': ...: the model output is invalid; check the checkpoint and its inputs
```

### 3.6 What it writes

- Its log: stderr, captured by the launcher into `<session>/logs/policy.log`.
- `<session>/policy/startup.json`: active kind, whether `--mock`, the merged settings, `describe()` of the policy
  (for `diffusion`: checkpoint path, size and modification time, epoch, model name, DDIM steps in the checkpoint and in
  use, device and GPU name, torch version, load time, the load report, the key map and the inputs used), the warm-up
  times, the endpoint and the pid.
- At each `reset`, one log line summarising the episode that just ended: number of inferences, mean and max
  inference time, and the number of clipped widths.

It never writes next to the checkpoint (it is opened read-only).

## 4. Interfaces and the policy classes

### 4.1 `Policy` (`base_policy.py`)

```
class Policy(ABC):
    KIND: ClassVar[str]                     # "diffusion", "hold_still"
    USES_PROMPT: ClassVar[bool] = False

    def __init__(self, gripper_width_max_m: float) -> None   # from ContractLimits (contracts/limits.yaml)

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: Settings) -> "Policy"     # build it (and load what it needs)

    @abstractmethod
    def inputs(self) -> tuple[str, ...]     # the Observation field names it actually reads
    @abstractmethod
    def reset(self) -> None                 # clear episode state; < reset_max_s
    @abstractmethod
    def infer(self, obs: Observation) -> ActionChunk    # raises BadRequest, or anything else = own failure

    def config(self) -> PolicyConfig        # concrete: OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE + inputs() + USES_PROMPT
    def describe(self) -> dict[str, Any]    # details for startup.json; default {"kind": KIND}

    def make_chunk(self, eef_pos: np.ndarray, eef_rot6d: np.ndarray, gripper_width_m: np.ndarray) -> ActionChunk
        # concrete; every policy builds its reply through it (see below)
```

`make_chunk` is where **every** policy's width is clipped, so no subclass can forget it:

1. `gripper_width_m` (float32) is clipped to `[0, w_max]`, where `w_max` is the **largest float32 value ≤
   `gripper_width_max_m`** (the max opening from `contracts/limits.yaml`, 0.08 m): `np.float32(max)` if that is ≤ max,
   else `np.nextafter(np.float32(max), np.float32(0))`. The rounding matters because the validator compares the
   float32 width with the float64 limit: `np.float32(0.07)` is 0.0700000003, above 0.07, and would be refused, while
   `np.float32(0.08)` is 0.0799999982 and needs no step down.
2. `n` = how many of the 16 steps were changed by the clip.
3. It returns `ActionChunk(eef_pos, eef_rot6d, gripper_width_m, width_clipped=n)`, which runs the C4 validator.

How the count reaches the status line: the chunk carries it. `width_clipped` is C4's always-present `c4.chunk` field
(uint64, `0 ≤ n ≤ ACTION_STEPS`, 0 when nothing was clipped); `message_comms` adds it up per episode and shows it on
the status line, as PLAN §5 asks for every normal-operation correction. The policy also logs the episode total at each
`reset` (section 3.6).

`Observation`, `ActionChunk` and `PolicyConfig` are the `contracts` message classes (contracts §4.6), so anything a
policy returns has already passed the C4 validator when it is built (contracts §3: "a Message object that exists is
valid").

### 4.2 `PromptablePolicy` (`promptable_policy.py`)

```
class PromptablePolicy(Policy):
    USES_PROMPT = True
    @abstractmethod
    def prompt(self, task_name: str, prompt_index: int) -> None    # raises BadRequest for an unknown task / index
```

No implementation exists yet. `config()` reports `uses_prompt: true` for it automatically. Section 12 shows what
adding `BehaviorPromptingPolicy` takes.

### 4.3 `HoldStillPolicy` (`hold_still_policy.py`, no torch)

"Stay at the anchor": every reply is the anchor itself, 16 times. It is the stand-in for `--mock` and, until a
rig-trained diffusion policy exists, the only policy used on the real arm (PLAN §6.2): it exercises frames, timing,
conversions and the safety gate end to end without moving the arm on purpose.

```
class HoldStillPolicy(Policy):
    KIND = "hold_still"
    def inputs(self) -> tuple[str, ...]          # ("gripper_width_m",)
    def reset(self) -> None                      # nothing
    def infer(self, obs: Observation) -> ActionChunk
```

`infer`: `eef_pos` = zeros (16, 3); `eef_rot6d` = `[1,0,0,0,1,0]` × 16; `gripper_width_m` = the newest observed width
(`obs.gripper_width_m[-1, 0]`) × 16, all float32, through `make_chunk` (so a measured width above the max opening is
clipped and counted like any other). The arm holds both its pose and its opening. Step 0 is exactly the anchor, so its chunk is
never odd.

### 4.4 `DiffusionPolicy` (`diffusion_policy.py`, torch)

```
class DiffusionPolicy(Policy):
    KIND = "diffusion"
    def __init__(self, runner: ModelRunner, model_reset: Callable[[], None], gripper_width_max_m: float) -> None
    @classmethod
    def from_settings(cls, settings: Settings) -> "DiffusionPolicy"
        # CheckpointLoader.load(...) -> LoadedModel; refuses a prompting checkpoint; builds KeyMap and ModelRunner
    def inputs(self) -> tuple[str, ...]          # runner.inputs, e.g. ("eef_pos", "eef_rot6d", "gripper_width_m", "main", "ultrawide")
    def reset(self) -> None                      # model.reset()
    def infer(self, obs: Observation) -> ActionChunk    # self.make_chunk(*runner.run(obs))
    def describe(self) -> dict[str, Any]         # LoadedModel.info + key map + inputs
```

The constructor takes an already-built `ModelRunner`, so the unit tests build one around a tiny dummy network
(section 10) without a checkpoint or `behavior_prompting`.

### 4.5 `ModelRunner` (`model_runner.py`, torch)

```
class ModelRunner:
    def __init__(self, model: torch.nn.Module, shape_meta: Mapping, key_map: KeyMap, device: torch.device) -> None
        # checks shape_meta against C4 (the shape rows of section 5.9: horizons, 224 images, action horizon >= 16,
        # component sizes) and runs KeyMap.check(shape_meta); raises PolicyLoadError
    inputs: tuple[str, ...]                       # contract fields whose model keys the model uses
    def run(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]
        # eef_pos (16, 3), eef_rot6d (16, 6), gripper_width_m (16, 1), float32, unclipped
```

`model` is anything with `predict_action(obs_dict) -> {"action_pred": Tensor (B, H, D)}`, `reset()`, `eval()`
(a behavior_prompting `BasePolicy` in real use, `TinyDiffusionModel` in tests). Section 5 describes `run` step by step.
`BehaviorPromptingPolicy` will reuse it unchanged.

### 4.6 `CheckpointLoader` and `LoadedModel` (`checkpoint_loader.py`, torch)

```
@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module                 # in eval mode, on the device, reset
    shape_meta: Mapping[str, Any]          # cfg.task.shape_meta, as plain dicts
    uses_prompting: bool                   # shape_meta.use_prompting
    info: Mapping[str, Any]                # epoch, cfg.name, model target, DDIM steps (checkpoint / in use),
                                           # num_train_timesteps, device, GPU name, torch version, load_s, load report

class CheckpointLoader:
    def __init__(self, training_timing: Mapping[str, int]) -> None
    def load(self, path: Path, device: str, diffusion_steps: int | None) -> LoadedModel   # raises PolicyLoadError
```

`load`, in order:

1. The file exists and ends in `.ckpt`; `device` is `cpu`, or `cuda` with `torch.cuda.is_available()` (else
   `PolicyLoadError("CUDA not available", "check the GPU driver, or set policy.device: cpu")`). No automatic fallback
   to the CPU: the backup is one explicit station setting (PLAN §9).
2. `light_workspace.install()` (before anything imports the real `TrainPolicyWorkspace`, which would pull in LIBERO,
   robosuite and wandb, and download the CLIP backbone only to overwrite it).
3. `torch.load(path, map_location="cpu", pickle_module=dill)`; `cfg = payload["cfg"]`; the workspace class from
   `cfg._target_` (now the light one); `workspace.load_payload(payload, exclude_keys=["optimizer"])` with strict key
   matching. The checkpoint's `model` weights are the EMA weights (training saves while the EMA copy is in the model),
   so no separate EMA step is needed.
4. `model = workspace.model.eval().to(device)`; `model.reset()`.
5. The checks of section 5.9 that need the checkpoint (weights, imported modules, colour normalisation, pose
   representation, timing, scheduler). Any failure → `PolicyLoadError` naming the key and what to do.
6. If `diffusion_steps` is set: `model.num_inference_steps = diffusion_steps` (section 5.8).

`CheckpointLoader` does not refuse a prompting checkpoint; `DiffusionPolicy.from_settings` does, so a future
`BehaviorPromptingPolicy` can use the same loader.

A checkpoint under `results/` is fine: it is only read, and nothing is written beside it (main-sim refused that folder
only because upstream wrote `<ckpt>.yaml` next to the checkpoint, and that code isn't used here).

### 4.7 `KeyMap` (`key_map.py`, numpy only)

Maps contract names to the model's own names. The defaults in `config.yaml` match every UMI single-arm model trained
by `behavior_prompting` (and the rig's dataproc output, which uses the same names):

| Contract (C4) | Model obs key (default) | Notes |
|---|---|---|
| `eef_pos` | `gripper_right_eef_pos` | |
| `eef_rot6d` | `gripper_right_eef_rot_axis_angle` | rot6d despite the name |
| `eef_rot6d_wrt_start` | `gripper_right_eef_rot_axis_angle_wrt_start` | `ignore_by_policy: true` in `run_001` and the default task config, so not sent to the model |
| `gripper_width_m` | `gripper_right_gripper_width` | |
| `main` | `camera_right_main_rgb` | |
| `ultrawide` | `camera_right_ultrawide_rgb` | |

| Contract (C4 chunk) | Model action component (default) | Size |
|---|---|---|
| `eef_pos` | `right_pos` | 3 |
| `eef_rot6d` | `right_rot` | 6 |
| `gripper_width_m` | `right_gripper_width` | 1 |

The action is one vector of size 10 per step; `shape_meta.action.components` gives the order and sizes, so the slices
are computed from the checkpoint, not hard-coded.

The map is an explicit `config.yaml` section rather than derived from `shape_meta` prefixes (as
`detached_inference.py` did): a model with other names is a config change, not a code change, and a mismatch fails at
start-up with the missing name.

```
class KeyMap:
    def __init__(self, obs: Mapping[str, str], action: Mapping[str, str]) -> None   # contract -> model
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "KeyMap"     # the key_map section of config.yaml
    def model_key(self, field: str) -> str
    def contract_field(self, model_key: str) -> str
    def check(self, shape_meta: Mapping) -> None                 # raises PolicyLoadError listing every problem
    def used_fields(self, shape_meta: Mapping) -> tuple[str, ...]          # contract fields the model uses
    def action_slices(self, shape_meta: Mapping) -> dict[str, slice]       # contract field -> columns of the vector
    def split_action(self, action: np.ndarray, slices: Mapping[str, slice]) -> dict[str, np.ndarray]
    def join_action(self, parts: Mapping[str, np.ndarray], slices: Mapping[str, slice]) -> np.ndarray   # tests, round trip
```

`check(shape_meta)` refuses: a model input the model uses (`ignore_by_policy` false) that no contract field maps to; a
mapped model key that isn't in `shape_meta`; an unknown contract field name; an action component without a contract
field; a component size different from the contract's (3 / 6 / 1). Both maps must be one-to-one.

### 4.8 `PolicyServer` (`policy_server.py`)

```
class PolicyServer:
    def __init__(self, policy: Policy, endpoint: str, reset_max_s: float,
                 clock: Callable[[], int] = time.monotonic_ns) -> None      # binds (contracts.RpcServer)
    endpoint: str                                     # the bound address
    def serve(self, stop: threading.Event, poll_s: float) -> None    # returns on stop; raises PolicyFatal
    def close(self) -> None
```

Handlers given to `contracts.RpcServer`:

| Request | Allowed when | Does | Reply |
|---|---|---|---|
| `PolicyConfigRequest` | always | `policy.config()` | `PolicyConfig` |
| `Reset` | always | `policy.reset()`, timed; logs the finished episode's stats | `Ack(ok=True)` |
| `Prompt` | promptable policy, **right after a reset** (once) | `policy.prompt(task_name, prompt_index)` | `Ack(ok=True)` |
| `Observation` (`c4.infer/v1`) | after a reset (and, if promptable, after its prompt) | `policy.infer(obs)`, timed | `ActionChunk` |

Order, tracked in one small state (`needs_reset` → `needs_prompt` (promptable only) → `ready`):

- The process starts in `needs_reset`: **`infer` before the first `reset` is refused** (`infer before reset`). So
  `message_comms` always resets before an episode's first `infer` (PLAN §6.4 step 2 already does); a missing reset is
  caught as its bug instead of silently starting from stale state.
- `prompt` to a policy that is not a `PromptablePolicy` is always a bad request: `prompt: this policy (kind
  hold_still) does not use prompts (config uses_prompt: false)`.
- For a promptable policy: `infer` before the prompt (`infer before prompt`) and a second `prompt` without a `reset`
  (`prompt already given since the last reset`) are bad requests. So it accepts **exactly one prompt after each reset**.

Error handling inside each handler: `BadRequest` is re-raised as is, so `RpcServer` sends `ErrorReply` and keeps
serving. Any other exception is stored as `PolicyFatal` (with the request name and the error) and re-raised, so the
same `ErrorReply` goes out first; `serve` checks for a stored fatal after every `serve_once` and raises it, and
`PolicyApp` turns it into the FATAL line.

### 4.9 `Settings` (`settings.py`)

```
@dataclass(frozen=True)
class Settings:
    kind: str | None
    mock: bool
    active_kind: str                 # "hold_still" if mock else kind
    endpoint: str                    # endpoints.policy from the station file
    session_dir: Path                # --session DIR / "policy", or <session_root>/<timestamp>_policy_alone/policy
    ...                              # one field per config.yaml key (section 7), paths resolved
    @classmethod
    def load(cls, station_file: Path, session: Path | None, mock: bool) -> "Settings"   # raises SettingsError
```

It applies deploy/STRUCTURE.md §6.3's merge rule itself (the package can't import `deploy`): `config.yaml` (found at
`Path(policy.__file__).parents[1] / "config.yaml"`), then the station's `policy:` section merged over it (maps key by
key, everything else replaced, an unknown key is an error with its path). Relative paths from the station are taken
from the station file's folder; defaults from `config.yaml` are taken from the `policy/` folder.

**Endpoint and session folder come from the station file and the common arguments only;** `policy/` repeats no
default of `deploy/`. `endpoint` is the station's `endpoints.policy`. The session folder is `--session` when given;
only without it (started alone) is the station's `session_root` read. A key needed but absent from the station file is
a `SettingsError` naming it (e.g. `station.yaml: endpoints.policy: missing`).

Validation (all problems collected, then one `SettingsError`): types and ranges of section 7; `kind` set unless
`--mock`; `kind` one of the factory's kinds; `checkpoint` set when the active kind is `diffusion`.

### 4.10 `build_policy` (`policy_factory.py`)

```
KINDS: Mapping[str, KindEntry]      # name -> (module path, class name, needs_torch)
    "hold_still": ("policy.hold_still_policy", "HoldStillPolicy", False)
    "diffusion":  ("policy.diffusion_policy",  "DiffusionPolicy",  True)

def build_policy(settings: Settings) -> Policy
```

It imports only the chosen module (`importlib.import_module`), checks the class's `KIND` matches, and calls
`from_settings`. If a kind that needs torch is chosen in an environment without it, the `ImportError` becomes
`PolicyLoadError("torch is not importable (kind diffusion)", "run policy with python.policy, the torch environment")`.

### 4.11 `PolicyApp` (`app.py`) and errors (`errors.py`)

```
class PolicyApp:
    def __init__(self, station: Path, session: Path | None, ready_file: Path | None, mock: bool) -> None
    def run(self) -> int        # 0: stopped normally; 1: FATAL line written
```

```
class SettingsError(ValueError):    problems: list[str]
class PolicyLoadError(RuntimeError): what: str; check: str
class BadRequest(ValueError):        # the message is the one-line reason sent in ErrorReply
class PolicyFatal(RuntimeError):     what: str; check: str
```

`run` catches `KeyboardInterrupt` (Ctrl-C before serving starts: exit 0, no FATAL line), and `SettingsError`,
`PolicyLoadError` and `PolicyFatal` (their FATAL line, exit 1); any other exception
is logged with its traceback and becomes `FATAL policy: unexpected error: <type>: <message>: see the log above`.

## 5. One `infer` request, from C4 to the model and back

What `DiffusionPolicy.infer(obs)` does, through `ModelRunner.run(obs)` and `Policy.make_chunk`. Everything is checked
to fit at load time (section 5.9), so at run time there are no shape surprises.

### 5.1 In: the C4 observation

The request has already been decoded and validated by `contracts` (dtypes, shapes, finite, anchor rows, rot6d). Its
arrays are read-only views of the received bytes (contracts §5).

| C4 field | Arrives as |
|---|---|
| `eef_pos` | float32 (2, 3) |
| `eef_rot6d` | float32 (2, 6) |
| `eef_rot6d_wrt_start` | float32 (2, 6) |
| `gripper_width_m` | float32 (2, 1) |
| `main`, `ultrawide` | uint8 (2, 224, 224, 3), RGB |

### 5.2 Which fields go to the model

Only the fields the model uses: those whose model key has `ignore_by_policy: false` in the checkpoint's `shape_meta`.
For `run_001`'s layout that is all of them except `eef_rot6d_wrt_start`. Passing an ignored key could even fail,
because the model's normalizer has no entry for it. `config()` reports exactly these fields as `inputs`.

### 5.3 History

The model needs **2 steps** of every input (`low_dim_obs_horizon: 2`, `img_obs_horizon: 2` in the checkpoint), oldest
first, and C4 provides exactly 2, with the same spacing as training (main and state 50 ms apart, ultrawide 100 ms
apart; `message_comms` and `process_frames` own that timing). The policy keeps no history of its own.

### 5.4 Conversions (numpy, on the CPU, identical to `detached_inference._process_obs`)

| Input | Conversion | To the model |
|---|---|---|
| image fields | copy into a writable array (torch warns on read-only buffers); `np.moveaxis(x, -1, 1)` (HWC → CHW); `.astype(np.float32) / 255.0` | float32 (2, 3, 224, 224) in [0, 1] |
| low-dim fields | `np.array(x, dtype=np.float32)` (already float32: a copy only) | float32 (2, D) |

Then each array becomes a tensor with a batch dimension in front (`torch.from_numpy(a).unsqueeze(0).to(device)`):
images (1, 2, 3, 224, 224), low-dim (1, 2, D), under the model's key names (`KeyMap`). The conversion is done on the
CPU exactly as the reference code does it, so the numbers the model sees are bit-identical to what the reference
server would feed it; the cost (four 224×224 images) is small next to the inference.

### 5.5 What the model does itself (not repeated here)

`model.predict_action(obs_tensors)` inside `torch.inference_mode()`, with the model in `eval()` mode. Inside the model,
from its own checkpoint:

- the **normalizer** scales the low-dim inputs (and leaves images as the training did);
- the evaluation image transform (`CenterCrop` ratio 0.95 in the UMI task config) and the backbone's colour
  normalisation (CLIP mean/std) are applied by the observation encoder because it is in `eval()` mode;
- the DDIM sampler runs `num_inference_steps` denoising steps (section 5.8);
- the output is **un-normalized** back to training units before it is returned.

So the policy process never normalizes or un-normalizes anything.

### 5.6 Out: slicing, splitting, clipping

1. `result["action_pred"][0]` → `.float().cpu().numpy()`: float32 (H, 10), H = the model's action horizon. The copy to
   the CPU also waits for the GPU to finish.
2. Keep the first `ACTION_STEPS` = 16 rows. H < 16 is refused at load; H = 16 for every config in the repo.
3. Split the 10 columns by `KeyMap.action_slices`: `eef_pos` (16, 3), `eef_rot6d` (16, 6), `gripper_width_m` (16, 1),
   each made C-contiguous float32. This is what `ModelRunner.run` returns.
4. `Policy.make_chunk` clips `gripper_width_m` to `[0, max opening]`, counts the clipped steps into `width_clipped`
   (section 4.1), and builds the `ActionChunk`, which runs the C4 validator (shapes and dtypes, finite, rot6d unit and
   perpendicular). Step 0's distance from the anchor is not checked here (`message_comms` does, section 3.5). The rot6d values are passed through as the model produced them; making
   them exactly orthonormal is `message_comms`' job when it converts to absolute poses. A validator failure here is
   the policy's own contract violation: fatal (section 3.5).

### 5.7 Units and frames

Unchanged from training: positions in metres relative to the anchor, rot6d = the first two **rows** of the relative
rotation matrix, widths absolute in metres. These are the same quantities as C4 (PLAN §4), which is why only names
need mapping.

### 5.8 Number of diffusion steps (PLAN §9 backup)

- The checkpoint's own number is `cfg.model.num_inference_steps` (16 for `run_001` and `diffusion_unet_base.yaml`;
  `null` there would mean `num_train_timesteps`, 50). This is the default (`diffusion_steps: null`).
- `diffusion_steps: N` sets `model.num_inference_steps = N` after loading; the sampler calls
  `scheduler.set_timesteps(N)` on every inference, so nothing else changes. Allowed: `1 ≤ N ≤ num_train_timesteps`,
  and only when the scheduler is DDIM (`diffusers.DDIMScheduler`, as in every config in the repo); otherwise a load
  error.
- `startup.json`, the log and `deploy check` (info line) show both numbers.
- On the CPU the vision encoder (a ViT-B per camera) runs once per inference whatever N is, so fewer steps help less
  there than on the GPU.

### 5.9 Load checks (each failure is a `PolicyLoadError` naming the key)

| Check | Why |
|---|---|
| strict weight load: no missing or unexpected keys | a mismatched architecture would otherwise run with random weights |
| no heavy simulation modules imported (LIBERO, robosuite, mujoco, wandb, ...) | as main-sim's start-up check; LIBERO's import can block on `input()` |
| backbone colour normalisation equals the backbone's reference constants, for both encoder types (`TimmObsEncoder`, `TransformerObsEncoder`): OpenAI CLIP for `*clip*` backbones; another backbone → error asking to add its constants to `light_workspace.py` | these constants are not stored in the checkpoint; a wrong value would silently change every image |
| `use_prompting` is false (in `DiffusionPolicy.from_settings`) | a prompting checkpoint needs its prompt; that is a future `BehaviorPromptingPolicy` |
| `task.pose_repr.obs_pose_repr` and `action_pose_repr` are `relative`; `action.rotation_rep` is `rotation_6d` | C4 is relative rot6d both ways |
| every used obs key has `horizon` = 2; images have shape `[3, 224, 224]`; low-dim shapes are 3 / 6 / 6 / 1 as in C4 | C4's fixed shapes; images are never resized here |
| action horizon ≥ 16; the action components add up to the action shape | C4's 16 steps |
| `task.obs_down_sample_steps` and `task.ultrawide_down_sample_steps` equal `training_timing` in `config.yaml` (3 and 1: 50 ms and 100 ms) | the model's step timing must match C1/C4 |
| `KeyMap.check(shape_meta)` | names (section 4.7) |
| DDIM-steps override valid (section 5.8) | |

## 6. Policy environment vs light environment

| Run | Python | Policy | Needs |
|---|---|---|---|
| `deploy run` (real) | `python.policy` (the `behavior_prompting` conda env) | `kind` from the station (`diffusion` or `hold_still`) | torch + `behavior_prompting` for `diffusion`; a GPU unless `device: cpu` |
| `deploy run --mock` | `python.light` | always `hold_still` | no torch, no checkpoint, no GPU |

With `--mock` the `kind` setting is not used (and may be unset). There is one stand-in, so there is no setting to
choose it.

## 7. `config.yaml`

Every setting of this process with its default (deploy §6.2, §6.3). Any key can be overridden in the station's
`policy:` section; the first block holds the ones meant to be.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `diffusion` \| `hold_still` \| null | `null`: **required** unless `--mock` | which policy runs (PLAN §6.2: one station line) |
| `checkpoint` | path \| null | `null`; required for `diffusion` | the `.ckpt` file; opened read-only |
| `device` | `cuda` \| `cpu` | `cuda` | PLAN §9 device backup |
| `diffusion_steps` | int ≥ 1 \| null | `null` (the checkpoint's own) | PLAN §9 fewer-steps backup (section 5.8) |
| `warmup_inferences` | int ≥ 1 | `2` | warm-up calls before ready |
| `warmup_gripper_width_m` | float ≥ 0 | `0.05` | width used in the warm-up observation |
| `reset_max_s` | float > 0 | `0.5` | a slower `reset` is logged as a warning (PLAN §4 limit) |
| `poll_s` | float > 0 | `0.05` | serve-loop poll; how quickly SIGINT is noticed between requests |
| `self_check_timeout_s` | float > 0 | `20.0` | `launch.py`: time allowed for `python -m policy.self_check` (importing torch takes a few seconds) |
| `training_timing.obs_down_sample_steps` | int | `3` | the checkpoint must match (60 Hz → 50 ms steps) |
| `training_timing.ultrawide_down_sample_steps` | int | `1` | the checkpoint must match (10 Hz ultrawide → 100 ms) |
| `key_map.obs.<C4 field>` | str | section 4.7 table | the model's obs key for each C4 observation field |
| `key_map.action.<C4 field>` | str | section 4.7 table | the model's action component type for each C4 chunk field |

Taken from elsewhere, not repeated here: `endpoints.policy` and `session_root` (station file, deploy §6.1); the max
opening and the C4 tolerances (`contracts/contracts/limits.yaml`); the C4 sizes `OBS_STEPS`, `ACTION_STEPS`,
`STEP_MS`, `IMAGE_SIZE` (`contracts/contracts/sizes.py`).

Example station section, first real-arm runs:

```yaml
policy:
  kind: hold_still
```

and later, with a rig-trained model:

```yaml
policy:
  kind: diffusion
  checkpoint: ~/models/block_raise_rig/best.ckpt
```

## 8. `launch.py` and the checks

### 8.1 `PolicyProcess`

```
class PolicyProcess(ManagedProcess):
    NAME = "policy"
    READY_HINT = ("checkpoint load + warm-up inference; check policy.kind, policy.checkpoint, policy.device "
                  "and the GPU")
    def program(self) -> list[str]
    def checks(self) -> list[CheckResult]

PROCESS_CLASS = PolicyProcess
```

- `program()`: `[self.station.python("light"), "-m", "policy"]` with `--mock`; otherwise
  `[self.station.python("policy"), "-m", "policy"]` (deploy §8). The base class appends `--station`, `--session`,
  `--ready-file` and `--mock`.
- `extra_env()`: not overridden. `is_ready()`: the default (the ready file, written after warm-up and bind, section
  3.1).
- Imports: standard library, `yaml`, `deploy`, `contracts` only (deploy §4).

### 8.2 `checks()` (what `deploy check` shows for `policy`)

`launch.py` cannot import the `policy` package (it may need torch), so it does two quick things itself and hands the
rest to the package, run in the right environment:

1. `kind` is set (unless `--mock`): from `self.config["kind"]`. Unset → `fail`: `policy.kind is not set: choose
   diffusion or hold_still in station.yaml`; the self-check is then skipped.
2. `port_free("policy", station.endpoint("policy"))`.
3. **Self-check:** runs `[program()[0], "-m", "policy.self_check", "--station", <station file>]` (+ `--mock`) with
   `self_check_timeout_s`, and turns each line it prints into a `CheckResult`. If it times out, crashes, or prints
   something that isn't a check line → one `fail` with the last lines of its stderr (e.g. the policy Python missing the
   package: `pip install -e policy` in that environment).

`python -m policy.self_check` (`self_check.py`) prints one JSON object per line, `{"name", "status", "detail"}`
(`status` as in `CheckResult`), and exits 0 unless it crashes. It never loads the checkpoint (2.5 GB, too slow) and
never binds a socket:

| Active kind | Checks |
|---|---|
| all | `settings`: `Settings.load` passes (every problem listed); `kind` (info: which policy runs, and `--mock`) |
| `diffusion` | `checkpoint`: set, exists, is a readable file ending in `.ckpt` (info: size, modification time); `imports`: `torch`, `dill`, `hydra`, `timm`, `diffusers` and `behavior_prompting` importable (info: torch version); `device`: `cuda` → `torch.cuda.is_available()` (info: GPU name and memory; fail with the CPU-backup hint), `cpu` → info `device: cpu (PLAN §9 backup)`; `diffusion_steps`: info when set (`diffusion steps: 8, PLAN §9 backup`) |
| `hold_still` | none beyond `settings` |

With `--mock` the self-check runs in the light environment for `hold_still`, so there are no checkpoint, GPU or
policy-environment checks (deploy §8).

## 9. `__main__` and running it alone

```
python -m policy --station FILE [--session DIR] [--ready-file FILE] [--mock]
```

Exactly the common arguments of deploy §4.2, nothing else (every other setting is in `station.yaml`).
`--station` is required. Without `--session` it writes into `<session_root>/<YYYY-MM-DD_HH-MM-SS>_policy_alone/policy/`,
the same `<session>/policy/` layout as under `deploy run`; without `--ready-file` it writes none.

```
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace
def main(argv: Sequence[str] | None = None) -> int     # PolicyApp(...).run()
```

Troubleshooting by hand:

```
~/envs/policy/bin/python -m policy --station deployment/station.yaml           # the real kind, in its terminal
python -m policy --station deployment/station.yaml --mock                       # hold_still, light environment
python -m policy.probe --station deployment/station.yaml [--infers 10]          # in another terminal
```

`probe.py` (light environment, no torch) connects to `endpoints.policy` with `contracts.RpcClient` and, like
`message_comms`, sends `config`, `reset` (5 times, timed), and `infer` with `neutral_observation()` (N times, timed),
then prints one line per step (`PASS reset 0.4 ms (max of 5)`, `PASS infer 87 ms mean / 95 ms max`, the total
`width_clipped`, or the `ErrorReply`). It leaves the policy reset. It sends no prompt unless `config` says
`uses_prompt` (then it needs `--task-name` and `--prompt-index`).

## 10. Unit tests

Run from `deployment/` in **both** environments:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest policy/tests                    # light: torch tests skipped
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/policy/bin/python -m pytest policy/tests  # policy env: all
```

Torch tests start with `torch = pytest.importorskip("torch")`; the one that needs `behavior_prompting` also has
`pytest.importorskip("behavior_prompting")`. So the light run skips them with a reason, and proves `hold_still` works
without torch. No test needs a GPU, a checkpoint or the network beyond `127.0.0.1`; sockets bind `tcp://127.0.0.1:*`.
Time is driven by injected clocks; nothing sleeps to wait.

C4 example observations are generated during the tests (`neutral_observation()`, the `contracts` example builder,
and fixtures in `conftest.py`); none are stored as files.

| File | Torch | Checks |
|---|---|---|
| `test_settings.py` | no | `config.yaml` loads with all defaults; station overrides merge (maps key by key); an unknown key is refused with its path (`policy.chekpoint`); `kind` unset → error naming `policy.kind`, but fine with `--mock`; unknown kind; `diffusion` without `checkpoint` refused; ranges (`diffusion_steps: 0`); relative paths resolved from the station's folder, defaults from `policy/`; `endpoints.policy` read from the station, and missing → error naming it; `session_root` read only without `--session` |
| `test_make_chunk.py` | no | widths inside the range unchanged and `width_clipped == 0`; widths of −0.001 and 0.085 clipped to 0 and the max opening, `width_clipped` counts exactly the changed steps; the max comes from `ContractLimits`; the result passes the C4 validator; **with a max of 0.07** (where `np.float32(0.07)` rounds up above 0.07) a width of 0.09 is clipped to `np.nextafter(np.float32(0.07), np.float32(0))`, which is ≤ 0.07, and the chunk passes the validator; with 0.08 the clip value is `np.float32(0.08)` itself; an **odd chunk** (step 0 at 5 cm / 10° from the anchor) is returned unchanged and valid, not refused |
| `test_key_map.py` | no | **round trip**: contract → model → contract names for every field; `split_action` then `join_action` gives back the 10-column vector; `action_slices` from `run_001`'s `components`; `check` passes for `run_001`'s and a plain-diffusion `shape_meta`, and fails naming the key for: a used model input with no contract field, a mapped key missing from `shape_meta`, an unknown contract field, a component size ≠ 3/6/1, a non-one-to-one map; `used_fields` leaves out `eef_rot6d_wrt_start` for `run_001` |
| `test_hold_still_policy.py` | no | **reply passes the C4 validator** (and survives `encode`/`decode`); zeros / identity rot6d / the newest width ×16; a width above the max clipped and counted; `config()` gives `OBS_STEPS`, `ACTION_STEPS`, `STEP_MS`, `IMAGE_SIZE` from `contracts/sizes.py`, `uses_prompt: false`, `inputs: ["gripper_width_m"]`; **`reset` < 0.5 s** |
| `test_policy_server.py` | no | over `tcp://127.0.0.1:*` with `contracts.RpcClient`: `config` reply; `reset` → `Ack` in **< 0.5 s**; **invalid requests give a clear `ErrorReply`, not a crash**, and the server keeps serving afterwards: wrong image shape (names the field), wrong schema version, an unknown request type (`c1.get_frames/v1`), **`infer` before the first `reset`**; **`prompt` is refused for a non-promptable policy** with the stated text; with `FakePromptablePolicy`: `infer` before the prompt refused, the prompt accepted, a second prompt refused, and over three reset/prompt/infer cycles the fake **receives exactly one prompt after each reset**; `RaisingPolicy` (raises `RuntimeError` in `infer`) and `BadChunkPolicy` (returns NaN) → the request gets `ErrorReply`, then `serve` raises `PolicyFatal`; `OddChunkPolicy` (step 0 at 5 cm from the anchor) → a normal `ActionChunk` reply and the server keeps serving; a `BadRequest` from a policy keeps serving; the stop event returns within `poll_s` |
| `test_neutral_observation.py` | no | passes the C4 validator; anchor rows exact; images black |
| `test_policy_factory.py` | no | each kind builds its class; `KIND` matches; choosing `diffusion` where torch is blocked gives the stated `PolicyLoadError` |
| `test_import_without_torch.py` | no | in a subprocess with an import blocker for `torch`, `behavior_prompting`, `hydra`, `dill`, `timm`, `diffusers`: `import policy`, building `hold_still`, `PolicyServer`, `self_check` and `probe` all work, and none of the blocked modules was imported |
| `test_app.py` | no | `python -m policy --mock` as a subprocess (light env, `hold_still`): the ready file appears only after the port answers; `startup.json` written; `config` answered; SIGINT → exit 0, no FATAL line; a broken station section → exit 1 with one `FATAL policy: ...` last line; no `--session` → the `_policy_alone` folder |
| `test_warm_up.py` | no | with a fake policy whose warm-up chunk fails a C4 value check (non-unit rot6d): a warning is logged and start-up continues; with one that raises another error (shape, exception): fatal |
| `test_self_check.py` | no | JSON-line format; `hold_still` all pass; `diffusion` with torch blocked → `fail` with the "policy environment" hint; missing checkpoint → `fail` |
| `test_launch.py` | no | `program()` uses `python.light` with `--mock` and `python.policy` otherwise; unset `kind` → `fail` and no self-check run; busy port → `fail`; self-check lines become `CheckResult`s; a self-check that crashes or times out → one `fail` with its stderr; `--mock` checks need no checkpoint, GPU or policy Python |
| `test_probe.py` | no | against a `hold_still` `PolicyServer` in the same test: all steps pass; against `RaisingPolicy`: the `ErrorReply` is printed |
| `test_diffusion_policy.py` | **yes** | `DiffusionPolicy(ModelRunner(TinyDiffusionModel, ...))` on the CPU with a hand-written plain-diffusion `shape_meta`: **chunk shapes (16, 3) / (16, 6) / (16, 1), dtype float32, C-contiguous, and it passes the C4 validator**; the model receives exactly the used keys under the model's names, float32, shapes (1, 2, 3, 224, 224) and (1, 2, D); a known pixel `(y, x, c)` in the uint8 image arrives at `[0, t, c, y, x]` with value `v / 255` (bit-identical to the numpy reference); horizon 20 → the first 16 rows are kept; horizon 12 → refused at construction; an out-of-range model width is clipped and counted; `reset` calls the model's `reset` and takes < 0.5 s; `ModelRunner` rejects a `shape_meta` with 112×112 images (no resize) |
| `test_checkpoint_loader.py` | **yes** (+ `behavior_prompting`) | the load checks on hand-made `cfg`/`shape_meta` objects without a real checkpoint: prompting refused by `DiffusionPolicy`, absolute pose representation refused, timing mismatch refused, `diffusion_steps` above `num_train_timesteps` or with a non-DDIM scheduler refused, a valid override sets `num_inference_steps`; the colour-normalisation check on both encoder types; missing file and wrong suffix; `device: cuda` with CUDA reported unavailable (monkeypatched) → the stated error. **Optional, skipped by default:** with `POLICY_TEST_CHECKPOINT=<path>` set to `run_001`'s slim checkpoint, load it on the CPU through `CheckpointLoader` only (not `DiffusionPolicy`, which refuses a prompting checkpoint) and check the strict load report, the colour-normalisation check and that no simulation modules were imported. A full inference test on a real checkpoint waits for a rig-trained plain-diffusion model. |

`tests/fakes.py`:

- `FakePromptablePolicy(PromptablePolicy)`: records every `prompt` and `reset` call; replies like `HoldStillPolicy`.
- `RaisingPolicy`, `BadChunkPolicy`: for the fatal path.
- `OddChunkPolicy`: a valid chunk whose step 0 is far from the anchor, to show it is not the policy's failure.

`tests/fakes_torch.py`: `TinyDiffusionModel(torch.nn.Module)`: one small linear layer; `predict_action(obs)` asserts
its inputs (keys, dtypes, shapes, image range [0, 1]), records them, and returns a deterministic `action_pred`
(1, H, 10) whose rot6d columns are identity and positions small, so the chunk is valid (one test sets a width outside
the range); `reset()` counts calls; `num_inference_steps` attribute.

## 11. Dependencies (`pyproject.toml`)

Python 3.10 (`requires-python = ">=3.10,<3.11"`), deploy §7.3. Installed editable in **both** environments
(`pip install -e policy`); the light environment gets only the base dependencies.

| Package | Version | Why |
|---|---|---|
| `contracts` | the local package (installed editable first) | messages, validators, limits, the C4 sizes `OBS_STEPS`, `ACTION_STEPS`, `STEP_MS`, `IMAGE_SIZE` (`contracts/sizes.py`), `RpcServer` / `RpcClient` |
| numpy | `>=1.26,<2.3` (as `contracts`) | 1.26.4 in the policy environment, 2.2.6 in the light one |
| pyyaml | `==6.0.3` | `config.yaml`, `station.yaml` |
| `[model]` extra: torch, dill, hydra-core, timm, diffusers | not pinned here | `DiffusionPolicy` only; versions come from `behavior_prompting/environment.yml` (torch 2.7.1, dill 0.3.7, hydra-core 1.2.0, timm 1.0.20, diffusers 0.35.1). `behavior_prompting` itself isn't on PyPI and is not listed; the self-check verifies it is importable |
| `[test]` extra: pytest | `==8.3.5` | tests |

`[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "--import-mode=importlib"` (deploy §7.4).

## 12. Adding prompting later (`BehaviorPromptingPolicy`)

1. **One new class**, `policy/policy/behavior_prompting_policy.py`: `BehaviorPromptingPolicy(PromptablePolicy)`,
   `KIND = "behavior_prompting"`. `from_settings` uses the same `CheckpointLoader` (accepting `use_prompting: true`) and
   the same `ModelRunner`; `infer` goes through `make_chunk` like every policy; `prompt()` loads the demo prompt from a
   prompt dataset (the `_handle_prompt` and `attach_prompt_dataset` logic of `detached_inference.py` / main-sim
   `server.py`) and raises `BadRequest` for an unknown task or index; `reset()` clears the prompt.
2. **One line** in `policy_factory.KINDS`, and its keys in `config.yaml` (e.g. `prompt_dataset: null`).
3. **One line** in `station.yaml`: `kind: behavior_prompting` (plus the dataset path).

Nothing else changes: `config()` reports `uses_prompt: true` because the class is a `PromptablePolicy`,
`PolicyServer` already enforces one prompt after each reset, and `message_comms` already sends `prompt` when `config`
says so (PLAN §6.4).

## 13. Requirements on other folders

These are designed in here and must match the other folders' STRUCTURE.md files:

- **`message_comms/`:** adds up `width_clipped` per episode and shows it on the status line (PLAN §5, normal
  operation), and checks step 0 against the anchor itself (odd chunk, section 3.5).
