"""The eval driver (DESIGN.md section 9, PLAN.md section 7): N seeded rollouts of one policy against the running sim.

    python -m sim_deploy.eval run --policy <RUN_FOLDER | hold_still | replay:<zarr>#<idx>> --rollouts N --seed S
                                  [--starts random | set | set:<FILE>] [--tag T] [--timeout 60] [--tap] [--dry-run]
                                  [--only-rollouts I,J] [--resume <EVAL>] [--max-new-rollouts K]

Per eval: a frozen copy of sim_deploy/stations/policy_sim.yaml in the eval folder (session_root -> the eval folder,
optional manager.save_tap), the policy set with the copy's `policy_deploy use <target> --station <copy>`, then
`policy_deploy run --mock --station <copy>` in a pty. Per rollout (`--home start`, the default, P2): h (home: the
laptop's key if the copy still has it, else the PC's key through scene REP :5651 `key`), wait for idle at home, the
PC's key p (scene `key`), wait for C2 `homing` then `idle` with every joint within homing.at_pose_tol_rad of
stations/start_pose.yaml's q_start_rad, the eraser placed relative to the start pose (scene/randomization_start.yaml,
the replay placement, or with `--starts set` start n of the start set eval/start_sets/w3.yaml), settle, c.
`--home deploy`: h only, the eraser in scene/randomization.yaml (the sim PC then runs with q_start_rad null: run.sh). Then watch ground truth (scene) + C2 until success / timeout / fault / stop, then
y or n. After the last one q, and the results under sim_evals/<task>/<run_folder>/<YYYY-MM-DD_HHMM>_<tag>/ + a row in
sim_evals/index.csv.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO, ROOT, looks
from sim_deploy.eval import gt_recorder as gtr
from sim_deploy.eval import start_set as sst
from sim_deploy.eval import report
from sim_deploy.eval.target import Target, resolve
from sim_deploy.evaltool import constants as C
from sim_deploy.model.build import git_commit, model_sha256
from sim_deploy.scene import start_pose as sp
from sim_deploy.scene.placement import HOME_ALIASES, HOMES, Scene, canonical_home

SIM_EVALS = REPO / "sim_evals"
POLICY_STATION = ROOT / "stations" / "policy_sim.yaml"
PD_DIR = REPO / "deployment" / "policy_deploy"
PD_ENV = Path("~/envs/sim_policy_deploy").expanduser()
HOME_DONE = re.compile(r"home pose reached")
REFUSED = re.compile(r"^(c|h): ")
NOT_AT_START = re.compile(r"not at the start pose")          # the PC refuses C3 start away from a set start pose

# the live judge only decides when to stop (evaltool/SCHEMA.md); success = the eraser centre >= its settled z + lift_m
# and its lowest corner >= table top + clear_m, both held hold_s continuously, grasp not required (user decisions
# 2026-11-22). The success values and the clearance test (C.clear_of_table) come from evaltool/constants.py and are
# tested on the gt.jsonl row as written, so the rollout stops exactly where `evaltool score` (which gives the scores)
# counts the success; off_table_m and poll_hz are the driver's own.
JUDGE = {"lift_m": C.LIFT_M, "hold_s": C.HOLD_S, "hold_tol_s": C.HOLD_TOL_S, "clear_m": C.CLEAR_M, "lifted_m": C.LIFTED_M,
         "off_table_m": 0.03, "poll_hz": C.GT_RATE_HZ, "success_needs_grasp": C.SUCCESS_NEEDS_GRASP}


class EvalError(RuntimeError):
    pass


EXIT_SIM_FAULT = 3          # E7: the sim PC latched a fault; run.sh restarts the sim and resumes (--resume)
PC_FAULT = re.compile(r"the PC reports fault")
HOME_REFUSED_FAULT = re.compile(r"key h refused: mode is fault")    # the sim PC latched a fault before the rollout


def log(msg: str) -> None:
    """One driver line on stdout. A closed stdout (run.sh's tee gone after a Ctrl-C) must not stop the driver from
    writing its results: the line then goes to stderr, or nowhere."""
    for stream in (sys.stdout, sys.stderr):
        try:
            print(f"[eval] {msg}", file=stream, flush=True)
            return
        except (BrokenPipeError, OSError, ValueError):
            continue


# --------------------------------------------------------------------------------------------- station freeze


def freeze_station(out: Path, tgt: Target, *, tap: bool, use_flags: list[str]) -> dict:
    """eval/station.yaml = policy_sim.yaml with session_root = eval/session (+ manager.save_tap), then `use`."""
    text = POLICY_STATION.read_text()
    session_root = out / "session"
    text, n = re.subn(r"(?m)^session_root:.*$", f"session_root: {session_root}", text)
    if n != 1:
        raise EvalError(f"{POLICY_STATION}: no single session_root line")
    if tap:
        text, n = re.subn(r"(?m)^manager: \{\}.*$", "manager: {save_tap: true, tap_images_every: 10}", text)
        if n == 0:   # a block `manager:` with 2-space children (W0: exec_h 12 as upstream's station)
            text, n = re.subn(r"(?m)^manager:[ \t]*(#.*)?$",
                              lambda m: m.group(0) + "\n  save_tap: true\n  tap_images_every: 10", text)
        if n != 1:
            raise EvalError(f"{POLICY_STATION}: --tap needs one 'manager: {{}}' or 'manager:' block line")
    header = (f"# frozen by sim_deploy.eval {dt.datetime.now().isoformat(timespec='seconds')} from "
              f"{POLICY_STATION.relative_to(REPO)}; then `policy_deploy use {tgt.use_arg}`\n")
    st = out / "station.yaml"
    st.write_text(header + text)
    cmd = [str(PD_ENV / "bin" / "policy_deploy"), "use", tgt.use_arg, "--station", str(st), *use_flags]
    r = subprocess.run(cmd, cwd=PD_DIR, capture_output=True, text=True, env=_env())
    (out / "use.log").write_text(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        raise EvalError(f"policy_deploy use failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-800:]}")
    doc = yaml.safe_load(st.read_text())
    return doc


def _env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


# ------------------------------------------------------------------------------------------------- the eval


class Eval:
    def __init__(self, args) -> None:
        self.a = args
        if args.model is None:                            # the look's build (run.sh passes both)
            args.model = str(looks.build_dir(args.look) / "model.json")
        elif looks.look_of_model(args.model) not in (None, args.look):
            raise EvalError(f"--model {args.model} is the build of look {looks.look_of_model(args.model)!r}, "
                            f"not of --look {args.look}")
        self.tgt = resolve(args.policy)
        requested = args.home
        args.home = canonical_home(args.home)              # P2: trainstart is an alias of start
        if requested != args.home:
            log(f"--home {requested} is now --home {args.home} (the start pose, PC key p)")
        self.scene = Scene.load(args.model, args.random, home=args.home,
                                start_pose=getattr(args, "start_pose", None))   # P4: --start-pose FILE
        cfg = sp.effective_config(sp.CONTROL_SIM)
        self.q_home = [float(x) for x in cfg["homing"]["q_home_rad"]]
        self.at_pose_tol = float(cfg["homing"].get("at_pose_tol_rad", sp.DEFAULT_AT_POSE_TOL))
        if args.home_key == "auto":
            args.home_key = "laptop" if sp.laptop_has_home_key() else "pc"
        self.model_dir = Path(args.model).parent
        now = dt.datetime.now()
        root = SIM_EVALS / "_dry_run" if args.dry_run else SIM_EVALS
        self.rollouts: list[dict] = []
        self.problems: list[str] = []
        self.sim_fault = False
        self.resumed = getattr(args, "resume", None)
        self.launcher = os.environ.get("SIM_DEPLOY_LAUNCHER") or "cli"      # evaltool/SCHEMA.md
        self.batch_id = os.environ.get("SIM_DEPLOY_BATCH_ID") or None
        if self.resumed:                                  # E7: continue an eval whose sim PC latched a fault
            self.out = Path(self.resumed).expanduser().resolve()
            doc = yaml.safe_load((self.out / "eval.yaml").read_text())
            self.created = doc["created"]
            f = self.out / "rollouts.jsonl"
            done = [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []
            self.rollouts = [r for r in done if r.get("outcome") != "not_run"]
            return
        self.out = root / self.tgt.task / self.tgt.run_folder / f"{now:%Y-%m-%d_%H%M}_{args.tag}"
        if self.out.exists():
            self.out = self.out.with_name(self.out.name + f"_{now:%S}")
        self.created = now.isoformat(timespec="seconds")

    def indices(self) -> list[int]:
        """The rollout indices this eval runs: 0..rollouts-1, or --only-rollouts (same seed key [seed, n] / set start
        n, so the same placements as in a full run)."""
        only = getattr(self.a, "only_rollouts", None)
        return list(only) if only else list(range(self.a.rollouts))

    # ---- preparation (also the dry run) ----------------------------------------------------------------------
    def prepare(self) -> dict:
        a, tgt = self.a, self.tgt
        if self.resumed:
            self.doc = yaml.safe_load((self.out / "eval.yaml").read_text())
            self.replay = self.doc["scene"].get("replay_placement")
            self.starts = self.doc["scene"].get("start_set")     # the starts as frozen at the eval's start
            self.start_check = None
            self.tgt.kind = self.doc["policy"]["kind"]
            self.launcher = self.doc.get("launcher", self.launcher)
            self.batch_id = self.doc.get("batch_id", self.batch_id)
            a.rollouts, a.seed, a.tag = int(self.doc["rollouts"]), int(self.doc["seed"]), self.doc["tag"]
            a.only_rollouts = self.doc.get("only_rollouts")
            log(f"resuming {self.out}: {len(self.rollouts)} of {len(self.indices())} rollouts done")
            return self.doc
        self.out.mkdir(parents=True)
        (self.out / "rollouts").mkdir()
        sha, _ = model_sha256(self.model_dir)
        if sha != self.scene.model["model_sha256"]:
            raise EvalError(f"{self.model_dir}: files do not match model.json (sha {sha[:12]} vs "
                            f"{self.scene.model['model_sha256'][:12]}); rebuild: python -m sim_deploy.model build")
        use_flags = []
        for flag, v in (("--precision", a.precision), ("--diffusion-steps", a.diffusion_steps), ("--exec-h", a.exec_h)):
            if v is not None:
                use_flags += [flag, str(v)]
        station = freeze_station(self.out, tgt, tap=a.tap, use_flags=use_flags)
        pol = station.get("policy", {})
        kind = pol.get("mock_kind") or pol.get("kind")
        if kind == "diffusion":
            tgt.kind = "prompting" if pol.get("prompt_dataset") else "diffusion"
        self.replay = None
        if tgt.kind == "replay":
            self.replay = self.scene.replay_placement(tgt.raw, a.behind_tip)
        self.starts, self.start_check = self._start_set()
        seeds = [[a.seed, n] for n in self.indices()]
        doc = {
            "schema": "sim_deploy.eval/v2", "created": self.created, "tag": a.tag, "dry_run": bool(a.dry_run),
            "command": " ".join(sys.argv),
            "policy": {"raw": tgt.raw, "kind": tgt.kind, "task": tgt.task, "run_folder": tgt.run_folder,
                       "checkpoint": tgt.checkpoint, "dataset": tgt.dataset, "use_arg": tgt.use_arg,
                       "run_info": tgt.info, "station_policy": station.get("policy"),
                       "station_manager": station.get("manager")},
            "rollouts": len(seeds), "seed": a.seed, "rollout_seeds": seeds, "only_rollouts": a.only_rollouts,
            "judge": {**JUDGE, "timeout_s": a.timeout},
            "home": self._home_doc(),
            "scene": {"randomization": {k: v for k, v in self.scene.random.items() if not k.startswith("_")},
                      "randomization_file": self.scene.random["_path"], "station_scene": self.scene.sc,
                      "replay_placement": self.replay, "placements": self._placements_kind(),
                      "start_set": self.starts},
            "look": {"name": a.look, "station": str(looks.station(a.look).relative_to(REPO)),
                     "model_dir": str(self.model_dir)},
            "model": {"dir": str(self.model_dir), "sha256": sha, "station": self.scene.model["station_file"]},
            "station_file": str(self.out / "station.yaml"),
            "git": {"sim_deploy": git_commit(), "training": tgt.training_commit},
            "endpoints": {"scene": a.scene_endpoint, "c2": a.c2_endpoint, "state": a.state_endpoint},
        }
        safety = sp.effective_config(sp.CONTROL_SIM).get("safety") or {}
        if self.starts is not None:                       # --starts set: every rollout's start and pose is hashed
            src, rules = f"set:{self.starts['name']}", self.starts["protocol_rules"]
        elif self.replay:
            src, rules = f"replay:{tgt.raw}", None
        else:
            src = f"random:{Path(self.scene.random['_path']).name}"
            rules = json.loads(json.dumps(doc["scene"]["randomization"]))
        prot = gtr.protocol(
            seed=a.seed, placement_source=src, placement_rules=rules,
            timeout_s=a.timeout, precision=a.precision, diffusion_steps=a.diffusion_steps, exec_h=a.exec_h,
            look=a.look, model_sha256=sha, q_start_rad=self.scene.start["q"] if a.home == "start" else None,
            safety=safety, seed_key=gtr.SEED_KEY_SET if self.starts is not None else gtr.SEED_KEY_RANDOM)
        doc.update(record_schema=gtr.RECORD_SCHEMA, launcher=self.launcher, batch_id=self.batch_id, protocol=prot,
                   protocol_sha=gtr.protocol_sha(prot))
        (self.out / "eval.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        if a.home == "start":                             # P2: the start pose file as it was at the start
            shutil.copy(self.scene.start["file"], self.out / "start_pose.yaml")
        self.doc = doc
        return doc

    def _placements_kind(self) -> str:
        return "replay" if self.replay else "set" if self.starts is not None else "random"

    def _start_set(self) -> tuple[dict | None, list | None]:
        """--starts set[:FILE]: the start-set file (eval/start_set.py) as frozen into eval.yaml, with this checkpoint's
        held-out rollouts, and each listed start recomputed in this eval's scene (model + start pose). Refused
        (EvalError) outside a dry run when a start differs from the file or fails a check, the policy is a replay,
        the home is not the start pose, the file has fewer starts than the rollouts, or the checkpoint is not one
        whose training split the file was built with (its held-out demo would be unknown)."""
        a, tgt = self.a, self.tgt
        spec = getattr(a, "starts", "random") or "random"
        if spec == "random":
            return None, None
        if spec.startswith("demos"):
            raise EvalError(f"--starts {spec!r}: the per-dataset demo starts were replaced by the shared start set "
                            f"(2026-11-22): --starts set[:FILE]")
        if spec != "set" and not spec.startswith("set:"):
            raise EvalError(f"--starts {spec!r}: random, set or set:<FILE | name>")
        if tgt.kind == "replay":
            raise EvalError("--starts set: a replay always uses its own replay placement")
        if a.home != "start":
            raise EvalError("--starts set needs --home start (the file is built at the start pose)")
        path = sst.resolve_file(spec[len("set:"):] if spec.startswith("set:") else None)
        if not path.exists():
            raise EvalError(f"{path}: no start-set file (python -m sim_deploy.eval starts build)")
        doc = sst.load(path)
        n_max = max(self.indices()) + 1
        if n_max > len(doc["starts"]):
            raise EvalError(f"{path.name} has {len(doc['starts'])} starts; rollout {n_max - 1} has none")
        ck = None
        if tgt.kind in ("diffusion", "prompting"):
            ck = sst.checkpoint_entry(doc, tgt.use_arg)
            if ck is None:
                raise EvalError(f"{path.name}: built without {tgt.run_folder}'s training split (its held-out demo is "
                                f"unknown): python -m sim_deploy.eval starts build --checkpoints ... with it")
        check = sst.verify(doc, self.scene)
        bad = [f"rollout {c['rollout']}: {'; '.join(c['problems'])}" for c in check
               if c["problems"] and c["rollout"] in self.indices()]
        if bad and not a.dry_run:
            raise EvalError(f"{path.name} does not hold in this scene (look {a.look}): {' | '.join(bad)}; rebuild: "
                            f"python -m sim_deploy.eval starts build")
        held = list(ck["held_out"]) if ck else []
        frozen = {"file": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                  "file_sha256": doc["_sha256"], "name": doc["name"], "built": doc["built"], "rule": doc["rule"],
                  "demo_source": {k: v for k, v in doc["demo_source"].items() if k != "shared_with"},
                  "real_trial_source": doc["real_trial_source"], "checkpoint_split": ck,
                  "held_out_rollouts": held, "protocol_rules": sst.protocol_rules(doc),
                  "starts": [{**s, "held_out": (s["rollout"] in held) if ck else None} for s in doc["starts"]]}
        return frozen, check

    def _home_doc(self) -> dict:
        """eval.yaml home: the variant, how h is pressed, the home q, and the start pose used (P2)."""
        a = self.a
        doc = {"variant": a.home, "home_key": a.home_key, "q_home_rad": self.q_home,
               "at_pose_tol_rad": self.at_pose_tol, "rollout_start_O_T_tcp": self.scene.home_tcp().tolist(),
               "sim_pc_station": os.environ.get("SIM_DEPLOY_SIM_PC_STATION")
               or "run.sh: build/control_sim_<home>_<look>.merged.yaml = stations/control_sim.yaml + "
                  + ("stations/start_pose.yaml" if a.home == "start" else "q_start_rad: null")}
        if a.home == "start":
            info = sp.describe(self.scene.start["q"], a.model)
            doc["start_pose"] = {
                "file": self.scene.start["file"], "q_start_rad": self.scene.start["q"],
                "file_text": Path(self.scene.start["file"]).read_text(),
                **{k: info[k] for k in ("tcp_O_m", "fingertip_dir_O", "tilt_deg", "heading_deg", "heading_words",
                                        "height_above_table_m", "cone_tilt_deg", "move_time_s", "ok", "vs_demos")},
                "checks": {k: c["ok"] for k, c in info["checks"].items()}, "O_T_tcp": info["O_T_tcp"]}
        return doc

    def dry_run_checks(self) -> list[tuple[str, str, str]]:
        rows = [("model", "pass", f"{self.model_dir} sha {self.doc['model']['sha256'][:16]} = model.json"),
                ("station", "pass", f"{self.out / 'station.yaml'} (use: {self.tgt.use_arg})"),
                ("policy", "pass", f"kind {self.tgt.kind}; checkpoint {self.tgt.checkpoint}")]
        pol = self.doc["policy"]["station_policy"] or {}
        for key in ("checkpoint", "prompt_dataset"):
            if pol.get(key):
                p = Path(str(pol[key])).expanduser()
                rows.append((f"station {key}", "pass" if p.exists() else "FAIL", str(p)))
        if self.replay:
            rp = self.replay
            rows.append(("replay placement", "pass" if rp["on_table"] else "FAIL",
                         f"eraser {np.round(rp['pose'][:3], 3).tolist()} yaw {rp['yaw_deg']:.1f}; grasp "
                         f"{rp['grasp_tip_above_table_m'] * 100:.1f} cm above the table (height error "
                         f"{rp['height_error_m'] * 100:+.1f} cm)"))
        elif self.starts is not None:
            d = self.starts
            ck = d["checkpoint_split"]
            prob = (ck or {}).get("problems") or []
            rows.append(("start set", "pass", f"{d['file']} ({d['name']}, {len(d['starts'])} starts"
                         + (f"; split of this checkpoint: {ck['sample_type']} on {ck['dataset']}, val "
                            f"{[v['index'] for v in ck['val_demos_checkpoint'] or []]} "
                            f"{'= recomputed' if ck['agree'] else 'DIFFERS from recomputed'}; held-out rollout(s) "
                            f"{d['held_out_rollouts']}" if ck else "; no checkpoint split (not a checkpoint)") + ")"))
            if prob:
                rows.append(("start set split", "info", "; ".join(prob)))
            chk = {c["rollout"]: c for c in self.start_check or []}
            for n in self.indices():
                s = sst.start_for(d, n)
                probs = (chk.get(n) or {}).get("problems") or []
                rows.append((f"start {n}", "FAIL" if probs else "pass",
                             sst.start_line(s, s.get("held_out")).split(None, 1)[1]
                             + (f"; {'; '.join(probs)}" if probs else "")))
        else:
            pl = [self.scene.random_placement(self.a.seed, n) for n in self.indices()]
            ok = all(self.scene.on_table(p["pose"]) and self.scene.in_view(p["pose"]) for p in pl)
            n_clip = sum(p["clipped"] for p in pl)
            rows.append(("placements", "pass" if ok else "FAIL",
                         f"{len(pl)} seeded poses on the table, in view at the {self.a.home} pose"
                         + (f"; {n_clip} clipped to the table / box" if self.a.home == "start" else "")))
        h = self.doc["home"]
        rows.append(("home key h", "info", f"{h['home_key']} ({'the laptop has h' if h['home_key'] == 'laptop' else 'scene key h: the copy has no laptop h'})"))
        if self.a.home == "start":
            st = h["start_pose"]
            bad = [k for k, ok in st["checks"].items() if not ok]
            rows.append(("start pose", "pass" if st["ok"] else "FAIL",
                         f"{st['file']}: TCP {st['tcp_O_m']}, {st['height_above_table_m'] * 100:.1f} cm above the "
                         f"table, tilt {st['tilt_deg']:.1f} deg, {st['heading_words']}"
                         + (f"; failed checks: {bad}" if bad else "")))
            rows.append(("key p in the copy", "pass" if sp.config_has_q_start() else "FAIL",
                         "franka_control config.yaml has homing.q_start_rad" if sp.config_has_q_start() else
                         "the copy's config.yaml has no homing.q_start_rad (key p not ported): --home start cannot run"))
        else:
            rows.append(("sim PC station", "info", "run.sh --home deploy merges q_start_rad: null (the real PC "
                         "refuses c away from a set start pose)"))
        r = subprocess.run([str(PD_ENV / "bin" / "policy_deploy"), "check", "--mock", "--station",
                            str(self.out / "station.yaml")], cwd=PD_DIR, capture_output=True, text=True, env=_env(),
                           timeout=300)
        (self.out / "check.log").write_text(r.stdout + "\n" + r.stderr)
        fails = [ln.strip() for ln in r.stdout.splitlines() if re.search(r"\bFAIL\b", ln)]
        rows.append(("policy_deploy check --mock", "info",
                     f"exit {r.returncode}; FAIL rows: {fails or 'none'} (Pi reachable / manager terminal fail "
                     f"without the sim Pi and a tty)"))
        for i, ep in enumerate([5555, 5620, 5630, 5650, 5651]):
            rows.append((f"port {ep}", "info", "in use (a sim process is up)" if _port_busy(ep) else "free"))
        return rows

    # ---- running ---------------------------------------------------------------------------------------------
    def run(self) -> int:
        from sim_deploy.eval.pd_session import EP_ENDED, EP_STARTED, PolicyDeploySession
        from sim_deploy.eval.sim_link import C2Watcher, SceneClient, StateMirror, VideoRecorder

        a = self.a
        self.sc = SceneClient(a.scene_endpoint)
        hello = self.sc.hello()
        if hello.get("model_sha256") != self.doc["model"]["sha256"]:
            raise EvalError(f"the sim PC runs model {str(hello.get('model_sha256'))[:12]}, model.json says "
                            f"{self.doc['model']['sha256'][:12]}: rebuild or restart the sim PC")
        self.c2 = C2Watcher(a.c2_endpoint)
        self.c2.start()
        self.mirror = StateMirror(a.state_endpoint, self.doc["model"]["sha256"])
        self.mirror.start()
        self.video = VideoRecorder(self.model_dir / "scene.xml", self.mirror, fps=a.video_fps)
        try:                                              # gt.jsonl's min_z: the driver's copy of the same build
            self.min_z = gtr.MinZ(self.model_dir / "scene.xml")
        except Exception as e:                            # noqa: BLE001 (gt rows then carry min_z null)
            self.min_z = None
            self.problems.append(f"gt min_z model: {e}")
        if a.video_fps > 0:
            self.video.start()
        self._EP_ENDED, self._EP_STARTED = EP_ENDED, EP_STARTED
        self._Session = PolicyDeploySession
        self.pd = None
        try:
            done = {r.get("n") for r in self.rollouts}
            todo = [n for n in self.indices() if n not in done]
            k = getattr(a, "max_new_rollouts", None)
            if k is not None and len(todo) > k:          # a piece (batch.py piece_rollouts): the rest stay for --resume
                log(f"--max-new-rollouts {k}: running rollouts {todo[:k]} now, {len(todo) - k} left for --resume")
                todo = todo[:k]
            for i, n in enumerate(todo):
                if self.pd is None or not self.pd.alive():
                    if not self._start_pd():
                        why = ("sim PC fault latched (as the real PC: restart it); run.sh restarts the sim and "
                               "resumes" if self.sim_fault else "policy_deploy did not start")
                        for m in todo[i:]:
                            self.rollouts.append(self._not_run(m, why))
                        break
                self._cur = self._gt_f = None
                try:
                    rec = self._rollout(n)
                except KeyboardInterrupt:
                    # E7 (UI runner): keep the rollout in progress, as "interrupted" (not rated: report.metrics
                    # leaves it out of every rate's denominator and counts it under outcomes/interrupted)
                    rec = dict(self._cur) if self._cur else self._not_run(n, "")
                    rec.update(outcome="interrupted", success=False, note=(rec.get("note") or "") + " interrupted (SIGINT)")
                    if getattr(self, "_gt_f", None) is not None and not self._gt_f.closed:
                        self._gt_f.close()
                    if rec.get("wall_start") and not rec.get("wall_end"):
                        rec["wall_end"] = dt.datetime.now().isoformat(timespec="milliseconds")
                        rec["loadavg_end"] = round(os.getloadavg()[0], 2)
                    try:
                        rec["c2"] = self.c2.close_window()
                    except Exception:                    # noqa: BLE001  (no window open yet)
                        pass
                    if self.pd is not None and self.pd.alive() and self.pd.phase == "running":
                        self.pd.send("n")
                        time.sleep(0.5)
                    self.rollouts.append(rec)
                    raise
                if rec.get("outcome") == "home_failed" and HOME_REFUSED_FAULT.search(rec.get("note") or ""):
                    # the sim PC latched a fault between rollouts (e.g. physics fell behind wall time under load:
                    # communication_constraints_violation) and refuses h: not this rollout's fault. Leave it and the
                    # rest of this call not_run and exit 3, so run.sh restarts the sim and resumes them.
                    self.sim_fault = True
                    why = f"sim PC in fault mode before the rollout ({rec['note']}); run.sh restarts the sim and resumes"
                    self.problems.append(why)
                    log(f"rollout {n}: {why}")
                    for m in todo[i:]:
                        self.rollouts.append(self._not_run(m, why))
                    break
                self.rollouts.append(rec)
                (self.out / "rollouts.jsonl").open("a").write(json.dumps(rec) + "\n")
                log(f"rollout {n}: {rec['outcome']} grasped={rec['grasped']} lifted={rec['lifted']} "
                    f"max lift {rec['max_lift_m'] * 100:.1f} cm")
        except KeyboardInterrupt:
            self.problems.append("interrupted (Ctrl-C)")
            log("interrupted: stopping policy_deploy")
        finally:
            if self.pd is not None:
                self.pd.send("s")
                time.sleep(0.3)
                self.pd.quit()
            self.c2.stop()
            self.mirror.stop()
            if a.video_fps > 0:
                self.video.stop()
            self.sc.close()
        self._collect()
        return self._write_results()

    def _start_pd(self) -> bool:
        if self.pd is not None:
            self.pd.quit()
        self.pd = self._Session(self.out / "station.yaml", self.out / "policy_deploy.log", python_env=PD_ENV,
                                cwd=PD_DIR)
        self.sessions = getattr(self, "sessions", [])
        self.pd.start()
        ok = self.pd.wait(lambda: self.pd.phase == "idle" or self.pd.not_started is not None or not self.pd.alive(),
                          self.a.start_timeout)
        if ok and self.pd.phase == "idle":                # E7: the ready line follows the [idle] status line
            self.pd.wait(lambda: self.pd.session_dir is not None, 5.0)
        if self.pd.session_dir:
            self.sessions.append(self.pd.session_dir)
        if not ok or self.pd.phase != "idle":
            why = self.pd.not_started or f"no [idle] within {self.a.start_timeout:.0f} s (phase {self.pd.phase!r})"
            if PC_FAULT.search(why):
                self.sim_fault = True
            self.problems.append(f"policy_deploy start: {why}")
            log(f"policy_deploy did not reach idle: {why} (see {self.out / 'policy_deploy.log'})")
            return False
        log(f"policy_deploy ready (session {self.pd.session_dir})")
        return True

    def _press_until(self, key: str, done, timeout_s: float, retry_s: float = 1.5, abort=None) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and self.pd.alive():
            t = time.monotonic()
            self.pd.send(key)
            if self.pd.wait(lambda: done(t), min(retry_s, end - time.monotonic())):
                return True
            if abort is not None and any(abort.search(m) for m in self.pd.messages_since(t)):
                return False
            if not any(REFUSED.match(m) for m in self.pd.messages_since(t)):
                if self.pd.wait(lambda: done(t), max(0.0, end - time.monotonic())):
                    return True
                return False
        return False

    def _home(self) -> bool:
        """h: the laptop's key (C3 home) when the copy has it, else the PC's key through the scene channel."""
        if self.a.home_key == "laptop":
            return self._press_until("h", lambda t: self.pd.find_since(t, HOME_DONE) is not None
                                     and self.pd.phase == "idle", self.a.home_timeout, retry_s=2.0)
        ok, why = self._pc_key_move("h", self.q_home, self.a.home_timeout)
        if not ok:
            self._cur["note"] = why
        return ok

    def _start_pose(self) -> tuple[bool, str]:
        """p: the PC's start-pose key through the scene channel, from home; done when C2 went homing -> idle and
        every joint is within homing.at_pose_tol_rad of q_start_rad."""
        return self._pc_key_move("p", self.scene.start["q"], self.a.home_timeout)

    def _pc_key_move(self, key: str, q_goal, timeout_s: float) -> tuple[bool, str]:
        from sim_deploy.eval.sim_link import SceneError

        t = time.monotonic()
        try:
            rep = self.sc.key(key)
        except SceneError as e:
            return False, f"key {key}: {e}"
        line = rep.get("log") or ""
        end = t + timeout_s
        seen = False
        while time.monotonic() < end:
            modes = self.c2.modes_since(t)
            seen = seen or "homing" in modes
            mode = self.c2.mode()
            if mode == "fault":
                latest = self.c2.latest
                return False, f"key {key}: C2 fault: {latest.error if latest else ''}"
            if not seen and time.monotonic() - t > 3.0:
                return False, f"key {key}: no C2 homing within 3 s (reply: {line or rep})"
            if seen and mode == "idle":
                latest = self.c2.latest
                err = float(np.max(np.abs(np.asarray(latest.q, float) - np.asarray(q_goal, float))))
                if err <= self.at_pose_tol:
                    return True, line
                return False, (f"key {key}: idle {err:.3f} rad from the expected pose (tol {self.at_pose_tol}): the "
                               f"sim PC runs another {'start pose' if key == 'p' else 'home'}; restart the sim "
                               f"(run.sh merges stations/start_pose.yaml at each start)")
            time.sleep(0.05)
        return False, f"key {key}: not back to idle within {timeout_s:.0f} s (C2 {self.c2.mode()})"

    def _measure_start(self, key: str) -> dict:
        """Where the rollout starts: C2 q and TCP, tilt, height above the table, the largest joint error."""
        latest = self.c2.latest
        if latest is None:
            return {"after": key}
        T = np.asarray(latest.O_T_TCP, float)
        num = sp.tcp_numbers(T, float(self.scene.sc["table"]["top_z_m"]))
        q = np.asarray(latest.q, float)
        return {"after": key, "q": q.round(5).tolist(), "tcp_O_m": num["tcp_O_m"], "tilt_deg": num["tilt_deg"],
                "heading_words": num["heading_words"], "height_above_table_m": num["height_above_table_m"],
                "max_joint_err_rad": float(np.max(np.abs(q - np.asarray(self.scene.home_q(), float)))),
                "gripper_width_m": float(latest.gripper_width_m)}

    def _check_home(self) -> str:
        """After homing the TCP must be at the --home variant's pose: the sim PC was started with the matching
        station (run.sh --home picks it). Returns the problem, or ""."""
        tcp = self.sc.ground_truth().get("tcp")
        if tcp is None:
            return ""
        want = self.scene.home_tcp()[:3, 3]
        err = float(np.linalg.norm(np.asarray(tcp[:3], float) - want))
        if err > 0.01:
            return (f"--home {self.a.home}: after homing the TCP is {np.round(tcp[:3], 3).tolist()}, {err * 100:.1f} cm "
                    f"from that home ({np.round(want, 3).tolist()}): the sim PC runs another home; start it with "
                    f"run.sh --home {self.a.home}")
        return ""

    def _not_run(self, n: int, why: str) -> dict:
        return {"n": n, "seed_key": [self.a.seed, n], "outcome": "not_run", "note": why, "grasped": False,
                "lifted": False, "success": False, "max_lift_m": None}

    def _rollout(self, n: int) -> dict:
        a, J = self.a, JUDGE
        rec = {"n": n, "seed_key": [a.seed, n], "grasped": False, "lifted": False, "success": False,
               "max_lift_m": 0.0, "t_grasp_s": None, "t_lift_s": None, "t_success_s": None, "fault": None,
               "stopped": False, "session_dir": str(self.pd.session_dir), "note": ""}
        self._cur = rec
        rec["home"] = a.home
        t_h = time.monotonic()
        if not self._home():
            rec.update(outcome="home_failed", note=rec["note"] or "no 'home pose reached'")
            return rec
        rec["t_home_s"] = round(time.monotonic() - t_h, 2)
        if a.home == "start":
            t_p = time.monotonic()
            ok, why = self._start_pose()
            if not ok:
                if "another start pose" in why:
                    raise EvalError(why)
                rec.update(outcome="start_pose_failed", note=why)
                return rec
            rec["t_start_pose_s"] = round(time.monotonic() - t_p, 2)
            rec["start_pose"] = self._measure_start("p")
        else:
            bad = self._check_home()
            if bad:
                raise EvalError(bad)
            rec["start_pose"] = self._measure_start("h")
        if self.replay:
            pose = self.replay["pose"]
        elif self.starts is not None:
            s = sst.start_for(self.starts, n)
            pose = s["pose"]
            rec["held_out"] = s.get("held_out")
            rec["placement"] = {"source": s["source"], "start_set": self.starts["name"], "file": self.starts["file"],
                                "rollout_slot": s["rollout"], "id": s["id"], "held_out": s.get("held_out"),
                                **({"demo": s["demo"], "height_error_m": s["height_error_m"],
                                    "approach_tilt_deg": s["approach_tilt_deg"]} if s["source"] == "demo"
                                   else {"real_trial": s["real_trial"]}),
                                "eraser": s["eraser"], "start_frame": s["start_frame"]}
        else:
            pl = self.scene.random_placement(a.seed, n)
            pose = pl["pose"]
            rec["placement"] = {k: v for k, v in pl.items() if k != "pose"}
        rec["eraser_start_pose"] = pose
        self.sc.set_object(pose)
        time.sleep(float(self.scene.random.get("settle_s", 0.3)))
        gt0 = self.sc.ground_truth()
        z0 = float(gt0["objects"]["eraser"]["pose"][2])
        rec["eraser_settled_pose"] = gt0["objects"]["eraser"]["pose"]
        rec["tcp_start"] = gt0.get("tcp")
        t_c = time.monotonic()
        started = self._press_until("c", lambda t: self.pd.find_since(t, self._EP_STARTED) is not None,
                                    a.start_rollout_timeout, abort=NOT_AT_START)
        if not started:
            msgs = self.pd.messages_since(t_c)
            note = "; ".join(msgs[-3:])
            if any(NOT_AT_START.search(m) for m in msgs):
                note += ("  [the sim PC has a start pose set: --home deploy needs run.sh --home deploy (q_start_rad "
                         "null)]" if a.home == "deploy" else "  [the arm is not at the start pose after p]")
            rec.update(outcome="start_failed", note=note)
            return rec
        m = self.pd.find_since(t_c, self._EP_STARTED)
        rec["pd_rollout_id"] = int(m.group("id"))         # policy_deploy's id ("episode N started")
        self.c2.open_window()
        if a.video_fps > 0:
            self.video.start_rollout()
        t0 = time.monotonic()
        rec["wall_start"] = dt.datetime.now().isoformat(timespec="milliseconds")
        rec["loadavg_start"] = round(os.getloadavg()[0], 2)
        rollout_dir = self.out / "rollouts" / f"{n:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        gt_f = self._gt_f = (rollout_dir / "gt.jsonl").open("w")   # SCHEMA gt rows, streamed (kept if the driver dies)
        rec["gt"] = f"rollouts/{n:03d}/gt.jsonl"
        n_gt = 0
        above_since = None
        trace = []
        outcome = None
        period = 1.0 / J["poll_hz"]
        top = float(self.scene.sc["table"]["top_z_m"])
        box = [float(v) for v in self.scene.eraser_size]
        try:
            while True:
                now = time.monotonic()
                t = now - t0
                if t > a.timeout:
                    outcome = "timeout"
                    break
                if not self.pd.alive():
                    outcome, rec["fault"] = "crashed", "policy_deploy exited during the rollout"
                    break
                ended = self.pd.find_since(t0, self._EP_ENDED)
                if ended:
                    outcome = "ended_by_policy_deploy"
                    rec["pd_outcome"], rec["pd_reason"] = ended.group("outcome"), ended.group("reason")
                    break
                mode = self.c2.mode()
                if mode in ("fault", "stopped"):
                    latest = self.c2.latest
                    rec["fault" if mode == "fault" else "note"] = f"C2 {mode}: {latest.error if latest else ''}"
                    rec["stopped"] = mode == "stopped"
                    outcome = mode
                    break
                try:
                    gt = self.sc.ground_truth()
                except Exception as e:                       # noqa: BLE001
                    rec["note"] = f"ground_truth: {e}"
                    outcome = "sim_error"
                    break
                ez = float(gt["objects"]["eraser"]["pose"][2])
                lift = ez - z0
                latest = self.c2.latest
                try:
                    mz = self.min_z(self.mirror.newest()) if self.min_z is not None else None
                except Exception:                            # noqa: BLE001
                    mz = None
                row = gtr.gt_row(t, gt, mz, latest.mode if latest else None, latest.error if latest else None)
                gt_f.write(json.dumps(row) + "\n")
                n_gt += 1
                grasp = gtr.sides_touching(gt.get("contacts", [])) == {"l", "r"}
                touch = sorted({c.get("a") if c.get("b") == "table" else c.get("b") for c in gt.get("contacts", [])
                                if "table" in (c.get("a"), c.get("b"))})
                if touch:                                     # anything but the eraser on the table (fingers, rig, arm)
                    rec["table_contact_polls"] = rec.get("table_contact_polls", 0) + 1
                    rec["table_contact_geoms"] = sorted(set(rec.get("table_contact_geoms", [])) | set(touch))
                if (gt.get("arm") or {}).get("reflex"):
                    rec["reflex_polls"] = rec.get("reflex_polls", 0) + 1
                trace.append([round(t, 3), round(lift, 4), int(grasp), gt.get("finger_gap_m"),
                              [round(float(v), 5) for v in gt["tcp"]] if gt.get("tcp") else None])   # P3: TCP path (G6)
                rec["max_lift_m"] = max(rec["max_lift_m"], lift)
                if grasp and not rec["grasped"]:
                    rec["grasped"], rec["t_grasp_s"] = True, t
                if grasp and lift >= J["lifted_m"] and not rec["lifted"]:
                    rec["lifted"], rec["t_lift_s"] = True, t
                t_row, lift_row = row["t"], row["eraser_pose"][2] - z0    # evaltool score._success, same row values
                if lift_row >= J["lift_m"] and C.clear_of_table(row["eraser_pose"], box, top):
                    if above_since is None:
                        above_since = t_row
                    if t_row - above_since >= J["hold_s"] - J["hold_tol_s"]:
                        rec["success"], rec["t_success_s"] = True, t
                        outcome = "success"
                        break
                else:
                    above_since = None
                if ez < top - J["off_table_m"]:
                    outcome, rec["note"] = "fail", "eraser fell off the table"
                    break
                time.sleep(max(0.0, period - (time.monotonic() - now)))
        finally:
            gt_f.close()
        rec["duration_s"] = time.monotonic() - t0
        rec["gt_rows"] = n_gt
        if outcome != "ended_by_policy_deploy" and self.pd.alive() and self.pd.phase == "running":
            t_k = time.monotonic()
            self.pd.send("y" if rec["success"] else "n")
            self.pd.wait(lambda: self.pd.find_since(t_k, self._EP_ENDED) is not None, 10.0)
        if outcome == "ended_by_policy_deploy":          # the manager ended it itself (a stop, an abort)
            outcome = "stopped" if rec.get("pd_outcome") in ("stopped", "aborted") else "fail"
            rec["stopped"] = outcome == "stopped"
        rec["outcome"] = outcome
        rec["c2"] = self.c2.close_window()
        rec["wall_end"] = dt.datetime.now().isoformat(timespec="milliseconds")
        rec["loadavg_end"] = round(os.getloadavg()[0], 2)
        if rec["c2"].get("modes", {}).get("fault"):
            rec["fault"] = rec["fault"] or f"C2 fault: {rec['c2'].get('errors')}"
        (rollout_dir / "trace.json").write_text(json.dumps({"columns": ["t_s", "lift_m", "grasped", "finger_gap_m", "tcp_O_xyz_wxyz"],
                                                       "rows": trace}))
        if a.video_fps > 0:
            frames = self.video.stop_rollout(rollout_dir / "external.mp4")
            if frames:
                rec["video"] = f"rollouts/{n:03d}/external.mp4"
                rec["video_frames"] = frames
                rec["video_fps"] = getattr(self.video, "last_rate", None)
            elif self.video.error:
                rec["note"] = (rec["note"] + "; " if rec["note"] else "") + self.video.error
        self._write_rollout_json(rec)                     # rewritten in _collect with steps and clamps
        return rec

    def _write_rollout_json(self, rec: dict) -> None:
        rollout_dir = self.out / "rollouts" / f"{rec['n']:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        doc = gtr.rollout_doc(rec, launcher=self.launcher, batch_id=self.batch_id)
        (rollout_dir / "rollout.json").write_text(json.dumps(doc, indent=1))

    # ---- after -----------------------------------------------------------------------------------------------
    def _collect(self) -> None:
        """Per rollout: its line of the session's manager/episodes.jsonl (policy_deploy calls a rollout an episode;
        pd_rollout_id = its episode_id), its steps.jsonl lines, its tap folder."""
        for rec in self.rollouts:
            sd, eid = rec.get("session_dir"), rec.get("pd_rollout_id")
            if not sd or eid is None or sd == "None":
                continue
            sd = Path(sd)
            rollout_dir = self.out / "rollouts" / f"{rec['n']:03d}"
            for f in sd.rglob("episodes.jsonl"):           # policy_deploy's file name
                for line in f.read_text().splitlines():
                    d = json.loads(line)
                    if d.get("episode_id") == eid:
                        rec["session_rollout"] = d
                        (rollout_dir / "session_rollout.json").write_text(json.dumps(d, indent=1))
            steps = []
            for f in sd.rglob("steps.jsonl"):
                steps += [d for d in map(json.loads, f.read_text().splitlines()) if d.get("episode_id") == eid]
            if steps:
                (rollout_dir / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
                rec["steps"] = _step_summary(steps)
            for tap in sd.rglob(f"tap/ep_{eid:04d}"):
                shutil.copytree(tap, rollout_dir / "tap", dirs_exist_ok=True)
                rec["tap"] = f"rollouts/{rec['n']:03d}/tap"
            rec["clamps"] = gtr.count_clamps(str(sd), eid, rec.get("wall_start"), rec.get("wall_end"))
        for rec in self.rollouts:                         # SCHEMA rollout.json for every rollout, run or not
            if rec.get("n") is not None:
                self._write_rollout_json(rec)
        (self.out / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rollouts))

    def _write_results(self) -> int:
        met = report.metrics(self.rollouts)
        met["problems"] = self.problems
        (self.out / "metrics.json").write_text(json.dumps(met, indent=1))
        (self.out / "summary.html").write_text(report.summary_html(self.doc, met, self.rollouts))
        s = met["success"]
        report.append_index(SIM_EVALS / "index.csv", {
            "date": self.created, "task": self.tgt.task, "run_folder": self.tgt.run_folder, "policy": self.tgt.raw,
            "kind": self.tgt.kind, "dataset": self.tgt.dataset or "", "rollouts": met["rollouts"], "seed": self.a.seed,
            "tag": self.a.tag, "look": (self.doc.get("look") or {}).get("name", ""), "successes": s["k"], "success_rate": f"{s['rate']:.3f}" if s["rate"] is not None else "",
            "ci95_lo": f"{s['ci95'][0]:.3f}", "ci95_hi": f"{s['ci95'][1]:.3f}",
            "grasp_rate": f"{met['grasped']['rate']:.3f}" if met["rollouts"] else "",
            "lift_rate": f"{met['lifted']['rate']:.3f}" if met["rollouts"] else "", "faults": met["faults"],
            "stops": met["stops"], "timeouts": met["timeouts"], "eval_dir": str(self.out.relative_to(SIM_EVALS))})
        log(f"{s['k']}/{met['rollouts']} successes; results in {self.out}")
        if self.sim_fault and any(e.get("outcome") == "not_run" for e in self.rollouts):
            log(f"the sim PC latched a fault: restart the sim and resume with --resume {self.out}")
            return EXIT_SIM_FAULT
        return 0 if not self.problems else 1


def _step_summary(steps: list[dict]) -> dict:
    sent = [s for s in steps if s.get("sent")]
    skipped = [s for s in steps if not s.get("sent")]
    reasons: dict = {}
    for s in skipped:
        reasons[s.get("skip_reason") or "?"] = reasons.get(s.get("skip_reason") or "?", 0) + 1

    def q(key, p):
        v = [s[key] for s in sent if s.get(key) is not None]
        return float(np.percentile(v, p)) if v else None

    margins = [s["margin_ms"] for s in sent if s.get("margin_ms") is not None]
    return {"sent": len(sent), "skipped": len(skipped), "skip_reasons": reasons,
            "policy_ms_p50": q("policy_ms", 50), "policy_ms_p95": q("policy_ms", 95),
            "cycle_ms_p50": q("cycle_ms", 50), "image_age_ms_p50": q("image_age_ms", 50),
            "margin_ms_min": min(margins) if margins else None, "margin_ms_p50": q("margin_ms", 50),
            "rejected": int(sum(s.get("rejected", 0) for s in steps)),
            "shortened": int(sum(s.get("shortened", 0) for s in steps))}


def _port_busy(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _index_list(text: str) -> list[int]:
    out = sorted({int(x) for x in text.replace(" ", "").split(",") if x != ""})
    if not out or out[0] < 0:
        raise ValueError(f"--only-rollouts {text!r}: give indices >= 0, e.g. 3,7")
    return out


def _non_negative(text: str) -> int:
    k = int(text)
    if k < 0:
        raise ValueError(f"{text!r}: an integer >= 0")
    return k


def add_arguments(p) -> None:
    p.add_argument("--policy", required=True, help="RUN_FOLDER | hold_still | replay:<zarr>#<index>")
    p.add_argument("--rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--starts", default="random", metavar="random|set|set:FILE",
                   help="where the eraser starts: random (default: rng([seed, n]) in --random's region), set (rollout "
                        "n = start n of the start set eval/start_sets/w3.yaml, the same for every checkpoint: 0-9 the "
                        "shared training demos 0-9 at their grasp carried to the start pose, 10-17 the real-arm trials' "
                        "fitted eraser poses; the checkpoint's val demo is recorded held_out), set:<FILE or name>; "
                        "needs --home start")
    p.add_argument("--tag", default="eval")
    p.add_argument("--timeout", type=float, default=60.0, help="per rollout, s")
    p.add_argument("--tap", action="store_true", help="manager.save_tap (tap images every 10 steps)")
    p.add_argument("--dry-run", action="store_true", help="check what can be checked without the sim running")
    # 2026-11-21 (the user): every run with the arm's settings, bf16 / 8 DDIM steps / exec_h 12 (as policy_sim.yaml),
    # passed to `policy_deploy use` explicitly so eval.yaml's station_policy / station_manager show them
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--diffusion-steps", type=int, default=8)
    p.add_argument("--exec-h", type=int, default=12)
    p.add_argument("--behind-tip", type=float, default=None,
                   help="replay placement: eraser centre behind the tip, m (default: half its height)")
    p.add_argument("--look", choices=tuple(looks.LOOKS), default=looks.DEFAULT_LOOK,
                   help="the scene look (sim_deploy/looks.py): robot (default, the real robot cell), train (the "
                        "training scene), current (sim_rig.yaml); run.sh starts the sims with it and passes it here")
    p.add_argument("--model", default=None, help="model.json (default: the --look's build)")
    p.add_argument("--random", default=None, help="eraser region file (default: the --home variant's)")
    p.add_argument("--home", choices=HOMES + tuple(HOME_ALIASES), default="start",
                   help="start (default, P2): h then the PC's p, the eraser relative to stations/start_pose.yaml; "
                        "deploy: h only (run.sh starts the sim PC with q_start_rad null); trainstart = start")
    p.add_argument("--start-pose", default=None,
                   help="--home start: the start pose file (default stations/start_pose.yaml; P4 sweep). run.sh reads "
                        "the same option and merges that file into the sim PC's station")
    p.add_argument("--home-key", choices=("auto", "laptop", "pc"), default="auto",
                   help="how h is pressed: the laptop's key (C3 home) or the PC's (scene key); auto: the laptop's "
                        "if the copy's policy_deploy still has it")
    p.add_argument("--scene-endpoint", default="tcp://127.0.0.1:5651")
    p.add_argument("--c2-endpoint", default="tcp://127.0.0.1:5620")
    p.add_argument("--state-endpoint", default="tcp://127.0.0.1:5650")
    p.add_argument("--video-fps", type=float, default=10.0, help="0 = no videos")
    p.add_argument("--start-timeout", type=float, default=180.0, help="policy_deploy start to [idle], s")
    p.add_argument("--home-timeout", type=float, default=60.0)
    p.add_argument("--start-rollout-timeout", type=float, default=30.0)
    p.add_argument("--only-rollouts", type=_index_list, default=None, metavar="I,J,...",
                   help="run only these rollout indices (0-based) with --seed's keys [seed, n] (or set starts I, J): "
                        "the placements of a full run's rollouts I, J (batch retries); the folders keep the indices "
                        "(rollouts/007/)")
    p.add_argument("--resume", default=None, help="an eval folder to continue (after the sim PC latched a fault; "
                   "run.sh does this itself): its frozen station and seeds, the rollouts not yet run")
    p.add_argument("--max-new-rollouts", type=_non_negative, default=None, metavar="K",
                   help="run at most K of the rollouts not yet run (index order, within --only-rollouts), then end "
                        "(exit 0) with the eval left incomplete for --resume; an execution limit only, not recorded in "
                        "the protocol (batch.py piece_rollouts: GPU cool-downs between pieces). run.sh counts it over "
                        "its own sim restarts")


def main_run(args) -> int:
    ev = Eval(args)
    ev.prepare()
    log(f"eval folder: {ev.out}")                     # stable line: the UI runner parses it
    if args.dry_run:
        rows = ev.dry_run_checks()
        w = max(len(r[0]) for r in rows)
        for name, st, detail in rows:
            print(f"  {name:<{w}}  {st:<5} {detail}")
        bad = [r for r in rows if r[1] == "FAIL"]
        (ev.out / "dry_run.json").write_text(json.dumps(rows, indent=1))
        print("dry run:", "FAIL" if bad else "OK")
        return 1 if bad else 0
    return ev.run()
