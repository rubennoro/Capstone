"""Package episodes for the W1 offline harness: package index <-> Pi episode name <-> umiproc slot <-> raw recording,
the Pi-format staging folder the copy's ``PiRecording.load`` reads, the checkpoint's train/val split per episode, and
the run folders (checkpoints) trained on a package.

Light (numpy + yaml only): works in both envs. Read-only on ``~/data`` and ``~/Capstone``; the staging folders are
written under ``sim_evals/_pipeline/W1/staging/<folder>/<slot>/``.

    python -m sim_deploy.offline.episodes pick1-v1 14            # one episode's mapping (JSON)
    python -m sim_deploy.offline.episodes --runs                  # the run folders with a checkpoint

Staging (why not the slot's raw/ or ~/Downloads/data/incoming directly): the slot's ``raw/`` is dataproc.raw.v1
(csv ``frame_idx,t_ns,sensor_seq,exposure_ns``, cam0 = main), which PiRecording does not read; the Pi's own folders
in ``~/Downloads/data/incoming`` reuse episode numbers across Pi sessions (day 1 and day 2). So each slot gets a
Pi-format folder of symlinks: ``meta.json`` with a ``cameras`` list (model, file, size, fps, frames, bitrate from the
slot's ``pi_meta``), ``main.mp4`` / ``ultrawide.mp4`` -> the slot's videos (bytes checked against its
MANIFEST.sha256), and ``<stem>_frames.csv`` = ``frame_index,t_ns,exposure_ns`` copied from the slot's csv (the
recorded exposure, 5 ms in every PICK episode, which is also the copy's default_exposure_ns).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from sim_deploy.offline.common import CAPSTONE_DATA, DATASETS, STAGING, TASKS, write_json

ROLE_MODEL = {"main": "imx519", "ultrawide": "imx219"}
INCOMING = Path("~/Downloads/data/incoming").expanduser()


@dataclass(frozen=True)
class EpisodeRef:
    dataset: str                # package name, e.g. pick1-v1
    dataset_path: str           # .../<name>.zarr.zip
    index: int                  # 0-based in meta/episode_ends (= package.json episodes order)
    pi_name: str                # replay buffer episode name, e.g. pi-2026-11-18_episode_000152
    folder: str                 # umiproc task folder: PICK_TEST1 (day 1) | PICK_TEST2 (day 2)
    slot: str                   # umiproc slot, e.g. episode_15
    slot_dir: str               # ~/data/umi/tasks/<folder>/<slot>
    n_steps: int                # rows (60 Hz)
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_path(name: str) -> Path:
    """~/data/umi/datasets/<group>/<name>/<name>.zarr.zip (exactly one match)."""
    hits = sorted(DATASETS.glob(f"*/{name}/{name}.zarr.zip"))
    if len(hits) != 1:
        raise FileNotFoundError(f"dataset {name!r}: {len(hits)} matches under {DATASETS}")
    return hits[0]


@lru_cache(maxsize=None)
def package_episodes(name: str) -> tuple[dict, ...]:
    return tuple(json.loads((dataset_path(name).parent / "package.json").read_text())["episodes"])


@lru_cache(maxsize=None)
def _slot_by_pi_name(folder: str) -> dict[str, str]:
    out = {}
    for piece in sorted((TASKS / folder).glob("*/processed/piece.json")):
        out[json.loads(piece.read_text())["pi_name"]] = piece.parent.parent.name
    return out


def episode(name: str, index: int) -> EpisodeRef:
    eps = package_episodes(name)
    if not 0 <= index < len(eps):
        raise IndexError(f"{name}: episode {index} not in 0..{len(eps) - 1}")
    e = eps[index]
    slot = _slot_by_pi_name(e["folder"]).get(e["pi_name"])
    if slot is None:
        raise FileNotFoundError(f"{e['pi_name']}: no slot in {TASKS / e['folder']} with that piece.json pi_name")
    if slot != e["slot"]:
        raise ValueError(f"{name}#{index}: package.json slot {e['slot']} but {e['pi_name']} is in {slot}")
    return EpisodeRef(dataset=name, dataset_path=str(dataset_path(name)), index=index, pi_name=e["pi_name"],
                      folder=e["folder"], slot=slot, slot_dir=str(TASKS / e["folder"] / slot),
                      n_steps=int(e["n_steps"]), label=e.get("label", e["folder"]))


def find_index(name: str, pi_name: str) -> int:
    for i, e in enumerate(package_episodes(name)):
        if e["pi_name"] == pi_name:
            return i
    raise KeyError(f"{pi_name} is not in {name}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pi_staging(ref: EpisodeRef) -> tuple[Path, dict]:
    """The Pi-format folder for PiRecording.load (built once; see the module docstring) and its provenance."""
    raw = Path(ref.slot_dir) / "raw"
    meta = json.loads((raw / "meta.json").read_text())
    out = STAGING / ref.folder / ref.slot
    info_path = out / "staging.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if info.get("pi_name") == ref.pi_name:
            return out, info
    manifest = {ln.split()[1]: ln.split()[0] for ln in (raw / "MANIFEST.sha256").read_text().splitlines() if ln.strip()}
    by_role = {c["role"]: (cam, c) for cam, c in meta["cameras"].items()}
    pi_cams = {c["model"]: c for c in meta["pi_meta"]["cameras"]}
    out.mkdir(parents=True, exist_ok=True)
    cameras, provenance = [], {}
    for role in ("main", "ultrawide"):
        cam, c = by_role[role]
        model = ROLE_MODEL[role]
        if c["sensor"] != model:
            raise ValueError(f"{raw}: {role} is {c['sensor']}, expected {model}")
        src = raw / f"{cam}.mp4"
        sha = _sha256(src)
        if manifest.get(f"{cam}.mp4") != sha:
            raise ValueError(f"{src}: sha256 does not match MANIFEST.sha256")
        link = out / f"{role}.mp4"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        with open(raw / f"{cam}_frames.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(out / f"{role}_frames.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_index", "t_ns", "exposure_ns"])
            for i, r in enumerate(rows):
                assert int(r["frame_idx"]) == i, (src, i)
                w.writerow([i, int(r["t_ns"]), int(r["exposure_ns"])])
        pc = pi_cams[model]
        if int(pc["frames"]) != len(rows) or [int(x) for x in pc["size"]] != [int(x) for x in c["resolution"]]:
            raise ValueError(f"{raw}: pi_meta {model} frames/size do not match the slot's {cam}")
        cameras.append({"file": f"{role}.mp4", "model": model, "size": list(c["resolution"]), "fps": pc["fps"],
                        "frames": len(rows), "bitrate": int(pc["bitrate"])})
        exposures = sorted({int(r["exposure_ns"]) for r in rows})
        provenance[role] = {"source": str(src), "sha256": sha, "slot_file": f"{cam}.mp4", "frames": len(rows),
                            "exposure_ns_values": exposures[:5], "pi_incoming_match": _incoming_match(ref, model, sha)}
    (out / "meta.json").write_text(json.dumps({"episode": meta.get("episode_number"), "cameras": cameras,
                                               "note": "W1 staging of " + str(raw)}, indent=1))
    info = {"pi_name": ref.pi_name, "slot_raw": str(raw), "cameras": provenance}
    write_json(info_path, info)
    return out, info


def _incoming_match(ref: EpisodeRef, model: str, sha: str) -> str | None:
    """The ~/Downloads/data/incoming/episode_NNNNNN Pi file with the same bytes, if that folder holds this episode."""
    num = ref.pi_name.rsplit("_", 1)[-1]
    d = INCOMING / f"episode_{num}"
    try:
        cams = json.loads((d / "meta.json").read_text())["cameras"]
    except (OSError, ValueError, KeyError):
        return None
    for c in cams:
        if c.get("model") == model and (d / c["file"]).is_file():
            return str(d / c["file"]) if _sha256(d / c["file"]) == sha else f"differs: {d / c['file']}"
    return None


# ------------------------------------------------------------------------------------------- runs and splits

def run_folders() -> list[dict]:
    """Run folders under ~/Capstone/data/*/policy/ with a checkpoint and run_config/config.yaml (not smoke runs)."""
    import yaml
    out = []
    for run in sorted(CAPSTONE_DATA.glob("*/policy/run_*")):
        if run.name.endswith("_smoke") or not list((run / "checkpoints").glob("*.ckpt")):
            continue
        cfg_path = run / "run_config/config.yaml"
        if not cfg_path.is_file():
            continue
        meta = json.loads((run / "train_meta.json").read_text()) if (run / "train_meta.json").is_file() else {}
        cfg = yaml.safe_load(cfg_path.read_text())
        prompting = bool(cfg.get("task", {}).get("shape_meta", {}).get("use_prompting", False))
        out.append({"run_folder": str(run), "task": run.parent.parent.name, "run": run.name,
                    "dataset": meta.get("dataset"), "kind": "prompting" if prompting else "plain",
                    "checkpoint": str(sorted((run / "checkpoints").glob("*.ckpt"))[0])})
    return out


def split_of(training_split_info: dict | None, pi_name: str) -> str:
    """'train' | 'val' | 'new' (not in the checkpoint's split info) | 'unknown' (no split info)."""
    if not training_split_info:
        return "unknown"
    hits = [bool(v) for k, v in training_split_info.items() if str(k).split(":", 1)[0] == pi_name]
    if not hits:
        return "new"
    return "train" if all(hits) else ("val" if not any(hits) else "mixed")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--runs":
        print(json.dumps(run_folders(), indent=1))
        return 0
    name, index = argv[0], int(argv[1])
    ref = episode(name, index)
    staging, info = pi_staging(ref)
    print(json.dumps({**ref.to_dict(), "staging": str(staging), "staging_info": info}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
