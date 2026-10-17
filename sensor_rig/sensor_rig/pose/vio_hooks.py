"""VIO hooks for ``process-session``: where each demo's trajectory comes from.

* ``import-dir``: an existing trajectory per episode, looked up in ``--traj-dir`` as ``<NAME>.tum``,
  ``<NAME>/trajectory.tum``, ``f_<NAME>.txt`` (ORB-SLAM3 EuRoC output) or, for episodes recorded by
  ``collect --mock-replay``, ``**/<replay source NAME>/sim/vio_trajectory.tum`` (the simulator's trajectory).
* ``cmd``: any VIO run by a shell command template, after ``pose-prep --target orbslam3``. Placeholders:
  ``{prep}`` (the episode's EuRoC folder), ``{settings}``, ``{times}``, ``{name}``, ``{mask}``, ``{out}`` (where the
  command must write the trajectory: TUM or ORB-SLAM3 EuRoC ``f_<NAME>.txt`` format).
* ``orbslam3-docker``: ORB-SLAM3 mono-inertial in the image of ``sensor_rig/docker/orbslam3`` (see its README).

Each hook returns ``(trajectory_path, format, frame)`` for ``pose-import``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

DEFAULT_IMAGE = "sensor-rig/orbslam3:v1.0-release-focal"
ORB_BIN = "/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc"
ORB_VOC = "/ORB_SLAM3/Vocabulary/ORBvoc.txt"


class VioError(RuntimeError):
    pass


def find_trajectory(ep, traj_dir) -> tuple[Path, str, str]:
    d = Path(traj_dir)
    name = ep.name
    for p, fmt in ((d / f"{name}.tum", "tum"), (d / name / "trajectory.tum", "tum"),
                   (d / f"f_{name}.txt", "orbslam3-euroc"), (d / name / f"f_{name}.txt", "orbslam3-euroc")):
        if p.is_file():
            return p, fmt, "imu"
    sim = ep.meta.get("sim") or {}
    src = sim.get("replay_source") if isinstance(sim, dict) else None
    if src:
        hits = sorted(d.glob(f"**/{src}/sim/vio_trajectory.tum"))
        if hits:
            return hits[0], "tum", "imu"
    raise VioError(f"no trajectory for {name} in {d} (looked for {name}.tum, {name}/trajectory.tum, f_{name}.txt"
                   + (f", **/{src}/sim/vio_trajectory.tum" if src else "") + ")")


def run_cmd(template: str, prep_dir: Path, name: str, timeout_s: float = 3600) -> tuple[Path, str, str]:
    out = prep_dir / f"f_{name}.txt"
    cmd = template.format(prep=shlex.quote(str(prep_dir)), settings=shlex.quote(str(prep_dir / "orbslam3_settings.yaml")),
                          times=shlex.quote(str(prep_dir / "times.txt")), name=shlex.quote(name),
                          mask=shlex.quote(str(prep_dir / "mask_ultrawide.png")), out=shlex.quote(str(out)))
    with open(prep_dir / "vio_cmd.log", "w") as log:
        log.write(f"$ {cmd}\n")
        log.flush()
        p = subprocess.run(cmd, shell=True, cwd=str(prep_dir), stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout_s)
    if p.returncode != 0:
        raise VioError(f"VIO command failed (rc={p.returncode}), see {prep_dir / 'vio_cmd.log'}")
    if not out.is_file():
        raise VioError(f"VIO command wrote no {out}")
    fmt = "tum" if _looks_like_seconds(out) else "orbslam3-euroc"
    return out, fmt, "imu"


def _looks_like_seconds(path: Path) -> bool:
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            return float(line.replace(",", " ").split()[0]) < 1e10
    return False


def docker_image_mask_supported(image: str, docker: str = "docker") -> bool:
    """True if the image is labelled ``sensor_rig.mask_supported=true`` (upstream ORB-SLAM3: false)."""
    try:
        p = subprocess.run([docker, "image", "inspect", "--format",
                            '{{ index .Config.Labels "sensor_rig.mask_supported" }}', image],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0 and p.stdout.strip().lower() == "true"


def orbslam3_docker_command(image: str, prep_dir: Path, name: str, mask: Path | None = None,
                            docker: str = "docker", user: str | None = None) -> list[str]:
    """``docker run`` for one episode: the EuRoC folder is mounted at /data (also the working dir, where
    mono_inertial_euroc writes ``f_<NAME>.txt``)."""
    prep_dir = Path(prep_dir).resolve()
    if user is None and hasattr(os, "getuid"):
        user = f"{os.getuid()}:{os.getgid()}"
    cmd = [docker, "run", "--rm"]
    if user:
        cmd += ["--user", user]
    cmd += ["-v", f"{prep_dir}:/data", "-w", "/data"]
    if mask is not None:
        cmd += ["-v", f"{Path(mask).resolve()}:/data/mask.png:ro"]
    cmd += [image, ORB_BIN, ORB_VOC, "/data/orbslam3_settings.yaml", "/data", "/data/times.txt", name]
    if mask is not None:
        cmd += ["--mask", "/data/mask.png"]
    return cmd


def run_orbslam3_docker(prep_dir: Path, name: str, image: str = DEFAULT_IMAGE, docker: str = "docker",
                        timeout_s: float = 3600) -> tuple[Path, str, str, dict]:
    if shutil.which(docker) is None:
        raise VioError(f"'{docker}' not found on PATH")
    mask = prep_dir / "mask_ultrawide.png"
    use_mask = mask.is_file() and docker_image_mask_supported(image, docker)
    cmd = orbslam3_docker_command(image, prep_dir, name, mask if use_mask else None, docker)
    with open(prep_dir / "orbslam3_docker.log", "w") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    out = prep_dir / f"f_{name}.txt"
    if p.returncode != 0:
        raise VioError(f"ORB-SLAM3 container failed (rc={p.returncode}), see {prep_dir / 'orbslam3_docker.log'}")
    if not out.is_file() or out.stat().st_size == 0:
        raise VioError(f"ORB-SLAM3 wrote no trajectory {out} (tracking never initialised?)")
    return out, "orbslam3-euroc", "imu", {"image": image, "mask_used": use_mask, "command": cmd}
