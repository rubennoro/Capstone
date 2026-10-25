"""B10: every `python -m franka_deploy ...` / `python -m franka_sim ...` command in hardware_day.md and the franka_deploy
README must still exist: its subcommand path plus --help exits 0. (sensor-rig lines are covered by sensor_rig's tests;
`python -c` / heredoc python lines are not commands of ours.)"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
SIM = PKG.parent / "franka_sim"
DOCS = [PKG / "docs" / "hardware_day.md", PKG / "README.md"]
CMD = re.compile(r"python -m ((?:franka_deploy|franka_sim)[\w.]*)([^`\n]*)")
WORD = re.compile(r"^[a-z][a-z0-9-]*$")


def commands():
    seen, out = set(), []
    for doc in DOCS:
        for m in CMD.finditer(doc.read_text()):
            mod, rest = m.group(1), m.group(2)
            if "sensor-rig" in m.group(0):
                continue
            subs = []
            for tok in rest.split():
                if WORD.match(tok):
                    subs.append(tok)
                else:
                    break
            key = (mod, tuple(subs))
            if key not in seen:
                seen.add(key)
                out.append((doc.name, mod, subs))
    return out


COMMANDS = commands()


def test_the_docs_contain_commands():
    mods = {(m, tuple(s)) for _, m, s in COMMANDS}
    for want in (("franka_deploy", ("fdcal", "capture", "stream")), ("franka_deploy", ("fdcal", "stream-intrinsics")),
                 ("franka_deploy", ("fdcal", "width")), ("franka_deploy", ("wireless-latency", "measure")),
                 ("franka_sim", ("eval",)), ("franka_sim", ("characterize",)),
                 ("franka_deploy.gripper.hand_server", ()), ("franka_deploy", ("check-config",))):
        assert want in mods, f"documented command missing: {want}"


@pytest.mark.parametrize("doc,mod,subs", COMMANDS, ids=[f"{m} {' '.join(s)}" for _, m, s in COMMANDS])
def test_documented_command_runs_help(doc, mod, subs):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"PYTHONPATH": os.pathsep.join([str(PKG), str(SIM)]), "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    p = subprocess.run([sys.executable, "-m", mod, *subs, "--help"], capture_output=True, text=True, env=env,
                       cwd=str(PKG), timeout=120)
    assert p.returncode == 0, f"{doc}: python -m {mod} {' '.join(subs)} --help -> {p.returncode}\n{p.stderr[-800:]}"


def test_section_headings_are_unique():
    """A bad edit once duplicated §0-§10 of hardware_day.md: every heading must appear exactly once."""
    for doc in DOCS:
        heads = [l.strip() for l in doc.read_text().splitlines() if re.match(r"^#{2,3} ", l)]
        dup = sorted({h for h in heads if heads.count(h) > 1})
        assert not dup, f"{doc.name}: duplicated headings {dup}"
    text = DOCS[0].read_text().split("\n## 15.")[0]          # §15's table quotes step numbers in its first column
    steps = re.findall(r"^\| (\d+\.\d+[a-z]?) \|", text, re.M)
    assert len(steps) == len(set(steps)), f"duplicated step numbers: {sorted({x for x in steps if steps.count(x) > 1})}"
