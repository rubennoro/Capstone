"""sim_deploy: the MuJoCo stand-in for the Pi and the PC (PLAN.md, DESIGN.md). Subpackages: rig (E2), adapter (E3),
model + scene + eval (E6), sim_pi (E5); sim_pc (E4) is C++."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ~/Capstone-main-sim/sim_deploy
REPO = ROOT.parent                              # ~/Capstone-main-sim
