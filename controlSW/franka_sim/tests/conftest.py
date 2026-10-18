import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]                 # controlSW/franka_sim
DEPLOY_ROOT = SIM_ROOT.parent / "franka_deploy"                # controlSW/franka_deploy
for p in (SIM_ROOT, DEPLOY_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
