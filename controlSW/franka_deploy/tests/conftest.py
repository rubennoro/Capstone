import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Opt-in heavy tests (real checkpoint + prompt zarr + GPU):
#   FRANKA_DEPLOY_CKPT=<.ckpt>  FRANKA_DEPLOY_PROMPT_ZARR=<.zarr.zip>  FRANKA_DEPLOY_WORK=<scratch dir>
CKPT = os.environ.get("FRANKA_DEPLOY_CKPT")
PROMPT_ZARR = os.environ.get("FRANKA_DEPLOY_PROMPT_ZARR")
WORK = os.environ.get("FRANKA_DEPLOY_WORK")
