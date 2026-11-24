"""Imports from the copied laptop code (``deployment/policy_deploy`` of this repo), which is not installed in
``~/envs/sim_deploy``: the rig-file loader, the camera models and the ``pi_stream/v1`` contract. Putting that folder
on ``sys.path`` makes the sim Pi use exactly the code the frames process uses (no second copy of any constant or
formula). Only modules with light dependencies (numpy, scipy, cv2, yaml, zmq-free) are imported from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]                      # ~/Capstone-main-sim
POLICY_DEPLOY = REPO / "deployment" / "policy_deploy"
DEFAULT_RIG_FILE = REPO / "deployment" / "contracts" / "rigs" / "rig_pi_v3_640.yaml"

if str(POLICY_DEPLOY) not in sys.path:
    sys.path.insert(0, str(POLICY_DEPLOY))

from policy_deploy.contracts.pi_stream import PiFrameStamp, PiStreamHeader, PiStreamPacket  # noqa: E402
from policy_deploy.frames._dataproc.config import RigConfig  # noqa: E402
from policy_deploy.frames._dataproc.geometry.camera_model import (FisheyeEquidistant, PinholeRadtan,  # noqa: E402
                                                                  VirtualPinhole)
from policy_deploy.frames._dataproc.imageprep import ImagePrep, letterbox_point, physical_model  # noqa: E402
from policy_deploy.frames.rig_file import load_live_rig  # noqa: E402

__all__ = ["REPO", "POLICY_DEPLOY", "DEFAULT_RIG_FILE", "PiFrameStamp", "PiStreamHeader", "PiStreamPacket",
           "RigConfig", "FisheyeEquidistant", "PinholeRadtan", "VirtualPinhole", "ImagePrep", "letterbox_point",
           "physical_model", "load_live_rig"]
