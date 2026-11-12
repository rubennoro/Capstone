"""process_frames: receives the Pi's camera frames (C0) and serves the policy's 224x224 images (C1)."""

from __future__ import annotations

from pathlib import Path

#: The process_frames/ folder (config.yaml, rigs/), found through the editable install.
FOLDER = Path(__file__).resolve().parents[1]
NAME = "process_frames"
