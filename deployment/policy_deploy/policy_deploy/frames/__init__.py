"""policy_deploy.frames: the process "frames" (STRUCTURE.md section 4).

Receives the Pi's stream (C0, pi_stream/v1), maps Pi times to the laptop clock, keeps the last second, tracks each
camera's health and serves the policy's 224x224 images (C1). Importing the package imports nothing else.
"""
from __future__ import annotations

NAME = "frames"
