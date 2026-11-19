# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/__init__.py @ 2ffe641
# changes: the docstring (records episodes and serves R1/U2 instead of serving C1 images).
"""teleop_collect.frames: the process "frames" (STRUCTURE.md section 5).

Receives the Pi's stream (C0, pi_stream/v1), maps Pi times to the laptop clock, tracks each camera's health, keeps
the newest JPEG of each camera for the page (U2) and, while an episode records (R1), writes every received JPEG
unchanged into one MJPEG .mkv per camera plus the timing CSVs. Images are never decoded. Importing the package
imports nothing else.
"""
from __future__ import annotations

NAME = "frames"
