"""sim_deploy.ui: a local web page to pick a policy and run sim rollouts on it with a Play button (ui/DESIGN.md).

`python -m sim_deploy.ui [--port 8770] [--stub] [--config FILE]` (env ~/envs/sim_deploy), then open
http://127.0.0.1:8770. Parts: server (U1), runner (U2), live (U3), static page (U4), catalog (U5).
"""
from __future__ import annotations

NAME = "sim_deploy.ui"
