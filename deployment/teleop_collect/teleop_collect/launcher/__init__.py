# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/__init__.py @ 2ffe641
# changes: teleop_collect; the command and its processes (fake_pc, frames, teleop, ui); the section number.
"""teleop_collect.launcher: `teleop_collect check | run [--mock] [--station FILE] [--data-root PATH] [--no-browser]`
for the laptop's processes (STRUCTURE.md section 10.1).

It imports only the standard library, yaml, teleop_collect.contracts and deploy_contracts, never a process
subpackage: the processes run as subprocesses (`python -m teleop_collect.<name>`), so OpenCV, PyAV or aiohttp never
load here. Importing this package imports nothing else.
"""
