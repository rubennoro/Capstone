# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/fake_pc/__init__.py @ 2ffe641
# changes: imports; the module and command names in the docstring and the bind hint say teleop_collect.
"""teleop_collect.fake_pc: the Python stand-in for franka_control under --mock (STRUCTURE.md section 10).

It binds C2 (PUB) and C3 (REP) on 127.0.0.1 with the link.yaml ports and answers as TOP 3.7.2 says, on a drifting,
offset clock, so every laptop test exercises the clock map. It stands in for the messages and modes, not the physics.
"""
