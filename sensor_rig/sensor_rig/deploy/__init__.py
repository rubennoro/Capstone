"""Deployment side of the rig: turn the physical cameras into iPhone-equivalent observation packets.

``RigFrameSource`` (``sensor_rig.deploy.frame_source``) holds all the logic and has no real-env imports; the real-env
peripheral ``real_env/peripherals/rig_camera.py: RigCamera`` is a thin adapter around it. See
``sensor_rig/docs/deployment.md``.
"""

from sensor_rig.deploy.frame_source import Packet, PairingConfig, RigFrameSource  # noqa: F401
