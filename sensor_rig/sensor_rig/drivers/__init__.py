"""Sensor backends (cameras, IMU). Shared by the recorder and by deployment (RigFrameSource).

Importing this package is cheap: hardware backends (GStreamer/Argus, spidev) are imported lazily by ``make_camera`` /
``make_imu``. Public names live in ``sensor_rig.drivers.base``.
"""

from sensor_rig.drivers.base import (  # noqa: F401
    CameraBackend,
    Frame,
    ImuBackend,
    ImuBatch,
    RealClock,
    SimClock,
    make_camera,
    make_imu,
)
