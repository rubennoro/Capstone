"""dataproc: Jetson-side offline processing of the Pi rig's recordings into training data.

``dataproc calibrate`` fits a session's gripper calibration and creates its task folder; ``dataproc run`` processes
a range of recordings into ``episode_N`` (or ``failed/``); ``dataproc package`` combines finished episodes into a
training ``.zarr.zip``. See ``docs/DESIGN.md`` (commands, layout), ``docs/PI_ADAPTER.md`` (the Pi's format) and
``docs/CONTRACT.md`` (dataproc's own raw format).
"""

__version__ = "0.1.0"
