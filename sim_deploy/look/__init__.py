"""Camera study B2 ("look"): opt-in realistic look for the sim (default off).

* ``camera_look.py``: the per-camera post-process (black level, gains, tone, saturation, colour shading, noise) the
  sim Pi applies to its 640x480 frames before the JPEG encode when a station's ``sim_pi.camera_look`` names a
  parameter file (``camera_look_real.yaml``).
* ``make_assets.py``: the real-data assets (table texture, workspace mesh) in ``assets/``.
* ``fit_look.py``: renders, region statistics, the camera-look fit and the comparison report.
"""
