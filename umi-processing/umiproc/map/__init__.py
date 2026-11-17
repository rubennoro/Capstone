"""``umiproc map``: the session map (UMI's 02_create_map + the mapping clip re-localized as in 03_batch_slam), the
table tag's pose in it (UMI's calibrate_slam_tag), and checks M1-M5 (docs/DESIGN.md "Checks -> map").

* :mod:`umiproc.map.build`: the command's steps, the pass/fail flow and where things go;
* :mod:`umiproc.map.checks`: M1-M5 and the diagnostics;
* :mod:`umiproc.map.tag`: the table-tag detection and UMI's aggregation;
* :mod:`umiproc.map.stored`: reading a folder's map back (status, "out of date");
* :mod:`umiproc.map.cli`: ``add_arguments`` / ``run``.
"""
