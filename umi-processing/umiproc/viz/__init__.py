"""Analysis: looking at what the processing did. Nothing here changes a task folder; outputs go to its ``analysis/``.

* ``target``: what a command points at (a task folder, a map folder or a run folder) and where it writes.
* ``slam_run``: ``umiproc plot-run``, one figure of a SLAM run (tracking timeline, path, numbers).
* ``slam_video``: ``umiproc slam-video``, a video of a run (each frame with its tracking state).
* ``slam_3d`` (+ ``slam_3d_images``: lens projection, frame decoding): ``umiproc slam-3d``, the sparse map, keyframes
  and path as a 3-D page.
* ``dense_3d``: EXPERIMENTAL trial, not a command yet: a dense coloured point cloud from the mapping clip's frames.
"""
