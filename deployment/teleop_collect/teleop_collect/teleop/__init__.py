"""teleop_collect.teleop: the 20 ms teleop loop; C2 in, C3 out (the only C3 client); the laptop -> PC clock map; the
episode flow; the U1 server the page talks to (STRUCTURE.md 6). Also pc_probe, the read-only link probe.

motion/ (keys -> twist -> waypoints) and episode/ (the task store and the recorders) are its pure-logic parts."""

NAME = "teleop"
