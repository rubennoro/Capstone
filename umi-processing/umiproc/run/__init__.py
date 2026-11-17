"""``umiproc run``: the demonstrations of a session, localized against its map (docs/DESIGN.md "Checks -> run").

* :mod:`umiproc.run.rules`: UMI's lost-frame rules at 30 fps (pure);
* :mod:`umiproc.run.pieces`: the table frame, the pose checks and finishing a piece;
* :mod:`umiproc.run.recording`: one recording (take, localize, wait, finish, episodes or failed);
* :mod:`umiproc.run.command`: the range (map and fingerprint checks, the plan, the report);
* :mod:`umiproc.run.cli`: ``add_arguments`` / ``run``.
"""
