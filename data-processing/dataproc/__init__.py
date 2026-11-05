"""dataproc: Jetson-side intake and offline processing of rig episodes.

``dataproc intake`` moves verified raw episodes from ``incoming/`` into a task folder; ``dataproc process`` turns
them into ``output/<folder>.zarr.zip`` for training. See ``docs/DESIGN.md`` (layout, commands, stages) and
``docs/CONTRACT.md`` (the raw episode format the Pi writes).
"""

__version__ = "0.1.0"
