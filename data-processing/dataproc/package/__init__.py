"""``dataproc package``: combine the exported episodes of task folders into one training ``.zarr.zip``.

* :mod:`.collect`: which episodes go in, and the checks (same rig, >= 1 episode, >= 2 per label warning);
* :mod:`.build`: ``datasets/<label>/<DATASET>/<DATASET>.zarr.zip`` + ``package.json``;
* :mod:`.cli`: printing, the cluster hand-off lines, exit codes.
"""

from dataproc.package.build import PackageResult, package_dataset
from dataproc.package.collect import PackageError

__all__ = ["package_dataset", "PackageResult", "PackageError"]
