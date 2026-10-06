# block_raise

Pick up an eraser (the "block") with a single iPhUMI gripper and raise it off the table.

- **Data:** recorded with one gripper held in the left hand (`L` in the app). With `sides=single`, the dataset stores it as `right`, which the single-arm configs read.
- **Datasets** (on the cluster): `~/CapstoneProject/datasets/block_raise/<dataset>/<dataset>.zarr.zip`
  - `firstreal-single`: session `firstreal` (2026-10-01), 45 demos, 37,971 frames. The demos were recorded without task labels, so each one is labeled as a single `block_raise` task covering the whole demo.

Each `run_NNN_<date>[_smoke]/` folder is one training job, with the same analysis layout every time: a README listing the dataset and settings, plus plots, videos, logs and config. Only the best checkpoint is kept (gitignored; it stays on the cluster).
Each run's `summary.json` holds its key numbers for comparing runs.
