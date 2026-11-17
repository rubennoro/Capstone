# umi-processing (`umiproc`)

UMI's SLAM recipe for the Pi 5 rig (IMX219 ultrawide + IMX519 main + LSM6DSOX IMU): one map per session, then every
demo localized against it, with checks, into per-task folders of episodes ready to package. Runs on the Jetson:
reads `/data/incoming`, writes `~/data/umi/`.

- What problem this solves and why it is built this way (start here if you're new): [PROBLEM.md](PROBLEM.md)
- How to use it: [DIRECTIONS.md](DIRECTIONS.md)
- How it is built and why: [docs/DESIGN.md](docs/DESIGN.md)

Self-contained: `umiproc/dp/` is a copy of `data-processing/dataproc` (nothing here imports `data-processing/`).

```bash
pip install -e umi-processing[test]
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest umi-processing/tests -q
```
