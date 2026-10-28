# deployment/ (placeholder, Wave 5)

Policy deployment on the Jetson (PIPELINE.md section 8, IMPLEMENTATION_PLAN.md Wave 5). Nothing is built here yet.

Planned flow, reusing `common/` unchanged so that deployment sees exactly the capture, timestamps and 224 images
that training data was made from:

```
CaptureService (cameras only) -> StreamHub -> shm rings -> ObsAssembler -> rigpy imageprep -> policy -> Franka
```

- **CaptureService / shm** (`rig/capture`, `rig/shm`): the same backends, clock and seqlock rings as `ec-record`.
  The IMU, live SLAM and finger tags are collection-only; deployment needs the cameras only (PIPELINE 2).
- **ObsAssembler**: picks 2 main frames at 20 Hz and 2 ultrawide frames at 10 Hz with the selection rule of
  `umi_utils.py:37-76`, and assembles the observation the policy was trained on.
- **Image prep** (`rig/imageprep`, `rigpy.imageprep`): main = letterbox -> 224 `INTER_AREA`; ultrawide = remap to the
  virtual iPhone ultrawide, then letterbox -> 224. The same code as the offline `images` stage.
- **Policy and robot**: pose comes from the Franka and width from the gripper (proprio = commanded action history);
  actions go to `controlSW/franka_deploy`.
