"""The fixed C1/C4 sizes (STRUCTURE.md section 4.6). Fixed by this contract version: changing one is a version bump."""
from __future__ import annotations

OBS_STEPS = 2        # observation steps per policy step (C1 frames, C4 Observation)
ACTION_STEPS = 16    # steps in one action chunk (C4 ActionChunk)
STEP_MS = 50         # time between action steps, ms
IMAGE_SIZE = 224     # images are IMAGE_SIZE x IMAGE_SIZE RGB
