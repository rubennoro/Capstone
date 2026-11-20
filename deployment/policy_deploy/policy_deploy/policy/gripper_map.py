"""GripperMap: the handheld gripper's widths (what the model learned) <-> the robot's fingertip opening.

Training's gripper_width is the ArUco marker distance on the handheld fingers (pick1 datasets: about 0.051 m closed on
the eraser to 0.090 m open). The robot's printed fingers open 0 to 0.040 m. One straight line through two matching
points, used both ways:

    training_closed_m (closed on the object)  <->  robot_closed_m (the object's width at the fingertips)
    training_open_m   (fully open)            <->  robot_open_m   (the robot's max opening)

Anchoring "closed" on the object's width keeps a grasp in distribution: the robot holding the object reads about
robot_closed_m, which goes back to the model as training_closed_m. A commanded width below the object's is fine: the
PC closes with grasp() and holds wherever the object stops the fingers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

KEYS = ("training_closed_m", "training_open_m", "robot_closed_m", "robot_open_m")


@dataclass(frozen=True)
class GripperMap:
    training_closed_m: float
    training_open_m: float
    robot_closed_m: float
    robot_open_m: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GripperMap":
        return cls(**{k: float(cfg[k]) for k in KEYS})

    @staticmethod
    def problems(cfg: Any) -> list[str]:
        """What is wrong with a policy.gripper_map value (a map with the four widths), as text."""
        if not isinstance(cfg, Mapping):
            return [f"expected a map with {', '.join(KEYS)} or null, got {cfg!r}"]
        out = [f"unknown key {k!r}" for k in cfg if k not in KEYS]
        for k in KEYS:
            v = cfg.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                out.append(f"{k}: expected a number >= 0, got {v!r}")
        if out:
            return out
        if cfg["training_open_m"] <= cfg["training_closed_m"]:
            out.append("training_open_m must be greater than training_closed_m")
        if cfg["robot_open_m"] <= cfg["robot_closed_m"]:
            out.append("robot_open_m must be greater than robot_closed_m")
        return out

    @property
    def _scale(self) -> float:
        """Robot metres per training metre."""
        return (self.robot_open_m - self.robot_closed_m) / (self.training_open_m - self.training_closed_m)

    def to_model(self, robot_width: np.ndarray) -> np.ndarray:
        """Robot opening (C4 observation) -> the width the model was trained on. float32 in, float32 out.

        The robot width is first clamped to [robot_closed_m, robot_open_m], so the model always gets a width inside
        [training_closed_m, training_open_m], the range it was trained on: fingers closed past the object (nothing
        grasped, e.g. 0.012 m) read as training_closed_m instead of extrapolating to a negative width, and a width
        above robot_open_m (the warm-up's, a sensor reading) as training_open_m."""
        w = np.clip(np.asarray(robot_width, dtype=np.float64), self.robot_closed_m, self.robot_open_m)
        return (self.training_closed_m + (w - self.robot_closed_m) / self._scale).astype(np.float32)

    def to_robot(self, model_width: np.ndarray) -> np.ndarray:
        """The model's width -> robot opening (clipped afterwards by Policy.make_chunk). float32 in, float32 out."""
        w = np.asarray(model_width, dtype=np.float64)
        return (self.robot_closed_m + (w - self.training_closed_m) * self._scale).astype(np.float32)

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in KEYS}
