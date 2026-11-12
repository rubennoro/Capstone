"""Private: rot6d row checks (unit length, perpendicular), for the C4 validators. Computed in float64."""
from __future__ import annotations

import numpy as np


def rot6d_problem(rot6d: np.ndarray, norm_tol: float, dot_max: float) -> str | None:
    """None if every row's two 3-vectors have length 1 +- norm_tol and |cos angle| <= dot_max.

    Otherwise the first problem, as text."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    for i, row in enumerate(rows):
        a, b = row[:3], row[3:]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if abs(na - 1.0) > norm_tol:
            return f"row {i}: first vector has length {na:.6g}, expected 1 +- {norm_tol}"
        if abs(nb - 1.0) > norm_tol:
            return f"row {i}: second vector has length {nb:.6g}, expected 1 +- {norm_tol}"
        cos = abs(float(np.dot(a, b))) / (na * nb)
        if cos > dot_max:
            return f"row {i}: vectors not perpendicular, |cos angle| = {cos:.6g} > {dot_max}"
    return None
