"""build_policy(): kind name -> Policy object (STRUCTURE.md 4.10).

Only the chosen policy's module is imported, so `hold_still` never imports torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Mapping, NamedTuple

from policy.errors import PolicyLoadError

if TYPE_CHECKING:
    from policy.base_policy import Policy
    from policy.settings import Settings


class KindEntry(NamedTuple):
    module: str
    class_name: str
    needs_torch: bool


KINDS: Mapping[str, KindEntry] = {
    "hold_still": KindEntry("policy.hold_still_policy", "HoldStillPolicy", False),
    "diffusion": KindEntry("policy.diffusion_policy", "DiffusionPolicy", True),
}


def build_policy(settings: "Settings") -> "Policy":
    """Build the policy of `settings.active_kind`. Raises PolicyLoadError."""
    kind = settings.active_kind
    entry = KINDS.get(kind)
    if entry is None:
        raise PolicyLoadError(f"unknown policy kind {kind!r}", f"choose one of {', '.join(sorted(KINDS))}")
    try:
        module = importlib.import_module(entry.module)
    except ImportError as e:
        if not entry.needs_torch:
            raise
        missing = e.name or "torch"
        raise PolicyLoadError(f"{missing} is not importable (kind {kind})",
                              "run policy with python.policy, the torch environment") from e
    cls = getattr(module, entry.class_name)
    if getattr(cls, "KIND", None) != kind:
        raise PolicyLoadError(f"{entry.module}.{entry.class_name}.KIND is {getattr(cls, 'KIND', None)!r}, "
                              f"expected {kind!r}", "fix policy_factory.KINDS")
    return cls.from_settings(settings)
