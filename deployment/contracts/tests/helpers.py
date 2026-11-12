"""Small helpers shared by the contracts tests."""
from __future__ import annotations

from typing import Any, Callable

import pytest

from contracts import ContractError


def contract_error(fn: Callable[[], Any], field: str, problem: str) -> ContractError:
    """Run fn, expect a ContractError with exactly this field and problem, and return it."""
    with pytest.raises(ContractError) as info:
        fn()
    e = info.value
    assert (e.field, e.problem) == (field, problem), str(e)
    return e
