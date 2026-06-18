"""DEER flat Error Reflection: Unseen -> False-Negative -> Boundary (single pass)."""
from .triggers import (
    covered_indices,
    detect_unseen_triggers,
    detect_fn_triggers,
    detect_boundary_triggers,
)
from .passes import reflect_flat

__all__ = [
    "covered_indices",
    "detect_unseen_triggers",
    "detect_fn_triggers",
    "detect_boundary_triggers",
    "reflect_flat",
]
