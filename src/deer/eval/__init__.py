"""Evaluation: strict span-level micro P/R/F1 + nested decomposition."""
from .metrics import (
    PRF,
    evaluate,
    nesting_roles,
    strict_prf,
)

__all__ = ["PRF", "evaluate", "nesting_roles", "strict_prf"]
