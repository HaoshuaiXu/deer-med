"""Offset aligner: map generated (name, type) back to [start, end) unit spans."""
from .aligner import AlignedPred, align_predictions, unit_level_of

__all__ = ["AlignedPred", "align_predictions", "unit_level_of"]
