"""Offset aligner — map generated ``(name, type)`` pairs back to ``[start, end)`` unit spans.

The DEER-style generator emits a *flat* list of ``{"name", "type"}`` with no offsets
(experiment-plan §4, output format B). Evaluation and the N1/N2 nested probes all need
character/token offsets, so this deterministic aligner recovers them.

Algorithm (experiment-plan §3.4):
  1. Find every occurrence of ``name`` in ``units`` (CMeEE: char subsequence; GENIA: token
     subsequence).
  2. Unique occurrence  -> assign it.
  3. Multiple occurrences (same-surface repeats, pre-exp A) -> greedy: pick the first
     candidate window whose exact ``[start, end)`` interval is not already taken by another
     prediction.  ``prefer_within`` lets an inner prediction be matched inside its owning
     outer span first (avoids mis-binding to an identical substring elsewhere in the
     sentence).  Exact-interval occupancy (not overlap) is used so nested spans — which
     always have *different* intervals — never block each other; only true duplicate
     surfaces compete.
  4. No occurrence (model rewrote / hallucinated the string) -> status ``"unaligned"``;
     callers count these as FP (experiment-plan §3.4 step 4).

The aligner is unit-agnostic: it works over ``units`` and splits ``name`` into a unit
sequence according to ``unit_level`` ("char" for CMeEE, "token" for GENIA).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

UnitLevel = str  # "char" | "token"


def unit_level_of(dataset: str) -> UnitLevel:
    """CMeEE is character-level, GENIA is token-level."""
    if dataset == "cmeee":
        return "char"
    if dataset == "genia":
        return "token"
    raise ValueError(f"unknown dataset: {dataset}")


def _name_to_units(name: str, unit_level: UnitLevel) -> List[str]:
    if unit_level == "char":
        return list(name)
    if unit_level == "token":
        return name.split()
    raise ValueError(f"unknown unit_level: {unit_level}")


def _find_occurrences(units: Sequence[str], target: Sequence[str]) -> List[Tuple[int, int]]:
    """Return all ``[start, end)`` windows where ``units[start:end] == target``."""
    n, m = len(units), len(target)
    if m == 0 or m > n:
        return []
    out: List[Tuple[int, int]] = []
    for i in range(0, n - m + 1):
        if list(units[i:i + m]) == list(target):
            out.append((i, i + m))
    return out


@dataclass
class AlignedPred:
    name: str
    type: str
    start: Optional[int]   # None if unaligned
    end: Optional[int]
    status: str            # "aligned" | "unaligned"
    n_candidates: int      # how many occurrences were found in the sentence

    def as_tuple(self) -> Optional[Tuple[int, int, str]]:
        if self.status != "aligned":
            return None
        return (self.start, self.end, self.type)


def align_predictions(
    units: Sequence[str],
    preds: Sequence[Tuple[str, str]],
    unit_level: UnitLevel,
    prefer_within: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[AlignedPred]:
    """Align a list of ``(name, type)`` predictions to unit spans.

    Args:
      units: the document's base units (chars for CMeEE, tokens for GENIA).
      preds: generated predictions as ``(name, type)`` pairs, in emission order.
      unit_level: "char" or "token".
      prefer_within: optional list of outer ``[start, end)`` spans; when a prediction has
        several candidate windows, prefer one falling inside any of these spans (used to
        bind inner predictions to their owning outer span).

    Greedy assignment processes predictions in order, marking exact intervals as occupied.
    """
    assigned: set[Tuple[int, int]] = set()
    results: List[AlignedPred] = []

    for name, etype in preds:
        target = _name_to_units(name, unit_level)
        occ = _find_occurrences(units, target)
        if not occ:
            results.append(AlignedPred(name, etype, None, None, "unaligned", 0))
            continue

        # Order candidates: optionally those inside a prefer_within span first, then by start.
        def sort_key(span: Tuple[int, int]) -> Tuple[int, int]:
            inside = 0
            if prefer_within:
                for (ws, we) in prefer_within:
                    if span[0] >= ws and span[1] <= we:
                        inside = -1  # sort earlier
                        break
            return (inside, span[0])

        ordered = sorted(occ, key=sort_key)
        chosen = next((sp for sp in ordered if sp not in assigned), None)
        if chosen is None:
            # all occurrences already claimed by earlier identical predictions
            results.append(AlignedPred(name, etype, None, None, "unaligned", len(occ)))
            continue
        assigned.add(chosen)
        results.append(AlignedPred(name, etype, chosen[0], chosen[1], "aligned", len(occ)))

    return results
