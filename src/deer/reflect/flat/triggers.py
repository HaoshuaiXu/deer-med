"""Trigger detection for DEER's three reflection sub-steps (spec §4).

All triggers are statistic-driven (DeerStats), never logprob.  Each returns the token
indices (or spans) to reflect on; the actual edit is decided by the LLM in passes.py.

Notes on items the paper left underspecified (spec §7) — implemented as tunable knobs:
  - Unseen: "surrounded by tokens frequently labelled entity/context" -> we check whether any
    token within ``C`` of the unseen token has P(t_e) or P(t_c) >= ``surround_thresh``.
  - Boundary: "predicted as entity but more often a context token in training" -> trigger when
    P(t_c) > P(t_e) for a boundary token.  Final keep/trim is the LLM's call.
"""
from __future__ import annotations

from typing import List, Sequence, Set, Tuple

Span = Tuple[int, int, str]


def covered_indices(spans: Sequence[Span]) -> Set[int]:
    """Unit indices currently inside any predicted entity."""
    idx: Set[int] = set()
    for (s, e, _t) in spans:
        for i in range(s, e):
            idx.add(i)
    return idx


def detect_unseen_triggers(
    units: Sequence[str],
    spans: Sequence[Span],
    stats,
    C: int = 2,
    surround_thresh: float = 0.5,
) -> List[int]:
    """Unseen tokens not predicted as entities, surrounded by high entity/context tokens."""
    covered = covered_indices(spans)
    n = len(units)
    out: List[int] = []
    for i, tok in enumerate(units):
        if stats.is_seen(tok) or i in covered:
            continue
        lo, hi = max(0, i - C), min(n, i + C + 1)
        surrounded = any(
            j != i and stats.is_seen(units[j]) and
            (stats.p_entity(units[j]) >= surround_thresh or stats.p_context(units[j]) >= surround_thresh)
            for j in range(lo, hi)
        )
        if surrounded:
            out.append(i)
    return out


def detect_fn_triggers(
    units: Sequence[str],
    spans: Sequence[Span],
    stats,
    theta_fn: float = 0.95,
) -> List[int]:
    """Seen tokens with entity-likelihood P(t_e) > theta_fn that were NOT predicted."""
    covered = covered_indices(spans)
    out: List[int] = []
    for i, tok in enumerate(units):
        if i in covered:
            continue
        if stats.is_seen(tok) and stats.p_entity(tok) > theta_fn:
            out.append(i)
    return out


def detect_boundary_triggers(
    units: Sequence[str],
    spans: Sequence[Span],
    stats,
    K: int = 2,
) -> List[Tuple[int, int, str]]:
    """Examine the 2K boundary tokens of each predicted span: K inside + K outside per edge
    (DEER spec §4.4 — "2K tokens, K on the entity side and K on the context side").

    Returns list of (span_index, token_index, where) where where in {"inside","outside"}:
      - inside  token triggers a TRIM candidate when P(t_c) > P(t_e) (predicted as entity but
        statistically more of a context word, e.g. "city" in "Wenchang city").
      - outside-adjacent token triggers an EXPAND candidate when P(t_e) > P(t_c) (statistically
        an entity word currently left out, e.g. "breast" before a predicted "cancer"); skipped
        if it already belongs to another predicted entity ("Don't consider it if it belongs to
        adjacent entities").
    """
    n = len(units)
    covered = covered_indices(spans)
    out: List[Tuple[int, int, str]] = []
    for si, (s, e, _t) in enumerate(spans):
        # inside edges (trim candidates)
        inside = set(range(s, min(e, s + K))) | set(range(max(s, e - K), e))
        for i in inside:
            tok = units[i]
            if stats.is_seen(tok) and stats.p_context(tok) > stats.p_entity(tok):
                out.append((si, i, "inside"))
        # outside-adjacent edges (expand candidates)
        outside = set(range(max(0, s - K), s)) | set(range(e, min(n, e + K)))
        for i in outside:
            if i in covered:           # belongs to another predicted entity -> skip
                continue
            tok = units[i]
            if stats.is_seen(tok) and stats.p_entity(tok) > stats.p_context(tok):
                out.append((si, i, "outside"))
    return out
