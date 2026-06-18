"""Strict span-level micro P/R/F1 with nested-aware decomposition (experiment-plan §3).

O4 decision (2026-06-14): self-written strict-F1 is the primary metric (decomposition
E1/E2/E3 and nested/flat splits are not provided by official scripts and must be hand-
rolled); a one-off run of the official script (CBLUE / standard nested-NER) calibrates
that this implementation matches their counting on the same predictions.

An entity is the triple ``(start, end, type)`` over UNIT space (chars for CMeEE, tokens
for GENIA).  "Strict" = a prediction is correct iff boundary AND type both match exactly.
Micro = pool TP/FP/FN across the whole corpus.

Decomposition (experiment-plan §3.2):
  - nested vs flat **gold** recall: a gold entity is "nested" if it is the inner or outer
    member of some containment pair in its own document; gain should concentrate on the
    nested subset.
  - E1 inner-miss rate / E2 outer-miss rate / E3 nested-pair both-recovered rate, mapping
    to the N1/N2/N3 reflection passes.
  - depth-stratified recall (mainly meaningful on GENIA; >=3 layers is descriptive only).
  - per-type P/R/F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Span = Tuple[int, int, str]  # (start, end, type)


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, other: "PRF") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


# --------------------------------------------------------------------------- #
# Containment / nesting structure                                             #
# --------------------------------------------------------------------------- #
def _strictly_contains(outer: Span, inner: Span) -> bool:
    """True if ``outer`` strictly span-contains ``inner`` (different intervals)."""
    (os_, oe, _), (is_, ie, _) = outer, inner
    if (os_, oe) == (is_, ie):
        return False
    return os_ <= is_ and ie <= oe


def nesting_roles(golds: Sequence[Span]) -> Dict[str, object]:
    """Classify gold entities of one document by their nesting role.

    Returns a dict with sets/maps:
      - inner: set of spans that are contained in some other span
      - outer: set of spans that contain some other span
      - nested: inner | outer
      - flat: spans in neither role
      - depth: {span: 1 + (#spans strictly containing it)}
      - pairs: list of (outer, inner) strict-containment pairs
    """
    spans = list(golds)
    inner: Set[Span] = set()
    outer: Set[Span] = set()
    depth: Dict[Span, int] = {s: 1 for s in spans}
    pairs: List[Tuple[Span, Span]] = []
    for a in spans:
        n_containers = 0
        for b in spans:
            if a is b:
                continue
            if _strictly_contains(b, a):  # b contains a
                inner.add(a)
                outer.add(b)
                n_containers += 1
                pairs.append((b, a))
        depth[a] = 1 + n_containers
    nested = inner | outer
    flat = set(spans) - nested
    return {
        "inner": inner, "outer": outer, "nested": nested, "flat": flat,
        "depth": depth, "pairs": pairs,
    }


# --------------------------------------------------------------------------- #
# Core strict micro PRF                                                        #
# --------------------------------------------------------------------------- #
def strict_prf(
    gold_docs: Sequence[Sequence[Span]],
    pred_docs: Sequence[Sequence[Span]],
) -> PRF:
    """Corpus-level strict micro PRF over aligned doc lists."""
    assert len(gold_docs) == len(pred_docs), "doc count mismatch"
    total = PRF()
    for golds, preds in zip(gold_docs, pred_docs):
        gset = set(golds)
        pset = set(preds)
        tp = len(gset & pset)
        total.tp += tp
        total.fp += len(pset - gset)
        total.fn += len(gset - pset)
    return total


# --------------------------------------------------------------------------- #
# Full evaluation with decomposition                                          #
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    overall: PRF = field(default_factory=PRF)
    per_type: Dict[str, PRF] = field(default_factory=dict)
    # gold-subset recall counts
    nested_gold_recovered: int = 0
    nested_gold_total: int = 0
    flat_gold_recovered: int = 0
    flat_gold_total: int = 0
    # E1/E2/E3
    inner_recovered: int = 0
    inner_total: int = 0
    outer_recovered: int = 0
    outer_total: int = 0
    pair_both_recovered: int = 0
    pair_total: int = 0
    # depth-stratified recall: depth -> [recovered, total]
    depth_recall: Dict[int, List[int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        def rate(num: int, den: int) -> float:
            return round(num / den, 4) if den else 0.0
        depth = {
            str(k): {"recovered": v[0], "total": v[1], "recall": rate(v[0], v[1])}
            for k, v in sorted(self.depth_recall.items())
        }
        return {
            "overall": self.overall.as_dict(),
            "per_type": {t: prf.as_dict() for t, prf in sorted(self.per_type.items())},
            "nested_gold": {
                "recovered": self.nested_gold_recovered,
                "total": self.nested_gold_total,
                "recall": rate(self.nested_gold_recovered, self.nested_gold_total),
            },
            "flat_gold": {
                "recovered": self.flat_gold_recovered,
                "total": self.flat_gold_total,
                "recall": rate(self.flat_gold_recovered, self.flat_gold_total),
            },
            "E1_inner_miss_rate": rate(self.inner_total - self.inner_recovered, self.inner_total),
            "E2_outer_miss_rate": rate(self.outer_total - self.outer_recovered, self.outer_total),
            "E3_pair_both_recovered_rate": rate(self.pair_both_recovered, self.pair_total),
            "depth_recall": depth,
        }


def evaluate(
    gold_docs: Sequence[Sequence[Span]],
    pred_docs: Sequence[Sequence[Span]],
) -> EvalResult:
    """Strict micro PRF + per-type + nested/flat + E1/E2/E3 + depth."""
    assert len(gold_docs) == len(pred_docs), "doc count mismatch"
    res = EvalResult()

    for golds, preds in zip(gold_docs, pred_docs):
        gset: Set[Span] = set(golds)
        pset: Set[Span] = set(preds)

        # overall
        res.overall.tp += len(gset & pset)
        res.overall.fp += len(pset - gset)
        res.overall.fn += len(gset - pset)

        # per-type
        types = {t for (_, _, t) in gset | pset}
        for t in types:
            gt = {s for s in gset if s[2] == t}
            pt = {s for s in pset if s[2] == t}
            prf = res.per_type.setdefault(t, PRF())
            prf.tp += len(gt & pt)
            prf.fp += len(pt - gt)
            prf.fn += len(gt - pt)

        # nesting structure of THIS doc's gold
        roles = nesting_roles(golds)
        inner: Set[Span] = roles["inner"]      # type: ignore[assignment]
        outer: Set[Span] = roles["outer"]      # type: ignore[assignment]
        nested: Set[Span] = roles["nested"]    # type: ignore[assignment]
        flat: Set[Span] = roles["flat"]        # type: ignore[assignment]
        depth: Dict[Span, int] = roles["depth"]  # type: ignore[assignment]
        pairs: List[Tuple[Span, Span]] = roles["pairs"]  # type: ignore[assignment]

        for s in nested:
            res.nested_gold_total += 1
            if s in pset:
                res.nested_gold_recovered += 1
        for s in flat:
            res.flat_gold_total += 1
            if s in pset:
                res.flat_gold_recovered += 1

        for s in inner:
            res.inner_total += 1
            if s in pset:
                res.inner_recovered += 1
        for s in outer:
            res.outer_total += 1
            if s in pset:
                res.outer_recovered += 1
        for (o, i) in pairs:
            res.pair_total += 1
            if o in pset and i in pset:
                res.pair_both_recovered += 1

        for s, d in depth.items():
            slot = res.depth_recall.setdefault(d, [0, 0])
            slot[1] += 1
            if s in pset:
                slot[0] += 1

    return res
