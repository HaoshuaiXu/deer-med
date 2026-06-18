"""DEER Step-0 token statistics — P(t_e) / P(t_c) / P(t_o) + span libraries.

Faithful to Bai et al. 2025 §2.2 (see notes/2026-06-14-deer-repro-spec.md):
  - Every token *occurrence* is assigned to exactly one class, priority entity > context > other.
  - entity  : a token inside any annotated entity span.
  - context : a token within C units of an entity boundary (default C=2), excluding entity
              tokens AND stopping at adjacent-entity boundaries (footnote 2: the context
              window of an entity is truncated when it meets another entity).
  - other   : everything else.
  - Counts are pooled per token *type* (surface form), then normalised:
        P(t_e)=count_e/N, P(t_c)=count_c/N, P(t_o)=count_o/N, N=count_e+count_c+count_o.
  - Training-free, pure counting; never uses LLM logprob.

Validation target (NCBI-Disease train, Disease single type, Figure 2 of the paper):
  cancer -> 268/0/5 (98.2%), of -> 178/937/4261 (3.3%), ataxia -> 57/0/1 (98.3%).

Span libraries (for the reflection passes' span-level retrieval): for each token type we
keep a capped list of focused span strings (the token plus a +-C window), one bucket per
class.  These mirror the paper's Entity/Context/Other span stores.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ENTITY, CONTEXT, OTHER = "entity", "context", "other"


def classify_units(n: int, entity_spans: Sequence[Tuple[int, int]], C: int = 2) -> List[str]:
    """Return per-index class labels for one document.

    Args:
      n: number of units in the document.
      entity_spans: list of (start, end) half-open spans (type-agnostic).
      C: context window on each side (default 2).
    """
    cls = [OTHER] * n
    ent_idx = set()
    for (s, e) in entity_spans:
        for i in range(max(0, s), min(n, e)):
            ent_idx.add(i)
    for i in ent_idx:
        cls[i] = ENTITY

    for (s, e) in entity_spans:
        # left window: walk outward, stop at an entity token (adjacent entity) or bounds
        cnt, i = 0, s - 1
        while i >= 0 and cnt < C:
            if i in ent_idx:
                break
            if cls[i] == OTHER:
                cls[i] = CONTEXT
            cnt += 1
            i -= 1
        # right window
        cnt, i = 0, e
        while i < n and cnt < C:
            if i in ent_idx:
                break
            if cls[i] == OTHER:
                cls[i] = CONTEXT
            cnt += 1
            i += 1
    return cls


@dataclass
class DeerStats:
    C: int
    counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {ENTITY: 0, CONTEXT: 0, OTHER: 0}))
    spans: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: defaultdict(lambda: {ENTITY: [], CONTEXT: [], OTHER: []}))
    span_cap: int = 50

    # -- probabilities ---------------------------------------------------- #
    def n(self, token: str) -> int:
        c = self.counts.get(token)
        if not c:
            return 0
        return c[ENTITY] + c[CONTEXT] + c[OTHER]

    def p(self, token: str, cls: str) -> float:
        total = self.n(token)
        if total == 0:
            return 0.0
        return self.counts[token][cls] / total

    def p_entity(self, token: str) -> float:
        return self.p(token, ENTITY)

    def p_context(self, token: str) -> float:
        return self.p(token, CONTEXT)

    def p_other(self, token: str) -> float:
        return self.p(token, OTHER)

    def is_seen(self, token: str) -> bool:
        return self.n(token) > 0

    # -- weight for retrieval (spec §3.1, eq.3) --------------------------- #
    def token_weight(self, token: str, w_e: float, w_c: float, w_o: float) -> float:
        if not self.is_seen(token):
            return 1.0  # unseen tokens get max weight
        return w_e * self.p_entity(token) + w_c * self.p_context(token) + w_o * self.p_other(token)

    # -- persistence ------------------------------------------------------ #
    def to_json(self) -> dict:
        return {
            "C": self.C,
            "span_cap": self.span_cap,
            "counts": {t: dict(c) for t, c in self.counts.items()},
            "spans": {t: {k: list(v) for k, v in d.items()} for t, d in self.spans.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DeerStats":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        st = cls(C=obj["C"], span_cap=obj.get("span_cap", 50))
        for t, c in obj["counts"].items():
            st.counts[t] = {ENTITY: c.get(ENTITY, 0), CONTEXT: c.get(CONTEXT, 0), OTHER: c.get(OTHER, 0)}
        for t, d in obj.get("spans", {}).items():
            st.spans[t] = {ENTITY: d.get(ENTITY, []), CONTEXT: d.get(CONTEXT, []), OTHER: d.get(OTHER, [])}
        return st


def _join(units: Sequence[str], a: int, b: int, unit_level: str) -> str:
    seg = units[max(0, a):b]
    return "".join(seg) if unit_level == "char" else " ".join(seg)


def compute_deer_stats(docs, C: int = 2, unit_level: str = "token", span_cap: int = 50) -> DeerStats:
    """Compute DEER token stats over a list of Doc objects (loaders.Doc).

    ``docs`` items must expose ``.units`` (list[str]) and ``.entities`` (each with
    ``.start`` / ``.end`` / ``.type`` / ``.surface``).

    Each span-library entry is a dict ``{"text", "name", "type"}`` where name/type are the
    GOLD entity the span is about (the containing entity for entity tokens; the nearest
    entity within C for context tokens; None for other tokens).  Faithful examples (the
    paper's Positive / Hard-Negative / Negative blocks) need the real gold entity, not the
    trigger token itself.
    """
    st = DeerStats(C=C, span_cap=span_cap)
    for doc in docs:
        units = doc.units
        n = len(units)
        espans = [(e.start, e.end) for e in doc.entities]
        cls = classify_units(n, espans, C=C)

        # associate each unit index with a gold entity (for example construction)
        assoc = [None] * n
        for e in doc.entities:
            for i in range(max(0, e.start), min(n, e.end)):
                assoc[i] = e
        for e in doc.entities:                      # context tokens -> nearest owning entity
            for i in range(max(0, e.start - C), e.start):
                if cls[i] == CONTEXT and assoc[i] is None:
                    assoc[i] = e
            for i in range(e.end, min(n, e.end + C)):
                if cls[i] == CONTEXT and assoc[i] is None:
                    assoc[i] = e

        for i, tok in enumerate(units):
            cl = cls[i]
            st.counts[tok][cl] += 1
            bucket = st.spans[tok][cl]
            if len(bucket) < span_cap:
                ent = assoc[i]
                bucket.append({
                    "text": _join(units, i - C, i + C + 1, unit_level),
                    "name": ent.surface if ent is not None else None,
                    "type": ent.type if ent is not None else None,
                })
    st.counts = {t: c for t, c in st.counts.items()}
    st.spans = {t: d for t, d in st.spans.items()}
    return st
