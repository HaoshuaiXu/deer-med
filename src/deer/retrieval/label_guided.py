"""Label-guided retrieval — faithful to DEER eq.1-5 (spec §3.1).

  S(s_i, s_t)       = lambda1 * S_token + lambda2 * S_embed                        (eq.1)
  S_token(s_i, s_t) = sum_{t in s_t} 1[t in s_i] * W(t)                            (eq.2)
  W(t)              = w_e*P(t_e)+w_c*P(t_c)+w_o*P(t_o)  if seen, else 1            (eq.3)
  v_s               = sum_{t in s} W(t) * v_t        (uncontextualized token vecs) (eq.4)
  S_embed           = cos(v_si, v_st)                                              (eq.5)

Token weights come from ``DeerStats`` (Step-0).  ``v_t`` are *uncontextualized* token
embeddings: each unique token string is encoded independently by the embedder and cached
(the paper uses text-embedding-3-small; here a local Qwen3-Embedding-0.6B is injected, see
llm/embedder.py).  ``MockEmbedder`` gives deterministic vectors for offline tests.

Demos are returned **ascending by similarity** (most similar closest to the query), per the
paper's prompt ordering.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class MockEmbedder:
    """Deterministic hash-based embeddings for offline unit tests (no network/model)."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) - 0.5 for i in range(self.dim)]
            out.append(vec)
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LabelGuidedRetriever:
    def __init__(
        self,
        pool_docs: Sequence,
        stats,
        w_e: float = 1.0,
        w_c: float = 1.0,
        w_o: float = 0.01,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        embedder: Optional[Embedder] = None,
        unit_level: str = "token",
    ):
        self.pool = list(pool_docs)
        self.stats = stats
        self.w_e, self.w_c, self.w_o = w_e, w_c, w_o
        self.l1, self.l2 = lambda1, lambda2
        self.embedder = embedder
        self.unit_level = unit_level
        self._tok_vec: Dict[str, List[float]] = {}

        self.n_pool = len(self.pool)
        # Precompute per-pool-doc token sets (for eq.2) and sentence vectors (eq.4).
        self.pool_tokens: List[set] = [set(d.units) for d in self.pool]

        # Inverted index token -> np.array(pool indices containing it) for fast eq.2.
        inv: Dict[str, List[int]] = {}
        for idx, toks in enumerate(self.pool_tokens):
            for t in toks:
                inv.setdefault(t, []).append(idx)
        self._inv: Dict[str, np.ndarray] = {t: np.asarray(ids, dtype=np.int64) for t, ids in inv.items()}

        # Normalised pool embedding matrix (n_pool x dim) for fast eq.5 (cosine = dot).
        self._P: Optional[np.ndarray] = None
        self.pool_vecs: List[Optional[List[float]]] = [None] * self.n_pool
        if self.embedder is not None and self.l2 != 0:
            uniq = {t for d in self.pool for t in d.units}
            print(f"[retriever] warming embedding index over {self.n_pool} pool docs "
                  f"({len(uniq)} unique tokens) — one-time, cached afterwards ...", flush=True)
            self._warm_token_vecs(list(uniq))
            self.pool_vecs = [self._sentence_vec(d.units) for d in self.pool]
            dim = next((len(v) for v in self.pool_vecs if v is not None), 0)
            P = np.zeros((self.n_pool, dim), dtype=np.float64)  # float64 to match prior precision
            for i, v in enumerate(self.pool_vecs):
                if v is not None:
                    P[i] = v
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._P = P / norms
            print("[retriever] embedding index ready.", flush=True)

    # -- weights & vectors -------------------------------------------------- #
    def _w(self, token: str) -> float:
        return self.stats.token_weight(token, self.w_e, self.w_c, self.w_o)

    def prewarm(self, docs) -> None:
        """Encode all token vectors for ``docs`` up front (single-threaded).

        Call before parallel prediction so the embedding cache is read-only during the
        concurrent loop (its dict/file mutations are not thread-safe).
        """
        if self.embedder is not None and self.l2 != 0:
            toks = list({t for d in docs for t in d.units})
            self._warm_token_vecs(toks)

    def _warm_token_vecs(self, tokens: Sequence[str]) -> None:
        need = sorted({t for t in tokens if t not in self._tok_vec})
        if not need:
            return
        vecs = self.embedder.encode(need)
        for t, v in zip(need, vecs):
            self._tok_vec[t] = list(v)

    def _sentence_vec(self, units: Sequence[str]) -> Optional[List[float]]:
        self._warm_token_vecs(units)
        acc: Optional[List[float]] = None
        for t in units:
            v = self._tok_vec.get(t)
            if v is None:
                continue
            w = self._w(t)
            if acc is None:
                acc = [w * x for x in v]
            else:
                for i in range(len(acc)):
                    acc[i] += w * v[i]
        return acc

    # -- scoring (vectorised) ----------------------------------------------- #
    def _s_token_vec(self, test_units: Sequence[str]) -> np.ndarray:
        """eq.2 via inverted index: each unique query token adds W(t) to pools containing it."""
        scores = np.zeros(self.n_pool, dtype=np.float64)
        for t in set(test_units):
            ids = self._inv.get(t)
            if ids is not None:
                scores[ids] += self._w(t)
        return scores

    def _s_embed_vec(self, test_units: Sequence[str]) -> np.ndarray:
        """eq.5 as a single matrix-vector product (cosine via normalised dot)."""
        if self._P is None:
            return np.zeros(self.n_pool, dtype=np.float64)
        v = self._sentence_vec(test_units)
        if v is None:
            return np.zeros(self.n_pool, dtype=np.float64)
        q = np.asarray(v, dtype=np.float64)
        nq = np.linalg.norm(q)
        if nq == 0:
            return np.zeros(self.n_pool, dtype=np.float64)
        return self._P @ (q / nq)

    def score_all(self, test_doc) -> np.ndarray:
        scores = self.l1 * self._s_token_vec(test_doc.units)
        if self._P is not None:
            scores = scores + self.l2 * self._s_embed_vec(test_doc.units)
        return scores

    def retrieve(self, test_doc, n: int = 8, exclude_self: bool = True) -> List[Tuple[int, float]]:
        """Return up to n (pool_index, score), **ascending by score** (most similar last)."""
        scores = self.score_all(test_doc)
        order = np.argsort(scores)[::-1]  # descending
        test_id = getattr(test_doc, "did", None)
        picked: List[Tuple[int, float]] = []
        for i in order:
            i = int(i)
            if exclude_self and test_id is not None and getattr(self.pool[i], "did", None) == test_id:
                continue
            picked.append((i, float(scores[i])))
            if len(picked) >= n:
                break
        picked.sort(key=lambda x: x[1])  # ascending: most similar closest to the query
        return picked
