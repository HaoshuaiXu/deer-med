"""Token embedder for the retrieval embedding path (DEER eq.4-5).

DEER used OpenAI text-embedding-3-small *uncontextualised* token vectors.  Per the
researcher's decision (2026-06-14) we use a **local Qwen3-Embedding-0.6B** instead — note the
encoder differs from the paper, so absolute NCBI numbers shift (record this in the fidelity
report); it does not bias the M2-vs-M4 relative gain.

Each unique token string is encoded independently (uncontextualised) and the results cached
to disk.  Runs on the researcher's machine (the model files live there; the sandbox has
neither the weights nor egress).  ``MockEmbedder`` in retrieval/label_guided.py covers offline
tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class Qwen3Embedder:
    """Local Qwen3-Embedding-0.6B token embedder with a disk cache.

    Args:
      model_path: local path to the model, e.g. /Users/xuhaoshuai/models/Qwen/Qwen3-Embedding-0.6B
      cache_path: cache base path; the legacy ``.json`` value is accepted and the on-disk
        cache is stored as ``<base>.npy`` (float32 token vectors) + ``<base>.keys.json``
        (token order). Falls back to reading a legacy ``<base>.json`` dict once and migrates
        it to the binary form on first flush.
      batch_size: encode batch size.
      device: passed to sentence-transformers ("cpu"/"cuda"/"mps").
    """

    def __init__(self, model_path: str, cache_path: str = "outs/embed_cache.json",
                 batch_size: int = 256, device: Optional[str] = None,
                 max_seq_length: Optional[int] = None):
        self.model_path = model_path
        # Legacy JSON-dict cache path (kept for read-back / migration); the binary cache lives
        # alongside it as <base>.npy + <base>.keys.json.
        self.cache_path = Path(cache_path)
        base = self.cache_path
        if base.suffix == ".json":
            base = base.with_suffix("")  # strip ".json" -> base name
        self._npy_path = base.with_suffix(".npy")
        self._keys_path = Path(str(base) + ".keys.json")
        self.batch_size = batch_size
        self.device = device
        # Cap sequence length when encoding long inputs (e.g. full sentences for GPT-NER's
        # sentence-level retrieval); Qwen3-Embedding's 32k default context blows up MPS memory
        # on a batch of long sentences. Harmless for short DEER tokens (well under the cap).
        self.max_seq_length = max_seq_length
        self._cache: Dict[str, List[float]] = {}
        if self._npy_path.exists() and self._keys_path.exists():
            # Preferred binary cache.
            keys = json.loads(self._keys_path.read_text(encoding="utf-8"))
            vecs = np.load(self._npy_path)
            self._cache = {k: vecs[i].tolist() for i, k in enumerate(keys)}
        elif self.cache_path.exists():
            # Legacy JSON-dict cache: load once, then migrate to binary so disk shrinks ~5x.
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self._cache:
                self._flush()
        self._model = None  # lazy

    def _pick_device(self) -> Optional[str]:
        if self.device:
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local dependency
            dev = self._pick_device()
            print(f"[Qwen3Embedder] loading {self.model_path} on {dev} ...", flush=True)
            self._model = SentenceTransformer(self.model_path, device=dev)
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
            print("[Qwen3Embedder] model loaded.", flush=True)

    def encode(self, texts: Sequence[str], chunk: int = 2000) -> List[List[float]]:
        need = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if need:
            self._ensure_model()
            total = len(need)
            print(f"[Qwen3Embedder] encoding {total} new tokens "
                  f"(cached={len(self._cache)}) ...", flush=True)
            for i in range(0, total, chunk):
                part = need[i:i + chunk]
                vecs = self._model.encode(
                    part, batch_size=self.batch_size, normalize_embeddings=True,
                    show_progress_bar=True,
                )
                for t, v in zip(part, vecs):
                    self._cache[t] = [float(x) for x in v]
                self._flush()  # persist each chunk so a kill doesn't lose progress
                print(f"[Qwen3Embedder]   {min(i + chunk, total)}/{total} done", flush=True)
        return [self._cache[t] for t in texts]

    def _flush(self):
        """Persist the cache as <base>.npy (float32 matrix) + <base>.keys.json (token order).

        Written atomically (temp file + os.replace) so a kill mid-flush cannot corrupt the
        cache. ~5x smaller than the legacy JSON-dict text and faithful (model outputs float32).
        """
        if not self._cache:
            return
        self._npy_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache.keys())
        vecs = np.asarray([self._cache[k] for k in keys], dtype=np.float32)

        tmp_npy = self._npy_path.with_name(self._npy_path.name + ".tmp")
        with open(tmp_npy, "wb") as f:  # file handle => np.save won't re-append ".npy"
            np.save(f, vecs)
        os.replace(tmp_npy, self._npy_path)

        tmp_keys = self._keys_path.with_name(self._keys_path.name + ".tmp")
        tmp_keys.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_keys, self._keys_path)
