"""NCBI-Disease loader (milestone-0 fidelity target).

Token-level, single entity type "Disease".  Reads a per-split JSON/JSONL where each record
has ``tokens`` plus BIO labels in ``ner_tags`` (HuggingFace ``ncbi_disease`` dump) or ``tags``.
Labels may be strings ("O"/"B-Disease"/"I-Disease") or HF integer ids (0=O,1=B,2=I).

Produces the same unified ``Doc``/``Entity`` records as loaders.py (units = word tokens,
spans = half-open token-index [start,end)).

To create the data on the researcher's machine:
    from datasets import load_dataset
    ds = load_dataset("ncbi_disease")
    import json
    for split in ["train","validation","test"]:
        out = [{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in ds[split]]
        json.dump(out, open(f"data/NCBI-disease/{ {'validation':'dev'}.get(split, split) }.json","w"))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .loaders import DATA_ROOT, Doc, Entity

NCBI_DIR = DATA_ROOT / "NCBI-disease"
NCBI_TYPES = ["Disease"]

_ID2BIO = {0: "O", 1: "B-Disease", 2: "I-Disease"}


def _norm_tag(tag) -> str:
    if isinstance(tag, int):
        return _ID2BIO.get(tag, "O")
    return str(tag)


def bio_to_spans(tokens: List[str], tags: List[str]):
    """Convert BIO tags to a list of Entity (token-index half-open spans)."""
    ents: List[Entity] = []
    i, n = 0, len(tags)
    while i < n:
        tag = _norm_tag(tags[i])
        if tag.startswith("B-") or (tag.startswith("I-") and (i == 0 or _norm_tag(tags[i - 1]) == "O")):
            etype = tag.split("-", 1)[1] if "-" in tag else "Disease"
            j = i + 1
            while j < n and _norm_tag(tags[j]).startswith("I-"):
                j += 1
            ents.append(Entity(start=i, end=j, type=etype, surface=" ".join(tokens[i:j])))
            i = j
        else:
            i += 1
    return ents


def _read_records(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def load_ncbi(split: str, data_dir: Path = NCBI_DIR) -> List[Doc]:
    path = Path(data_dir) / f"{split}.json"
    if not path.exists():
        path = Path(data_dir) / f"{split}.jsonl"
    records = _read_records(path)
    docs: List[Doc] = []
    for i, r in enumerate(records):
        tokens = r["tokens"]
        tags = r.get("ner_tags", r.get("tags"))
        ents = bio_to_spans(tokens, tags)
        docs.append(Doc("ncbi", split, f"ncbi-{split}-{i}", tokens, " ".join(tokens), ents))
    return docs
