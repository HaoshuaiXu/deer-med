"""Data loading → unified internal format.

Unified record (Doc):
  - dataset: "cmeee" | "genia"
  - split:   "train" | "dev" | "test"
  - did:     stable id within (dataset, split)
  - units:   list[str]  — basic statistical units. CMeEE = characters, GENIA = word-tokens.
  - text:    str        — original text (CMeEE) or space-joined tokens (GENIA, for display)
  - entities: list[Entity], each span is half-open [start, end) in UNIT space.

Offset conventions (empirically verified, see README):
  - CMeEE raw uses start_idx/end_idx with text[start_idx:end_idx] == entity  → half-open [start,end).
  - GENIA raw uses token start/end with 0<=start<end<=len(tokens)            → half-open [start,end).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

# Repo root = three levels up from this file (src/deer/data/loaders.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"

CMEEE_DIR = DATA_ROOT / "CMeEE-V2" / "raw"
GENIA_DIR = DATA_ROOT / "genia_term_corpus" / "raw"

CMEEE_TYPES = ["bod", "dep", "dis", "dru", "equ", "ite", "mic", "pro", "sym"]
GENIA_TYPES = ["DNA", "RNA", "cell_line", "cell_type", "protein"]


@dataclass
class Entity:
    start: int          # inclusive, unit space
    end: int            # exclusive, unit space
    type: str
    surface: str        # surface string of the span

    def as_tuple(self):
        return (self.start, self.end, self.type)


@dataclass
class Doc:
    dataset: str
    split: str
    did: str
    units: List[str]
    text: str
    entities: List[Entity] = field(default_factory=list)

    def __len__(self):
        return len(self.units)


def load_cmeee(split: Literal["train", "dev", "test"]) -> List[Doc]:
    path = CMEEE_DIR / f"{split}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Doc] = []
    for i, d in enumerate(raw):
        text = d["text"]
        units = list(text)  # character units
        ents: List[Entity] = []
        for e in d.get("entities", []):
            s, en = int(e["start_idx"]), int(e["end_idx"])
            surface = e["entity"]
            ents.append(Entity(start=s, end=en, type=e["type"], surface=surface))
        docs.append(Doc("cmeee", split, f"cmeee-{split}-{i}", units, text, ents))
    return docs


def load_genia(split: Literal["train", "dev", "test"]) -> List[Doc]:
    path = GENIA_DIR / f"{split}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Doc] = []
    for i, d in enumerate(raw):
        toks = d["tokens"]
        text = " ".join(toks)
        ents: List[Entity] = []
        for e in d.get("entities", []):
            s, en = int(e["start"]), int(e["end"])
            surface = " ".join(toks[s:en])
            ents.append(Entity(start=s, end=en, type=e["type"], surface=surface))
        docs.append(Doc("genia", split, f"genia-{split}-{i}", toks, text, ents))
    return docs


def load_dataset(dataset: str, split: str) -> List[Doc]:
    if dataset == "cmeee":
        return load_cmeee(split)  # type: ignore[arg-type]
    if dataset == "genia":
        return load_genia(split)  # type: ignore[arg-type]
    raise ValueError(f"unknown dataset: {dataset}")


def verify_offsets(docs: List[Doc]) -> dict:
    """Sanity-check that span surfaces match the unit slices."""
    total = 0
    mismatch = 0
    for doc in docs:
        for e in doc.entities:
            total += 1
            sliced = "".join(doc.units[e.start:e.end]) if doc.dataset == "cmeee" else " ".join(doc.units[e.start:e.end])
            if sliced != e.surface:
                mismatch += 1
    return {"total": total, "mismatch": mismatch}
