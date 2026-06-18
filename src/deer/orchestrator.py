"""DEER pipeline orchestrator (config-driven).

Wires the stages: label-guided retrieval -> generation -> (optional) reflection.  The method
tier is a config flag so ablations (M1 = no reflection, M2 = full flat DEER) change only
config, not control flow (DEER paper §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .align.aligner import align_predictions
from .generate.generator import generate_entities
from .reflect.flat.passes import ReflectTrace, reflect_flat

Span = Tuple[int, int, str]


@dataclass
class PipelineConfig:
    method: str = "m2"            # "m1" (no reflection) | "m2" (full flat DEER)
    entity_types: Sequence[str] = field(default_factory=lambda: ["Disease"])
    unit_level: str = "token"     # "token" (GENIA/NCBI) | "char" (CMeEE)
    domain: str = "text"          # domain phrase in reflection prompts (e.g. "biomedical abstract")
    prompt_lang: str = "en"       # "en" (DEER-verbatim; NCBI/GENIA) | "zh" (CMeEE scaffold)
    reflect_passes: tuple = ("unseen", "fn", "boundary")  # flat ER sub-steps to run (ablation)
    n_shots: int = 8
    # reflection (NCBI defaults, Table 10)
    C: int = 2
    theta_fn: float = 0.95
    K: int = 2
    M: int = 1
    surround_thresh: float = 0.5


class DeerPipeline:
    def __init__(self, config: PipelineConfig, retriever, stats, llm):
        self.cfg = config
        self.retriever = retriever
        self.stats = stats
        self.llm = llm

    def _demos(self, test_doc) -> List[Tuple[str, List[Tuple[str, str]]]]:
        picks = self.retriever.retrieve(test_doc, n=self.cfg.n_shots)
        demos = []
        for (idx, _score) in picks:
            d = self.retriever.pool[idx]
            ents = [(e.surface, e.type) for e in d.entities]
            demos.append((d.text, ents))
        return demos

    def predict(self, test_doc, return_trace: bool = False):
        demos = self._demos(test_doc)
        preds, raw = generate_entities(self.llm, test_doc.text, demos, self.cfg.entity_types,
                                       lang=self.cfg.prompt_lang)
        aligned = align_predictions(test_doc.units, preds, self.cfg.unit_level)
        spans: List[Span] = [a.as_tuple() for a in aligned if a.status == "aligned"]
        # dedupe
        spans = list(dict.fromkeys(spans))

        trace = ReflectTrace() if (return_trace or self.cfg.method != "m1") else None
        if self.cfg.method != "m1":  # frozen flat-only path (DEER fidelity)
            spans = reflect_flat(
                test_doc.units, spans, self.stats, self.llm, self.cfg.entity_types,
                C=self.cfg.C, theta_fn=self.cfg.theta_fn, K=self.cfg.K, M=self.cfg.M,
                surround_thresh=self.cfg.surround_thresh, unit_level=self.cfg.unit_level,
                domain=self.cfg.domain, lang=self.cfg.prompt_lang,
                passes=self.cfg.reflect_passes, trace=trace,
            )

        spans = list(dict.fromkeys(spans))
        if return_trace:
            return spans, (trace.entries if trace else [])
        return spans

    def predict_corpus(self, test_docs: Sequence) -> List[List[Span]]:
        return [self.predict(d) for d in test_docs]
