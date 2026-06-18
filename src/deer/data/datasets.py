"""Dataset registry — one place that knows, per dataset, everything the pipeline needs to
differ on: loader, prompt-facing entity-type set, unit level (char/token), domain phrase,
prompt language, and the (start-of-migration) DEER hyper-parameters.

Why a registry: the DEER/REINDEER stages (retrieval / generation / align / stats / reflect /
eval) are already dataset-agnostic — they take ``unit_level`` / ``entity_types`` / ``domain``
/ ``prompt_lang`` as parameters.  The ONLY thing NCBI-specific was the runner.  This module
holds the per-dataset values so a single general runner (scripts/run_pipeline.py) can drive
ncbi / cmeee / genia without forking code.

Hyper-parameters
----------------
NCBI uses Table 10 (Bai et al. 2025).  **CMeEE and GENIA are NOT in DEER**, so there is no
published Table-10 row for them; we start from DEER's documented default grid value
(λ1=λ2=1, C=2, w_e=w_c=1, w_o=0.01, θ_FN=0.95, M=1, K=2, N=8) and tune on dev later
(experiment-plan §2.2/§8).  Faithfulness is only claimed on NCBI; CMeEE/GENIA setup is our
controlled design.

CMeEE type labels
-----------------
CMeEE raw stores English abbreviations ("dis"/"sym"/...), but the researcher's decision
(2026-06-15) is to show the model **Chinese full type names** under a **Chinese scaffold**.
So for CMeEE we remap every gold entity's ``type`` abbrev -> Chinese name **at load time**,
making the Chinese name the single canonical label across demos / generation / alignment /
eval.  The flat DEER statistics (P(t_e) etc.) never look at the type string, so this remap is
inert for them; the precomputed nested_stats (keyed by abbrev) are only used by M3+ nested
passes and are out of scope for this flat migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .loaders import CMEEE_TYPES, GENIA_TYPES, Doc, load_dataset
from .ncbi import NCBI_TYPES, load_ncbi

# CMeEE abbrev -> Chinese full name (data/CMeEE-V2/type_mapping/type_mapping.json)
CMEEE_ABBR_TO_ZH: Dict[str, str] = {
    "dis": "疾病", "sym": "临床表现", "dru": "药物", "equ": "医疗设备",
    "pro": "医疗程序", "bod": "身体", "ite": "医学检验项目", "mic": "微生物类", "dep": "科室",
}
# canonical prompt-facing order = CMEEE_TYPES order, mapped to Chinese
CMEEE_TYPES_ZH: List[str] = [CMEEE_ABBR_TO_ZH[t] for t in CMEEE_TYPES]


# CMeEE held-out evaluation regions (notes/2026-06-16-test-significance-plan.md §1).
# CMeEE official test labels are hidden (CBLUE leaderboard), so we carve mutually-exclusive,
# reproducible regions out of the 5000-doc dev split (a gap is left after HELDOUT to avoid
# any boundary contamination; the rest is simply unused). Tuning happens ONLY on TUNE; the
# held-out region is scored ONCE on the final systems and never used to pick hyper-params.
CMEEE_TUNE = (0, 1000)        # dev[:1000]  — tuning / structural-decision confirmation
CMEEE_HELDOUT = (1000, 2500)  # dev[1000:2500] (n=1500) — final held-out test, eval once

# DEER default grid hyper-params (most datasets, incl. NCBI's Table-10 row)
_DEER_DEFAULT_HP: Dict[str, float] = {
    "lambda1": 1.0, "lambda2": 1.0, "C": 2, "w_e": 1.0, "w_c": 1.0, "w_o": 0.01,
    "theta_fn": 0.95, "M": 1, "K": 2, "n_shots": 8,
}


@dataclass
class DatasetSpec:
    name: str
    entity_types: Sequence[str]          # prompt-facing labels (CMeEE = Chinese)
    unit_level: str                      # "char" (CMeEE) | "token" (GENIA/NCBI)
    domain: str                          # domain phrase for reflection prompts
    prompt_lang: str                     # "zh" (CMeEE) | "en"
    hp: Dict[str, float] = field(default_factory=lambda: dict(_DEER_DEFAULT_HP))
    _loader: Callable[[str], List[Doc]] = None  # type: ignore[assignment]
    _type_remap: Optional[Dict[str, str]] = None

    def load(self, split: str) -> List[Doc]:
        docs = self._loader(split)
        if self._type_remap:
            for d in docs:
                for e in d.entities:
                    e.type = self._type_remap.get(e.type, e.type)
        return docs

    def display_to_stat(self) -> Optional[Dict[str, str]]:
        """Map pipeline (display) type -> nested-stat type. CMeEE spans carry Chinese names
        but the nested stats (c_a_in_b / p_end) are keyed by English abbrevs, so N1/N2 must
        translate. None = identity (GENIA/NCBI types already match the stats)."""
        if not self._type_remap:
            return None
        return {disp: abbr for abbr, disp in self._type_remap.items()}


DATASETS: Dict[str, DatasetSpec] = {
    "ncbi": DatasetSpec(
        name="ncbi", entity_types=NCBI_TYPES, unit_level="token",
        domain="biomedical abstract", prompt_lang="en",
        _loader=lambda split: load_ncbi(split),
    ),
    "genia": DatasetSpec(
        name="genia", entity_types=GENIA_TYPES, unit_level="token",
        domain="biomedical literature abstract", prompt_lang="en",
        _loader=lambda split: load_dataset("genia", split),
    ),
    "cmeee": DatasetSpec(
        name="cmeee", entity_types=CMEEE_TYPES_ZH, unit_level="char",
        domain="中文临床医学文本", prompt_lang="zh",
        _loader=lambda split: load_dataset("cmeee", split),
        _type_remap=CMEEE_ABBR_TO_ZH,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name} (have {list(DATASETS)})")
    return DATASETS[name]
