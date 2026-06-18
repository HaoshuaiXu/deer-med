"""DEER generation stage — flat (name, type) extraction via ICL.

Faithful to the paper (notes/2026-06-14-deer-repro-spec.md §3.5):
  - Output is a JSON object with key ``named_entities`` -> list of {"name","type"};
    empty list when no entities.  Generation is identical across all method tiers
    (M0-M4): the gain must come from reflection, not output format.
  - Prompt = JSON template + task instruction + N demos (``Input:.../Output:{json}``) +
    final ``Input: <sentence>\nOutput:``.  Demos are sorted ascending by similarity so the
    most similar demo sits closest to the query (handled by the retrieval module).
  - Decoding temperature = 0.

The parser is tolerant: it extracts the first JSON object containing ``named_entities``
(or the legacy spaced key ``named entities``), tolerates code fences, and drops malformed
items rather than crashing.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

# A demo = (sentence_text, list[(name, type)])
Demo = Tuple[str, List[Tuple[str, str]]]


def _entities_to_json(ents: Sequence[Tuple[str, str]]) -> str:
    return json.dumps(
        {"named_entities": [{"name": n, "type": t} for (n, t) in ents]},
        ensure_ascii=False,
    )


def build_generation_prompt(
    sentence: str,
    demos: Sequence[Demo],
    entity_types: Sequence[str],
    task_hint: Optional[str] = None,
    lang: str = "en",
) -> List[dict]:
    """Build the chat message for one generation query.

    Verbatim to DEER A.1 "In-Context Learning - Input": a single prompt block =
    generic JSON template + task instruction + N ``Input:/Output:`` demos + final
    ``Input:/Output:``.  Only the entity-type enumeration is parameterised — the paper's
    CoNLL03 example reads "Please identify the four types of named entities: \"PER\", ..."; per
    the established convention (used by the reflection prompts too) we substitute the type set
    and drop the hard-coded count word, leaving the rest of the sentence intact.  The template
    keeps the paper's generic ``ent_name_1 ... ent_name_n`` placeholders (NOT a type-substituted
    form), and there is NO system message (the paper's generation prompt has none).

    Args:
      sentence: the test sentence (already detokenised to display form).
      demos: retrieved demos, **in ascending similarity order** (most similar last).
      entity_types: the label set, e.g. ["Disease"] for NCBI.
      task_hint: optional extra instruction appended to the instruction block.
      lang: "en" (DEER-verbatim English scaffold; NCBI/GENIA) | "zh" (Chinese scaffold for
        CMeEE — DEER never ran CMeEE, so the Chinese wording is our design, kept structurally
        isomorphic to A.1: same JSON template + task instruction + demos + final prompt). The
        JSON key stays ``named_entities`` in both langs so the parser is language-agnostic.

    The JSON ``template`` and demo Input/Output structure are identical across langs; only the
    natural-language instruction and the Input/Output labels differ.
    """
    template = ('{"named_entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., '
                '{"name": "ent_name_n", "type": "ent_type_n"}]}')
    if lang == "zh":
        types_str = "、".join(f'"{t}"' for t in entity_types)
        instr = (
            "下面是命名实体识别的 JSON 模板：\n"
            f"{template}\n\n"
            f"请从下面的文本中识别以下类型的命名实体：{types_str}，按上面的 JSON 模板输出 JSON 对象。"
            '若未识别到任何命名实体，输出 {"named_entities": []}。'
        )
        in_label, out_label = "输入", "输出"
    else:
        types_str = ", ".join(f'"{t}"' for t in entity_types)
        instr = (
            "Here is the JSON template for named entity recognition:\n"
            f"{template}\n\n"
            f"Please identify the named entities of the following types: {types_str}, following the "
            "JSON template listed above, and output the JSON object. If no named entities identified, "
            'output {"named_entities": []}.'
        )
        in_label, out_label = "Input", "Output"
    if task_hint:
        instr += "\n" + task_hint

    parts: List[str] = [instr]
    for (demo_sent, demo_ents) in demos:
        parts.append(f"{in_label}: {demo_sent}\n{out_label}: {_entities_to_json(demo_ents)}")
    parts.append(f"{in_label}: {sentence}\n{out_label}:")
    return [{"role": "user", "content": "\n\n".join(parts)}]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_named_entities(text: str) -> List[Tuple[str, str]]:
    """Parse model output into a list of (name, type); tolerant of noise."""
    if not text:
        return []
    s = text.strip()
    # strip code fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    obj = None
    # try direct, then first {...} blob
    for candidate in (s, (_JSON_OBJ_RE.search(s).group(0) if _JSON_OBJ_RE.search(s) else None)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return []
    items = obj.get("named_entities", obj.get("named entities", []))
    if not isinstance(items, list):
        return []
    out: List[Tuple[str, str]] = []
    for it in items:
        if isinstance(it, dict) and "name" in it and "type" in it:
            name, typ = it.get("name"), it.get("type")
            if isinstance(name, str) and isinstance(typ, str) and name.strip():
                out.append((name.strip(), typ.strip()))
    return out


def generate_entities(
    llm,
    sentence: str,
    demos: Sequence[Demo],
    entity_types: Sequence[str],
    task_hint: Optional[str] = None,
    lang: str = "en",
) -> Tuple[List[Tuple[str, str]], str]:
    """Run one generation call; returns (parsed entities, raw text)."""
    messages = build_generation_prompt(sentence, demos, entity_types, task_hint, lang=lang)
    resp = llm.call(messages, temperature=0.0)
    return parse_named_entities(resp.text), resp.text
