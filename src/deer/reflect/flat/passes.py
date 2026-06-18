"""DEER flat reflection passes — Unseen -> False-Negative -> Boundary (single sequential pass).

The prompts are transcribed **verbatim** from DEER's Appendix A.1 (Bai et al. 2025); see
notes/2026-06-14-deer-repro-spec.md §8.  The paper illustrates them on CoNLL03; per the
authors' own per-dataset practice we substitute ONLY the entity-type set and the domain
phrase (e.g. "biomedical abstract" / "Disease").  We do NOT otherwise reword the instructions
— for a fidelity reproduction the prompt is part of the method and must not be tuned to the
metric.

Faithful structure (corrected from the earlier reconstruction):
  - Unseen / FalseNegative evaluate ALL candidate tokens of a sentence in ONE call, with a
    CoT output format, and end with a final ``{"named_entities":[...]}`` list of entities to
    ADD ("no change" -> empty list).
  - FalseNegative / Boundary inject a fixed ``token_stat`` and Positive / Hard-Negative /
    Negative example buckets (drawn from the Step-0 span libraries).
  - Boundary returns a single corrected entity, the original (keep), or ``{}`` (delete).

Example-selection mechanics (which exact spans to show) are underspecified in the paper; we
draw from the Step-0 span libraries (Positive=entity spans, Hard-Neg=context spans,
Neg=other spans) — documented as a faithful approximation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ...align.aligner import _find_occurrences, _name_to_units
from .triggers import detect_unseen_triggers, detect_fn_triggers, detect_boundary_triggers

Span = Tuple[int, int, str]


# --------------------------------------------------------------------------- #
# A.1 JSON templates + CoT "Output Format" scaffolds (verbatim, generic       #
# ent_name_1..ent_name_n / token_1..token_n placeholders as printed in the    #
# paper; the field-value "..." are the paper's own template ellipses).        #
# --------------------------------------------------------------------------- #
NE_TEMPLATE = ('{"named_entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., '
               '{"name": "ent_name_n", "type": "ent_type_n"}]}')
SINGLE_TEMPLATE = '{"name": "ent_name", "type": "ent_type"}'

UNSEEN_OUTFMT = (
    "Output Format:\n"
    "Candidate Token: token_1\n"
    "Contextual Meaning: ...\n"
    "Relation to Examples Provided: ...\n"
    "Rationale: ...\n"
    "Updates: ... (add a new entity/no change)\n\n"
    "Candidate Token: token_n\n"
    "Contextual Meaning: ...\n"
    "Relation to Examples Provided: ...\n"
    "Rationale: ...\n"
    "Updates: ... (add a new entity/no change)\n\n"
    "Final predicted entities for the input text (JSON format):"
)

FN_OUTFMT = (
    "Output Format:\n"
    "Candidate Token: token_1\n"
    "Training Data Stats: ... entity ... context ... regular ...\n"
    "Contextual Meaning: ...\n"
    "Relation to Examples Provided: ... positive examples ... negative examples ...\n"
    "Rationale: ...\n"
    "Updates: ... (add a new entity/no change)\n\n"
    "Candidate Token: token_n\n"
    "Training Data Stats: ... entity ... context ... regular ...\n"
    "Contextual Meaning: ...\n"
    "Relation to Examples Provided: ... positive examples ... negative examples ...\n"
    "Rationale: ...\n"
    "Updates: ... (add a new entity/no change)\n\n"
    "Final predicted entities for the input text (JSON format):"
)

BOUNDARY_OUTFMT = (
    "Output Format:\n"
    "Boundary Token: token_1\n"
    "Training Data Stats: ... entity ... context ... regular ...\n"
    "Contextual Meaning: ...\n"
    "Rationale: ... positive examples ... negative examples ... data stats ...\n\n"
    "Boundary Token: token_n\n"
    "Training Data Stats: ... entity ... context ... regular ...\n"
    "Contextual Meaning: ...\n"
    "Rationale: ... positive examples ... negative examples ... data stats ...\n\n"
    "Updated Predicted Entity (JSON format):"
)

# --------------------------------------------------------------------------- #
# Chinese scaffold (CMeEE only, prompt_lang="zh"). Structurally isomorphic to  #
# the A.1 English versions above; only natural-language wording + CoT field    #
# labels are translated. JSON templates/keys and <...> structural tags stay    #
# English (they are code/delimiters). DEER never ran CMeEE, so the Chinese      #
# wording is our design — kept structurally faithful, not metric-tuned.        #
# --------------------------------------------------------------------------- #
ZH_FINAL_MARKER = "最终预测实体"           # mirrors "Final predicted entities ..."
ZH_BOUNDARY_MARKER = "修正后的预测实体"     # mirrors "Updated Predicted Entity ..."

UNSEEN_OUTFMT_ZH = (
    "输出格式：\n"
    "候选 token：token_1\n"
    "上下文含义：...\n"
    "与所给示例的关系：...\n"
    "依据：...\n"
    "更新：...（新增一个实体／无变化）\n\n"
    "候选 token：token_n\n"
    "上下文含义：...\n"
    "与所给示例的关系：...\n"
    "依据：...\n"
    "更新：...（新增一个实体／无变化）\n\n"
    f"{ZH_FINAL_MARKER}（JSON 格式）："
)

FN_OUTFMT_ZH = (
    "输出格式：\n"
    "候选 token：token_1\n"
    "训练数据统计：... 实体 ... 上下文 ... 普通 ...\n"
    "上下文含义：...\n"
    "与所给示例的关系：... 正例 ... 负例 ...\n"
    "依据：...\n"
    "更新：...（新增一个实体／无变化）\n\n"
    "候选 token：token_n\n"
    "训练数据统计：... 实体 ... 上下文 ... 普通 ...\n"
    "上下文含义：...\n"
    "与所给示例的关系：... 正例 ... 负例 ...\n"
    "依据：...\n"
    "更新：...（新增一个实体／无变化）\n\n"
    f"{ZH_FINAL_MARKER}（JSON 格式）："
)

BOUNDARY_OUTFMT_ZH = (
    "输出格式：\n"
    "边界 token：token_1\n"
    "训练数据统计：... 实体 ... 上下文 ... 普通 ...\n"
    "上下文含义：...\n"
    "依据：... 正例 ... 负例 ... 数据统计 ...\n\n"
    "边界 token：token_n\n"
    "训练数据统计：... 实体 ... 上下文 ... 普通 ...\n"
    "上下文含义：...\n"
    "依据：... 正例 ... 负例 ... 数据统计 ...\n\n"
    f"{ZH_BOUNDARY_MARKER}（JSON 格式）："
)

SYS_MSG = {"en": "You are an expert named-entity recognition system.",
           "zh": "你是一个专业的命名实体识别系统。"}


# --------------------------------------------------------------------------- #
# parsing / span helpers                                                      #
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _last_named_entities(text: str) -> List[Tuple[str, str]]:
    """Parse the FINAL {"named_entities":[...]} object from a CoT response."""
    if not text:
        return []
    # prefer content after the 'Final predicted entities' marker if present (en or zh)
    marker = max(text.rfind("Final predicted"), text.rfind(ZH_FINAL_MARKER))
    s = text[marker:] if marker != -1 else text
    # find all {...} blobs, try from the last
    objs = re.findall(r"\{.*\}", s, re.DOTALL)
    for blob in reversed(objs):
        try:
            o = json.loads(blob)
        except json.JSONDecodeError:
            continue
        items = o.get("named_entities", o.get("named entities"))
        if isinstance(items, list):
            out = []
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("name"), str) and it["name"].strip():
                    out.append((it["name"].strip(), str(it.get("type", "")).strip()))
            return out
    return []


def _last_single_entity(text: str):
    """Parse a single {"name","type"} or {} from the end of a boundary response.

    Returns: ("entity", (name,type)) | ("delete", None) | ("keep", None)
    """
    if not text:
        return ("keep", None)
    marker = max(text.rfind("Updated Predicted Entity"), text.rfind(ZH_BOUNDARY_MARKER))
    s = text[marker:] if marker != -1 else text
    objs = re.findall(r"\{.*?\}", s, re.DOTALL)
    for blob in reversed(objs):
        try:
            o = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if o == {}:
            return ("delete", None)
        if isinstance(o.get("name"), str) and o["name"].strip():
            return ("entity", (o["name"].strip(), str(o.get("type", "")).strip()))
    return ("keep", None)


def _join(units: Sequence[str], a: int, b: int, unit_level: str) -> str:
    seg = units[max(0, a):b]
    return "".join(seg) if unit_level == "char" else " ".join(seg)


def _resolve_span(units, name, etype, unit_level, anchors=()) -> Optional[Span]:
    target = _name_to_units(name, unit_level)
    occ = _find_occurrences(units, target)
    if not occ:
        return None
    if anchors:
        covering = [sp for sp in occ if any(sp[0] <= a < sp[1] for a in anchors)]
        if covering:
            occ = covering
    s, e = occ[0]
    return (s, e, etype or "")


# --------------------------------------------------------------------------- #
# prompt building (verbatim A.1 instructions, type/domain parameterised)      #
# --------------------------------------------------------------------------- #
def _types_phrase(entity_types: Sequence[str], lang: str = "en") -> str:
    sep = "、" if lang == "zh" else ", "
    return sep.join(f'"{t}"' for t in entity_types)


def _token_stat_json(stats, tok: str) -> str:
    c = stats.counts.get(tok, {})
    e, ct, o = c.get("entity", 0), c.get("context", 0), c.get("other", 0)
    return json.dumps({
        "num_occurrences_as_entity": e,
        "num_occurrences_as_context_tokens": ct,
        "num_occurrences_as_other_tokens": o,
        "entity_vs_context_count": f"{e} vs {ct}",
        "entity_vs_non_entity_count": f"{e} vs {ct + o}",
    })


def _ex_text(entry) -> str:
    return entry["text"] if isinstance(entry, dict) else entry


def _ex_entity(entry, default_type: str):
    """Return the gold (name, type) a span entry is about, or None."""
    if isinstance(entry, dict) and entry.get("name"):
        return (entry["name"], entry.get("type") or default_type)
    return None


# neg_label differs per pass (FalseNegative vs Boundary), per language.
NEG_LABEL = {
    "fn": {"en": "other tokens, not entity nor context", "zh": "其他 token，既非实体也非上下文"},
    "boundary": {"en": "regular tokens, neither entity nor context", "zh": "普通 token，既非实体也非上下文"},
}
_EX_LABELS = {
    "en": ("Positive Examples (part of entity):", "Hard Negative Examples (context tokens):",
           "Negative Examples", "Input", "Output"),
    "zh": ("正例（属于实体的一部分）：", "难负例（上下文 token）：", "负例", "输入", "输出"),
}


def _example_block_fn(stats, tok: str, default_type: str, M: int,
                      neg_label: str = "other tokens, not entity nor context",
                      lang: str = "en") -> str:
    """Positive / Hard-Negative / Negative example block (A.1).

    ``neg_label`` differs between passes: FalseNegative prints "other tokens, not entity nor
    context"; Boundary prints "regular tokens, neither entity nor context" (paper A.1).
    """
    pos_l, hn_l, neg_l, in_l, out_l = _EX_LABELS["zh" if lang == "zh" else "en"]
    sp = stats.spans.get(tok, {})
    lines = [pos_l]
    for e in sp.get("entity", [])[:M]:
        ent = _ex_entity(e, default_type) or (tok, default_type)
        lines.append(f"{in_l}: {_ex_text(e)}\n{out_l}: {json.dumps({'name': ent[0], 'type': ent[1]}, ensure_ascii=False)}")
    lines.append(hn_l)
    for e in sp.get("context", [])[:M]:
        ent = _ex_entity(e, default_type)  # the real nearby entity (token itself is NOT it)
        out = json.dumps({"name": ent[0], "type": ent[1]}, ensure_ascii=False) if ent else "{}"
        lines.append(f"{in_l}: {_ex_text(e)}\n{out_l}: {out}")
    lines.append(f"{neg_l}（{neg_label}）：" if lang == "zh" else f"{neg_l} ({neg_label}):")
    for e in sp.get("other", [])[:M]:
        lines.append(f"{in_l}: {_ex_text(e)}\n{out_l}: {{}}")
    return "\n".join(lines)


@dataclass
class ReflectTrace:
    entries: List[dict] = field(default_factory=list)

    def log(self, pass_name: str, action: str, **kw):
        self.entries.append({"pass": pass_name, "action": action, **kw})


# --------------------------------------------------------------------------- #
# passes                                                                       #
# --------------------------------------------------------------------------- #
def run_unseen(units, spans, stats, llm, entity_types, *, C=2, surround_thresh=0.5, M=1,
               unit_level="token", domain="text", lang="en",
               trace: Optional[ReflectTrace] = None) -> List[Span]:
    triggers = detect_unseen_triggers(units, spans, stats, C=C, surround_thresh=surround_thresh)
    if not triggers:
        return list(spans)
    spans = list(spans)
    n = len(units)
    sent = _join(units, 0, len(units), unit_level)
    cand = [units[i] for i in triggers]
    tphrase = _types_phrase(entity_types, lang)
    in_l, out_l = ("输入", "输出") if lang == "zh" else ("Input", "Output")
    # one A.1 block per candidate: the candidate token, its surrounding *seen* ("potential
    # context") tokens, the context token used to retrieve examples, and that token's
    # entity-span examples.
    blocks = []
    for i in triggers:
        neigh = [j for j in range(max(0, i - C), min(n, i + C + 1))
                 if j != i and stats.is_seen(units[j])]
        pot_ctx = [units[j] for j in neigh]
        ctx_tok = units[neigh[0]] if neigh else ""
        ex_lines = []
        for j in neigh:
            for e in stats.spans.get(units[j], {}).get("entity", [])[:M]:
                ent = _ex_entity(e, entity_types[0]) or (units[j], entity_types[0])
                ex_lines.append(f"{in_l}: {_ex_text(e)}\n{out_l}: {json.dumps({'name': ent[0], 'type': ent[1]}, ensure_ascii=False)}")
        examples = "\n".join(ex_lines) if ex_lines else ("（无）" if lang == "zh" else "(none)")
        blocks.append(f"<candidate_token>\n{units[i]}\n</candidate_token>\n"
                      f"<potential_context_tokens_around>\n{pot_ctx}\n</potential_context_tokens_around>\n"
                      f"<context_token>\n{ctx_tok}\n</context_token>\n"
                      f"<examples>\n{examples}\n</examples>")

    if lang == "zh":
        instr = (
            f"给定来自{domain}的输入文本，评估每个候选 token 及其周围 token，判断它是否应被归类为以下命名实体"
            f"类型之一（或其一部分）：{tphrase}。**如有示例，请参考所给示例。** 若它应成为一个新实体，请从句子中"
            f"抽取确切的文本片段（包含任何空格），以 JSON 格式给出。确保不拆分 token（例如保持带连字符的词完整）。"
            f'注意：缩写与全称是不同的实体。若未做任何修改，返回 {{"named_entities": []}}。'
        )
        outfmt = UNSEEN_OUTFMT_ZH
        tpl_label = "JSON 模板"
    else:
        instr = (
            f"Given the input text from a {domain}, evaluate each candidate token along with the "
            f"surrounding tokens to determine if it should be categorized as (part of) one of the "
            f"named entity types: {tphrase}. **Use provided examples, if available, for reference.** "
            f"If it should be a new entity, extract the exact text span in the sentence, including any "
            f"spaces, in JSON format. Ensure tokens are not split (e.g., maintain hyphenated words "
            f"intact). Note that abbreviations and full names are separate entities. If no changes are "
            f'made, return {{"named_entities": []}}.'
        )
        outfmt = UNSEEN_OUTFMT
        tpl_label = "JSON Template"
    user = (f"<input_text>\n{sent}\n</input_text>\n\n"
            f"<candidate_tokens>\n{cand}\n</candidate_tokens>\n\n"
            + "\n\n".join(blocks) + f"\n\n{instr}\n\n"
            f"{tpl_label}:\n{NE_TEMPLATE}\n\n{outfmt}")
    resp = llm.call([{"role": "system", "content": SYS_MSG["zh" if lang == "zh" else "en"]},
                     {"role": "user", "content": user}])
    for (name, typ) in _last_named_entities(resp.text):
        sp = _resolve_span(units, name, typ or entity_types[0], unit_level, anchors=triggers)
        if sp and sp not in spans:
            spans.append(sp)
            if trace:
                trace.log("unseen", "add", token=name, span=sp)
    return spans


def run_fn(units, spans, stats, llm, entity_types, *, theta_fn=0.95, M=1,
           unit_level="token", domain="text", lang="en",
           trace: Optional[ReflectTrace] = None) -> List[Span]:
    triggers = detect_fn_triggers(units, spans, stats, theta_fn=theta_fn)
    if not triggers:
        return list(spans)
    spans = list(spans)
    sent = _join(units, 0, len(units), unit_level)
    tphrase = _types_phrase(entity_types, lang)
    neg = NEG_LABEL["fn"]["zh" if lang == "zh" else "en"]
    blocks = []
    for i in triggers:
        tok = units[i]
        blocks.append(f"<candidate_token>\n{tok}\n</candidate_token>\n"
                      f"<token_stat>\n{_token_stat_json(stats, tok)}\n</token_stat>\n"
                      f"<examples>\n{_example_block_fn(stats, tok, entity_types[0], M, neg_label=neg, lang=lang)}\n</examples>")
    cand = [units[i] for i in triggers]

    if lang == "zh":
        instr = (
            "请遵循以下说明：\n"
            f"1. 评估上面列出的每个候选 token，判断它是否应被归类为以下命名实体类型之一（或其一部分）：{tphrase}。"
            "请仔细考虑所提供的正例与负例。尤其关注整体统计数据——token 究竟被纳入还是排除在实体之外。难负例"
            "标示的是那些不属于实体、但位于实体附近的 token。\n"
            "2. 仔细对照两类示例。很多时候，同一 token 在正例中被识别为实体的一部分，而在负例中却不是，这通常源于"
            "标注过程中的不一致。若正例与（难）负例看起来相似，**请以统计数据为准**，例如该片段被识别为实体相对于"
            "作为上下文的频次，尤其当数据界限分明时（例如某一频次显著更高）。\n"
            "3. 若该 token 在训练数据中未出现过或极少出现，请运用你的最佳判断，确定它是否应被视为某个**具体**实体的"
            "一部分或其完整名称。\n"
            "4. 若需要任何修改，请通过**从句子中抽取确切的文本片段**给出更新后的实体，包含任何空格、不添加额外 token，"
            "以 JSON 格式给出。确保不拆分 token（例如保持带连字符的词完整）。注意：缩写与全称是不同的实体。"
            '若无需修改，返回 {"named_entities": []}。'
        )
        outfmt, tpl_label = FN_OUTFMT_ZH, "JSON 模板"
    else:
        instr = (
            "Please follow the instructions below:\n"
            f"1. Evaluate each candidate token listed above to determine if it should be categorized as "
            f"(part of) one of the named entity types: {tphrase}. Consider both the positive and negative "
            "examples provided carefully. Pay particular attention to the overall statistical data on "
            "whether tokens are included or excluded from the entity. Hard negative examples highlight "
            "tokens that are not part of the entity but are located near it.\n"
            "2. Review both sets carefully. In many cases, a token may be identified as part of an entity "
            "in positive examples but not in negative ones, likely due to inconsistencies in the annotation "
            "process. if positive and (hard) negative examples seem similar, **base your decision on "
            "statistical data**, such as the frequency of the span being recognized as an entity versus its "
            "context, particularly when the data is clear-cut (e.g., one frequency is significantly higher).\n"
            "3. If the token has not been seen or is rarely seen in the training data, use your best "
            "judgment to determine whether it should be considered as part of or the entire name of a "
            "**specific** entity.\n"
            "4. If any modifications are necessary, provide the updated entities by **extracting the exact "
            "text span in the sentence**, including any spaces and no outside tokens added, in JSON format. "
            "Ensure tokens are not split (e.g., maintain hyphenated words intact). Note that abbreviations "
            'and full names are separate entities. If no changes are required, return {"named_entities": []}.'
        )
        outfmt, tpl_label = FN_OUTFMT, "JSON Template"
    user = (f"<input_text>\n{sent}\n</input_text>\n\n<candidate_tokens>\n{cand}\n</candidate_tokens>\n\n"
            + "\n\n".join(blocks) + f"\n\n{instr}\n\n"
            f"{tpl_label}:\n{NE_TEMPLATE}\n\n{outfmt}")
    resp = llm.call([{"role": "system", "content": SYS_MSG["zh" if lang == "zh" else "en"]},
                     {"role": "user", "content": user}])
    for (name, typ) in _last_named_entities(resp.text):
        sp = _resolve_span(units, name, typ or entity_types[0], unit_level, anchors=triggers)
        if sp and sp not in spans:
            spans.append(sp)
            if trace:
                trace.log("fn", "add", token=name, span=sp)
    return spans


def run_boundary(units, spans, stats, llm, entity_types, *, K=2, M=1,
                 unit_level="token", domain="text", lang="en",
                 trace: Optional[ReflectTrace] = None) -> List[Span]:
    spans = list(spans)
    trig = detect_boundary_triggers(units, spans, stats, K=K)
    by_span: Dict[int, List[Tuple[int, str]]] = {}
    for (si, ti, where) in trig:
        by_span.setdefault(si, []).append((ti, where))
    if not by_span:
        return spans
    sent = _join(units, 0, len(units), unit_level)
    tphrase = _types_phrase(entity_types, lang)
    neg = NEG_LABEL["boundary"]["zh" if lang == "zh" else "en"]
    is_zh = lang == "zh"
    out = list(spans)
    for si, toks in by_span.items():
        s, e, t = spans[si]
        ent_text = _join(units, s, e, unit_level)
        btoks = [units[ti] for ti, _w in toks]
        stat_blocks = []
        for ti, where in toks:
            if is_zh:
                status = "属于该实体" if where == "inside" else "与该实体相邻（当前未包含）"
            else:
                status = "part of the entity" if where == "inside" else "adjacent to the entity (not currently included)"
            stat_blocks.append(f"<boundary_token>\n{units[ti]}\n</boundary_token>\n"
                               f"<status>\n{status}\n</status>\n"
                               f"<token_stat>\n{_token_stat_json(stats, units[ti])}\n</token_stat>\n"
                               "<examples>\n"
                               + _example_block_fn(stats, units[ti], entity_types[0], M,
                                                   neg_label=neg, lang=lang)
                               + "\n</examples>")
        if is_zh:
            instr = (
                "请遵循以下说明：\n"
                "1. 通过评估上面列出的每个边界 token 与预测实体的关系，校准预测实体的边界。若该 token 属于相邻实体，"
                "则不予考虑。请对照所提供的正例与负例，尤其关注边界 token 周围的上下文，以及关于纳入/排除实体的整体"
                "统计数据。难负例标示的是不属于实体、但位于实体附近的 token。\n"
                "2. 仔细对照两类示例。很多时候，同一 token 在正例中被识别为实体的一部分，而在负例中却不是，这通常源于"
                "标注过程中的不一致。若正例与（难）负例看起来相似，**请以统计数据为准**，例如该片段被识别为实体相对于"
                "作为上下文的频次，尤其当数据界限分明时（例如某一频次显著更高）。\n"
                "3. 若该 token 在训练数据中未出现过或极少出现，请运用你的最佳判断，确定它是否应被视为某个**具体**实体的"
                "一部分或其完整名称。\n"
                "4. 判断是否需要任何修改，例如增删边界 token。若需要修改，请通过**从句子中抽取确切的文本片段**给出更新后"
                "的实体，包含任何空格、不添加额外 token，以 JSON 格式给出。注意：缩写与全称是不同的实体。若相对预测实体"
                "既未增添也未移除 token，则输出原实体。若原实体的全部 token 都被移除，则输出 {}。"
            )
            outfmt, tpl_label = BOUNDARY_OUTFMT_ZH, "JSON 模板"
        else:
            instr = (
                "Please follow the instructions below:\n"
                "1. Calibrate the boundary of the predicted entity by evaluating each boundary token "
                "listed above in relation to the predicted entity. Don't consider it if it belongs to "
                "adjacent entities. Check both the provided positive and negative examples, with particular "
                "attention to the context surrounding the boundary token and overall statistical data on "
                "inclusion or exclusion from the entity. Hard negative examples highlight tokens that are "
                "not part of the entity but are located near it.\n"
                "2. Review both sets carefully. In many cases, a token may be identified as part of an entity "
                "in positive examples but not in negative ones, likely due to inconsistencies in the annotation "
                "process. if positive and (hard) negative examples seem similar, **base your decision on "
                "statistical data**, such as the frequency of the span being recognized as an entity versus its "
                "context, particularly when the data is clear-cut (e.g., one frequency is significantly higher).\n"
                "3. If the token has not been seen or is rarely seen in the training data, use your best "
                "judgment to determine whether it should be considered as part of or the entire name of a "
                "**specific** entity.\n"
                "4. Determine whether any modifications, such as adding or removing boundary tokens, are "
                "necessary. If changes are required, provide the updated entity by **extracting the exact "
                "text span in the sentence**, including any spaces and no outside tokens added, in JSON "
                "format. Note that abbreviations and full names are separate entities. If no tokens are added "
                "to or removed from the predicted entity, output the original entity. If all original tokens "
                "are removed, output {}."
            )
            outfmt, tpl_label = BOUNDARY_OUTFMT, "JSON Template"
        user = (f"<input_text>\n{sent}\n</input_text>\n\n"
                f"<predicted_entity>\n{json.dumps({'name': ent_text, 'type': t}, ensure_ascii=False)}\n</predicted_entity>\n\n"
                f"<boundary_tokens>\n{btoks}\n</boundary_tokens>\n\n" + "\n\n".join(stat_blocks)
                + f"\n\n{instr}\n\n{tpl_label}:\n{SINGLE_TEMPLATE}\n\n{outfmt}")
        resp = llm.call([{"role": "system", "content": SYS_MSG["zh" if is_zh else "en"]},
                         {"role": "user", "content": user}])
        action, payload = _last_single_entity(resp.text)
        if action == "delete":
            out[si] = None
            if trace:
                trace.log("boundary", "delete", span=(s, e, t))
        elif action == "entity":
            name, typ = payload
            if name != ent_text:
                sp = _resolve_span(units, name, typ or t, unit_level, anchors=list(range(s, e)))
                if sp:
                    out[si] = sp
                    if trace:
                        trace.log("boundary", "trim", old=(s, e, t), new=sp)
    return [sp for sp in out if sp is not None]


def reflect_flat(units, spans, stats, llm, entity_types, *, C=2, theta_fn=0.95, K=2, M=1,
                 surround_thresh=0.5, unit_level="token", domain="text", lang="en",
                 passes=("unseen", "fn", "boundary"),
                 trace: Optional[ReflectTrace] = None) -> List[Span]:
    """Full flat ER: Unseen -> FN -> Boundary, once (no iteration; spec §4.1).

    ``passes`` selects which sub-steps run, preserving the fixed Unseen->FN->Boundary order
    (DEER §4.1). Default = all three (faithful DEER). Used for per-pass ablation — e.g. on
    CMeEE char-level the flat Boundary over-trims, so ("unseen","fn") isolates the recall passes.
    """
    if "unseen" in passes:
        spans = run_unseen(units, spans, stats, llm, entity_types, C=C, surround_thresh=surround_thresh,
                           M=M, unit_level=unit_level, domain=domain, lang=lang, trace=trace)
    if "fn" in passes:
        spans = run_fn(units, spans, stats, llm, entity_types, theta_fn=theta_fn, M=M,
                       unit_level=unit_level, domain=domain, lang=lang, trace=trace)
    if "boundary" in passes:
        spans = run_boundary(units, spans, stats, llm, entity_types, K=K, M=M,
                             unit_level=unit_level, domain=domain, lang=lang, trace=trace)
    seen, ded = set(), []
    for sp in spans:
        if sp not in seen:
            seen.add(sp)
            ded.append(sp)
    return ded
