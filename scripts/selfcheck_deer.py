"""Offline self-check for the DEER pipeline (no network, no model weights).

Covers: Step-0 token stats (classification + P() + adjacent-entity truncation), trigger
detection, and a full retrieval->generate->reflect->eval run with a MockEmbedder + mock LLM.

Run:  python scripts/selfcheck_deer.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deer.data.loaders import Doc, Entity  # noqa: E402
from deer.data.ncbi import bio_to_spans  # noqa: E402
from deer.stats.deer_token_stats import classify_units, compute_deer_stats  # noqa: E402
from deer.retrieval import LabelGuidedRetriever, MockEmbedder  # noqa: E402
from deer.reflect.flat.triggers import detect_fn_triggers, detect_boundary_triggers  # noqa: E402
from deer.orchestrator import DeerPipeline, PipelineConfig  # noqa: E402
from deer.eval import evaluate  # noqa: E402
from deer.llm import LLMClient, LLMConfig  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [XX] {name}")


def doc(did, tokens, spans):
    ents = [Entity(s, e, t, " ".join(tokens[s:e])) for (s, e, t) in spans]
    return Doc("ncbi", "train", did, tokens, " ".join(tokens), ents)


def test_stats():
    print("\n== DEER step-0 stats ==")
    toks = ["The", "risk", "of", "cancer", ",", "especially"]
    cls = classify_units(len(toks), [(3, 4)], C=2)
    check("classify: cancer=entity", cls[3] == "entity")
    check("classify: risk/of/comma/especially=context",
          cls[1] == "context" and cls[2] == "context" and cls[4] == "context" and cls[5] == "context")
    check("classify: The=other", cls[0] == "other")

    # adjacent-entity truncation
    toks2 = ["Lionel", "Messi", "and", "Cristiano", "Ronaldo"]
    cls2 = classify_units(len(toks2), [(0, 2), (3, 5)], C=2)
    check("adjacent truncation: and=context, names=entity",
          cls2 == ["entity", "entity", "context", "entity", "entity"])

    st = compute_deer_stats([doc("d0", toks, [(3, 4, "Disease")])], C=2, unit_level="token")
    check("P(cancer entity)=1.0", abs(st.p_entity("cancer") - 1.0) < 1e-9)
    check("P(of context)=1.0", abs(st.p_context("of") - 1.0) < 1e-9)
    check("counts cancer entity=1", st.counts["cancer"]["entity"] == 1)
    check("span lib has cancer entity span", len(st.spans["cancer"]["entity"]) == 1)
    check("token_weight unseen=1.0", st.token_weight("NEVERSEEN", 1, 1, 0.01) == 1.0)


def test_bio():
    print("\n== BIO -> spans ==")
    toks = ["patients", "with", "breast", "cancer", "."]
    tags = ["O", "O", "B-Disease", "I-Disease", "O"]
    ents = bio_to_spans(toks, tags)
    check("one span (2,4)", len(ents) == 1 and (ents[0].start, ents[0].end) == (2, 4))
    # integer HF tags
    ents2 = bio_to_spans(toks, [0, 0, 1, 2, 0])
    check("int tags same result", len(ents2) == 1 and (ents2[0].start, ents2[0].end) == (2, 4))


def test_triggers():
    print("\n== triggers ==")
    train = [
        doc("d1", ["the", "lung", "cancer", "case"], [(1, 3, "Disease")]),
        doc("d2", ["a", "lung", "cancer", "study"], [(1, 3, "Disease")]),
        doc("d3", ["severe", "lung", "cancer", "here"], [(1, 3, "Disease")]),
    ]
    st = compute_deer_stats(train, C=2, unit_level="token")
    # 'cancer' is always an entity -> high P_entity; if not predicted, FN should fire
    units = ["new", "lung", "cancer", "found"]
    fn = detect_fn_triggers(units, [], st, theta_fn=0.9)
    check("FN fires on unpredicted high-entity token", any(units[i] == "cancer" for i in fn))
    fn2 = detect_fn_triggers(units, [(1, 3, "Disease")], st, theta_fn=0.9)
    check("FN silent when already covered", all(units[i] != "cancer" for i in fn2))


def mock_fn(messages):
    user = messages[1]["content"] if len(messages) > 1 else ""
    if "<predicted_entity>" in user:          # boundary reflection -> keep (no parseable JSON)
        return "Updated Predicted Entity (JSON format): no change"
    if "<input_text>" in user:                # unseen / fn reflection -> add nothing
        return 'Final predicted entities for the input text (JSON format):\n{"named_entities": []}'
    return '{"named_entities": [{"name": "cancer", "type": "Disease"}]}'  # generation


def test_pipeline():
    print("\n== full pipeline (mock embedder + mock llm) ==")
    train = [
        doc("d1", ["the", "lung", "cancer", "case"], [(1, 3, "Disease")]),
        doc("d2", ["a", "breast", "cancer", "study"], [(1, 3, "Disease")]),
        doc("d3", ["severe", "skin", "cancer", "here"], [(2, 3, "Disease")]),
    ]
    st = compute_deer_stats(train, C=2, unit_level="token")
    retr = LabelGuidedRetriever(train, st, embedder=MockEmbedder(), unit_level="token")
    with tempfile.TemporaryDirectory() as tmp:
        llm = LLMClient(LLMConfig(model="deepseek-v4-flash"), cache_dir=tmp, mock=True, mock_fn=mock_fn)
        test = doc("t1", ["the", "cancer", "spread"], [(1, 2, "Disease")])

        cfg1 = PipelineConfig(method="m1", entity_types=["Disease"], unit_level="token", n_shots=2)
        pipe1 = DeerPipeline(cfg1, retr, st, llm)
        spans1 = pipe1.predict(test)
        check("M1 predicts cancer span (1,2,Disease)", (1, 2, "Disease") in spans1)

        cfg2 = PipelineConfig(method="m2", entity_types=["Disease"], unit_level="token", n_shots=2)
        pipe2 = DeerPipeline(cfg2, retr, st, llm)
        spans2, trace = pipe2.predict(test, return_trace=True)
        check("M2 runs reflection and returns spans", (1, 2, "Disease") in spans2)

        res = evaluate([[(1, 2, "Disease")]], [spans2])
        check("eval F1 = 1.0 on this toy case", res.as_dict()["overall"]["f1"] == 1.0)


def main():
    test_stats()
    test_bio()
    test_triggers()
    test_pipeline()
    print(f"\nDEER PIPELINE SELF-CHECK: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
