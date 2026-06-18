"""Milestone-0 runner — DEER on NCBI-Disease (run on the researcher's machine).

Needs: DEEPSEEK_API_KEY env var (generation) + local Qwen3-Embedding-0.6B (retrieval embed
path).  The sandbox cannot reach DeepSeek nor load the model, so run this locally.

Examples:
  # 1) sanity: validate Step-0 stats against Figure 2 of the paper (no API needed)
  python scripts/run_deer.py --validate-stats

  # 2) full DEER (M2) on 1000 sampled NCBI test sentences
  DEEPSEEK_API_KEY=sk-... python scripts/run_deer.py --method m2 --split test \
      --n-test 1000 --seed 13 --embed-model /Users/xuhaoshuai/models/Qwen/Qwen3-Embedding-0.6B

Method tiers: m1 = DEER w/o ER ; m2 = full flat DEER.
NCBI hyper-params follow Table 10: lambda1=lambda2=1, C=2, w_e=w_c=1, w_o=0.01, theta_FN=0.95,
M=1, K=2, N=8 shots.  Target (DEER full, varies by LLM): 73.2-84.8; DeepSeek-V4 Flash expected
~77-85 (see notes/2026-06-14-deer-repro-spec.md §6).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deer.data.ncbi import load_ncbi, NCBI_TYPES  # noqa: E402
from deer.stats.deer_token_stats import compute_deer_stats  # noqa: E402
from deer.retrieval import LabelGuidedRetriever, MockEmbedder  # noqa: E402
from deer.orchestrator import DeerPipeline, PipelineConfig  # noqa: E402
from deer.eval import evaluate  # noqa: E402
from deer.llm import LLMClient, LLMConfig  # noqa: E402

# Figure 2 validation targets (NCBI-Disease train)
FIG2 = {"cancer": (268, 0, 5), "of": (178, 937, 4261), "ataxia": (57, 0, 1)}


def validate_stats(stats, prob_tol: float = 0.01):
    """Validate Step-0 against Figure 2.

    Judged on class *probabilities* within ``prob_tol`` (retrieval/triggers use P(), not raw
    counts), since the exact counts depend on the dataset IOB version.  A faithful口径 keeps
    every P() within ~0.01 of the paper even when the corpus differs by a few tokens.
    """
    print("\n=== Step-0 stats validation vs Figure 2 (NCBI train) ===")
    ok = True
    for tok, (e, c, o) in FIG2.items():
        N = e + c + o
        exp_p = (e / N, c / N, o / N)
        cc = stats.counts.get(tok, {})
        got = (cc.get("entity", 0), cc.get("context", 0), cc.get("other", 0))
        gn = sum(got) or 1
        got_p = (got[0] / gn, got[1] / gn, got[2] / gn)
        dmax = max(abs(a - b) for a, b in zip(exp_p, got_p))
        match = dmax <= prob_tol
        ok = ok and match
        print(f"  {tok:8s} counts exp={e}/{c}/{o} got={got[0]}/{got[1]}/{got[2]} | "
              f"P(e/c/o) exp={exp_p[0]:.3f}/{exp_p[1]:.3f}/{exp_p[2]:.3f} "
              f"got={got_p[0]:.3f}/{got_p[1]:.3f}/{got_p[2]:.3f} | dmax={dmax:.4f} "
              f"{'OK' if match else 'OFF'}")
    print(f"  => {'PASS — Step-0 faithful (P within tol; count diffs = IOB version)' if ok else 'CHECK context-window / counting口径 (spec §7)'}")
    return ok


def build_embedder(args):
    if args.embed_model:
        from deer.llm.embedder import Qwen3Embedder
        return Qwen3Embedder(args.embed_model, cache_path=str(Path(args.out_dir) / "embed_cache.json"))
    print("WARNING: no --embed-model; using MockEmbedder (NOT faithful, sanity only).")
    return MockEmbedder()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="NCBI-disease dir (default data/NCBI-disease)")
    ap.add_argument("--method", default="m2", choices=["m1", "m2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-shots", type=int, default=8)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--out-dir", default="outs/deer_ncbi")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel sentences (thread pool over API calls); 1 = serial")
    ap.add_argument("--validate-stats", action="store_true", help="only validate Step-0 and exit")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else None
    train = load_ncbi("train", data_dir) if data_dir else load_ncbi("train")
    stats = compute_deer_stats(train, C=2, unit_level="token")

    ok = validate_stats(stats)
    if args.validate_stats:
        sys.exit(0 if ok else 1)

    test = load_ncbi(args.split, data_dir) if data_dir else load_ncbi(args.split)
    if args.n_test and len(test) > args.n_test:
        rng = random.Random(args.seed)
        test = rng.sample(test, args.n_test)
    print(f"\ntrain={len(train)}  test(sampled)={len(test)}  method={args.method}")

    embedder = build_embedder(args)
    retr = LabelGuidedRetriever(
        train, stats, w_e=1.0, w_c=1.0, w_o=0.01, lambda1=1.0, lambda2=1.0,
        embedder=embedder, unit_level="token",
    )
    llm = LLMClient(LLMConfig(model="deepseek-v4-flash", pool_size=max(32, args.workers)),
                    cache_dir=str(Path(args.out_dir) / "llm_cache"))
    cfg = PipelineConfig(
        method=args.method, entity_types=NCBI_TYPES, unit_level="token",
        n_shots=args.n_shots, C=2, theta_fn=0.95, K=2, M=1,
        domain="biomedical abstract",
    )
    pipe = DeerPipeline(cfg, retr, stats, llm)

    golds = [[(e.start, e.end, e.type) for e in d.entities] for d in test]
    gold_sets = [set(g) for g in golds]
    preds = [None] * len(test)
    traces = [None] * len(test)

    def work(d):
        return pipe.predict(d, return_trace=True)  # (spans, trace_entries)

    import time as _time
    t0 = _time.time()
    failures = []  # (index, error) for sentences that failed even after retries
    if args.workers > 1:
        retr.prewarm(test)  # encode test tokens single-threaded before concurrent loop
        print(f"  running {len(test)} sentences with {args.workers} workers ...")
        done = 0
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, d): i for i, d in enumerate(test)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    preds[i], traces[i] = fut.result()
                except Exception as e:  # noqa: BLE001 — one bad sentence must not kill the batch
                    failures.append((i, repr(e)))
                with lock:
                    done += 1
                    if done % 50 == 0:
                        print(f"  ...{done}/{len(test)}")
    else:
        for k, d in enumerate(test):
            try:
                preds[k], traces[k] = work(d)
            except Exception as e:  # noqa: BLE001
                failures.append((k, repr(e)))
            if (k + 1) % 50 == 0:
                print(f"  ...{k+1}/{len(test)}")

    if failures:
        print(f"\n  WARNING: {len(failures)}/{len(test)} sentences failed after retries "
              f"(e.g. {failures[0][1][:120]}).")
        print("  Completed sentences are cached — just re-run the SAME command to fill the gaps "
              "(cache resumes instantly). Skipping eval until all sentences succeed.")
        sys.exit(2)

    # ---- reflection diagnostics (which pass helps/hurts) ----
    refl = {}
    if args.method != "m1":
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(int))
        wrong_add_examples = []  # sample of bad additions for inspection
        for i, tr in enumerate(traces):
            if not tr:
                continue
            gs = gold_sets[i]
            units = test[i].units
            for ev in tr:
                p, a = ev["pass"], ev["action"]
                if a == "add":
                    span = tuple(ev["span"])
                    correct = span in gs
                    key = "add_correct" if correct else "add_wrong"
                    agg[p][key] += 1
                    if not correct and len(wrong_add_examples) < 25:
                        added = " ".join(units[span[0]:span[1]])
                        golds_here = sorted(gs)
                        # is the added span a sub/super-string of a gold span? (boundary error)
                        overlap = [g for g in golds_here if not (g[1] <= span[0] or g[0] >= span[1])]
                        wrong_add_examples.append({
                            "pass": p, "did": test[i].did,
                            "trigger_token": ev.get("token"),
                            "added": f"{added} [{span[0]}:{span[1]}]",
                            "overlapping_gold": [f"{' '.join(units[g[0]:g[1]])} [{g[0]}:{g[1]}]" for g in overlap],
                            "boundary_error": bool(overlap),  # True = overlaps a gold but wrong boundary
                        })
                elif a == "trim":
                    old_in = tuple(ev["old"]) in gs
                    new_in = tuple(ev["new"]) in gs
                    if new_in and not old_in:
                        agg[p]["trim_fixed"] += 1       # bad->good (good)
                    elif old_in and not new_in:
                        agg[p]["trim_broke"] += 1       # good->bad (bad)
                    else:
                        agg[p]["trim_neutral"] += 1
                elif a == "delete":
                    agg[p]["delete_good_entity" if tuple(ev["span"]) in gs else "delete_fp"] += 1
        refl = {p: dict(d) for p, d in agg.items()}
        print("\n  reflection edits (vs gold):")
        for p, d in refl.items():
            print(f"    {p}: {dict(d)}")
        n_be = sum(1 for e in wrong_add_examples if e["boundary_error"])
        print(f"\n  wrong-add examples ({n_be}/{len(wrong_add_examples)} are boundary errors "
              f"= overlap a gold span but wrong edge):")
        for e in wrong_add_examples[:12]:
            tag = "BOUNDARY" if e["boundary_error"] else "SPURIOUS"
            print(f"    [{e['pass']}/{tag}] token='{e['trigger_token']}' added='{e['added']}' "
                  f"gold_overlap={e['overlapping_gold']}")
        refl["_wrong_add_examples"] = wrong_add_examples

    elapsed = _time.time() - t0
    lstats = llm.stats()
    print(f"\n  wall={elapsed:.1f}s  live_calls={lstats['live_calls']}  cache_hits={lstats['cache_hits']}  "
          f"throughput={(lstats['live_calls']/elapsed if elapsed else 0):.1f} calls/s")

    res = evaluate(golds, preds).as_dict()
    report = {
        "config": vars(args),
        "hyperparams": {"lambda1": 1.0, "lambda2": 1.0, "C": 2, "w_e": 1.0, "w_c": 1.0,
                        "w_o": 0.01, "theta_fn": 0.95, "M": 1, "K": 2, "n_shots": args.n_shots},
        "n_train": len(train), "n_test": len(test),
        "llm_stats": llm.stats(),
        "reflection_edits": refl,
        "result": res,
    }
    out = Path(args.out_dir) / f"ncbi_{args.method}_{args.split}_n{len(test)}_seed{args.seed}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== RESULT ({args.method}) ===")
    print(f"  strict micro  P={res['overall']['precision']}  R={res['overall']['recall']}  F1={res['overall']['f1']}")
    print(f"  saved -> {out}")
    print("  fidelity check: M1(w/oER) should beat KATE by ~1-5; M2 adds ~1-2 over M1; "
          "absolute ~77-85 expected for Flash (spec §6).")


if __name__ == "__main__":
    main()
