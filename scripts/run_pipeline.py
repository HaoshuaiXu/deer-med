"""Dataset-general flat-DEER runner — ncbi | cmeee | genia, method tiers m1/m2.

The pipeline stages are dataset-agnostic; everything that differs per dataset (loader, entity
types, unit level, domain phrase, prompt language, hyper-params) lives in
``deer.data.datasets``.  NCBI's frozen runner is ``run_deer.py``; this one generalizes it to
CMeEE / GENIA.

Note on nested datasets: CMeEE and GENIA contain nested entities, but DEER is a *flat* method —
it predicts a non-overlapping span set, so nested gold entities it cannot recover show up as the
``nested-gold recall`` gap in the result block. That gap is the expected behaviour of a flat
baseline on nested data, not a bug.

Examples (run on the researcher's machine — needs DEEPSEEK_API_KEY + local Qwen3 embedder):
  # offline smoke (no API/embedder): build prompts + align on a few dev sentences
  python scripts/run_pipeline.py --dataset cmeee --selfcheck

  # M2 (full flat DEER) on 300 CMeEE dev sentences
  DEEPSEEK_API_KEY=sk-... python scripts/run_pipeline.py --dataset cmeee --method m2 \
      --split dev --n-test 300 --seed 42 --workers 8 \
      --embed-model /path/to/Qwen3-Embedding-0.6B

  # same for GENIA
  ... --dataset genia --method m2 --split dev --n-test 300 --seed 42 ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deer.data.datasets import get_dataset  # noqa: E402
from deer.stats.deer_token_stats import compute_deer_stats  # noqa: E402
from deer.retrieval import LabelGuidedRetriever, MockEmbedder  # noqa: E402
from deer.orchestrator import DeerPipeline, PipelineConfig  # noqa: E402
from deer.eval import evaluate  # noqa: E402
from deer.llm import LLMClient, LLMConfig  # noqa: E402

# method tier -> flat reflection pass list (override with --passes)
METHOD_PASSES = {
    "m1": (),
    "m2": ("unseen", "fn", "boundary"),       # full flat DEER
}
FLAT_PASSES = {"unseen", "fn", "boundary"}


def build_embedder(args):
    if args.embed_model:
        from deer.llm.embedder import Qwen3Embedder
        return Qwen3Embedder(args.embed_model, cache_path=str(Path(args.out_dir) / "embed_cache.json"))
    print("WARNING: no --embed-model; using MockEmbedder (NOT faithful, sanity only).")
    return MockEmbedder()


def selfcheck(spec, n: int = 3):
    """Offline: load train+dev, compute stats, build a generation prompt + dry-align gold.

    No API / no embedder — proves the dataset flows through load -> stats -> prompt -> align
    -> eval wiring with consistent types/units before spending any API budget.
    """
    from deer.generate.generator import build_generation_prompt
    from deer.align.aligner import align_predictions

    train = spec.load("train")
    dev = spec.load("dev")
    stats = compute_deer_stats(train, C=int(spec.hp["C"]), unit_level=spec.unit_level)
    print(f"[{spec.name}] train={len(train)} dev={len(dev)} unit={spec.unit_level} "
          f"types={list(spec.entity_types)} lang={spec.prompt_lang}")
    print(f"  stats: {len(stats.counts)} unit-types; domain={spec.domain!r}")

    for d in dev[:n]:
        gold = [(e.surface, e.type) for e in d.entities]
        # round-trip: feed gold (name,type) through the deterministic aligner; spans must match
        aligned = align_predictions(d.units, gold, spec.unit_level)
        rec = {tuple(a.as_tuple()) for a in aligned if a.status == "aligned"}
        gld = {e.as_tuple() for e in d.entities}
        ok = gld.issubset(rec)
        print(f"  [{d.did}] len={len(d.units)} gold_ents={len(gold)} align_recovers_gold={ok}")
    # one full generation prompt (2 demos) to eyeball the scaffold
    demos = [(dev[i].text, [(e.surface, e.type) for e in dev[i].entities]) for i in range(1, 3)]
    msg = build_generation_prompt(dev[0].text, demos, spec.entity_types, lang=spec.prompt_lang)
    print("\n--- sample generation prompt (truncated) ---")
    print(msg[0]["content"][:600])
    print("...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ncbi", "cmeee", "genia"])
    ap.add_argument("--method", default="m1", choices=["m1", "m2"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slice", default=None,
                    help="contiguous reproducible region 'START:END' over the loaded split "
                         "(half-open). Takes the full slice and SKIPS random --n-test sampling.")
    ap.add_argument("--n-shots", type=int, default=None, help="override hp n_shots")
    ap.add_argument("--theta-fn", type=float, default=None,
                    help="override FN-pass entity-likelihood threshold (default hp 0.95)")
    ap.add_argument("--surround-thresh", type=float, default=None,
                    help="override Unseen-pass surround threshold (default 0.5)")
    ap.add_argument("--passes", default=None,
                    help="comma list of flat reflection passes (default per --method); "
                         "flat: unseen,fn,boundary ; e.g. 'unseen,fn'")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--diag", action="store_true",
                    help="aggregate per-pass reflection edits vs gold (attribution)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="offline wiring check (load/stats/prompt/align), no API; then exit")
    args = ap.parse_args()

    spec = get_dataset(args.dataset)
    hp = dict(spec.hp)
    n_shots = args.n_shots or int(hp["n_shots"])
    if args.out_dir is None:
        args.out_dir = f"outs/pipeline_{spec.name}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.selfcheck:
        selfcheck(spec)
        sys.exit(0)

    train = spec.load("train")
    stats = compute_deer_stats(train, C=int(hp["C"]), unit_level=spec.unit_level)

    test = spec.load(args.split)
    if args.slice is not None:
        s, e = (int(x) for x in args.slice.split(":"))
        test = test[s:e]
        sel = f"{args.split}[{s}:{e}]={len(test)}"
    else:
        if args.n_test and len(test) > args.n_test:
            test = random.Random(args.seed).sample(test, args.n_test)
        sel = f"{args.split}(sampled)={len(test)}"
    print(f"[{spec.name}] train={len(train)} {sel} "
          f"method={args.method} unit={spec.unit_level} lang={spec.prompt_lang}")

    embedder = build_embedder(args)
    retr = LabelGuidedRetriever(
        train, stats, w_e=hp["w_e"], w_c=hp["w_c"], w_o=hp["w_o"],
        lambda1=hp["lambda1"], lambda2=hp["lambda2"],
        embedder=embedder, unit_level=spec.unit_level,
    )
    llm = LLMClient(LLMConfig(model="deepseek-v4-flash", pool_size=max(32, args.workers)),
                    cache_dir=str(Path(args.out_dir) / "llm_cache"))
    theta_fn = args.theta_fn if args.theta_fn is not None else hp["theta_fn"]
    passes = (tuple(p.strip() for p in args.passes.split(",") if p.strip())
              if args.passes is not None else METHOD_PASSES[args.method])
    bad = [p for p in passes if p not in FLAT_PASSES]
    if bad:
        ap.error(f"this is a flat-DEER runner; unsupported pass(es): {bad} "
                 f"(allowed: {sorted(FLAT_PASSES)})")
    cfg = PipelineConfig(
        method=args.method, entity_types=spec.entity_types, unit_level=spec.unit_level,
        domain=spec.domain, prompt_lang=spec.prompt_lang, n_shots=n_shots,
        C=int(hp["C"]), theta_fn=theta_fn, K=int(hp["K"]), M=int(hp["M"]),
        reflect_passes=passes,
    )
    if args.surround_thresh is not None:
        cfg.surround_thresh = args.surround_thresh
    print(f"  reflect_passes={cfg.reflect_passes} theta_fn={cfg.theta_fn} "
          f"surround_thresh={cfg.surround_thresh}")

    pipe = DeerPipeline(cfg, retr, stats, llm)

    golds = [[e.as_tuple() for e in d.entities] for d in test]
    gold_sets = [set(g) for g in golds]
    preds = [None] * len(test)
    traces = [None] * len(test)
    want_trace = args.diag and args.method != "m1"

    def work(d):
        if want_trace:
            return pipe.predict(d, return_trace=True)  # (spans, trace_entries)
        return pipe.predict(d), None

    t0 = _time.time()
    failures = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        retr.prewarm(test)
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
        print("  Completed sentences are cached — re-run the SAME command to fill gaps. "
              "Skipping eval until all succeed.")
        sys.exit(2)

    elapsed = _time.time() - t0
    lstats = llm.stats()
    print(f"\n  wall={elapsed:.1f}s  live_calls={lstats['live_calls']}  "
          f"cache_hits={lstats['cache_hits']}  "
          f"throughput={(lstats['live_calls']/elapsed if elapsed else 0):.1f} calls/s")

    # ---- per-pass reflection attribution (which pass helps/hurts vs gold) ----
    refl = {}
    if want_trace and any(traces):
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(int))
        for i, tr in enumerate(traces):
            if not tr:
                continue
            gs = gold_sets[i]
            for ev in tr:
                p, a = ev["pass"], ev["action"]
                if a == "add":
                    agg[p]["add_correct" if tuple(ev["span"]) in gs else "add_wrong"] += 1
                elif a == "trim":
                    old_in, new_in = tuple(ev["old"]) in gs, tuple(ev["new"]) in gs
                    agg[p]["trim_fixed" if (new_in and not old_in) else
                           "trim_broke" if (old_in and not new_in) else "trim_neutral"] += 1
                elif a == "delete":
                    agg[p]["delete_good" if tuple(ev["span"]) in gs else "delete_fp"] += 1
        refl = {p: dict(d) for p, d in agg.items()}
        print("\n  reflection edits vs gold (add_wrong / trim_broke / delete_good = precision-hurting):")
        for p, d in refl.items():
            print(f"    {p}: {dict(d)}")

    res = evaluate(golds, preds).as_dict()
    report = {
        "config": vars(args), "dataset": spec.name, "hyperparams": hp,
        "theta_fn": cfg.theta_fn, "surround_thresh": cfg.surround_thresh,
        "n_train": len(train), "n_test": len(test),
        "llm_stats": llm.stats(), "reflection_edits": refl, "result": res,
    }
    suffix = ""
    if args.slice is not None:
        suffix += "_sl" + args.slice.replace(":", "-")
    if args.theta_fn is not None:
        suffix += f"_tfn{cfg.theta_fn}"
    if args.surround_thresh is not None:
        suffix += f"_sur{cfg.surround_thresh}"
    if args.passes is not None:
        suffix += "_" + "-".join(cfg.reflect_passes)
    out = Path(args.out_dir) / f"{spec.name}_{args.method}_{args.split}_n{len(test)}_seed{args.seed}{suffix}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # per-doc gold/pred dump for offline paired-bootstrap significance (scripts/significance.py).
    perdoc = {
        "dataset": spec.name, "method": args.method, "split": args.split,
        "passes": list(cfg.reflect_passes), "n_test": len(test),
        "docs": [
            {"did": test[i].did,
             "gold": [list(g) for g in golds[i]],
             "pred": [list(p) for p in (preds[i] or [])]}
            for i in range(len(test))
        ],
    }
    perdoc_out = out.with_name(out.stem + "_perdoc.json")
    perdoc_out.write_text(json.dumps(perdoc, ensure_ascii=False), encoding="utf-8")
    ov = res["overall"]
    print(f"\n=== RESULT ({spec.name}/{args.method}  passes={cfg.reflect_passes}) ===")
    print(f"  strict micro  P={ov['precision']}  R={ov['recall']}  F1={ov['f1']}")
    ng, fg = res["nested_gold"], res["flat_gold"]
    print(f"  nested-gold recall = {ng['recall']} ({ng['recovered']}/{ng['total']})   "
          f"flat-gold recall = {fg['recall']} ({fg['recovered']}/{fg['total']})")
    print(f"  E1 inner-miss={res['E1_inner_miss_rate']}  E2 outer-miss={res['E2_outer_miss_rate']}  "
          f"E3 pair-both={res['E3_pair_both_recovered_rate']}")
    print(f"  saved -> {out}")
    print(f"  per-doc dump -> {perdoc_out}")


if __name__ == "__main__":
    main()
