"""Run the ranker against the local proxy ground truth and report the composite,
plus ablations and a small sensitivity sweep. Diagnostics only.

    python -m tools.evaluate_run --candidates ./dataset/candidates.jsonl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import behavior, evaluate, jdfit, parse, score


def _ranked_ids(candidates, anchor, **kw):
    scored = score.rank(candidates, anchor, top_n=100, **kw)
    return [s.candidate_id for s in scored]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./dataset/candidates.jsonl", type=Path)
    args = ap.parse_args(argv)

    t0 = time.time()
    cands = parse.load_candidates(args.candidates)
    anchor = parse.compute_time_anchor(cands)
    gt = evaluate.build_proxy_ground_truth(cands, anchor)
    rel = sum(1 for t in gt.values() if t >= evaluate.RELEVANT_TIER)
    t5 = sum(1 for t in gt.values() if t == 5)
    print(f"loaded {len(cands)} in {time.time()-t0:.1f}s | proxy relevant(>=3): {rel} | tier5: {t5}")

    print("\n== Full model ==")
    m = evaluate.evaluate(_ranked_ids(cands, anchor), gt)
    print(f"  NDCG@10={m.ndcg10} NDCG@50={m.ndcg50} MAP={m.map} P@10={m.p10} "
          f"P@5={m.p5} => composite={m.composite}")

    # Ablation: turn off honeypot exclusion (should drop composite / add tier-0s).
    print("\n== Ablation: honeypots NOT excluded ==")
    m_hp = evaluate.evaluate(_ranked_ids(cands, anchor, exclude_honeypots=False), gt)
    print(f"  NDCG@10={m_hp.ndcg10} NDCG@50={m_hp.ndcg50} MAP={m_hp.map} "
          f"P@10={m_hp.p10} => composite={m_hp.composite}")

    # Sensitivity: vary jdfit ALPHA and behavior bounds, restore afterward.
    print("\n== Sensitivity: jdfit ALPHA ==")
    orig_alpha = jdfit.ALPHA
    for alpha in (0.0, 0.15, 0.30, 0.45):
        jdfit.ALPHA = alpha
        m_a = evaluate.evaluate(_ranked_ids(cands, anchor), gt)
        print(f"  ALPHA={alpha:>4}: composite={m_a.composite} "
              f"(NDCG@10={m_a.ndcg10}, NDCG@50={m_a.ndcg50})")
    jdfit.ALPHA = orig_alpha

    print("\n== Sensitivity: behavior MAX_MULT ==")
    orig_max = behavior.MAX_MULT
    for mx in (1.0, 1.10, 1.25):
        behavior.MAX_MULT = mx
        m_b = evaluate.evaluate(_ranked_ids(cands, anchor), gt)
        print(f"  MAX_MULT={mx:>4}: composite={m_b.composite} "
              f"(NDCG@10={m_b.ndcg10})")
    behavior.MAX_MULT = orig_max
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
