#!/usr/bin/env python3
"""Redrob Intelligent Candidate Ranking — single-command entrypoint.

Produces the top-100 ranked submission CSV from the candidate pool. Runs on
CPU only, with no network access and no precomputed artifacts, in well under
the 5-minute budget.

    python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv

Output columns (spec Section 2): candidate_id, rank, score, reasoning.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from src import parse, reasoning, score


def build_rows(scored: list[score.ScoredCandidate]) -> list[dict]:
    rows = []
    for rank_pos, sc in enumerate(scored, start=1):
        rows.append({
            "candidate_id": sc.candidate_id,
            "rank": rank_pos,
            "score": f"{sc.composite:.6f}",
            "reasoning": reasoning.generate(sc, rank_pos),
        })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path,
                        help="Path to candidates.jsonl (plain or gzipped).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Path to write the submission CSV.")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of candidates to rank (default 100).")
    args = parser.parse_args(argv)

    t0 = time.time()
    candidates = parse.load_candidates(args.candidates)
    anchor = parse.compute_time_anchor(candidates)
    t_load = time.time() - t0

    t1 = time.time()
    scored = score.rank(candidates, anchor, top_n=args.top, exclude_honeypots=True)
    rows = build_rows(scored)
    write_csv(rows, args.out)
    t_rank = time.time() - t1

    # Reproducibility / sanity summary (no PII, structured).
    honeypots_in_top = sum(1 for s in scored if s.trap.is_honeypot)
    tiers = [s.tier for s in scored]
    print(f"[rank] loaded {len(candidates)} candidates (anchor {anchor.date}) "
          f"in {t_load:.1f}s")
    print(f"[rank] ranked top-{len(scored)} in {t_rank:.1f}s "
          f"(total {time.time() - t0:.1f}s)")
    print(f"[rank] honeypots in output: {honeypots_in_top} (must be 0)")
    if scored:
        print(f"[rank] top score {scored[0].composite:.4f} -> "
              f"bottom score {scored[-1].composite:.4f}")
        print(f"[rank] evidence tiers in top-{len(scored)}: "
              f"min={min(tiers)} max={max(tiers)}")
    print(f"[rank] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
