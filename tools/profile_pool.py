"""Scratch script for sanity-checking evidence/trap output on the full pool.

Prints grade distribution, timing, and a handful of honeypot + clean examples
so I can eyeball whether the numbers look right before trusting rank.py.

    python -m tools.profile_pool --candidates ./dataset/candidates.jsonl
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

from src import evidence, parse, traps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./dataset/candidates.jsonl", type=Path)
    args = ap.parse_args(argv)

    t0 = time.time()
    cands = parse.load_candidates(args.candidates)
    t_load = time.time() - t0

    anchor = parse.compute_time_anchor(cands)

    t1 = time.time()
    grade_hist = collections.Counter()
    honeypots = 0
    honeypots_strong = 0
    stuffers = 0
    strong = 0  # evidence >= 0.78 (tier 5)
    tier4plus = 0  # >= 0.58
    honeypot_examples = []
    strong_clean = []

    for c in cands:
        ev = evidence.score_candidate(c, anchor)
        tr = traps.detect(c, anchor, ev.grade)
        bucket = round(ev.grade, 1)
        grade_hist[bucket] += 1
        is_strong = ev.grade >= 0.78
        if is_strong:
            strong += 1
        if ev.grade >= 0.58:
            tier4plus += 1
        if tr.is_honeypot:
            honeypots += 1
            if is_strong and len(honeypot_examples) < 12:
                honeypot_examples.append(
                    (parse.candidate_id(c), round(ev.grade, 3), tr.honeypot_reasons)
                )
            if is_strong:
                honeypots_strong += 1
        if tr.is_keyword_stuffer:
            stuffers += 1
        if is_strong and not tr.is_honeypot and len(strong_clean) < 8:
            p = parse.profile(c)
            strong_clean.append((parse.candidate_id(c), round(ev.grade, 3),
                                 p.get("current_title"), p.get("location")))

    t_score = time.time() - t1

    print(f"loaded {len(cands)} candidates in {t_load:.1f}s; scored in {t_score:.1f}s")
    print(f"time anchor: {anchor.date}")
    print(f"strong (evidence>=0.78, ~tier5): {strong}")
    print(f"tier4+ (evidence>=0.58): {tier4plus}")
    print(f"honeypots flagged: {honeypots} (of which strong-pool: {honeypots_strong})")
    print(f"keyword stuffers flagged: {stuffers}")
    print("\nevidence grade histogram (rounded to 0.1):")
    for bucket in sorted(grade_hist):
        print(f"  {bucket:>4}: {grade_hist[bucket]}")
    print("\nstrong-pool honeypot examples (excluded from top-100):")
    for cid, g, reasons in honeypot_examples:
        print(f"  {cid} grade={g}: {reasons}")
    print("\nclean strong-pool examples (eligible):")
    for cid, g, title, loc in strong_clean:
        print(f"  {cid} grade={g} | {title} | {loc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
