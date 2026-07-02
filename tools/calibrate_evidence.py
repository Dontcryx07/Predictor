"""Side-by-side comparison of the scorer's output vs the hand-graded audit.

Used this constantly while tuning the ontology weights in evidence.py --
change a weight, rerun this, see which of the 44 templates moved tier.
Exits non-zero on any mismatch so it can also run as a quick check.

Usage:
    python -m tools.calibrate_evidence --templates ./dataset/career_description_templates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import evidence
from src.templates_audit import AUDIT, grade_to_tier, match_description


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", default="./dataset/career_description_templates.json", type=Path)
    args = parser.parse_args(argv)

    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    by_prefix = {e.template_id: e for e in AUDIT}

    mismatches = 0
    print(f"{'T':>3} {'aud_tier':>8} {'scr_tier':>8} {'aud_grd':>8} {'scr_grd':>8}  families")
    print("-" * 88)
    for tmpl in sorted(templates, key=lambda t: t["id"]):
        text = tmpl["text"]
        entry = match_description(text)
        audit_entry = by_prefix.get(tmpl["id"])
        result = evidence.score_text(text)
        scored_tier = grade_to_tier(result.grade)
        ok = audit_entry is not None and scored_tier == audit_entry.tier
        flag = "" if ok else "  <-- MISMATCH"
        if not ok:
            mismatches += 1
        aud_tier = audit_entry.tier if audit_entry else -1
        aud_grade = audit_entry.grade if audit_entry else -1
        fam = ",".join(result.families)
        extra = []
        if result.nontech_marker:
            extra.append(f"NONTECH:{result.nontech_marker}")
        if result.cv_primary:
            extra.append("CV")
        if result.disclaimer:
            extra.append("DISC")
        print(f"T{tmpl['id']:02d} {aud_tier:>8} {scored_tier:>8} {aud_grade:>8.2f} "
              f"{result.grade:>8.3f}  {fam} {' '.join(extra)}{flag}")

    print("-" * 88)
    print(f"mismatches: {mismatches} / {len(templates)}")
    # Also verify prefix matching covers all 44.
    unmatched = [t["id"] for t in templates if match_description(t["text"]) is None]
    if unmatched:
        print(f"PREFIX MATCH FAILURES for template ids: {unmatched}")
        mismatches += len(unmatched)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
