"""Extract the distinct career-history description templates from the pool.

The dataset was found (during EDA) to be composed of a small, fixed set of
canonical `career_history[].description` strings. This tool enumerates them,
counts their frequency, and writes a stable, index-ordered JSON artifact used
to build and validate the evidence audit in `src/templates_audit.py`.

Usage:
    python tools/extract_templates.py --candidates ./dataset/candidates.jsonl \
        --out ./dataset/_templates_extracted.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def extract(candidates_path: Path) -> list[dict]:
    counts: collections.Counter[str] = collections.Counter()
    with open(candidates_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for role in record.get("career_history", []):
                desc = role.get("description")
                if desc:
                    counts[desc] += 1
    # Sort by descending frequency, then text, for a deterministic index order.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"id": idx, "count": count, "text": text}
        for idx, (text, count) in enumerate(ordered)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    templates = extract(args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(templates, fh, ensure_ascii=False, indent=2)

    print(f"Extracted {len(templates)} distinct templates -> {args.out}")
    for tmpl in templates:
        preview = tmpl["text"][:60].replace("\n", " ")
        print(f"  T{tmpl['id']:02d}  {tmpl['count']:>6}x  {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
