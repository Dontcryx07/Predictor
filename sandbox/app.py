"""Streamlit sandbox for the Redrob candidate ranker.

Satisfies the submission spec's Section 10.5 sandbox requirement: accepts a
small candidate sample (<=100), runs the *same* ranking pipeline used for the
full submission, and returns a ranked CSV — end-to-end, on CPU, in seconds.

Run locally:
    streamlit run sandbox/app.py

Deploy free on Streamlit Community Cloud pointing at this file.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank import build_rows, write_csv  # noqa: E402
from src import parse, score  # noqa: E402


st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Intelligent Candidate Ranker — Sandbox")
st.caption(
    "Upload a small candidate sample (JSONL or a JSON array, <=100 records). "
    "The exact production pipeline scores it on CPU in seconds and returns a "
    "ranked CSV. Evidence is read from career descriptions — never the skills "
    "list — honeypots are excluded, and each row carries fact-grounded reasoning."
)

uploaded = st.file_uploader(
    "Candidate sample (.jsonl or .json)", type=["jsonl", "json"]
)
use_bundled = st.checkbox(
    "Use bundled sample (dataset/sample_candidates.json) instead", value=False
)


def _load_uploaded(file) -> list[dict]:
    raw = file.read().decode("utf-8")
    raw = raw.strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


candidates: list[dict] = []
if use_bundled:
    p = Path(__file__).resolve().parent.parent / "dataset" / "sample_candidates.json"
    if p.exists():
        candidates = json.loads(p.read_text(encoding="utf-8"))
    else:
        st.warning("Bundled sample not found in this deployment.")
elif uploaded is not None:
    candidates = _load_uploaded(uploaded)

if candidates:
    top_n = st.slider("How many to rank", 1, min(100, len(candidates)),
                      min(20, len(candidates)))
    anchor = parse.compute_time_anchor(candidates)
    scored = score.rank(candidates, anchor, top_n=top_n, exclude_honeypots=True)
    rows = build_rows(scored)

    st.subheader(f"Top {len(rows)} (anchor date {anchor.date})")
    st.dataframe(rows, use_container_width=True)

    buf = io.StringIO()
    import csv as _csv
    writer = _csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="submission_sample.csv", mime="text/csv")

    excluded = sum(1 for c in candidates
                   if score.score_candidate(c, anchor).trap.is_honeypot)
    st.info(f"Honeypots detected and excluded from the ranking: {excluded}")
else:
    st.write("Awaiting a candidate sample.")
