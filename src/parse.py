"""Stage 0 — parsing, normalization, and the dataset time anchor.

Responsibilities:
    * Robustly load candidates from a plain or gzipped JSONL file.
    * Provide safe accessors for the nested profile structure.
    * Derive a *deterministic* "today" from the data itself (the maximum
      ``last_active_date`` in the pool) so that any time-based feature is
      reproducible across machines and runs — never wall-clock dependent.

No scoring logic lives here; this module only turns bytes into clean,
typed-enough Python structures for the rest of the pipeline.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def parse_iso_date(value: Optional[str]) -> Optional[tuple[int, int, int]]:
    """Parse a ``YYYY-MM-DD`` string into a ``(year, month, day)`` tuple.

    Returns ``None`` for missing/malformed dates rather than raising, because
    the dataset is synthetic and we must degrade gracefully on bad input.
    """
    if not value or not isinstance(value, str):
        return None
    parts = value.strip()[:10].split("-")
    if len(parts) != 3:
        return None
    try:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    return (year, month, day)


def month_index(value: Optional[str]) -> Optional[int]:
    """Convert a date string to an absolute month index (year * 12 + month).

    Handy for cheap month-granularity duration/recency math without pulling in
    ``datetime`` and its timezone/locale surface area.
    """
    parsed = parse_iso_date(value)
    if parsed is None:
        return None
    year, month, _day = parsed
    return year * 12 + month


def months_between(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    """Whole months from ``earlier`` to ``later`` (may be negative)."""
    a = month_index(earlier)
    b = month_index(later)
    if a is None or b is None:
        return None
    return b - a


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _open_maybe_gzip(path: Path):
    """Open ``path`` as text, transparently handling gzip by extension/magic."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    # Sniff the gzip magic bytes so a mislabeled .jsonl still works.
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate records from a (optionally gzipped) JSONL file.

    Malformed lines are skipped rather than aborting the whole run; this keeps
    the ranker robust to a single bad record in a 100k-line file.
    """
    path = Path(path)
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Load all candidates into a list. ~100k records fit comfortably in RAM."""
    return list(iter_candidates(path))


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------
def profile(candidate: dict) -> dict:
    return candidate.get("profile") or {}


def signals(candidate: dict) -> dict:
    return candidate.get("redrob_signals") or {}


def career_history(candidate: dict) -> list[dict]:
    history = candidate.get("career_history") or []
    return [role for role in history if isinstance(role, dict)]


def skills(candidate: dict) -> list[dict]:
    items = candidate.get("skills") or []
    return [s for s in items if isinstance(s, dict)]


def education(candidate: dict) -> list[dict]:
    items = candidate.get("education") or []
    return [e for e in items if isinstance(e, dict)]


def candidate_id(candidate: dict) -> str:
    return str(candidate.get("candidate_id", "")).strip()


def evidence_text_blocks(candidate: dict) -> list[str]:
    """Ordered text blocks that describe *what the candidate actually did*.

    Deliberately excludes the raw ``skills[]`` list — per the JD and our EDA,
    the skills list is the adversarial keyword-stuffing surface. Career
    descriptions, headline and summary are the honest signal of real work.
    """
    prof = profile(candidate)
    blocks: list[str] = []
    headline = prof.get("headline")
    if isinstance(headline, str):
        blocks.append(headline)
    summary = prof.get("summary")
    if isinstance(summary, str):
        blocks.append(summary)
    for role in career_history(candidate):
        desc = role.get("description")
        if isinstance(desc, str):
            blocks.append(desc)
    return blocks


# ---------------------------------------------------------------------------
# Dataset time anchor
# ---------------------------------------------------------------------------
DEFAULT_ANCHOR = "2026-05-27"  # observed max last_active_date during EDA


@dataclass(frozen=True)
class Anchor:
    """A deterministic 'today' derived from the data."""

    date: str
    month: int


def compute_time_anchor(
    candidates: Iterable[dict], fallback: str = DEFAULT_ANCHOR
) -> Anchor:
    """Return the latest ``last_active_date`` across the pool as the anchor.

    Using the data's own maximum activity date (rather than the wall clock)
    makes recency features fully reproducible — a Stage-3 requirement.
    """
    latest: Optional[str] = None
    for candidate in candidates:
        active = signals(candidate).get("last_active_date")
        if isinstance(active, str) and parse_iso_date(active) is not None:
            if latest is None or active > latest:
                latest = active
    chosen = latest or fallback
    return Anchor(date=chosen, month=month_index(chosen) or (month_index(fallback) or 0))
