"""A rough local stand-in for the hidden leaderboard.

There's no public leaderboard and no feedback during the competition, so
there's no honest way to know the real score before submitting. What we can
do is build our own "ground truth" from the template audit (best tier across
a candidate's roles, forced to 0 for honeypots) and compute the same metrics
the spec says they'll use:

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

Worth being honest about the limits here: this proxy ground truth comes from
the same audit that drives the scorer, so getting a composite of 1.0 against
it mostly just proves there are no bugs in the tier logic -- it says nothing
about whether the ordering *within* the strong candidates matches what the
real judges think. Used this for sanity checks and weight sweeps, not as
proof of a good score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import parse, traps
from .templates_audit import match_description


RELEVANT_TIER = 3  # tier >= 3 counts as "relevant" for P@k and MAP


def proxy_tier(candidate: dict, anchor: parse.Anchor) -> int:
    """Best audited tier across the candidate's roles; 0 if honeypot/unknown."""
    best = 0
    for role in parse.career_history(candidate):
        entry = match_description(role.get("description", ""))
        if entry is not None:
            best = max(best, entry.tier)
    # Honeypots are forced to tier 0 in the real ground truth (spec S7).
    tr = traps.detect(candidate, anchor, evidence_grade=1.0 if best >= 4 else 0.0)
    if tr.is_honeypot:
        return 0
    return best


def build_proxy_ground_truth(candidates: list[dict], anchor: parse.Anchor) -> dict[str, int]:
    return {parse.candidate_id(c): proxy_tier(c, anchor) for c in candidates}


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], gt: dict[str, int], k: int) -> float:
    gains = [float(gt.get(cid, 0)) for cid in ranked_ids[:k]]
    dcg = _dcg(gains)
    ideal = sorted(gt.values(), reverse=True)[:k]
    idcg = _dcg([float(g) for g in ideal])
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], gt: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if gt.get(cid, 0) >= RELEVANT_TIER)
    return hits / k


def mean_average_precision(ranked_ids: list[str], gt: dict[str, int]) -> float:
    total_relevant = sum(1 for t in gt.values() if t >= RELEVANT_TIER)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if gt.get(cid, 0) >= RELEVANT_TIER:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(total_relevant, len(ranked_ids))


@dataclass
class Metrics:
    ndcg10: float
    ndcg50: float
    map: float
    p10: float
    composite: float
    p5: float


def evaluate(ranked_ids: list[str], gt: dict[str, int]) -> Metrics:
    ndcg10 = ndcg_at_k(ranked_ids, gt, 10)
    ndcg50 = ndcg_at_k(ranked_ids, gt, 50)
    mapv = mean_average_precision(ranked_ids, gt)
    p10 = precision_at_k(ranked_ids, gt, 10)
    p5 = precision_at_k(ranked_ids, gt, 5)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10
    return Metrics(
        ndcg10=round(ndcg10, 4),
        ndcg50=round(ndcg50, 4),
        map=round(mapv, 4),
        p10=round(p10, 4),
        composite=round(composite, 4),
        p5=round(p5, 4),
    )
