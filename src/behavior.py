"""Behavioral availability modifier.

The JD flat out says it: a perfect-on-paper candidate who hasn't logged in
for 6 months and ignores 95% of recruiter messages isn't actually available,
and should be down-weighted. While poking around the pool I also found pairs
of candidates with near-identical career histories and titles that differ
mainly in engagement (one responds 88% of the time, the other 60%) -- which
suggests the ground truth probably does use this to separate otherwise-tied
candidates, not just as a coin-flip tiebreak.

Returns a multiplier roughly in [0.6, 1.1] combining activity recency,
response rate, open-to-work flag, and interview completion. Kept fairly
narrow on purpose -- this should reorder people within a tier, not let a
hyperactive Tier-2 candidate leapfrog a quiet Tier-5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import parse


# Bounds for the final modifier.
MIN_MULT = 0.60
MAX_MULT = 1.10


@dataclass
class BehaviorResult:
    multiplier: float
    months_since_active: float
    components: dict = field(default_factory=dict)


def _recency_component(candidate: dict, anchor: parse.Anchor) -> tuple[float, float]:
    """Return (component in [-1,1], months_since_active)."""
    last_active = parse.signals(candidate).get("last_active_date")
    months = parse.months_between(last_active, anchor.date)
    if months is None:
        return 0.0, -1.0
    months = max(0.0, float(months))
    # <=1 month: fully active (+1). ~6 months: 0. >=12 months: strongly stale.
    comp = max(-1.0, min(1.0, 1.0 - months / 6.0))
    return comp, months


def _response_component(candidate: dict) -> float:
    rate = parse.signals(candidate).get("recruiter_response_rate")
    if not isinstance(rate, (int, float)):
        return 0.0
    # 0.5 is neutral; 0.9 -> +0.8; 0.1 -> -0.8.
    return max(-1.0, min(1.0, (rate - 0.5) / 0.5))


def _open_component(candidate: dict) -> float:
    return 1.0 if parse.signals(candidate).get("open_to_work_flag") else -0.4


def _interview_component(candidate: dict) -> float:
    rate = parse.signals(candidate).get("interview_completion_rate")
    if not isinstance(rate, (int, float)):
        return 0.0
    return max(-1.0, min(1.0, (rate - 0.5) / 0.5))


# Component weights (sum used to normalize into [-1, 1]).
_WEIGHTS = {
    "recency": 0.40,
    "response": 0.30,
    "open_to_work": 0.20,
    "interview": 0.10,
}


def modifier(candidate: dict, anchor: parse.Anchor) -> BehaviorResult:
    recency, months = _recency_component(candidate, anchor)
    response = _response_component(candidate)
    open_c = _open_component(candidate)
    interview = _interview_component(candidate)

    comps = {
        "recency": recency,
        "response": response,
        "open_to_work": open_c,
        "interview": interview,
    }
    blended = sum(_WEIGHTS[k] * comps[k] for k in comps) / sum(_WEIGHTS.values())
    blended = max(-1.0, min(1.0, blended))

    # Map [-1, 1] -> [MIN_MULT, MAX_MULT], with 0 -> ~1.0 (neutral).
    if blended >= 0:
        mult = 1.0 + blended * (MAX_MULT - 1.0)
    else:
        mult = 1.0 + blended * (1.0 - MIN_MULT)
    mult = round(max(MIN_MULT, min(MAX_MULT, mult)), 6)

    return BehaviorResult(
        multiplier=mult,
        months_since_active=round(months, 1),
        components=comps,
    )
