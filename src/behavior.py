"""Stage 3 — behavioral availability modifier.

The JD is explicit: "a perfect-on-paper candidate who hasn't logged in for 6
months and has a 5% recruiter response rate is, for hiring purposes, not
actually available. Down-weight them appropriately." EDA also found behavioral
twins — pairs of otherwise-identical strong candidates separated only by
engagement — so behavior is a genuine within-tier ordering signal, not a
tiebreak afterthought.

This returns a multiplicative factor in roughly [0.6, 1.1]: a strongly engaged,
open-to-work, recently active candidate gets a small boost; a stale, unengaged
one is meaningfully discounted. It is multiplicative and bounded so it refines
ordering within an evidence tier without inverting tiers.
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
