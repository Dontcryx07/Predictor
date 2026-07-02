"""Where evidence + jdfit + behavior + traps get combined into one number.

    within_score = evidence_grade * (1 + ALPHA*jdfit) * behavior_mult * trap_penalty
    composite    = evidence_tier + squash(within_score)

The tier is added as a whole number so it dominates -- no amount of good
location/notice/behavior can push a Tier-3 candidate above a Tier-4 one. Only
within the same tier does the continuous stuff matter. Took a couple of
iterations to land on this; my first version was a pure weighted sum and it
let a very-online Tier-2 candidate outrank a quiet Tier-4, which is obviously
wrong.

Honeypots get dropped before ranking (not just penalized -- see traps.py for
why). Sorting is deterministic: composite descending, candidate_id ascending
on ties, matching exactly what the validator checks for.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import behavior, evidence, jdfit, parse, traps
from .templates_audit import grade_to_tier


# Continuous penalty for soft traps (keyword stuffers not severe enough to be
# honeypots). Bounded strictly in (0, 1].
STUFFER_PENALTY = 0.5
# Divisor to squash within_score (max ~1*(1+ALPHA)*MAX_MULT) into [0, 1).
_WITHIN_DIVISOR = 1.55


@dataclass
class ScoredCandidate:
    candidate_id: str
    candidate: dict
    evidence: evidence.CandidateEvidence
    jd: jdfit.JDFitResult
    beh: behavior.BehaviorResult
    trap: traps.TrapResult
    tier: int
    within_score: float
    composite: float
    confidence: float


def _confidence(ev: evidence.CandidateEvidence, jd: jdfit.JDFitResult,
                beh: behavior.BehaviorResult, trap: traps.TrapResult) -> float:
    """A [0,1] confidence that this ranking is well-supported.

    High when strong, corroborated evidence aligns with decent JD-fit and
    engagement; low when signals disagree (e.g. a keyword stuffer, or a strong
    profile that is stale/unreachable). Used for reasoning tone, not ordering.
    """
    conf = 0.35 + 0.5 * ev.grade
    if ev.strong_role_count >= 2:
        conf += 0.08
    if jd.score >= 0.0:
        conf += 0.05
    else:
        conf -= 0.05
    if beh.multiplier >= 1.0:
        conf += 0.05
    elif beh.multiplier < 0.8:
        conf -= 0.10
    if trap.is_keyword_stuffer:
        conf -= 0.25
    return round(max(0.0, min(1.0, conf)), 4)


def score_candidate(candidate: dict, anchor: parse.Anchor) -> ScoredCandidate:
    ev = evidence.score_candidate(candidate, anchor)
    jd = jdfit.score(candidate)
    beh = behavior.modifier(candidate, anchor)
    trap = traps.detect(candidate, anchor, ev.grade)

    trap_penalty = STUFFER_PENALTY if trap.is_keyword_stuffer else 1.0
    within = ev.grade * (1.0 + jdfit.ALPHA * jd.score) * beh.multiplier * trap_penalty
    within = max(0.0, within)
    tier = grade_to_tier(ev.grade)

    within01 = min(0.999999, within / _WITHIN_DIVISOR)
    composite = tier + within01

    return ScoredCandidate(
        candidate_id=parse.candidate_id(candidate),
        candidate=candidate,
        evidence=ev,
        jd=jd,
        beh=beh,
        trap=trap,
        tier=tier,
        within_score=round(within, 6),
        composite=round(composite, 6),
        confidence=_confidence(ev, jd, beh, trap),
    )


def rank(candidates: list[dict], anchor: parse.Anchor, top_n: int = 100,
         exclude_honeypots: bool = True) -> list[ScoredCandidate]:
    """Score all candidates and return the deterministic top-N shortlist.

    Honeypots are dropped before ranking. Ordering: composite descending,
    ties broken by candidate_id ascending (validator-compliant).
    """
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        sc = score_candidate(candidate, anchor)
        if exclude_honeypots and sc.trap.is_honeypot:
            continue
        scored.append(sc)

    # Sort on the *rounded* composite so that any emitted-score ties are
    # guaranteed to be ordered by candidate_id ascending — precisely the
    # validator's tie-break rule — with no rounding-induced violations.
    scored.sort(key=lambda s: (-s.composite, s.candidate_id))
    return scored[:top_n]
