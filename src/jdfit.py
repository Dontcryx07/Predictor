"""JD-fit: the "logistics" score, separate from whether they can do the job.

evidence.py answers "did they actually do this kind of work". This module
answers the more mundane question of whether the rest of the JD lines up --
years of experience, location, notice period, product vs. consulting
background, and whether their claimed skills hold up against Redrob's own
assessment scores.

This is deliberately a *small* correction on top of evidence, not a second
vote of equal weight -- see ALPHA below. A candidate with amazing evidence but
a slightly long notice period should still clearly outrank a mediocre one
sitting in Noida. The weights here are my best read of the JD's stated
priorities (5-9y band, Pune/Noida offices, sub-30-day notice preferred), not
anything derived from the data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import parse


# Weight applied to jdfit in the final multiplier (1 + ALPHA * jdfit).
ALPHA = 0.30

# Location tiers (lowercased city keywords).
PUNE_NOIDA = ("pune", "noida")
INDIA_TIER1 = (
    "hyderabad", "mumbai", "bangalore", "bengaluru", "delhi", "gurgaon",
    "gurugram", "chennai", "new delhi",
)

_SUBSCORE_WEIGHTS = {
    "experience": 0.34,
    "location": 0.30,
    "notice": 0.15,
    "product": 0.11,
    "skill_trust": 0.10,
}

_PRODUCT_MARKERS = (
    "product company", "consumer product", "consumer-product", "marketplace",
    "b2b saas", "saas product", "saas company", "growth-stage startup",
    "mid-stage startup", "e-commerce", "consumer-app", "consumer app",
    "flagship product",
)
_CONSULTING_MARKER = "consulting firm"


@dataclass
class JDFitResult:
    score: float                       # clamped [-1, 1]
    experience: float
    location: float
    notice: float
    product: float
    skill_trust: float
    location_label: str = ""
    breakdown: dict = field(default_factory=dict)


def _experience_fit(years: float) -> float:
    """Peak at 6-8y, gentle shoulders across 5-9, negative at the extremes."""
    if years <= 0:
        return -1.0
    if 6.0 <= years <= 8.0:
        return 1.0
    if 5.0 <= years < 6.0 or 8.0 < years <= 9.0:
        return 0.7
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.3
    if 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return -0.1
    if years < 3.0:
        return -0.6
    return -0.4  # > 13 years: over-senior for this IC-heavy role


def _location_fit(candidate: dict) -> tuple[float, str]:
    prof = parse.profile(candidate)
    location = str(prof.get("location", "")).lower()
    country = str(prof.get("country", "")).lower()
    signals = parse.signals(candidate)
    relocate = bool(signals.get("willing_to_relocate"))

    if any(city in location for city in PUNE_NOIDA):
        return 1.0, "Pune/Noida (office location)"
    if any(city in location for city in INDIA_TIER1):
        return 0.6, "India Tier-1 city"
    if country == "india":
        return 0.3, "elsewhere in India"
    if relocate:
        return -0.1, "outside India, willing to relocate"
    return -0.6, "outside India, not open to relocation"


def _notice_fit(candidate: dict) -> float:
    notice = parse.signals(candidate).get("notice_period_days")
    if not isinstance(notice, (int, float)):
        return 0.0
    if notice <= 30:
        return 0.6
    if notice <= 60:
        return 0.2
    if notice <= 90:
        return 0.0
    if notice <= 150:
        return -0.3
    return -0.5


def _product_fit(candidate: dict) -> float:
    text = " ".join(parse.evidence_text_blocks(candidate)).lower()
    has_product = any(m in text for m in _PRODUCT_MARKERS)
    has_consulting = _CONSULTING_MARKER in text
    if has_product and not has_consulting:
        return 0.4
    if has_consulting and not has_product:
        return -0.5
    if has_product and has_consulting:
        return 0.0
    return 0.0


def _skill_trust(candidate: dict) -> float:
    """Reward assessment scores that corroborate claimed proficiency.

    Never penalizes missing assessments (many genuine candidates have none);
    only rewards demonstrated, verified skill depth.
    """
    scores = parse.signals(candidate).get("skill_assessment_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return 0.0
    values = [v for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    # avg 50 -> 0; avg 80 -> ~0.6; avg 30 -> ~-0.4 (poor verified performance).
    return max(-0.5, min(1.0, (avg - 50.0) / 50.0))


def score(candidate: dict) -> JDFitResult:
    prof = parse.profile(candidate)
    years = prof.get("years_of_experience")
    years = float(years) if isinstance(years, (int, float)) else 0.0

    exp = _experience_fit(years)
    loc, loc_label = _location_fit(candidate)
    notice = _notice_fit(candidate)
    product = _product_fit(candidate)
    skill_trust = _skill_trust(candidate)

    subs = {
        "experience": exp,
        "location": loc,
        "notice": notice,
        "product": product,
        "skill_trust": skill_trust,
    }
    blended = sum(_SUBSCORE_WEIGHTS[k] * subs[k] for k in subs)
    total_weight = sum(_SUBSCORE_WEIGHTS.values())
    blended = blended / total_weight  # normalize to [-1, 1]
    blended = max(-1.0, min(1.0, blended))

    return JDFitResult(
        score=round(blended, 6),
        experience=exp,
        location=loc,
        notice=notice,
        product=product,
        skill_trust=skill_trust,
        location_label=loc_label,
        breakdown=subs,
    )
