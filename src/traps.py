"""Honeypot and keyword-stuffer detection.

The spec mentions ~80 honeypot candidates with subtly impossible profiles, and
says a >10% honeypot rate in the top 100 gets you disqualified. So this matters
more than almost anything else in the pipeline.

My first instinct was to check tenure against real company founding dates
(e.g. flag anyone who "worked at CRED" before 2018). Turned out that's a dead
end -- during EDA I found company names are assigned completely independently
of role content and dates (Infosys/Wipro and fictional names like "Pied Piper"
all show up ~23.5k times regardless of what the person actually did), so that
check would have false-flagged ~177 perfectly normal candidates. Ditched it.

What actually works is checking for contradictions *within* a candidate's own
data:
    1. Experience inflation - stated years_of_experience is way more than the
       career history actually spans.
    2. Phantom expertise - "expert" in several skills with 0 months of use.

Any flagged candidate gets dropped from the shortlist entirely rather than
just penalized. We flag more candidates than the ~80 honeypots the spec
mentions, but that's fine -- dropping a borderline real candidate costs
basically nothing, whereas letting one honeypot through is a big risk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import parse


# Experience is considered inflated if stated YoE exceeds the career-history
# span by more than this many years. EDA: honeypots overshoot by 7-13 years,
# so a threshold of 3 catches them with margin while tolerating normal
# truncated histories.
YOE_SPAN_GAP_YEARS = 3.0
# Number of expert-proficiency, zero-duration skills that constitutes phantom
# expertise.
PHANTOM_EXPERT_MIN = 3

# Skill tokens that are "AI/ML core" for keyword-stuffer detection. This mirrors
# the trap the JD calls out: a profile whose *skills list* is full of AI terms
# but whose *described work* contains no such evidence.
_AI_SKILL_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:"
    r"nlp|llm|rag|ml|machine learning|deep learning|pytorch|tensorflow|"
    r"fine-?tuning|fine-?tuning llms|langchain|pinecone|faiss|milvus|weaviate|"
    r"qdrant|vector|embedding|embeddings|transformer|transformers|hugging ?face|"
    r"bert|gpt|lora|semantic search|recommendation|ranking|learning to rank|"
    r"information retrieval|mlops|feature engineering|gans|speech recognition|"
    r"image classification|object detection|tts|statistical modeling"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass
class TrapResult:
    is_honeypot: bool
    honeypot_reasons: list[str] = field(default_factory=list)
    is_keyword_stuffer: bool = False
    stuffer_ai_skill_count: int = 0
    yoe_span_gap: float = 0.0
    phantom_expert_count: int = 0


def _career_span_years(candidate: dict, anchor: parse.Anchor) -> float:
    """Years from the earliest role start to the anchor date.

    Returns 0.0 if there are no parseable start dates.
    """
    starts = [
        parse.month_index(role.get("start_date"))
        for role in parse.career_history(candidate)
    ]
    starts = [s for s in starts if s is not None]
    if not starts:
        return 0.0
    return max(0.0, (anchor.month - min(starts)) / 12.0)


def _phantom_expert_count(candidate: dict) -> int:
    count = 0
    for skill in parse.skills(candidate):
        prof = str(skill.get("proficiency", "")).lower()
        duration = skill.get("duration_months")
        if prof == "expert" and (duration == 0 or duration is None):
            # Only count an explicit zero, not merely missing, to stay precise.
            if duration == 0:
                count += 1
    return count


def _ai_skill_count(candidate: dict) -> int:
    count = 0
    for skill in parse.skills(candidate):
        name = str(skill.get("name", ""))
        if _AI_SKILL_PATTERN.search(name):
            count += 1
    return count


def detect(candidate: dict, anchor: parse.Anchor, evidence_grade: float) -> TrapResult:
    """Detect honeypot and keyword-stuffer traps for a single candidate.

    ``evidence_grade`` is the candidate's aggregated evidence grade (from
    ``evidence.score_candidate``); it lets us identify keyword-stuffers as
    "many AI skills listed, but no evidence of that work in the described
    career history".
    """
    reasons: list[str] = []

    prof = parse.profile(candidate)
    stated_yoe = prof.get("years_of_experience")
    stated_yoe = float(stated_yoe) if isinstance(stated_yoe, (int, float)) else 0.0
    span = _career_span_years(candidate, anchor)
    gap = stated_yoe - span
    if span > 0.0 and gap > YOE_SPAN_GAP_YEARS:
        reasons.append(
            f"stated {stated_yoe:.1f}y experience vs only ~{span:.1f}y of career history"
        )

    phantom = _phantom_expert_count(candidate)
    if phantom >= PHANTOM_EXPERT_MIN:
        reasons.append(
            f"{phantom} skills claimed 'expert' with 0 months of use"
        )

    ai_skills = _ai_skill_count(candidate)
    # Keyword stuffer: AI-dense skills list but the described work shows no
    # meaningful ML/IR evidence. Soft signal (already reflected in a ~0 evidence
    # grade); surfaced for reasoning and penalties, not for hard exclusion.
    is_stuffer = ai_skills >= 5 and evidence_grade < 0.20

    return TrapResult(
        is_honeypot=bool(reasons),
        honeypot_reasons=reasons,
        is_keyword_stuffer=is_stuffer,
        stuffer_ai_skill_count=ai_skills,
        yoe_span_gap=round(gap, 2),
        phantom_expert_count=phantom,
    )
