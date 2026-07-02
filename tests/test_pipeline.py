"""Unit tests for parsing, traps, jd-fit, behavior, scoring, reasoning, evaluate."""
from __future__ import annotations

from src import behavior, evaluate, evidence, jdfit, parse, reasoning, score, traps


# --------------------------------------------------------------------------- parse
def test_month_index_and_between():
    assert parse.month_index("2026-05-27") == 2026 * 12 + 5
    assert parse.months_between("2026-01-01", "2026-05-01") == 4
    assert parse.month_index("garbage") is None


def test_anchor_from_data():
    cands = [
        {"redrob_signals": {"last_active_date": "2026-01-01"}},
        {"redrob_signals": {"last_active_date": "2026-05-27"}},
        {"redrob_signals": {"last_active_date": "2025-11-11"}},
    ]
    a = parse.compute_time_anchor(cands)
    assert a.date == "2026-05-27"


# --------------------------------------------------------------------------- evidence
def test_strong_candidate_is_tier5(strong_candidate, anchor):
    ev = evidence.score_candidate(strong_candidate, anchor)
    assert ev.grade >= 0.78


def test_stuffer_evidence_is_low(stuffer_candidate, anchor):
    ev = evidence.score_candidate(stuffer_candidate, anchor)
    assert ev.grade < 0.20  # skills list is ignored; described work is non-tech


def test_cv_candidate_capped(cv_candidate, anchor):
    ev = evidence.score_candidate(cv_candidate, anchor)
    assert ev.grade <= 0.20


# --------------------------------------------------------------------------- traps
def test_honeypot_yoe_inflation(honeypot_candidate, anchor):
    tr = traps.detect(honeypot_candidate, anchor, evidence_grade=0.95)
    assert tr.is_honeypot
    assert any("experience" in r for r in tr.honeypot_reasons)


def test_honeypot_phantom_expertise(honeypot_candidate, anchor):
    tr = traps.detect(honeypot_candidate, anchor, evidence_grade=0.95)
    assert tr.phantom_expert_count >= 3


def test_strong_candidate_not_honeypot(strong_candidate, anchor):
    tr = traps.detect(strong_candidate, anchor, evidence_grade=0.95)
    assert not tr.is_honeypot


def test_keyword_stuffer_flagged(stuffer_candidate, anchor):
    ev = evidence.score_candidate(stuffer_candidate, anchor)
    tr = traps.detect(stuffer_candidate, anchor, ev.grade)
    assert tr.is_keyword_stuffer


# --------------------------------------------------------------------------- jdfit
def test_pune_beats_abroad(strong_candidate):
    fit = jdfit.score(strong_candidate)
    assert fit.location == 1.0
    assert "Pune" in fit.location_label


def test_experience_band_peaks_at_7():
    assert jdfit._experience_fit(7.0) == 1.0
    assert jdfit._experience_fit(2.0) < 0
    assert jdfit._experience_fit(15.0) < 0


def _with_roles(candidate, roles):
    out = dict(candidate)
    out["career_history"] = roles
    return out


def test_tenure_penalizes_job_hoppers(strong_candidate):
    hopper_roles = [
        {"duration_months": 14, "description": "java backend development at a large enterprise"},
        {"duration_months": 15, "description": "java backend development at a large enterprise"},
        {"duration_months": 13, "description": "java backend development at a large enterprise"},
    ]
    stable_roles = [
        {"duration_months": 48, "description": "java backend development at a large enterprise"},
        {"duration_months": 40, "description": "java backend development at a large enterprise"},
    ]
    hopper = jdfit._tenure_fit(_with_roles(strong_candidate, hopper_roles))
    stable = jdfit._tenure_fit(_with_roles(strong_candidate, stable_roles))
    assert hopper < 0 < stable


def test_tenure_neutral_for_single_role(strong_candidate):
    # One role tells us nothing about hopping; must not reward or punish.
    assert jdfit._tenure_fit(strong_candidate) == 0.0


def test_ml_depth_rewards_sustained_ml_work(strong_candidate):
    from tests.conftest import STRONG_DESC, NONTECH_DESC
    deep = [
        {"duration_months": 36, "description": STRONG_DESC},
        {"duration_months": 24, "description": STRONG_DESC},
    ]
    shallow = [
        {"duration_months": 6, "description": STRONG_DESC},
        {"duration_months": 60, "description": NONTECH_DESC},
    ]
    assert jdfit._ml_depth_fit(_with_roles(strong_candidate, deep)) == 1.0
    assert jdfit._ml_depth_fit(_with_roles(strong_candidate, shallow)) < 0


# --------------------------------------------------------------------------- behavior
def test_behavior_bounds(strong_candidate, anchor):
    b = behavior.modifier(strong_candidate, anchor)
    assert behavior.MIN_MULT <= b.multiplier <= behavior.MAX_MULT


def test_stale_inactive_downweighted(strong_candidate, anchor):
    stale = dict(strong_candidate)
    stale["redrob_signals"] = dict(strong_candidate["redrob_signals"])
    stale["redrob_signals"]["last_active_date"] = "2025-01-01"
    stale["redrob_signals"]["recruiter_response_rate"] = 0.05
    stale["redrob_signals"]["open_to_work_flag"] = False
    b = behavior.modifier(stale, anchor)
    assert b.multiplier < 0.85


# --------------------------------------------------------------------------- score
def test_tier_never_inverts(strong_candidate, cv_candidate, anchor):
    """A tier-5 candidate must always outrank a tier-1 one regardless of extras."""
    s_strong = score.score_candidate(strong_candidate, anchor)
    s_cv = score.score_candidate(cv_candidate, anchor)
    assert s_strong.composite > s_cv.composite
    assert s_strong.tier > s_cv.tier


def test_ranking_excludes_honeypots(strong_candidate, honeypot_candidate, anchor):
    ranked = score.rank([honeypot_candidate, strong_candidate], anchor, top_n=10)
    ids = [s.candidate_id for s in ranked]
    assert honeypot_candidate["candidate_id"] not in ids
    assert strong_candidate["candidate_id"] in ids


def test_ranking_deterministic_and_nonincreasing(strong_candidate, cv_candidate,
                                                  stuffer_candidate, anchor):
    pool = [cv_candidate, strong_candidate, stuffer_candidate]
    r1 = [s.candidate_id for s in score.rank(pool, anchor, top_n=10)]
    r2 = [s.candidate_id for s in score.rank(list(reversed(pool)), anchor, top_n=10)]
    assert r1 == r2  # order-independent / deterministic
    comps = [s.composite for s in score.rank(pool, anchor, top_n=10)]
    assert comps == sorted(comps, reverse=True)


def test_equal_scores_break_by_candidate_id(strong_candidate, anchor):
    twin_a = dict(strong_candidate); twin_a["candidate_id"] = "CAND_0000900"
    twin_b = dict(strong_candidate); twin_b["candidate_id"] = "CAND_0000800"
    ranked = score.rank([twin_a, twin_b], anchor, top_n=10)
    assert ranked[0].candidate_id == "CAND_0000800"  # ascending on equal score


# --------------------------------------------------------------------------- reasoning
def test_reasoning_no_hallucinated_skill(strong_candidate, anchor):
    sc = score.score_candidate(strong_candidate, anchor)
    text = reasoning.generate(sc, rank=1)
    skill_names = [s["name"] for s in strong_candidate["skills"]]
    # If a skill is named in the reasoning, it must be a real one.
    assert any(name in text for name in skill_names)
    assert "Senior AI Engineer" in text


def test_reasoning_tone_tracks_rank(strong_candidate, anchor):
    sc = score.score_candidate(strong_candidate, anchor)
    top = reasoning.generate(sc, rank=1)
    bottom = reasoning.generate(sc, rank=95)
    assert top != bottom
    assert "depth" in bottom.lower()


def test_reasoning_varies_across_candidates(strong_candidate, cv_candidate, anchor):
    a = reasoning.generate(score.score_candidate(strong_candidate, anchor), 1)
    b = reasoning.generate(score.score_candidate(cv_candidate, anchor), 2)
    assert a != b


# --------------------------------------------------------------------------- evaluate
def test_perfect_ranking_scores_one():
    gt = {"a": 5, "b": 4, "c": 0, "d": 3}
    ranked = ["a", "b", "d", "c"]
    m = evaluate.evaluate(ranked, gt)
    assert m.ndcg10 == 1.0


def test_bad_ranking_scores_low():
    gt = {"a": 5, "b": 4, "c": 0, "d": 3}
    ranked = ["c", "d", "b", "a"]
    m = evaluate.evaluate(ranked, gt)
    assert m.ndcg10 < 1.0
