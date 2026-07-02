"""Shared pytest fixtures: small, hand-built candidates covering each archetype."""
from __future__ import annotations

import pytest

from src import parse


def _signals(**overrides):
    base = {
        "profile_completeness_score": 80.0,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 24.0,
        "skill_assessment_scores": {},
        "connection_count": 200,
        "endorsements_received": 30,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 50,
        "search_appearance_30d": 100,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.8,
        "offer_acceptance_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


STRONG_DESC = (
    "Built a RAG-based ranking pipeline serving 50M+ queries per month for an "
    "internal recruiter-facing search product. The architecture combined BM25 + "
    "dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based re-ranker on "
    "the top-50, falling back to a learning-to-rank model. Designed the offline "
    "evaluation framework from scratch — NDCG, MRR, recall@K calibrated against "
    "online A/B engagement metrics."
)
NONTECH_DESC = (
    "Content writing and SEO strategy for a tech-focused publication. Wrote "
    "longform articles on developer tools, cloud platforms, and AI/ML topics. "
    "Managed a freelance writer pool and the editorial calendar. Recent work has "
    "been on AI-assisted content production, using LLM tools for drafting."
)
CV_DESC = (
    "Built computer vision models for our product's image moderation feature "
    "using PyTorch — fine-tuned ResNet variants. Most of my project work has "
    "been in CV; my NLP/LLM professional experience there is limited."
)


@pytest.fixture
def anchor():
    return parse.Anchor(date="2026-05-27", month=2026 * 12 + 5)


@pytest.fixture
def strong_candidate():
    return {
        "candidate_id": "CAND_0000101",
        "profile": {
            "anonymized_name": "A B", "headline": "ML Engineer | Search & Ranking",
            "summary": "Built retrieval and ranking systems.",
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": 7.0, "current_title": "Senior AI Engineer",
            "current_company": "Hooli", "current_company_size": "501-1000",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "Hooli", "title": "Senior AI Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Software", "company_size": "501-1000",
            "description": STRONG_DESC,
        }],
        "education": [],
        "skills": [
            {"name": "Semantic Search", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        ],
        "redrob_signals": _signals(),
    }


@pytest.fixture
def honeypot_candidate():
    """Strong template but stated experience wildly exceeds career span."""
    return {
        "candidate_id": "CAND_0000102",
        "profile": {
            "anonymized_name": "C D", "headline": "AI Engineer",
            "summary": "Ranking systems.", "location": "Noida, Uttar Pradesh",
            "country": "India", "years_of_experience": 16.0,
            "current_title": "AI Engineer", "current_company": "Initech",
            "current_company_size": "1001-5000", "current_industry": "Software",
        },
        "career_history": [{
            "company": "Initech", "title": "AI Engineer",
            "start_date": "2022-01-01", "end_date": None, "duration_months": 52,
            "is_current": True, "industry": "Software", "company_size": "1001-5000",
            "description": STRONG_DESC,
        }],
        "education": [],
        "skills": [
            {"name": "RAG", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "Ranking", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        ],
        "redrob_signals": _signals(),
    }


@pytest.fixture
def stuffer_candidate():
    """Non-tech role, AI-dense skills list — the keyword-stuffer trap."""
    return {
        "candidate_id": "CAND_0000103",
        "profile": {
            "anonymized_name": "E F", "headline": "Content Writer",
            "summary": "Curious about AI tools.", "location": "Jaipur, Rajasthan",
            "country": "India", "years_of_experience": 6.0,
            "current_title": "Content Writer", "current_company": "Acme Corp",
            "current_company_size": "201-500", "current_industry": "Media",
        },
        "career_history": [{
            "company": "Acme Corp", "title": "Content Writer",
            "start_date": "2020-01-01", "end_date": None, "duration_months": 72,
            "is_current": True, "industry": "Media", "company_size": "201-500",
            "description": NONTECH_DESC,
        }],
        "education": [],
        "skills": [
            {"name": "NLP", "proficiency": "expert", "endorsements": 10, "duration_months": 20},
            {"name": "LLM", "proficiency": "expert", "endorsements": 10, "duration_months": 20},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
            {"name": "Fine-tuning LLMs", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
            {"name": "Vector", "proficiency": "advanced", "endorsements": 10, "duration_months": 20},
        ],
        "redrob_signals": _signals(),
    }


@pytest.fixture
def cv_candidate():
    return {
        "candidate_id": "CAND_0000104",
        "profile": {
            "anonymized_name": "G H", "headline": "Computer Vision Engineer",
            "summary": "CV models.", "location": "Bangalore, Karnataka",
            "country": "India", "years_of_experience": 6.0,
            "current_title": "Computer Vision Engineer", "current_company": "Globex Inc",
            "current_company_size": "501-1000", "current_industry": "Software",
        },
        "career_history": [{
            "company": "Globex Inc", "title": "Computer Vision Engineer",
            "start_date": "2020-06-01", "end_date": None, "duration_months": 71,
            "is_current": True, "industry": "Software", "company_size": "501-1000",
            "description": CV_DESC,
        }],
        "education": [],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "endorsements": 30, "duration_months": 60}],
        "redrob_signals": _signals(),
    }
