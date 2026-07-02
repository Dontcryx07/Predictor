"""The evidence scorer -- this is the actual core of the ranker.

Idea: read what a candidate *actually did* from the free-text career
descriptions (plus headline/summary as a minor tie-breaker), and turn that
into a 0-1 "evidence grade" for how much their work looks like the JD's real
ask (retrieval / ranking / search / recsys / NLP-IR), not how many buzzwords
are in their skills list.

A few things this has to handle, learned the hard way from looking at actual
profiles:

- It can't be a keyword lookup against the skills array, because that's
  exactly the surface the JD warns is gamed (see the Marketing Manager with
  9 "AI core skills" in the sample submission).
- Some of the strongest candidates describe real ranking/retrieval work in
  completely plain English -- "built systems that connect users to the most
  relevant matches" -- with zero ML jargon. The phrase list below has to
  include that kind of language, not just BM25/FAISS/embeddings-speak.
- Content writers who namedrop "AI/ML topics" and "LLM tools" need to still
  score near zero, because the actual job is writing, not building anything.
- CV-only work gets capped rather than zeroed, since it's real ML, just not
  the NLP/IR the role needs.

Matching uses word-boundary regex, not naive substring search -- otherwise
short tokens like "rag" or "ltr" match inside words like "sto[rag]e" or
"fi[ltr]ate", which was a real bug in an earlier version of this.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Pattern

from . import parse


# ---------------------------------------------------------------------------
# Concept ontology
# ---------------------------------------------------------------------------
# Positive families: name -> (weight, [phrases]). Present-family weights sum
# into "primary_raw" which is then squashed. Core IR/ranking/recsys families
# carry the most weight because they are the JD's actual target work.
# ``eng_general`` and ``scale`` are handled specially (see below).
POSITIVE_FAMILIES: dict[str, tuple[float, list[str]]] = {
    "retrieval_search": (0.55, [
        "semantic search", "search product", "search engine", "search and discovery",
        "search relevance", "retrieval", "vector search", "nearest-neighbor",
        "nearest neighbor", "faiss", "pinecone", "milvus", "weaviate", "qdrant",
        "elasticsearch", "opensearch", "bm25", "information retrieval",
        "surface relevant", "surface the right thing", "most relevant results",
        "most relevant matches", "query understanding", "query expansion",
        "keyword-based to embedding", "embedding-based search", "hybrid retrieval",
        "dense retrieval", "sparse and dense", "index refresh",
        "connect them to the most relevant",
    ]),
    "ranking": (0.55, [
        "ranking layer", "ranking model", "ranking models", "ranking pipeline",
        "ranking algorithm", "ranking algorithms", "ranking calibration", "re-rank",
        "re-ranker", "re-ranking", "reranker", "learning-to-rank", "learning to rank",
        "scoring function", "relevance labeling", "discovery feed", "ltr",
    ]),
    "recsys_matching": (0.50, [
        "recommendation system", "recommendation-style", "recommender",
        "recommendations-heavy", "collaborative filtering", "matrix factorization",
        "content-based ranking", "content recommendation", "matching layer",
        "matching system", "personalization", "personalized", "cold-start",
        "cold start", "candidate-jd matching",
    ]),
    "embeddings": (0.28, [
        "embedding", "embeddings", "sentence-transformer", "sentence transformers",
        "bge", "mpnet", "all-minilm", "minilm", "dense vector",
    ]),
    "nlp_llm": (0.20, [
        "nlp", "natural language", "llm", "rag", "retrieval-augmented",
        "transformer", "transformers", "fine-tune", "fine-tuned", "fine tuning",
        "lora", "qlora", "distilbert", "bert", "hugging face", "huggingface",
        "llama", "mistral", "gpt-4", "openai embeddings", "sentiment analysis",
        "document classification", "language model",
    ]),
    "eval_framework": (0.20, [
        "ndcg", "mrr", "recall@", "offline evaluation", "offline metrics",
        "offline-online", "offline and online", "a/b test", "a/b testing",
        "evaluation framework", "evaluation methodology", "relevance judgments",
        "held-out eval", "held-out", "eval harness", "offline experimentation",
        "experimentation framework", "simulated a/b", "online engagement",
        "explicit modeling and evaluation",
    ]),
    "mlops": (0.28, [
        "mlflow", "kubeflow", "feature store", "model serving", "model-serving",
        "drift detection", "embedding drift", "data drift", "model monitoring",
        "bentoml", "index versioning", "embedding versioning", "retraining",
        "inference service", "production ml pipelines",
    ]),
    "ml_general": (0.26, [
        "machine learning", "xgboost", "lightgbm", "scikit-learn", "sklearn",
        "gradient-boosted", "gradient boosted", "predictive model",
        "predictive modeling", "churn prediction", "churn model", "classification",
        "clustering", "forecasting", "feature engineering", "prophet", "lstm",
        "reinforcement learning", "pytorch",
    ]),
    "data_eng": (0.16, [
        "airflow", "spark", "pyspark", "dbt", "snowflake", "data warehouse",
        "data pipelines", "data pipeline", "data quality", "data infrastructure",
        "looker", "dimensional modeling",
    ]),
    "eng_general": (0.12, [
        "backend", "microservices", "spring boot", "react", "typescript",
        "full-stack", "fullstack", "kubernetes", "docker", "terraform", "devops",
        "rest api", "android", "kotlin", "frontend", "selenium", "ci/cd",
        "fastapi", "node.js", "postgres",
    ]),
    "scale": (0.06, [
        "10m+", "50m+", "30m+", "35m+", "millions of", "billions of",
        "queries per month", "500k", "200k",
    ]),
}

NONTECH_MARKERS: list[str] = [
    "enterprise sales", "sales cycle", "arr quota", "quota", "support agents",
    "customer support team", "demand-generation", "demand generation",
    "content marketing", "performance marketing", "account-based marketing",
    "brand identity", "brand design", "packaging design", "adobe suite",
    "creative direction", "mechanical engineering", "solidworks", "creo",
    "month-end close", "statutory compliance", "fixed-asset", "staff accountants",
    "fulfillment operations", "picking, packing", "warehouses", "consulting firm",
    "business diagnostics", "slide-craft", "content writing", "seo strategy",
    "editorial calendar", "freelance writer",
]

CV_PRIMARY_MARKERS: list[str] = [
    "computer vision", "image moderation", "resnet", "object detection",
]

DISCLAIMER_MARKERS: list[str] = [
    "adjacent ml exposure", "some adjacent ml", "wouldn't call myself an ml specialist",
    "technical depth in ai is limited", "not the model itself",
    "lighter on technical depth", "professional experience there is limited",
    "my own modeling work was secondary", "haven't done much application development",
    "modeling work was secondary", "limited backend exposure",
    "deployment was handled by", "handled by the platform team", "lighter weight than",
]

# Tuning constants (calibrated against the 44-template audit).
SQUASH_K = 2.2               # grade = 1 - exp(-K * primary_raw)
ENG_FLOOR = 0.13             # floor for real-but-unrelated software engineering
DISCLAIMER_MULT = 0.72       # applied once if any hedging disclaimer present
NONTECH_MULT = 0.06          # crushes keyword-bait non-tech roles to tier 0
CV_CAP = 0.16                # cap for CV-primary work lacking retrieval/ranking

CORE_FAMILIES = ("retrieval_search", "ranking", "recsys_matching")
_FLOOR_ONLY = ("eng_general",)


# ---------------------------------------------------------------------------
# Regex compilation (alphanumeric-boundary matching)
# ---------------------------------------------------------------------------
def _compile_alternation(phrases: list[str]) -> Pattern[str]:
    # Longest-first so the alternation prefers the most specific phrase.
    ordered = sorted(set(phrases), key=len, reverse=True)
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


_FAMILY_PATTERNS: dict[str, tuple[float, Pattern[str]]] = {
    name: (weight, _compile_alternation(phrases))
    for name, (weight, phrases) in POSITIVE_FAMILIES.items()
}
_NONTECH_PATTERN = _compile_alternation(NONTECH_MARKERS)
_CV_PATTERN = _compile_alternation(CV_PRIMARY_MARKERS)
_DISCLAIMER_PATTERN = _compile_alternation(DISCLAIMER_MARKERS)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceResult:
    grade: float
    families: tuple[str, ...]
    nontech_marker: Optional[str] = None
    cv_primary: bool = False
    disclaimer: bool = False


@functools.lru_cache(maxsize=None)
def score_text(text: str) -> EvidenceResult:
    """Grade a single free-text block for JD evidence. Pure function.

    Memoized: the pool contains only ~44 distinct descriptions and a few
    thousand distinct headlines/summaries, so caching collapses ~300k scoring
    calls into a few thousand and keeps full-pool ranking to a few seconds.
    """
    if not text or not isinstance(text, str):
        return EvidenceResult(grade=0.0, families=())
    norm = _normalize(text)

    present: dict[str, float] = {}
    for name, (weight, pattern) in _FAMILY_PATTERNS.items():
        if pattern.search(norm):
            present[name] = weight

    # eng_general is a floor, not an additive term, so a Kafka-mentioning
    # backend role doesn't inflate to data-engineering territory.
    primary_raw = sum(w for n, w in present.items() if n not in _FLOOR_ONLY)
    grade = 1.0 - math.exp(-SQUASH_K * primary_raw)
    if "eng_general" in present:
        grade = max(grade, ENG_FLOOR)

    disclaimer_match = _DISCLAIMER_PATTERN.search(norm)
    if disclaimer_match:
        grade *= DISCLAIMER_MULT

    nontech_match = _NONTECH_PATTERN.search(norm)
    if nontech_match:
        grade *= NONTECH_MULT

    cv_match = _CV_PATTERN.search(norm)
    cv_primary = cv_match is not None
    if cv_primary and not any(f in present for f in CORE_FAMILIES):
        grade = min(grade, CV_CAP)

    return EvidenceResult(
        grade=round(grade, 6),
        families=tuple(sorted(present.keys())),
        nontech_marker=nontech_match.group(0) if nontech_match else None,
        cv_primary=cv_primary,
        disclaimer=disclaimer_match is not None,
    )


# ---------------------------------------------------------------------------
# Candidate-level aggregation
# ---------------------------------------------------------------------------
def _recency_factor(role: dict, anchor: parse.Anchor) -> float:
    """Discount factor in [0.6, 1.0]; recent roles count fully, old ones less.

    Uses the role end date (or the anchor for current roles). The JD's "hasn't
    written production code in 18 months" concern motivates favouring recent
    demonstrated work over ancient strong work.
    """
    if role.get("is_current"):
        return 1.0
    end_month = parse.month_index(role.get("end_date"))
    if end_month is None:
        return 0.9
    years_since = max(0.0, (anchor.month - end_month) / 12.0)
    return max(0.6, min(1.0, 1.0 - 0.04 * max(0.0, years_since - 3.0)))


@dataclass
class CandidateEvidence:
    grade: float                       # final aggregated evidence grade [0,1]
    best_role_index: int               # index into career_history of top role
    best_role_grade: float
    families: tuple[str, ...]          # families matched in the best role
    per_role: list[float] = field(default_factory=list)
    strong_role_count: int = 0         # roles with grade >= 0.58 (tier 3+)
    cv_primary_only: bool = False


def score_candidate(candidate: dict, anchor: parse.Anchor) -> CandidateEvidence:
    """Aggregate per-role evidence into a single candidate evidence grade.

    Aggregation: the recency-adjusted best role sets the grade (evidence is
    about the strongest demonstrated work), with a small bonus for a second
    independent strong role. A minor headline/summary booster can lift — never
    lower — the grade, since summaries in this dataset are sometimes noisy.
    """
    roles = parse.career_history(candidate)
    per_role_grades: list[float] = []
    best_families: tuple[str, ...] = ()
    best_idx = -1
    best_adjusted = 0.0
    best_raw = 0.0
    strong_count = 0
    cv_flags = 0
    core_or_ml_roles = 0

    for i, role in enumerate(roles):
        result = score_text(role.get("description", ""))
        per_role_grades.append(result.grade)
        factor = _recency_factor(role, anchor)
        adjusted = result.grade * factor
        if result.grade >= 0.58:
            strong_count += 1
        if result.grade >= 0.36:
            core_or_ml_roles += 1
        if result.cv_primary:
            cv_flags += 1
        if adjusted > best_adjusted:
            best_adjusted = adjusted
            best_raw = result.grade
            best_families = result.families
            best_idx = i

    prof = parse.profile(candidate)
    aux_grade = 0.0
    for block in (prof.get("headline"), prof.get("summary")):
        if isinstance(block, str):
            aux_grade = max(aux_grade, score_text(block).grade)

    base = best_adjusted
    if strong_count >= 2:
        base = min(1.0, base + 0.03)
    grade = max(base, min(base + 0.05, aux_grade * 0.5))
    grade = round(min(1.0, grade), 6)

    return CandidateEvidence(
        grade=grade,
        best_role_index=best_idx,
        best_role_grade=round(best_raw, 6),
        families=best_families,
        per_role=[round(g, 6) for g in per_role_grades],
        strong_role_count=strong_count,
        cv_primary_only=(cv_flags > 0 and core_or_ml_roles == 0),
    )
