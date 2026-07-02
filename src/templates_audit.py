"""The 44-template evidence audit — the human-graded reference set.

EDA established that every ``career_history[].description`` in the 100k pool is
one of exactly 44 canonical strings. That makes the text universe small enough
to audit *exhaustively* rather than by sampling. This module records, for each
template, a human judgment of how strongly it evidences fit for the JD
(Senior AI Engineer building production retrieval / ranking / search / recsys
systems), expressed as:

    * ``tier``  — integer 0..5 relevance tier (mirrors the hidden GT scale).
    * ``grade`` — continuous 0..1 target evidence grade (finer ordering).
    * ``family``— a short role-family label (for reporting / reasoning).
    * ``note``  — the written justification for the grade.

IMPORTANT — how this is used:
    * At *ranking time* we do NOT look candidates up in this table. Evidence is
      produced by the general lexical scorer in ``evidence.py`` so the system
      generalizes to unseen text (e.g. the sandbox sample).
    * This audit is the *validation set*: a unit test asserts the general
      scorer places every one of the 44 templates in the correct tier band.
    * ``evaluate.py`` uses it to build a proxy ground truth for local NDCG.

Templates are matched by a normalized 45-char prefix, which is unique across
all 44 canonical strings (verified in tests). Full texts live in the committed
``dataset/_templates_extracted.json`` artifact produced by
``tools/extract_templates.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEntry:
    template_id: int
    prefix: str        # unique normalized prefix used for matching
    tier: int          # 0..5
    grade: float       # 0..1 target evidence grade
    family: str
    note: str


# Tier band boundaries used to map a continuous evidence grade -> tier.
# Chosen so the audited grades below each fall inside their intended band, and
# so the scorer's continuous output can be validated tier-for-tier.
TIER_BANDS: list[tuple[float, int]] = [
    (0.05, 0),
    (0.20, 1),
    (0.36, 2),
    (0.58, 3),
    (0.78, 4),
    (1.01, 5),
]


def grade_to_tier(grade: float) -> int:
    """Map a continuous evidence grade in [0,1] to an integer tier 0..5."""
    for upper, tier in TIER_BANDS:
        if grade < upper:
            return tier
    return 5


AUDIT: list[AuditEntry] = [
    # ---- Tier 0: non-technical archetypes (keyword-stuffer trap population) --
    AuditEntry(0, "enterprise sales of cloud software solutions", 0, 0.02,
               "sales", "Enterprise SaaS sales; quota carrier. No engineering, ML or IR content whatsoever."),
    AuditEntry(1, "customer support team lead at a saas product", 0, 0.02,
               "customer_support", "Support team lead; explicitly 'lighter on technical depth'. Not an engineering role."),
    AuditEntry(2, "marketing leadership role at a b2b saas company", 0, 0.02,
               "marketing", "Demand-gen / marketing leadership. SEO/content, no ML or IR."),
    AuditEntry(3, "business analyst at a consulting firm", 0, 0.03,
               "consulting_ba", "Consulting BA; 'AI-strategy advisory but my own technical depth in AI is limited'. Maps to the JD's explicit consulting negative."),
    AuditEntry(4, "brand design and creative direction at a consu", 0, 0.02,
               "design", "Brand/creative design. No technical relevance."),
    AuditEntry(5, "mechanical engineering design role at a hardwa", 0, 0.02,
               "mechanical", "Mechanical/CAD engineering. Non-software; JD wants NLP/IR, not hardware."),
    AuditEntry(6, "senior accounting role at a mid-sized company", 0, 0.02,
               "accounting", "Accounting/finance close. No technical relevance."),
    AuditEntry(7, "content writing and seo strategy for a tech-fo", 0, 0.03,
               "content_seo", "Content writer who *mentions* AI/ML topics and LLM tools — classic keyword bait. Actual work is writing/SEO; tier 0."),
    AuditEntry(8, "operations management role at a logistics comp", 0, 0.02,
               "operations", "Logistics/fulfillment ops management. No technical relevance."),

    # ---- Tier 1: general software engineering (real eng, no ML/IR) ----------
    AuditEntry(9, "cloud infrastructure and devops work at an ent", 1, 0.12,
               "devops", "Cloud/DevOps infra; 'haven't done much application development'. Real engineering but no ML/IR."),
    AuditEntry(10, "android mobile development using java", 1, 0.10,
               "mobile", "Android/mobile dev. Engineering, but no ML/IR; explicitly mobile-only."),
    AuditEntry(11, "frontend engineering at a media company", 1, 0.09,
               "frontend", "Frontend/React; 'limited backend exposure'. No ML/IR."),
    AuditEntry(12, "java backend development at a large enterprise", 1, 0.12,
               "backend", "Java/Spring backend. Solid engineering, no ML/IR."),
    AuditEntry(13, "full-stack web application development at a saa", 1, 0.13,
               "fullstack", "Full-stack web dev. General engineering, no ML/IR."),
    AuditEntry(14, "test automation and qa engineering for a finte", 1, 0.09,
               "qa", "QA/test automation; 'entirely in QA/test engineering'. No ML/IR."),

    # ---- Tier 2: data engineering / analytics (adjacent, no real ML) --------
    AuditEntry(15, "designed and maintained the analytical data wa", 2, 0.24,
               "data_eng", "Analytics data warehouse / dbt / SQL. Data-adjacent, no modeling."),
    AuditEntry(16, "built and maintained data pipelines on apache", 2, 0.27,
               "data_eng", "Airflow/Spark pipelines 'support a few internal ML models' — adjacent, not ML itself."),
    AuditEntry(17, "backend + data hybrid role at a growth-stage s", 2, 0.26,
               "data_eng", "Backend+data warehouse; 'a couple of small predictive features' but mostly data infra."),
    AuditEntry(18, "implemented streaming data pipelines on kafka", 2, 0.27,
               "data_eng", "Kafka/Spark streaming; 'some adjacent ML exposure'. Data engineering."),
    AuditEntry(19, "mixed data science and analytics-engineering r", 3, 0.40,
               "ds_light", "~30% lightweight ML (clustering/churn in sklearn/XGBoost) + A/B experimentation framework. Genuine but light ML; low tier 3."),
    AuditEntry(20, "backend development with python (fastapi)", 2, 0.22,
               "backend_ml_integ", "Backend that *integrates* a model-serving service 'not the model itself'. Engineering, minimal ML."),

    # ---- Tier 3: ML-adjacent modeling (real ML, not retrieval/ranking) ------
    AuditEntry(21, "contributed to ml feature engineering and mode", 3, 0.46,
               "ml_prod_eng", "Production ML engineer (fraud): serving API, feature store, observability. Real ML production work, modeling secondary."),
    AuditEntry(22, "built recommendation-style features at a mid-s", 4, 0.67,
               "recsys", "Production recommendation features: collaborative filtering + gradient-boosted re-ranking. Directly relevant recsys, but scope-limited ('lighter weight than FAANG', 'production deployment was handled by the platform team'), so tier 4 not tier 5."),
    AuditEntry(23, "built computer vision models for our product's", 1, 0.16,
               "cv_primary", "CV-primary (image moderation, ResNet); 'NLP/LLM professional experience there is limited'. JD explicitly de-prioritizes CV-primary without NLP/IR."),
    AuditEntry(24, "worked on time-series forecasting models", 3, 0.38,
               "ml_forecasting", "Time-series forecasting (Prophet/LightGBM/LSTM) + some RL. Real ML, but not IR/ranking/retrieval."),
    AuditEntry(25, "worked on customer-facing predictive modeling", 3, 0.42,
               "ml_predictive", "Predictive modeling (churn/LTV) with sklearn/XGBoost at e-commerce. Real applied ML, not retrieval/ranking."),
    AuditEntry(26, "built nlp pipelines for sentiment analysis", 4, 0.62,
               "nlp_classification", "Production NLP with transformers (DistilBERT, PyTorch/HF). Classification rather than retrieval/ranking, but genuine transformer-based NLP is a core JD competency, so high tier 3 / low tier 4."),

    # ---- Tier 4: strong production ranking / retrieval / MLOps --------------
    AuditEntry(27, "owned the ranking layer for an e-commerce sear", 5, 0.85,
               "ranking_search", "Owned a search product's ranking layer; hand-tuned -> learning-to-rank; relevance labeling + eval workflow. Core JD match."),
    AuditEntry(28, "trained and shipped multiple ranking models", 5, 0.84,
               "ranking_recsys", "Shipped ranking models for a discovery feed; offline-online correlation analysis. Ranking + evaluation rigor — core JD."),
    AuditEntry(29, "developed a semantic search feature for an inte", 5, 0.90,
               "semantic_search", "Semantic search with sentence-transformers + FAISS vs BM25, human relevance judgments. Textbook JD retrieval work."),
    AuditEntry(30, "implemented a rag-based customer support chatb", 5, 0.85,
               "rag", "RAG chatbot: document ingestion + embeddings + Pinecone vector store + a real eval framework + measured production impact. Squarely the JD's 'embeddings-based retrieval + vector databases + RAG' — tier 5 despite the support-chatbot domain."),
    AuditEntry(31, "built a content recommendation system serving", 5, 0.89,
               "recsys_ranking", "10M-user content recsys: CF + content-based ranking via sentence-transformer embeddings + A/B. Core recsys/ranking at scale."),
    AuditEntry(32, "built and operated production ml pipelines usin", 4, 0.64,
               "mlops", "Production MLOps (MLflow/Kubeflow/feature store/monitoring). Strong production ML — a JD 'must have' — but underlying model is churn, not ranking."),

    # ---- Tier 5: elite explicit retrieval + ranking at scale ----------------
    AuditEntry(33, "built a rag-based ranking pipeline serving 50m", 5, 0.99,
               "elite_recruiter_search", "The dream candidate: recruiter-facing hybrid BM25+dense (BGE/FAISS HNSW) + LLM re-ranker + LTR fallback + NDCG/MRR eval vs A/B. Verbatim the JD."),
    AuditEntry(34, "fine-tuned llama-2-7b and mistral-7b variants", 5, 0.93,
               "elite_llm_matching", "LoRA/QLoRA fine-tuning for candidate-JD matching + ranking-metric eval harness + production serving. Fine-tuning (JD nice-to-have) on the exact domain."),
    AuditEntry(35, "built and shipped a production recommendation", 5, 0.90,
               "recsys_full", "Marketplace recsys: CF + content (TF-IDF + sentence-transformer) + behavioral re-ranking + cold-start + A/B. Full-stack recsys."),
    AuditEntry(36, "owned the end-to-end ranking pipeline at a reco", 5, 0.97,
               "elite_hybrid_ranking", "End-to-end ranking: embeddings (BGE-large) -> Pinecone retrieval -> LTR re-scoring (XGBoost) -> behavioral integration + eval calibration. Elite hybrid."),
    AuditEntry(37, "led the migration from keyword-based to embeddi", 5, 0.95,
               "elite_search_migration", "Keyword->embedding search migration over 30M-candidate corpus, A/B, index/embedding versioning, recruiter engagement. Matches JD ops experience."),
    AuditEntry(38, "owned the design and rollout of a large-scale s", 5, 0.96,
               "elite_semantic_scale", "35M-item semantic search, BM25->hybrid sparse+dense, NDCG@10, index refresh, embedding-drift monitoring. Matches JD 'embedding drift/index refresh' verbatim."),

    # ---- Tier 5: plain-language elites (no jargon — the JD's stated trap) ----
    AuditEntry(39, "built systems that understand what users are lo", 5, 0.86,
               "plain_matching", "Plain-language: 'connect users to the most relevant matches', overhauled the matching layer from heuristics to 'explicit modeling and evaluation'. Tier 5 with zero AI keywords."),
    AuditEntry(40, "designed the ranking layer for the company's fl", 5, 0.85,
               "plain_ranking", "Plain-language ranking layer: 'surface the right thing at the right time across millions of items', owned data pipeline + evaluation framework. Tier 5, no jargon."),
    AuditEntry(41, "shipped the personalization infrastructure", 5, 0.84,
               "plain_personalization", "Plain-language personalization/relevance system + offline/online experimentation + drift detection. Tier 5, no jargon."),
    AuditEntry(42, "owned the search and discovery experience end-t", 5, 0.87,
               "plain_search", "Plain-language search & discovery end-to-end: 'most relevant results for each user's intent', ranking algorithms + evaluation methodology. Tier 5, no jargon."),
    AuditEntry(43, "led the engineering team building infrastructu", 5, 0.85,
               "plain_retrieval", "Plain-language retrieval at scale: 'surface relevant content', billions of docs, index refresh + query understanding + ranking calibration. Tier 5, no jargon."),
]


_PREFIX_LEN = 45


def normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace for robust prefix matching."""
    return " ".join(text.lower().split())


def match_description(description: str) -> AuditEntry | None:
    """Return the audit entry whose prefix the description starts with.

    Used only by tests and the proxy-ground-truth builder — never in the
    ranking path. Returns ``None`` for text that is not one of the known
    canonical templates (e.g. an unseen sandbox sample).
    """
    norm = normalize_for_match(description)
    for entry in AUDIT:
        if norm.startswith(entry.prefix[:_PREFIX_LEN]):
            return entry
    return None


def audit_by_id() -> dict[int, AuditEntry]:
    return {entry.template_id: entry for entry in AUDIT}
