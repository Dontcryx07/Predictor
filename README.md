# Redrob Intelligent Candidate Discovery & Ranking

An evidence-first candidate ranking system for the Redrob AI All-India Hackathon
(Track 1). It ranks the top-100 candidates for the **Senior AI Engineer –
Founding Team** JD out of a 100,000-candidate pool, producing a fact-grounded
reasoning for every pick.

Pure Python standard library. No GPU, no network, no precomputed artifacts,
no third-party runtime dependencies. Full-pool ranking runs in **~20-25
seconds** on a laptop CPU — well under the 5-minute budget.

---

## TL;DR — reproduce the submission

```bash
python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv
python dataset/validate_submission.py submission.csv   # -> "Submission is valid."
```

That single command reads the pool, scores every candidate, excludes honeypots,
and writes `candidate_id,rank,score,reasoning` for the top 100.

---

## The core insight

We spent time with the data before building (the CEO's stated advice). The
decisive finding: **every candidate's career history is assembled from just 44
canonical `career_history[].description` templates.** That collapses a fuzzy
semantic-matching problem into one we can audit *exhaustively*:

- ~25k uses each of nine non-technical archetypes (sales, support, marketing,
  consulting, design, mechanical, accounting, content/SEO, operations) — the
  keyword-stuffer trap population.
- ~10k each of general software roles; ~1.8k each of data engineering.
- A few hundred each of ML-adjacent roles (fraud ML, CV, forecasting, NLP
  classification).
- **Only ~179 candidates** carry a strong retrieval / ranking / search / recsys
  template (used 2–78 times each). The entire top-100 lives inside this set.

Three consequences shaped the design:

1. **Evidence comes from described work, never the skills list.** The JD warns
   that a Marketing Manager with every AI keyword is not a fit; the skills array
   is the adversarial surface. We grade the *career descriptions*.
2. **Company names are randomized noise** (Infosys, Wipro, and fictional firms
   like Pied Piper each appear ~23,500× independently of role content). So
   product-vs-services signal is read from description *text*, and impossible
   tenure at a real company is *not* used as a honeypot signal (it would
   false-flag ~177 innocents).
3. **No embeddings needed.** With a 44-string text universe, a transparent
   lexical scorer can be validated tier-for-tier against a human audit — giving
   us explainability and zero Stage-3 reproduction risk.

---

## Architecture

```
candidates.jsonl
      │
      ▼
[parse]        robust JSONL load + deterministic time anchor (max last_active_date)
      │
      ▼
[evidence]     general lexical scorer over description text → grade ∈ [0,1] → tier 0–5
      │            (validated against the 44-template audit)
      ▼
[jdfit]        experience band, location, notice, product context, skill trust → [-1,1]
      │
      ▼
[behavior]     activity recency, response rate, open-to-work, interview rate → ×[0.6,1.1]
      │
      ▼
[traps]        honeypot detection (YoE inflation, phantom expertise) + stuffer flag
      │
      ▼
[score]        composite = evidence_tier + squash(evidence×(1+α·jdfit)×behavior×penalty)
      │            honeypots excluded; deterministic tie-break by candidate_id
      ▼
[reasoning]    fact-slotted 1–2 sentence justification per candidate
      │
      ▼
submission.csv
```

### Scoring formulation

```
within_score = evidence_grade · (1 + α·jdfit) · behavior_mult · trap_penalty
composite    = evidence_tier + squash(within_score)          # α = 0.30
```

`composite` is **tier-lexicographic**: the integer evidence tier dominates, so a
Tier-5 candidate can never be pushed below a Tier-4 by the bounded JD-fit and
behavioral refinements. Within a tier, every ordering signal implements a
specific statement in the JD itself:

| Within-tier signal | JD line it implements |
| --- | --- |
| Experience band (peak 6–8y) | "6-8 years total experience" / "5-9 ... a range, not a requirement" |
| Location tiers | "Located in or willing to relocate to Noida or Pune" |
| Notice period | The JD's note on preferring short/buyable notice |
| Product-vs-consulting context | "at product companies (not pure services)" + the consulting-firms-only disqualifier |
| Tenure stability | "switching companies every 1.5 years, we're not a fit... plans to be here for 3+ years" |
| ML career depth | "of which 4-5 are in applied ML/AI roles" |
| Skill trust (assessments) | Redrob's own verified skill signals over self-claims |
| Behavioral multiplier | "hasn't logged in for 6 months and has a 5% recruiter response rate is... not actually available" |

### Honeypot defense

Honeypots (~80 in the pool, forced to Tier 0 in the hidden ground truth) are
detected by **internal profile contradictions**, not company names:

- stated `years_of_experience` exceeds the career-history span by > 3 years;
- ≥ 3 skills claimed `expert` with `duration_months == 0` (phantom expertise).

Any flagged candidate is excluded from the shortlist. We caught all 7 honeypots
that hide inside the strong-template pool (claiming 15–17y over 4–8y careers)
while keeping every legitimate plain-language Tier-5. Excluding a borderline
candidate is nearly free; ranking a honeypot risks disqualification, so the
filter is deliberately conservative (asymmetric cost).

---

## Repository layout

| Path | What it is |
| --- | --- |
| `rank.py` | Single-command entrypoint → submission CSV |
| `src/parse.py` | JSONL loading, safe accessors, deterministic time anchor |
| `src/evidence.py` | General lexical evidence scorer (the ontology + squashing) |
| `src/templates_audit.py` | The human-graded 44-template reference (tiers + justifications) |
| `src/traps.py` | Honeypot + keyword-stuffer detection |
| `src/jdfit.py` | Experience / location / notice / product / skill-trust fit |
| `src/behavior.py` | Behavioral-availability multiplier |
| `src/score.py` | Composite scoring, tier-safe ranking, deterministic tie-break |
| `src/reasoning.py` | Fact-slotting reasoning generator |
| `src/evaluate.py` | Local proxy ground truth + NDCG@10/@50, MAP, P@10 |
| `tools/` | EDA / calibration / evaluation scripts (not on the ranking path) |
| `tests/` | Unit tests, incl. the scorer-vs-audit tier validation |
| `sandbox/app.py` | Streamlit demo that ranks an uploaded small sample |
| `docs/notes.md` | Informal working notes on the data / decisions made along the way |

---

## Sandbox demo

`sandbox/app.py` is a small Streamlit wrapper around the exact same `src/`
pipeline, so a reviewer can upload a handful of candidates and see ranked
output without needing the full 465MB pool. Run it locally with:

```bash
streamlit run sandbox/app.py
```

It's deployed at: https://predictor-aqcz3wtcjbfnypjh2qmd2y.streamlit.app/.

---

## Local evaluation (and an honest caveat)

`src/evaluate.py` builds a proxy ground truth from the template audit and
computes the exact competition composite. Run it:

```bash
python -m tools.evaluate_run --candidates ./dataset/candidates.jsonl
```

**What it validates:** tier separation is correct, honeypots are handled, and
there are no tier inversions (structural correctness). **What it cannot
validate:** because the proxy is derived from the same audit that drives the
scorer, its composite is ~1.0 by construction and says nothing about the *hidden*
ground truth's ordering *within* the ~169 strong candidates. No local metric can
score that ordering — the proxy has only 6 relevance levels, so all strong
candidates tie under it. What we did instead is make every within-tier signal
traceable to an explicit sentence in the JD (see the table under Scoring
formulation), so the ordering is the hiring manager's own stated priorities
rather than our invented ones. That is a principled bet, not a measured score,
and we call it out rather than overclaim.

---

## Testing

```bash
pip install -r requirements.txt   # pytest only; runtime needs nothing
python -m pytest tests/ -q         # 31 tests
```

The key test, `test_scorer_reproduces_audit_tier_for_all_44`, asserts the
general scorer places all 44 canonical templates in their audited tier — so any
future ontology change that regresses a template fails CI.

---

## Compute & reproducibility

- **Runtime:** ~20-25s for 100k candidates (load ~10-13s, rank+score ~9-10s) on a laptop CPU.
- **Memory:** the full pool fits comfortably under 16 GB.
- **Determinism:** no RNG; stable sorts; "today" is the dataset's max
  `last_active_date` (`2026-05-27`), never the wall clock.
- **No precomputation, no network, no GPU.** `pip install` is optional (tests
  only). The `dataset/candidates.jsonl` file is git-ignored due to size; place
  it under `dataset/` before running.
