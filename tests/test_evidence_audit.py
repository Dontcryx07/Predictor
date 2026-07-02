"""Validate that the general evidence scorer reproduces the 44-template audit.

This is the central correctness guarantee: the scorer is a *general* lexical
model, and here we assert it places every one of the 44 canonical templates in
its human-audited relevance tier. If a future ontology change regresses any
template, this test fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import evidence
from src.templates_audit import AUDIT, grade_to_tier, match_description

TEMPLATES_PATH = Path("dataset/_templates_extracted.json")


def _load_templates():
    if not TEMPLATES_PATH.exists():
        pytest.skip("dataset/_templates_extracted.json not present")
    return json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))


def test_audit_has_44_entries():
    assert len(AUDIT) == 44
    tiers = {e.tier for e in AUDIT}
    assert tiers <= {0, 1, 2, 3, 4, 5}


def test_prefixes_unique():
    prefixes = [e.prefix[:45] for e in AUDIT]
    assert len(set(prefixes)) == len(prefixes)


def test_every_template_matches_an_audit_entry():
    for tmpl in _load_templates():
        assert match_description(tmpl["text"]) is not None, tmpl["id"]


def test_scorer_reproduces_audit_tier_for_all_44():
    templates = _load_templates()
    by_id = {e.template_id: e for e in AUDIT}
    mismatches = []
    for tmpl in templates:
        audit_entry = by_id[tmpl["id"]]
        scored_tier = grade_to_tier(evidence.score_text(tmpl["text"]).grade)
        if scored_tier != audit_entry.tier:
            mismatches.append((tmpl["id"], audit_entry.tier, scored_tier))
    assert not mismatches, f"tier mismatches (id, audit, scored): {mismatches}"


def test_elite_templates_are_tier5():
    # Templates 27-43 (minus the deliberately-tier4 ones) should be tier 5.
    templates = {t["id"]: t["text"] for t in _load_templates()}
    for tid in (27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43):
        grade = evidence.score_text(templates[tid]).grade
        assert grade_to_tier(grade) == 5, (tid, grade)


def test_nontech_templates_are_tier0():
    templates = {t["id"]: t["text"] for t in _load_templates()}
    for tid in range(0, 9):
        grade = evidence.score_text(templates[tid]).grade
        assert grade_to_tier(grade) == 0, (tid, grade)
