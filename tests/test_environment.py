"""
Automated tests for Scheme Env — covers J1, J2, J3.
Run with: pytest tests/test_environment.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from server.schemes import get_eligible_schemes, get_optimal_scheme
from server.scheme_env_environment import (
    SchemeEnvEnvironment,
    _compute_grader_score,
    generate_dynamic_persona,
)


# =========================================================
# J1 — Unit tests for get_eligible_schemes() and get_optimal_scheme()
# =========================================================

class TestGetEligibleSchemes:

    def test_pmkvy_eligible(self):
        profile = {"age": "25", "income": "8000", "occupation": "mason", "has_aadhaar": "False"}
        assert "PMKVY" in get_eligible_schemes(profile)

    def test_pmkvy_income_boundary_pass(self):
        profile = {"age": "25", "income": "9999", "occupation": "mason", "has_aadhaar": "False"}
        assert "PMKVY" in get_eligible_schemes(profile)

    def test_pmkvy_income_boundary_fail(self):
        # income=10000 must NOT qualify — strict less-than
        profile = {"age": "25", "income": "10000", "occupation": "mason", "has_aadhaar": "False"}
        assert "PMKVY" not in get_eligible_schemes(profile)

    def test_pmkvy_age_max_boundary_pass(self):
        profile = {"age": "35", "income": "5000", "occupation": "carpenter", "has_aadhaar": "False"}
        assert "PMKVY" in get_eligible_schemes(profile)

    def test_pmkvy_age_max_boundary_fail(self):
        profile = {"age": "36", "income": "5000", "occupation": "carpenter", "has_aadhaar": "False"}
        assert "PMKVY" not in get_eligible_schemes(profile)

    def test_mgnregs_eligible(self):
        profile = {"age": "35", "income": "3000", "occupation": "farm_labourer", "has_aadhaar": "True"}
        assert "MGNREGS" in get_eligible_schemes(profile)

    def test_mgnregs_requires_aadhaar(self):
        profile = {"age": "35", "income": "3000", "occupation": "farm_labourer", "has_aadhaar": "False"}
        assert "MGNREGS" not in get_eligible_schemes(profile)

    def test_mgnregs_wrong_occupation(self):
        profile = {"age": "35", "income": "3000", "occupation": "mason", "has_aadhaar": "True"}
        assert "MGNREGS" not in get_eligible_schemes(profile)

    def test_pmay_eligible(self):
        profile = {"age": "30", "income": "4000", "occupation": "mason", "has_aadhaar": "True"}
        assert "PMAY" in get_eligible_schemes(profile)

    def test_pmay_income_boundary_pass(self):
        profile = {"age": "30", "income": "5999", "occupation": "mason", "has_aadhaar": "True"}
        assert "PMAY" in get_eligible_schemes(profile)

    def test_pmay_income_boundary_fail(self):
        # income=6000 must NOT qualify
        profile = {"age": "30", "income": "6000", "occupation": "mason", "has_aadhaar": "True"}
        assert "PMAY" not in get_eligible_schemes(profile)

    def test_pmay_requires_aadhaar(self):
        profile = {"age": "30", "income": "4000", "occupation": "mason", "has_aadhaar": "False"}
        assert "PMAY" not in get_eligible_schemes(profile)

    def test_pmay_age_min_boundary(self):
        # age=20 is below PMAY min (21)
        profile = {"age": "20", "income": "4000", "occupation": "mason", "has_aadhaar": "True"}
        assert "PMAY" not in get_eligible_schemes(profile)

    def test_no_eligible_schemes(self):
        # income too high for all schemes, wrong occupation for MGNREGS
        profile = {"age": "25", "income": "50000", "occupation": "doctor", "has_aadhaar": "True"}
        result = get_eligible_schemes(profile)
        # Only check benchmark schemes
        assert "PMKVY" not in result
        assert "MGNREGS" not in result
        assert "PMAY" not in result

    def test_dual_eligible_pmkvy_and_pmay(self):
        # age 25, income 4000, mason, aadhaar=True → both PMKVY and PMAY eligible
        profile = {"age": "25", "income": "4000", "occupation": "mason", "has_aadhaar": "True"}
        result = get_eligible_schemes(profile)
        assert "PMKVY" in result
        assert "PMAY" in result


class TestGetOptimalScheme:

    def test_pmay_beats_pmkvy(self):
        # Both eligible — PMAY must win (higher benefit)
        profile = {"age": "25", "income": "4000", "occupation": "mason", "has_aadhaar": "True"}
        assert get_optimal_scheme(profile) == "PMAY"

    def test_mgnregs_beats_pmkvy(self):
        # farm_labourer with aadhaar and income in PMKVY range
        # MGNREGS has no income ceiling so both eligible — MGNREGS must win
        profile = {"age": "30", "income": "8000", "occupation": "farm_labourer", "has_aadhaar": "True"}
        assert get_optimal_scheme(profile) == "MGNREGS"

    def test_pmkvy_only(self):
        # income too high for PMAY, occupation wrong for MGNREGS
        profile = {"age": "25", "income": "8000", "occupation": "mason", "has_aadhaar": "False"}
        assert get_optimal_scheme(profile) == "PMKVY"

    def test_none_when_ineligible(self):
        profile = {"age": "25", "income": "50000", "occupation": "doctor", "has_aadhaar": "True"}
        # No benchmark scheme qualifies
        result = get_optimal_scheme(profile)
        assert result not in ["PMKVY", "MGNREGS", "PMAY"]


# =========================================================
# J2 — Determinism test
# =========================================================

class TestDeterminism:

    def test_same_seed_same_persona(self):
        """reset(seed=N) twice must produce identical personas."""
        env = SchemeEnvEnvironment()
        obs1 = env.reset(seed=1)
        p1_age    = obs1.known_profile.get("age")
        p1_income = obs1.known_profile.get("income")

        obs2 = env.reset(seed=1)
        p2_age    = obs2.known_profile.get("age")
        p2_income = obs2.known_profile.get("income")

        assert p1_age == p2_age, f"Age mismatch: {p1_age} vs {p2_age}"
        assert p1_income == p2_income, f"Income mismatch: {p1_income} vs {p2_income}"

    def test_different_seeds_different_personas(self):
        """reset(seed=1) and reset(seed=2) should produce different tasks."""
        env = SchemeEnvEnvironment()
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=2)
        # Task 1 and Task 2 have different missing_data
        assert obs1.missing_data != obs2.missing_data or \
               obs1.notification != obs2.notification

    def test_task5_aadhaar_age_valid_range(self):
        """Task 5 aadhaar_age must always be 36, 37, or 38."""
        env = SchemeEnvEnvironment()
        env.reset(seed=5)
        aadhaar_age = env._persona.get("_aadhaar_age")
        assert aadhaar_age in ["36", "37", "38"], \
            f"Unexpected aadhaar_age: {aadhaar_age}"


# =========================================================
# J3 — Grader score range test: 0.0 <= score <= 1.0 always
# =========================================================

class TestGraderScoreRange:

    def _assert_in_range(self, score: float):
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0.0, 1.0]"

    def test_perfect_score_no_penalties(self):
        score = _compute_grader_score(
            task=1, base_score=1.0, step_count=3,
            noise_queries=0, redundant_queries=0
        )
        self._assert_in_range(score)
        assert score == 1.0

    def test_score_with_document_bonus_clamped(self):
        # bonus should not push above 1.0
        score = _compute_grader_score(
            task=4, base_score=1.0, step_count=2,
            noise_queries=0, redundant_queries=0, document_verified=True
        )
        self._assert_in_range(score)

    def test_score_with_heavy_penalties_floored(self):
        # many noise queries should not push below 0.30 for correct answer
        score = _compute_grader_score(
            task=1, base_score=1.0, step_count=15,
            noise_queries=10, redundant_queries=5
        )
        self._assert_in_range(score)
        assert score >= 0.30, "Correct answer should always score >= 0.30"

    def test_wrong_answer_always_zero(self):
        score = _compute_grader_score(
            task=1, base_score=0.0, step_count=1,
            noise_queries=0, redundant_queries=0
        )
        assert score == 0.0

    def test_all_tasks_all_penalty_combos(self):
        """Exhaustive range check across tasks and penalty combinations."""
        for task in [1, 2, 3, 4, 5]:
            for noise in [0, 1, 3, 5, 10]:
                for redundant in [0, 1, 3, 5]:
                    for doc_verified in [True, False]:
                        for base in [0.0, 0.5, 1.0]:
                            score = _compute_grader_score(
                                task=task,
                                base_score=base,
                                step_count=10,
                                noise_queries=noise,
                                redundant_queries=redundant,
                                document_verified=doc_verified,
                            )
                            self._assert_in_range(score)