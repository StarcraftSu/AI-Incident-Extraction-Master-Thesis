"""Tests for `compare_values` — the core leaf-level comparison.

Covers all four field-type branches: constrained, bertscore, exact-match,
and excluded. The constrained and exact branches must handle list-typed
extractions (a real failure mode on Llama PS3 and Haiku at various
points). The BERTScore branch is exercised only on the substring
shortcut path; calls to BERTScore itself are slow (model load) and
covered by the integration tests in `test_evaluate_extraction.py`.
"""

from evaluation import compare_values, _normalize_str, _strip_harm_suffix


# ---------------------------------------------------------------------------
# Constrained fields
# ---------------------------------------------------------------------------


class TestConstrained:
    """Constrained fields use vocab-aware set-intersection matching."""

    def test_exact_match(self):
        assert compare_values("AI incident", "AI incident", "event.event_type") == "correct"
        assert compare_values("ai incident", "AI incident", "event.event_type") == "correct"

    def test_mismatch(self):
        assert compare_values("AI hazard", "AI incident", "event.event_type") == "incorrect"

    def test_comma_separated_gt(self):
        # Multi-value GT: any one value extracted is correct.
        assert compare_values("physical", "physical, psychological", "harm.harm_type") == "correct"
        assert compare_values("psychological", "physical, psychological", "harm.harm_type") == "correct"
        assert compare_values("economic", "physical, psychological", "harm.harm_type") == "incorrect"

    def test_comma_separated_extracted(self):
        # Multi-value extraction: any one matching GT is correct.
        assert compare_values("rights violation, economic", "economic", "harm.harm_type") == "correct"

    def test_list_typed_extracted(self):
        # bug_011 territory: model returns a Python list.
        assert compare_values(["physical", "psychological"], "physical", "harm.harm_type") == "correct"
        assert compare_values(["other"], "other", "ai_system.system_type") == "correct"
        assert compare_values(["unknown"], "other", "ai_system.system_type") == "incorrect"

    def test_harm_suffix_stripped(self):
        # KI3 "Physical harm" parent label should match vocab "physical"
        assert compare_values("Physical harm", "physical", "harm.harm_type") == "correct"
        assert compare_values("Economic harm", "economic", "harm.harm_type") == "correct"
        assert compare_values("Reputational harm", "reputational", "harm.harm_type") == "correct"

    def test_harm_suffix_per_element_in_list(self):
        # Multi-label string: per-element strip handles "Physical harm, Economic harm"
        assert compare_values("Physical harm, Economic harm", "physical", "harm.harm_type") == "correct"
        assert compare_values("Physical harm, Economic harm", "economic", "harm.harm_type") == "correct"
        # Same in list form
        assert compare_values(["Physical harm", "Economic harm"], "physical", "harm.harm_type") == "correct"


# ---------------------------------------------------------------------------
# Exact-match open fields
# ---------------------------------------------------------------------------


class TestExactMatch:
    """ai_system.name, developer, deployer, event_date — exact match w/ substring."""

    def test_exact(self):
        assert compare_values("OpenAI", "OpenAI", "ai_system.developer") == "correct"

    def test_substring_match_for_org_names(self):
        # Common: GT has a fuller name, extraction has the short form.
        assert compare_values("Reno Police", "Reno Police Department", "ai_system.deployer") == "correct"

    def test_list_typed_extracted(self):
        # bug_011: list of names with the right value present.
        assert compare_values(["ChatGPT", "Claude"], "ChatGPT", "ai_system.name") == "correct"
        assert compare_values(["OpenAI"], "OpenAI", "ai_system.developer") == "correct"
        assert compare_values(["2024-09-01"], "2024-09-01", "event.event_date") == "correct"

    def test_list_typed_no_match(self):
        assert compare_values(["Foo", "Bar"], "Baz", "ai_system.name") == "incorrect"


# ---------------------------------------------------------------------------
# BERTScore-style fields (substring path only — BERTScore call is slow)
# ---------------------------------------------------------------------------


class TestBertScoreFastPath:
    """affected_parties, event_location — exercise the substring shortcut."""

    def test_substring_match(self):
        assert compare_values("ChatGPT users", "ChatGPT users", "harm.affected_parties") == "correct"
        assert compare_values("users", "ChatGPT users", "harm.affected_parties") == "correct"

    def test_substring_normalization(self):
        # Case insensitive
        assert compare_values("FLORIDA", "Florida, United States", "event.event_location") == "correct"

    def test_list_extraction_per_element_substring(self):
        # bug_005 territory: a single matching list element should count.
        r = compare_values(["employees", "store operations"], "Andon Market employees", "harm.affected_parties")
        assert r == "correct"

    def test_empty_extraction_not_falsely_matched(self):
        # Must NOT return correct just because "" is a substring of GT.
        # bug_003 territory.
        assert compare_values([None], "Andon Market employees", "harm.affected_parties") in ("incorrect", "missing")
        assert compare_values([""], "users", "harm.affected_parties") in ("incorrect", "missing")
        assert compare_values(["   "], "general public", "harm.affected_parties") in ("incorrect", "missing")


# ---------------------------------------------------------------------------
# Excluded field (description) — never scored
# ---------------------------------------------------------------------------


class TestExcluded:
    def test_description_always_excluded(self):
        assert compare_values("anything", "anything else", "event.description") == "excluded"
        assert compare_values(None, "something", "event.description") == "excluded"


# ---------------------------------------------------------------------------
# Empty / missing / hallucinated dispatch
# ---------------------------------------------------------------------------


class TestEmptyHandling:
    def test_both_empty_is_correct(self):
        assert compare_values("not stated", "not stated", "ai_system.deployer") == "correct"
        assert compare_values(None, None, "ai_system.deployer") == "correct"
        assert compare_values("not stated", None, "ai_system.deployer") == "correct"

    def test_extracted_empty_when_gt_has_value_is_missing(self):
        assert compare_values("not stated", "OpenAI", "ai_system.deployer") == "missing"
        assert compare_values(None, "OpenAI", "ai_system.deployer") == "missing"
        assert compare_values("", "OpenAI", "ai_system.deployer") == "missing"

    def test_extracted_has_value_when_gt_empty_is_hallucinated(self):
        assert compare_values("OpenAI", "not stated", "ai_system.deployer") == "hallucinated"
        assert compare_values("OpenAI", None, "ai_system.deployer") == "hallucinated"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestNormalizeStr:
    """`_normalize_str` should NOT strip 'harm' suffix (that lives in
    `_strip_harm_suffix` which is scoped to constrained context only)."""

    def test_lowercases_and_strips(self):
        assert _normalize_str("  AI Incident  ") == "ai incident"

    def test_corp_suffixes_removed(self):
        assert _normalize_str("OpenAI, Inc.") == "openai"
        assert _normalize_str("Anthropic Corp.") == "anthropic"

    def test_does_not_strip_harm_suffix(self):
        # Was over-broad before; harm-strip is now scoped to _strip_harm_suffix
        assert _normalize_str("Physical harm") == "physical harm"
        assert _normalize_str("victims of harm") == "victims of harm"


class TestStripHarmSuffix:
    """`_strip_harm_suffix` is the per-element helper used by constrained
    field aggregation. End-anchored, so it strips trailing ' harm' only."""

    def test_strips_trailing_harm(self):
        assert _strip_harm_suffix("physical harm") == "physical"
        assert _strip_harm_suffix("economic harm") == "economic"

    def test_does_not_strip_middle_or_no_match(self):
        assert _strip_harm_suffix("physical") == "physical"
        assert _strip_harm_suffix("rights violation") == "rights violation"
