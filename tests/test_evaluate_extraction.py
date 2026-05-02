"""Tests for `evaluate_extraction` — the per-incident verdict assembler.

Focuses on integration-level behavior: nested traversal, organization
matching, the description-excluded rule, hallucinated-org detection.
"""

from evaluation import evaluate_extraction


def _full_gt(**overrides):
    """Default GT skeleton; override specific fields per test."""
    base = {
        "event": {
            "event_type": "AI incident",
            "event_date": "2024-01-01",
            "event_location": "United States",
            "description": "x",
        },
        "ai_system": {
            "name": "X",
            "system_type": "chatbot",
            "developer": "X",
            "deployer": "X",
        },
        "harm": {
            "harm_type": "physical",
            "severity": "minor",
            "affected_parties": "users",
        },
        "organizations": [{"name": "X", "role": "developer"}],
    }
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Nested traversal
# ---------------------------------------------------------------------------


class TestNestedTraversal:
    def test_perfect_match(self):
        gt = _full_gt()
        ext = _full_gt()
        results = evaluate_extraction(ext, gt)
        # All fields except description should be 'correct'
        for k, v in results.items():
            assert v == "correct", f"{k} should be correct, got {v}"

    def test_description_excluded(self):
        gt = _full_gt()
        ext = _full_gt()
        ext["event"]["description"] = "completely different"
        results = evaluate_extraction(ext, gt)
        # description must NOT appear in results
        assert "event.description" not in results
        # ... and all other fields should still be correct.
        for k, v in results.items():
            assert v == "correct", f"{k} got {v}"

    def test_partial_match(self):
        gt = _full_gt()
        ext = _full_gt()
        ext["harm"]["harm_type"] = "economic"  # wrong
        results = evaluate_extraction(ext, gt)
        assert results["harm.harm_type"] == "incorrect"
        assert results["event.event_type"] == "correct"


# ---------------------------------------------------------------------------
# Flat output handling
# ---------------------------------------------------------------------------


class TestFlatOutput:
    def test_flat_extraction_normalized(self):
        gt = _full_gt()
        flat_ext = {
            "event_type": "AI incident",
            "event_date": "2024-01-01",
            "event_location": "United States",
            "description": "x",
            "ai_system_name": "X",
            "system_type": "chatbot",
            "developer": "X",
            "deployer": "X",
            "harm_type": "physical",
            "severity": "minor",
            "affected_parties": "users",
            "organizations": [{"name": "X", "role": "developer"}],
        }
        results = evaluate_extraction(flat_ext, gt)
        # All real fields should be correct; no flat duplicates remaining.
        for k in ("event.event_type", "ai_system.name", "harm.harm_type"):
            assert results[k] == "correct"
        for k in ("event_type", "ai_system_name", "harm_type"):
            assert k not in results

    def test_bare_json_input_safe(self):
        # bug_003: bare-literal parsed_output must not crash.
        gt = _full_gt()
        for bare in (123, "AI incident", [1, 2, 3], True, None):
            results = evaluate_extraction(bare, gt)
            # All real fields should be missing (model produced nothing useful).
            assert results.get("event.event_type") in ("missing", None)


# ---------------------------------------------------------------------------
# Organizations matching
# ---------------------------------------------------------------------------


class TestOrganizationsMatching:
    def test_positional_match_when_names_align(self):
        gt = _full_gt(organizations=[{"name": "OpenAI", "role": "developer"}, {"name": "ACLU", "role": "other"}])
        ext = _full_gt(organizations=[{"name": "OpenAI", "role": "developer"}, {"name": "ACLU", "role": "other"}])
        results = evaluate_extraction(ext, gt)
        assert results["organizations[0].name"] == "correct"
        assert results["organizations[0].role"] == "correct"
        assert results["organizations[1].name"] == "correct"
        assert results["organizations[1].role"] == "correct"

    def test_best_match_handles_reordering(self):
        gt = _full_gt(organizations=[{"name": "OpenAI", "role": "developer"}, {"name": "ACLU", "role": "other"}])
        # Reordered: should still match by name similarity.
        ext = _full_gt(organizations=[{"name": "ACLU", "role": "other"}, {"name": "OpenAI", "role": "developer"}])
        results = evaluate_extraction(ext, gt)
        assert results["organizations[0].name"] == "correct"
        assert results["organizations[0].role"] == "correct"

    def test_extra_org_marked_hallucinated(self):
        gt = _full_gt(organizations=[{"name": "OpenAI", "role": "developer"}])
        ext = _full_gt(organizations=[
            {"name": "OpenAI", "role": "developer"},
            {"name": "Hallucinated Co", "role": "deployer"},
        ])
        results = evaluate_extraction(ext, gt)
        # The extra org should appear as organizations[hN].
        hkeys = [k for k in results if k.startswith("organizations[h")]
        assert any("name" in k for k in hkeys)
        assert all(results[k] == "hallucinated" for k in hkeys)

    def test_empty_gt_orgs_with_extracted_orgs_counted_as_hallucinated(self):
        # bug_006: empty GT orgs branch must still see hallucinated extractions.
        gt = _full_gt(organizations=[])
        ext = _full_gt(organizations=[
            {"name": "Tesla", "role": "developer"},
            {"name": "ACLU", "role": "other"},
        ])
        results = evaluate_extraction(ext, gt)
        hkeys = [k for k in results if k.startswith("organizations[h")]
        # 2 extra orgs × 2 fields each = 4 hallucinated rows
        assert len(hkeys) == 4
        assert all(results[k] == "hallucinated" for k in hkeys)

    def test_single_dict_org_coerced_and_matched(self):
        # bug_006: single dict org should be coerced to [dict] and then matched.
        gt = _full_gt(organizations=[{"name": "Anthropic", "role": "developer"}])
        ext = _full_gt(organizations={"name": "Anthropic", "role": "developer"})
        results = evaluate_extraction(ext, gt)
        assert results["organizations[0].name"] == "correct"
        assert results["organizations[0].role"] == "correct"


# ---------------------------------------------------------------------------
# Hallucination tracking: schema-extra fields
# ---------------------------------------------------------------------------


class TestSchemaExtras:
    def test_extra_top_level_field_marked_hallucinated(self):
        gt = _full_gt()
        ext = _full_gt()
        ext["culprit"] = "Tesla"  # not in schema
        results = evaluate_extraction(ext, gt)
        assert results.get("culprit") == "hallucinated"

    def test_extra_nested_field_marked_hallucinated(self):
        # Haiku PS3 quirk: invents *_quote fields under nested groups.
        gt = _full_gt()
        ext = _full_gt()
        ext["event"]["event_date_quote"] = "the article said 2024-01-01"
        results = evaluate_extraction(ext, gt)
        assert results.get("event.event_date_quote") == "hallucinated"
