"""Tests for `_normalize_to_nested`.

This function bridges between the model's freely-chosen output schema
and the canonical nested GT structure. It is the most-iterated piece
of the evaluator (commits 0b01c2f, 067a242, a66c749, f229a14, 0f7f606
each fixed something here). These tests pin the contract so any
regression is caught immediately.
"""

from evaluation import _normalize_to_nested


# ---------------------------------------------------------------------------
# Pure shapes
# ---------------------------------------------------------------------------


class TestPureShapes:
    """The three canonical input shapes the evaluator must handle."""

    def test_pure_flat_promoted(self):
        flat = {"event_type": "AI incident", "harm_type": "economic"}
        out = _normalize_to_nested(flat)
        assert out["event"]["event_type"] == "AI incident"
        assert out["harm"]["harm_type"] == "economic"
        # Original flat keys must NOT remain at top level
        assert "event_type" not in out
        assert "harm_type" not in out

    def test_pure_nested_preserved(self):
        nested = {
            "event": {"event_type": "AI incident", "event_date": "2024-01-01"},
            "ai_system": {"name": "X"},
            "harm": {"harm_type": "physical"},
            "organizations": [{"name": "OpenAI", "role": "developer"}],
        }
        out = _normalize_to_nested(nested)
        assert out["event"]["event_type"] == "AI incident"
        assert out["ai_system"]["name"] == "X"
        assert out["organizations"] == [{"name": "OpenAI", "role": "developer"}]


# ---------------------------------------------------------------------------
# Mixed shape (the bug from f229a14)
# ---------------------------------------------------------------------------


class TestMixedShape:
    """Some PS3 outputs nest one group but leave others flat. The
    normalizer must merge: nested wins on conflict, flat fills gaps."""

    def test_flat_keys_lifted_into_nested(self):
        mixed = {
            "event": {"event_type": "AI incident"},
            "harm_type": "physical",   # flat extra
            "severity": "severe",      # flat extra
        }
        out = _normalize_to_nested(mixed)
        assert out["event"]["event_type"] == "AI incident"
        assert out["harm"]["harm_type"] == "physical"
        assert out["harm"]["severity"] == "severe"
        # Originals not at top level after merge
        assert "harm_type" not in out
        assert "severity" not in out

    def test_nested_wins_on_conflict(self):
        conflict = {
            "event": {"event_type": "AI hazard"},      # nested
            "event_type": "AI incident",                # flat duplicate (wrong)
        }
        out = _normalize_to_nested(conflict)
        assert out["event"]["event_type"] == "AI hazard"


# ---------------------------------------------------------------------------
# Inner-key renaming (a66c749, 0f7f606)
# ---------------------------------------------------------------------------


class TestInnerKeyRenaming:
    """Models use shorthand for inner keys; rename to canonical."""

    def test_event_type_short_form(self):
        out = _normalize_to_nested({"event": {"type": "AI incident", "date": "2024-01-01", "location": "x"}})
        assert out["event"]["event_type"] == "AI incident"
        assert out["event"]["event_date"] == "2024-01-01"
        assert out["event"]["event_location"] == "x"

    def test_ai_system_name_in_nested_block(self):
        # Haiku PS3 quirk (commit 0f7f606): emits ai_system_name inside ai_system block.
        out = _normalize_to_nested({"ai_system": {"ai_system_name": "Mythos", "system_type": "AI model"}})
        assert out["ai_system"]["name"] == "Mythos"
        assert "ai_system_name" not in out["ai_system"]

    def test_harm_type_short_form(self):
        out = _normalize_to_nested({"harm": {"type": "economic"}})
        assert out["harm"]["harm_type"] == "economic"


# ---------------------------------------------------------------------------
# Organizations: list, dict, other shapes
# ---------------------------------------------------------------------------


class TestOrganizations:
    """Models occasionally emit organizations as a single dict, sometimes
    using role-keyed or column-oriented shapes."""

    def test_list_passed_through(self):
        orgs = [{"name": "OpenAI", "role": "developer"}, {"name": "ACLU", "role": "other"}]
        out = _normalize_to_nested({"event": {}, "ai_system": {}, "harm": {}, "organizations": orgs})
        assert out["organizations"] == orgs

    def test_single_dict_coerced_to_list(self):
        # bug_006 territory.
        single = {"name": "Anthropic", "role": "developer"}
        out = _normalize_to_nested({"organizations": single})
        assert out["organizations"] == [single]

    def test_role_keyed_dict_dropped(self):
        # role-keyed dict: {developer: {...}, deployer: {...}} — has neither
        # 'name' nor 'role' as a key, so it's dropped (left as []).
        weird = {"developer": {"name": "X"}, "deployer": {"name": "Y"}}
        out = _normalize_to_nested({"organizations": weird})
        assert out["organizations"] == []

    def test_column_oriented_dict_coerced(self):
        # column-oriented {name: [...], role: [...]} HAS 'name' and 'role'
        # keys, so the coercion path treats it as a single-org dict and
        # wraps in a list. The list-typed values then surface as
        # mismatches in the per-org comparison — natural failure mode,
        # not a parse error.
        weird = {"name": ["X", "Y"], "role": ["a", "b"]}
        out = _normalize_to_nested({"organizations": weird})
        assert out["organizations"] == [weird]


# ---------------------------------------------------------------------------
# Defensive cases
# ---------------------------------------------------------------------------


class TestDefensive:
    """Should not crash on bare-JSON inputs from `parse_json_output`."""

    def test_none_returns_empty(self):
        assert _normalize_to_nested(None) == {}

    def test_non_dict_returns_empty(self):
        # bug_003: bare json.loads("123") shouldn't crash downstream.
        assert _normalize_to_nested(123) == {}
        assert _normalize_to_nested("AI incident") == {}
        assert _normalize_to_nested([1, 2, 3]) == {}
        assert _normalize_to_nested(True) == {}

    def test_empty_dict_returns_baseline_structure(self):
        out = _normalize_to_nested({})
        assert out == {"event": {}, "ai_system": {}, "harm": {}, "organizations": []}

    def test_unknown_top_level_keys_preserved(self):
        # Schema-extra keys (e.g., 'incidents', 'underlying_model') must
        # surface so downstream marks them as hallucinated structure.
        out = _normalize_to_nested({"event_type": "AI incident", "culprit": "Tesla"})
        assert out.get("culprit") == "Tesla"
