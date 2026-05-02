"""Tests for `parse_json_output`.

The function must be tolerant of model output formats: bare JSON,
markdown code blocks, "FINAL JSON:" prefixes, and free-text wrappers.
It must also be safe — never crash on garbage.
"""

import pytest
from evaluation import parse_json_output


class TestPlainJSON:
    def test_simple_object(self):
        parsed, valid = parse_json_output('{"a": 1, "b": "x"}')
        assert valid is True
        assert parsed == {"a": 1, "b": "x"}

    def test_nested_object(self):
        parsed, valid = parse_json_output('{"event": {"event_type": "AI incident"}}')
        assert valid is True
        assert parsed == {"event": {"event_type": "AI incident"}}


class TestCodeBlocks:
    def test_json_code_block(self):
        text = '```json\n{"a": 1}\n```'
        parsed, valid = parse_json_output(text)
        assert valid is True
        assert parsed == {"a": 1}

    def test_generic_code_block(self):
        text = '```\n{"a": 1}\n```'
        parsed, valid = parse_json_output(text)
        assert valid is True
        assert parsed == {"a": 1}

    def test_final_json_prefix(self):
        text = 'After verification, here is the final answer:\nFINAL JSON: {"a": 1}'
        parsed, valid = parse_json_output(text)
        assert valid is True
        assert parsed == {"a": 1}


class TestEdgeCases:
    def test_empty_string_invalid(self):
        parsed, valid = parse_json_output("")
        assert valid is False
        assert parsed is None

    def test_whitespace_only_invalid(self):
        parsed, valid = parse_json_output("   \n\t   ")
        assert valid is False
        assert parsed is None

    def test_invalid_json_returns_none(self):
        parsed, valid = parse_json_output('not actually json {{ }}')
        assert valid is False
        assert parsed is None

    def test_json_inside_prose(self):
        # The fallback regex extracts the first {...} block.
        parsed, valid = parse_json_output('Here is the result: {"a": 1} thanks!')
        assert valid is True
        assert parsed == {"a": 1}

    def test_bare_literal_returns_value_not_dict(self):
        # bug_003: parse_json_output returns valid=True for json.loads("123").
        # The caller (evaluate_extraction) is responsible for handling non-dict inputs.
        parsed, valid = parse_json_output("123")
        assert valid is True
        assert parsed == 123

        parsed, valid = parse_json_output('"AI incident"')
        assert valid is True
        assert parsed == "AI incident"

        parsed, valid = parse_json_output('[1, 2, 3]')
        assert valid is True
        assert parsed == [1, 2, 3]
