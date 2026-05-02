# Evaluation test suite

72 unit tests guarding `src/evaluation.py` against the regressions the
project has historically been bitten by. Run from the project root:

```bash
source .venv/bin/activate
pytest tests/
```

Expected: 72 passed in ~4s. The first run loads BERTScore (slow);
subsequent runs are fast because BERTScore is only invoked once.

## File layout

| File | What it covers |
|------|----------------|
| `test_compare_values.py` | The leaf-level field comparator across all 4 branches (constrained, exact, BERTScore-fast-path, excluded) plus list-typed handling, harm-suffix stripping, and empty-value dispatch |
| `test_normalize_to_nested.py` | The flat→nested translator. Pins all five shape cases (pure flat, pure nested, mixed, organizations-as-dict, defensive non-dict input) and inner-key renaming (e.g., `ai_system_name → name`) |
| `test_evaluate_extraction.py` | End-to-end per-incident scoring: nested traversal, organization best-match pairing, schema-extra hallucination tracking |
| `test_aggregation.py` | `BenchmarkMetrics` and `calculate_metrics`: pure-hallucination row exclusion, macro vs micro accuracy, F1 algebra, empty/None parsed_output handling |
| `test_parse_json_output.py` | JSON extraction from raw model output: bare JSON, code blocks, FINAL JSON: prefix, edge cases |

## What each test class is guarding against

The test names directly reference the bug they prevent. Search the
git log for the bug ID (e.g., `git log --all --grep "bug_006"`) to
see when it was originally surfaced.

- `bug_003` — bare-literal `parsed_output` (`json.loads("123")`)
  crashing `_normalize_to_nested`
- `bug_005` — BERTScore branch returning "correct" for `[None]` /
  `[""]` because empty string is a substring of every GT value
- `bug_006` — single-dict org coerced; empty-GT-orgs branch must
  still see hallucinated extractions
- `bug_008` — empty `{}` parsed_output silently dropped from
  per-field totals
- `bug_011` — exact-match branch crashed on list-typed extractions
- `merged_bug_002` — over-broad harm-suffix strip in `_normalize_str`
- Date/Country/few-shot pipeline fixes — covered by integration via
  the `evaluate_extraction` and `compare_values` tests

## How to add a new test when a bug is fixed

1. Reproduce the failure as a minimal `compare_values(...)` or
   `evaluate_extraction(...)` call inside a `def test_<bugname>():`
2. Add it to the relevant file under the closest existing class
3. Run `pytest -k test_<bugname>` to confirm it fails first
4. Apply the fix in `src/evaluation.py`
5. Re-run the test to confirm it passes
6. Commit both the fix and the test in the same commit, with the
   commit message referencing the bug

Tests are deliberately **regression-style** rather than coverage-
maximizing: each test exists because someone (often you) found a real
bug and wanted to make sure it doesn't come back.
