"""Tests for `BenchmarkMetrics` and `calculate_metrics` aggregation.

The aggregation logic has been the source of multiple subtle bugs:
  - bug_006: hallucinated-org pseudo-fields dragging accuracy avg
  - bug_008: empty-{} parsed_output silently dropped from totals
  - generalized fix: any pure-hallucination row excluded from accuracy
  - micro-vs-macro: small-n org cells distorting macro means

These tests pin the contract on both metrics.
"""

import pytest

from evaluation import (
    BenchmarkMetrics,
    ExtractionResult,
    FieldMetrics,
    calculate_metrics,
)


def _make_field(correct=0, incorrect=0, missing=0, hallucinated=0):
    return FieldMetrics(
        field_name="x",
        correct=correct,
        incorrect=incorrect,
        missing_in_extraction=missing,
        hallucinated=hallucinated,
        total=correct + incorrect + missing + hallucinated,
    )


# ---------------------------------------------------------------------------
# Pure-hallucination row exclusion (the generalized filter)
# ---------------------------------------------------------------------------


class TestPureHallucinationFiltering:
    def test_org_pseudo_field_excluded_from_accuracy(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=2)
        bm.field_metrics = {
            "harm.harm_type": _make_field(correct=2),
            "ai_system.name": _make_field(correct=2),
            "organizations[h0].name": _make_field(hallucinated=3),
            "organizations[h0].role": _make_field(hallucinated=3),
        }
        # 2 real fields at 100% — accuracy should be 100%, not dragged.
        assert bm.overall_accuracy == pytest.approx(1.0)

    def test_schema_extra_field_excluded_from_accuracy(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=2)
        bm.field_metrics = {
            "harm.harm_type": _make_field(correct=2),
            # Haiku PS3 invents these — they have NO GT signal at all.
            "event.event_date_quote": _make_field(hallucinated=2),
            "ai_system.underlying_model": _make_field(hallucinated=1),
            "harm.scale": _make_field(hallucinated=1),
        }
        # Only the real field counts.
        assert bm.overall_accuracy == pytest.approx(1.0)

    def test_pure_hallucination_rows_still_in_hallucination_rate(self):
        # The hallucination signal must NOT be lost — that's where it lives.
        bm = BenchmarkMetrics(model="m", template="t", total_samples=2)
        bm.field_metrics = {
            "harm.harm_type": _make_field(correct=2),               # 2/2
            "organizations[h0].name": _make_field(hallucinated=3),  # 3/3 hall
            "organizations[h0].role": _make_field(hallucinated=3),  # 3/3 hall
            "event.event_date_quote": _make_field(hallucinated=2),  # 2/2 hall
            "ai_system.underlying_model": _make_field(hallucinated=1),
        }
        # total_extracted = 2 + 3 + 3 + 2 + 1 = 11
        # total_hallucinated = 3 + 3 + 2 + 1 = 9
        # rate = 9/11 ≈ 0.818
        assert bm.overall_hallucination_rate == pytest.approx(9 / 11)


# ---------------------------------------------------------------------------
# Macro vs micro accuracy
# ---------------------------------------------------------------------------


class TestMacroVsMicro:
    """Small-n cells are weighted equally with high-n cells under macro.
    Micro is sample-weighted (total correct / total scored)."""

    def test_macro_treats_fields_equally(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=50)
        bm.field_metrics = {
            "high_n":  _make_field(correct=25, incorrect=25),    # 50% on n=50
            "small_n": _make_field(correct=1),                    # 100% on n=1
        }
        # Macro = (0.5 + 1.0) / 2 = 0.75
        assert bm.overall_accuracy == pytest.approx(0.75)

    def test_micro_weights_by_sample_count(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=50)
        bm.field_metrics = {
            "high_n":  _make_field(correct=25, incorrect=25),
            "small_n": _make_field(correct=1),
        }
        # Micro = (25 + 1) / (50 + 1) = 26/51 ≈ 0.510
        assert bm.overall_accuracy_micro == pytest.approx(26 / 51)

    def test_both_metrics_exclude_pure_halluc_rows(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=50)
        bm.field_metrics = {
            "real":  _make_field(correct=4, incorrect=1),
            "halluc_only": _make_field(hallucinated=10),
        }
        assert bm.overall_accuracy == pytest.approx(4/5)         # macro
        assert bm.overall_accuracy_micro == pytest.approx(4/5)   # micro


# ---------------------------------------------------------------------------
# F1 algebra (per-field property)
# ---------------------------------------------------------------------------


class TestF1Algebra:
    def test_per_field_f1_consistent(self):
        f = _make_field(correct=8, incorrect=1, missing=1)
        # precision = 8/(8+1+0) = 8/9; recall = 8/(8+1) = 8/9; F1 = 2pr/(p+r)
        assert f.precision == pytest.approx(8 / 9)
        assert f.recall == pytest.approx(8 / 9)
        assert f.f1_score == pytest.approx(2 * (8/9) * (8/9) / (8/9 + 8/9))

    def test_zero_division_safe(self):
        f = _make_field()  # all zero
        assert f.accuracy == 0.0
        assert f.precision == 0.0
        assert f.recall == 0.0
        assert f.f1_score == 0.0


# ---------------------------------------------------------------------------
# calculate_metrics: full pipeline aggregation
# ---------------------------------------------------------------------------


def _result(parsed_output, gt, incident_id="x"):
    return ExtractionResult(
        incident_id=incident_id,
        model="m",
        template="t",
        raw_output="",
        parsed_output=parsed_output,
        ground_truth=gt,
        is_valid_json=True,
        latency_seconds=0.0,
    )


class TestCalculateMetrics:
    def test_empty_dict_parsed_output_counted_as_missing(self):
        # bug_008: empty {} was being silently dropped from per-field totals.
        gt = {
            "event": {"event_type": "AI incident"},
            "ai_system": {}, "harm": {}, "organizations": [],
        }
        results = [
            _result({"event": {"event_type": "AI incident"}}, gt, "a"),
            _result({}, gt, "b"),                               # empty dict
            _result({"event": {"event_type": "AI incident"}}, gt, "c"),
        ]
        m = calculate_metrics(results)
        # event.event_type should be counted on all 3 records.
        ev = m.field_metrics["event.event_type"]
        assert ev.total == 3
        assert ev.missing_in_extraction == 1
        assert ev.correct == 2

    def test_none_parsed_output_counted_as_missing(self):
        gt = {"event": {"event_type": "AI incident"}, "ai_system": {}, "harm": {}, "organizations": []}
        results = [
            _result(None, gt, "a"),
            _result({"event": {"event_type": "AI incident"}}, gt, "b"),
        ]
        m = calculate_metrics(results)
        ev = m.field_metrics["event.event_type"]
        assert ev.total == 2

    def test_total_samples_counted(self):
        gt = {"event": {"event_type": "AI incident"}, "ai_system": {}, "harm": {}, "organizations": []}
        results = [_result({"event": {"event_type": "AI incident"}}, gt, str(i)) for i in range(5)]
        m = calculate_metrics(results)
        assert m.total_samples == 5
