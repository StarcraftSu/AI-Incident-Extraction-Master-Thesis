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

    def test_hallucination_rate_uses_in_scope_fields_only(self):
        # After the orgs-as-sub-task change, hallucination_rate is
        # computed only over fields that contribute to accuracy
        # aggregation. Org-pseudo-fields, schema-extras, and the
        # organizations array proper are all excluded.
        bm = BenchmarkMetrics(model="m", template="t", total_samples=2)
        bm.field_metrics = {
            "ai_system.developer": _make_field(correct=8, hallucinated=2),  # in-scope, 2 halluc
            "harm.harm_type": _make_field(correct=10),                       # in-scope, 0 halluc
            "organizations[h0].name": _make_field(hallucinated=3),           # excluded (org)
            "event.event_date_quote": _make_field(hallucinated=2),           # excluded (pure-halluc)
        }
        # Only the in-scope rows contribute:
        # total_extracted = (8+0+2) + (10+0+0) = 20
        # total_hallucinated = 2 + 0 = 2
        # rate = 2/20 = 0.1
        assert bm.overall_hallucination_rate == pytest.approx(0.1)


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


class TestMicroFamily:
    """Micro precision/recall/F1: the global-counts variants paired with
    micro accuracy. Added to address the metric-aggregation
    inconsistency flagged in the peer review of Chapter 4."""

    def test_micro_precision_recall_consistent(self):
        # Construct a 2-field example with known global TP/FP/FN.
        # Field A: 8 correct, 2 incorrect, 0 missing → contribute TP=8, FP=2
        # Field B: 6 correct, 1 incorrect, 3 missing → contribute TP=6, FP=1, FN=3
        # Global TP = 14, FP = 3, FN = 3
        # Precision_micro = 14/17, Recall_micro = 14/17, F1_micro = 14/17
        bm = BenchmarkMetrics(model="m", template="t", total_samples=10)
        bm.field_metrics = {
            "A": _make_field(correct=8, incorrect=2, missing=0),
            "B": _make_field(correct=6, incorrect=1, missing=3),
        }
        assert bm.overall_precision_micro == pytest.approx(14/17)
        assert bm.overall_recall_micro == pytest.approx(14/17)
        assert bm.overall_f1_micro == pytest.approx(14/17)

    def test_micro_f1_harmonic_mean(self):
        # Asymmetric P and R: TP=10, FP=10, FN=0 → P=0.5, R=1.0, F1=2*0.5*1/(1.5)=2/3
        bm = BenchmarkMetrics(model="m", template="t", total_samples=10)
        bm.field_metrics = {
            "A": _make_field(correct=10, incorrect=10, missing=0),
        }
        assert bm.overall_precision_micro == pytest.approx(0.5)
        assert bm.overall_recall_micro == pytest.approx(1.0)
        assert bm.overall_f1_micro == pytest.approx(2/3)

    def test_micro_excludes_pure_halluc_rows(self):
        # Pure-hallucination rows excluded from accuracy aggregation.
        # Plus: as of the organizations-as-sub-task change, ALL
        # organizations[*] rows are excluded from both accuracy AND
        # hallucination_rate (they are now reported separately).
        bm = BenchmarkMetrics(model="m", template="t", total_samples=10)
        bm.field_metrics = {
            "real": _make_field(correct=8, incorrect=2),
            "organizations[h0].name": _make_field(hallucinated=5),
        }
        # Micro should compute over only the real row.
        assert bm.overall_precision_micro == pytest.approx(8/10)
        assert bm.overall_recall_micro == pytest.approx(1.0)
        # Hallucination_rate ALSO excludes organizations now.
        # Only "real" is in scope: total_extracted = 10, halluc = 0.
        assert bm.overall_hallucination_rate == pytest.approx(0.0)


class TestOrganizationsExcludedFromHeadline:
    """Organizations array is treated as a separate sub-task. All
    `organizations*` rows are excluded from accuracy/precision/recall/
    F1/hallucination_rate, but their per-cell data is preserved in
    field_metrics for supplementary analysis."""

    def test_organizations_data_preserved_but_excluded_from_aggregation(self):
        bm = BenchmarkMetrics(model="m", template="t", total_samples=10)
        bm.field_metrics = {
            "event.event_type": _make_field(correct=10),  # 10/10
            "organizations[0].name": _make_field(correct=2, incorrect=8),  # 2/10 — would drag mean if included
            "organizations[0].role": _make_field(hallucinated=8),  # would inflate halluc if included
            "organizations[h0].name": _make_field(hallucinated=3),
        }
        # All headline metrics see only event.event_type (perfect).
        assert bm.overall_accuracy == pytest.approx(1.0)
        assert bm.overall_accuracy_micro == pytest.approx(1.0)
        assert bm.overall_f1_micro == pytest.approx(1.0)
        # Hallucination on the in-scope fields is 0 (no halluc on event.event_type).
        assert bm.overall_hallucination_rate == pytest.approx(0.0)
        # But the org rows are still in field_metrics — accessible for
        # supplementary analysis.
        assert "organizations[0].name" in bm.field_metrics
        assert "organizations[0].role" in bm.field_metrics
        assert bm.field_metrics["organizations[0].role"].hallucinated == 8


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
