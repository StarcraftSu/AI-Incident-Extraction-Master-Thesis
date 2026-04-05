"""
Evaluation metrics for comparing LLM extractions against ground truth.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractionResult:
    """Result of a single extraction attempt."""
    incident_id: str
    model: str
    template: str
    raw_output: str
    parsed_output: Optional[dict]
    ground_truth: dict
    is_valid_json: bool
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class FieldMetrics:
    """Metrics for a single field."""
    field_name: str
    correct: int = 0
    incorrect: int = 0
    missing_in_extraction: int = 0  # In ground truth but not extracted
    hallucinated: int = 0  # In extraction but not in ground truth
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        extracted = self.correct + self.incorrect + self.hallucinated
        return self.correct / extracted if extracted > 0 else 0.0

    @property
    def recall(self) -> float:
        relevant = self.correct + self.incorrect + self.missing_in_extraction
        return self.correct / relevant if relevant > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    model: str
    template: str
    total_samples: int = 0
    valid_json_count: int = 0
    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)
    total_latency: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def json_validity_rate(self) -> float:
        return self.valid_json_count / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def overall_accuracy(self) -> float:
        if not self.field_metrics:
            return 0.0
        return sum(m.accuracy for m in self.field_metrics.values()) / len(self.field_metrics)

    @property
    def overall_f1(self) -> float:
        if not self.field_metrics:
            return 0.0
        return sum(m.f1_score for m in self.field_metrics.values()) / len(self.field_metrics)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "template": self.template,
            "total_samples": self.total_samples,
            "valid_json_count": self.valid_json_count,
            "json_validity_rate": self.json_validity_rate,
            "overall_accuracy": self.overall_accuracy,
            "overall_f1": self.overall_f1,
            "avg_latency_seconds": self.avg_latency,
            "field_metrics": {
                name: {
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "correct": m.correct,
                    "incorrect": m.incorrect,
                    "missing": m.missing_in_extraction,
                    "hallucinated": m.hallucinated,
                }
                for name, m in self.field_metrics.items()
            },
            "error_count": len(self.errors),
        }


def parse_json_output(raw_output: str) -> tuple[Optional[dict], bool]:
    """
    Parse JSON from LLM output, handling common issues.
    Returns (parsed_dict, is_valid).
    """
    if not raw_output or not raw_output.strip():
        return None, False

    text = raw_output.strip()

    # Try direct parse first
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",       # ``` ... ```
        r"FINAL JSON:\s*([\s\S]*?)$",    # For chain-of-verification
        r"\{[\s\S]*\}",                   # Raw JSON object
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str), True
            except (json.JSONDecodeError, IndexError):
                continue

    return None, False


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison.

    Treats null, None, "null", "not stated", "n/a", and empty string
    as equivalent empty values.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.lower().strip()
        # Treat these as equivalent to None (empty/unknown)
        if stripped in ("null", "not stated", "n/a", "none", "unknown", ""):
            return None
        return stripped
    if isinstance(value, (int, float)):
        return str(value).lower().strip()
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        normalized = [normalize_value(v) for v in value]
        # Sort only if list contains sortable items (not dicts)
        if normalized and isinstance(normalized[0], dict):
            return normalized  # Don't sort list of dicts
        try:
            return sorted(normalized)
        except TypeError:
            return normalized  # Return unsorted if comparison fails
    return value


def compare_values(extracted: Any, ground_truth: Any) -> str:
    """
    Compare extracted value with ground truth.
    Returns: 'correct', 'incorrect', 'missing', or 'hallucinated'
    """
    ext_norm = normalize_value(extracted)
    gt_norm = normalize_value(ground_truth)

    # Both null/empty
    if ext_norm in (None, "", []) and gt_norm in (None, "", []):
        return "correct"

    # Ground truth exists but extraction is missing
    if ext_norm in (None, "", []) and gt_norm not in (None, "", []):
        return "missing"

    # Extraction exists but ground truth is missing (hallucination)
    if ext_norm not in (None, "", []) and gt_norm in (None, "", []):
        return "hallucinated"

    # Both exist - compare
    if ext_norm == gt_norm:
        return "correct"

    # For strings, check partial match
    if isinstance(ext_norm, str) and isinstance(gt_norm, str):
        if ext_norm in gt_norm or gt_norm in ext_norm:
            return "correct"  # Partial match is acceptable

    # For lists, check overlap
    if isinstance(ext_norm, list) and isinstance(gt_norm, list):
        ext_set = set(str(x) for x in ext_norm)
        gt_set = set(str(x) for x in gt_norm)
        if ext_set & gt_set:  # Any overlap
            return "correct"

    return "incorrect"


def evaluate_extraction(
    extracted: dict,
    ground_truth: dict,
    prefix: str = ""
) -> dict[str, str]:
    """
    Recursively evaluate extraction against ground truth.
    Returns dict mapping field paths to comparison results.
    """
    results = {}

    # Get all keys from both dicts
    all_keys = set(ground_truth.keys()) | set(extracted.keys() if extracted else [])

    for key in all_keys:
        field_path = f"{prefix}.{key}" if prefix else key
        gt_value = ground_truth.get(key)
        ext_value = extracted.get(key) if extracted else None

        if isinstance(gt_value, dict) and not isinstance(gt_value, list):
            # Recurse into nested dict
            nested_results = evaluate_extraction(
                ext_value if isinstance(ext_value, dict) else {},
                gt_value,
                field_path
            )
            results.update(nested_results)
        else:
            # Compare leaf values
            results[field_path] = compare_values(ext_value, gt_value)

    return results


def calculate_metrics(results: list[ExtractionResult]) -> BenchmarkMetrics:
    """Calculate aggregated metrics from a list of extraction results."""
    if not results:
        return BenchmarkMetrics(model="", template="")

    metrics = BenchmarkMetrics(
        model=results[0].model,
        template=results[0].template,
        total_samples=len(results),
    )

    for result in results:
        # Count valid JSON
        if result.is_valid_json:
            metrics.valid_json_count += 1

        # Add latency
        metrics.total_latency += result.latency_seconds

        # Track errors
        if result.error:
            metrics.errors.append(f"{result.incident_id}: {result.error}")

        # Evaluate fields if we have valid output
        if result.parsed_output and result.ground_truth:
            field_results = evaluate_extraction(result.parsed_output, result.ground_truth)

            for field_path, comparison in field_results.items():
                if field_path not in metrics.field_metrics:
                    metrics.field_metrics[field_path] = FieldMetrics(field_name=field_path)

                fm = metrics.field_metrics[field_path]
                fm.total += 1

                if comparison == "correct":
                    fm.correct += 1
                elif comparison == "incorrect":
                    fm.incorrect += 1
                elif comparison == "missing":
                    fm.missing_in_extraction += 1
                elif comparison == "hallucinated":
                    fm.hallucinated += 1

    return metrics


def print_metrics_report(metrics: BenchmarkMetrics):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {metrics.model} + {metrics.template}")
    print("=" * 60)

    print(f"\nOverall Statistics:")
    print(f"  Samples:          {metrics.total_samples}")
    print(f"  JSON Validity:    {metrics.json_validity_rate:.1%}")
    print(f"  Overall Accuracy: {metrics.overall_accuracy:.1%}")
    print(f"  Overall F1:       {metrics.overall_f1:.3f}")
    print(f"  Avg Latency:      {metrics.avg_latency:.2f}s")

    print(f"\nField-Level Metrics:")
    print(f"  {'Field':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for name, fm in sorted(metrics.field_metrics.items()):
        print(f"  {name:<30} {fm.accuracy:>5.1%} {fm.precision:>5.1%} {fm.recall:>5.1%} {fm.f1_score:>5.3f}")

    if metrics.errors:
        print(f"\nErrors ({len(metrics.errors)}):")
        for err in metrics.errors[:5]:  # Show first 5
            print(f"  - {err[:80]}...")


# Quick test
if __name__ == "__main__":
    # Test JSON parsing
    test_outputs = [
        '{"event": {"type": "malfunction"}}',
        '```json\n{"event": {"type": "bias"}}\n```',
        'Here is the extraction:\n\n{"event": {"type": "test"}}',
        'FINAL JSON:\n{"event": {"type": "verified"}}',
        'invalid json {{{',
    ]

    print("Testing JSON parsing:")
    for output in test_outputs:
        parsed, valid = parse_json_output(output)
        print(f"  Valid: {valid}, Parsed: {parsed}")
