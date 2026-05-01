"""
Re-evaluate saved experiment results against the current evaluation logic.

Each `*_results.json` file already stores `parsed_output` and `ground_truth`
per incident, so re-scoring requires no LLM calls. This script walks the
results directory, recomputes BenchmarkMetrics for every condition, and
overwrites `*_metrics.json` in place.

Usage (from the project root):
    source .venv/bin/activate
    python src/reevaluate.py                         # all runs under data/results
    python src/reevaluate.py path/to/single/run_dir  # one run only

A backup of each overwritten metrics file is written next to it as
`*_metrics.json.bak` on the first re-evaluation, so the original numbers
remain inspectable.
"""

import json
import sys
from pathlib import Path

try:
    from evaluation import (
        ExtractionResult,
        calculate_metrics,
    )
except ImportError:  # pragma: no cover - fallback when run as a module
    from .evaluation import (
        ExtractionResult,
        calculate_metrics,
    )


def _result_from_dict(d: dict) -> ExtractionResult:
    """Reconstruct an ExtractionResult from a saved JSON record."""
    return ExtractionResult(
        incident_id=d.get("incident_id", ""),
        model=d.get("model", ""),
        template=d.get("condition", d.get("template", "")),
        raw_output=d.get("raw_output", ""),
        parsed_output=d.get("parsed_output"),
        ground_truth=d.get("ground_truth", {}),
        is_valid_json=d.get("is_valid_json", False),
        latency_seconds=d.get("latency_seconds", 0.0),
        error=d.get("error"),
    )


def reevaluate_condition(results_path: Path) -> dict:
    """Recompute metrics for one `*_results.json` file. Returns the new dict."""
    with results_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    results = [_result_from_dict(r) for r in records]
    metrics = calculate_metrics(results)

    # Preserve the model/condition labels from the saved records.
    if records:
        metrics.model = records[0].get("model", metrics.model)
        metrics.template = records[0].get("condition", metrics.template)

    metrics_path = results_path.with_name(
        results_path.name.replace("_results.json", "_metrics.json")
    )
    backup_path = metrics_path.with_suffix(metrics_path.suffix + ".bak")
    if metrics_path.exists() and not backup_path.exists():
        backup_path.write_bytes(metrics_path.read_bytes())

    metrics_dict = metrics.to_dict()
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics_dict


def reevaluate_run_dir(run_dir: Path) -> list[dict]:
    """Re-evaluate every condition under a single run directory."""
    out = []
    for results_path in sorted(run_dir.glob("*_results.json")):
        m = reevaluate_condition(results_path)
        hall = m["overall_hallucination_rate"]
        print(
            f"  {results_path.parent.name}/{results_path.stem}: "
            f"acc={m['overall_accuracy']:.1%} "
            f"f1={m['overall_f1']:.3f} "
            f"hall={hall:.1%}"
        )
        out.append(m)
    return out


def reevaluate_all(root: Path) -> list[dict]:
    """Walk every subdirectory of `root` and re-evaluate found result files."""
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return []

    all_metrics: list[dict] = []
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not run_dirs:
        # Maybe the user passed a leaf run dir directly.
        return reevaluate_run_dir(root)

    for run_dir in run_dirs:
        results_files = list(run_dir.glob("*_results.json"))
        if not results_files:
            continue
        print(f"\n{run_dir.name}/")
        all_metrics.extend(reevaluate_run_dir(run_dir))

    return all_metrics


def main():
    project_root = Path(__file__).parent.parent
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = project_root / "data" / "results"

    print(f"Re-evaluating saved results under: {target}")
    metrics = reevaluate_all(target)
    print(f"\nRe-evaluated {len(metrics)} condition(s).")


if __name__ == "__main__":
    main()
