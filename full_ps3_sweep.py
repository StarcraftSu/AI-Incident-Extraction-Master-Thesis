"""Full PS3 sweep on Llama 3.1 8B after the CoVe fix.

Runs all 4 KI conditions (PS3_KI1..PS3_KI4) on the full 50-incident
dataset. Expected runtime ~60-90 minutes (50 incidents x 4 conditions
x 2 LLM calls per incident, ~10-30s per call on Llama).

Pre-fix baseline (the numbers in the current results table):
  PS3_KI1: 36.4%   PS3_KI2: 38.0%   PS3_KI3: 38.0%   PS3_KI4: 37.8%

If the fix works, expect a substantial jump on at least KI3/KI4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_dataset_with_ground_truth
from experiment import ExperimentRunner

project_root = Path(__file__).parent
dataset = load_dataset_with_ground_truth(
    str(project_root / "data/raw/experimental_incidents_50.xlsx"),
    str(project_root / "data/annotated/ground_truth_50.json"),
)
print(f"Dataset: {len(dataset)} incidents")
print("Sweep: Llama 3.1 8B x PS3 x [KI1, KI2, KI3, KI4]")

runner = ExperimentRunner(output_dir=str(project_root / "data/results"))
metrics = runner.run_benchmark(
    dataset=dataset,
    model_keys=["llama3.1:8b"],
    conditions=[("PS3", "KI1"), ("PS3", "KI2"), ("PS3", "KI3"), ("PS3", "KI4")],
    verbose=True,
)

print("\n" + "=" * 70)
print("FULL PS3 SWEEP RESULT (Llama 3.1 8B, 50 incidents)")
print("Baseline:  PS3_KI1 36.4%   PS3_KI2 38.0%   PS3_KI3 38.0%   PS3_KI4 37.8%")
print("=" * 70)
for key, m in metrics.items():
    print(
        f"{key}: accuracy={m.overall_accuracy_micro*100:.1f}%, "
        f"F1={m.overall_f1_micro:.3f}, hallucination={m.overall_hallucination_rate*100:.1f}%"
    )

report = runner.generate_summary_report(metrics)
print("\n" + report)
