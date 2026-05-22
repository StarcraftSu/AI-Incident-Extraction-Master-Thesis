"""Smoke test for the PS3 CoVe fix.

Runs first 5 incidents on Llama 3.1 8B at PS3 x KI4 to confirm the
verification step now respects the KI schema. Pre-fix baseline on
the full 50 incidents was 37.8% accuracy; if the fix works, the
5-incident smoke run should land notably above that.
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
print(f"Full dataset: {len(dataset)} incidents — smoke test uses first 5")

subset = dataset[:5]

runner = ExperimentRunner(output_dir=str(project_root / "data/results"))
metrics = runner.run_benchmark(
    dataset=subset,
    model_keys=["llama3.1:8b"],
    conditions=[("PS3", "KI4")],
    verbose=True,
)

print("\n" + "=" * 60)
print("SMOKE TEST RESULT (PS3 x KI4 fix, 5 incidents)")
print("Pre-fix full-50 baseline: 37.8% accuracy")
print("=" * 60)
for key, m in metrics.items():
    print(
        f"{key}: accuracy={m.overall_accuracy_micro*100:.1f}%, "
        f"F1={m.overall_f1_micro:.3f}, hallucination={m.overall_hallucination_rate*100:.1f}%"
    )
