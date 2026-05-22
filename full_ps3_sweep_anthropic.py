"""Full PS3 sweep on Haiku 4.5 + Opus 4.6 after the CoVe fix.

Runs all 4 KI conditions (PS3_KI1..PS3_KI4) on the full 50-incident
dataset for both Anthropic models. Expected runtime ~45-55 min total
(Haiku ~17 min, Opus ~30 min).

Pre-fix baselines (org-excluded micro accuracy):
  Haiku PS3_KI1 50.2%   KI2 50.3%   KI3 49.6%   KI4 49.8%
  Opus  PS3_KI1 52.7%   KI2 51.0%   KI3 51.4%   KI4 51.4%
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
print("Sweep: Haiku 4.5 + Opus 4.6 x PS3 x [KI1, KI2, KI3, KI4]")

runner = ExperimentRunner(output_dir=str(project_root / "data/results"))
metrics = runner.run_benchmark(
    dataset=dataset,
    model_keys=["claude-haiku-4-5-20251001", "claude-opus-4-6"],
    conditions=[("PS3", "KI1"), ("PS3", "KI2"), ("PS3", "KI3"), ("PS3", "KI4")],
    verbose=True,
)

print("\n" + "=" * 70)
print("ANTHROPIC PS3 SWEEP RESULT (50 incidents)")
print("Haiku baseline: KI1 50.2% / KI2 50.3% / KI3 49.6% / KI4 49.8%")
print("Opus  baseline: KI1 52.7% / KI2 51.0% / KI3 51.4% / KI4 51.4%")
print("=" * 70)
for key, m in metrics.items():
    print(
        f"{key}: accuracy={m.overall_accuracy_micro*100:.1f}%, "
        f"F1={m.overall_f1_micro:.3f}, hallucination={m.overall_hallucination_rate*100:.1f}%"
    )
