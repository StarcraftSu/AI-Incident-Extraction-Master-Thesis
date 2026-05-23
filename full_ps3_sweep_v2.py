"""Full PS3 sweep using the orthodox 3-call CoVe implementation.

Runs PS3 × {KI1, KI2, KI3, KI4} × {Llama 3.1 8B, Haiku 4.5, Opus 4.6}
on all 50 incidents. 600 extractions × 3 calls = 1,800 LLM calls.

The 3-call CoVe pattern:
  Call 1: baseline extract (PS1-style + KI component)
  Call 2: independent per-field WH-question verification (no draft values)
  Call 3: critical revision (draft + verifications, anti-rubber-stamp framing)

Implementation guided by Dhuliawala et al. (2024) "Chain-of-Verification
Reduces Hallucination in LLMs" (ACL Findings 2024), specifically the
"2-Step" / Factored variant where verification execution is independent
of the baseline.

Expected runtime (based on smoke-test latencies):
  Llama 3.1 8B local: ~2 h
  Haiku 4.5 API:      ~30 min
  Opus 4.6 API:       ~1.5 h
  Total (sequential): ~4 h
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
print(f"Loaded {len(dataset)} incidents — full PS3 sweep over all 50")

runner = ExperimentRunner(output_dir=str(project_root / "data/results"))

ps3_conditions = [("PS3", "KI1"), ("PS3", "KI2"), ("PS3", "KI3"), ("PS3", "KI4")]

models_in_order = [
    "claude-haiku-4-5-20251001",   # fastest API, run first
    "claude-opus-4-6",             # slowest API
    "llama3.1:8b",                 # local (independent of API quota)
]

print(f"\nModels: {models_in_order}")
print(f"Conditions: {ps3_conditions}")
print(f"Total extractions: {len(dataset)} × {len(ps3_conditions)} × {len(models_in_order)} = {len(dataset) * len(ps3_conditions) * len(models_in_order)}")
print(f"Total LLM calls (3 per extraction):              = {len(dataset) * len(ps3_conditions) * len(models_in_order) * 3}\n")

metrics = runner.run_benchmark(
    dataset=dataset,
    model_keys=models_in_order,
    conditions=ps3_conditions,
    verbose=True,
)

print("\n" + "=" * 60)
print("FULL PS3 3-CALL CoVe SWEEP COMPLETE")
print("=" * 60)
for key, m in sorted(metrics.items()):
    print(
        f"{key}: acc={m.overall_accuracy_micro*100:.1f}%, "
        f"F1={m.overall_f1_micro:.3f}, halluc={m.overall_hallucination_rate*100:.1f}%"
    )

print("\nResults saved under data/results/<model>_<timestamp>/PS3_KI*.json")
