"""
Full Llama 3.1 8B benchmark: all 12 PS x KI conditions on the
50-incident OECD AIM dataset. Free (local Ollama).

Estimated runtime: 60-90 min (PS3 conditions take 2x because of
the verification call). No API cost.

Output: data/results/llama3.1_8b_<timestamp>/PS{1-3}_KI{1-4}_*.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# .env is harmless here (Ollama doesn't need a key) but we load it
# for consistency with the other rerun_* scripts.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data_loader import load_dataset_with_ground_truth
from experiment import ExperimentRunner


def main():
    project_root = Path(__file__).parent.parent
    runner = ExperimentRunner(output_dir=str(project_root / "data" / "results"))

    dataset = load_dataset_with_ground_truth(
        str(project_root / "data" / "raw" / "experimental_incidents_50.xlsx"),
        str(project_root / "data" / "annotated" / "ground_truth_50.json"),
    )
    print(f"\nDataset: {len(dataset)} incidents")
    print("Running ALL 12 PS x KI conditions on llama3.1:8b (local)\n")

    runner.run_benchmark(
        dataset=dataset,
        model_keys=["llama3.1:8b"],
        conditions=None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
