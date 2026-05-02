"""
Full Claude Opus 4.6 benchmark: all 12 PS x KI conditions on the
50-incident OECD AIM dataset. Estimated cost ~$25-35 (800 LLM calls
counting PS3 doubles) at Opus pricing ($3/M in, $15/M out).

Output: data/results/claude_opus_4_6_<timestamp>/PS{1-3}_KI{1-4}_*.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from data_loader import load_dataset_with_ground_truth
from experiment import ExperimentRunner


def main():
    project_root = Path(__file__).resolve().parents[3]
    runner = ExperimentRunner(output_dir=str(project_root / "data" / "results"))

    dataset = load_dataset_with_ground_truth(
        str(project_root / "data" / "raw" / "experimental_incidents_50.xlsx"),
        str(project_root / "data" / "annotated" / "ground_truth_50.json"),
    )
    print(f"\nDataset: {len(dataset)} incidents")
    print("Running ALL 12 PS x KI conditions on claude-opus-4-6")
    print("Estimated cost: ~$25-35 (800 LLM calls)\n")

    runner.run_benchmark(
        dataset=dataset,
        model_keys=["claude-opus-4-6"],
        conditions=None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
