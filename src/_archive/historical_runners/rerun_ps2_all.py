"""
Re-run only PS2 conditions on all three models, after fixing the
few-shot examples to include a Date line in the article and the
corresponding event_date in the output JSON. Without this fix, the
examples were teaching the model to write "not stated" even when the
real article had a date — depressing PS2 numbers across all tiers.

Estimated cost: ~$2 Haiku, ~$5 Opus, free Llama. ~30 min wall-clock.
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

    conditions = [("PS2", ki) for ki in ("KI1", "KI2", "KI3", "KI4")]
    print(f"\nDataset: {len(dataset)} incidents")
    print(f"Re-running 4 PS2 conditions on three model tiers under fixed few-shot examples\n")

    runner.run_benchmark(
        dataset=dataset,
        model_keys=[
            "llama3.1:8b",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
        ],
        conditions=conditions,
        verbose=True,
    )


if __name__ == "__main__":
    main()
