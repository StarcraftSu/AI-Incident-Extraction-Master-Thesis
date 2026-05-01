"""
Rerun only KI3 and KI4 conditions on Llama 3.1 8B with the current
prompts (system_type taxonomy aligned with vocab in 633749c, harm_type
parents aligned in a66c749). KI1/KI2 prompts are unchanged so their
saved results remain valid.

Output: data/results/llama3.1_8b_<timestamp>/PS{1,2,3}_KI{3,4}_*.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load ANTHROPIC_API_KEY from .env at project root (no-op for local Ollama,
# but harmless and consistent with experiment.py's loader).
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
    print(f"Dataset: {len(dataset)} incidents")

    conditions = [
        ("PS1", "KI3"), ("PS1", "KI4"),
        ("PS2", "KI3"), ("PS2", "KI4"),
        ("PS3", "KI3"), ("PS3", "KI4"),
    ]
    print(f"Rerunning {len(conditions)} KI3/KI4 conditions on llama3.1:8b")

    runner.run_benchmark(
        dataset=dataset,
        model_keys=["llama3.1:8b"],
        conditions=conditions,
        verbose=True,
    )


if __name__ == "__main__":
    main()
