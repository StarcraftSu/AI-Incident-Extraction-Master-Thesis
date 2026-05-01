"""
Preflight: run Claude Haiku 4.5 on the FIRST incident only, across all
12 PS x KI conditions. Validates the API path, parsing, and prompts
end-to-end before committing to the full 50-incident run.

Cost estimate: ~16 LLM calls (12 PS1+PS2 + 4 PS3 doubles) at Haiku
pricing ≈ $0.10. Output goes into a fresh data/results/<timestamp>/ dir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data_loader import load_dataset_with_ground_truth, Dataset
from experiment import ExperimentRunner


def main():
    project_root = Path(__file__).parent.parent
    runner = ExperimentRunner(output_dir=str(project_root / "data" / "results"))

    full = load_dataset_with_ground_truth(
        str(project_root / "data" / "raw" / "experimental_incidents_50.xlsx"),
        str(project_root / "data" / "annotated" / "ground_truth_50.json"),
    )
    preflight = Dataset(name="haiku_preflight", incidents=full.incidents[:1])
    print(f"\nPreflight dataset: {len(preflight)} incident "
          f"(id={preflight[0].id})")
    print(f"Running ALL 12 PS x KI conditions on claude-haiku-4-5-20251001\n")

    runner.run_benchmark(
        dataset=preflight,
        model_keys=["claude-haiku-4-5-20251001"],
        conditions=None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
