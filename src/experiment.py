"""
Main experiment runner for AI incident extraction benchmark.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from data_loader import Dataset
from prompts import format_prompt, AVAILABLE_TEMPLATES
from llm_client import OllamaClient, LLMResponse
from evaluation import (
    ExtractionResult,
    BenchmarkMetrics,
    parse_json_output,
    calculate_metrics,
    print_metrics_report,
)


class ExperimentRunner:
    """Runs extraction experiments across models and templates."""

    def __init__(
        self,
        output_dir: str = "data/results",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OllamaClient()

    def check_setup(self) -> bool:
        """Verify Ollama is running."""
        if not self.client.is_available():
            print("ERROR: Ollama is not running!")
            print("Start it with: ollama serve")
            return False

        models = self.client.list_models()
        print(f"Ollama is running. Available models: {models}")
        return True

    def ensure_model(self, model_name: str) -> bool:
        """Ensure a model is available, pulling if necessary."""
        models = self.client.list_models()

        # Check if model exists (handle version tags)
        for m in models:
            if model_name in m or m in model_name:
                return True

        print(f"Model {model_name} not found. Pulling...")
        return self.client.pull_model(model_name)

    def run_single_extraction(
        self,
        incident_id: str,
        article_text: str,
        ground_truth: dict,
        model: str,
        template: str,
    ) -> ExtractionResult:
        """Run a single extraction and return results."""
        # Format the prompt
        prompt = format_prompt(template, article_text)

        # Call the model
        response: LLMResponse = self.client.generate(
            model=model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse the output
        parsed, is_valid = parse_json_output(response.text)

        return ExtractionResult(
            incident_id=incident_id,
            model=model,
            template=template,
            raw_output=response.text,
            parsed_output=parsed,
            ground_truth=ground_truth,
            is_valid_json=is_valid,
            latency_seconds=response.latency_seconds,
            error=response.error if not response.success else None,
        )

    def run_benchmark(
        self,
        dataset: Dataset,
        models: list[str],
        templates: list[str],
        verbose: bool = True,
    ) -> dict[str, BenchmarkMetrics]:
        """
        Run full benchmark across all model-template combinations.
        Returns dict mapping "model+template" to metrics.
        """
        all_metrics = {}

        for model in models:
            # Ensure model is available
            if not self.ensure_model(model):
                print(f"Skipping model {model} - not available")
                continue

            for template in templates:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Running: {model} + {template}")
                    print(f"{'='*60}")

                results = []

                for i, incident in enumerate(dataset):
                    if verbose:
                        print(f"  [{i+1}/{len(dataset)}] Processing {incident.id}...", end=" ")

                    result = self.run_single_extraction(
                        incident_id=incident.id,
                        article_text=incident.article_text,
                        ground_truth=incident.ground_truth or {},
                        model=model,
                        template=template,
                    )
                    results.append(result)

                    if verbose:
                        status = "OK" if result.is_valid_json else "INVALID JSON"
                        print(f"{status} ({result.latency_seconds:.1f}s)")

                # Calculate metrics for this combination
                metrics = calculate_metrics(results)
                key = f"{model}+{template}"
                all_metrics[key] = metrics

                if verbose:
                    print_metrics_report(metrics)

                # Save individual results
                self._save_results(model, template, results, metrics)

        return all_metrics

    def _save_results(
        self,
        model: str,
        template: str,
        results: list[ExtractionResult],
        metrics: BenchmarkMetrics,
    ):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = model.replace(":", "_").replace("/", "_")

        # Save detailed results
        results_file = self.output_dir / f"{model_safe}_{template}_{timestamp}_results.json"
        results_data = [
            {
                "incident_id": r.incident_id,
                "model": r.model,
                "template": r.template,
                "raw_output": r.raw_output,
                "parsed_output": r.parsed_output,
                "ground_truth": r.ground_truth,
                "is_valid_json": r.is_valid_json,
                "latency_seconds": r.latency_seconds,
                "error": r.error,
            }
            for r in results
        ]

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save metrics summary
        metrics_file = self.output_dir / f"{model_safe}_{template}_{timestamp}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")

    def generate_summary_report(
        self,
        all_metrics: dict[str, BenchmarkMetrics]
    ) -> str:
        """Generate a summary comparison report."""
        lines = [
            "# AI Incident Extraction Benchmark Summary",
            f"\nGenerated: {datetime.now().isoformat()}",
            "\n## Overall Comparison\n",
            "| Model | Template | JSON Valid | Accuracy | F1 Score | Avg Latency |",
            "|-------|----------|------------|----------|----------|-------------|",
        ]

        for key, metrics in sorted(all_metrics.items()):
            lines.append(
                f"| {metrics.model} | {metrics.template} | "
                f"{metrics.json_validity_rate:.1%} | "
                f"{metrics.overall_accuracy:.1%} | "
                f"{metrics.overall_f1:.3f} | "
                f"{metrics.avg_latency:.1f}s |"
            )

        report = "\n".join(lines)

        # Save report
        report_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"\nSummary report saved to: {report_file}")
        return report


def main():
    """Main entry point for running experiments."""
    print("=" * 60)
    print("AI Incident Extraction Benchmark")
    print("=" * 60)

    # Initialize runner (use project root for output)
    project_root = Path(__file__).parent.parent
    runner = ExperimentRunner(output_dir=str(project_root / "data/results"))

    # Check setup
    if not runner.check_setup():
        sys.exit(1)

    # Load annotated dataset (resolve path relative to project root)
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data/annotated/incidents_20.json"
    if dataset_path.exists():
        dataset = Dataset.load(dataset_path)
    else:
        print(f"\nERROR: Dataset not found at {dataset_path}")
        print("Please ensure incidents_20.json exists in data/annotated/")
        sys.exit(1)

    # Define what to test
    # Use available models (check with: ollama list)
    models_to_test = [
        "llama3.2:latest",   # 3B params
        # "qwen2.5:latest",  # Disabled - too slow
    ]

    templates_to_test = [
        "zero_shot",
        "simple_schema",
        "rich_ontology",
        "few_shot",
        "chain_of_verification",
    ]

    # Run benchmark
    print(f"\nRunning benchmark with:")
    print(f"  Models: {models_to_test}")
    print(f"  Templates: {templates_to_test}")
    print(f"  Dataset size: {len(dataset)} incidents")

    all_metrics = runner.run_benchmark(
        dataset=dataset,
        models=models_to_test,
        templates=templates_to_test,
        verbose=True,
    )

    # Generate summary
    report = runner.generate_summary_report(all_metrics)
    print("\n" + report)


if __name__ == "__main__":
    main()
