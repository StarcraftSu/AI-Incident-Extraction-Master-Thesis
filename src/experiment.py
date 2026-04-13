"""
Main experiment runner for AI incident extraction benchmark.

Supports the PS × KI factorial design:
  3 Prompting Strategies (PS1-PS3) × 4 Knowledge Injection levels (KI1-KI4)
  = 12 conditions per model.

Models:
  - Llama 3.1 8B (Ollama, local)
  - Claude Haiku 4.5 (Anthropic API)
  - Claude Opus 4.6 (Anthropic API)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from data_loader import Dataset
from llm_client import OllamaClient, AnthropicClient, LLMResponse, create_client
from evaluation import (
    ExtractionResult,
    BenchmarkMetrics,
    parse_json_output,
    calculate_metrics,
    print_metrics_report,
)

# Import new modular templates
try:
    from templates import (
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
    )
    from templates.prompting_strategy import build_ps3_verification_prompt
except ImportError:
    from .templates import (
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
    )
    from .templates.prompting_strategy import build_ps3_verification_prompt


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "llama3.1:8b": {
        "provider": "ollama",
        "model_id": "llama3.1:8b",
        "label": "Llama 3.1 8B (local, 4-bit)",
        "tier": "low",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5 (API)",
        "tier": "mid",
    },
    "claude-opus-4-6-20250918": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6-20250918",
        "label": "Claude Opus 4.6 (API)",
        "tier": "high",
    },
}


class ExperimentRunner:
    """Runs extraction experiments across models and PS×KI conditions."""

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
        self._clients = {}  # cache clients by provider

    def _get_client(self, provider: str):
        """Get or create a client for the given provider."""
        if provider not in self._clients:
            self._clients[provider] = create_client(provider)
        return self._clients[provider]

    def check_model(self, model_key: str) -> bool:
        """Verify a model is available."""
        model_info = MODELS[model_key]
        client = self._get_client(model_info["provider"])

        if model_info["provider"] == "ollama":
            if not client.is_available():
                print(f"  ERROR: Ollama is not running. Start with: ollama serve")
                return False
            models = client.list_models()
            for m in models:
                if model_info["model_id"] in m or m in model_info["model_id"]:
                    return True
            print(f"  Model {model_info['model_id']} not found locally. Pulling...")
            return client.pull_model(model_info["model_id"])

        elif model_info["provider"] == "anthropic":
            if not client.is_available():
                print(f"  ERROR: ANTHROPIC_API_KEY not set.")
                return False
            return True

        return False

    def _call_llm(self, model_key: str, prompt: str) -> LLMResponse:
        """Call the appropriate LLM based on model config."""
        model_info = MODELS[model_key]
        client = self._get_client(model_info["provider"])

        if model_info["provider"] == "ollama":
            return client.generate(
                model=model_info["model_id"],
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                format="json",
            )
        else:
            return client.generate(
                model=model_info["model_id"],
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def run_single_extraction(
        self,
        incident_id: str,
        article_text: str,
        ground_truth: dict,
        model_key: str,
        ps: str,
        ki: str,
    ) -> ExtractionResult:
        """Run a single extraction for one PS×KI condition."""
        condition = f"{ps}_{ki}"

        if ps in ("PS1", "PS2"):
            # Single-call strategies
            prompt = build_condition_prompt(ps, ki, article_text)
            response = self._call_llm(model_key, prompt)
            parsed, is_valid = parse_json_output(response.text)

            return ExtractionResult(
                incident_id=incident_id,
                model=model_key,
                template=condition,
                raw_output=response.text,
                parsed_output=parsed,
                ground_truth=ground_truth,
                is_valid_json=is_valid,
                latency_seconds=response.latency_seconds,
                error=response.error if not response.success else None,
            )

        elif ps == "PS3":
            # Two-call CoVe strategy
            # Step 1: Extract
            extraction_prompt = build_condition_prompt(ps, ki, article_text)
            response1 = self._call_llm(model_key, extraction_prompt)
            parsed1, is_valid1 = parse_json_output(response1.text)

            if not is_valid1 or not parsed1:
                # If first extraction fails, return the failure
                return ExtractionResult(
                    incident_id=incident_id,
                    model=model_key,
                    template=condition,
                    raw_output=response1.text,
                    parsed_output=parsed1,
                    ground_truth=ground_truth,
                    is_valid_json=is_valid1,
                    latency_seconds=response1.latency_seconds,
                    error=response1.error or "Step 1 extraction failed",
                )

            # Step 2: Verify
            verification_prompt = build_ps3_verification_prompt(article_text, parsed1)
            response2 = self._call_llm(model_key, verification_prompt)
            parsed2, is_valid2 = parse_json_output(response2.text)

            total_latency = response1.latency_seconds + response2.latency_seconds

            # Use verified output if valid, otherwise fall back to initial
            final_parsed = parsed2 if is_valid2 and parsed2 else parsed1
            final_valid = is_valid2 if parsed2 else is_valid1

            return ExtractionResult(
                incident_id=incident_id,
                model=model_key,
                template=condition,
                raw_output=f"--- STEP 1 ---\n{response1.text}\n--- STEP 2 ---\n{response2.text}",
                parsed_output=final_parsed,
                ground_truth=ground_truth,
                is_valid_json=final_valid,
                latency_seconds=total_latency,
                error=None,
            )

        else:
            raise ValueError(f"Unknown PS: {ps}")

    def run_benchmark(
        self,
        dataset: Dataset,
        model_keys: list[str],
        conditions: Optional[list[tuple[str, str]]] = None,
        verbose: bool = True,
    ) -> dict[str, BenchmarkMetrics]:
        """
        Run full benchmark across models and PS×KI conditions.

        Args:
            dataset: The incident dataset.
            model_keys: List of model keys from MODELS dict.
            conditions: List of (PS, KI) tuples. Defaults to all 12.
            verbose: Print progress.

        Returns:
            Dict mapping "model+PS_KI" to BenchmarkMetrics.
        """
        if conditions is None:
            conditions = [
                (ps, ki)
                for ps in ["PS1", "PS2", "PS3"]
                for ki in ["KI1", "KI2", "KI3", "KI4"]
            ]

        all_metrics = {}

        for model_key in model_keys:
            model_info = MODELS[model_key]
            print(f"\n{'#'*60}")
            print(f"Model: {model_info['label']}")
            print(f"{'#'*60}")

            if not self.check_model(model_key):
                print(f"  Skipping — model not available")
                continue

            for ps, ki in conditions:
                condition = f"{ps}_{ki}"
                if verbose:
                    print(f"\n  {'='*50}")
                    print(f"  Condition: {PS_LABELS[ps]} × {KI_LABELS[ki]}")
                    print(f"  {'='*50}")

                results = []
                for i, incident in enumerate(dataset):
                    if verbose:
                        print(f"    [{i+1}/{len(dataset)}] {incident.id}...", end=" ")

                    result = self.run_single_extraction(
                        incident_id=incident.id,
                        article_text=incident.article_text,
                        ground_truth=incident.ground_truth or {},
                        model_key=model_key,
                        ps=ps,
                        ki=ki,
                    )
                    results.append(result)

                    if verbose:
                        status = "OK" if result.is_valid_json else "INVALID"
                        print(f"{status} ({result.latency_seconds:.1f}s)")

                metrics = calculate_metrics(results)
                # Override model/template labels for new design
                metrics.model = model_key
                metrics.template = condition
                key = f"{model_key}+{condition}"
                all_metrics[key] = metrics

                if verbose:
                    print_metrics_report(metrics)

                self._save_results(model_key, condition, results, metrics)

        return all_metrics

    def _save_results(
        self,
        model: str,
        condition: str,
        results: list[ExtractionResult],
        metrics: BenchmarkMetrics,
    ):
        """Save results and metrics to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = model.replace(":", "_").replace("/", "_").replace("-", "_")

        # Create model-specific subdirectory
        run_dir = self.output_dir / f"{model_safe}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = run_dir / f"{condition}_results.json"
        results_data = [
            {
                "incident_id": r.incident_id,
                "model": r.model,
                "condition": r.template,
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

        # Save metrics
        metrics_file = run_dir / f"{condition}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        print(f"    Saved: {results_file.name}")

    def generate_summary_report(
        self,
        all_metrics: dict[str, BenchmarkMetrics]
    ) -> str:
        """Generate a summary comparison report."""
        lines = [
            "# AI Incident Extraction Benchmark Summary",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\nDesign: 3 PS × 4 KI = 12 conditions per model",
            "\n## Overall Comparison\n",
            "| Model | PS | KI | JSON Valid | Accuracy | F1 | Halluc. | Latency |",
            "|-------|----|----|------------|----------|-----|---------|---------|",
        ]

        for key, m in sorted(all_metrics.items()):
            parts = m.template.split("_")
            ps = parts[0] if len(parts) >= 2 else m.template
            ki = parts[1] if len(parts) >= 2 else ""

            # Calculate hallucination rate
            total_hall = sum(fm.hallucinated for fm in m.field_metrics.values())
            total_extracted = sum(
                fm.correct + fm.incorrect + fm.hallucinated
                for fm in m.field_metrics.values()
            )
            hall_rate = total_hall / total_extracted if total_extracted > 0 else 0.0

            lines.append(
                f"| {m.model} | {ps} | {ki} | "
                f"{m.json_validity_rate:.0%} | "
                f"{m.overall_accuracy:.1%} | "
                f"{m.overall_f1:.3f} | "
                f"{hall_rate:.1%} | "
                f"{m.avg_latency:.1f}s |"
            )

        report = "\n".join(lines)

        report_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"\nSummary saved: {report_file}")
        return report


def main():
    """Main entry point for running experiments."""
    print("=" * 60)
    print("AI Incident Extraction Benchmark")
    print("Design: 3 PS × 4 KI × 3 Models = 36 runs")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    runner = ExperimentRunner(output_dir=str(project_root / "data/results"))

    # Load dataset
    dataset_path = project_root / "data/raw/experimental_incidents_50.xlsx"
    if not dataset_path.exists():
        print(f"\nERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    from data_loader import load_dataset
    dataset = load_dataset(str(dataset_path))
    print(f"\nDataset: {len(dataset)} incidents")

    # --- Configure what to run ---
    # Uncomment/modify as needed:

    # Run all models:
    # model_keys = list(MODELS.keys())

    # Run only local model:
    model_keys = ["llama3.1:8b"]

    # Run only Anthropic models:
    # model_keys = ["claude-haiku-4-5-20251001", "claude-opus-4-6-20250918"]

    # Run all 12 conditions:
    conditions = None  # None = all 12

    # Run a subset for testing:
    # conditions = [("PS1", "KI1"), ("PS1", "KI2")]

    print(f"Models: {model_keys}")
    print(f"Conditions: {'all 12' if conditions is None else conditions}")

    all_metrics = runner.run_benchmark(
        dataset=dataset,
        model_keys=model_keys,
        conditions=conditions,
        verbose=True,
    )

    report = runner.generate_summary_report(all_metrics)
    print("\n" + report)


if __name__ == "__main__":
    main()
