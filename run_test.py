#!/usr/bin/env python3
"""
Quick test script to verify the setup and run a minimal benchmark.

Usage:
    cd ai_incident_extraction
    python run_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_client import OllamaClient
from data_loader import create_sample_dataset
from prompts import format_prompt, AVAILABLE_TEMPLATES
from evaluation import parse_json_output


def main():
    print("=" * 60)
    print("AI Incident Extraction - Quick Test")
    print("=" * 60)

    # Step 1: Check Ollama
    print("\n1. Checking Ollama connection...")
    client = OllamaClient()

    if not client.is_available():
        print("   ERROR: Ollama is not running!")
        print("   Please start Ollama with: ollama serve")
        print("   Then pull a model: ollama pull llama3.2:1b")
        return False

    print("   OK - Ollama is running")

    # Step 2: List models
    print("\n2. Checking available models...")
    models = client.list_models()

    if not models:
        print("   No models found. Pulling llama3.2:1b...")
        if not client.pull_model("llama3.2:1b"):
            print("   ERROR: Failed to pull model")
            return False
        models = ["llama3.2:1b"]

    print(f"   Available models: {models}")

    # Step 3: Create test data
    print("\n3. Creating test dataset...")
    dataset = create_sample_dataset()
    print(f"   Created {len(dataset)} sample incidents")

    # Save dataset
    dataset.save("data/annotated/sample_dataset.json")

    # Step 4: Test prompt formatting
    print("\n4. Testing prompt templates...")
    test_article = dataset[0].article_text
    for template in AVAILABLE_TEMPLATES:
        prompt = format_prompt(template, test_article)
        print(f"   {template}: {len(prompt)} chars")

    # Step 5: Test a single extraction
    print("\n5. Running test extraction...")
    model = models[0] if models else "llama3.2:1b"
    print(f"   Using model: {model}")

    # Use simple template for speed
    prompt = format_prompt("zero_shot", dataset[0].article_text)

    print("   Sending request to Ollama...")
    response = client.generate(
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000,
    )

    if response.success:
        print(f"   OK - Response received in {response.latency_seconds:.1f}s")
        print(f"   Tokens: {response.total_tokens}")

        # Try to parse JSON
        parsed, valid = parse_json_output(response.text)
        if valid:
            print("   OK - Valid JSON output")
            print(f"\n   Extracted data preview:")
            import json
            print(json.dumps(parsed, indent=2)[:500] + "...")
        else:
            print("   WARNING: Output is not valid JSON")
            print(f"   Raw output preview: {response.text[:300]}...")
    else:
        print(f"   ERROR: {response.error}")
        return False

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Setup is working!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run full benchmark: python src/experiment.py")
    print("  2. Add more models: ollama pull qwen2.5:1.5b")
    print("  3. Add real data to: data/annotated/")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
