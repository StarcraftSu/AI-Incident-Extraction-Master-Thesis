#!/usr/bin/env python3
"""
Quick test script to verify the setup and run a minimal extraction.

Usage:
    cd ai_incident_extraction
    python run_test.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_client import OllamaClient, AnthropicClient
from templates import build_condition_prompt, ALL_CONDITIONS, KI_LABELS, PS_LABELS
from evaluation import parse_json_output


def main():
    print("=" * 60)
    print("AI Incident Extraction - Quick Test")
    print("Design: 3 PS × 4 KI = 12 conditions")
    print("=" * 60)

    # Step 1: Check Ollama
    print("\n1. Checking Ollama connection...")
    ollama = OllamaClient()

    if not ollama.is_available():
        print("   ERROR: Ollama is not running!")
        print("   Start with: ollama serve")
        return False

    models = ollama.list_models()
    print(f"   OK - Models: {models}")

    # Step 2: Check Anthropic
    print("\n2. Checking Anthropic API...")
    anthropic = AnthropicClient()
    print(f"   Available: {anthropic.is_available()}")

    # Step 3: Test prompt generation for all 12 conditions
    print("\n3. Testing all 12 prompt conditions...")
    test_article = """Title: AI Chatbot Gives Harmful Medical Advice
Summary: A popular AI chatbot provided incorrect medical advice to a user, leading to a hospital visit. The chatbot recommended a dangerous drug interaction that could have been fatal.
Concepts: AI chatbot, medical advice, harmful, drug interaction"""

    for ps in ["PS1", "PS2", "PS3"]:
        for ki in ["KI1", "KI2", "KI3", "KI4"]:
            prompt = build_condition_prompt(ps, ki, test_article)
            print(f"   {ps}_{ki} ({PS_LABELS[ps]} × {KI_LABELS[ki]}): {len(prompt)} chars")

    # Step 4: Run a single extraction with Ollama
    print("\n4. Running test extraction (PS1_KI2 on Ollama)...")
    model = models[0] if models else "llama3.1:8b"
    print(f"   Model: {model}")

    prompt = build_condition_prompt("PS1", "KI2", test_article)
    response = ollama.generate(
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=2000,
        format="json",
    )

    if response.success:
        print(f"   OK - Response in {response.latency_seconds:.1f}s")
        parsed, valid = parse_json_output(response.text)
        if valid:
            print(f"   OK - Valid JSON")
            print(json.dumps(parsed, indent=2)[:500])
        else:
            print(f"   WARNING: Invalid JSON")
            print(f"   Raw: {response.text[:300]}...")
    else:
        print(f"   ERROR: {response.error}")
        return False

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  python src/experiment.py    # Run full benchmark")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
