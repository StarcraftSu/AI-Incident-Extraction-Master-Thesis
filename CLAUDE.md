# Claude Context for ai_incident_extraction

## What This Is

Experiment code for a master thesis: "Knowledge Injection for LLM-based AI Incident Information Extraction" at DSV, Stockholm University. Benchmarks 12 prompt conditions (3 PS x 4 KI) across 3 LLM capability tiers on 50 AI incident records.

## Architecture

```
build_condition_prompt(ps, ki, article) → LLM call → parse JSON → evaluate vs ground truth
```

- `templates/knowledge_injection.py` — KI1-KI4 prompt components (cumulative: KI3 = KI2 + taxonomy, KI4 = KI3 + ontology)
- `templates/prompting_strategy.py` — PS1 (zero-shot), PS2 (few-shot, 3 examples), PS3 (two-step verification)
- `evaluation.py` — field-level comparison with flat-to-nested normalization, BERTScore for free-text fields
- `experiment.py` — ExperimentRunner: loops models x conditions x incidents, saves per-condition results
- `llm_client.py` — Ollama (local, JSON mode) and Anthropic (API) clients

## Critical Design Decisions

1. **KI levels are cumulative**: KI4 includes everything from KI1+KI2+KI3. Changing KI2 affects KI3 and KI4.
2. **PS3 verification step does NOT receive the KI component** — only step 1 does. This is intentional (method chapter says step 2 gets "only the article and field names").
3. **Flat-to-nested normalization** (`_normalize_to_nested`): Models without schema (especially PS1_KI1) produce flat JSON. The evaluator maps flat keys to nested ground truth paths before comparison. Without this, flat outputs get double-counted as both "missing" (nested path) and "hallucinated" (flat path).
4. **Constrained fields use set intersection** for multi-value matching: both GT and extraction can be comma-separated.
5. **description field is excluded** from evaluation (it's a summary, not an extractable fact).
6. **BERTScore**: distilbert-base-uncased, threshold 0.5 on F1, with rescale_with_baseline=True.

## Constrained Field Vocabulary

These are the ONLY valid values for constrained fields (used in both ground truth and evaluation):

- **event_type**: "AI incident", "AI hazard"
- **system_type**: facial recognition, recommendation system, generative AI, autonomous vehicle, decision support, chatbot, content moderation, predictive system, ai agent, other
- **harm_type**: physical, psychological, reputational, economic, environmental, rights violation, other
- **severity**: minor, moderate, significant, severe
- **org role**: developer, deployer, regulator, victim, other

## Environment

- Python 3.12 in `.venv/`
- Key deps: requests, openpyxl, bert-score, transformers<5, numpy<2, torch 2.2.2
- Ollama for local Llama 3.1 8B (must be running: `ollama serve`)
- ANTHROPIC_API_KEY env var for Claude models

## Running

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python src/experiment.py   # Full 12-condition run
python run_test.py                             # Quick verification
python src/evaluation.py                       # Unit tests for eval logic
```

## Common Pitfalls

- **Don't change KI2 without checking KI3/KI4** — they inherit via string concatenation.
- **Don't add subcategory values to ground truth** — evaluation only matches top-level constrained values.
- **Saved results can be re-evaluated** without re-calling LLMs — reconstruct ExtractionResult from *_results.json files.
- **Ollama must be running** before experiment starts. Check with `curl http://localhost:11434/api/tags`.
- **torch 2.2.2 is the max for x86_64 macOS** — transformers must be <5 to be compatible.

## Recent Bug Fixes (2026-04-30)

1. **Flat-to-nested normalization**: PS1_KI1 outputs flat JSON → was double-counted as missing+hallucinated. Fixed with `_normalize_to_nested()`.
2. **Multi-value constrained matching**: "rights violation, economic" vs GT "economic, rights violation" was scored incorrect. Fixed with set intersection.
3. **GT typo**: "right violation" → "rights violation" (one incident).
4. **GT orgs**: Replaced `{"name": "not stated", "role": "not stated"}` with empty `[]` in 3 incidents.
5. **KI3 taxonomy instruction**: Changed "select the most specific applicable category" → "use the TOP-LEVEL category name in JSON output" to prevent subcategory values that evaluation rejects.
