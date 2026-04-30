# AI Incident Extraction Benchmark

Benchmark LLM-based structured extraction from AI incident news using a factorial design: 3 Prompting Strategies x 4 Knowledge Injection levels x 3 Models.

## Research Question

> How do prompting strategies and knowledge injection affect the accuracy and reliability of structured extraction from AI incident news?

## Experimental Design

**Factor A: Prompting Strategy (PS)**

| Condition | Strategy | Description |
|-----------|----------|-------------|
| PS1 | Zero-shot | Task instruction + KI component + article |
| PS2 | Few-shot | Three worked examples + KI component + article |
| PS3 | Verification (CoVe) | Two-step: extract then independently verify each field |

**Factor B: Knowledge Injection (KI)** — cumulative hierarchy

| Condition | Form | What is added |
|-----------|------|---------------|
| KI1 | No injection | Field names and JSON format only |
| KI2 | Schema-guided | Field definitions, data types, enumerated allowed values |
| KI3 | Taxonomy-guided | KI2 + hierarchical category trees with subtypes |
| KI4 | Ontology-guided | KI3 + typed entities and relational constraints |

**Models (3 capability tiers)**

| Tier | Model | Run via |
|------|-------|---------|
| Low | Llama 3.1 8B (4-bit quantized) | Ollama (local) |
| Mid | Claude Haiku 4.5 | Anthropic API |
| High | Claude Opus 4.6 | Anthropic API |

## Project Structure

```
ai_incident_extraction/
├── data/
│   ├── raw/
│   │   └── experimental_incidents_50.xlsx   # 50 OECD AIM incident records
│   ├── annotated/
│   │   └── ground_truth_50.json             # Manual annotations (12 fields per incident)
│   └── results/                             # Experiment outputs (one dir per condition)
├── src/
│   ├── templates/
│   │   ├── __init__.py                      # Condition builder: build_condition_prompt(ps, ki, article)
│   │   ├── knowledge_injection.py           # KI1-KI4 prompt components
│   │   └── prompting_strategy.py            # PS1-PS3 prompt builders
│   ├── llm_client.py                        # Ollama + Anthropic API clients
│   ├── data_loader.py                       # Excel/JSON dataset loading
│   ├── evaluation.py                        # Field-level comparison + metrics
│   └── experiment.py                        # Main experiment runner
├── run_test.py                              # Quick setup verification
├── .venv/                                   # Python 3.12 virtual environment
└── README.md
```

## Setup

### 1. Create virtual environment and install dependencies

```bash
cd ai_incident_extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install requests openpyxl bert-score "transformers<5" "numpy<2"
```

### 2. Install Ollama and pull model

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

### 3. Set Anthropic API key (for Haiku/Opus)

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 4. Verify setup

```bash
source .venv/bin/activate
python run_test.py
```

## Usage

### Full benchmark (all 12 conditions on Llama 3.1 8B)

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python src/experiment.py
```

### Custom run

Edit `src/experiment.py::main()` to select specific models and conditions:

```python
model_keys = ["llama3.1:8b"]
conditions = [("PS1", "KI1"), ("PS2", "KI2")]  # Or None for all 12
```

### Re-evaluate saved results (no LLM calls needed)

After fixing evaluation code, re-evaluate from saved raw outputs:

```python
# See the re-evaluation script pattern in the experiment output analysis
```

## Extraction Schema

12 fields across 4 groups, derived from OECD Common Reporting Framework and AIID:

| Group | Field | Type | Evaluation |
|-------|-------|------|------------|
| event | event_type | constrained: "AI incident" / "AI hazard" | exact match |
| event | event_date | open: YYYY-MM-DD | exact match |
| event | event_location | open: free text | BERTScore >= 0.5 |
| event | description | open: free text | **excluded** |
| ai_system | name | open: string | exact match + substring |
| ai_system | system_type | constrained: 10 values | exact match |
| ai_system | developer | open: string | exact match + substring |
| ai_system | deployer | open: string | exact match + substring |
| harm | harm_type | constrained: 7 values | exact match |
| harm | severity | constrained: 4 levels | exact match |
| harm | affected_parties | open: free text | BERTScore >= 0.5 |
| organizations | name + role | array of objects | BERTScore name matching + exact role |

## Evaluation Metrics

Each field is classified as one of: **correct**, **incorrect**, **missing**, **hallucinated**.

| Metric | Formula |
|--------|---------|
| Accuracy | correct / total fields |
| Precision | correct / (correct + incorrect + hallucinated) |
| Recall | correct / (correct + missing) |
| F1 Score | harmonic mean of precision and recall |
| Hallucination Rate | hallucinated / (correct + incorrect + hallucinated) |
| JSON Validity | % of outputs parseable as valid JSON |

## Data Source

50 incidents from the [OECD AI Incidents Monitor](https://oecd.ai/en/incidents):
- Country: USA
- Downloaded: 12 April 2026
- Each record: title + summary + concepts (OECD AIM fields)
- Ground truth: 50 manually reviewed annotations with constrained vocabulary

## Known Issues and Design Notes

- **Flat-to-nested normalization**: Models without schema guidance (PS1_KI1) produce flat JSON. The evaluation normalizes flat output to nested structure before comparison.
- **PS3 verification step does not receive KI component**: Only step 1 (extraction) gets the KI prompt. Step 2 (verification) gets a generic field list. This reduces KI effect for PS3.
- **Ollama JSON mode vs Anthropic**: Ollama uses `format: "json"` to force valid JSON output. Anthropic relies on prompt instructions. This may cause format-related differences.
- **BERTScore model**: distilbert-base-uncased with rescale_with_baseline=True.
