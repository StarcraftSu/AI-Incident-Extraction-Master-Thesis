# AI Incident Extraction Benchmark

Benchmark LLM-based structured extraction from AI incident news using a factorial design: 3 Prompting Strategies × 4 Knowledge Injection levels × 3 Models = 36 experimental runs.

## Research Question

> How do prompting strategies and knowledge injection affect the accuracy and reliability of structured extraction from AI incident news?

## Experimental Design

**Factor A: Prompting Strategy (PS)**

| Condition | Strategy | Description |
|-----------|----------|-------------|
| PS1 | Zero-shot | Task instruction + KI + article |
| PS2 | Few-shot | Two worked examples + KI + article |
| PS3 | Verification (CoVe) | Two-step: extract then verify each field |

**Factor B: Knowledge Injection (KI)**

| Condition | Form | What is added |
|-----------|------|---------------|
| KI1 | No injection | Field names only |
| KI2 | Schema-guided | Field definitions + enumerated values |
| KI3 | Taxonomy-guided | Schema + hierarchical category labels |
| KI4 | Ontology-guided | Taxonomy + relational constraints |

**Models (3 tiers)**

| Tier | Model | Run via |
|------|-------|---------|
| Low | Llama 3.1 8B | Ollama (local, 4-bit) |
| Mid | Claude Haiku 4.5 | Anthropic API |
| High | Claude Opus 4.6 | Anthropic API |

## Project Structure

```
ai_incident_extraction/
├── configs/
│   └── config.yaml                    # Experiment configuration
├── data/
│   ├── annotated/
│   │   └── incidents_20.json          # 20 annotated incidents (ground truth)
│   └── results/                       # Experiment outputs
├── src/
│   ├── templates/
│   │   ├── knowledge_injection.py     # KI1-KI4 prompt components
│   │   ├── prompting_strategy.py      # PS1-PS3 prompt builders
│   │   └── __init__.py                # Condition builder
│   ├── llm_client.py                  # Ollama + Anthropic clients
│   ├── data_loader.py                 # Dataset handling
│   ├── evaluation.py                  # Metrics calculation
│   ├── experiment.py                  # Main experiment runner
│   └── prompts.py                     # Re-exports
├── run_test.py                        # Quick setup test
└── README.md
```

## Setup

### 1. Install Ollama and pull model

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

### 2. Set Anthropic API key (for Haiku/Opus)

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Install Python dependencies

```bash
cd ai_incident_extraction
pip install -r requirements.txt
```

### 4. Run test

```bash
python run_test.py
```

## Usage

### Full Benchmark (all 12 conditions on Llama 3.1 8B)

```bash
cd src
python experiment.py
```

### Custom Run

Edit `experiment.py::main()` to select specific models and conditions:

```python
model_keys = ["llama3.1:8b"]
conditions = [("PS1", "KI1"), ("PS1", "KI2"), ("PS1", "KI3"), ("PS1", "KI4")]
```

## Evaluation Metrics

- **Accuracy**: correct / total fields
- **Precision**: correct / (correct + incorrect + hallucinated)
- **Recall**: correct / (correct + missing)
- **F1 Score**: harmonic mean of precision and recall
- **Hallucination Rate**: hallucinated / total extracted
- **JSON Validity Rate**: % of parseable JSON outputs

## Data Source

Incidents sourced from the [OECD AI Incidents Monitor](https://oecd.ai/en/incidents):
- Country: USA
- Date: January 2026
- 20 incidents with ground truth annotations
- Ground truth vocabulary aligned with KI2 schema (v2.0)
