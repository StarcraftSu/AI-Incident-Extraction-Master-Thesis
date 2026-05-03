# AI Incident Extraction Benchmark

Benchmark LLM-based structured extraction from AI incident news using a factorial design: **3 Prompting Strategies × 4 Knowledge Injection levels × 3 Models = 36 condition cells × 50 incidents per cell**.

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
| Mid | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Anthropic API |
| High | Claude Opus 4.6 (`claude-opus-4-6`) | Anthropic API |

## Project Structure

```
ai_incident_extraction/
├── data/
│   ├── raw/
│   │   └── experimental_incidents_50.xlsx     # 50 OECD AIM incident records
│   ├── annotated/
│   │   └── ground_truth_50.json               # Manual annotations (12 fields per incident)
│   └── results/                               # Per-condition outputs (36 dirs after a full run)
│       ├── <model>_<timestamp>/
│       │   ├── PS<i>_KI<j>_results.json       # Raw model outputs + GT pairs
│       │   └── PS<i>_KI<j>_metrics.json       # Computed BenchmarkMetrics
│       └── _archive/                          # Pre-fix runs (kept as diff targets)
├── src/
│   ├── templates/
│   │   ├── __init__.py                        # build_condition_prompt(ps, ki, article)
│   │   ├── knowledge_injection.py             # KI1-KI4 prompt components
│   │   └── prompting_strategy.py              # PS1-PS3 prompt builders
│   ├── llm_client.py                          # Ollama + Anthropic REST clients (no SDK)
│   ├── data_loader.py                         # Excel/JSON dataset loading; prepends Date+Country to article_text
│   ├── evaluation.py                          # Field-level comparison + BenchmarkMetrics aggregation
│   ├── experiment.py                          # ExperimentRunner — loops models × conditions × incidents
│   ├── reevaluate.py                          # Re-score saved *_results.json without re-calling LLMs
│   ├── generate_summary.py                    # Cross-condition summary tables (feeds Chapter 4)
│   └── independent_audit.py                   # Confirmability check — stdlib-only recompute
├── tests/                                     # 76-test pytest suite guarding evaluation logic
│   ├── test_aggregation.py
│   ├── test_compare_values.py
│   ├── test_evaluate_extraction.py
│   └── test_normalize_to_nested.py
├── run_test.py                                # Quick setup smoke test (1 LLM call)
├── requirements.txt                           # Pinned deps (torch 2.2.2, transformers<5, numpy<2)
├── .venv/                                     # Python 3.12 virtual environment
├── CLAUDE.md                                  # Working notes for AI assistants
└── README.md
```

## Setup

### 1. Create virtual environment and install dependencies

```bash
cd ai_incident_extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The version pins (`torch==2.2.2`, `transformers<5`, `numpy<2`) are required for x86_64 macOS compatibility — see comments in `requirements.txt`.

### 2. Install Ollama and pull model

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

### 3. Set Anthropic API key (for Haiku/Opus)

Create a `.env` file at the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

`run_test.py` and `experiment.py` both auto-load this on startup via `python-dotenv`.

### 4. Verify setup

```bash
python run_test.py    # Builds all 12 prompt conditions and runs one Ollama extraction
```

## Usage

### Run the full benchmark (36 cells)

`src/experiment.py::main()` uses a manual model-keys toggle. By default it runs Llama only:

```python
model_keys = ["llama3.1:8b"]                                          # default (active)
# model_keys = list(MODELS.keys())                                    # all 3 tiers
# model_keys = ["claude-haiku-4-5-20251001", "claude-opus-4-6"]       # API tiers only
```

To run all three tiers, uncomment the appropriate line and:

```bash
PYTHONUNBUFFERED=1 python src/experiment.py
```

A full 36-cell run costs ~$40 in Anthropic API charges and ~12 hours of wall time (mostly Llama on local hardware; Anthropic calls are ~2-4 s each).

### Re-evaluate saved results (no LLM calls)

After fixing or refining `evaluation.py`, re-score every saved `*_results.json` without re-calling any model:

```bash
python src/reevaluate.py
```

This rebuilds `ExtractionResult` objects from disk, runs them through the current evaluator, and overwrites `*_metrics.json` with fresh numbers. Used heavily during the iterative bug-fix process.

### Generate cross-condition summary

After a full run, produce the summary tables that feed Chapter 4:

```bash
python src/generate_summary.py
```

### Verify metrics independently

For confirmability — re-derive headline accuracy using only stdlib (no BERTScore, no constrained-vocab set-intersection, no harm-suffix stripping, no organization best-match pairing):

```bash
python src/independent_audit.py
```

The audit applies a strictly simpler rule set than the canonical evaluator, so its numbers are a *lower bound* on canonical accuracy. As of the most recent run, the audit lands within ±3 pp of the canonical numbers on **29 / 36 cells (81 %)**, supporting the Discussion §5.3 confirmability claim.

### Run the test suite

```bash
pytest tests/ -q
```

76 tests guard `compare_values`, `_normalize_to_nested`, `evaluate_extraction`, and the `BenchmarkMetrics` aggregation. Each test pins behaviour against a specific evaluator bug or design decision so future code changes can't silently regress.

## Extraction Schema

12 fields across 4 groups, derived from OECD Common Reporting Framework and AIID. The headline metric covers **10 scalar fields** (description and the variable-length organizations array are excluded — see "Methodology" below).

| Group | Field | Type | Evaluation |
|-------|-------|------|------------|
| event | event_type | constrained: "AI incident" / "AI hazard" | exact match |
| event | event_date | open: YYYY-MM-DD | exact match |
| event | event_location | open: free text | substring → BERTScore ≥ 0.5 |
| event | description | open: free text | **excluded** (free-text summary) |
| ai_system | name | open: string | exact match + substring |
| ai_system | system_type | constrained: 10 values | exact match |
| ai_system | developer | open: string | exact match + substring |
| ai_system | deployer | open: string | exact match + substring |
| harm | harm_type | constrained: 7 values | exact match (set intersection on multi-value) |
| harm | severity | constrained: 4 levels | exact match |
| harm | affected_parties | open: free text | substring → BERTScore ≥ 0.5 |
| organizations | name + role (variable-length array) | array of objects | **excluded from headline** (separate sub-task) |

## Evaluation Metrics

Each scored field is classified as one of: **correct**, **incorrect**, **missing**, **hallucinated**. Aggregation produces both micro (sample-weighted) and macro (unweighted mean of per-field metrics) variants.

| Metric | Formula | Notes |
|--------|---------|-------|
| Accuracy (micro) | sum(correct) / sum(scored) over all fields × incidents | **Primary headline metric** |
| Accuracy (macro) | mean(per-field accuracy) | Reported in `*_metrics.json` for sensitivity |
| Precision | correct / (correct + incorrect + hallucinated) | |
| Recall | correct / (correct + incorrect + missing) | |
| F1 Score | harmonic mean of precision and recall | Both micro and macro variants |
| Hallucination rate | hallucinated / (correct + incorrect + hallucinated) | Computed over in-scope fields only |
| JSON Validity | % of outputs parseable as valid JSON | 100 % across all 36 cells in the locked benchmark |

## Methodology Notes

- **Headline scope = 10 scalar fields.** The variable-length `organizations` array is excluded from headline aggregation because its small per-slot row counts, role-coding subjectivity, and prevalence of generic entity mentions ("users", "Republican") would put ~28 % weight on the noisiest sub-task. Per-cell organization data is preserved in `*_results.json` for future supplementary analysis. The `description` field is excluded because exact-match scoring of free-text summaries is not meaningful (BERTScore at the standard threshold saturates near 100 % on Haiku and Opus).
- **Date and country are prepended to the article text** at load time (`data_loader.py`). Without this, models often returned `"not stated"` for `event_date` and `event_location` even when the values were available in the OECD AIM metadata — see `_archive/no_date_in_input_*` and `_archive/no_country_in_input_*` for the pre-fix runs.
- **PS3 verification step does NOT receive the KI component.** Only step 1 (extraction) gets the KI prompt. Step 2 (verification) gets a generic field list with quote-grounding instructions. This is intentional: it follows the chain-of-verification design (Dhuliawala et al., 2023) where the verifier independently re-derives values without seeing the first call's output.
- **BERTScore model**: `distilbert-base-uncased` with `rescale_with_baseline=True`, threshold 0.5 on F1. The substring shortcut at the top of the BERTScore branch in `evaluation.py` short-circuits the common cases (e.g., `"United States"` ⊂ `"California, United States"`) before the threshold is consulted.
- **Reproducibility.** The Llama tier is exactly reproducible (open weights, Ollama with `format: "json"`, temperature 0.0). The proprietary tiers are pinned via the snapshot ID `claude-haiku-4-5-20251001` (dated snapshot) and the alias `claude-opus-4-6` (no dated snapshot was available at the time of the runs); Anthropic's API may evolve the alias over time.

## Data Source

50 incidents from the [OECD AI Incidents Monitor](https://oecd.ai/en/incidents):
- Country filter: USA
- Downloaded: 12 April 2026
- Each record: title + summary + concepts + date + country (OECD AIM fields)
- Ground truth: 50 manually annotated records using the constrained vocabulary above

## Constrained Field Vocabulary

The only valid values, used in both ground truth and the evaluator's set-intersection matching:

- **event_type**: `AI incident`, `AI hazard`
- **system_type**: facial recognition, recommendation system, generative AI, autonomous vehicle, decision support, chatbot, content moderation, predictive system, ai agent, other
- **harm_type**: physical, psychological, reputational, economic, environmental, rights violation, other
- **severity**: minor, moderate, significant, severe
- **organizations[*].role**: developer, deployer, regulator, victim, other

## Audit Trail

The evaluation pipeline went through ~12 bug-fix and design-change commits during data preparation. The pre-fix runs are preserved in `data/results/_archive/` with READMEs documenting why each was superseded. Each evaluator change ships with a regression test in `tests/`. The full commit range is `0b01c2f` (initial pipeline) through the current `master`. See Discussion §5.3 of the thesis for the confirmability story.
