# AI Incident Extraction Benchmark Analysis Report

**Generated:** 2026-02-01
**Model:** llama3.2:latest (3B parameters)
**Dataset:** 20 AI incidents from AIID
**Templates Tested:** 5 knowledge injection strategies

---

## Executive Summary

This benchmark evaluates how different prompting strategies (knowledge injection templates) affect LLM accuracy in extracting structured information from AI incident news articles. The experiment tests the research question:

> *How to inject knowledge (prompting) into LLMs to improve their accuracy in extracting risk information from AI incident news?*

**Key Finding:** The `simple_schema` template achieved the best overall performance with 38.5% accuracy and 0.411 F1 score, suggesting that providing structured category definitions improves extraction quality more than complex ontologies or multi-step verification.

---

## Overall Results Comparison

| Rank | Template | JSON Valid | Accuracy | F1 Score | Precision | Recall | Avg Latency |
|------|----------|------------|----------|----------|-----------|--------|-------------|
| 1 | **simple_schema** | 100% | **38.5%** | **0.411** | ~45% | ~47% | 20.5s |
| 2 | few_shot | 100% | 30.0% | 0.322 | ~40% | ~42% | 21.7s |
| 3 | rich_ontology | 100% | 28.3% | 0.304 | ~38% | ~40% | 35.3s |
| 4 | zero_shot | 100% | 23.8% | 0.275 | ~28% | ~35% | 16.2s |
| 5 | chain_of_verification | 95% | 10.7% | 0.133 | ~25% | ~20% | 11.1s |

---

## Detailed Analysis by Template

### 1. Simple Schema (Best Performer)

**Description:** Provides flat category definitions with enumerated options (e.g., event_type: "malfunction | bias | privacy_breach | ...")

**Performance:**
- Overall Accuracy: 38.5%
- Overall F1: 0.411
- JSON Validity: 100%

**Field-Level Performance:**

| Field | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| harm.severity | 70% | 0.700 | Excellent - clear severity levels help |
| ai_system.name | 70% | 0.700 | Good entity recognition |
| ai_system.developer | 55% | 0.629 | Reasonable company extraction |
| harm.harm_type | 50% | 0.500 | Moderate - some confusion between types |
| harm.affected_parties | 45% | 0.450 | Challenging - varied descriptions |
| ai_system.system_type | 45% | 0.450 | Moderate - enumerated types help |
| ai_system.deployer | 40% | 0.485 | Often confused with developer |
| event.event_type | 40% | 0.400 | Moderate classification accuracy |
| harm.affected_count | 35% | 0.452 | Numbers often hallucinated |
| organizations | 40% | 0.432 | List extraction challenging |
| event.description | 5% | 0.050 | Free-text matching is difficult |
| event.event_date | 0% | 0.000 | Date format mismatch issue |
| event.event_location | 5% | 0.095 | Location extraction weak |

**Why it works:** The explicit enumeration of valid values (e.g., severity levels, event types) constrains the model's output to expected categories, reducing hallucination and improving consistency.

---

### 2. Few-Shot (Second Best)

**Description:** Provides 2 complete extraction examples before the target article.

**Performance:**
- Overall Accuracy: 30.0%
- Overall F1: 0.322
- JSON Validity: 100%

**Field-Level Highlights:**

| Field | Accuracy | Notes |
|-------|----------|-------|
| harm.affected_count | 65% | Examples showed numeric extraction |
| ai_system.developer | 60% | Good pattern matching from examples |
| ai_system.name | 60% | Learned from example structure |
| harm.affected_parties | 55% | Better than schema for this field |
| organizations | 45% | Improved list handling |
| harm.severity | 45% | Lower than schema (less constrained) |

**Insight:** Few-shot excels at fields requiring pattern recognition (affected_parties, organizations) but underperforms on fields where enumerated constraints help (severity, event_type).

---

### 3. Rich Ontology (gUFO-based)

**Description:** Uses detailed ontological framework with concepts like Endurants, Perdurants, Participation relations, and Causal chains.

**Performance:**
- Overall Accuracy: 28.3%
- Overall F1: 0.304
- JSON Validity: 100%

**Field-Level Highlights:**

| Field | Accuracy | Notes |
|-------|----------|-------|
| ai_system.status | 85% | New field - well extracted |
| harm.affected_count | 75% | Strong performance |
| ai_system.name | 65% | Good |
| ai_system.deployer | 50% | Moderate |
| ai_system.system_type | 45% | Similar to simple_schema |
| participation | 0% | Too abstract - all hallucinated |
| causal_chain | 0% | Too abstract - all hallucinated |

**Why it underperformed:**
1. **Complexity overhead:** The 3B model struggles with abstract ontological concepts
2. **Field proliferation:** Added fields (participation, causal_chain, temporal) were consistently hallucinated
3. **Longer prompts:** 3742 chars vs 1525 chars for simple_schema, leading to slower inference (35.3s)

**Recommendation:** Rich ontology may work better with larger models (7B+) or simplified to core concepts.

---

### 4. Zero-Shot (Baseline)

**Description:** Minimal instruction listing field names without definitions.

**Performance:**
- Overall Accuracy: 23.8%
- Overall F1: 0.275
- JSON Validity: 100%

**Observations:**
- Model often used different field names than expected (e.g., `event.date` instead of `event.event_date`)
- High hallucination rate for some fields
- Fastest template at 16.2s average

**Use case:** Quick baseline testing, not recommended for production.

---

### 5. Chain-of-Verification (Worst Performer)

**Description:** Multi-step process: extract, verify against source, remove unverified info, output final JSON.

**Performance:**
- Overall Accuracy: 10.7%
- Overall F1: 0.133
- JSON Validity: 95% (1 invalid output)

**Why it failed:**
1. **Format confusion:** Model often mixed verification reasoning with JSON output
2. **Over-conservative:** Verification step led to many fields being set to null
3. **Field name inconsistency:** Output used flat field names (e.g., `event_date`) instead of nested (`event.event_date`)

**Interesting finding:** When the model did extract data (e.g., `event_date`), precision was very high (84-100%), but recall was extremely low due to excessive null values.

---

## Field-by-Field Analysis

### Best Extracted Fields (across all templates)

| Field | Best Template | Best Accuracy | Notes |
|-------|---------------|---------------|-------|
| ai_system.name | simple_schema | 70% | Core entity - well recognized |
| harm.severity | simple_schema | 70% | Enumerated values help |
| ai_system.developer | few_shot | 60% | Company names recognizable |
| harm.affected_count | rich_ontology | 75% | Numeric extraction works |

### Worst Extracted Fields

| Field | Issue | Recommendation |
|-------|-------|----------------|
| event.description | Free-text matching fails | Use semantic similarity instead of exact match |
| event.event_date | Date format mismatch | Normalize date formats in evaluation |
| event.event_location | Inconsistent granularity | Define location hierarchy |
| organizations | List of dicts complex | Simplify to list of strings |

---

## Hallucination Analysis

Hallucination Rate by Template (fields present in extraction but not in ground truth):

| Template | High Hallucination Fields | Notes |
|----------|--------------------------|-------|
| zero_shot | event.type (100%), event.location (55%) | Uses wrong field names |
| rich_ontology | participation (100%), causal_chain (100%), temporal (100%) | Abstract fields fabricated |
| few_shot | harm.affected_count (25%) | Over-extracts numbers |
| simple_schema | harm.affected_count (40%) | Moderate |
| chain_of_verification | Low overall | Verification reduces hallucination but at cost of recall |

---

## Latency Analysis

| Template | Avg Latency | Prompt Length | Tokens/sec (est.) |
|----------|-------------|---------------|-------------------|
| chain_of_verification | 11.1s | 2126 chars | ~38 |
| zero_shot | 16.2s | 945 chars | ~26 |
| simple_schema | 20.5s | 1525 chars | ~35 |
| few_shot | 21.7s | 2496 chars | ~42 |
| rich_ontology | 35.3s | 3742 chars | ~48 |

**Observation:** Longer prompts generally produce longer outputs, increasing latency. Rich ontology's complex schema leads to verbose responses.

---

## Recommendations

### For This Research

1. **Use `simple_schema` as primary template** - Best balance of accuracy and consistency
2. **Test with larger models** - 7B+ may better handle rich_ontology complexity
3. **Improve evaluation metrics** - Current exact match is too strict for description fields
4. **Normalize date formats** - Many "incorrect" dates are actually correct but formatted differently

### Template Design Guidelines

1. **Enumerate valid values** - Explicit options (e.g., "malfunction | bias | ...") reduce hallucination
2. **Keep schema flat** - Nested structures increase extraction errors
3. **Limit abstract concepts** - Concrete fields (name, date, count) outperform abstract ones (participation, causal_chain)
4. **Balance prompt length** - Longer isn't always better; diminishing returns after ~1500 chars

### Future Work

1. Test with GPT-4, Llama3 70B, Qwen 7B for comparison
2. Implement semantic similarity for description matching
3. Add confidence scores to extractions
4. Create domain-specific fine-tuned model

---

## Appendix: Experiment Configuration

```yaml
Model: llama3.2:latest
Temperature: 0.0 (deterministic)
Max Tokens: 2000
Dataset: 20 incidents from AIID
Templates: zero_shot, simple_schema, rich_ontology, few_shot, chain_of_verification
Evaluation: Field-level accuracy, precision, recall, F1, JSON validity
```

---

## Files Generated

- `llama3.2_latest_zero_shot_*_results.json` - Raw extraction outputs
- `llama3.2_latest_zero_shot_*_metrics.json` - Aggregated metrics
- `llama3.2_latest_simple_schema_*_results.json`
- `llama3.2_latest_simple_schema_*_metrics.json`
- `llama3.2_latest_rich_ontology_*_results.json`
- `llama3.2_latest_rich_ontology_*_metrics.json`
- `llama3.2_latest_few_shot_*_results.json`
- `llama3.2_latest_few_shot_*_metrics.json`
- `llama3.2_latest_chain_of_verification_*_results.json`
- `llama3.2_latest_chain_of_verification_*_metrics.json`

---

*Report generated as part of Master's Thesis: "Knowledge Injection for LLM-based AI Incident Information Extraction"*
