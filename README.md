# AI Incident Extraction Benchmark

Benchmark different LLM prompting strategies for extracting structured data from AI incident news articles.

## Research Question

> How to inject knowledge (prompting) into LLMs to improve their accuracy in extracting risk information from AI incident news?

## Project Structure

```
ai_incident_extraction/
├── configs/
│   └── config.yaml              # Experiment configuration
├── data/
│   ├── raw/                     # Raw news articles (xlsx)
│   ├── annotated/               # Human-annotated ground truth (json)
│   └── results/                 # Experiment outputs (grouped by run)
├── src/
│   ├── templates/               # Knowledge injection templates
│   │   ├── zero_shot.py         # T1: Baseline
│   │   ├── simple_schema.py     # T2: Enumerated values
│   │   ├── rich_ontology.py     # T3: gUFO-based
│   │   ├── few_shot.py          # T4: Examples
│   │   └── chain_of_verification.py  # T5: Verify against source
│   ├── prompts.py               # Template re-exports
│   ├── llm_client.py            # Ollama/OpenAI client
│   ├── data_loader.py           # Dataset handling
│   ├── evaluation.py            # Metrics calculation
│   └── experiment.py            # Main experiment runner
├── notebooks/                   # Analysis notebooks
├── requirements.txt
├── run_test.py                  # Quick setup test
└── README.md
```

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 2. Start Ollama and pull models

```bash
# Start the server
ollama serve

# Pull lightweight models (in another terminal)
ollama pull llama3.2:1b      # 1.3GB, fastest
ollama pull llama3.2:3b      # 2GB, better quality
ollama pull qwen2.5:1.5b     # 1GB, good for testing
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

### Quick Test

```bash
python run_test.py
```

### Full Benchmark

```bash
cd src
python experiment.py
```

### Custom Experiment

```python
from data_loader import Dataset
from experiment import ExperimentRunner

# Load your dataset
dataset = Dataset.load("data/annotated/my_dataset.json")

# Run benchmark
runner = ExperimentRunner()
metrics = runner.run_benchmark(
    dataset=dataset,
    models=["llama3.2:1b", "qwen2.5:1.5b"],
    templates=["zero_shot", "simple_schema", "rich_ontology"],
)
```

## Knowledge Injection Templates

| Template | Description |
|----------|-------------|
| `zero_shot` | Minimal instruction, no schema |
| `simple_schema` | Flat category definitions |
| `rich_ontology` | gUFO-based ontological framework |
| `few_shot` | Examples of correct extractions |
| `chain_of_verification` | Extract then verify against source |

## Evaluation Metrics

- **Accuracy**: % of correctly extracted fields
- **Precision**: Correct / (Correct + Incorrect + Hallucinated)
- **Recall**: Correct / (Correct + Incorrect + Missing)
- **F1 Score**: Harmonic mean of precision and recall
- **JSON Validity Rate**: % of valid JSON outputs
- **Hallucination Rate**: % of fabricated information

## Dataset Format

```json
{
  "name": "dataset_name",
  "incidents": [
    {
      "id": "incident_001",
      "article_text": "Full news article text...",
      "source_url": "https://...",
      "ground_truth": {
        "event": {
          "event_type": "malfunction",
          "event_date": "2024-03-15",
          "event_location": "USA",
          "description": "..."
        },
        "ai_system": {
          "name": "System Name",
          "system_type": "autonomous_vehicle",
          "developer": "Company",
          "deployer": "Company"
        },
        "harm": {
          "harm_type": "physical",
          "severity": "fatal",
          "affected_parties": ["driver"],
          "affected_count": 1
        },
        "organizations": [
          {"name": "Company", "role": "developer"}
        ]
      }
    }
  ]
}
```

## Data Source

Incidents are collected from the [OECD AI Policy Observatory - AI Incidents Monitor](https://oecd.ai/en/incidents):

**Query parameters used:**
- Country: USA
- Date range: 2026-01-01 to 2026-02-01
- Harm level: AI incident
- Results: 20 incidents

**Direct link:** [OECD AI Incidents Query](https://oecd.ai/en/incidents?search_terms=%5B%5D&and_condition=false&countries=USA&from_date=2026-01-01&to_date=2026-02-01&properties_config=%7B%22principles%22:%5B%5D,%22industries%22:%5B%5D,%22harm_types%22:%5B%5D,%22harm_levels%22:%5B%22AI%20incident%22%5D,%22harmed_entities%22:%5B%5D,%22business_functions%22:%5B%5D,%22ai_tasks%22:%5B%5D,%22autonomy_levels%22:%5B%5D,%22languages%22:%5B%5D%7D&order_by=date&num_results=20)

## Adding More Data

1. Collect incidents from [OECD AI Incidents Monitor](https://oecd.ai/en/incidents)
2. Create ground truth annotations following the schema above
3. Save to `data/annotated/your_dataset.json`
4. Run benchmark with your dataset

## Results

Results are saved to `data/results/`:
- `{model}_{template}_{timestamp}_results.json` - Detailed extraction outputs
- `{model}_{template}_{timestamp}_metrics.json` - Aggregated metrics
- `summary_{timestamp}.md` - Comparison report
