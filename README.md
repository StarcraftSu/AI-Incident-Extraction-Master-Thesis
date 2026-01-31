# AI Incident Extraction Benchmark

Benchmark different LLM prompting strategies for extracting structured data from AI incident news articles.

## Research Question

> How to inject knowledge (prompting) into LLMs to improve their accuracy in extracting risk information from AI incident news?

## Project Structure

```
ai_incident_extraction/
├── configs/
│   └── config.yaml          # Experiment configuration
├── data/
│   ├── raw/                  # Raw news articles
│   ├── annotated/            # Human-annotated ground truth
│   └── results/              # Experiment outputs
├── prompts/                  # Additional prompt templates
├── src/
│   ├── prompts.py           # Knowledge injection templates
│   ├── llm_client.py        # Ollama/OpenAI client
│   ├── data_loader.py       # Dataset handling
│   ├── evaluation.py        # Metrics calculation
│   └── experiment.py        # Main experiment runner
├── notebooks/               # Analysis notebooks
├── requirements.txt
├── run_test.py             # Quick setup test
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

## Adding Real Data

1. Collect news articles from [AI Incident Database](https://incidentdatabase.ai/)
2. Create ground truth annotations following the schema above
3. Save to `data/annotated/your_dataset.json`
4. Run benchmark with your dataset

## Results

Results are saved to `data/results/`:
- `{model}_{template}_{timestamp}_results.json` - Detailed extraction outputs
- `{model}_{template}_{timestamp}_metrics.json` - Aggregated metrics
- `summary_{timestamp}.md` - Comparison report
