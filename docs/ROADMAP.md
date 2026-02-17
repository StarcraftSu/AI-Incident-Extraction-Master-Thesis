# Development Roadmap

## Overview

This document outlines planned enhancements to make the extraction framework more flexible and extensible.

---

## Phase 1: Add Closed-Source Model Support (Anthropic/Claude)

**Priority:** High
**Effort:** ~2-4 hours
**Breaking Changes:** None (backward compatible)

### Files to Modify

| File | Changes |
|------|---------|
| `src/llm_client.py` | Add `AnthropicClient` class |
| `configs/config.yaml` | Add Anthropic configuration section |
| `src/experiment.py` | Accept `provider` parameter |

### Implementation Details

#### 1. AnthropicClient (src/llm_client.py)

```python
class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models (Haiku, Sonnet, Opus)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        return self.api_key is not None

    def generate(self, model: str, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 2000) -> LLMResponse:
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return LLMResponse(
            text=response.content[0].text,
            model=model,
            usage={"input": response.usage.input_tokens,
                   "output": response.usage.output_tokens}
        )
```

#### 2. Config Update (configs/config.yaml)

```yaml
models:
  ollama:
    - name: "llama3.2:3b"
    - name: "qwen2.5:7b"

  anthropic:
    - name: "claude-haiku-4-5-20251001"
      api_key_env: "ANTHROPIC_API_KEY"
    - name: "claude-sonnet-4-5-20251022"
      api_key_env: "ANTHROPIC_API_KEY"
```

#### 3. Factory Update (src/llm_client.py)

```python
def create_client(provider: str, **kwargs) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### Dependencies

```
pip install anthropic
```

---

## Phase 2: Custom Schema Support

**Priority:** Medium
**Effort:** ~8-12 hours
**Breaking Changes:** Yes (config format change)

### Goal

Allow researchers to define custom extraction fields without modifying code.

### Files to Modify

| File | Changes |
|------|---------|
| `configs/config.yaml` | New declarative schema format |
| `src/template_generator.py` | NEW: Dynamic prompt generation |
| `src/data_loader.py` | Support dynamic schema |
| `src/evaluation.py` | Schema-agnostic metrics |
| `src/templates/*.py` | Migrate to generator functions |

### New Schema Format

```yaml
# Current (hardcoded)
schema:
  event:
    - event_type
    - event_date

# New (declarative)
schema:
  fields:
    - name: event_type
      description: "Type of AI incident"
      type: enum
      values: [malfunction, bias, privacy_breach, misuse, other]
      required: true

    - name: event_date
      description: "Date the incident occurred"
      type: date
      format: "YYYY-MM-DD"
      required: false

    - name: custom_field  # Researcher-defined
      description: "My custom extraction target"
      type: string
      required: false
```

### Template Generator

```python
# src/template_generator.py

def generate_simple_schema_prompt(schema: list[dict], article: str) -> str:
    """Generate simple_schema prompt from dynamic schema."""
    field_definitions = []
    for field in schema:
        if field['type'] == 'enum':
            field_definitions.append(
                f"- {field['name']}: {field['description']} "
                f"(options: {', '.join(field['values'])})"
            )
        else:
            field_definitions.append(
                f"- {field['name']}: {field['description']}"
            )

    return f"""Extract the following fields from this AI incident article:

{chr(10).join(field_definitions)}

Article:
{article}

Output in JSON format."""
```

### Migration Steps

1. Create `src/template_generator.py`
2. Update `configs/config.yaml` schema section
3. Modify `src/data_loader.py` to parse new format
4. Update `src/evaluation.py` to extract field names dynamically
5. Deprecate hardcoded templates, use generator
6. Update tests and documentation

---

## Phase 3: Future Enhancements

- [ ] Batch processing for large datasets
- [ ] Async API calls for parallel model queries
- [ ] Result caching to avoid redundant API calls
- [ ] Web UI for annotation and result visualization
- [ ] Export to OECD AIM format

---

*Created: 2026-02-17*
