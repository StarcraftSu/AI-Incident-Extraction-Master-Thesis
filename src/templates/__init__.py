"""
Prompt templates for AI incident extraction benchmark.

New design (PS × KI):
  3 Prompting Strategies × 4 Knowledge Injection levels = 12 conditions.

Legacy templates (zero_shot, simple_schema, etc.) are preserved for
backward compatibility with old benchmark results.
"""

# New modular design
from .knowledge_injection import KI_COMPONENTS, KI_LABELS
from .prompting_strategy import (
    PS_BUILDERS,
    PS_LABELS,
    build_ps1_prompt,
    build_ps2_prompt,
    build_ps3_extraction_prompt,
    build_ps3_verification_prompt,
)

# Legacy templates (kept for backward compat with old results)
from .zero_shot import ZERO_SHOT_TEMPLATE
from .simple_schema import SIMPLE_SCHEMA_TEMPLATE
from .rich_ontology import RICH_ONTOLOGY_TEMPLATE
from .few_shot import FEW_SHOT_TEMPLATE
from .chain_of_verification import CHAIN_OF_VERIFICATION_TEMPLATE

# Legacy registry
TEMPLATES = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "simple_schema": SIMPLE_SCHEMA_TEMPLATE,
    "rich_ontology": RICH_ONTOLOGY_TEMPLATE,
    "few_shot": FEW_SHOT_TEMPLATE,
    "chain_of_verification": CHAIN_OF_VERIFICATION_TEMPLATE,
}

AVAILABLE_TEMPLATES = list(TEMPLATES.keys())


def get_template(template_name: str) -> str:
    """Get a legacy prompt template by name."""
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {AVAILABLE_TEMPLATES}")
    return TEMPLATES[template_name]


def format_prompt(template_name: str, article_text: str) -> str:
    """Format a legacy prompt template with the article text."""
    template = get_template(template_name)
    return template.format(article_text=article_text)


# ---------------------------------------------------------------------------
# New PS × KI condition builder
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    f"{ps}_{ki}" for ps in ["PS1", "PS2", "PS3"]
    for ki in ["KI1", "KI2", "KI3", "KI4"]
]


def build_condition_prompt(ps: str, ki: str, article_text: str) -> str:
    """
    Build a prompt for a given PS × KI condition.

    For PS1 and PS2: returns a single prompt string.
    For PS3: returns a tuple (extraction_prompt, ) — the verification
             prompt is built later after the first extraction completes.
    """
    ki_component = KI_COMPONENTS[ki]

    if ps == "PS1":
        return build_ps1_prompt(ki_component, article_text)
    elif ps == "PS2":
        return build_ps2_prompt(ki_component, article_text)
    elif ps == "PS3":
        return build_ps3_extraction_prompt(ki_component, article_text)
    else:
        raise ValueError(f"Unknown PS: {ps}. Available: PS1, PS2, PS3")
