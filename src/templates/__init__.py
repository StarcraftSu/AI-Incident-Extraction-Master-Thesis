"""
Prompt templates for AI incident extraction benchmark.

Each template represents a different "knowledge injection" strategy.
"""

from .zero_shot import ZERO_SHOT_TEMPLATE
from .simple_schema import SIMPLE_SCHEMA_TEMPLATE
from .rich_ontology import RICH_ONTOLOGY_TEMPLATE
from .few_shot import FEW_SHOT_TEMPLATE
from .chain_of_verification import CHAIN_OF_VERIFICATION_TEMPLATE

TEMPLATES = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "simple_schema": SIMPLE_SCHEMA_TEMPLATE,
    "rich_ontology": RICH_ONTOLOGY_TEMPLATE,
    "few_shot": FEW_SHOT_TEMPLATE,
    "chain_of_verification": CHAIN_OF_VERIFICATION_TEMPLATE,
}

AVAILABLE_TEMPLATES = list(TEMPLATES.keys())


def get_template(template_name: str) -> str:
    """Get a prompt template by name."""
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {AVAILABLE_TEMPLATES}")
    return TEMPLATES[template_name]


def format_prompt(template_name: str, article_text: str) -> str:
    """Format a prompt template with the article text."""
    template = get_template(template_name)
    return template.format(article_text=article_text)
