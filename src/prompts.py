"""
Prompt templates for AI incident extraction benchmark.

Each template is defined in its own file under src/templates/.
This module re-exports them for backward compatibility.
"""

# Handle imports whether running as package or directly
try:
    from .templates import (
        ZERO_SHOT_TEMPLATE,
        SIMPLE_SCHEMA_TEMPLATE,
        RICH_ONTOLOGY_TEMPLATE,
        FEW_SHOT_TEMPLATE,
        CHAIN_OF_VERIFICATION_TEMPLATE,
        TEMPLATES,
        AVAILABLE_TEMPLATES,
        get_template,
        format_prompt,
    )
except ImportError:
    from templates import (
        ZERO_SHOT_TEMPLATE,
        SIMPLE_SCHEMA_TEMPLATE,
        RICH_ONTOLOGY_TEMPLATE,
        FEW_SHOT_TEMPLATE,
        CHAIN_OF_VERIFICATION_TEMPLATE,
        TEMPLATES,
        AVAILABLE_TEMPLATES,
        get_template,
        format_prompt,
    )

__all__ = [
    "ZERO_SHOT_TEMPLATE",
    "SIMPLE_SCHEMA_TEMPLATE",
    "RICH_ONTOLOGY_TEMPLATE",
    "FEW_SHOT_TEMPLATE",
    "CHAIN_OF_VERIFICATION_TEMPLATE",
    "TEMPLATES",
    "AVAILABLE_TEMPLATES",
    "get_template",
    "format_prompt",
]
