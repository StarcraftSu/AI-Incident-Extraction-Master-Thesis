"""
Prompt templates for AI incident extraction benchmark.

This module re-exports from templates/ for convenience.
Supports both legacy templates and new PS × KI modular design.
"""

# Handle imports whether running as package or directly
try:
    from .templates import (
        # New PS × KI design
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
        # Legacy templates
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
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
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
    # New design
    "build_condition_prompt",
    "ALL_CONDITIONS",
    "KI_COMPONENTS",
    "KI_LABELS",
    "PS_LABELS",
    # Legacy
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
