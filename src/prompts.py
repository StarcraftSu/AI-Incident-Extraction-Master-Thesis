"""
Prompt templates for AI incident extraction benchmark.

Re-exports from templates/ for convenience.
"""

try:
    from .templates import (
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
    )
except ImportError:
    from templates import (
        build_condition_prompt,
        ALL_CONDITIONS,
        KI_COMPONENTS,
        KI_LABELS,
        PS_LABELS,
    )

__all__ = [
    "build_condition_prompt",
    "ALL_CONDITIONS",
    "KI_COMPONENTS",
    "KI_LABELS",
    "PS_LABELS",
]
