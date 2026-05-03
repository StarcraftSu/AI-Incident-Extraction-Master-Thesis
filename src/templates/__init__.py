"""
Prompt templates for AI incident extraction benchmark.

Design: 3 Prompting Strategies × 4 Knowledge Injection levels = 12 conditions.

  PS1: Zero-shot      KI1: No injection
  PS2: Few-shot       KI2: Schema-guided
  PS3: Verification   KI3: Taxonomy-guided
                      KI4: Ontology-guided
"""

from .knowledge_injection import KI_COMPONENTS, KI_LABELS
from .prompting_strategy import (
    PS_BUILDERS,
    PS_LABELS,
    build_ps1_prompt,
    build_ps2_prompt,
    build_ps3_extraction_prompt,
)

ALL_CONDITIONS = [
    f"{ps}_{ki}" for ps in ["PS1", "PS2", "PS3"]
    for ki in ["KI1", "KI2", "KI3", "KI4"]
]


def build_condition_prompt(ps: str, ki: str, article_text: str) -> str:
    """
    Build a prompt for a given PS × KI condition.

    For PS1 and PS2: returns a single prompt string.
    For PS3: returns the extraction prompt (step 1).
             The verification prompt (step 2) is built separately
             after the first extraction completes.
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
