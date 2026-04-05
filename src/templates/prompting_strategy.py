"""
Prompting Strategy (PS) wrappers for prompt construction.

Three strategies that wrap around any KI component:
  PS1: Zero-shot — task instruction + KI + article
  PS2: Few-shot — two worked examples + KI + article
  PS3: Verification (CoVe) — two-step: extract then verify

Sources:
  - Brown et al. (2020) — few-shot prompting
  - Dhuliawala et al. (2023) — Chain-of-Verification
  - Grothey et al. (2025) — extraction benchmark
"""

# ---------------------------------------------------------------------------
# Few-shot examples (used by PS2)
# These are fixed across all KI levels and models to avoid confounding.
# Example 1: physical harm (autonomous vehicle)
# Example 2: rights violation (facial recognition bias)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLE_1 = {
    "article": """Title: Tesla Autopilot Involved in Fatal Highway Crash
Summary: A Tesla Model S crashed into a concrete barrier on Highway 101 while the Autopilot semi-autonomous driving feature was engaged. The 38-year-old driver died in the collision. NHTSA has opened a formal investigation into the incident.
Concepts: Tesla, Autopilot, autonomous driving, fatal crash, NHTSA, investigation""",
    "output": """{
  "event_type": "AI incident",
  "event_date": "not stated",
  "event_location": "Highway 101, United States",
  "description": "Tesla Model S with Autopilot engaged crashed into concrete barrier, killing the driver",
  "ai_system_name": "Tesla Autopilot",
  "system_type": "autonomous vehicle",
  "developer": "Tesla",
  "deployer": "Tesla",
  "harm_type": "physical",
  "severity": "severe",
  "affected_parties": "driver",
  "affected_count": "1",
  "organizations": [
    {"name": "Tesla", "role": "developer"},
    {"name": "NHTSA", "role": "regulator"}
  ]
}""",
}

FEW_SHOT_EXAMPLE_2 = {
    "article": """Title: Facial Recognition Software Misidentifies Congress Members as Criminals
Summary: Amazon's Rekognition facial recognition software incorrectly matched 28 members of Congress to criminal mugshots in a test conducted by the ACLU. The false matches disproportionately affected people of color, raising concerns about racial bias in AI systems used by law enforcement.
Concepts: Amazon, Rekognition, facial recognition, bias, ACLU, Congress, racial bias, law enforcement""",
    "output": """{
  "event_type": "AI incident",
  "event_date": "not stated",
  "event_location": "United States",
  "description": "Facial recognition incorrectly matched 28 Congress members to criminal mugshots with racial bias",
  "ai_system_name": "Amazon Rekognition",
  "system_type": "facial recognition",
  "developer": "Amazon",
  "deployer": "not stated",
  "harm_type": "rights violation",
  "severity": "significant",
  "affected_parties": "Congress members, people of color",
  "affected_count": "28",
  "organizations": [
    {"name": "Amazon", "role": "developer"},
    {"name": "ACLU", "role": "other"}
  ]
}""",
}


# ---------------------------------------------------------------------------
# PS1: Zero-shot
# ---------------------------------------------------------------------------
def build_ps1_prompt(ki_component: str, article_text: str) -> str:
    """Zero-shot: task instruction + KI component + article."""
    return f"""{ki_component}

Article:
{article_text}

Return ONLY valid JSON. Do not include any other text."""


# ---------------------------------------------------------------------------
# PS2: Few-shot
# ---------------------------------------------------------------------------
def build_ps2_prompt(ki_component: str, article_text: str) -> str:
    """Few-shot: two worked examples + KI component + article."""
    return f"""{ki_component}

Here are two examples of the expected extraction:

EXAMPLE 1:
Article:
{FEW_SHOT_EXAMPLE_1["article"]}

Output:
{FEW_SHOT_EXAMPLE_1["output"]}

EXAMPLE 2:
Article:
{FEW_SHOT_EXAMPLE_2["article"]}

Output:
{FEW_SHOT_EXAMPLE_2["output"]}

Now extract from this article:
{article_text}

Return ONLY valid JSON in the same format as the examples above."""


# ---------------------------------------------------------------------------
# PS3: Verification (CoVe) — returns TWO prompts
# ---------------------------------------------------------------------------
def build_ps3_extraction_prompt(ki_component: str, article_text: str) -> str:
    """PS3 step 1: Extract using the KI component (same as PS1)."""
    return build_ps1_prompt(ki_component, article_text)


def build_ps3_verification_prompt(article_text: str, extracted_json: dict) -> str:
    """
    PS3 step 2: Verify each field against the source text.

    IMPORTANT: This prompt does NOT include the original extraction,
    only the article and the fields to verify. This independence is
    the key mechanism by which CoVe reduces hallucination.
    (Dhuliawala et al., 2023)
    """
    # Build verification questions for each field
    questions = []
    for key, value in _flatten_dict(extracted_json).items():
        if value and value != "not stated" and value != "null":
            questions.append(
                f'- Field "{key}" = "{value}": '
                f'Does the article explicitly state or directly support this? '
                f'Answer YES (with a brief quote) or NO.'
            )

    questions_text = "\n".join(questions)

    return f"""You are verifying extracted information against a source article.
For each field below, determine whether the article explicitly states or directly supports the extracted value.

Article:
{article_text}

Fields to verify:
{questions_text}

After verifying all fields, output the corrected JSON.
- Keep fields verified as YES with their original values.
- Change fields verified as NO to "not stated".
- Keep the same JSON structure.

Return ONLY valid JSON."""


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, f"{key}[{i}]"))
                else:
                    items[f"{key}[{i}]"] = item
        else:
            items[key] = v
    return items


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PS_BUILDERS = {
    "PS1": build_ps1_prompt,
    "PS2": build_ps2_prompt,
    "PS3": None,  # PS3 uses two-step; handled specially in experiment runner
}

PS_LABELS = {
    "PS1": "Zero-shot",
    "PS2": "Few-shot",
    "PS3": "Verification (CoVe)",
}
