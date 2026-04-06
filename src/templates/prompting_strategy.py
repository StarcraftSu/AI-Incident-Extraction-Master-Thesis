"""
Prompting Strategy (PS) wrappers for prompt construction.

Three strategies that wrap around any KI component:
  PS1: Zero-shot — task instruction + KI + article
  PS2: Few-shot — two worked examples + KI + article
  PS3: Verification (CoVe) — two-step: extract then verify

All prompts share a common structure following prompt engineering
best practices (Anthropic, 2025):
  - Role assignment ("You are an expert AI incident analyst")
  - XML tags separating <instructions>, <article>, <examples>
  - Article text placed before instructions (long context first)
  - Quote-grounding in CoVe verification step

Sources:
  - Brown et al. (2020) — few-shot prompting
  - Dhuliawala et al. (2023) — Chain-of-Verification
  - Chen et al. (2025) — role prompting +20% on AI incidents
  - Grothey et al. (2025) — extraction benchmark
"""

# ---------------------------------------------------------------------------
# Shared role prefix (applied to ALL conditions)
# Justified by Chen et al. (2025): role prompting improves AI incident
# processing by ~20%. This is a fixed component, not an experimental variable.
# ---------------------------------------------------------------------------
ROLE_PREFIX = """You are an expert AI incident analyst. Your task is to extract structured information from AI incident news articles. Extract only what is explicitly stated in the article. If information is not mentioned, write "not stated"."""

# ---------------------------------------------------------------------------
# Few-shot examples (used by PS2)
# Fixed across all KI levels and models to avoid confounding.
# Example 1: physical harm (autonomous vehicle)
# Example 2: rights violation (facial recognition bias)
# Example 3: economic harm (AI-generated fraud)
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

FEW_SHOT_EXAMPLE_3 = {
    "article": """Title: AI-Powered Trading Bot Causes $20 Million Loss for Investors
Summary: An AI-powered trading algorithm deployed by QuantFund Capital malfunctioned during volatile market conditions, executing thousands of unauthorized trades within minutes. The system, developed by AlgoTech Solutions, caused approximately $20 million in losses for retail investors before manual intervention shut it down.
Concepts: AI trading, algorithm, financial loss, malfunction, QuantFund Capital, AlgoTech Solutions, investors""",
    "output": """{
  "event_type": "AI incident",
  "event_date": "not stated",
  "event_location": "not stated",
  "description": "AI trading algorithm malfunctioned during volatile conditions, executing unauthorized trades causing $20 million losses",
  "ai_system_name": "not stated",
  "system_type": "predictive system",
  "developer": "AlgoTech Solutions",
  "deployer": "QuantFund Capital",
  "harm_type": "economic",
  "severity": "severe",
  "affected_parties": "retail investors",
  "affected_count": "not stated",
  "organizations": [
    {"name": "AlgoTech Solutions", "role": "developer"},
    {"name": "QuantFund Capital", "role": "deployer"}
  ]
}""",
}


# ---------------------------------------------------------------------------
# PS1: Zero-shot
# ---------------------------------------------------------------------------
def build_ps1_prompt(ki_component: str, article_text: str) -> str:
    """Zero-shot: role + article (top) + KI instructions (bottom)."""
    return f"""{ROLE_PREFIX}

<article>
{article_text}
</article>

<instructions>
{ki_component}
</instructions>

Return ONLY valid JSON. Do not include any other text."""


# ---------------------------------------------------------------------------
# PS2: Few-shot
# ---------------------------------------------------------------------------
def build_ps2_prompt(ki_component: str, article_text: str) -> str:
    """Few-shot: role + article (top) + 3 examples + KI instructions."""
    return f"""{ROLE_PREFIX}

<article>
{article_text}
</article>

<examples>
<example index="1">
<example_article>
{FEW_SHOT_EXAMPLE_1["article"]}
</example_article>
<example_output>
{FEW_SHOT_EXAMPLE_1["output"]}
</example_output>
</example>

<example index="2">
<example_article>
{FEW_SHOT_EXAMPLE_2["article"]}
</example_article>
<example_output>
{FEW_SHOT_EXAMPLE_2["output"]}
</example_output>
</example>

<example index="3">
<example_article>
{FEW_SHOT_EXAMPLE_3["article"]}
</example_article>
<example_output>
{FEW_SHOT_EXAMPLE_3["output"]}
</example_output>
</example>
</examples>

<instructions>
{ki_component}
</instructions>

Now extract from the article above. Return ONLY valid JSON in the same format as the examples."""


# ---------------------------------------------------------------------------
# PS3: Verification (CoVe) — returns TWO prompts
# ---------------------------------------------------------------------------
def build_ps3_extraction_prompt(ki_component: str, article_text: str) -> str:
    """PS3 step 1: Extract using the KI component (same structure as PS1)."""
    return build_ps1_prompt(ki_component, article_text)


def build_ps3_verification_prompt(article_text: str, extracted_json: dict) -> str:
    """
    PS3 step 2: Verify each field against the source text.

    Uses quote-grounding: the model must cite the relevant passage
    before confirming or rejecting each field. This prevents the model
    from simply rubber-stamping its own prior output.

    IMPORTANT: This prompt does NOT include the original extraction,
    only the article and the fields to verify. This independence is
    the key mechanism by which CoVe reduces hallucination.
    (Dhuliawala et al., 2023)
    """
    questions = []
    for key, value in _flatten_dict(extracted_json).items():
        if value and value != "not stated" and value != "null":
            questions.append(
                f'- Field "{key}" = "{value}": '
                f'First quote the relevant sentence from the article, '
                f'then answer YES or NO.'
            )

    questions_text = "\n".join(questions)

    return f"""{ROLE_PREFIX}

You are now verifying previously extracted information against the source article.

<article>
{article_text}
</article>

<verification_tasks>
For each field below, find and quote the relevant sentence from the article, then determine whether the article explicitly states or directly supports the extracted value.

{questions_text}
</verification_tasks>

<instructions>
After verifying all fields, output the corrected JSON:
- Keep fields verified as YES with their original values.
- Change fields verified as NO to "not stated".
- Keep the same JSON structure.
</instructions>

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
