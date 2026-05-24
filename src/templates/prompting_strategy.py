import json
import re

"""
Prompting Strategy (PS) wrappers for prompt construction.

Three strategies that wrap around any KI component:
  PS1: Zero-shot — task instruction + KI + article
  PS2: Few-shot — three worked examples + KI + article
  PS3: Verification (CoVe) — three-step: extract, verify independently, revise

All prompts share a common structure following prompt engineering
best practices (Anthropic, 2025):
  - Role assignment ("You are an expert AI incident analyst")
  - XML tags separating <instructions>, <article>, <examples>
  - Article text placed before instructions (long context first)
  - Quote-grounding in CoVe verification step

Sources:
  - Brown et al. (2020) — few-shot prompting
  - Dhuliawala et al. (2023) — Chain-of-Verification (4-step pattern)
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
    "article": """Date: 2024-03-15
Country: United States
Title: Tesla Autopilot Involved in Fatal Highway Crash
Summary: A Tesla Model S crashed into a concrete barrier on Highway 101 while the Autopilot semi-autonomous driving feature was engaged. The 38-year-old driver died in the collision. NHTSA has opened a formal investigation into the incident.
Concepts: Tesla, Autopilot, autonomous driving, fatal crash, NHTSA, investigation""",
    "output": """{
  "event": {
    "event_type": "AI incident",
    "event_date": "2024-03-15",
    "event_location": "Highway 101, United States",
    "description": "Tesla Model S with Autopilot engaged crashed into concrete barrier, killing the driver"
  },
  "ai_system": {
    "name": "Tesla Autopilot",
    "system_type": "autonomous vehicle",
    "developer": "Tesla",
    "deployer": "Tesla"
  },
  "harm": {
    "harm_type": "physical",
    "severity": "severe",
    "affected_parties": "driver"
  },
  "organizations": [
    {"name": "Tesla", "role": "developer"},
    {"name": "NHTSA", "role": "regulator"}
  ]
}""",
}

FEW_SHOT_EXAMPLE_2 = {
    "article": """Date: 2018-07-26
Country: United States
Title: Facial Recognition Software Misidentifies Congress Members as Criminals
Summary: Amazon's Rekognition facial recognition software incorrectly matched 28 members of Congress to criminal mugshots in a test conducted by the ACLU. The false matches disproportionately affected people of color, raising concerns about racial bias in AI systems used by law enforcement.
Concepts: Amazon, Rekognition, facial recognition, bias, ACLU, Congress, racial bias, law enforcement""",
    "output": """{
  "event": {
    "event_type": "AI incident",
    "event_date": "2018-07-26",
    "event_location": "United States",
    "description": "Facial recognition incorrectly matched 28 Congress members to criminal mugshots with racial bias"
  },
  "ai_system": {
    "name": "Amazon Rekognition",
    "system_type": "facial recognition",
    "developer": "Amazon",
    "deployer": "not stated"
  },
  "harm": {
    "harm_type": "rights violation",
    "severity": "significant",
    "affected_parties": "Congress members, people of color"
  },
  "organizations": [
    {"name": "Amazon", "role": "developer"},
    {"name": "ACLU", "role": "other"}
  ]
}""",
}

FEW_SHOT_EXAMPLE_3 = {
    "article": """Date: 2024-09-12
Country: United States
Title: AI-Powered Trading Bot Causes $20 Million Loss for Investors
Summary: An AI-powered trading algorithm deployed by QuantFund Capital malfunctioned during volatile market conditions, executing thousands of unauthorized trades within minutes. The system, developed by AlgoTech Solutions, caused approximately $20 million in losses for retail investors before manual intervention shut it down.
Concepts: AI trading, algorithm, financial loss, malfunction, QuantFund Capital, AlgoTech Solutions, investors""",
    "output": """{
  "event": {
    "event_type": "AI incident",
    "event_date": "2024-09-12",
    "event_location": "United States",
    "description": "AI trading algorithm malfunctioned during volatile conditions, executing unauthorized trades causing $20 million losses"
  },
  "ai_system": {
    "name": "not stated",
    "system_type": "predictive system",
    "developer": "AlgoTech Solutions",
    "deployer": "QuantFund Capital"
  },
  "harm": {
    "harm_type": "economic",
    "severity": "severe",
    "affected_parties": "retail investors"
  },
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
# PS3: Verification (CoVe) — three calls: extract, verify, revise (three builders below)
# ---------------------------------------------------------------------------
def build_ps3_extraction_prompt(ki_component: str, article_text: str) -> str:
    """PS3 step 1: Extract using the KI component (same structure as PS1)."""
    return build_ps1_prompt(ki_component, article_text)


# Field name → open WH question mapping for PS3 step 2 verification.
# Dhuliawala (2023) §4.4 Finding 5: "Open verification questions outperform
# yes/no questions — avoids sycophancy." Handles both flat (KI1) and nested
# (KI2-4) field names produced by step 1.
_FIELD_QUESTIONS = {
    # Flat form (KI1 typical)
    "event_type": "What type of event is described — choose the best-fitting category from the schema's allowed values (AI incident vs. AI hazard) based on whether the article describes actual harm or only a risk.",
    "event_date": "When did the event occur?",
    "event_location": "Where did the event occur?",
    "description": "What happened in the event?",
    "ai_system_name": "What is the name of the AI system involved?",
    "system_type": "What type of AI system is described — choose the single best-fitting category from the schema's allowed values based on what the article says about the system's function.",
    "developer": "Who developed the AI system?",
    "deployer": "Who deployed or used the AI system?",
    "harm_type": "What type of harm is described — choose the single best-fitting category from the schema's allowed values based on what the article says about the harm.",
    "severity": "How severe is the harm described in the article — choose the best-fitting level from the schema's allowed values.",
    "affected_parties": "Who was affected by the harm?",
    "organizations": "What organizations are mentioned in the article and what is each one's role (choose role from the schema's allowed values)?",
    # Nested form (KI2-4 typical)
    "event.event_type": "What type of event is described — choose the best-fitting category from the schema's allowed values (AI incident vs. AI hazard) based on whether the article describes actual harm or only a risk.",
    "event.event_date": "When did the event occur?",
    "event.event_location": "Where did the event occur?",
    "event.description": "What happened in the event?",
    "ai_system.name": "What is the name of the AI system involved?",
    "ai_system.system_type": "What type of AI system is described — choose the single best-fitting category from the schema's allowed values based on what the article says about the system's function.",
    "ai_system.developer": "Who developed the AI system?",
    "ai_system.deployer": "Who deployed or used the AI system?",
    "harm.harm_type": "What type of harm is described — choose the single best-fitting category from the schema's allowed values based on what the article says about the harm.",
    "harm.severity": "How severe is the harm described in the article — choose the best-fitting level from the schema's allowed values.",
    "harm.affected_parties": "Who was affected by the harm?",
    "organizations.name": "What organizations are mentioned in the article?",
    "organizations.role": "What role does each mentioned organization play — choose from the schema's allowed values (developer, deployer, regulator, victim, other).",
}


def _field_to_wh_question(field_name: str) -> str:
    """Map a flat or nested field name to an open WH question."""
    if field_name in _FIELD_QUESTIONS:
        return _FIELD_QUESTIONS[field_name]
    # Fallback for unexpected field names
    return f'What does the article say about "{field_name}"?'


def build_ps3_verification_questions_prompt(
    ki_component: str, article_text: str, draft_json: dict
) -> str:
    """
    PS3 step 2: Independent per-field verification (does NOT see draft values).

    Implements Dhuliawala (2023) CoVe's independence principle: the verification
    model sees only the article, the schema, and a list of open WH questions —
    NOT the draft values from step 1. This prevents the model from rubber-stamping
    its own prior output when answering verification questions.

    Per Dhuliawala (2023) §4.4 Finding 5, questions are phrased as open WH
    questions ("When/Where/Who/What…?") rather than yes/no probes
    ("Does the article state X?") to avoid sycophancy.

    Output: a JSON object mapping field name to verification object
    {"quote": "...", "value": "..."} or {"value": "not stated"}.
    Used downstream as evidence for the step 3 revision prompt.
    """
    # Collect field names only — NOT their values (independence principle)
    field_names = []
    for key in _flatten_dict(draft_json).keys():
        # Collapse organizations[0].name → organizations.name for clean questions
        clean_key = re.sub(r'\[\d+\]\.', '.', key)
        if clean_key not in field_names:
            field_names.append(clean_key)

    questions = []
    for field_name in field_names:
        wh = _field_to_wh_question(field_name)
        questions.append(f'- "{field_name}": {wh}')
    questions_text = "\n".join(questions)

    return f"""{ROLE_PREFIX}

You are independently verifying which fields from a schema are explicitly supported by the source article.
You do NOT see any prior extraction; answer each question from scratch using only the article.

<article>
{article_text}
</article>

<schema>
{ki_component}
</schema>

<verification_tasks>
For each field below, find evidence in the article.

- For free-text fields (names, dates, locations, descriptions, affected parties): extract only what is explicitly stated; otherwise write "not stated".
- For categorical fields (event type, system type, harm type, severity, organization role): map the article's content to the best-fitting allowed value from the schema; if the article describes the relevant content (even without using the exact category label), choose the closest category from the schema. If the entity is clearly described in the article but does not fit any specific category, use the schema's "other" value (when allowed) — do NOT use "not stated" for categorical fields when the entity exists in the article.

{questions_text}
</verification_tasks>

<instructions>
Output a JSON object mapping each field name (as listed above) to a verification object:
- If the article contains direct evidence for the field: {{"quote": "verbatim sentence from the article", "value": "extracted value"}}
- If the article does NOT contain direct evidence: {{"value": "not stated"}}

Example shape:
{{
  "event.event_date": {{"quote": "Date: 2026-03-31", "value": "2026-03-31"}},
  "ai_system.developer": {{"value": "not stated"}}
}}

Return ONLY valid JSON.
</instructions>"""


def build_ps3_revision_prompt(
    ki_component: str,
    article_text: str,
    draft_json: dict,
    verifications_text: str,
) -> str:
    """
    PS3 step 3: Critical revision using independent verifications.

    Implements Dhuliawala (2023) CoVe's final integration step (§3.4): the model
    sees the step 1 draft + the step 2 verifications and produces a revised final.

    Uses anti-rubber-stamp framing in the system prompt: the model is explicitly
    told NOT to defend the draft, to treat it as potentially incorrect, and to
    prefer "not stated" over weak inference. This compensates for the fact that
    step 3 architecturally sees the draft (unlike step 2 which is independent).
    """
    draft_str = json.dumps(draft_json, indent=2)

    return f"""{ROLE_PREFIX}

You are a critical fact-checking system producing a final revised extraction by integrating an initial draft with independent verification results.

Your task is NOT to defend the previous extraction.
Treat the step 1 draft as potentially incorrect, exaggerated, or hallucinated.
Prefer "not stated" over weak inference.
A field is valid ONLY if the verification provides direct quote evidence from the article.

<article>
{article_text}
</article>

<schema>
{ki_component}
</schema>

<step1_draft>
{draft_str}
</step1_draft>

<step2_verifications>
{verifications_text}
</step2_verifications>

<instructions>
Combine the draft and the verifications to produce a final revised JSON.

CRITICAL — STRUCTURE PRESERVATION:
- Use the EXACT SAME JSON structure and key names as in the <step1_draft> above.
- Do NOT rename keys (e.g., do not shorten "event_type" to "type", do not change "harm_type" to "type").
- Do NOT change the shape (e.g., do not turn a single object into an array, or vice versa).
- Only change the VALUES of fields based on the verifications.

Decision rules per field:
- If the verification provides a quote that supports the draft value: keep the draft value.
- If the verification provides a quote that supports a different value: use the verification's value (the draft was wrong).
- If the verification says "not stated" (no quote): write "not stated" for that field, EVEN IF the draft had a value. Do not defend the draft.

Return ONLY valid JSON, in the exact same structure as <step1_draft>.
</instructions>"""


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
    "PS3": None,  # PS3 uses three calls; handled specially in experiment runner
}

PS_LABELS = {
    "PS1": "Zero-shot",
    "PS2": "Few-shot",
    "PS3": "Verification (CoVe)",
}
