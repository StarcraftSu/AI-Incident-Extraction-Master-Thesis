"""
Prompt templates for AI incident extraction benchmark.

Each template represents a different "knowledge injection" strategy.
"""

# T1: Zero-shot (baseline) - minimal instruction
ZERO_SHOT_TEMPLATE = """Extract information about the AI incident from the following news article.

Return a JSON object with these fields:
- event: type, date, location, description
- ai_system: name, system_type, developer, deployer
- harm: harm_type, severity, affected_parties, affected_count
- organizations: list of {name, role}

If information is not mentioned in the article, use null.

Article:
{article_text}

Output only valid JSON, no other text."""


# T2: Simple Schema - structured categories
SIMPLE_SCHEMA_TEMPLATE = """Extract information from this AI incident news article according to the schema below.

SCHEMA:
{{
  "event": {{
    "event_type": "malfunction | bias | privacy_breach | misuse | security | other",
    "event_date": "YYYY-MM-DD or null",
    "event_location": "location string or null",
    "description": "brief description"
  }},
  "ai_system": {{
    "name": "system name",
    "system_type": "autonomous_vehicle | facial_recognition | recommendation | chatbot | other",
    "developer": "company name or null",
    "deployer": "company name or null"
  }},
  "harm": {{
    "harm_type": "physical | psychological | financial | reputational | societal | privacy",
    "severity": "minor | moderate | severe | fatal | unknown",
    "affected_parties": ["list of affected groups"],
    "affected_count": "number or null"
  }},
  "organizations": [
    {{"name": "org name", "role": "developer | deployer | regulator | victim | investigator"}}
  ]
}}

Article:
{article_text}

Output only valid JSON matching the schema above."""


# T3: Rich Ontology (gUFO-based) - detailed ontological categories
RICH_ONTOLOGY_TEMPLATE = """Extract information from this AI incident news article using the following ontological framework.

ONTOLOGICAL CATEGORIES:

1. OBJECTS (Endurants - things that persist through time):
   - AI_System: An artificial intelligence system involved in the incident
     * name: The specific name of the AI system
     * system_type: [autonomous_vehicle | facial_recognition | recommendation_system | chatbot | content_moderation | predictive_system | generative_ai | other]
     * developer: Organization that created the system
     * deployer: Organization that operates/deployed the system
     * status: Current operational status if mentioned

   - Person/Group: Humans affected by or involved in the incident
     * role: [victim | operator | developer | regulator | witness]
     * outcome: What happened to them

   - Organization: Companies, agencies, or institutions involved
     * name: Organization name
     * type: [company | government_agency | research_institution | ngo]
     * role_in_incident: [developer | deployer | regulator | victim | investigator]

2. EVENT (Perdurant - what unfolded over time):
   - event_type: [malfunction | bias | privacy_breach | misuse | security_vulnerability | unintended_consequence | other]
   - temporal:
     * start_date: When the incident began (YYYY-MM-DD or null)
     * end_date: When it ended if applicable (YYYY-MM-DD or null)
   - location:
     * country: Country where it occurred
     * region: State/province/city
     * specific_place: Specific location if mentioned

3. PARTICIPATION (How objects relate to the event):
   - participated_in: Which objects were directly involved
   - was_created_in: What emerged from the event (e.g., investigation, lawsuit)
   - was_terminated_in: What ended due to the event (e.g., service shutdown, contract)

4. HARM (Consequences of the event):
   - harm_type: [physical | psychological | financial | reputational | societal | privacy | none]
   - severity: [minor | moderate | severe | fatal | unknown]
   - affected_parties: List of who/what was harmed
   - affected_count: Number of affected entities if mentioned

5. CAUSAL_CHAIN (If discernible):
   - trigger: What initiated the incident
   - contributing_factors: Other factors that contributed
   - immediate_consequence: Direct result
   - long_term_consequence: Longer-term impacts if mentioned

OUTPUT FORMAT (JSON):
{{
  "event": {{
    "event_type": "",
    "description": "",
    "temporal": {{"start_date": null, "end_date": null}},
    "location": {{"country": null, "region": null, "specific_place": null}}
  }},
  "ai_system": {{
    "name": "",
    "system_type": "",
    "developer": null,
    "deployer": null,
    "status": null
  }},
  "harm": {{
    "harm_type": "",
    "severity": "",
    "affected_parties": [],
    "affected_count": null
  }},
  "organizations": [],
  "participation": {{
    "participated_in": [],
    "was_created_in": [],
    "was_terminated_in": []
  }},
  "causal_chain": {{
    "trigger": null,
    "contributing_factors": [],
    "immediate_consequence": null,
    "long_term_consequence": null
  }}
}}

Article:
{article_text}

Extract ONLY information explicitly stated in the article. Use null for missing information. Output only valid JSON."""


# T4: Few-shot - with examples
FEW_SHOT_TEMPLATE = """Extract information about AI incidents from news articles. Here are examples:

EXAMPLE 1:
Article: "Tesla's Autopilot system was involved in a fatal crash on Highway 101 on March 15, 2024. The 38-year-old driver died when his Model S collided with a concrete barrier while the semi-autonomous driving feature was engaged. NHTSA has opened an investigation."

Output:
{{
  "event": {{
    "event_type": "malfunction",
    "event_date": "2024-03-15",
    "event_location": "Highway 101, USA",
    "description": "Tesla Model S with Autopilot engaged crashed into concrete barrier"
  }},
  "ai_system": {{
    "name": "Tesla Autopilot",
    "system_type": "autonomous_vehicle",
    "developer": "Tesla",
    "deployer": "Tesla"
  }},
  "harm": {{
    "harm_type": "physical",
    "severity": "fatal",
    "affected_parties": ["driver"],
    "affected_count": 1
  }},
  "organizations": [
    {{"name": "Tesla", "role": "developer"}},
    {{"name": "NHTSA", "role": "investigator"}}
  ]
}}

EXAMPLE 2:
Article: "Amazon's facial recognition software, Rekognition, incorrectly matched 28 members of Congress to criminal mugshots in a test conducted by the ACLU in July 2018. The false matches disproportionately affected people of color."

Output:
{{
  "event": {{
    "event_type": "bias",
    "event_date": "2018-07-01",
    "event_location": "USA",
    "description": "Facial recognition incorrectly matched Congress members to mugshots with racial bias"
  }},
  "ai_system": {{
    "name": "Amazon Rekognition",
    "system_type": "facial_recognition",
    "developer": "Amazon",
    "deployer": null
  }},
  "harm": {{
    "harm_type": "reputational",
    "severity": "moderate",
    "affected_parties": ["Congress members", "people of color"],
    "affected_count": 28
  }},
  "organizations": [
    {{"name": "Amazon", "role": "developer"}},
    {{"name": "ACLU", "role": "investigator"}}
  ]
}}

Now extract from this article:
{article_text}

Output only valid JSON in the same format as the examples above."""


# T5: Chain-of-Verification - extract then verify
CHAIN_OF_VERIFICATION_TEMPLATE = """You will extract information from an AI incident news article using a verification process.

STEP 1: Read the article carefully and identify all relevant information.

STEP 2: Extract the following information:
- Event details (type, date, location, description)
- AI system (name, type, developer, deployer)
- Harm caused (type, severity, who was affected, how many)
- Organizations involved (name and their role)

STEP 3: For EACH extracted piece of information, verify it against the original article:
- Is this information EXPLICITLY stated in the article?
- If you inferred it, mark it as null instead

STEP 4: Output ONLY verified information. Use null for anything not explicitly stated.

Article:
{article_text}

Now perform the extraction with verification. Think through each step, then output the final JSON:

{{
  "event": {{
    "event_type": "malfunction | bias | privacy_breach | misuse | security | other",
    "event_date": "YYYY-MM-DD or null",
    "event_location": "location or null",
    "description": "verified description"
  }},
  "ai_system": {{
    "name": "verified name or null",
    "system_type": "verified type or null",
    "developer": "verified developer or null",
    "deployer": "verified deployer or null"
  }},
  "harm": {{
    "harm_type": "verified type or null",
    "severity": "verified severity or unknown",
    "affected_parties": ["verified parties"],
    "affected_count": "verified number or null"
  }},
  "organizations": [
    {{"name": "verified name", "role": "verified role"}}
  ]
}}

First show your verification reasoning, then output the final JSON after "FINAL JSON:"."""


def get_template(template_name: str) -> str:
    """Get a prompt template by name."""
    templates = {
        "zero_shot": ZERO_SHOT_TEMPLATE,
        "simple_schema": SIMPLE_SCHEMA_TEMPLATE,
        "rich_ontology": RICH_ONTOLOGY_TEMPLATE,
        "few_shot": FEW_SHOT_TEMPLATE,
        "chain_of_verification": CHAIN_OF_VERIFICATION_TEMPLATE,
    }
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    return templates[template_name]


def format_prompt(template_name: str, article_text: str) -> str:
    """Format a prompt template with the article text."""
    template = get_template(template_name)
    return template.format(article_text=article_text)


# List all available templates
AVAILABLE_TEMPLATES = [
    "zero_shot",
    "simple_schema",
    "rich_ontology",
    "few_shot",
    "chain_of_verification",
]
