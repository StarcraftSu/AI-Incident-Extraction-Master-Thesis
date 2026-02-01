"""
T2: Simple Schema - structured categories with enumerated values.

This template provides explicit value options for each field,
helping the model choose from predefined categories.
"""

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
