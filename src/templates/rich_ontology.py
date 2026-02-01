"""
T3: Rich Ontology (gUFO-based) - detailed ontological framework.

This template uses concepts from the Unified Foundational Ontology (UFO):
- Endurants (objects that persist through time)
- Perdurants (events that unfold over time)
- Participation relations
- Causal chains

Based on: Hooshyar et al. (2025) - gOCED ontology for event data.
"""

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
