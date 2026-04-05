"""
Knowledge Injection (KI) components for prompt construction.

Four levels of structural expressiveness, each building on the previous:
  KI1: No injection — field names only
  KI2: Schema-guided — field definitions + enumerated values
  KI3: Taxonomy-guided — schema + hierarchical category labels
  KI4: Ontology-guided — taxonomy + relational constraints

Sources:
  - OECD Common Reporting Framework (Perset, Aranda & Rispal, 2025)
  - CSET AI Harm Taxonomy (Hoffman & Frase, 2023)
  - GMF Taxonomy (Pittaras & McGregor, 2022)
  - gOCED / gUFO ontology (Hooshyar et al., 2025)
  - ODKE+ approach (Khorshidi et al., 2025)
"""

# ---------------------------------------------------------------------------
# KI1: No injection — field names and JSON format only
# ---------------------------------------------------------------------------
KI1_NO_INJECTION = """Extract the following fields from the AI incident article below and return valid JSON:
event_type, event_date, event_location, description,
ai_system_name, system_type, developer, deployer,
harm_type, severity, affected_parties, affected_count,
organizations (name and role for each).

If information is not mentioned in the article, use null."""


# ---------------------------------------------------------------------------
# KI2: Schema-guided — field definitions with data types and enums
# ---------------------------------------------------------------------------
KI2_SCHEMA = """Extract information using the schema below. Use ONLY the allowed values where specified.
If a field cannot be determined from the article, write "not stated".

Fields:
- event_type (string): one of ["AI incident", "AI hazard"]
- event_date (string): date in YYYY-MM-DD format, or "not stated"
- event_location (string): country or city name
- description (string): one-sentence summary of what happened
- ai_system_name (string): name of the AI system, if mentioned
- system_type (string): one of ["facial recognition", "recommendation system", "generative AI", "autonomous vehicle", "decision support", "chatbot", "content moderation", "predictive system", "other"]
- developer (string): organization that built the AI system
- deployer (string): organization that used/deployed the AI system
- harm_type (string): one of ["physical", "psychological", "reputational", "economic", "environmental", "rights violation", "other"]
- severity (string): one of ["minor", "moderate", "significant", "severe"]
- affected_parties (string): who was harmed
- affected_count (string): number of people affected, if stated
- organizations (array of objects): each with "name" (string) and "role" (string, one of ["developer", "deployer", "regulator", "victim", "other"])"""


# ---------------------------------------------------------------------------
# KI3: Taxonomy-guided — schema + hierarchical category labels
# ---------------------------------------------------------------------------
KI3_TAXONOMY = KI2_SCHEMA + """

TAXONOMIES — select the most specific applicable category:

harm_type taxonomy:
  Physical harm
    ├── Injury (bodily harm to individuals)
    ├── Death (fatalities caused by or related to AI system)
    └── Health impact (long-term health consequences)
  Psychological harm
    ├── Distress (emotional suffering, anxiety)
    ├── Manipulation (deception, coercion via AI)
    └── Dignity violation (humiliation, dehumanization)
  Economic harm
    ├── Financial loss (direct monetary damage)
    ├── Property damage (destruction of physical assets)
    └── Job displacement (employment loss due to AI)
  Reputational harm
    └── Defamation or identity misuse
  Rights violation
    ├── Privacy breach (unauthorized data use or surveillance)
    ├── Discrimination (biased decisions based on protected attributes)
    └── Due process violation (denial of fair treatment)
  Environmental harm
    └── Ecological damage or resource waste

severity taxonomy:
  Minor — no lasting impact on individuals
  Moderate — recoverable harm to individuals
  Significant — lasting harm to individuals or groups
  Severe — widespread, irreversible, or life-threatening harm

system_type taxonomy:
  Recognition / detection
    ├── Facial recognition
    ├── Object detection
    └── Speech recognition
  Decision support
    ├── Recommendation system
    ├── Predictive system
    └── Content moderation
  Autonomous systems
    ├── Autonomous vehicle
    └── Robotic system
  Generative AI
    ├── Text generation (chatbot, LLM)
    ├── Image generation
    └── Video / audio generation (deepfake)
  Other"""


# ---------------------------------------------------------------------------
# KI4: Ontology-guided — taxonomy + relational constraints
# ---------------------------------------------------------------------------
KI4_ONTOLOGY = KI3_TAXONOMY + """

RELATIONAL CONSTRAINTS — ensure your output respects these:

ENTITIES AND THEIR TYPES:
- AI_SYSTEM (endurant): a persistent technical artifact
- ORGANIZATION (endurant): a company, agency, or institution
- PERSON_GROUP (endurant): affected individuals or communities
- INCIDENT (event): what happened, unfolding over time

RELATIONSHIPS:
- An INCIDENT involves exactly one AI_SYSTEM
- The AI_SYSTEM was developed_by one ORGANIZATION (the developer)
- The AI_SYSTEM was deployed_by one ORGANIZATION (the deployer)
  Note: developer and deployer may be different entities
- The INCIDENT caused one or more HARMs
- Each HARM has a type (from the taxonomy above) and a severity
- Each HARM affects one or more PERSON_GROUPs
- If stated, the INCIDENT has a CAUSE that links the AI_SYSTEM behavior to the HARM (e.g., "the facial recognition system misidentified the person, leading to wrongful arrest")

CONSTRAINTS:
- developer and deployer must be organizations, not people
- affected_parties must be the group harmed, not the developer
- if the article does not state a relationship, write "not stated" rather than inferring"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
KI_COMPONENTS = {
    "KI1": KI1_NO_INJECTION,
    "KI2": KI2_SCHEMA,
    "KI3": KI3_TAXONOMY,
    "KI4": KI4_ONTOLOGY,
}

KI_LABELS = {
    "KI1": "No injection",
    "KI2": "Schema-guided",
    "KI3": "Taxonomy-guided",
    "KI4": "Ontology-guided",
}
