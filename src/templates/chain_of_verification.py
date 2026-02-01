"""
T5: Chain-of-Verification - extract then verify against source.

This template implements a multi-step verification process:
1. Read and identify information
2. Extract structured data
3. Verify each field against the original text
4. Remove unverified (hallucinated) information

Based on: Grothey et al. (2025) - LLM extraction from pathology reports.
"""

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
