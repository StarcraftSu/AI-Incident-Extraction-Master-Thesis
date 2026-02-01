"""
T4: Few-shot - learning from examples.

This template provides 2 complete extraction examples before the target article,
allowing the model to learn the expected output format and extraction patterns.
"""

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
