"""
T1: Zero-shot (baseline) - minimal instruction, no schema definitions.

This template provides only field names without explaining what values are expected.
Used as a baseline to measure the impact of knowledge injection.
"""

ZERO_SHOT_TEMPLATE = """Extract information about the AI incident from the following news article.

Return a JSON object with these fields:
- event: type, date, location, description
- ai_system: name, system_type, developer, deployer
- harm: harm_type, severity, affected_parties, affected_count
- organizations: list of {{name, role}}

If information is not mentioned in the article, use null.

Article:
{article_text}

Output only valid JSON, no other text."""
