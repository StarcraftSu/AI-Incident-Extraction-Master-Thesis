"""
Data loading utilities for AI incident news articles and annotations.

Reads experimental incidents from an Excel file (experimental_incidents_50.xlsx)
with columns: id, title, date, summary, concepts, companies, country.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import openpyxl


@dataclass
class AIIncident:
    """Represents a single AI incident with article text and metadata."""
    id: str
    article_text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    concepts: Optional[str] = None
    date: Optional[str] = None
    country: Optional[str] = None
    ground_truth: Optional[dict] = None  # Human-annotated extraction

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "article_text": self.article_text,
            "title": self.title,
            "summary": self.summary,
            "concepts": self.concepts,
            "date": self.date,
            "country": self.country,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AIIncident":
        return cls(
            id=data["id"],
            article_text=data["article_text"],
            title=data.get("title"),
            summary=data.get("summary"),
            concepts=data.get("concepts"),
            date=data.get("date"),
            country=data.get("country"),
            ground_truth=data.get("ground_truth"),
        )


@dataclass
class Dataset:
    """A collection of AI incidents."""
    name: str
    incidents: list[AIIncident] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.incidents)

    def __iter__(self):
        return iter(self.incidents)

    def __getitem__(self, idx: int) -> AIIncident:
        return self.incidents[idx]

    def add(self, incident: AIIncident):
        self.incidents.append(incident)

    def save(self, filepath: str):
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "count": len(self.incidents),
            "incidents": [inc.to_dict() for inc in self.incidents],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.incidents)} incidents to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Dataset":
        """Load dataset from JSON file."""
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", filepath.stem))
        for inc_data in data.get("incidents", []):
            dataset.add(AIIncident.from_dict(inc_data))

        print(f"Loaded {len(dataset)} incidents from {filepath}")
        return dataset

    def split(self, train_ratio: float = 0.7) -> tuple["Dataset", "Dataset"]:
        """Split dataset into train and test sets."""
        split_idx = int(len(self.incidents) * train_ratio)
        train = Dataset(name=f"{self.name}_train", incidents=self.incidents[:split_idx])
        test = Dataset(name=f"{self.name}_test", incidents=self.incidents[split_idx:])
        return train, test


def load_dataset(path: str) -> Dataset:
    """
    Load incidents from an Excel file (.xlsx).

    Expected columns: A=id, B=title, C=date, D=summary, E=concepts, F=companies, G=country.
    Constructs article_text by combining: "Title: {title}\\nSummary: {summary}\\nConcepts: {concepts}"

    Args:
        path: Path to the Excel file.

    Returns:
        A Dataset containing AIIncident objects.
    """
    filepath = Path(path)
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    dataset = Dataset(name=filepath.stem)

    rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
    for row in rows:
        if not row or not row[0]:
            continue

        inc_id = str(row[0]).strip()
        title = str(row[1]).strip() if row[1] else ""
        date = str(row[2]).strip() if row[2] else ""
        summary = str(row[3]).strip() if row[3] else ""
        concepts = str(row[4]).strip() if row[4] else ""
        # companies = row[5]  # available but not used in article_text
        country = str(row[6]).strip() if row[6] else ""

        article_text = f"Title: {title}\nSummary: {summary}\nConcepts: {concepts}"

        dataset.add(AIIncident(
            id=inc_id,
            article_text=article_text,
            title=title,
            summary=summary,
            concepts=concepts,
            date=date,
            country=country,
            ground_truth=None,
        ))

    wb.close()
    print(f"Loaded {len(dataset)} incidents from {filepath}")
    return dataset


def create_empty_ground_truth() -> dict:
    """Create an empty ground truth template for annotation.

    Allowed values:
      - event_type: "AI incident" or "AI hazard"
      - system_type: facial recognition, recommendation system, generative AI,
            autonomous vehicle, decision support, chatbot, content moderation,
            predictive system, ai agent, other
      - harm_type: physical, psychological, reputational, economic,
            environmental, rights violation, other
      - severity: minor, moderate, significant, severe
      - org role: developer, deployer, regulator, victim, other
    """
    return {
        "event": {
            "event_type": None,      # "AI incident" or "AI hazard"
            "event_date": None,      # YYYY-MM-DD
            "event_location": None,
            "description": None,
        },
        "ai_system": {
            "name": None,
            "system_type": None,     # from allowed values above
            "developer": None,
            "deployer": None,
        },
        "harm": {
            "harm_type": None,       # from allowed values above
            "severity": None,        # minor, moderate, significant, severe
            "affected_parties": [],
        },
        "organizations": [],         # List of {"name": "", "role": ""}
    }


def create_sample_dataset() -> Dataset:
    """Create a sample dataset with a few AI incidents for testing.

    All annotation values use ONLY allowed constrained values:
      - event_type: "AI incident" | "AI hazard"
      - system_type: facial recognition | recommendation system | generative AI |
            autonomous vehicle | decision support | chatbot | content moderation |
            predictive system | ai agent | other
      - harm_type: physical | psychological | reputational | economic |
            environmental | rights violation | other
      - severity: minor | moderate | significant | severe
      - org role: developer | deployer | regulator | victim | other
    """
    dataset = Dataset(name="sample_ai_incidents")

    # Sample 1: Tesla Autopilot crash
    dataset.add(AIIncident(
        id="sample_001",
        article_text="""Title: Tesla Autopilot Involved in Fatal Highway Crash
Summary: A Tesla Model S crashed into a concrete barrier on Highway 101 while the Autopilot semi-autonomous driving feature was engaged. The 38-year-old driver died in the collision. NHTSA has opened a formal investigation into the incident.
Concepts: Tesla, Autopilot, autonomous driving, fatal crash, NHTSA, investigation""",
        title="Tesla Autopilot Involved in Fatal Highway Crash",
        summary="A Tesla Model S crashed into a concrete barrier on Highway 101 while the Autopilot semi-autonomous driving feature was engaged.",
        concepts="Tesla, Autopilot, autonomous driving, fatal crash, NHTSA, investigation",
        date="2024-03-16",
        country="United States",
        ground_truth={
            "event": {
                "event_type": "AI incident",
                "event_date": "not stated",
                "event_location": "Highway 101, United States",
                "description": "Tesla Model S with Autopilot crashed into concrete barrier, killing the driver",
            },
            "ai_system": {
                "name": "Tesla Autopilot",
                "system_type": "autonomous vehicle",
                "developer": "Tesla",
                "deployer": "Tesla",
            },
            "harm": {
                "harm_type": "physical",
                "severity": "severe",
                "affected_parties": "driver",
            },
            "organizations": [
                {"name": "Tesla", "role": "developer"},
                {"name": "NHTSA", "role": "regulator"},
            ],
        },
    ))

    # Sample 2: Facial recognition bias
    dataset.add(AIIncident(
        id="sample_002",
        article_text="""Title: Facial Recognition Software Misidentifies Congress Members as Criminals
Summary: Amazon's Rekognition facial recognition software incorrectly matched 28 members of Congress to criminal mugshots in a test conducted by the ACLU. The false matches disproportionately affected people of color, raising concerns about racial bias in AI systems used by law enforcement.
Concepts: Amazon, Rekognition, facial recognition, bias, ACLU, Congress, racial bias, law enforcement""",
        title="Facial Recognition Software Misidentifies Congress Members as Criminals",
        summary="Amazon's Rekognition facial recognition software incorrectly matched 28 members of Congress to criminal mugshots.",
        concepts="Amazon, Rekognition, facial recognition, bias, ACLU, Congress, racial bias, law enforcement",
        date="2018-07-26",
        country="United States",
        ground_truth={
            "event": {
                "event_type": "AI incident",
                "event_date": "not stated",
                "event_location": "United States",
                "description": "Facial recognition incorrectly matched 28 Congress members to criminal mugshots with racial bias",
            },
            "ai_system": {
                "name": "Amazon Rekognition",
                "system_type": "facial recognition",
                "developer": "Amazon",
                "deployer": "not stated",
            },
            "harm": {
                "harm_type": "rights violation",
                "severity": "significant",
                "affected_parties": "Congress members, people of color",
            },
            "organizations": [
                {"name": "Amazon", "role": "developer"},
                {"name": "ACLU", "role": "other"},
            ],
        },
    ))

    # Sample 3: ChatGPT data leak
    dataset.add(AIIncident(
        id="sample_003",
        article_text="""Title: ChatGPT Bug Exposes User Data
Summary: OpenAI disclosed a data breach where a bug in ChatGPT's Redis client library caused some users to see chat history titles belonging to other users. Payment information of about 1.2% of ChatGPT Plus subscribers may have been exposed, including names, email addresses, and partial credit card numbers. OpenAI took ChatGPT offline to fix the issue.
Concepts: OpenAI, ChatGPT, data breach, privacy, payment information""",
        title="ChatGPT Bug Exposes User Data",
        summary="OpenAI disclosed a data breach where a bug in ChatGPT exposed user data.",
        concepts="OpenAI, ChatGPT, data breach, privacy, payment information",
        date="2023-03-24",
        country="not stated",
        ground_truth={
            "event": {
                "event_type": "AI incident",
                "event_date": "2023-03-20",
                "event_location": "not stated",
                "description": "Bug exposed users' chat histories and payment information",
            },
            "ai_system": {
                "name": "ChatGPT",
                "system_type": "chatbot",
                "developer": "OpenAI",
                "deployer": "OpenAI",
            },
            "harm": {
                "harm_type": "rights violation",
                "severity": "moderate",
                "affected_parties": "ChatGPT Plus subscribers",
            },
            "organizations": [
                {"name": "OpenAI", "role": "developer"},
            ],
        },
    ))

    # Sample 4: YouTube recommendation algorithm
    dataset.add(AIIncident(
        id="sample_004",
        article_text="""Title: YouTube Algorithm Promotes Extremist Content
Summary: YouTube's recommendation algorithm was found to be promoting conspiracy theory videos and extremist content to users in a study published by researchers at UC Berkeley. The study analyzed over 300,000 videos and found that the algorithm consistently recommended increasingly extreme content to users who watched political videos.
Concepts: YouTube, recommendation algorithm, extremist content, conspiracy theories, UC Berkeley""",
        title="YouTube Algorithm Promotes Extremist Content",
        summary="YouTube's recommendation algorithm was found to be promoting extremist content.",
        concepts="YouTube, recommendation algorithm, extremist content, conspiracy theories, UC Berkeley",
        date="2020-01-15",
        country="not stated",
        ground_truth={
            "event": {
                "event_type": "AI hazard",
                "event_date": "not stated",
                "event_location": "not stated",
                "description": "YouTube algorithm promoted extremist and conspiracy content",
            },
            "ai_system": {
                "name": "YouTube recommendation algorithm",
                "system_type": "recommendation system",
                "developer": "YouTube",
                "deployer": "YouTube",
            },
            "harm": {
                "harm_type": "psychological",
                "severity": "moderate",
                "affected_parties": "YouTube users",
            },
            "organizations": [
                {"name": "YouTube", "role": "developer"},
                {"name": "UC Berkeley", "role": "other"},
            ],
        },
    ))

    # Sample 5: Hiring algorithm bias
    dataset.add(AIIncident(
        id="sample_005",
        article_text="""Title: Amazon Scraps AI Recruiting Tool Over Gender Bias
Summary: Amazon scrapped a secret AI recruiting tool after discovering it showed bias against women. The system was designed to review resumes and rate candidates, but it penalized resumes containing words like "women's" and downgraded graduates of all-women's colleges. The tool was never used as the sole basis for hiring decisions.
Concepts: Amazon, AI recruiting, gender bias, hiring, discrimination""",
        title="Amazon Scraps AI Recruiting Tool Over Gender Bias",
        summary="Amazon scrapped a secret AI recruiting tool after discovering it showed bias against women.",
        concepts="Amazon, AI recruiting, gender bias, hiring, discrimination",
        date="2018-10-10",
        country="not stated",
        ground_truth={
            "event": {
                "event_type": "AI incident",
                "event_date": "not stated",
                "event_location": "not stated",
                "description": "AI recruiting tool showed bias against women in resume screening",
            },
            "ai_system": {
                "name": "Amazon AI recruiting tool",
                "system_type": "predictive system",
                "developer": "Amazon",
                "deployer": "Amazon",
            },
            "harm": {
                "harm_type": "rights violation",
                "severity": "moderate",
                "affected_parties": "women job applicants, graduates of women's colleges",
            },
            "organizations": [
                {"name": "Amazon", "role": "developer"},
            ],
        },
    ))

    return dataset


if __name__ == "__main__":
    import sys

    # If an Excel path is provided, load from it; otherwise use sample dataset
    if len(sys.argv) > 1:
        ds = load_dataset(sys.argv[1])
    else:
        ds = create_sample_dataset()
        ds.save("data/annotated/sample_dataset.json")

    print(f"\nFirst incident: {ds[0].id}")
    print(f"Article preview: {ds[0].article_text[:100]}...")
