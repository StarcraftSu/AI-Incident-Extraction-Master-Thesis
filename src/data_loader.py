"""
Data loading utilities for AI incident news articles and annotations.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AIIncident:
    """Represents a single AI incident with article and ground truth."""
    id: str
    article_text: str
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    publish_date: Optional[str] = None
    ground_truth: Optional[dict] = None  # Human-annotated extraction

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "article_text": self.article_text,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "publish_date": self.publish_date,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AIIncident":
        return cls(
            id=data["id"],
            article_text=data["article_text"],
            source_url=data.get("source_url"),
            source_name=data.get("source_name"),
            publish_date=data.get("publish_date"),
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

    def save(self, filepath: str | Path):
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
    def load(cls, filepath: str | Path) -> "Dataset":
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


def create_empty_ground_truth() -> dict:
    """Create an empty ground truth template for annotation."""
    return {
        "event": {
            "event_type": None,  # malfunction, bias, privacy_breach, misuse, security, other
            "event_date": None,  # YYYY-MM-DD
            "event_location": None,
            "description": None,
        },
        "ai_system": {
            "name": None,
            "system_type": None,  # autonomous_vehicle, facial_recognition, etc.
            "developer": None,
            "deployer": None,
        },
        "harm": {
            "harm_type": None,  # physical, psychological, financial, reputational, societal, privacy
            "severity": None,  # minor, moderate, severe, fatal, unknown
            "affected_parties": [],
            "affected_count": None,
        },
        "organizations": [],  # List of {"name": "", "role": ""}
    }


def create_sample_dataset() -> Dataset:
    """Create a sample dataset with a few AI incidents for testing."""
    dataset = Dataset(name="sample_ai_incidents")

    # Sample 1: Tesla Autopilot crash
    dataset.add(AIIncident(
        id="sample_001",
        article_text="""Tesla's Autopilot system was involved in a fatal crash on Highway 101
        on March 15, 2024. The 38-year-old driver, John Smith, died when his Model S collided
        with a concrete barrier while the semi-autonomous driving feature was engaged.
        The National Highway Traffic Safety Administration (NHTSA) has opened an investigation.
        Tesla stated that the driver had received multiple warnings to keep hands on the wheel.
        This marks the third fatal incident involving Tesla's Autopilot this year.""",
        source_url="https://example.com/tesla-crash",
        source_name="Tech News Daily",
        publish_date="2024-03-16",
        ground_truth={
            "event": {
                "event_type": "malfunction",
                "event_date": "2024-03-15",
                "event_location": "Highway 101, USA",
                "description": "Tesla Model S with Autopilot crashed into concrete barrier",
            },
            "ai_system": {
                "name": "Tesla Autopilot",
                "system_type": "autonomous_vehicle",
                "developer": "Tesla",
                "deployer": "Tesla",
            },
            "harm": {
                "harm_type": "physical",
                "severity": "fatal",
                "affected_parties": ["driver"],
                "affected_count": 1,
            },
            "organizations": [
                {"name": "Tesla", "role": "developer"},
                {"name": "NHTSA", "role": "investigator"},
            ],
        },
    ))

    # Sample 2: Facial recognition bias
    dataset.add(AIIncident(
        id="sample_002",
        article_text="""Amazon's facial recognition software, Rekognition, incorrectly matched
        28 members of Congress to criminal mugshots in a test conducted by the ACLU in July 2018.
        The false matches disproportionately affected people of color, with nearly 40% of the
        incorrect matches being minorities despite making up only 20% of Congress. The ACLU
        called for a moratorium on the use of facial recognition by law enforcement.""",
        source_url="https://example.com/amazon-rekognition",
        source_name="Civil Liberties Watch",
        publish_date="2018-07-26",
        ground_truth={
            "event": {
                "event_type": "bias",
                "event_date": "2018-07-01",
                "event_location": "USA",
                "description": "Facial recognition falsely matched Congress members to mugshots with racial bias",
            },
            "ai_system": {
                "name": "Amazon Rekognition",
                "system_type": "facial_recognition",
                "developer": "Amazon",
                "deployer": None,
            },
            "harm": {
                "harm_type": "reputational",
                "severity": "moderate",
                "affected_parties": ["Congress members", "people of color"],
                "affected_count": 28,
            },
            "organizations": [
                {"name": "Amazon", "role": "developer"},
                {"name": "ACLU", "role": "investigator"},
            ],
        },
    ))

    # Sample 3: ChatGPT data leak
    dataset.add(AIIncident(
        id="sample_003",
        article_text="""OpenAI disclosed a data breach on March 20, 2023, where a bug in
        ChatGPT's Redis client library caused some users to see chat history titles belonging
        to other users. The company also found that during a 9-hour window, payment information
        of about 1.2% of ChatGPT Plus subscribers may have been exposed, including names,
        email addresses, payment addresses, and the last four digits of credit card numbers.
        OpenAI took ChatGPT offline to fix the issue.""",
        source_url="https://example.com/chatgpt-breach",
        source_name="Security Weekly",
        publish_date="2023-03-24",
        ground_truth={
            "event": {
                "event_type": "privacy_breach",
                "event_date": "2023-03-20",
                "event_location": None,
                "description": "Bug exposed users' chat histories and payment information",
            },
            "ai_system": {
                "name": "ChatGPT",
                "system_type": "chatbot",
                "developer": "OpenAI",
                "deployer": "OpenAI",
            },
            "harm": {
                "harm_type": "privacy",
                "severity": "moderate",
                "affected_parties": ["ChatGPT Plus subscribers"],
                "affected_count": None,  # 1.2% mentioned but not absolute number
            },
            "organizations": [
                {"name": "OpenAI", "role": "developer"},
            ],
        },
    ))

    # Sample 4: Content moderation failure
    dataset.add(AIIncident(
        id="sample_004",
        article_text="""YouTube's recommendation algorithm was found to be promoting
        conspiracy theory videos and extremist content to users in a study published by
        researchers at UC Berkeley in January 2020. The study analyzed over 300,000 videos
        and found that the algorithm consistently recommended increasingly extreme content
        to users who watched political videos. YouTube responded by saying they had made
        changes to reduce recommendations of borderline content by 70%.""",
        source_url="https://example.com/youtube-algorithm",
        source_name="Research Digest",
        publish_date="2020-01-15",
        ground_truth={
            "event": {
                "event_type": "unintended_consequence",
                "event_date": "2020-01-01",
                "event_location": None,
                "description": "YouTube algorithm promoted extremist and conspiracy content",
            },
            "ai_system": {
                "name": "YouTube recommendation algorithm",
                "system_type": "recommendation_system",
                "developer": "YouTube",
                "deployer": "YouTube",
            },
            "harm": {
                "harm_type": "societal",
                "severity": "moderate",
                "affected_parties": ["YouTube users"],
                "affected_count": None,
            },
            "organizations": [
                {"name": "YouTube", "role": "developer"},
                {"name": "UC Berkeley", "role": "investigator"},
            ],
        },
    ))

    # Sample 5: Hiring algorithm bias
    dataset.add(AIIncident(
        id="sample_005",
        article_text="""Amazon scrapped a secret AI recruiting tool in 2018 after discovering
        it showed bias against women. The system, developed since 2014, was designed to review
        resumes and rate candidates. However, it penalized resumes containing words like
        "women's" and downgraded graduates of all-women's colleges. Amazon edited the program
        but couldn't guarantee it wouldn't find other discriminatory patterns, leading to its
        abandonment. The tool was never used as the sole basis for hiring decisions.""",
        source_url="https://example.com/amazon-hiring",
        source_name="Reuters",
        publish_date="2018-10-10",
        ground_truth={
            "event": {
                "event_type": "bias",
                "event_date": "2018-10-01",
                "event_location": None,
                "description": "AI recruiting tool showed bias against women in resume screening",
            },
            "ai_system": {
                "name": "Amazon AI recruiting tool",
                "system_type": "predictive_system",
                "developer": "Amazon",
                "deployer": "Amazon",
            },
            "harm": {
                "harm_type": "societal",
                "severity": "moderate",
                "affected_parties": ["women job applicants", "graduates of women's colleges"],
                "affected_count": None,
            },
            "organizations": [
                {"name": "Amazon", "role": "developer"},
            ],
        },
    ))

    return dataset


if __name__ == "__main__":
    # Create and save sample dataset
    sample = create_sample_dataset()
    sample.save("data/annotated/sample_dataset.json")

    # Test loading
    loaded = Dataset.load("data/annotated/sample_dataset.json")
    print(f"\nFirst incident: {loaded[0].id}")
    print(f"Article preview: {loaded[0].article_text[:100]}...")
