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

        # Include the OECD AIM-recorded date AND country in the article text
        # so the model has the same evidence the human annotator did. Without
        # the date, every model correctly returned "not stated" for event_date
        # (epistemically right, scored 0% against GT). Same pattern would
        # apply to event_location: GT routinely contains "United States"
        # (or "<state>, United States") because the annotator used the AIM
        # `country` column, but Opus etc. refused to invent a country from
        # an article that didn't mention one. Adding both as metadata
        # restores parity. Date format is YYYY-MM-DD; openpyxl may yield a
        # "YYYY-MM-DD HH:MM:SS" string for datetime cells but that's still
        # parseable. Country is filtered to USA-only at dataset level, but
        # the column is still loaded per row in case future datasets vary.
        date_line = f"Date: {date}\n" if date else ""
        country_line = f"Country: {country}\n" if country else ""
        article_text = f"{date_line}{country_line}Title: {title}\nSummary: {summary}\nConcepts: {concepts}"

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


def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth annotations from a JSON file.

    Args:
        gt_path: Path to the ground truth JSON file.

    Returns:
        Dict mapping incident id to ground truth annotation dict.
    """
    filepath = Path(gt_path)
    with open(filepath, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    gt_by_id = {}
    for ann in annotations:
        inc_id = ann.get("id", "")
        gt_by_id[inc_id] = {k: v for k, v in ann.items() if k != "id"}

    print(f"Loaded {len(gt_by_id)} ground truth annotations from {filepath}")
    return gt_by_id


def load_dataset_with_ground_truth(
    data_path: str,
    gt_path: str,
) -> Dataset:
    """Load incidents from Excel and merge ground truth annotations.

    Args:
        data_path: Path to the Excel file with incident records.
        gt_path: Path to the ground truth JSON file.

    Returns:
        Dataset with ground_truth populated for each incident.
    """
    dataset = load_dataset(data_path)
    gt_by_id = load_ground_truth(gt_path)

    matched = 0
    for incident in dataset:
        if incident.id in gt_by_id:
            incident.ground_truth = gt_by_id[incident.id]
            matched += 1

    print(f"Matched {matched}/{len(dataset)} incidents with ground truth")
    if matched < len(dataset):
        missing = [inc.id for inc in dataset if inc.ground_truth is None]
        print(f"  Missing GT for: {missing[:5]}...")

    return dataset


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/data_loader.py <path/to/incidents.xlsx>")
        sys.exit(1)

    ds = load_dataset(sys.argv[1])
    print(f"\nFirst incident: {ds[0].id}")
    print(f"Article preview: {ds[0].article_text[:100]}...")
