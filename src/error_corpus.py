"""
Error corpus extractor.

Walks every (tier, ps, ki, incident_id, field) triple, re-runs the
canonical evaluator, and dumps each row to a CSV with the raw extracted
value and ground-truth value alongside the comparison label. On top of
the corpus it computes three aggregate views:

  1. Hallucinated values — for cells with comparison == "hallucinated",
     count the most frequent fabricated values per field per tier.
  2. Constrained-field confusion — for the closed-vocabulary fields,
     count (ground_truth, predicted) substitution pairs.
  3. Incident hardness — per incident, count how many of the 36 cells
     got any of its 10 scored fields wrong.

Output:
  data/results/error_corpus.csv   (full row-level dump, 36×50×10 ≈ 18 000 rows)
  stdout                          (three aggregate views)

Run from project root or src/:
    python src/error_corpus.py
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the canonical evaluator so comparison labels match metrics.json.
from evaluation import evaluate_extraction, _normalize_to_nested


# The 10 scored fields (organizations excluded from headline aggregation
# in the canonical evaluator).
CONSTRAINED_FIELDS = [
    "event.event_type",
    "ai_system.system_type",
    "harm.harm_type",
    "harm.severity",
]
OPEN_EXACT_FIELDS = [
    "event.event_date",
    "ai_system.name",
    "ai_system.developer",
    "ai_system.deployer",
]
OPEN_BERT_FIELDS = [
    "event.event_location",
    "harm.affected_parties",
]
SCORED_FIELDS = CONSTRAINED_FIELDS + OPEN_EXACT_FIELDS + OPEN_BERT_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_CSV = RESULTS_DIR / "error_corpus.csv"


def tier_of(run: str) -> str | None:
    if "llama" in run.lower(): return "Llama"
    if "haiku" in run.lower(): return "Haiku"
    if "opus" in run.lower():  return "Opus"
    return None


def get_leaf(obj, path: str):
    """Walk a dotted path through a nested dict; return None on miss."""
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def to_text(v) -> str:
    """Render a leaf value as a flat string for CSV; None/[] → ''."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x is not None)
    return str(v)


# Discover the latest *_results.json per (tier, condition). Each run
# directory holds files for one condition; sorted-asc + dict overwrite
# means the latest timestamp wins.
latest_results: dict[tuple[str, str], Path] = {}
for d in sorted(os.listdir(RESULTS_DIR)):
    full = RESULTS_DIR / d
    if not full.is_dir() or d == "_archive":
        continue
    t = tier_of(d)
    if not t:
        continue
    for f in os.listdir(full):
        if f.endswith("_results.json"):
            cond = f.replace("_results.json", "")
            latest_results[(t, cond)] = full / f


# Walk and build the corpus.
rows: list[dict] = []
for (tier, cond), results_path in sorted(latest_results.items()):
    ps, ki = cond.split("_")
    with open(results_path) as fh:
        incidents = json.load(fh)

    for inc in incidents:
        iid = inc["incident_id"]
        parsed = inc.get("parsed_output") or {}
        gt = inc.get("ground_truth") or {}

        # Normalize flat → nested using the same helper the evaluator
        # uses, so leaf lookups line up with comparisons.
        parsed_norm = _normalize_to_nested(parsed)
        field_results = evaluate_extraction(parsed_norm, gt)

        for field in SCORED_FIELDS:
            comparison = field_results.get(field, "missing-from-eval")
            ext_v = get_leaf(parsed_norm, field)
            gt_v = get_leaf(gt, field)
            rows.append({
                "tier": tier,
                "ps": ps,
                "ki": ki,
                "incident_id": iid,
                "field": field,
                "comparison": comparison,
                "extracted": to_text(ext_v),
                "ground_truth": to_text(gt_v),
            })


with open(OUTPUT_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[
        "tier", "ps", "ki", "incident_id", "field",
        "comparison", "extracted", "ground_truth",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
print()


def banner(title: str):
    print("=" * 90)
    print(title)
    print("=" * 90)


# View 1 — most frequent hallucinated values per (tier, field).
banner("View 1 — Most frequent hallucinated values per (tier, field)")
halluc_by_tier_field: dict[tuple[str, str], Counter] = defaultdict(Counter)
for r in rows:
    if r["comparison"] == "hallucinated":
        halluc_by_tier_field[(r["tier"], r["field"])][r["extracted"].lower().strip()] += 1

for (tier, field), counter in sorted(halluc_by_tier_field.items()):
    total = sum(counter.values())
    print(f"\n  {tier} | {field}  (n={total})")
    for value, n in counter.most_common(5):
        print(f"    {n:>3}  {value!r}")


# View 2 — constrained-field confusion (GT → predicted), incorrect rows only.
print()
banner("View 2 — Constrained-field confusion: (GT → predicted) on incorrect rows")
for field in CONSTRAINED_FIELDS:
    confusion: Counter = Counter()
    for r in rows:
        if r["field"] != field or r["comparison"] != "incorrect":
            continue
        gt_norm = r["ground_truth"].lower().strip()
        pr_norm = r["extracted"].lower().strip()
        if gt_norm and pr_norm:
            confusion[(gt_norm, pr_norm)] += 1
    if not confusion:
        continue
    total = sum(confusion.values())
    print(f"\n  {field}  (total incorrect rows: {total})")
    for (gt_v, pr_v), n in confusion.most_common(10):
        print(f"    {n:>3}  {gt_v!r:<40} → {pr_v!r}")


# View 3 — per-incident hardness across all 36 cells.
print()
banner("View 3 — Incident hardness: fraction of cells that mis-extract any field")
# For each (incident, cell), OR-accumulate "any field wrong" across the
# 10 scored fields, then aggregate by incident across the 36 cells.
incident_wrong_cells: dict[str, int] = defaultdict(int)
incident_total_cells: dict[str, int] = defaultdict(int)
cell_any_wrong: dict[tuple, bool] = {}
for r in rows:
    key = (r["incident_id"], r["tier"], r["ps"], r["ki"])
    flag = r["comparison"] in ("incorrect", "hallucinated", "missing")
    cell_any_wrong[key] = cell_any_wrong.get(key, False) or flag

for (iid, _t, _p, _k), any_wrong in cell_any_wrong.items():
    incident_total_cells[iid] += 1
    if any_wrong:
        incident_wrong_cells[iid] += 1

print(f"\n  {'incident_id':<30} {'wrong/total':>14} {'pct':>6}")
print(f"  {'-'*30} {'-'*14} {'-'*6}")
ranked = sorted(
    incident_total_cells.keys(),
    key=lambda i: incident_wrong_cells[i] / incident_total_cells[i],
    reverse=True,
)
for iid in ranked:
    w = incident_wrong_cells[iid]
    t = incident_total_cells[iid]
    pct = w / t * 100
    print(f"  {iid:<30} {w:>6}/{t:<6} {pct:>5.1f}%")
