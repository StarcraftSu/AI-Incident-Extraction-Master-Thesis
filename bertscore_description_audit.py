"""Compute mean BERTScore F1 on the `description` field across all
PS x KI x model conditions, using the post-fix PS3 runs and the
canonical PS1/PS2 runs.

Replaces the now-stale 0.03-0.06 / 0.67-0.74 numbers in
Results.md §4.4 supplementary observation.

For each *_results.json file under data/results/ (excluding _archive/),
walks the per-incident parsed_output and ground_truth, extracts
event.description, and computes BERTScore F1 with the same settings
as evaluation.py: distilbert-base-uncased, rescale_with_baseline=True.

Reports mean F1 per (model, PS, KI) cell, grouped by tier.

Note: For PS3 conditions, only the post-fix runs (timestamps from
2026-05-21 onward) are used. Pre-fix PS3 runs live in
data/results/_archive/buggy_ps3_no_ki_step2_20260502/ and are skipped.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from evaluation import compute_bert_score, _normalize_to_nested

RESULTS_DIR = Path(__file__).parent / "data/results"

CONDITION_RE = re.compile(r"PS(\d)_KI(\d)_results\.json$")
MODEL_PATTERNS = {
    "llama": "Llama 3.1 8B",
    "haiku": "Haiku 4.5",
    "opus": "Opus 4.6",
}


def model_label(dir_name: str) -> str:
    """Map result directory to model label."""
    if "llama" in dir_name:
        return "Llama 3.1 8B"
    if "haiku" in dir_name:
        return "Haiku 4.5"
    if "opus" in dir_name:
        return "Opus 4.6"
    raise ValueError(f"Unknown model dir: {dir_name}")


def is_post_fix_ps3(dir_name: str) -> bool:
    """Return True for post-fix PS3 directories (created 2026-05-21+)."""
    # Llama post-fix: llama3.1_8b_20260521_2* (after fix at 22:50)
    # Haiku post-fix: claude_haiku_4_5_20251001_20260522_*
    # Opus post-fix: claude_opus_4_6_20260522_*
    if "20260522" in dir_name:
        return True
    if "llama" in dir_name and "20260521_2" in dir_name:
        return True
    return False


def latest_per_condition(results_dir: Path) -> dict:
    """Return {(model, PS, KI): file_path} using the latest dir per condition.
    For PS3, only post-fix runs are eligible. PS1/PS2 use the latest of any
    timestamp (the May-2 canonical runs).
    """
    by_condition = {}
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        try:
            model = model_label(run_dir.name)
        except ValueError:
            continue
        for results_file in run_dir.glob("PS*_KI*_results.json"):
            match = CONDITION_RE.search(results_file.name)
            if not match:
                continue
            ps = f"PS{match.group(1)}"
            ki = f"KI{match.group(2)}"
            if ps == "PS3" and not is_post_fix_ps3(run_dir.name):
                continue  # skip pre-fix PS3
            key = (model, ps, ki)
            # Sorted order means later timestamps win
            by_condition[key] = results_file
    return by_condition


def compute_mean_description_bertscore(results_file: Path) -> tuple[float, int]:
    """Walk all incidents in a results file, compute BERTScore F1 on
    event.description. Returns (mean_score, n_valid_pairs)."""
    with open(results_file) as f:
        data = json.load(f)
    incidents = data if isinstance(data, list) else data.get("results", [])
    scores = []
    coverage = 0  # incidents where the model produced a non-empty, non-"not stated" description
    for inc in incidents:
        parsed = inc.get("parsed_output") or {}
        gt = inc.get("ground_truth") or {}
        # Use the canonical normalizer so flat outputs (PS1/PS3 with no schema)
        # are mapped onto the same nested structure as ground truth.
        parsed_n = _normalize_to_nested(parsed) if isinstance(parsed, dict) else {}
        ext_desc = (parsed_n.get("event") or {}).get("description", "")
        gt_desc = (gt.get("event") or {}).get("description", "")
        if not isinstance(gt_desc, str) or not gt_desc.strip():
            continue  # GT has no description — skip incident from BERTScore base
        gt_desc = gt_desc.strip()
        if not isinstance(ext_desc, str):
            ext_desc = ""
        ext_desc = ext_desc.strip()
        # Missing or "not stated" extraction scores 0 (apples-to-apples vs other conditions)
        if not ext_desc or ext_desc.lower() == "not stated":
            scores.append(0.0)
            continue
        coverage += 1
        scores.append(compute_bert_score(ext_desc, gt_desc))
    if not scores:
        return 0.0, 0, 0
    return sum(scores) / len(scores), len(scores), coverage


def main():
    print("Locating result files...")
    cells = latest_per_condition(RESULTS_DIR)
    print(f"Found {len(cells)} (model, PS, KI) cells\n")
    print(f"{'Model':<14} {'PS':<4} {'KI':<4} {'Mean F1':<10} {'N':<5} {'Cov %':<7}  Run")
    print("-" * 95)
    by_tier = defaultdict(list)
    for (model, ps, ki), path in sorted(cells.items()):
        mean, n, coverage = compute_mean_description_bertscore(path)
        cov_pct = (coverage / n * 100) if n else 0
        run_label = path.parent.name[-40:]
        print(f"{model:<14} {ps:<4} {ki:<4} {mean:.3f}      {n:<5} {cov_pct:5.1f}%  {run_label}")
        by_tier[(model, ps)].append((ki, mean))
    print()
    print("PS-level summary (mean across KI1-KI4):")
    print(f"{'Model':<14} {'PS':<4} {'Mean BERTScore F1 (avg over KI)':<35}")
    print("-" * 60)
    for (model, ps), entries in sorted(by_tier.items()):
        scores = [e[1] for e in entries]
        avg = sum(scores) / len(scores) if scores else 0.0
        rng = (min(scores), max(scores)) if scores else (0.0, 0.0)
        print(f"{model:<14} {ps:<4} {avg:.3f}  (range {rng[0]:.3f}-{rng[1]:.3f})")


if __name__ == "__main__":
    main()
