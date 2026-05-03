"""
Independent audit script: re-derives the headline accuracy numbers
from raw *_results.json files using only stdlib (no BERTScore, no
constrained-vocab set-intersection, no taxonomy harm-suffix stripping,
no best-match organization pairing), without importing anything from
src/. Use this to verify the canonical metrics independently of
evaluation.py.

Scope matches the canonical headline (10 scalar fields):
  event.{event_type, event_date, event_location}
  ai_system.{name, system_type, developer, deployer}
  harm.{harm_type, severity, affected_parties}

Description and organizations[*] are excluded (same as canonical).

Logic:
  - For each (model, condition), load the raw results
  - For each incident: walk the GT structure, compare each leaf
    against the parsed_output's value (case-insensitive substring)
  - Aggregate to a per-cell micro accuracy

The audit applies a strictly simpler rule set than the canonical, so
the audit's numbers are a LOWER BOUND on canonical accuracy. Cells
where the audit and canonical agree closely indicate the canonical
score is not driven by BERTScore or vocabulary-normalization rules.

Run:
    python src/independent_audit.py
"""

import glob
import json
import os
from pathlib import Path


def _norm(v):
    if v is None:
        return ""
    return str(v).lower().strip()


def _is_empty(v):
    s = _norm(v)
    return s in ("", "not stated", "n/a", "none", "unknown", "not_stated", "null")


def compare_simple(extracted, gt):
    """Bare-bones comparison: empty/empty=correct, both-non-empty
    use case-insensitive substring; no BERTScore."""
    e_empty = _is_empty(extracted)
    g_empty = _is_empty(gt)
    if e_empty and g_empty:
        return "correct"
    if e_empty:
        return "missing"
    if g_empty:
        return "hallucinated"
    e, g = _norm(extracted), _norm(gt)
    if e == g or e in g or g in e:
        return "correct"
    return "incorrect"


def normalize_to_nested(p):
    """Lift flat keys into nested groups, mirroring evaluation.py."""
    if not isinstance(p, dict):
        return {}
    nested = {"event": {}, "ai_system": {}, "harm": {}, "organizations": []}
    flat_map = {
        "event_type": ("event", "event_type"),
        "event_date": ("event", "event_date"),
        "event_location": ("event", "event_location"),
        "description": ("event", "description"),
        "ai_system_name": ("ai_system", "name"),
        "name": ("ai_system", "name"),
        "system_type": ("ai_system", "system_type"),
        "developer": ("ai_system", "developer"),
        "deployer": ("ai_system", "deployer"),
        "harm_type": ("harm", "harm_type"),
        "severity": ("harm", "severity"),
        "affected_parties": ("harm", "affected_parties"),
    }
    for k, v in p.items():
        if k in ("event", "ai_system", "harm"):
            if isinstance(v, dict):
                if k == "ai_system" and "ai_system_name" in v and "name" not in v:
                    v = {**v, "name": v.pop("ai_system_name")}
                nested[k].update(v)
        elif k == "organizations":
            if isinstance(v, list):
                nested[k] = v
            elif isinstance(v, dict) and ("name" in v or "role" in v):
                nested[k] = [v]
        elif k in flat_map:
            grp, ik = flat_map[k]
            nested[grp][ik] = v
    return nested


def score_incident(extracted, gt):
    """Returns dict {field_path: verdict}.

    Scope: the 10 headline scalar fields only.
      - event.{event_type, event_date, event_location}
      - ai_system.{name, system_type, developer, deployer}
      - harm.{harm_type, severity, affected_parties}

    Excludes:
      - description: not in headline aggregation (free text, no
        meaningful exact-match scoring).
      - organizations[*]: treated as a separate sub-task in the
        canonical evaluator (variable-length, role subjectivity,
        generic entity prevalence). See evaluation.py.
    """
    norm = normalize_to_nested(extracted)
    res = {}
    for grp in ("event", "ai_system", "harm"):
        gt_grp = gt.get(grp, {}) or {}
        ex_grp = norm.get(grp, {}) or {}
        for k in gt_grp:
            if k == "description":
                continue
            res[f"{grp}.{k}"] = compare_simple(ex_grp.get(k), gt_grp.get(k))
    return res


def main():
    project_root = Path(__file__).parent.parent
    results_root = project_root / "data" / "results"

    rows = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        for results_file in sorted(run_dir.glob("PS*_KI*_results.json")):
            condition = results_file.stem.replace("_results", "")
            with results_file.open() as f:
                records = json.load(f)
            if not records:
                continue
            model = records[0]["model"]
            total_c = total_n = 0
            for r in records:
                if r.get("parsed_output") is None or not r.get("ground_truth"):
                    continue
                verdicts = score_incident(r["parsed_output"], r["ground_truth"])
                total_c += sum(1 for v in verdicts.values() if v == "correct")
                total_n += sum(1 for v in verdicts.values() if v in ("correct", "incorrect", "missing"))
            rows.append({
                "model": model,
                "condition": condition,
                "incidents": len(records),
                "audit_correct": total_c,
                "audit_total": total_n,
                "audit_acc_micro": total_c / total_n if total_n else 0.0,
            })

    # Compare to canonical metrics
    print(f"{'Model':<30} {'Cond':<10} {'Audit acc':>11} {'Canonical micro':>18} {'Δ':>8}")
    print("-" * 80)
    for r in rows:
        canon_path = next(results_root.glob(f"*/{r['condition']}_metrics.json"), None)
        canon_micro = "?"
        delta = "?"
        # find by model
        for mp in results_root.glob(f"*/{r['condition']}_metrics.json"):
            with mp.open() as f:
                m = json.load(f)
            if m["model"] == r["model"]:
                canon_micro = m.get("overall_accuracy_micro", m["overall_accuracy"])
                delta = (r["audit_acc_micro"] - canon_micro) * 100
                break
        if isinstance(canon_micro, float):
            print(f"{r['model']:<30} {r['condition']:<10} {r['audit_acc_micro']:>10.1%} {canon_micro:>17.1%} {delta:>+7.1f}pp")


if __name__ == "__main__":
    main()
