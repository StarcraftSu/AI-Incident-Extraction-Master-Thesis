"""
Verify derived benchmark statistics against pinned expected values.

For each (tier, prompting strategy, knowledge injection level) cell,
loads the canonical metrics.json and computes PS-averaged accuracy,
KI-averaged accuracy, schema-step deltas, hallucination relative
reductions, and other derived statistics. Each value is checked
against a pinned expected number and reported as PASS or FAIL.

Useful as a regression check after evaluator or normalization changes.

Run from project root or src/:
    python src/verify_benchmark_stats.py
"""

import json
import os
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"


def tier_of(run: str) -> str | None:
    if "llama" in run: return "Llama"
    if "haiku" in run: return "Haiku"
    if "opus" in run:  return "Opus"
    return None


# Load latest *_metrics.json and *_results.json for each (tier, condition)
latest_metrics: dict[tuple[str, str], Path] = {}
latest_results: dict[tuple[str, str], Path] = {}
for run in sorted(os.listdir(RESULTS)):
    if run == "_archive" or not (RESULTS / run).is_dir():
        continue
    t = tier_of(run)
    if not t:
        continue
    for f in os.listdir(RESULTS / run):
        if f.endswith("_metrics.json"):
            cond = f.replace("_metrics.json", "")
            latest_metrics[(t, cond)] = RESULTS / run / f
        elif f.endswith("_results.json"):
            cond = f.replace("_results.json", "")
            latest_results[(t, cond)] = RESULTS / run / f

cells: dict[tuple[str, str, str], dict] = {}
for (tier, cond), p in latest_metrics.items():
    ps, ki = cond.split("_")
    with open(p) as f:
        m = json.load(f)
    cells[(tier, ps, ki)] = {
        "acc":  m["overall_accuracy_micro"] * 100,
        "prec": m["overall_precision_micro"] * 100,
        "rec":  m["overall_recall_micro"] * 100,
        "f1":   m["overall_f1_micro"],
        "halluc": m["overall_hallucination_rate"] * 100,
        "field_metrics": m["field_metrics"],
    }


def cell(t, ps, ki, k):
    return cells[(t, ps, ki)][k]


def ps_avg_raw(t, ps, k):
    return sum(cell(t, ps, ki, k) for ki in ("KI1", "KI2", "KI3", "KI4")) / 4


def ki_avg_raw(t, ki, k):
    return sum(cell(t, ps, ki, k) for ps in ("PS1", "PS2", "PS3")) / 3


def ki_avg_table(t, ki, k, decimals=1):
    """Average raw values across PS, then round.
    Round-after-averaging convention for KI-averaged display values."""
    return round(ki_avg_raw(t, ki, k), decimals)


def display(t, ps, ki, k, decimals=1):
    """Per-cell value rounded to 1 decimal for table display."""
    return round(cell(t, ps, ki, k), decimals)


PASS = 0
FAIL = 0


def check(label, expected, computed, formula, tol=0.05):
    global PASS, FAIL
    diff = abs(computed - expected)
    if diff <= tol:
        status = "PASS"
        PASS += 1
    else:
        status = "FAIL"
        FAIL += 1
    print(f"[{status}] {label}")
    print(f"        expected  = {expected}")
    print(f"        computed  = {computed:.4f}")
    print(f"        diff      = {diff:.4f}")
    print(f"        formula   = {formula}")
    print()


def info(label, computed, formula):
    print(f"[INFO] {label}")
    print(f"        computed  = {computed}")
    print(f"        formula   = {formula}")
    print()


def banner(title):
    print("=" * 90)
    print(title)
    print("=" * 90)


# Best-cell accuracy and F1 per tier (direct table lookup).
banner("Best cells — accuracy and F1 per tier")
check("Llama best PS3×KI2 acc", 71.2, display("Llama","PS3","KI2","acc"),
      "cell(Llama,PS3,KI2,acc)")
check("Haiku best PS2×KI3 acc", 81.0, display("Haiku","PS2","KI3","acc"),
      "cell(Haiku,PS2,KI3,acc)")
check("Opus best PS1×KI3 acc",  85.8, display("Opus","PS1","KI3","acc"),
      "cell(Opus,PS1,KI3,acc)")
check("Opus best PS2×KI4 acc",  85.8, display("Opus","PS2","KI4","acc"),
      "cell(Opus,PS2,KI4,acc) (tied)")
check("Llama best F1", 0.815, round(cell("Llama","PS3","KI2","f1"), 3),
      "cell(Llama,PS3,KI2,f1)")
check("Haiku best F1", 0.879, round(cell("Haiku","PS2","KI3","f1"), 3),
      "cell(Haiku,PS2,KI3,f1)")
check("Opus best F1",  0.911, round(cell("Opus","PS1","KI3","f1"), 3),
      "cell(Opus,PS1,KI3,f1)")


# Accuracy gaps between tier best-cells.
banner("Tier gaps between best cells")
check("Llama-to-Opus gap",
      14.6,
      display("Opus","PS1","KI3","acc") - display("Llama","PS3","KI2","acc"),
      "Opus[PS1×KI3] - Llama[PS3×KI2] = 85.8 - 71.2")
check("Llama-to-Haiku gap",
      9.8,
      display("Haiku","PS2","KI3","acc") - display("Llama","PS3","KI2","acc"),
      "Haiku[PS2×KI3] - Llama[PS3×KI2] = 81.0 - 71.2")
check("Haiku-to-Opus gap",
      4.8,
      display("Opus","PS1","KI3","acc") - display("Haiku","PS2","KI3","acc"),
      "Opus[PS1×KI3] - Haiku[PS2×KI3] = 85.8 - 81.0")


# PS-averaged recall and precision per tier (averaged over KI1..KI4).
banner("Highest average recall / precision per tier")
for label, t, ps, k, expected in [
    ("Llama PS3 avg recall",     "Llama","PS3","rec", 89.8),
    ("Haiku PS2 avg recall",     "Haiku","PS2","rec", 93.6),
    ("Opus PS2 avg recall",      "Opus", "PS2","rec", 96.6),
    ("Llama PS2 avg precision",  "Llama","PS2","prec", 69.7),
    ("Haiku PS2 avg precision",  "Haiku","PS2","prec", 77.7),
    ("Opus PS2 avg precision",   "Opus", "PS2","prec", 82.0),
]:
    check(label, expected, ps_avg_raw(t,ps,k),
          f"mean(cell({t},{ps},KI{{1..4}},{k}))")


# Recall-minus-precision gap at PS1×KI1 vs PS1×KI2.
banner("Recall − precision gap under PS1, KI1 vs KI2")
for label, t, expected_ki1, expected_ki2 in [
    ("Llama PS1×KI1 gap", "Llama", 41.5, 11.3),
    ("Haiku PS1×KI1 gap", "Haiku", 27.3, 11.3),
    ("Opus PS1×KI1 gap",  "Opus",  35.2, 11.1),
]:
    g1_disp = display(t,"PS1","KI1","rec") - display(t,"PS1","KI1","prec")
    g2_disp = display(t,"PS1","KI2","rec") - display(t,"PS1","KI2","prec")
    check(f"{label} KI1", expected_ki1, g1_disp,
          f"display(rec) - display(prec) at PS1×KI1")
    check(f"{label} KI2", expected_ki2, g2_disp,
          f"display(rec) - display(prec) at PS1×KI2")


# KI1→KI2 schema step, averaged across PS, on accuracy.
# Reported in both raw form and table-display form to expose the
# round-after-averaging convention.
banner("KI1 → KI2 PS-averaged schema step (accuracy)")
for label, t, expected in [
    ("Llama +14.8 pp", "Llama", 14.8),
    ("Haiku +22.0 pp", "Haiku", 22.0),
    ("Opus +23.7 pp",  "Opus",  23.7),
]:
    raw = ki_avg_raw(t,"KI2","acc") - ki_avg_raw(t,"KI1","acc")
    disp = ki_avg_table(t,"KI2","acc") - ki_avg_table(t,"KI1","acc")
    check(f"{label} (raw)", expected, raw,
          f"ki_avg_raw({t},KI2,acc) - ki_avg_raw({t},KI1,acc)")
    check(f"{label} (display-derived)", expected, disp,
          f"ki_avg_table({t},KI2,acc) - ki_avg_table({t},KI1,acc)")


# Max − min spread of PS-averaged accuracy across KI2, KI3, KI4
# (the plateau region after the schema step).
banner("KI2 / KI3 / KI4 plateau spread (max − min of PS-averaged accuracy)")
for label, t, expected_range in [
    ("Llama plateau range",  "Llama", 1.5),
    ("Haiku plateau range",  "Haiku", 0.8),
    ("Opus plateau range",   "Opus",  0.7),
]:
    vals_disp = [ki_avg_table(t, ki, "acc") for ki in ("KI2","KI3","KI4")]
    r = max(vals_disp) - min(vals_disp)
    check(label, expected_range, r,
          f"max - min of display KI2,KI3,KI4 averages = {vals_disp}")


# Relative reduction in hallucination rate from KI1 to KI4, PS-averaged.
banner("Hallucination KI1 → KI4 relative reduction (PS-averaged)")
for label, t, expected_rel in [
    ("Llama 22%", "Llama", 22),
    ("Haiku 55%", "Haiku", 55),
    ("Opus 66%",  "Opus",  66),
]:
    h1_disp = ki_avg_table(t,"KI1","halluc")
    h4_disp = ki_avg_table(t,"KI4","halluc")
    rel_disp = (h1_disp - h4_disp) / h1_disp * 100
    h1_raw  = ki_avg_raw(t,"KI1","halluc")
    h4_raw  = ki_avg_raw(t,"KI4","halluc")
    rel_raw = (h1_raw - h4_raw) / h1_raw * 100
    check(f"{label} (display: {h1_disp}→{h4_disp})", expected_rel, rel_disp,
          f"(ki_avg_table(KI1)-ki_avg_table(KI4))/ki_avg_table(KI1)*100", tol=1.0)
    check(f"{label} (raw: {h1_raw:.2f}→{h4_raw:.2f})", expected_rel, rel_raw,
          f"(ki_avg_raw(KI1)-ki_avg_raw(KI4))/ki_avg_raw(KI1)*100", tol=1.0)


# KI1 → KI2 schema step within each PS (per-cell delta on accuracy).
banner("Schema-step accuracy gain within PS1")
for t, expected in [("Llama", 19.6), ("Haiku", 29.6), ("Opus", 30.6)]:
    d = display(t,"PS1","KI2","acc") - display(t,"PS1","KI1","acc")
    check(f"{t} PS1 schema jump", expected, d,
          f"cell({t},PS1,KI2,acc) - cell({t},PS1,KI1,acc)")

banner("Schema-step accuracy gain within PS2")
for t, expected in [("Llama", 4.6), ("Haiku", 13.0), ("Opus", 12.0)]:
    d = display(t,"PS2","KI2","acc") - display(t,"PS2","KI1","acc")
    check(f"{t} PS2 schema jump", expected, d,
          f"cell({t},PS2,KI2,acc) - cell({t},PS2,KI1,acc)")

banner("Schema-step accuracy gain within PS3")
for t, expected in [("Llama", 20.2), ("Haiku", 23.2), ("Opus", 28.6)]:
    d = display(t,"PS3","KI2","acc") - display(t,"PS3","KI1","acc")
    check(f"{t} PS3 schema jump", expected, d,
          f"cell({t},PS3,KI2,acc) - cell({t},PS3,KI1,acc)")


# Few-shot effect at KI3: PS2 accuracy minus PS1 accuracy.
banner("Few-shot effect at KI3 (PS2 − PS1 accuracy)")
for t, expected in [("Llama", 3.8), ("Haiku", 0.2), ("Opus", -1.0)]:
    d = display(t,"PS2","KI3","acc") - display(t,"PS1","KI3","acc")
    check(f"{t} KI3 PS2-PS1", expected, d,
          f"cell({t},PS2,KI3,acc) - cell({t},PS1,KI3,acc)")


# Llama PS3×KI2 vs PS1×KI4: overall accuracy delta and per-field deltas
# on the fields with the largest reversals.
banner("Llama PS3×KI2 vs PS1×KI4 — overall and per-field deltas")
d_overall = display("Llama","PS3","KI2","acc") - display("Llama","PS1","KI4","acc")
check("Llama overall PS3×KI2 vs PS1×KI4", 1.2, d_overall,
      "cell(Llama,PS3,KI2,acc) - cell(Llama,PS1,KI4,acc)")


def field_acc(t, ps, ki, fkey):
    fm = cell(t, ps, ki, "field_metrics")
    if fkey in fm:
        return fm[fkey]["accuracy"] * 100
    return None

for label, fkey, expected_delta in [
    ("Llama harm_type PS3×KI2 - PS1×KI4",  "harm.harm_type", 28),
    ("Llama severity PS3×KI2 - PS1×KI4",   "harm.severity",  10),
    ("Llama deployer PS3×KI2 - PS1×KI4",   "ai_system.deployer", -16),
]:
    a1 = field_acc("Llama","PS3","KI2", fkey)
    a2 = field_acc("Llama","PS1","KI4", fkey)
    d = a1 - a2
    check(label, expected_delta, d,
          f"field({fkey}) PS3×KI2 acc - PS1×KI4 acc = {a1:.0f}% - {a2:.0f}%",
          tol=1.0)


# Per-tier best-cell accuracy on each field, grouped into difficulty bands.
banner("Per-field accuracy at each tier's best cell, grouped by difficulty")
best_cells = {
    "Llama": ("PS3","KI2"),
    "Haiku": ("PS2","KI3"),
    "Opus":  ("PS1","KI3"),
}
field_groups = {
    "Easy (94-100%)":   ["event.event_date", "event.event_location"],
    "Medium (40-92%)":  ["ai_system.name", "ai_system.developer", "ai_system.deployer"],
    "Hard - harm_type": ["harm.harm_type"],
    "Hard - severity":  ["harm.severity"],
}
for group, fields in field_groups.items():
    print(f"\n  {group}")
    for fkey in fields:
        for t in ("Llama","Haiku","Opus"):
            ps, ki = best_cells[t]
            a = field_acc(t, ps, ki, fkey)
            if a is not None:
                print(f"    {t} {ps}×{ki} {fkey} = {a:.0f}%")


# KI1 → KI4 accuracy gradient, averaged across PS.
print()
banner("KI1 → KI4 PS-averaged accuracy gradient")
for t, expected in [("Llama", 16.3), ("Haiku", 21.2), ("Opus", 24.4)]:
    raw = ki_avg_raw(t,"KI4","acc") - ki_avg_raw(t,"KI1","acc")
    disp = ki_avg_table(t,"KI4","acc") - ki_avg_table(t,"KI1","acc")
    check(f"{t} KI1→KI4 (raw)", expected, raw,
          f"ki_avg_raw({t},KI4) - ki_avg_raw({t},KI1)")
    check(f"{t} KI1→KI4 (display-derived)", expected, disp,
          f"ki_avg_table({t},KI4) - ki_avg_table({t},KI1)")


# KI-averaged PS1 → PS2 delta on accuracy.
banner("PS1 → PS2 accuracy delta, averaged across KI")
for t in ("Llama","Haiku","Opus"):
    d = ps_avg_raw(t,"PS2","acc") - ps_avg_raw(t,"PS1","acc")
    info(f"{t} PS1→PS2 delta", round(d,1),
         f"ps_avg_raw({t},PS2,acc) - ps_avg_raw({t},PS1,acc)")


# Few-shot effect at KI1: PS2 − PS1 accuracy.
banner("Few-shot effect at KI1 (PS2 − PS1 accuracy)")
for t in ("Llama","Haiku","Opus"):
    d = display(t,"PS2","KI1","acc") - display(t,"PS1","KI1","acc")
    info(f"{t} KI1 PS2-PS1", round(d,1),
         f"cell({t},PS2,KI1,acc) - cell({t},PS1,KI1,acc)")


# Verification effect on Llama at KI1: PS3 − PS1.
banner("Verification effect on Llama at KI1 (PS3 − PS1 accuracy)")
d = display("Llama","PS3","KI1","acc") - display("Llama","PS1","KI1","acc")
check("Llama PS3-PS1 at KI1", 10.2, d,
      "cell(Llama,PS3,KI1,acc) - cell(Llama,PS1,KI1,acc)")


# event_type vocabulary collapse on Haiku from PS1×KI1 to PS1×KI2:
# unique-value count drops, distribution collapses to the closed
# vocabulary, accuracy jumps.
banner("Haiku event_type vocabulary collapse, PS1×KI1 → PS1×KI2")
haiku_ki1_path = latest_results.get(("Haiku","PS1_KI1"))
haiku_ki2_path = latest_results.get(("Haiku","PS1_KI2"))

if haiku_ki1_path:
    with open(haiku_ki1_path) as f:
        ki1_raw = json.load(f)
    et_values_ki1 = []
    for inc in ki1_raw:
        parsed = inc.get("parsed_output", {})
        et = None
        if isinstance(parsed, dict):
            if "event" in parsed and isinstance(parsed["event"], dict):
                et = parsed["event"].get("event_type")
            else:
                et = parsed.get("event_type")
        et_values_ki1.append(et)
    unique_ki1 = len(set(str(v) for v in et_values_ki1 if v))
    info(f"Haiku PS1×KI1 event_type unique values (expected: 50)",
         unique_ki1, f"len(set(parsed_output.event_type)) over 50 incidents")
    print(f"        first 5 values: {et_values_ki1[:5]}")
    print()

if haiku_ki2_path:
    with open(haiku_ki2_path) as f:
        ki2_raw = json.load(f)
    et_values_ki2 = []
    for inc in ki2_raw:
        parsed = inc.get("parsed_output", {})
        et = None
        if isinstance(parsed, dict):
            if "event" in parsed and isinstance(parsed["event"], dict):
                et = parsed["event"].get("event_type")
            else:
                et = parsed.get("event_type")
        et_values_ki2.append(et)
    counter = Counter(et_values_ki2)
    info(f"Haiku PS1×KI2 event_type distribution (expected: 31 incident + 19 hazard)",
         dict(counter), "Counter(parsed_output.event_type) over 50 incidents")

    et_field_acc = field_acc("Haiku","PS1","KI2","event.event_type")
    info(f"Haiku PS1×KI2 event_type accuracy (expected: 44/50 = 88%)",
         round(et_field_acc, 1), "field_metrics[event.event_type].accuracy * 100")


# Opus best-cell error rate vs the 1-in-7 framing.
banner("Opus best-cell error rate vs 1-in-7 framing")
err_pct = 100 - display("Opus","PS1","KI3","acc")
check("Opus best-cell error rate ≈ 1/7", 100/7, err_pct,
      "100 - cell(Opus,PS1,KI3,acc) = 100 - 85.8", tol=0.5)


# Constrained-vs-open subset accuracy. 10 scored fields grouped by
# evaluator dispatch in evaluation.py:
#   CONSTRAINED : closed-vocabulary set intersection
#   OPEN_EXACT  : exact-match string fields
#   OPEN_BERT   : BERTScore semantic fields
# OPEN = OPEN_EXACT ∪ OPEN_BERT (6 fields).
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
OPEN_FIELDS = OPEN_EXACT_FIELDS + OPEN_BERT_FIELDS


def subset_accuracy(t, ps, ki, fields):
    """Micro-averaged accuracy over a subset of fields:
        sum(TP + TN) / sum(TP + TN + FP_incorrect + FP_halluc + FN)
    Skips fields absent from this cell's field_metrics."""
    fm = cell(t, ps, ki, "field_metrics")
    num = 0
    den = 0
    for f in fields:
        if f not in fm:
            continue
        m = fm[f]
        tp = m.get("correct", 0)
        tn = m.get("true_negative", 0)
        fp_i = m.get("incorrect", 0)
        fp_h = m.get("hallucinated", 0)
        fn = m.get("missing", 0)
        num += tp + tn
        den += tp + tn + fp_i + fp_h + fn
    return (num / den * 100) if den else float("nan")


print()
banner("Constrained (4 fields) vs Open (6 fields) accuracy — all 36 cells")
print(f"  CONSTRAINED_FIELDS = {CONSTRAINED_FIELDS}")
print(f"  OPEN_FIELDS        = {OPEN_FIELDS}")
print(f"    (= OPEN_EXACT {OPEN_EXACT_FIELDS} + OPEN_BERT {OPEN_BERT_FIELDS})")
print()
print(f"  {'Tier':<6} {'Cell':<10} {'Overall':>8} {'Constr':>8} {'Open':>8} "
      f"{'OpenExc':>8} {'OpenBrt':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t in ("Llama", "Haiku", "Opus"):
    for ps in ("PS1", "PS2", "PS3"):
        for ki in ("KI1", "KI2", "KI3", "KI4"):
            ov = display(t, ps, ki, "acc")
            c = subset_accuracy(t, ps, ki, CONSTRAINED_FIELDS)
            o = subset_accuracy(t, ps, ki, OPEN_FIELDS)
            oe = subset_accuracy(t, ps, ki, OPEN_EXACT_FIELDS)
            ob = subset_accuracy(t, ps, ki, OPEN_BERT_FIELDS)
            print(f"  {t:<6} {ps+'×'+ki:<10} {ov:>8.1f} {c:>8.1f} {o:>8.1f} "
                  f"{oe:>8.1f} {ob:>8.1f}")
    print()


# Constrained vs open at each tier's best cell.
print("Best cell per tier — overall vs constrained vs open")
print(f"  {'Tier':<6} {'Best cell':<10} {'Overall':>8} {'Constr':>8} {'Open':>8} "
      f"{'OpenExc':>8} {'OpenBrt':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t, (ps, ki) in best_cells.items():
    ov = display(t, ps, ki, "acc")
    c = subset_accuracy(t, ps, ki, CONSTRAINED_FIELDS)
    o = subset_accuracy(t, ps, ki, OPEN_FIELDS)
    oe = subset_accuracy(t, ps, ki, OPEN_EXACT_FIELDS)
    ob = subset_accuracy(t, ps, ki, OPEN_BERT_FIELDS)
    print(f"  {t:<6} {ps+'×'+ki:<10} {ov:>8.1f} {c:>8.1f} {o:>8.1f} "
          f"{oe:>8.1f} {ob:>8.1f}")
print()


# Constrained vs open, averaged across all 12 cells per tier.
print("Tier-averaged across 12 cells")
print(f"  {'Tier':<6} {'Constr':>8} {'Open':>8} {'OpenExc':>8} {'OpenBrt':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for t in ("Llama", "Haiku", "Opus"):
    c_vals, o_vals, oe_vals, ob_vals = [], [], [], []
    for ps in ("PS1", "PS2", "PS3"):
        for ki in ("KI1", "KI2", "KI3", "KI4"):
            c_vals.append(subset_accuracy(t, ps, ki, CONSTRAINED_FIELDS))
            o_vals.append(subset_accuracy(t, ps, ki, OPEN_FIELDS))
            oe_vals.append(subset_accuracy(t, ps, ki, OPEN_EXACT_FIELDS))
            ob_vals.append(subset_accuracy(t, ps, ki, OPEN_BERT_FIELDS))
    print(f"  {t:<6} {sum(c_vals)/12:>8.1f} {sum(o_vals)/12:>8.1f} "
          f"{sum(oe_vals)/12:>8.1f} {sum(ob_vals)/12:>8.1f}")
print()


banner(f"SUMMARY: {PASS} PASS, {FAIL} FAIL")
