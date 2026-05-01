"""
Build the cross-condition summary report from the latest *_metrics.json
file for each (model, PS, KI) cell. KI1/KI2 prompts have not changed
since 2026-04-29, so those metrics come from the original run dirs.
KI3/KI4 prompts were updated in a66c749 and 633749c, so those metrics
come from the 2026-05-01 rerun dirs.

Output: data/results/summary_<timestamp>.md (overwrites existing summary).
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def latest_metrics_for_condition(results_root: Path, model_safe: str, condition: str) -> dict | None:
    """Return the parsed metrics dict from the most recent run dir that
    contains `<condition>_metrics.json` for this model. Most-recent wins
    so the latest prompt version's results are used.

    `_archive/` is deliberately excluded — see results/_archive/README.md
    for why those runs are kept but not picked up by the summary.
    """
    candidates = [
        p for p in results_root.glob(f"{model_safe}_*")
        if p.is_dir() and p.parent.name != "_archive"
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        metrics_path = run_dir / f"{condition}_metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                m = json.load(f)
            m["_run_dir"] = run_dir.name
            return m
    return None


def build_summary(results_root: Path) -> str:
    models = [
        ("llama3.1:8b", "llama3.1_8b", "Llama 3.1 8B (local, 4-bit)"),
        # Haiku and Opus runs not yet executed; rows will be omitted if
        # no metrics file is found for them.
        ("claude-haiku-4-5-20251001", "claude_haiku_4_5_20251001", "Claude Haiku 4.5 (API)"),
        ("claude-opus-4-6-20250918", "claude_opus_4_6_20250918", "Claude Opus 4.6 (API)"),
    ]
    conditions = [
        f"{ps}_{ki}"
        for ps in ("PS1", "PS2", "PS3")
        for ki in ("KI1", "KI2", "KI3", "KI4")
    ]

    lines = [
        "# AI Incident Extraction Benchmark Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Design: 3 PS × 4 KI = 12 conditions per model. "
        "Each row uses the most recent run-dir's metrics for that condition; "
        "KI3/KI4 numbers reflect the 2026-05-01 rerun under the vocab-aligned prompts.",
        "",
        "## Overall Comparison",
        "",
        "| Model | PS | KI | JSON Valid | Accuracy | Precision | Recall | F1 | Halluc. | Latency | Source |",
        "|-------|----|----|------------|----------|-----------|--------|-----|---------|---------|--------|",
    ]

    for _, model_safe, label in models:
        any_row = False
        for cond in conditions:
            m = latest_metrics_for_condition(results_root, model_safe, cond)
            if m is None:
                continue
            any_row = True
            ps, ki = cond.split("_", 1)
            lines.append(
                f"| {label} | {ps} | {ki} "
                f"| {m['json_validity_rate']:.0%} "
                f"| {m['overall_accuracy']:.1%} "
                f"| {m['overall_precision']:.1%} "
                f"| {m['overall_recall']:.1%} "
                f"| {m['overall_f1']:.3f} "
                f"| {m['overall_hallucination_rate']:.1%} "
                f"| {m['avg_latency_seconds']:.1f}s "
                f"| `{m['_run_dir']}` |"
            )
        if not any_row:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — | — | _no run yet_ |")

    return "\n".join(lines) + "\n"


def main():
    project_root = Path(__file__).parent.parent
    results_root = project_root / "data" / "results"
    if not results_root.exists():
        print(f"ERROR: {results_root} not found", file=sys.stderr)
        sys.exit(1)

    report = build_summary(results_root)
    out_path = results_root / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
