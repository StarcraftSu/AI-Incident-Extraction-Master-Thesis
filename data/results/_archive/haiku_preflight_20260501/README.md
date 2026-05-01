# Haiku preflight (2026-05-01)

This folder holds the 12 preflight runs that exercised all PS × KI
conditions on **one** OECD AIM incident (`2026-04-10-eb21`) before the
full 50-incident Claude Haiku 4.5 benchmark.

## Why archived (not deleted)

1. **Audit trail.** Confirms that all 12 prompt templates parse on
   Haiku before paying for the full run.
2. **Bug surface.** The preflight data revealed the
   `ai_system_name` → `name` rename issue inside the nested `ai_system`
   block on PS3 (committed as `0f7f606` in the main evaluator). The raw
   outputs in this folder are the smallest reproducible test case.
3. **N=1, not the canonical scores.** Each `*_metrics.json` here is
   computed over **one** incident, so accuracies are binary (0% or
   100%) and not comparable to the full-run numbers. The canonical
   Haiku metrics live in
   `data/results/claude_haiku_4_5_20251001_20260501_22{15,17,18,20,22,24,26,28,32,35,38,42}**`
   (12 dirs, one condition each).

## Provenance

Generated 2026-05-01 by `src/preflight_haiku.py` on incident
`2026-04-10-eb21`. Total cost ≈ \$0.10. The full benchmark followed
immediately (commits `0f7f606` and onward).
