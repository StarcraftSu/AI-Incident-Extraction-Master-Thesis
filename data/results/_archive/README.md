# Archived run directories

This folder holds **superseded** experiment runs whose `parsed_output`
records were generated under a prompt version that has since been
revised. Their `_metrics.json` numbers were re-scored with the latest
evaluator (so the metrics here reflect "old prompt + new evaluator"),
but the prompt that produced the underlying model outputs is no longer
the canonical version cited in the thesis.

The canonical numbers for every (model, PS, KI) condition live in the
main `data/results/` directory and are listed by directory name in the
last column of `summary_*.md`. `src/generate_summary.py` deliberately
ignores anything under `_archive/`.

## What is here, and why

| Run dir (archived)               | Condition | Superseded by                       | Prompt change(s)                                                                       |
|----------------------------------|-----------|-------------------------------------|----------------------------------------------------------------------------------------|
| `llama3.1_8b_20260428_235454`    | PS1_KI3   | `llama3.1_8b_20260501_195727`       | `harm_type` parents lost " harm" suffix (a66c749); `system_type` taxonomy flattened to vocab-aligned labels with "AI agent" added (633749c) |
| `llama3.1_8b_20260429_000209`    | PS1_KI4   | `llama3.1_8b_20260501_200456`       | Same two prompt changes above (KI4 inherits KI3's taxonomy)                            |
| `llama3.1_8b_20260429_002721`    | PS2_KI3   | `llama3.1_8b_20260501_201337`       | Same two prompt changes                                                                |
| `llama3.1_8b_20260429_003731`    | PS2_KI4   | `llama3.1_8b_20260501_202251`       | Same two prompt changes                                                                |
| `llama3.1_8b_20260429_011503`    | PS3_KI3   | `llama3.1_8b_20260501_203555`       | Same two prompt changes                                                                |
| `llama3.1_8b_20260429_012843`    | PS3_KI4   | `llama3.1_8b_20260501_204941`       | Same two prompt changes                                                                |

KI1 and KI2 prompts were **not** revised, so the original 2026-04-29
runs for those six conditions remain canonical and stay in the main
`data/results/` tree.

## Why keep these at all

1. **Audit trail for the prompt fix.** Allows direct inspection of what
   the model used to produce under "Physical harm" / "Recognition /
   detection" parent labels versus the new vocab-aligned forms.
2. **Reproducibility.** Anyone reading the thesis can verify, from the
   commit log, what state of the prompt produced which `parsed_output`.
3. **Diff target.** The +6pp / +8pp `system_type` accuracy gains
   reported in the rerun commit (68fdc52) are reproducible by comparing
   `_metrics.json` here against the canonical replacement.

## Provenance

Moved here in commit (see `git log -- data/results/_archive`) on
2026-05-01. Originally produced on 2026-04-28 / 2026-04-29 by
`src/experiment.py` under prompts predating commits a66c749 (harm_type
taxonomy alignment) and 633749c (system_type taxonomy alignment).
