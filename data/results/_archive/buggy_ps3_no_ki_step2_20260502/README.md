# buggy_ps3_no_ki_step2_20260502

Pre-fix PS3 (CoVe) results from the 2026-05-02 benchmark sweep.

## Why archived

`build_ps3_verification_prompt` was stripping the KI component before sending step 2 to the LLM, so step 2 always ran with no schema / taxonomy / ontology constraints. As a result PS3 collapsed across KI levels to roughly the same accuracy regardless of which KI condition was nominally being tested.

## Fix

`build_ps3_verification_prompt` now accepts `ki_component` and includes it in a `<schema>` block in the step 2 prompt. The model still cannot see step 1's extracted values (Dhuliawala et al. 2023 independence is preserved), but the schema constraints now flow through to the verified output.

## Superseded by

The 2026-05-21 / 2026-05-22 PS3 re-sweep using the fixed code:

- `llama3.1_8b_20260521_23*` — Llama PS3 conditions on fixed code
- `claude_haiku_4_5_20251001_20260522_*` — Haiku PS3 conditions
- `claude_opus_4_6_20260522_*` — Opus PS3 conditions

## Pre-fix headline numbers (org-excluded micro accuracy)

| Model | KI1 | KI2 | KI3 | KI4 |
|---|---|---|---|---|
| Llama 3.1 8B | 36.4% | 38.0% | 38.0% | 37.8% |
| Haiku 4.5 | 50.2% | 50.3% | 49.6% | 49.8% |
| Opus 4.6 | 52.7% | 51.0% | 51.4% | 51.4% |

The flat-across-KI signature was the diagnostic giveaway that step 2 was ignoring the injection.
