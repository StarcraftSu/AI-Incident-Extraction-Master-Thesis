# simplified_2call_ps3_20260523

Pre-fix PS3 results archived 2026-05-23.

## Why archived

These PS3 results were generated with the SIMPLIFIED 2-call CoVe variant
that did NOT match Dhuliawala et al. (2023)'s original 4-step pattern:

- Call 1: extract draft
- Call 2 (merged plan+execute+revise): asked verification questions
  and output final JSON; step 2 did NOT see step 1's values.

The simplification lost the integration step where step 2 (verification)
output and step 1 (baseline) draft are explicitly compared in a separate
revision call.

## Superseded by

The 2026-05-23 PS3 re-sweep using the orthodox 3-call CoVe implementation:

- Call 1: baseline extract (PS1-style + KI)
- Call 2: independent per-field WH-question verification (no draft values)
- Call 3: critical revision (sees draft + verifications, anti-rubber-stamp framing)

Plus per Dhuliawala (2024) §4.4 Finding 5: open WH questions
("When did the event occur?") instead of yes/no probes.
Plus categorical-field fallback rule: "other" is preferred over
"not stated" when an entity is described but does not fit specific
categories.

## Subdirectories

- llama3.1_8b_20260521_2{30738,31933,33305,34836}/   — 4 KI levels, Llama simplified 2-call
- claude_haiku_4_5_..._20260522_08{2710,3124,3554,4046}/ — 4 KI, Haiku simplified
- claude_opus_4_6_20260522_08{4937,5624,...}/       — 4 KI, Opus simplified
- llama3.1_8b_20260523_18{2610,5427,...}/           — Llama smoke tests today
- claude_haiku_4_5_..._20260523_19{3627,4131}/      — Haiku smoke tests today

