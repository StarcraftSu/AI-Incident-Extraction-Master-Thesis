# Pre-date-fix runs (archived 2026-05-01)

These 33 run directories used a version of `data_loader.py` that built
`article_text` from only **Title + Summary + Concepts**, omitting the
OECD AIM-recorded **Date** column. As a result every model correctly
returned `event_date = "not stated"` (the date is not in the input it
sees) while the GT contained the actual date — scoring 0% on
`event.event_date` across all 12 conditions for all three model tiers.

## What changed

In `src/data_loader.py:143`, the article_text now includes a leading
`Date: YYYY-MM-DD\n` line. The fix is small but invalidates every
saved result, since the model now sees additional evidence. All three
models (Llama 3.1 8B, Haiku 4.5, Opus 4.6) were re-run from scratch
with the corrected pipeline; the canonical numbers live outside this
archive folder.

## Contents

```
no_date_in_input_20260501/
  llama3.1_8b_20260428_*/   ... 12 dirs (full original Llama benchmark, KI3+KI4 reruns)
  claude_haiku_4_5_*/       ... 12 dirs (full Haiku benchmark)
  claude_opus_4_6_*/        ... 8 dirs (PS1 + PS2 only — PS3 stopped before fix)
  haiku_preflight_20260501/ ... already nested (preflight from same era)
```

The Opus directories cover only PS1 × KI{1..4} and PS2 × KI{1..4} (8/12 conditions).
PS3 conditions on Opus were never run on the pre-fix pipeline; they
were stopped after PS2_KI4 because the date issue was identified.

## Why kept

Audit trail showing the impact of the pipeline fix on `event.event_date`
specifically. Comparing pre-fix vs post-fix metrics quantifies how
much the missing date depressed each tier's accuracy on that single
field, which is useful for the Discussion chapter's epistemic-
uncertainty paragraph.

## Provenance

- Llama: 2026-04-28 / 04-29 / 05-01 (multiple sessions; archives include
  the original 2026-04-29 runs and the 2026-05-01 KI3/KI4 rerun under
  vocab-aligned prompts)
- Haiku: 2026-05-01 evening (~$5)
- Opus: 2026-05-01 evening, partial (PS1+PS2 only, ~$12)
