# Pre-country-fix runs (archived 2026-05-02)

These 36 directories were the post-date-fix benchmark sweep that
INCLUDED Date in `article_text` but did NOT include Country. The GT
also still contained 3 "not stated" values in constrained-vocab cells.

## What changed for the next run

`src/data_loader.py` now prepends both `Date: <date>\n` and `Country:
<country>\n` to article_text. The few-shot examples in
`templates/prompting_strategy.py` were updated to also include a
`Country: United States` line in their article texts (and to set
event_location consistently). Three GT cells (`2026-04-06-bee5`
system_type and `2026-04-05-a12d` two org roles) were normalized from
`"not stated"` to `"other"` to match the constrained vocabulary.

## Why kept

Audit trail showing the impact of the country-in-input fix and the GT
cleanup on the per-condition headline numbers. The pre-fix summary
file is `data/results/summary_20260502_022842.md` (still in the main
results dir, since it pre-dates this archive — the post-fix summary
will land alongside it).

## Contents

```
no_country_in_input_20260502/
  llama3.1_8b_*/      ... 12 dirs
  claude_haiku_4_5_*/ ... 12 dirs
  claude_opus_4_6_*/  ... 12 dirs
```

## Provenance

Generated 2026-05-02 between 13:00 and 22:30 by `rerun_llama.py`,
`rerun_haiku.py`, `rerun_opus.py`, and `rerun_ps2_all.py`. Total cost
of the runs in this archive ≈ $35 (Llama free + Haiku $5 + Opus $25
+ PS2 rerun $5).
