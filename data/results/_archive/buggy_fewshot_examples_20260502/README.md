# PS2 runs with buggy few-shot examples (archived 2026-05-02)

These 12 directories (4 PS2 conditions × 3 models) were produced by the
2026-05-02 post-date-fix benchmark sweep, but BEFORE the few-shot
example fix in `templates/prompting_strategy.py` (commit `f229a14`).

## What was wrong

The three `FEW_SHOT_EXAMPLE_*` dicts in `prompting_strategy.py` had:

```
"article": "Title: ...\nSummary: ...\nConcepts: ..."   # no Date line
"output": '{ ..., "event_date": "not stated", ... }'   # always not stated
```

After the data_loader was fixed (commit `abcb707`) to prepend `Date:
YYYY-MM-DD\n` to real article text, real articles HAD a date but the
few-shot examples shown to the model in PS2 still didn't — and their
canonical outputs still said `event_date: "not stated"`. This taught
the model the wrong pattern: "date in article ⇒ output is not stated".

Symptom: Llama PS1_KI2 event_date = 8% accuracy, while other Llama
conditions hit 76-98%. Opus PS2 underperformed Opus PS1 by ~19pp.
Strongest on Opus because Opus follows few-shot patterns more rigidly.

## What was fixed

`prompting_strategy.py`: each example now has a plausible `Date: YYYY-
MM-DD` line in the article and the matching `event_date` value in the
output JSON:

| Example | Article date | event_date in output |
|---|---|---|
| Tesla Autopilot | 2024-03-15 | "2024-03-15" |
| Amazon Rekognition | 2018-07-26 | "2018-07-26" |
| AlgoTech trading bot | 2024-09-12 | "2024-09-12" |

The 12 directories in this archive were produced under the buggy
examples and are kept for diff comparison only. The canonical post-fix
PS2 metrics live in the main `data/results/` tree under timestamps
≥ 2026-05-02 13:00 (Llama) / 20:00 (Haiku/Opus).

## Provenance

Generated 2026-05-02 between ~00:09 and 00:32 by `rerun_*.py`. Total
cost ≈ $4 for the affected dirs (Haiku PS2 ~$0.50 + Opus PS2 ~$3 +
Llama PS2 free).
