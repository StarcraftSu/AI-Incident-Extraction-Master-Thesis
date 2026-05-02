# Historical runner scripts (archived 2026-05-02)

These six scripts were created during the iterative debugging of the
benchmark pipeline. Each was a thin wrapper around
`ExperimentRunner.run_benchmark()` parameterized for a specific rerun
that happened in response to a bug fix or pipeline change. After the
final benchmark sweep on 2026-05-02 produced canonical numbers, these
scripts were moved here to keep `src/` clean while preserving the
audit trail.

**They are still runnable from this location.** The path resolution
was updated to use `Path(__file__).resolve().parents[3]` so each
script still finds the project root, the `.env` file, and the `src/`
modules correctly. To run any of them:

```bash
cd ai_incident_extraction
source .venv/bin/activate
python src/_archive/historical_runners/<script>.py
```

## What each script did, and which commit it produced

| Script | Purpose | Commit |
|---|---|---|
| `rerun_ki34.py` | Re-run only KI3+KI4 conditions on Llama 3.1 8B after the harm/system_type taxonomy alignment (`a66c749`, `633749c`) | `68fdc52` |
| `preflight_haiku.py` | 1-incident smoke test across all 12 conditions on Haiku (~$0.10) before launching the full Haiku run; surfaced the `ai_system_name` nested-key bug | `0f7f606` |
| `rerun_haiku.py` | First full Haiku 4.5 sweep (12 conditions × 50 incidents), launched after preflight passed | `e5c22d8` |
| `rerun_opus.py` | First full Opus 4.6 sweep, launched after the model-id correction (`claude-opus-4-6` not `claude-opus-4-6-20250918`) | `0c2f5b1` |
| `rerun_llama.py` | Full Llama 3.1 8B sweep after the date-in-input fix | `abcb707` |
| `rerun_ps2_all.py` | PS2-only re-run on all 3 models after the few-shot example fix (Date+Country added to example articles) | `4ef3213` |

## Why kept (not deleted)

1. **Reproducibility.** Each commit message references the script
   that produced its result dirs. Keeping the scripts lets a future
   reader (or you, six months from now) match commits to scripts to
   the actual run dirs in `data/results/_archive/`.
2. **Templates.** If you later need a similar selective rerun (e.g.,
   "re-run only PS3 conditions on a new model"), the simplest path
   is to copy one of these files and edit the `model_keys` /
   `conditions` arguments.
3. **Cost record.** Each script's docstring documents the estimated
   API cost of the rerun it represents — useful provenance for the
   thesis cost summary.

## Why NOT in main `src/`

After the final sweep, future reruns are unlikely. The canonical
runner is `src/experiment.py`, which already supports any subset of
models / conditions through its `model_keys` and `conditions`
arguments. Anyone needing a fresh run should edit `experiment.py`
rather than create another rerun_*.py.
