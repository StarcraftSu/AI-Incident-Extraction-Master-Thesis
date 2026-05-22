# smoke_test_5_incidents_20260521

A 5-incident smoke test on Llama 3.1 8B × PS3 × KI4 run on 2026-05-21 to validate the PS3 CoVe fix before launching the full sweep.

The smoke test showed accuracy jumping from the pre-fix 37.8% (full 50) to 58.0% (smoke 5), confirming the fix worked. The full 50-incident sweep was then run; PS3_KI4 landed at 54.4% over the full set.

Use the full-sweep numbers, not these — the n=5 sample here is too small for headline reporting.

## Reproduce

```bash
cd ai_incident_extraction
source .venv/bin/activate
python smoke_test_ps3.py
```
