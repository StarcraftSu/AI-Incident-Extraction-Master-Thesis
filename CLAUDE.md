# Claude Context for ai_incident_extraction

## What This Is

Experiment code for a master thesis: "Knowledge Injection for LLM-based AI Incident Information Extraction" at DSV, Stockholm University. Benchmarks 12 prompt conditions (3 PS × 4 KI) across 3 LLM capability tiers on 50 OECD AIM incidents. Total: 36 condition cells, 1800 LLM calls, ~$40 in API charges, dataset locked.

Headline numbers under the current evaluator (org-excluded micro accuracy):

| Tier | Best cell | Accuracy | F1 | Halluc. |
|------|-----------|----------|-----|---------|
| Llama 3.1 8B | PS1×KI4 | 73.4 % | 0.824 | 5.0 % |
| Haiku 4.5 | PS2×KI3 | 82.7 % | 0.895 | 2.1 % |
| Opus 4.6 | PS1×KI3 | 86.7 % | 0.924 | 1.0 % |

## Architecture

```
build_condition_prompt(ps, ki, article) → LLM call → parse JSON → evaluate vs GT → BenchmarkMetrics
```

- `templates/knowledge_injection.py` — KI1-KI4 prompt components (cumulative: KI3 = KI2 + taxonomy, KI4 = KI3 + ontology)
- `templates/prompting_strategy.py` — PS1 (zero-shot), PS2 (few-shot, 3 examples), PS3 (two-step verification / CoVe)
- `evaluation.py` — field-level comparison (constrained / bertscore / exact / excluded), flat-to-nested normalization, BenchmarkMetrics aggregation (micro + macro)
- `experiment.py` — `ExperimentRunner`: loops models × conditions × incidents, saves per-condition `*_results.json` + `*_metrics.json`
- `llm_client.py` — Ollama (local, JSON mode) and Anthropic (REST, no SDK) clients
- `data_loader.py` — Excel reader; prepends `Date:` and `Country:` to `article_text` at load time
- `reevaluate.py` — re-score saved outputs without re-calling LLMs (used heavily during bug fixes)
- `generate_summary.py` — cross-condition summary tables (feeds Chapter 4)
- `independent_audit.py` — confirmability check, stdlib-only recompute (lands within ±3 pp on 81 % of cells)
- `tests/` — 76-test pytest suite guarding the evaluator

## Critical Design Decisions

1. **KI levels are cumulative.** KI4 includes KI3 includes KI2. Changing KI2 affects KI3 and KI4.
2. **PS3 verification step does NOT receive the KI component.** Only step 1 (extraction) does. Step 2 (verification) gets a generic field list with quote-grounding instructions. Intentional — follows Dhuliawala et al. (2023).
3. **Flat-to-nested normalization** (`_normalize_to_nested`). Models without schema (especially PS1×KI1) produce flat JSON. The evaluator maps flat keys to nested GT paths before comparison; without this, flat outputs get double-counted as both "missing" (nested path) and "hallucinated" (flat path).
4. **Constrained fields use set intersection** for multi-value matching. Both extracted and GT can be comma-separated; any-pair-overlap counts as correct.
5. **Date and country are prepended to `article_text`** at load time. Without this, models returned `"not stated"` for `event_date` and `event_location` even when the values were in the OECD AIM metadata. Pre-fix runs preserved in `data/results/_archive/`.
6. **`description` field excluded** from evaluation. Free-text summary; exact-match scoring not meaningful, BERTScore saturates near 100 % on Haiku/Opus.
7. **`organizations[*]` array excluded from headline metrics.** Variable-length, role-coding subjectivity, generic entity prevalence — would put ~28 % weight on the noisiest sub-task. Per-cell org data still saved in `*_results.json` for supplementary analysis. Treated as a separate sub-task (Chapter 4 §4.7).
8. **Headline metric = micro accuracy** (sample-weighted). Macro is also computed and saved in `*_metrics.json` for sensitivity but is NOT the primary metric (large per-field outliers — fields hitting 0 % at KI1 — distort macro on Llama).
9. **BERTScore**: `distilbert-base-uncased`, threshold 0.5 on F1, with `rescale_with_baseline=True`. **Substring shortcut applied before threshold** — `"United States"` ⊂ `"California, United States"` short-circuits to `correct` without calling BERTScore.

## Constrained Field Vocabulary

The ONLY valid values for constrained fields (used in both ground truth and evaluation):

- **event_type**: `AI incident`, `AI hazard`
- **system_type**: facial recognition, recommendation system, generative AI, autonomous vehicle, decision support, chatbot, content moderation, predictive system, ai agent, other
- **harm_type**: physical, psychological, reputational, economic, environmental, rights violation, other
- **severity**: minor, moderate, significant, severe
- **organizations[*].role**: developer, deployer, regulator, victim, other

## Environment

- Python 3.12 in `.venv/`
- Pinned deps in `requirements.txt`: `requests`, `python-dotenv`, `openpyxl`, `bert-score`, `torch==2.2.2`, `transformers<5`, `numpy<2`, `pytest`
- The torch / transformers / numpy pins are required for x86_64 macOS compatibility — DO NOT unpin
- Ollama for local Llama 3.1 8B (must be running: `ollama serve`)
- `ANTHROPIC_API_KEY` in `.env` at project root (auto-loaded by `python-dotenv`)
- No vendor SDKs — `llm_client.py` hits both Ollama and Anthropic via `requests` directly

## Running

```bash
source .venv/bin/activate
python run_test.py                              # Quick smoke test (1 LLM call, all 12 prompts built)
PYTHONUNBUFFERED=1 python src/experiment.py     # Full benchmark (toggle model_keys in main())
python src/reevaluate.py                        # Re-score saved outputs after eval changes
python src/generate_summary.py                  # Build summary tables
python src/independent_audit.py                 # Confirmability cross-check
pytest tests/ -q                                # 76-test regression suite
```

`experiment.py::main()` uses a manual model-keys toggle (uncomment one of three lines at ~line 420). Default is Llama-only. `data/results/<model>_<timestamp>/` is created per run.

## Common Pitfalls

- **Don't change KI2 without checking KI3/KI4** — they inherit via string concatenation.
- **Don't add subcategory values to ground truth** — evaluation only matches top-level constrained values.
- **Don't add fields to `EXACT_MATCH_OPEN_FIELDS`** — that constant is documentation-only; the dispatch uses fallthrough. Real registration goes in `CONSTRAINED_FIELDS`, `BERTSCORE_FIELDS`, or `EXCLUDED_FIELDS`.
- **Saved results can be re-evaluated without re-calling LLMs** — use `reevaluate.py`. Raw outputs in `*_results.json` are evaluator-agnostic.
- **Ollama must be running** before experiment starts. Check with `curl http://localhost:11434/api/tags`.
- **`independent_audit.py` and the canonical evaluator must stay in scope-aligned.** If you change which fields the headline covers (e.g. excluding orgs), update `score_incident` in the audit script too — otherwise the §5.3 confirmability claim ("within 3 pp on 80 % of conditions") will silently break.
- **The audit script is intentionally simpler than canonical.** No BERTScore, no constrained-vocab set-intersection, no harm-suffix stripping, no org best-match pairing. Audit numbers are a lower bound on canonical accuracy by design.

## Audit Trail (commits)

The evaluator and data pipeline went through ~12 bug-fix and design-change commits during data preparation. Highlights:

- `0b01c2f` — initial pipeline
- `abcb707`, `789322b` — date and country added to `article_text` (event_date/event_location went from 0 % to 90+ %)
- bug_003 through bug_011 — evaluator fixes (flat-to-nested, set intersection, harm-suffix stripping, empty-{} as missing, hallucination rate scope, etc.)
- `692e508` — added 76-test pytest suite
- `842b1e7` — dropped organizations from headline; treat as separate sub-task
- `c282d63` — fixed `requirements.txt` (5 missing deps added, 4 unused dropped)
- `0489a74` — realigned `independent_audit.py` with the post-org-exclusion canonical scope

Pre-fix runs preserved in `data/results/_archive/` with READMEs explaining why each was superseded. Discussion §5.3 cites this audit trail as part of the study's confirmability.

## Paper Cross-References

- Method §3.4 — evaluation rules and field-type table
- Chapter 4 §4.1 — Table 1 (all 36 cells)
- Chapter 4 §4.4 — PS×KI interaction sub-patterns (the headline finding)
- Chapter 4 §4.7 — methodology notes (10-field scope, micro vs macro, audit trail)
- Discussion §5.3 — limitations / confirmability (cites this codebase explicitly)
