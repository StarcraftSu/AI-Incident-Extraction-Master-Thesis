"""
Evaluation metrics for comparing LLM extractions against ground truth.

Evaluation strategy:
- Constrained fields (event_type, system_type, harm_type, severity, org role):
  case-insensitive exact match. Comma-separated ground truth values: match any one.
- Open fields - structured (event_date, ai_system name, developer, deployer):
  case-insensitive exact match with normalization.
- Open fields - free text (event_location, affected_parties):
  BERTScore similarity with threshold.
- description: EXCLUDED from evaluation (summarization, not extraction).
- organizations: array comparison — name uses BERTScore, role uses exact match.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Lazy-load BERTScore to avoid import overhead when not needed
_bert_scorer = None

def _get_bert_scorer():
    """Lazy-load BERTScorer on first use."""
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(model_type="distilbert-base-uncased", lang="en", rescale_with_baseline=True)
    return _bert_scorer


def compute_bert_score(candidate: str, reference: str) -> float:
    """Compute BERTScore F1 between candidate and reference strings."""
    if not candidate or not reference:
        return 0.0
    scorer = _get_bert_scorer()
    P, R, F1 = scorer.score([candidate], [reference])
    return F1.item()


# BERTScore threshold: above this = correct, below = incorrect
BERT_SCORE_THRESHOLD = 0.5


@dataclass
class ExtractionResult:
    """Result of a single extraction attempt."""
    incident_id: str
    model: str
    template: str
    raw_output: str
    parsed_output: Optional[dict]
    ground_truth: dict
    is_valid_json: bool
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class FieldMetrics:
    """Metrics for a single field."""
    field_name: str
    correct: int = 0
    incorrect: int = 0
    missing_in_extraction: int = 0
    hallucinated: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP). TP = correct, FP = incorrect + hallucinated."""
        tp_fp = self.correct + self.incorrect + self.hallucinated
        return self.correct / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN). TP = correct, FN = missing."""
        tp_fn = self.correct + self.missing_in_extraction
        return self.correct / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    model: str
    template: str
    total_samples: int = 0
    valid_json_count: int = 0
    field_metrics: dict = field(default_factory=dict)
    total_latency: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def json_validity_rate(self) -> float:
        return self.valid_json_count / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_samples if self.total_samples > 0 else 0.0

    def _accuracy_aggregation_fields(self) -> dict:
        """Return only field rows that should contribute to per-field
        accuracy/precision/recall/F1 averages and to hallucination_rate.

        Two exclusion rules:

        1. **Pure-hallucination rows** (correct + incorrect + missing
           == 0). The model invented a field that does not exist in
           the schema (e.g. `event.event_date_quote`,
           `ai_system.underlying_model`, top-level `incidents`). They
           are at 0% accuracy by construction.

        2. **Organizations array** (any key starting with
           `organizations`). The variable-length nested-array nature
           of `organizations` introduces methodological noise — small-
           *n* slot rows, role subjectivity (developer vs deployer
           for self-deploying AI), and generic entity mentions
           ("users", "Republican") — that is not present in the 10
           scalar fields. To keep the headline metric on a fixed,
           reproducible 10-cell-per-incident basis, organizations is
           treated as a separate sub-task and excluded from headline
           aggregation. The per-cell organization data is still
           computed and saved in `*_results.json` for any future
           supplementary analysis.
        """
        return {
            k: v for k, v in self.field_metrics.items()
            if (v.correct + v.incorrect + v.missing_in_extraction) > 0
            and not k.startswith("organizations")
        }

    @property
    def overall_accuracy(self) -> float:
        """Macro accuracy: unweighted mean of per-field accuracies.

        Each field counts equally regardless of how many incidents it
        was scored on. Sensitive to small-n cells: e.g., a single
        incident with 5 GT orgs can flip `organizations[4].name` from
        100% to 0% based on a single output difference, which moves
        the macro mean by ~5pp.

        For headline cross-tier comparisons, prefer
        `overall_accuracy_micro` below.
        """
        agg = self._accuracy_aggregation_fields()
        if not agg:
            return 0.0
        return sum(m.accuracy for m in agg.values()) / len(agg)

    @property
    def overall_accuracy_micro(self) -> float:
        """Micro accuracy: total correct cells / total scored cells.

        Sample-weighted across all field rows that contribute to
        accuracy aggregation. The denominator includes correct,
        incorrect, missing, AND hallucinated counts so that
        hallucinated cells (model invented a value where ground truth
        was empty) are penalised in accuracy directly, matching the
        per-field `accuracy` and the macro `overall_accuracy`
        definitions and ensuring all four micro/macro metrics use the
        same scoring universe. Hallucination is additionally reported
        via `overall_hallucination_rate` for direct interpretation.
        """
        agg = self._accuracy_aggregation_fields()
        if not agg:
            return 0.0
        total_c = sum(m.correct for m in agg.values())
        total_n = sum(m.correct + m.incorrect + m.missing_in_extraction + m.hallucinated for m in agg.values())
        return total_c / total_n if total_n else 0.0

    @property
    def overall_precision_micro(self) -> float:
        """Micro precision: TP / (TP + FP) computed from global counts.
        TP = total correct cells. FP = total incorrect + hallucinated.
        Sample-weighted, paired with overall_accuracy_micro."""
        agg = self._accuracy_aggregation_fields()
        total_tp = sum(m.correct for m in agg.values())
        total_fp = sum(m.incorrect + m.hallucinated for m in agg.values())
        denom = total_tp + total_fp
        return total_tp / denom if denom else 0.0

    @property
    def overall_recall_micro(self) -> float:
        """Micro recall: TP / (TP + FN) computed from global counts.
        TP = total correct cells. FN = total missing cells."""
        agg = self._accuracy_aggregation_fields()
        total_tp = sum(m.correct for m in agg.values())
        total_fn = sum(m.missing_in_extraction for m in agg.values())
        denom = total_tp + total_fn
        return total_tp / denom if denom else 0.0

    @property
    def overall_f1_micro(self) -> float:
        """Micro F1: harmonic mean of micro precision and micro recall.
        This is the standard NLP-extraction F1 and is paired with
        overall_accuracy_micro as the recommended headline metrics.
        Less volatile than `overall_f1` (macro), which is the
        unweighted mean of per-field F1 scores and inherits the
        small-n org-slot artifact described in `overall_accuracy`."""
        p = self.overall_precision_micro
        r = self.overall_recall_micro
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def overall_f1(self) -> float:
        agg = self._accuracy_aggregation_fields()
        if not agg:
            return 0.0
        return sum(m.f1_score for m in agg.values()) / len(agg)

    @property
    def overall_precision(self) -> float:
        agg = self._accuracy_aggregation_fields()
        if not agg:
            return 0.0
        return sum(m.precision for m in agg.values()) / len(agg)

    @property
    def overall_recall(self) -> float:
        agg = self._accuracy_aggregation_fields()
        if not agg:
            return 0.0
        return sum(m.recall for m in agg.values()) / len(agg)

    @property
    def overall_hallucination_rate(self) -> float:
        """Hallucination rate over the same 10 scalar fields used for
        accuracy aggregation. Organizations are excluded because they
        are reported as a separate sub-task; this also avoids the
        artifact where ~90% of headline hallucinations on Haiku/Opus
        were org-related, dominating the metric with a noisy field."""
        agg = self._accuracy_aggregation_fields()
        total_extracted = sum(
            m.correct + m.incorrect + m.hallucinated
            for m in agg.values()
        )
        total_hallucinated = sum(m.hallucinated for m in agg.values())
        return total_hallucinated / total_extracted if total_extracted > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "template": self.template,
            "total_samples": self.total_samples,
            "valid_json_count": self.valid_json_count,
            "json_validity_rate": self.json_validity_rate,
            "overall_accuracy": self.overall_accuracy,
            "overall_accuracy_micro": self.overall_accuracy_micro,
            "overall_precision": self.overall_precision,
            "overall_precision_micro": self.overall_precision_micro,
            "overall_recall": self.overall_recall,
            "overall_recall_micro": self.overall_recall_micro,
            "overall_f1": self.overall_f1,
            "overall_f1_micro": self.overall_f1_micro,
            "overall_hallucination_rate": self.overall_hallucination_rate,
            "avg_latency_seconds": self.avg_latency,
            "field_metrics": {
                name: {
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "correct": m.correct,
                    "incorrect": m.incorrect,
                    "missing": m.missing_in_extraction,
                    "hallucinated": m.hallucinated,
                }
                for name, m in self.field_metrics.items()
            },
            "error_count": len(self.errors),
        }


def parse_json_output(raw_output: str) -> tuple:
    """
    Parse JSON from LLM output, handling common issues.
    Returns (parsed_dict, is_valid).
    """
    if not raw_output or not raw_output.strip():
        return None, False

    text = raw_output.strip()

    # Try direct parse first
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"FINAL JSON:\s*([\s\S]*?)$",
        r"\{[\s\S]*\}",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str), True
            except (json.JSONDecodeError, IndexError):
                continue

    return None, False


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

CONSTRAINED_FIELDS = {
    "event_type": ["ai incident", "ai hazard"],
    "system_type": [
        "facial recognition", "recommendation system", "generative ai",
        "autonomous vehicle", "decision support", "chatbot",
        "content moderation", "predictive system", "ai agent", "other",
    ],
    "harm_type": [
        "physical", "psychological", "reputational", "economic",
        "environmental", "rights violation", "other",
    ],
    "severity": ["minor", "moderate", "significant", "severe"],
    "role": ["developer", "deployer", "regulator", "victim", "other"],
}

# Fields evaluated with BERTScore (free text, semantic comparison)
BERTSCORE_FIELDS = {"event_location", "affected_parties"}

# Fields evaluated with exact match (structured open fields).
# Documentation only: enumerates which fields are *intended* to fall
# through to the exact-match bucket in `_get_field_type`. The dispatch
# uses a fallthrough default (anything not in CONSTRAINED/BERTSCORE/
# EXCLUDED gets "exact"), so this set is not consulted at runtime —
# adding or removing entries here has no behavioural effect.
EXACT_MATCH_OPEN_FIELDS = {"event_date", "name", "developer", "deployer"}

# Fields excluded from evaluation
EXCLUDED_FIELDS = {"description"}


def _normalize_str(value: str) -> str:
    """Normalize a string for comparison: lowercase, strip, remove common suffixes."""
    s = value.lower().strip()
    # Remove common corporate suffixes
    for suffix in [", inc.", ", inc", " inc.", " inc", ", corp.", ", corp",
                   " corp.", " corp", ", ltd.", ", ltd", " ltd.", " ltd",
                   " llc", ", llc", " co.", ", co."]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
    return s


def _strip_harm_suffix(s: str) -> str:
    """Strip a trailing " harm" so taxonomy parent labels ("physical harm",
    "economic harm") match the harm_type vocab ("physical", "economic").

    Scoped to the constrained-options builder rather than baked into
    `_normalize_str`, so it does not over-fire on ai_system.* or other
    fields whose values may legitimately end in "harm" (e.g. "victims of
    harm", "Online Harm" as a brand). Also applied per element so
    multi-label strings like "Physical harm, Economic harm" both get
    stripped — the previous end-anchored regex only stripped the last.
    """
    return re.sub(r"\s+harm$", "", s)


def _is_empty(value: Any) -> bool:
    """Check if a value represents empty/missing."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.lower().strip()
        return stripped in ("null", "not stated", "n/a", "none", "unknown", "not_stated", "")
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _get_field_type(field_path: str) -> str:
    """Determine the evaluation type for a field path.

    Returns: 'constrained', 'bertscore', 'exact', or 'excluded'
    """
    last_segment = field_path.rsplit(".", 1)[-1]
    last_segment = re.sub(r"\[\d+\]$", "", last_segment)

    if last_segment in EXCLUDED_FIELDS:
        return "excluded"
    if last_segment in CONSTRAINED_FIELDS:
        return "constrained"
    if last_segment in BERTSCORE_FIELDS:
        return "bertscore"
    return "exact"


def compare_values(extracted: Any, ground_truth: Any, field_path: str = "") -> str:
    """
    Compare extracted value with ground truth.
    Returns: 'correct', 'incorrect', 'missing', 'hallucinated', or 'excluded'
    """
    field_type = _get_field_type(field_path)

    # Skip excluded fields
    if field_type == "excluded":
        return "excluded"

    ext_empty = _is_empty(extracted)
    gt_empty = _is_empty(ground_truth)

    # Both empty = correct (both agree info is not stated)
    if ext_empty and gt_empty:
        return "correct"

    # Ground truth has value but extraction is empty = missing
    if ext_empty and not gt_empty:
        return "missing"

    # Extraction has value but ground truth is empty = hallucinated
    if not ext_empty and gt_empty:
        return "hallucinated"

    # Both have values — compare based on field type
    ext_str = _normalize_str(str(extracted)) if not isinstance(extracted, (dict, list)) else extracted
    gt_str = _normalize_str(str(ground_truth)) if not isinstance(ground_truth, (dict, list)) else ground_truth

    if field_type == "constrained":
        # Constrained field: build sets of allowed values from each side.
        # Both can be: a comma-separated string, or a list of strings.
        def _to_options(v):
            if isinstance(v, list):
                items = (_normalize_str(str(x)) for x in v if x is not None)
            elif isinstance(v, str):
                items = (_normalize_str(opt) for opt in v.split(","))
            else:
                return {str(v)}
            # Strip taxonomy "harm" suffix per element so "Physical harm,
            # Economic harm" → {"physical", "economic"} matches the vocab.
            return {_strip_harm_suffix(item) for item in items if item}

        gt_options = _to_options(ground_truth if isinstance(ground_truth, list) else gt_str)
        ext_options = _to_options(extracted if isinstance(extracted, list) else ext_str)
        # Correct if any extracted value matches any ground truth value
        if gt_options & ext_options:
            return "correct"
        return "incorrect"

    elif field_type == "bertscore":
        # Free text field. Join lists to comma-separated text so BERTScore
        # does not score Python repr noise. Also try substring on each list
        # element individually — a single matching element should count.
        def _to_text(v):
            if isinstance(v, list):
                return ", ".join(str(x) for x in v if x is not None)
            return str(v)

        ext_text = _to_text(extracted)
        gt_text = _to_text(ground_truth)
        ext_norm = _normalize_str(ext_text)
        gt_norm = _normalize_str(gt_text)
        # Gate the substring shortcut on both sides being non-empty —
        # otherwise [None] / [""] flatten to "" and trivially pass
        # `"" in gt_norm` for any non-empty GT, silently inflating accuracy.
        if ext_norm and gt_norm and (ext_norm == gt_norm or ext_norm in gt_norm or gt_norm in ext_norm):
            return "correct"
        # Per-element substring shortcut (only when one side is a list).
        if isinstance(extracted, list) or isinstance(ground_truth, list):
            ext_parts = [_normalize_str(str(x)) for x in extracted] if isinstance(extracted, list) else [ext_norm]
            gt_parts = [_normalize_str(str(x)) for x in ground_truth] if isinstance(ground_truth, list) else [gt_norm]
            for e in ext_parts:
                if not e:
                    continue
                for g in gt_parts:
                    if not g:
                        continue
                    if e == g or e in g or g in e:
                        return "correct"
        score = compute_bert_score(ext_text, gt_text)
        return "correct" if score >= BERT_SCORE_THRESHOLD else "incorrect"

    else:
        # Exact match open field (names, dates).
        if ext_str == gt_str:
            return "correct"
        # Also try substring for organization names.
        if isinstance(ext_str, str) and isinstance(gt_str, str):
            if ext_str in gt_str or gt_str in ext_str:
                return "correct"
        # Per-element shortcut for list-typed extractions (mirrors the
        # BERTScore branch). Without this, `["ChatGPT", "Claude"]` against
        # GT `"ChatGPT"` is scored incorrect even though the correct value
        # is right there in the list. ~30 records hit this path on Llama.
        if isinstance(extracted, list) or isinstance(ground_truth, list):
            ext_parts = [_normalize_str(str(x)) for x in extracted] if isinstance(extracted, list) else (
                [ext_str] if isinstance(ext_str, str) else []
            )
            gt_parts = [_normalize_str(str(x)) for x in ground_truth] if isinstance(ground_truth, list) else (
                [gt_str] if isinstance(gt_str, str) else []
            )
            for e in ext_parts:
                if not e:
                    continue
                for g in gt_parts:
                    if not g:
                        continue
                    if e == g or e in g or g in e:
                        return "correct"
        return "incorrect"


_INNER_KEY_MAP = {
    "event": {
        "type": "event_type",
        "date": "event_date",
        "location": "event_location",
    },
    "ai_system": {
        "type": "system_type",
        "system": "system_type",
        # Some models (notably Haiku PS3) emit the flat-style key inside
        # the nested ai_system block: {ai_system: {ai_system_name: ...}}.
        # Rename to the canonical inner key so it scores against name.
        "ai_system_name": "name",
    },
    "harm": {
        "type": "harm_type",
    },
}

_FLAT_TO_NESTED = {
    "event_type": ("event", "event_type"),
    "event_date": ("event", "event_date"),
    "event_location": ("event", "event_location"),
    "description": ("event", "description"),
    "ai_system_name": ("ai_system", "name"),
    "name": ("ai_system", "name"),
    "system_type": ("ai_system", "system_type"),
    "developer": ("ai_system", "developer"),
    "deployer": ("ai_system", "deployer"),
    "harm_type": ("harm", "harm_type"),
    "severity": ("harm", "severity"),
    "affected_parties": ("harm", "affected_parties"),
}


def _normalize_to_nested(extracted: dict) -> dict:
    """Normalize an extraction output to the nested ground truth structure.

    Three cases handled by a single pass (fixes the previous all-or-nothing
    logic that left flat keys in place when the output was partially nested):

    1. Pure flat output  → flat keys promoted into nested groups.
    2. Pure nested output → inner keys renamed (e.g., 'type' → 'event_type').
    3. Mixed output      → flat keys promoted AND nested blocks merged in;
                           nested values win on conflict.

    Unknown top-level keys are preserved (rather than silently dropped) so
    that downstream evaluation counts them as hallucinations instead of
    discarding evidence of structure-level invention.
    """
    if not isinstance(extracted, dict):
        # Treats None, scalars (json.loads("123")), bare strings, and lists
        # uniformly as unparseable structure → fields score as missing rather
        # than crashing downstream on .keys().
        return {}

    result = {"event": {}, "ai_system": {}, "harm": {}, "organizations": []}

    # Pass 1: lift recognised flat keys into their nested groups.
    for key, value in extracted.items():
        if key in ("event", "ai_system", "harm", "organizations"):
            continue
        if key in _FLAT_TO_NESTED:
            group, nested_key = _FLAT_TO_NESTED[key]
            result[group][nested_key] = value
        else:
            # Unknown top-level key: preserve so it surfaces as a hallucination.
            result[key] = value

    # Pass 2: merge nested blocks on top, overriding any flat-promoted values.
    for group_key in ("event", "ai_system", "harm"):
        block = extracted.get(group_key)
        if isinstance(block, dict):
            mapping = _INNER_KEY_MAP.get(group_key, {})
            for k, v in block.items():
                renamed = mapping.get(k, k)
                result[group_key][renamed] = v

    # Pass 3: organizations. Models occasionally emit a single dict instead
    # of a list (≈15% of records, concentrated in PS3_KI4). Coerce a single
    # {name, role} dict into [dict] so it can be matched; drop other shapes
    # (role-keyed, column-oriented) — they're left as the initialized [],
    # so GT orgs count as missing rather than being silently zeroed.
    orgs = extracted.get("organizations")
    if isinstance(orgs, list):
        result["organizations"] = orgs
    elif isinstance(orgs, dict) and ("name" in orgs or "role" in orgs):
        result["organizations"] = [orgs]

    return result


def evaluate_extraction(
    extracted: dict,
    ground_truth: dict,
    prefix: str = ""
) -> dict:
    """
    Recursively evaluate extraction against ground truth.
    Returns dict mapping field paths to comparison results.
    Excludes 'description' field.

    Before comparison, normalizes flat extraction outputs to the nested
    ground truth structure to avoid false hallucination/missing counts
    caused by structural mismatch.
    """
    # Normalize flat output to nested structure (only at top level)
    if not prefix:
        extracted = _normalize_to_nested(extracted)

    results = {}

    all_keys = set(ground_truth.keys()) | set(extracted.keys() if extracted else [])

    for key in all_keys:
        field_path = f"{prefix}.{key}" if prefix else key

        gt_value = ground_truth.get(key)
        ext_value = extracted.get(key) if extracted else None

        if isinstance(gt_value, dict) and not isinstance(gt_value, list):
            # Recurse into nested dict
            nested_results = evaluate_extraction(
                ext_value if isinstance(ext_value, dict) else {},
                gt_value,
                field_path
            )
            results.update(nested_results)
        elif key == "organizations" or (
            isinstance(gt_value, list)
            and gt_value
            and isinstance(gt_value[0], dict)
        ):
            # Organizations array: compare lists of {name, role} dicts.
            # Enter this branch even when GT is an empty list, so that
            # hallucinated orgs are counted per-element.
            ext_list = ext_value if isinstance(ext_value, list) else []
            gt_list = gt_value if isinstance(gt_value, list) else []

            # For each ground truth org, find best match in extracted
            for i, gt_org in enumerate(gt_list):
                org_path = f"{field_path}[{i}]"

                if not ext_list:
                    # All orgs missing
                    for org_key in gt_org:
                        full_path = f"{org_path}.{org_key}"
                        result = compare_values(None, gt_org[org_key], full_path)
                        if result != "excluded":
                            results[full_path] = result
                    continue

                # Find best matching extracted org (by name similarity)
                best_match = None
                best_score = -1
                for ext_org in ext_list:
                    if isinstance(ext_org, dict):
                        ext_name = str(ext_org.get("name", "")).lower()
                        gt_name = str(gt_org.get("name", "")).lower()
                        if ext_name == gt_name:
                            best_match = ext_org
                            best_score = 1.0
                            break
                        # Try substring match (e.g., "US Treasury" in "US Treasury Secretary Scott Besant")
                        if gt_name and ext_name and (gt_name in ext_name or ext_name in gt_name):
                            best_match = ext_org
                            best_score = 1.0
                            break
                        # Try BERTScore for fuzzy name matching
                        if not _is_empty(ext_org.get("name")) and not _is_empty(gt_org.get("name")):
                            score = compute_bert_score(ext_name, gt_name)
                            if score > best_score:
                                best_score = score
                                best_match = ext_org

                if best_match and best_score >= BERT_SCORE_THRESHOLD:
                    # Compare each field in the org
                    for org_key in gt_org:
                        full_path = f"{org_path}.{org_key}"
                        result = compare_values(
                            best_match.get(org_key),
                            gt_org[org_key],
                            full_path
                        )
                        if result != "excluded":
                            results[full_path] = result
                else:
                    # No match found — all fields missing
                    for org_key in gt_org:
                        full_path = f"{org_path}.{org_key}"
                        result = compare_values(None, gt_org[org_key], full_path)
                        if result != "excluded":
                            results[full_path] = result

            # Check for hallucinated orgs (in extraction but not in ground truth)
            for j, ext_org in enumerate(ext_list):
                if not isinstance(ext_org, dict):
                    continue
                matched = False
                for gt_org in gt_list:
                    ext_name = str(ext_org.get("name", "")).lower()
                    gt_name = str(gt_org.get("name", "")).lower()
                    if ext_name == gt_name:
                        matched = True
                        break
                    if gt_name and ext_name and (gt_name in ext_name or ext_name in gt_name):
                        matched = True
                        break
                    if (not _is_empty(ext_org.get("name")) and
                        not _is_empty(gt_org.get("name")) and
                        compute_bert_score(ext_name, gt_name) >= BERT_SCORE_THRESHOLD
                    ):
                        matched = True
                        break
                if not matched and not _is_empty(ext_org.get("name")):
                    hall_path = f"{field_path}[h{j}]"
                    results[f"{hall_path}.name"] = "hallucinated"
                    results[f"{hall_path}.role"] = "hallucinated"
        else:
            # Leaf value comparison
            result = compare_values(ext_value, gt_value, field_path)
            if result != "excluded":
                results[field_path] = result

    return results


def calculate_metrics(results: list) -> BenchmarkMetrics:
    """Calculate aggregated metrics from a list of extraction results."""
    if not results:
        return BenchmarkMetrics(model="", template="")

    metrics = BenchmarkMetrics(
        model=results[0].model,
        template=results[0].template,
        total_samples=len(results),
    )

    for result in results:
        if result.is_valid_json:
            metrics.valid_json_count += 1

        metrics.total_latency += result.latency_seconds

        if result.error:
            metrics.errors.append(f"{result.incident_id}: {result.error}")

        # Drop the truthy guard on parsed_output so empty {} / [] still
        # contribute "missing" to per-field totals (rather than silently
        # dropping out of denominators and inflating overall accuracy).
        # `_normalize_to_nested` already returns {} for non-dict inputs.
        if result.ground_truth:
            field_results = evaluate_extraction(result.parsed_output, result.ground_truth)

            for field_path, comparison in field_results.items():
                if field_path not in metrics.field_metrics:
                    metrics.field_metrics[field_path] = FieldMetrics(field_name=field_path)

                fm = metrics.field_metrics[field_path]
                fm.total += 1

                if comparison == "correct":
                    fm.correct += 1
                elif comparison == "incorrect":
                    fm.incorrect += 1
                elif comparison == "missing":
                    fm.missing_in_extraction += 1
                elif comparison == "hallucinated":
                    fm.hallucinated += 1

    return metrics


def print_metrics_report(metrics: BenchmarkMetrics):
    """Print a formatted metrics report."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {metrics.model} + {metrics.template}")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Samples:            {metrics.total_samples}")
    print(f"  JSON Validity:      {metrics.json_validity_rate:.1%}")
    print(f"  Overall Accuracy:   {metrics.overall_accuracy:.1%}")
    print(f"  Overall Precision:  {metrics.overall_precision:.1%}")
    print(f"  Overall Recall:     {metrics.overall_recall:.1%}")
    print(f"  Overall F1:         {metrics.overall_f1:.3f}")
    print(f"  Hallucination Rate: {metrics.overall_hallucination_rate:.1%}")
    print(f"  Avg Latency:        {metrics.avg_latency:.2f}s")

    print(f"\nField-Level Metrics:")
    print(f"  {'Field':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Type':>12}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*12}")

    for name, fm in sorted(metrics.field_metrics.items()):
        ftype = _get_field_type(name)
        print(f"  {name:<35} {fm.accuracy:>5.1%} {fm.precision:>5.1%} {fm.recall:>5.1%} {fm.f1_score:>5.3f} {ftype:>12}")

    if metrics.errors:
        print(f"\nErrors ({len(metrics.errors)}):")
        for err in metrics.errors[:5]:
            print(f"  - {err[:80]}...")


# Quick test
if __name__ == "__main__":
    print("Testing evaluation logic...")

    # Test constrained field with comma-separated values
    assert compare_values("physical", "physical, psychological", "harm.harm_type") == "correct"
    assert compare_values("psychological", "physical, psychological", "harm.harm_type") == "correct"
    assert compare_values("economic", "physical, psychological", "harm.harm_type") == "incorrect"
    print("  Comma-separated constrained: OK")

    # Test exact match
    assert compare_values("AI incident", "AI incident", "event.event_type") == "correct"
    assert compare_values("ai incident", "AI incident", "event.event_type") == "correct"
    assert compare_values("AI hazard", "AI incident", "event.event_type") == "incorrect"
    print("  Constrained exact match: OK")

    # Test not stated
    assert compare_values("not stated", "not stated", "ai_system.deployer") == "correct"
    assert compare_values(None, "not stated", "ai_system.deployer") == "correct"
    assert compare_values("not stated", "OpenAI", "ai_system.deployer") == "missing"
    assert compare_values("OpenAI", "not stated", "ai_system.deployer") == "hallucinated"
    print("  Not stated handling: OK")

    # Test excluded field
    assert compare_values("some text", "other text", "event.description") == "excluded"
    print("  Description excluded: OK")

    # Test open field substring match
    assert compare_values("OpenAI", "OpenAI", "ai_system.developer") == "correct"
    assert compare_values("Reno Police", "Reno Police Department", "ai_system.deployer") == "correct"
    print("  Open field substring: OK")

    # ---- Regression tests for the 30-Apr / 01-May evaluation fixes ----

    # Bug 1: mixed flat + nested output should not double-count.
    mixed = {
        "event": {"event_type": "AI incident"},
        "harm_type": "physical",  # flat extra
        "severity": "severe",      # flat extra
    }
    normalized = _normalize_to_nested(mixed)
    assert normalized["event"]["event_type"] == "AI incident"
    assert normalized["harm"]["harm_type"] == "physical"
    assert normalized["harm"]["severity"] == "severe"
    assert "harm_type" not in normalized and "severity" not in normalized
    print("  Mixed flat+nested merge: OK")

    # Bug 1b: pure flat still promoted.
    flat_only = {"event_type": "AI incident", "harm_type": "economic"}
    norm_flat = _normalize_to_nested(flat_only)
    assert norm_flat["event"]["event_type"] == "AI incident"
    assert norm_flat["harm"]["harm_type"] == "economic"
    assert "event_type" not in norm_flat
    print("  Pure flat promotion: OK")

    # Bug 1c: nested wins on conflict with flat duplicate.
    conflict = {"event": {"event_type": "AI hazard"}, "event_type": "AI incident"}
    norm_c = _normalize_to_nested(conflict)
    assert norm_c["event"]["event_type"] == "AI hazard"
    print("  Nested-wins-on-conflict: OK")

    # Bug 3: unknown top-level keys preserved (counted as hallucination downstream).
    unknown = {"event_type": "AI incident", "culprit": "Tesla"}
    norm_u = _normalize_to_nested(unknown)
    assert norm_u.get("culprit") == "Tesla"
    print("  Unknown key preservation: OK")

    # Bug 2: hallucinated orgs counted per-element when GT has empty list.
    gt_empty_orgs = {
        "event": {"event_type": "AI incident"},
        "ai_system": {},
        "harm": {},
        "organizations": [],
    }
    ext_two_orgs = {
        "event": {"event_type": "AI incident"},
        "ai_system": {},
        "harm": {},
        "organizations": [
            {"name": "Tesla", "role": "developer"},
            {"name": "ACLU", "role": "other"},
        ],
    }
    org_results = evaluate_extraction(ext_two_orgs, gt_empty_orgs)
    org_paths = [p for p in org_results if p.startswith("organizations[h")]
    assert len(org_paths) == 4, f"expected 4 org-hall paths, got {org_paths}"
    assert all(org_results[p] == "hallucinated" for p in org_paths)
    print("  Empty-GT-orgs hallucination counting: OK")

    # ---- Regression tests for ultrareview findings (May 2026) ----

    # Ultrareview #3: bare-JSON parsed_output must not crash evaluate_extraction.
    for bare in (123, "AI incident", [1, 2, 3], True):
        out = evaluate_extraction(bare, {"event": {"event_type": "AI incident"}})
        assert out.get("event.event_type") == "missing", f"bare {bare!r} → {out}"
    print("  Bare-JSON parsed_output safe: OK")

    # Ultrareview #2: single-dict organizations coerced to list and matched.
    gt_one_org = {
        "event": {}, "ai_system": {}, "harm": {},
        "organizations": [{"name": "Anthropic", "role": "developer"}],
    }
    ext_dict_org = {
        "event": {}, "ai_system": {}, "harm": {},
        "organizations": {"name": "Anthropic", "role": "developer"},
    }
    r = evaluate_extraction(ext_dict_org, gt_one_org)
    assert r.get("organizations[0].name") == "correct"
    assert r.get("organizations[0].role") == "correct"
    print("  Single-dict organizations coercion: OK")

    # Ultrareview #1: list-typed BERTScore extraction matched via substring.
    r = compare_values(
        ["employees", "store operations"],
        "Andon Market employees",
        "harm.affected_parties",
    )
    assert r == "correct", f"expected correct, got {r}"
    print("  BERTScore list extraction: OK")

    # C3: trailing " harm" stripped so taxonomy parent labels match vocab.
    assert compare_values("Physical harm", "physical", "harm.harm_type") == "correct"
    assert compare_values("Economic harm", "economic", "harm.harm_type") == "correct"
    assert compare_values("Reputational harm", "reputational", "harm.harm_type") == "correct"
    print("  Trailing-'harm' suffix strip: OK")

    # ---- Regression tests for ultrareview round 2 (May 2026) ----

    # Round-2 #1 (bug_011): exact-match branch handles list-typed extractions.
    assert compare_values(["ChatGPT", "Claude"], "ChatGPT", "ai_system.name") == "correct"
    assert compare_values(["OpenAI"], "OpenAI", "ai_system.developer") == "correct"
    assert compare_values(["2024-09-01"], "2024-09-01", "event.event_date") == "correct"
    assert compare_values(["MicroBio Corp"], "MicroBio", "ai_system.deployer") == "correct"
    print("  Exact-match list extraction: OK")

    # Round-2 #2 (bug_003): BERTScore branch must not return correct for
    # all-empty list extractions (the empty-substring trap).
    assert compare_values([None], "Anyone harmed", "harm.affected_parties") in ("incorrect", "missing")
    assert compare_values([""], "users", "harm.affected_parties") in ("incorrect", "missing")
    assert compare_values(["   "], "general public", "harm.affected_parties") in ("incorrect", "missing")
    print("  BERTScore empty-substring guard: OK")

    # Round-2 #3 (merged_bug_002): harm-suffix strip is scoped to the
    # constrained _to_options builder, not baked into _normalize_str.
    assert _normalize_str("Physical harm") == "physical harm", (
        "_normalize_str must not strip 'harm' (was over-broad before)"
    )
    assert _normalize_str("victims of harm") == "victims of harm"
    # Multi-label asymmetry: previously the end-anchored regex only
    # stripped the last token, so "Physical harm, Economic harm" became
    # {"physical harm", "economic"} which did not intersect {"physical"}.
    assert compare_values("Physical harm, Economic harm", "physical", "harm.harm_type") == "correct"
    assert compare_values("Physical harm, Economic harm", "economic", "harm.harm_type") == "correct"
    print("  Harm-suffix strip scoped to constrained: OK")

    # Pure-hallucination rows (no GT signal at all) must NOT inflate
    # per-field accuracy averages, but must STILL count in
    # hallucination_rate. Triggered by:
    #   - organizations[hN].* (extra orgs not in GT)
    #   - schema-extra fields like event.description_quote, harm.scale,
    #     ai_system.underlying_model, top-level "incidents" array
    bm = BenchmarkMetrics(model="m", template="t", total_samples=2)
    bm.field_metrics = {
        "harm.harm_type":            FieldMetrics("harm.harm_type",            correct=2, incorrect=0, missing_in_extraction=0, hallucinated=0, total=2),
        "ai_system.name":            FieldMetrics("ai_system.name",            correct=2, incorrect=0, missing_in_extraction=0, hallucinated=0, total=2),
        "organizations[h0].name":    FieldMetrics("organizations[h0].name",    correct=0, incorrect=0, missing_in_extraction=0, hallucinated=3, total=3),
        "organizations[h0].role":    FieldMetrics("organizations[h0].role",    correct=0, incorrect=0, missing_in_extraction=0, hallucinated=3, total=3),
        "event.event_date_quote":    FieldMetrics("event.event_date_quote",    correct=0, incorrect=0, missing_in_extraction=0, hallucinated=2, total=2),
        "ai_system.underlying_model":FieldMetrics("ai_system.underlying_model",correct=0, incorrect=0, missing_in_extraction=0, hallucinated=1, total=1),
    }
    # Two real fields at 100% — accuracy must be 100%, not pulled down
    # by hallucinated-only rows (org pseudo-fields OR schema extras).
    assert abs(bm.overall_accuracy - 1.0) < 1e-9, (
        f"pure-hallucination rows must not drag accuracy; got {bm.overall_accuracy}"
    )
    # Hallucination rate must still see all 9 hallucinated extractions.
    # total_extracted = 2 + 2 + 3 + 3 + 2 + 1 = 13; halluc = 9; rate = 9/13 ≈ 0.692.
    assert abs(bm.overall_hallucination_rate - 9/13) < 1e-6, (
        f"hallucination_rate must include all rows; got {bm.overall_hallucination_rate}"
    )
    print("  Pure-hallucination rows excluded from accuracy: OK")

    # Haiku PS3 quirk: ai_system_name inside nested ai_system block.
    haiku_ps3_shape = {
        "event": {"event_type": "AI hazard"},
        "ai_system": {"ai_system_name": "Mythos", "system_type": "AI model"},
        "harm": {},
        "organizations": [],
    }
    norm = _normalize_to_nested(haiku_ps3_shape)
    assert norm["ai_system"].get("name") == "Mythos", (
        f"ai_system_name should rename to name; got {norm['ai_system']}"
    )
    assert "ai_system_name" not in norm["ai_system"]
    print("  Inner ai_system_name → name rename: OK")

    # Round-2 #4 (bug_008): empty {} / [] parsed_output should produce
    # missings rather than disappearing from per-field totals.
    from evaluation import calculate_metrics, ExtractionResult
    full_gt = {
        "event": {"event_type": "AI incident", "event_date": "2024-01-01",
                  "event_location": "USA", "description": "x"},
        "ai_system": {"name": "X", "system_type": "chatbot",
                      "developer": "X", "deployer": "X"},
        "harm": {"harm_type": "physical", "severity": "minor",
                 "affected_parties": "users"},
        "organizations": [{"name": "X", "role": "developer"}],
    }
    rs = [
        ExtractionResult(incident_id="a", model="m", template="t",
                         raw_output="", parsed_output={"event": {"event_type": "AI incident"}},
                         ground_truth=full_gt, is_valid_json=True, latency_seconds=0.0),
        ExtractionResult(incident_id="b", model="m", template="t",
                         raw_output="", parsed_output={},
                         ground_truth=full_gt, is_valid_json=True, latency_seconds=0.0),
        ExtractionResult(incident_id="c", model="m", template="t",
                         raw_output="", parsed_output={"event": {"event_type": "AI incident"}},
                         ground_truth=full_gt, is_valid_json=True, latency_seconds=0.0),
    ]
    m = calculate_metrics(rs)
    assert m.field_metrics["event.event_type"].total == 3, (
        f"expected 3 records counted, got {m.field_metrics['event.event_type'].total}"
    )
    assert m.field_metrics["event.event_type"].missing_in_extraction == 1
    print("  Empty parsed_output counted as missing: OK")

    print("\nAll tests passed!")
