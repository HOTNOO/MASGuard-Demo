"""Deterministic patch-intent gate for MASGuard recovery attempts.

The gate checks whether a produced source patch is aligned with the bounded
observation that promoted the recovery action. It is intentionally conservative:
it does not decide correctness, it only blocks patches that do not touch the
observed source boundary or do not mention the observed semantic contract.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from swe_mas.utils.path_filters import classify_changed_files, parse_unified_diff_paths


INTENT_CLASSES = {
    "not_enabled",
    "no_fresh_patch",
    "non_semantic_source_diff",
    "semantic_noop_source_diff",
    "no_target_boundary_touch",
    "repeated_failed_patch_shape",
    "semantic_contract_not_addressed",
    "patch_intent_aligned",
}


def evaluate_patch_intent_gate(
    *,
    patch_text: str,
    patch_summary: dict[str, Any] | None = None,
    patcher_failure_evidence: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return a pre-oracle patch-intent gate result.

    The result is read-only and no-credit. A non-blocked patch may still fail the
    oracle; a blocked patch is simply not aligned enough to spend oracle budget.
    """

    if not enabled:
        return {
            "attempted": False,
            "blocked": False,
            "intent_class": "not_enabled",
            "violations": [],
            "revision_required": False,
            "claim_boundary": "not_enabled",
        }

    patch_summary = dict(patch_summary or {})
    evidence = dict(patcher_failure_evidence or {})
    changed_files = _changed_files(patch_text=patch_text, patch_summary=patch_summary)
    source_files = _source_files(changed_files)
    target_boundary_files = _target_boundary_files(evidence)
    contract_tokens = _contract_tokens(evidence)
    target_touched = _touches_target_boundary(source_files, target_boundary_files)
    contract_mentioned = _patch_mentions_contract(patch_text, contract_tokens)
    repeated_failed_lines = _repeated_failed_patch_lines(
        patch_text=patch_text,
        evidence=evidence,
    )
    semantic_noop_lines = _semantic_noop_added_lines(patch_text)
    violations: list[str] = []

    if not source_files:
        intent_class = "no_fresh_patch"
        violations.append("no_fresh_source_diff")
    elif not _has_nontrivial_source_change(patch_text):
        intent_class = "non_semantic_source_diff"
        violations.append("fresh_source_diff_is_whitespace_or_comment_only")
    elif semantic_noop_lines:
        intent_class = "semantic_noop_source_diff"
        violations.append("fresh_source_diff_contains_semantic_noop")
    elif target_boundary_files and not target_touched:
        intent_class = "no_target_boundary_touch"
        violations.append("fresh_source_diff_outside_bounded_observation_boundary")
    elif repeated_failed_lines:
        intent_class = "repeated_failed_patch_shape"
        violations.append("repeats_failed_patch_shape")
    elif contract_tokens and not contract_mentioned:
        intent_class = "semantic_contract_not_addressed"
        violations.append("observed_semantic_contract_not_mentioned_in_patch")
    else:
        intent_class = "patch_intent_aligned"

    return {
        "attempted": True,
        "blocked": bool(violations),
        "intent_class": intent_class,
        "violations": violations,
        "revision_required": bool(violations),
        "changed_files": changed_files,
        "fresh_source_files": source_files,
        "target_boundary_files": target_boundary_files,
        "contract_tokens": contract_tokens,
        "target_boundary_touched": target_touched,
        "semantic_contract_mentioned": contract_mentioned,
        "failed_patch_shape_repeated": bool(repeated_failed_lines),
        "repeated_failed_patch_lines": repeated_failed_lines,
        "semantic_noop_lines": semantic_noop_lines,
        "claim_boundary": "pre_oracle_patch_intent_gate",
    }


def patch_intent_gate_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "blocked": False,
        "intent_class": "not_enabled",
        "violations": [],
        "revision_required": False,
        "skip_reason": reason,
        "claim_boundary": "bounded_validation_disabled_or_gate_not_requested",
    }


def _changed_files(*, patch_text: str, patch_summary: dict[str, Any]) -> list[str]:
    if "fresh_changed_files" in patch_summary:
        files = patch_summary.get("fresh_changed_files", [])
    elif "changed_files" in patch_summary:
        files = patch_summary.get("changed_files", [])
    else:
        files = parse_unified_diff_paths(str(patch_text or ""))
    return _dedupe_paths(files)


def _source_files(changed_files: list[str]) -> list[str]:
    return _dedupe_paths(classify_changed_files(changed_files).get("source_files", []) or [])


def _target_boundary_files(evidence: dict[str, Any]) -> list[str]:
    files: list[str] = []
    files.extend(str(path) for path in list(evidence.get("selected_target_candidates", []) or []))
    files.extend(str(path) for path in list(evidence.get("changed_files", []) or []))
    files.extend(str(path) for path in list(dict(evidence.get("changed_file_classes", {}) or {}).get("source_files", []) or []))
    for key in ("candidate_patch_source_files", "candidate_patch_replay_source_files", "source_files"):
        files.extend(str(path) for path in list(evidence.get(key, []) or []))
    patch_summary = dict(evidence.get("patch_summary", {}) or {})
    files.extend(str(path) for path in list(patch_summary.get("changed_files", []) or []))
    files.extend(str(path) for path in list(dict(patch_summary.get("changed_file_classes", {}) or {}).get("source_files", []) or []))
    return _dedupe_paths(files)


def _touches_target_boundary(changed_files: list[str], target_files: list[str]) -> bool:
    if not changed_files or not target_files:
        return False
    changed = [_normalize_path(path) for path in changed_files]
    targets = [_normalize_path(path) for path in target_files]
    for changed_path in changed:
        for target_path in targets:
            if not changed_path or not target_path:
                continue
            if (
                changed_path == target_path
                or changed_path.endswith("/" + target_path)
                or target_path.endswith("/" + changed_path)
            ):
                return True
    return False


def _contract_tokens(evidence: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for item in list(evidence.get("contract_keywords", []) or []):
        tokens.append(str(item))
    span = dict(evidence.get("failure_span_family", {}) or {})
    api = dict(span.get("api_surface_mismatch", {}) or evidence.get("api_surface_mismatch", {}) or {})
    for key in ("parameter", "callable"):
        value = str(api.get(key, "") or "").strip()
        if value:
            tokens.append(value)
    for key in ("undefined_symbol", "verified_failure_cause", "semantic_retry_reason"):
        value = str(span.get(key, "") or evidence.get(key, "") or "")
        tokens.extend(_identifier_tokens(value))
    for key in (
        "semantic_invariants",
        "expected_behavior_constraints",
        "forbidden_patch_directions",
        "failure_output_excerpt",
    ):
        values = list(evidence.get(key, []) or span.get(key, []) or [])
        for value in values:
            tokens.extend(_identifier_tokens(str(value)))
    tokens.extend(_identifier_tokens(str(span.get("evidence_excerpt", "") or "")))
    return _dedupe(
        token
        for token in tokens
        if len(str(token)) >= 2 and str(token).lower() not in _TOKEN_STOPWORDS
    )


def _identifier_tokens(text: str) -> list[str]:
    raw = str(text or "")
    quoted = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]{1,80})['\"]", raw)
    dotted = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]{2,})+\b", raw)
    snake = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", raw)
    code_literals = re.findall(r"\b[A-Z][A-Za-z0-9_]{1,20}\b", raw)
    call_names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{1,80})\s*\(", raw)
    return quoted + dotted + call_names + code_literals + [token for token in snake if "_" in token]


def _patch_mentions_contract(patch_text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    lowered = str(patch_text or "").lower()
    return any(str(token).lower() in lowered for token in tokens)


def _repeated_failed_patch_lines(*, patch_text: str, evidence: dict[str, Any]) -> list[str]:
    previous = _failed_patch_added_lines(evidence)
    current = _normalized_added_lines(patch_text)
    if not previous or not current:
        return []
    current_set = set(current)
    repeated = [line for line in previous if line in current_set]
    if len(repeated) >= 2:
        return repeated[:8]
    if len(repeated) == 1 and len(repeated[0]) >= 32:
        return repeated
    return []


def _has_nontrivial_source_change(patch_text: str) -> bool:
    changed: list[str] = []
    for raw_line in str(patch_text or "").splitlines():
        if not raw_line or raw_line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if raw_line[0] not in {"+", "-"}:
            continue
        text = raw_line[1:].strip()
        if not text:
            continue
        if text.startswith("#"):
            continue
        changed.append(text)
    return bool(changed)


def _semantic_noop_added_lines(patch_text: str) -> list[str]:
    lines: list[str] = []
    for line in _normalized_added_lines(patch_text):
        if _contains_literal_noop_replace(line):
            lines.append(line)
    return lines[:8]


def _contains_literal_noop_replace(line: str) -> bool:
    for match in re.finditer(
        r"\.replace\(\s*(?P<q1>['\"])(?P<old>(?:\\.|(?!\1).)*?)(?P=q1)\s*,\s*"
        r"(?P<q2>['\"])(?P<new>(?:\\.|(?!\3).)*?)(?P=q2)",
        str(line or ""),
    ):
        if match.group("old") == match.group("new"):
            return True
    return False


def _failed_patch_added_lines(evidence: dict[str, Any]) -> list[str]:
    feedback = dict(evidence.get("failed_patch_feedback", {}) or {})
    lines: list[Any] = []
    for key in (
        "failed_patch_added_lines",
        "failed_added_lines",
        "previous_patch_added_lines",
        "rejected_patch_added_lines",
    ):
        lines.extend(list(feedback.get(key, []) or []))
        lines.extend(list(evidence.get(key, []) or []))
    for key in (
        "failed_patch_diff",
        "previous_failed_patch_diff",
        "rejected_patch_diff",
        "workspace_diff",
    ):
        lines.extend(_extract_added_lines(str(feedback.get(key, "") or "")))
        lines.extend(_extract_added_lines(str(evidence.get(key, "") or "")))
    lines.extend(_extract_added_lines("\n".join(str(item) for item in list(evidence.get("previous_patch_excerpt", []) or []))))
    return _dedupe(_normalize_added_line(line) for line in lines if _normalize_added_line(line))


def _normalized_added_lines(patch_text: str) -> list[str]:
    return _dedupe(_normalize_added_line(line) for line in _extract_added_lines(str(patch_text or "")) if _normalize_added_line(line))


def _extract_added_lines(text: str) -> list[str]:
    added: list[str] = []
    for line in str(text or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added.append(line[1:])
    return added


def _normalize_added_line(line: Any) -> str:
    text = str(line or "").strip()
    if text.startswith("+") and not text.startswith("+++"):
        text = text[1:].strip()
    if not text or text in {"{", "}", ")", "];"}:
        return ""
    if text.startswith("#"):
        return ""
    return re.sub(r"\s+", " ", text)


def _normalize_path(path: str) -> str:
    text = str(path or "").strip().strip("`")
    text = re.sub(r"^[ab]/", "", text)
    return text.replace("\\", "/")


def _dedupe_paths(values: Iterable[Any]) -> list[str]:
    return _dedupe(_normalize_path(str(value)) for value in values)


def _dedupe(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


_TOKEN_STOPWORDS = {
    "attribute",
    "object",
    "source",
    "return",
    "expected",
    "observed",
    "focused",
    "failure",
    "semantic",
    "path",
    "test",
    "tests",
    "pytest",
    "value",
    "values",
}
