"""Verify MAS-DX-R patcher outputs against evidence-conditioned contracts."""

from __future__ import annotations

from typing import Any

from bcmr_swe.recovery.semantic_invariant_gate import (
    semantic_invariant_patch_violations,
    semantic_invariant_row_violations,
)


def verify_patch_contract(
    row: dict[str, Any],
    *,
    baseline_evidence: dict[str, Any] | None = None,
    retry_budget_exhausted: bool = False,
    contract_mode: str = "fixed_localization_source_only",
) -> dict[str, Any]:
    """Audit one patcher raw row against MAS-DX-R recovery constraints.

    The verifier is intentionally post-hoc and deterministic. It does not decide
    whether the semantic fix is correct; it decides whether the recovery action
    respected the evidence-conditioned contract well enough to justify counting
    or retrying the patcher attempt.
    """

    baseline_evidence = dict(baseline_evidence or {})
    observed_signature = _signature_from_row(row)
    baseline_signature = dict(baseline_evidence.get("post_patch_failure_signature", {}) or {})
    action = _action_observation(row)
    patch_summary = dict(row.get("patch_summary", {}) or {})
    changed_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
    patch_legitimacy = str(
        row.get("patch_legitimacy")
        or patch_summary.get("fresh_target_legitimacy")
        or patch_summary.get("target_legitimacy")
        or ""
    )
    violations: list[str] = []
    warnings: list[str] = []
    fixed_localization_contract = contract_mode == "fixed_localization_source_only"

    if fixed_localization_contract and bool(action.get("locator_attempted", False)):
        violations.append("fixed_localization_locator_rerun")
    if fixed_localization_contract and not bool(action.get("fixed_localization_reused", False)):
        violations.append("fixed_localization_not_reused")
    if not bool(action.get("patch_attempted", False)):
        violations.append("no_patch_attempted")
    if not bool(action.get("verifier_attempted", False)):
        violations.append("verifier_not_attempted")

    required_source_only = _requires_source_only(baseline_evidence)
    test_files = [str(path) for path in list(changed_classes.get("test_files", []) or []) if str(path).strip()]
    if required_source_only and test_files:
        violations.append("source_only_required_but_test_files_changed")
    if required_source_only and patch_legitimacy == "source_mixed":
        violations.append("source_only_required_but_patch_is_source_mixed")

    source_files = [str(path) for path in list(changed_classes.get("source_files", []) or []) if str(path).strip()]
    if not source_files and not bool(row.get("oracle_success", False)):
        violations.append("no_source_patch")

    if _same_failing_tests(baseline_signature, observed_signature):
        warnings.append("same_failing_tests_remain")
    if _same_failure_family(baseline_signature, observed_signature):
        warnings.append("same_failure_family_remains")
    if _rms_worsened(baseline_signature, observed_signature):
        violations.append("visual_rms_worsened")
    semantic_invariant_violations = _semantic_invariant_violations(
        baseline_evidence=baseline_evidence,
        row=row,
    )
    candidate_patch_violations = _candidate_patch_semantic_violations(
        baseline_evidence=baseline_evidence,
        row=row,
    )
    violations.extend(candidate_patch_violations)
    violations.extend(semantic_invariant_violations)
    violations = list(dict.fromkeys(violations))

    protocol_blocked = bool(row.get("evaluation_protocol_error", False)) or bool(row.get("error_type"))
    if protocol_blocked:
        warnings.append("protocol_or_runtime_blocked")

    oracle_success = bool(row.get("oracle_success", False))
    contract_adhered = not violations
    eligible_for_bounded_retry = (
        not oracle_success
        and not protocol_blocked
        and (
            "source_only_required_but_test_files_changed" in violations
            or "source_only_required_but_patch_is_source_mixed" in violations
            or "visual_rms_worsened" in violations
        )
    )
    if not oracle_success and contract_adhered and _same_failing_tests(baseline_signature, observed_signature):
        eligible_for_bounded_retry = True
    if retry_budget_exhausted:
        eligible_for_bounded_retry = False
        if not oracle_success:
            warnings.append("retry_budget_exhausted")

    return {
        "schema": "mas_dx_r_patch_contract_verification_v1",
        "instance_id": str(row.get("instance_id", "") or ""),
        "contract_mode": contract_mode,
        "contract_adhered": contract_adhered,
        "protocol_or_runtime_blocked": protocol_blocked,
        "contract_violation_types": violations,
        "contract_warning_types": warnings,
        "semantic_invariant_violations": semantic_invariant_violations,
        "required_source_only": required_source_only,
        "patch_legitimacy": patch_legitimacy,
        "changed_files": list(patch_summary.get("changed_files", []) or []),
        "changed_file_classes": changed_classes,
        "baseline_failure_signature": baseline_signature,
        "observed_failure_signature": observed_signature,
        "worsened_signature": "visual_rms_worsened" in violations,
        "same_failing_tests_remain": "same_failing_tests_remain" in warnings,
        "eligible_for_bounded_retry": eligible_for_bounded_retry,
        "oracle_success": oracle_success,
        "token_cost": float(row.get("token_cost", 0.0) or 0.0),
        "model_call_count": int(row.get("model_call_count", 0) or 0),
    }


def _action_observation(row: dict[str, Any]) -> dict[str, Any]:
    action = dict(row.get("action_adherence_observed", {}) or {})
    patch_summary = dict(row.get("patch_summary", {}) or {})
    stage_outputs = dict(row.get("stage_outputs", {}) or {})
    if "patch_attempted" not in action:
        action["patch_attempted"] = bool(list(patch_summary.get("changed_files", []) or []))
    if "verifier_attempted" not in action:
        verifier = dict(stage_outputs.get("verifier", {}) or {})
        action["verifier_attempted"] = bool(verifier) or bool(
            list(verifier.get("commands", []) or [])
        ) or _official_validation_attempted(row)
    if "locator_attempted" not in action:
        locator = dict(stage_outputs.get("locator", {}) or {})
        action["locator_attempted"] = bool(locator)
    return action


def _official_validation_attempted(row: dict[str, Any]) -> bool:
    if (
        bool(row.get("evaluation_protocol_error", False))
        or bool(row.get("oracle_protocol_error", False))
        or bool(row.get("fail_to_pass_protocol_error", False))
    ):
        return False
    if row.get("fail_to_pass_returncode") is not None:
        return True
    if row.get("oracle_returncode") is not None:
        return True
    return bool(
        str(row.get("fail_to_pass_output", "") or "").strip()
        or str(row.get("oracle_output", "") or "").strip()
    )


def _requires_source_only(baseline_evidence: dict[str, Any]) -> bool:
    next_step = str(baseline_evidence.get("next_recommended_step", "") or "")
    patch_legitimacy = str(baseline_evidence.get("patch_legitimacy", "") or "")
    signature = dict(baseline_evidence.get("post_patch_failure_signature", {}) or {})
    hard_constraints = {
        str(item)
        for item in list(baseline_evidence.get("hard_constraints", []) or [])
        if str(item).strip()
    }
    return (
        "source_only" in next_step
        or "separate_source_and_test_edits" in next_step
        or patch_legitimacy == "source_mixed"
        or "source_patch_only" in hard_constraints
        or "preserve_source_only_patch_contract" in hard_constraints
        or "do_not_edit_tests_for_recovery_claim" in hard_constraints
        or "forbid_test_edits_unless_evidence_proves_test_bug" in hard_constraints
        or str(signature.get("family", "") or "") in {"assertion_semantic_regression", "visual_semantic_regression"}
    )


def _semantic_invariant_violations(
    *,
    baseline_evidence: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    return semantic_invariant_row_violations(
        row=row,
        evidence=baseline_evidence,
    )


def _candidate_patch_semantic_violations(
    *,
    baseline_evidence: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    patch_text = str(row.get("patch", "") or "")
    if not patch_text.strip():
        return []
    violations = semantic_invariant_patch_violations(
        patch_text=patch_text,
        evidence=baseline_evidence,
    )
    return list(dict.fromkeys(violations))


def _signature_from_row(row: dict[str, Any]) -> dict[str, Any]:
    existing = row.get("post_patch_failure_signature", {})
    if isinstance(existing, dict) and existing:
        return dict(existing)
    signature = row.get("patcher_failure_evidence_signature", {})
    if isinstance(signature, dict) and signature and not _has_output(row):
        return dict(signature)
    output = str(row.get("fail_to_pass_output", "") or row.get("oracle_output", "") or "")
    failed_tests = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            target = stripped.split(" ", 1)[1].split(" ", 1)[0].strip()
            if target and target not in failed_tests:
                failed_tests.append(target)
    exception_excerpt = []
    rms = ""
    family = "unknown"
    for line in output.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if "rms " in lowered and not rms:
            rms = _rms_value(stripped)
        if (
            stripped.startswith("E ")
            or stripped.startswith("E       ")
            or "assertionerror" in lowered
            or "imagecomparisonfailure" in lowered
        ):
            exception_excerpt.append(stripped[:240])
        if "imagecomparisonfailure" in lowered or "images not close" in lowered:
            family = "visual_semantic_regression"
        elif "assertionerror" in lowered and family == "unknown":
            family = "assertion_semantic_regression"
    if not family or family == "unknown":
        if bool(row.get("oracle_success", False)):
            family = "none_recovered"
        elif int(row.get("fail_to_pass_returncode", 0) or 0) != 0:
            family = "target_test_still_failing"
    return {
        "family": family,
        "failed_tests": failed_tests,
        "exception_excerpt": exception_excerpt[:8],
        "headline": exception_excerpt[0] if exception_excerpt else (f"FAILED {failed_tests[0]}" if failed_tests else ""),
        "rms": rms,
    }


def _has_output(row: dict[str, Any]) -> bool:
    return bool(str(row.get("fail_to_pass_output", "") or row.get("oracle_output", "") or "").strip())


def _rms_value(line: str) -> str:
    marker = "RMS "
    if marker not in line:
        return ""
    tail = line.split(marker, 1)[1]
    value = []
    for ch in tail:
        if ch.isdigit() or ch == ".":
            value.append(ch)
        elif value:
            break
    return "".join(value)


def _same_failing_tests(baseline: dict[str, Any], observed: dict[str, Any]) -> bool:
    base = {str(item) for item in list(baseline.get("failed_tests", []) or []) if str(item)}
    obs = {str(item) for item in list(observed.get("failed_tests", []) or []) if str(item)}
    return bool(base and obs and base == obs)


def _same_failure_family(baseline: dict[str, Any], observed: dict[str, Any]) -> bool:
    base = str(baseline.get("family", "") or "")
    obs = str(observed.get("family", "") or "")
    return bool(base and obs and base == obs and base != "none_recovered")


def _rms_worsened(baseline: dict[str, Any], observed: dict[str, Any]) -> bool:
    try:
        base = float(str(baseline.get("rms", "") or "nan"))
        obs = float(str(observed.get("rms", "") or "nan"))
    except ValueError:
        return False
    return bool(obs > base)
