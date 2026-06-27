"""AAAI-facing graph-calibrated recovery preference profiles.

The profile built here is a pre-oracle control signal.  It can be attached to
the recovery blueprint to test an explicit preference branch without changing
the default MASGuard execution path.
"""

from __future__ import annotations

from typing import Any


GRAPH_FAILURE_TO_OPERATOR = {
    "environment_or_infra_failure": "environment_preflight_or_oracle_target_repair",
    "test_environment_blocker": "environment_preflight_or_oracle_target_repair",
    "test_collection_blocker": "environment_preflight_or_oracle_target_repair",
    "handoff_information_loss": "handoff_correction_or_ablation",
    "single_agent_local_failure": "patch_contract_nonregression_repatch",
    "verifier_protocol_drift": "protocol_first_target_validation",
    "verifier_acceptance_gap": "protocol_first_target_validation",
    "verifier_feedback_not_propagated": "protocol_first_target_validation",
    "wrong_trust_in_upstream_artifact": "protocol_first_target_validation",
    "shared_fact_contamination": "shared_fact_quarantine_then_repatch",
    "local_patch_regression": "patch_contract_nonregression_repatch",
    "patch_incomplete": "patch_contract_nonregression_repatch",
}

ACTION_TO_OPERATOR = {
    "environment_preflight_then_verifier": "environment_preflight_or_oracle_target_repair",
    "environment_repair": "environment_preflight_or_oracle_target_repair",
    "propagate_verifier_feedback": "handoff_correction_or_ablation",
    "verifier_only_replay": "protocol_first_target_validation",
    "rerun_verifier": "protocol_first_target_validation",
    "protocol_first_target_validation": "protocol_first_target_validation",
    "patcher_fixed_localization": "patch_contract_nonregression_repatch",
    "evidence_conditioned_repatch": "patch_contract_nonregression_repatch",
    "repatch_with_existing_localization": "patch_contract_nonregression_repatch",
    "shared_fact_quarantine_then_repatch": "shared_fact_quarantine_then_repatch",
    "handoff_correction_then_verifier": "handoff_correction_or_ablation",
    "handoff_correction_or_ablation": "handoff_correction_or_ablation",
}

MUTATING_OPERATORS = {
    "handoff_correction_or_ablation",
    "patch_contract_nonregression_repatch",
    "shared_fact_quarantine_then_repatch",
}

PROTOCOL_OPERATORS = {
    "environment_preflight_or_oracle_target_repair",
    "protocol_first_target_validation",
}


def build_graph_calibrated_preference_profile(
    *,
    instance_id: str = "",
    graph: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    diagnosis_row: dict[str, Any] | None = None,
    min_preference_score: float = 0.55,
) -> dict[str, Any]:
    """Build a conservative graph-calibrated operator preference profile.

    Inputs must be available before recovery execution: propagation graph,
    ranked hypotheses, and optionally a provider-free diagnosis row.  The
    profile is intentionally not used by the default blueprint path; callers
    must pass it explicitly to test the AAAI preference branch.
    """

    graph = dict(graph or {})
    summary = dict(graph.get("summary", {}) or {})
    hypotheses = [dict(item) for item in list(hypotheses or []) if isinstance(item, dict)]
    diagnosis_row = dict(diagnosis_row or {})
    top = dict(hypotheses[0]) if hypotheses else {}
    resolved_instance_id = str(instance_id or graph.get("instance_id", "") or diagnosis_row.get("instance_id", "") or "")

    graph_failure_type = str(top.get("failure_type", "") or summary.get("graph_top_failure_type", "") or "")
    diagnosis_failure_type = str(
        diagnosis_row.get("primary_failure_type", "")
        or diagnosis_row.get("failure_type", "")
        or graph_failure_type
        or ""
    )
    recommended_action = str(
        diagnosis_row.get("recommended_recovery_action", "")
        or diagnosis_row.get("selected_action", "")
        or top.get("candidate_recovery_action", "")
        or ""
    )
    graph_operator = GRAPH_FAILURE_TO_OPERATOR.get(graph_failure_type, "")
    diagnosis_operator = (
        GRAPH_FAILURE_TO_OPERATOR.get(diagnosis_failure_type, "")
        or ACTION_TO_OPERATOR.get(recommended_action, "")
    )
    selected_operator = diagnosis_operator or graph_operator or ACTION_TO_OPERATOR.get(recommended_action, "")
    candidate_actions = [str(item) for item in list(summary.get("candidate_recovery_actions", []) or []) if str(item)]
    score, reasons = _score_profile(
        graph_failure_type=graph_failure_type,
        diagnosis_failure_type=diagnosis_failure_type,
        graph_operator=graph_operator,
        selected_operator=selected_operator,
        recommended_action=recommended_action,
        candidate_actions=candidate_actions,
        summary=summary,
        top_hypothesis=top,
    )
    admissible = bool(selected_operator and score >= float(min_preference_score))
    if selected_operator in MUTATING_OPERATORS and bool(summary.get("has_invalid_test_target_signal", False)):
        admissible = False
        reasons.append("invalid_test_target_blocks_mutating_preference")
    if selected_operator in MUTATING_OPERATORS and not (
        bool(summary.get("has_reusable_localization", False))
        or bool(summary.get("has_patch_state_artifact", False))
        or "patcher_fixed_localization" in candidate_actions
    ):
        admissible = False
        reasons.append("mutating_preference_lacks_localization_or_patch_state_support")

    return {
        "schema": "mas_dx_r_aaai_graph_calibrated_preference_profile_v1",
        "instance_id": resolved_instance_id,
        "mode": "aaai_graph_calibrated_preference_v1",
        "selected_policy_action": selected_operator,
        "planned_operator": selected_operator,
        "run_scope": _run_scope(selected_operator),
        "preference_score": round(score, 4),
        "min_preference_score": float(min_preference_score),
        "admissible": admissible,
        "override_allowed": bool(admissible and score >= max(float(min_preference_score), 0.75)),
        "signals": {
            "graph_failure_type": graph_failure_type,
            "diagnosis_failure_type": diagnosis_failure_type,
            "recommended_recovery_action": recommended_action,
            "graph_operator": graph_operator,
            "diagnosis_operator": diagnosis_operator,
            "candidate_recovery_actions": candidate_actions,
            "top_hypothesis_confidence": str(top.get("confidence", "") or ""),
            "supporting_evidence_span_count": len(list(top.get("supporting_evidence_span_ids", []) or [])),
        },
        "preference_reasons": _dedupe(reasons or ["graph_calibrated_preference_default"]),
        "claim_boundary": {
            "pre_oracle_policy": True,
            "provider_free": True,
            "does_not_execute_recovery": True,
            "does_not_call_models": True,
            "does_not_use_target_outcome": True,
            "does_not_use_expert_label_as_runtime_input": True,
            "does_not_add_recovery_credit": True,
            "explicit_branch_only": True,
        },
    }


def normalize_preference_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Return the executable subset of a preference profile."""

    profile = dict(profile or {})
    if not profile:
        return {}
    return {
        "schema": str(profile.get("schema", "") or ""),
        "mode": str(profile.get("mode", "") or ""),
        "selected_policy_action": str(profile.get("selected_policy_action", "") or profile.get("planned_operator", "") or ""),
        "planned_operator": str(profile.get("planned_operator", "") or profile.get("selected_policy_action", "") or ""),
        "run_scope": str(profile.get("run_scope", "") or ""),
        "preference_score": float(profile.get("preference_score", 0.0) or 0.0),
        "admissible": bool(profile.get("admissible", False)),
        "override_allowed": bool(profile.get("override_allowed", False)),
        "evidence_gate": dict(profile.get("evidence_gate", {}) or {}),
        "preference_reasons": [str(item) for item in list(profile.get("preference_reasons", []) or []) if str(item)],
        "claim_boundary": dict(profile.get("claim_boundary", {}) or {}),
    }


def build_mas_evidence_fail_closed_preference_profile(
    *,
    instance_id: str = "",
    selected_policy_action: str,
    selector_rule: str,
    evidence_summary: dict[str, Any],
    min_all69_overlap: int = 69,
) -> dict[str, Any]:
    """Wrap MAS-evidence selector evidence as an explicit fail-closed branch.

    This profile is intentionally stricter than the graph-calibrated profile:
    it is admissible only when the selector is all69-covered, pre-oracle,
    positive-lift, and zero-hurt in the frozen shadow audit. It never overrides
    an existing retrieval policy and it does not add recovery credit by itself.
    """

    evidence = dict(evidence_summary or {})
    action = str(selected_policy_action or "").strip()
    all69_overlap = int(evidence.get("all69_overlap_count", 0) or 0)
    shadow_lift = int(evidence.get("shadow_lift", evidence.get("all_lift", 0)) or 0)
    shadow_hurt = int(evidence.get("shadow_hurt", evidence.get("all_hurt", 0)) or 0)
    pre_oracle = bool(evidence.get("pre_oracle", True))
    provider_free = bool(evidence.get("provider_free", True))
    fail_closed = bool(evidence.get("fail_closed_to_current_masguard", True))
    blocked_reasons: list[str] = []
    if not action:
        blocked_reasons.append("selected_policy_action_missing")
    if all69_overlap < int(min_all69_overlap):
        blocked_reasons.append("all69_overlap_below_threshold")
    if shadow_lift <= 0:
        blocked_reasons.append("shadow_lift_not_positive")
    if shadow_hurt != 0:
        blocked_reasons.append("shadow_hurt_not_zero")
    if not pre_oracle:
        blocked_reasons.append("selector_not_pre_oracle")
    if not provider_free:
        blocked_reasons.append("selector_not_provider_free")
    if not fail_closed:
        blocked_reasons.append("selector_not_fail_closed")
    admissible = not blocked_reasons
    score = 0.45
    if all69_overlap >= int(min_all69_overlap):
        score += 0.15
    if shadow_lift > 0:
        score += min(0.2, 0.05 * shadow_lift)
    if shadow_hurt == 0:
        score += 0.15
    if pre_oracle and provider_free and fail_closed:
        score += 0.05
    return {
        "schema": "mas_dx_r_mas_evidence_fail_closed_preference_profile_v1",
        "instance_id": str(instance_id or evidence.get("instance_id", "") or ""),
        "mode": "mas_evidence_fail_closed_selector_v1",
        "selected_policy_action": action,
        "planned_operator": action,
        "run_scope": _run_scope(action),
        "preference_score": round(max(0.0, min(1.0, score)), 4),
        "min_preference_score": 0.75,
        "admissible": bool(admissible),
        "override_allowed": False,
        "selector_rule": str(selector_rule or evidence.get("selector_rule", "") or ""),
        "evidence_gate": {
            "all69_overlap_count": all69_overlap,
            "min_all69_overlap": int(min_all69_overlap),
            "shadow_lift": shadow_lift,
            "shadow_hurt": shadow_hurt,
            "pre_oracle": pre_oracle,
            "provider_free": provider_free,
            "fail_closed_to_current_masguard": fail_closed,
        },
        "preference_reasons": (
            [
                "mas_evidence_selector_positive_shadow_lift",
                "mas_evidence_selector_zero_shadow_hurt",
                "mas_evidence_selector_fail_closed_to_current_masguard",
            ]
            if admissible
            else ["mas_evidence_selector_evidence_gate_blocked", *blocked_reasons]
        ),
        "claim_boundary": {
            "pre_oracle_policy": True,
            "provider_free": provider_free,
            "does_not_execute_recovery": True,
            "does_not_call_models": True,
            "does_not_use_target_outcome": True,
            "does_not_add_recovery_credit": True,
            "explicit_branch_only": True,
            "fail_closed_to_current_masguard": fail_closed,
        },
    }


def _score_profile(
    *,
    graph_failure_type: str,
    diagnosis_failure_type: str,
    graph_operator: str,
    selected_operator: str,
    recommended_action: str,
    candidate_actions: list[str],
    summary: dict[str, Any],
    top_hypothesis: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    confidence = str(top_hypothesis.get("confidence", "") or "")
    if confidence == "high":
        score += 0.25
        reasons.append("high_confidence_graph_hypothesis")
    elif confidence == "medium":
        score += 0.15
        reasons.append("medium_confidence_graph_hypothesis")
    if graph_failure_type and diagnosis_failure_type and graph_failure_type == diagnosis_failure_type:
        score += 0.2
        reasons.append("diagnosis_failure_type_matches_graph_hypothesis")
    if graph_operator and selected_operator and graph_operator == selected_operator:
        score += 0.15
        reasons.append("diagnosis_operator_matches_graph_operator")
    if recommended_action and (
        recommended_action in candidate_actions
        or ACTION_TO_OPERATOR.get(recommended_action, "") == selected_operator
    ):
        score += 0.15
        reasons.append("recommended_action_supported_by_graph_candidates")
    if len(list(top_hypothesis.get("supporting_evidence_span_ids", []) or [])) >= 2:
        score += 0.1
        reasons.append("multiple_supporting_evidence_spans")
    if bool(summary.get("has_semantic_edges", False)) or bool(summary.get("semantic_edge_counts", {})):
        score += 0.05
        reasons.append("semantic_graph_edges_present")
    if selected_operator in PROTOCOL_OPERATORS:
        score += 0.05
        reasons.append("non_mutating_validation_preference")
    if selected_operator in MUTATING_OPERATORS and bool(summary.get("has_reusable_localization", False)):
        score += 0.05
        reasons.append("reusable_localization_supports_mutating_preference")
    if selected_operator in MUTATING_OPERATORS and bool(summary.get("has_invalid_test_target_signal", False)):
        score -= 0.35
        reasons.append("invalid_test_target_penalizes_mutating_preference")
    return max(0.0, min(1.0, score)), reasons


def _run_scope(operator: str) -> str:
    if operator == "environment_preflight_or_oracle_target_repair":
        return "environment_preflight_then_verifier"
    if operator == "protocol_first_target_validation":
        return "verifier_only_replay"
    if operator in MUTATING_OPERATORS:
        return "patcher_fixed_localization"
    return "verifier_only_replay"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "")
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
