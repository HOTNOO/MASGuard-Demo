"""Fusion controller for MAS-DX-R recovery-operator selection.

This module is intentionally provider-free. It fuses three pre-oracle signals:
propagation-graph hypotheses, retrieval-conditioned operator rankings, and
bounded-validation risk gates. It does not execute recovery or inspect oracle
outcomes.
"""

from __future__ import annotations

from typing import Any

from bcmr_swe.mas_recovery.preference_design import normalize_preference_profile


OPERATOR_TO_LEGACY_LABEL = {
    "environment_preflight_or_oracle_target_repair": "environment_repair",
    "protocol_first_target_validation": "rerun_verifier",
    "handoff_correction_or_ablation": "rerun_verifier",
    "patch_contract_nonregression_repatch": "repatch_with_existing_localization",
    "shared_fact_quarantine_then_repatch": "repatch_with_existing_localization",
    "confidence_gated_clean_start_fallback": "clean_start_or_swe_repair",
}

GRAPH_FAILURE_TO_OPERATOR = {
    "test_environment_blocker": "environment_preflight_or_oracle_target_repair",
    "test_collection_blocker": "environment_preflight_or_oracle_target_repair",
    "handoff_information_loss": "handoff_correction_or_ablation",
    "verifier_protocol_drift": "protocol_first_target_validation",
    "verifier_acceptance_gap": "protocol_first_target_validation",
    "shared_fact_contamination": "shared_fact_quarantine_then_repatch",
    "local_patch_regression": "patch_contract_nonregression_repatch",
    "patch_incomplete": "patch_contract_nonregression_repatch",
    "selection_to_patch_execution_gap": "patch_contract_nonregression_repatch",
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


def select_fusion_recovery_decision(
    *,
    instance_id: str,
    retrieval_policy_row: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    preference_profile: dict[str, Any] | None = None,
    min_mutating_confidence_margin: float = 4.0,
    min_neighbor_similarity_for_mutation: float = 0.65,
) -> dict[str, Any]:
    """Select a recovery operator from graph, retrieval, and validation gates.

    The selector is conservative: protocol/environment validation can override
    mutating retrieval suggestions when graph evidence names a protocol boundary.
    Mutating operators need both enough score margin and enough historical
    neighbor support; otherwise the decision falls back to protocol validation.
    """

    row = dict(retrieval_policy_row or {})
    runtime = dict(row.get("runtime_feature_digest", {}) or {})
    if not runtime and graph:
        runtime = _runtime_from_graph(graph, hypotheses or [])
    graph_operator = _graph_operator(runtime, hypotheses or [])
    retrieval_operator = str(row.get("selected_policy_action", "") or row.get("planned_operator", "") or "")
    planned_operator = str(row.get("planned_operator", "") or "")
    preference = normalize_preference_profile(preference_profile)
    preference_operator = str(preference.get("selected_policy_action", "") or "")
    preference_gate = _preference_evidence_gate(preference)
    preference_admissible = bool(
        preference.get("admissible", False)
        and preference_operator
        and preference_gate["admissible"]
    )
    ranked = [dict(item) for item in list(row.get("ranked_operators", []) or []) if isinstance(item, dict)]
    neighbor = dict(row.get("neighbor_evidence_summary", {}) or {})
    risk = dict(row.get("risk_calibration", {}) or {})
    candidates = [str(item) for item in list(runtime.get("candidate_recovery_actions", []) or []) if str(item)]
    protocol_reasons = _protocol_reasons(runtime=runtime, graph_operator=graph_operator)
    margin = _score_margin(ranked)
    total_neighbor_similarity = float(neighbor.get("total_neighbor_similarity", 0.0) or 0.0)

    selected = (
        retrieval_operator
        or (preference_operator if preference_admissible else "")
        or graph_operator
        or "protocol_first_target_validation"
    )
    decision_mode = "retrieval_conditioned_operator"
    reasons: list[str] = []
    validation_required = False
    mutating_allowed = selected in MUTATING_OPERATORS

    if preference_admissible and not retrieval_operator:
        selected = preference_operator
        decision_mode = _preference_decision_mode(preference, default="aaai_graph_calibrated_preference")
        reasons.append(_preference_selected_reason(preference))
        reasons.extend(str(item) for item in list(preference.get("preference_reasons", []) or []))
        validation_required = selected in PROTOCOL_OPERATORS or selected in MUTATING_OPERATORS
        mutating_allowed = selected in MUTATING_OPERATORS
    elif (
        preference_admissible
        and retrieval_operator
        and retrieval_operator != preference_operator
        and bool(preference.get("override_allowed", False))
        and not protocol_reasons
    ):
        selected = preference_operator
        decision_mode = "aaai_graph_calibrated_preference_override"
        reasons.append("explicit_aaai_preference_profile_overrode_retrieval_operator")
        reasons.extend(str(item) for item in list(preference.get("preference_reasons", []) or []))
        validation_required = selected in PROTOCOL_OPERATORS or selected in MUTATING_OPERATORS
        mutating_allowed = selected in MUTATING_OPERATORS
    elif retrieval_operator == "confidence_gated_clean_start_fallback":
        if planned_operator in PROTOCOL_OPERATORS:
            selected = planned_operator
            decision_mode = "protocol_first_history_fallback_override"
            reasons.append("retrieval_policy_requested_fallback_but_planned_operator_is_protocol_validation")
            validation_required = True
        elif graph_operator in PROTOCOL_OPERATORS:
            selected = graph_operator
            decision_mode = "protocol_first_history_fallback_override"
            reasons.append("retrieval_policy_requested_fallback_but_graph_supports_non_mutating_validation")
            validation_required = True
        else:
            selected = "confidence_gated_clean_start_fallback"
            decision_mode = "history_risk_fallback"
            reasons.append("retrieval_policy_requested_confidence_gated_fallback")
        mutating_allowed = False
    elif protocol_reasons and retrieval_operator in MUTATING_OPERATORS:
        selected = graph_operator if graph_operator in PROTOCOL_OPERATORS else "protocol_first_target_validation"
        decision_mode = "protocol_first_graph_override"
        reasons.extend(protocol_reasons)
        reasons.append("mutating_retrieval_operator_deferred_until_protocol_validation")
        validation_required = True
        mutating_allowed = False
    elif retrieval_operator in MUTATING_OPERATORS:
        weak_margin = margin < float(min_mutating_confidence_margin)
        weak_neighbor = total_neighbor_similarity < float(min_neighbor_similarity_for_mutation)
        high_risk = str(risk.get("risk_level", "") or "") == "high"
        if weak_margin or weak_neighbor or high_risk:
            selected = graph_operator if graph_operator in MUTATING_OPERATORS else retrieval_operator
            decision_mode = "bounded_validation_before_mutation"
            if weak_margin:
                reasons.append("mutating_operator_score_margin_below_threshold")
            if weak_neighbor:
                reasons.append("mutating_operator_neighbor_similarity_below_threshold")
            if high_risk:
                reasons.append("mutating_operator_high_history_risk")
            validation_required = True
            mutating_allowed = False
        else:
            reasons.append("mutating_operator_has_graph_or_retrieval_support")
    elif selected in PROTOCOL_OPERATORS:
        decision_mode = "protocol_or_environment_validation"
        validation_required = True
        reasons.append("selected_operator_is_non_mutating_validation")
    else:
        selected = graph_operator or "protocol_first_target_validation"
        decision_mode = "graph_fallback_operator"
        reasons.append("retrieval_operator_missing_or_unknown")
        validation_required = selected in PROTOCOL_OPERATORS
        mutating_allowed = selected in MUTATING_OPERATORS

    low_support_collection_blocker = (
        selected == "environment_preflight_or_oracle_target_repair"
        and graph_operator == "environment_preflight_or_oracle_target_repair"
        and str(runtime.get("graph_top_failure_type", "") or "") == "test_collection_blocker"
        and str(runtime.get("graph_top_responsible_stage", "") or "") == "environment"
        and str(runtime.get("shared_fact_count_bucket", "") or "") == "1"
        and total_neighbor_similarity < 1.0
        and "patcher_fixed_localization" in candidates
    )
    if low_support_collection_blocker:
        selected = "patch_contract_nonregression_repatch"
        decision_mode = "low_support_collection_repatch_override"
        reasons.append("test_collection_blocker_with_low_neighbor_support_prefers_localized_repatch")
        reasons.append("patcher_candidate_available_for_low_support_collection_blocker")
        validation_required = True
        mutating_allowed = True

    graph_only = not retrieval_operator
    if (
        graph_only
        and selected in MUTATING_OPERATORS
        and bool(runtime.get("has_invalid_test_target_signal", False))
    ):
        selected = "protocol_first_target_validation"
        decision_mode = "graph_invalid_target_validation_override"
        reasons.append("graph_summary_invalid_target_defers_mutating_recovery")
        validation_required = True
        mutating_allowed = False

    if (
        graph_only
        and selected in MUTATING_OPERATORS
        and bool(runtime.get("has_missing_dependency_signal", False))
        and bool(runtime.get("has_oracle_verifier_contradiction", False))
        and bool(runtime.get("has_source_mixed_patch", False))
        and not bool(runtime.get("has_test_collection_blocker", False))
    ):
        selected = "environment_preflight_or_oracle_target_repair"
        decision_mode = "graph_environment_dominates_source_mixed_patch_override"
        reasons.append("missing_dependency_and_oracle_verifier_contradiction_before_source_mixed_patch_mutation")
        validation_required = True
        mutating_allowed = False

    graph_agrees = bool(graph_operator and selected == graph_operator)
    retrieval_agrees = bool(retrieval_operator and selected == retrieval_operator)
    if graph_operator and retrieval_operator and graph_operator == retrieval_operator:
        reasons.append("graph_and_retrieval_operator_agree")
    elif graph_operator and retrieval_operator and graph_operator != retrieval_operator:
        reasons.append("graph_and_retrieval_operator_disagree")

    selected = selected or "protocol_first_target_validation"
    return {
        "schema": "mas_dx_r_fusion_recovery_decision_v1",
        "instance_id": str(instance_id or row.get("instance_id", "") or ""),
        "selected_policy_action": selected,
        "planned_operator": selected if selected != "confidence_gated_clean_start_fallback" else retrieval_operator,
        "legacy_recovery_action": OPERATOR_TO_LEGACY_LABEL.get(selected, ""),
        "run_scope": _run_scope(selected),
        "decision_mode": decision_mode,
        "fusion_inputs": {
            "graph_operator": graph_operator,
            "retrieval_operator": retrieval_operator,
            "preference_operator": preference_operator,
            "preference_admissible": preference_admissible,
            "preference_evidence_gate": preference_gate,
            "preference_score": preference.get("preference_score"),
            "retrieval_planned_operator": planned_operator,
            "graph_agrees_with_selected": graph_agrees,
            "retrieval_agrees_with_selected": retrieval_agrees,
            "graph_top_failure_type": str(runtime.get("graph_top_failure_type", "") or ""),
            "graph_top_confidence": str(runtime.get("graph_top_confidence", "") or ""),
            "candidate_recovery_actions": candidates,
            "policy_score_margin": margin,
            "total_neighbor_similarity": total_neighbor_similarity,
            "retrieval_policy_confidence": str(row.get("policy_confidence", "") or ""),
            "retrieval_policy_score": float(row.get("policy_score", 0.0) or 0.0),
        },
        "bounded_validation": {
            "required_before_mutation": bool(validation_required and selected in MUTATING_OPERATORS),
            "validation_operator": (
                selected
                if selected in PROTOCOL_OPERATORS
                else "protocol_first_target_validation"
            ),
            "mutating_operator_allowed": bool(mutating_allowed),
            "max_mutating_interventions": 1 if mutating_allowed else 0,
        },
        "policy_reasons": _dedupe(reasons or ["fusion_default_selected_operator"]),
        "claim_boundary": {
            "pre_oracle_policy": True,
            "provider_free": True,
            "does_not_execute_recovery": True,
            "does_not_use_target_outcome": True,
            "aaai_preference_branch_explicit_only": True,
        },
    }


def _runtime_from_graph(graph: dict[str, Any], hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(graph.get("summary", {}) or {})
    top = dict(hypotheses[0]) if hypotheses else {}
    runtime = {
        "graph_top_failure_type": str(top.get("failure_type", "") or ""),
        "graph_top_confidence": str(top.get("confidence", "") or ""),
        "graph_top_responsible_stage": str(top.get("responsible_stage", "") or ""),
        "candidate_recovery_actions": list(summary.get("candidate_recovery_actions", []) or []),
    }
    for key, value in summary.items():
        runtime.setdefault(str(key), value)
    return runtime


def _graph_operator(runtime: dict[str, Any], hypotheses: list[dict[str, Any]]) -> str:
    failure_type = str(runtime.get("graph_top_failure_type", "") or "")
    if not failure_type and hypotheses:
        failure_type = str(dict(hypotheses[0]).get("failure_type", "") or "")
    return GRAPH_FAILURE_TO_OPERATOR.get(failure_type, "")


def _protocol_reasons(*, runtime: dict[str, Any], graph_operator: str) -> list[str]:
    reasons: list[str] = []
    failure_type = str(runtime.get("graph_top_failure_type", "") or "")
    if graph_operator in PROTOCOL_OPERATORS:
        reasons.append("graph_operator_requires_protocol_or_environment_validation")
    if failure_type in {"test_environment_blocker", "test_collection_blocker", "verifier_protocol_drift", "verifier_acceptance_gap"}:
        reasons.append(f"graph_failure_type_requires_validation:{failure_type}")
    return _dedupe(reasons)


def _score_margin(ranked: list[dict[str, Any]]) -> float:
    if len(ranked) < 2:
        return 999.0
    scores = [float(item.get("score", 0.0) or 0.0) for item in ranked[:2]]
    return round(scores[0] - scores[1], 4)


def _run_scope(operator: str) -> str:
    if operator == "environment_preflight_or_oracle_target_repair":
        return "environment_preflight_then_verifier"
    if operator == "protocol_first_target_validation":
        return "verifier_only_replay"
    if operator in MUTATING_OPERATORS:
        return "patcher_fixed_localization"
    if operator == "confidence_gated_clean_start_fallback":
        return "clean_start_or_swe_repair"
    return "verifier_only_replay"


def _preference_evidence_gate(preference: dict[str, Any]) -> dict[str, Any]:
    mode = str(preference.get("mode", "") or "")
    if mode != "mas_evidence_fail_closed_selector_v1":
        return {"admissible": True, "mode": mode, "blockers": []}
    gate = dict(preference.get("evidence_gate", {}) or {})
    blockers: list[str] = []
    all69_overlap = int(gate.get("all69_overlap_count", 0) or 0)
    min_all69_overlap = int(gate.get("min_all69_overlap", 69) or 69)
    shadow_lift = int(gate.get("shadow_lift", 0) or 0)
    shadow_hurt = int(gate.get("shadow_hurt", 0) or 0)
    if all69_overlap < min_all69_overlap:
        blockers.append("all69_overlap_below_threshold")
    if shadow_lift <= 0:
        blockers.append("shadow_lift_not_positive")
    if shadow_hurt != 0:
        blockers.append("shadow_hurt_not_zero")
    if not bool(gate.get("pre_oracle", False)):
        blockers.append("selector_not_pre_oracle")
    if not bool(gate.get("provider_free", False)):
        blockers.append("selector_not_provider_free")
    if not bool(gate.get("fail_closed_to_current_masguard", False)):
        blockers.append("selector_not_fail_closed")
    return {
        "admissible": not blockers,
        "mode": mode,
        "all69_overlap_count": all69_overlap,
        "min_all69_overlap": min_all69_overlap,
        "shadow_lift": shadow_lift,
        "shadow_hurt": shadow_hurt,
        "blockers": blockers,
    }


def _preference_decision_mode(preference: dict[str, Any], *, default: str) -> str:
    if str(preference.get("mode", "") or "") == "mas_evidence_fail_closed_selector_v1":
        return "mas_evidence_fail_closed_preference"
    return default


def _preference_selected_reason(preference: dict[str, Any]) -> str:
    if str(preference.get("mode", "") or "") == "mas_evidence_fail_closed_selector_v1":
        return "explicit_mas_evidence_fail_closed_profile_selected_operator"
    return "explicit_aaai_preference_profile_selected_operator"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "")
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
