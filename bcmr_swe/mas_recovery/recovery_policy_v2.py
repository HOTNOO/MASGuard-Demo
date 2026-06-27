"""Harm-aware recovery policy layer for MAS-DX-R v2.

This module wraps the existing v1 recovery decision. It does not execute
experiments and does not look at oracle outcomes. Its job is to decide whether a
graph-selected recovery action is sufficiently grounded, whether a protocol gate
must run first, or whether a clean-start fallback should be used.
"""

from __future__ import annotations

from typing import Any

from bcmr_swe.mas_recovery.failure_span_family import infer_failure_span_family
from bcmr_swe.mas_recovery.recovery_controller import select_recovery_decision


FALLBACK_ACTION = "confidence_gated_clean_start_fallback"
MINIMAL_PROBE_ACTION = "uncertainty_gated_minimal_validation_probe"


def select_harm_aware_recovery_decision(
    graph: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
    *,
    min_graph_confidence: str = "medium",
) -> dict[str, Any]:
    observations = list(observations or [])
    base = select_recovery_decision(graph, hypotheses, observations)
    selected = _selected_hypothesis(base, hypotheses)
    gate = _policy_gate(
        graph=graph,
        selected_hypothesis=selected,
        base_decision=base,
        observations=observations,
        min_graph_confidence=min_graph_confidence,
    )
    if gate["decision_mode"] == "fallback_clean_start":
        return _fallback_decision(base, gate)
    if gate["decision_mode"] == "minimal_probe_required":
        return _minimal_probe_decision(base, gate)
    if gate["decision_mode"] == "protocol_first":
        return _protocol_first_decision(base, gate)
    return _augment_decision(base, gate)


def _policy_gate(
    *,
    graph: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    base_decision: dict[str, Any],
    observations: list[dict[str, Any]],
    min_graph_confidence: str,
) -> dict[str, Any]:
    summary = dict(graph.get("summary", {}) or {})
    confidence = str(selected_hypothesis.get("confidence", "") or base_decision.get("confidence", "") or "low")
    failure_type = str(selected_hypothesis.get("failure_type", "") or base_decision.get("failure_type", "") or "")
    selected_action = str(base_decision.get("selected_action", "") or "")
    failure_span = infer_failure_span_family(observations, top_updated=selected_hypothesis) if observations else {}
    protocol_signals = _protocol_signals(summary, selected_hypothesis, observations, base_decision)
    evidence_score = _evidence_score(summary, selected_hypothesis)
    reasons: list[str] = []

    if selected_action in {
        "no_recovery_protocol_blocked",
        "record_recovered_previous_patch_state",
        "record_recovered_candidate_patch_state",
    }:
        return {
            "schema": "mas_dx_r_recovery_policy_gate_v2",
            "decision_mode": "use_graph_recovery",
            "operator_gate": selected_action,
            "evidence_score": evidence_score,
            "confidence": confidence,
            "failure_span_family": failure_span,
            "policy_reasons": ["terminal_or_protocol_decision"],
            "rejected_actions": [],
        }

    if protocol_signals:
        reasons.extend(protocol_signals)
        return {
            "schema": "mas_dx_r_recovery_policy_gate_v2",
            "decision_mode": "protocol_first",
            "operator_gate": "protocol_first_target_validation",
            "evidence_score": evidence_score,
            "confidence": confidence,
            "failure_span_family": failure_span,
            "policy_reasons": reasons,
            "rejected_actions": [selected_action] if selected_action else [],
        }

    if _confidence_rank(confidence) < _confidence_rank(min_graph_confidence):
        reasons.append("graph_confidence_below_policy_threshold")
    if failure_type in {"ambiguous", ""}:
        reasons.append("ambiguous_failure_hypothesis")
    if evidence_score < 2 and selected_action in {
        "patcher_fixed_localization",
        "evidence_conditioned_repatch",
        "shared_fact_quarantine_then_repatch",
    }:
        reasons.append("mutating_recovery_without_sufficient_graph_evidence")

    if reasons and _has_recovery_promoting_observation(observations) and selected_action in {
        "patcher_fixed_localization",
        "evidence_conditioned_repatch",
        "shared_fact_quarantine_then_repatch",
        "bounded_source_only_retry_after_candidate_replay",
    }:
        operator_gate = _operator_gate_for_action(selected_action, failure_type)
        span_gate = str(failure_span.get("routed_operator_gate", "") or "")
        if (
            span_gate == "semantic_invariant_guarded_repatch"
            and selected_action
            in {
                "patcher_fixed_localization",
                "evidence_conditioned_repatch",
                "bounded_source_only_retry_after_candidate_replay",
            }
        ):
            operator_gate = span_gate
        return {
            "schema": "mas_dx_r_recovery_policy_gate_v2",
            "decision_mode": "use_graph_recovery",
            "operator_gate": operator_gate,
            "evidence_score": evidence_score,
            "confidence": confidence,
            "failure_span_family": failure_span,
            "policy_reasons": _dedupe(
                ["bounded_minimal_probe_observation_supports_constrained_recovery"] + reasons
            ),
            "rejected_actions": [],
            "minimal_probe": {
                "required_before_this_decision": True,
                "satisfied_by_observation": True,
                "credit_boundary": "observation_conditions_action_only_not_recovery_credit",
            },
        }

    if reasons:
        minimal_probe = _minimal_probe_gate(
            selected_action=selected_action,
            failure_type=failure_type,
            observations=observations,
            reasons=reasons,
        )
        if minimal_probe:
            return {
                "schema": "mas_dx_r_recovery_policy_gate_v2",
                "decision_mode": "minimal_probe_required",
                "operator_gate": MINIMAL_PROBE_ACTION,
                "evidence_score": evidence_score,
                "confidence": confidence,
                "failure_span_family": failure_span,
                "policy_reasons": _dedupe(
                    reasons + ["uncertainty_requires_bounded_minimal_probe_before_action"]
                ),
                "rejected_actions": [selected_action] if selected_action else [],
                "minimal_probe": minimal_probe,
            }
        return {
            "schema": "mas_dx_r_recovery_policy_gate_v2",
            "decision_mode": "fallback_clean_start",
            "operator_gate": FALLBACK_ACTION,
            "evidence_score": evidence_score,
            "confidence": confidence,
            "failure_span_family": failure_span,
            "policy_reasons": reasons,
            "rejected_actions": [selected_action] if selected_action else [],
        }

    operator_gate = _operator_gate_for_action(selected_action, failure_type)
    span_gate = str(failure_span.get("routed_operator_gate", "") or "")
    if (
        span_gate == "semantic_invariant_guarded_repatch"
        and selected_action
        in {
            "patcher_fixed_localization",
            "evidence_conditioned_repatch",
            "bounded_source_only_retry_after_candidate_replay",
        }
    ):
        operator_gate = span_gate
        reasons.append(str(failure_span.get("route_reason", "") or "semantic_span_family_route"))
    return {
        "schema": "mas_dx_r_recovery_policy_gate_v2",
        "decision_mode": "use_graph_recovery",
        "operator_gate": operator_gate,
        "evidence_score": evidence_score,
        "confidence": confidence,
        "failure_span_family": failure_span,
        "policy_reasons": ["graph_evidence_sufficient"] + _dedupe(reasons),
        "rejected_actions": [],
    }


def _fallback_decision(base: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    out = _augment_decision(base, gate)
    out["selected_action"] = FALLBACK_ACTION
    out["selected_run_scope"] = "clean_start_or_swe_repair"
    out["legacy_recovery_action"] = "clean_start_or_swe_repair"
    out["fallback_action"] = ""
    out["compatibility_mode"] = "external_controller_only"
    out["expected_action_adherence"] = {
        "patch_allowed": True,
        "verifier_required": True,
        "environment_preflight_required": False,
        "fixed_localization_required": False,
        "fallback_selected_before_oracle": True,
    }
    out["do_not_do"] = [
        "do_not_execute_low_confidence_graph_mutation",
        "do_not_use_expert_label_as_runtime_input",
    ]
    return out


def _minimal_probe_decision(base: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    out = _augment_decision(base, gate)
    probe = dict(gate.get("minimal_probe", {}) or {})
    base_action = str(base.get("selected_action", "") or "")
    out["selected_action"] = MINIMAL_PROBE_ACTION
    out["selected_run_scope"] = str(probe.get("run_scope", "") or "verifier_only_replay")
    out["legacy_recovery_action"] = "minimal_validation_probe"
    out["fallback_action"] = "confidence_gated_clean_start_fallback"
    out["deferred_graph_recovery_action"] = base_action
    out["compatibility_mode"] = "external_controller_only"
    out["expected_action_adherence"] = {
        "patch_allowed": False,
        "verifier_required": str(probe.get("intervention_type", "") or "") == "verifier_replay",
        "environment_preflight_required": str(probe.get("intervention_type", "") or "") == "environment_preflight",
        "fixed_localization_required": False,
        "minimal_probe_required": True,
        "fallback_selected_before_oracle": False,
    }
    do_not_do = list(out.get("do_not_do", []) or [])
    do_not_do.extend(
        [
            "do_not_patch_before_minimal_probe",
            "do_not_select_deferred_recovery_without_probe_update",
            "do_not_count_probe_as_recovery_credit",
        ]
    )
    out["do_not_do"] = _dedupe(do_not_do)
    return out


def _protocol_first_decision(base: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    out = _augment_decision(base, gate)
    base_action = str(base.get("selected_action", "") or "")
    out["selected_action"] = "protocol_first_target_validation"
    out["selected_run_scope"] = "verifier_only_replay"
    out["legacy_recovery_action"] = "protocol_first_target_validation"
    out["fallback_action"] = "confidence_gated_clean_start_fallback"
    out["deferred_graph_recovery_action"] = base_action
    out["compatibility_mode"] = "external_controller_only"
    out["expected_action_adherence"] = {
        "patch_allowed": False,
        "verifier_required": True,
        "environment_preflight_required": False,
        "fixed_localization_required": False,
        "target_validation_required": True,
        "fallback_selected_before_oracle": False,
    }
    do_not_do = list(out.get("do_not_do", []) or [])
    do_not_do.extend(
        [
            "do_not_patch_during_protocol_first_validation",
            "do_not_execute_deferred_graph_recovery_until_protocol_gate_passes",
        ]
    )
    out["do_not_do"] = _dedupe(do_not_do)
    return out


def _augment_decision(base: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["schema"] = "mas_dx_r_recovery_decision_v2"
    out["policy_gate"] = gate
    out["decision_mode"] = str(gate.get("decision_mode", "") or "")
    out["operator_gate"] = str(gate.get("operator_gate", "") or "")
    out["pre_oracle_policy"] = True
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "")
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _selected_hypothesis(base: dict[str, Any], hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    selected_id = str(base.get("confirmed_hypothesis_id", "") or "")
    if selected_id:
        for hypothesis in hypotheses:
            if str(hypothesis.get("hypothesis_id", "") or "") == selected_id:
                return dict(hypothesis)
    if hypotheses:
        return dict(hypotheses[0])
    return {}


def _protocol_signals(
    summary: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    observations: list[dict[str, Any]],
    base_decision: dict[str, Any],
) -> list[str]:
    reasons = []
    failure_type = str(selected_hypothesis.get("failure_type", "") or base_decision.get("failure_type", "") or "")
    selected_action = str(base_decision.get("selected_action", "") or "")
    if observations and selected_action in {
        "evidence_conditioned_repatch",
        "patcher_fixed_localization",
        "bounded_source_only_retry_after_candidate_replay",
    }:
        if _has_recovery_promoting_observation(observations):
            return reasons
    if summary.get("has_invalid_test_target_signal"):
        reasons.append("invalid_or_uncollectable_test_target_signal")
    if summary.get("has_test_collection_blocker") and failure_type not in {"local_patch_regression"}:
        reasons.append("collection_or_import_blocker_before_recovery")
    if summary.get("has_missing_dependency_signal") or summary.get("has_external_dependency_blocker_signal"):
        reasons.append("environment_dependency_blocker_signal")
    if observations and all(bool(item.get("protocol_blocked", False)) for item in observations):
        reasons.append("all_interventions_protocol_blocked")
    return reasons


def _has_recovery_promoting_observation(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        signal = str(observation.get("observed_signal", "") or "")
        if signal in {
            "verifier_replay_clean_failure",
            "validated_targets_collectable_but_tests_fail",
            "environment_preflight_clean_but_tests_fail",
            "candidate_patch_replay_clean_failure",
            "graph_static_patch_failure_span",
        }:
            return True
    return False


def _minimal_probe_gate(
    *,
    selected_action: str,
    failure_type: str,
    observations: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    if _has_recovery_promoting_observation(observations):
        return {}
    if selected_action in {
        "no_recovery_protocol_blocked",
        "record_recovered_previous_patch_state",
        "record_recovered_candidate_patch_state",
        "protocol_first_target_validation",
        "environment_preflight_then_verifier",
        "verifier_only_replay",
    }:
        return {}
    if not (
        selected_action
        in {
            "patcher_fixed_localization",
            "evidence_conditioned_repatch",
            "shared_fact_quarantine_then_repatch",
            "bounded_source_only_retry_after_candidate_replay",
        }
        or "ambiguous_failure_hypothesis" in reasons
        or "graph_confidence_below_policy_threshold" in reasons
        or "mutating_recovery_without_sufficient_graph_evidence" in reasons
    ):
        return {}

    intervention_type = _minimal_probe_type(failure_type)
    run_scope = {
        "environment_preflight": "environment_preflight_then_verifier",
        "pytest_collect_only": "verifier_only_replay",
        "verifier_replay": "verifier_only_replay",
    }.get(intervention_type, "verifier_only_replay")
    return {
        "schema": "mas_dx_r_minimal_validation_probe_v1",
        "intervention_type": intervention_type,
        "run_scope": run_scope,
        "mutation_scope": "environment_check_only" if intervention_type == "environment_preflight" else "none",
        "tests_failure_type": failure_type or "ambiguous",
        "updates_deferred_action": selected_action,
        "stop_condition": "stop_after_confirmed_weakened_contradicted_or_protocol_blocked",
        "credit_boundary": "probe_updates_hypothesis_only_not_recovery_credit",
    }


def _minimal_probe_type(failure_type: str) -> str:
    if failure_type in {"test_environment_blocker", "test_collection_blocker"}:
        return "environment_preflight"
    if failure_type in {"ambiguous", ""}:
        return "pytest_collect_only"
    return "verifier_replay"


def _evidence_score(summary: dict[str, Any], selected_hypothesis: dict[str, Any]) -> int:
    score = 0
    spans = list(selected_hypothesis.get("supporting_evidence_span_ids", []) or [])
    if spans:
        score += 1
    if summary.get("has_semantic_edges") or summary.get("semantic_edge_counts"):
        score += 1
    if summary.get("has_handoff_edges") or summary.get("has_shared_fact_artifact"):
        score += 1
    if summary.get("has_oracle_verifier_contradiction") or summary.get("has_invalid_test_target_signal"):
        score += 1
    return score


def _operator_gate_for_action(action: str, failure_type: str) -> str:
    if action in {"shared_fact_quarantine_then_repatch"} or failure_type == "shared_fact_contamination":
        return "shared_fact_quarantine_then_repatch"
    if failure_type == "handoff_information_loss":
        return "handoff_correction_or_ablation"
    if action == "environment_preflight_then_verifier":
        return "environment_preflight_or_oracle_target_repair"
    if action in {
        "patcher_fixed_localization",
        "evidence_conditioned_repatch",
        "bounded_source_only_retry_after_candidate_replay",
    }:
        return "patch_contract_nonregression_repatch"
    if action == "verifier_only_replay":
        return "verifier_replay_gate"
    return action or "unknown_operator_gate"


def _confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(confidence, 0)
