"""Select compatible recovery decisions from MAS-DX-R hypotheses and observations."""

from __future__ import annotations

from typing import Any


ACTION_TO_LEGACY_LABEL = {
    "environment_preflight_then_verifier": "environment_repair",
    "verifier_only_replay": "rerun_verifier",
    "patcher_fixed_localization": "repatch_with_existing_localization",
    "evidence_conditioned_repatch": "repatch_with_existing_localization",
    "shared_fact_quarantine_then_repatch": "repatch_with_existing_localization",
    "handoff_correction_then_verifier": "rerun_verifier",
    "protocol_first_target_validation": "protocol_first_target_validation",
    "stage_checkpoint_replay": "repatch_with_existing_localization",
    "no_recovery_protocol_blocked": "protocol_blocked",
    "record_recovered_previous_patch_state": "rerun_verifier",
    "record_recovered_candidate_patch_state": "repatch_with_existing_localization",
    "bounded_source_only_retry_after_candidate_replay": "repatch_with_existing_localization",
}


def select_recovery_decision(
    graph: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    observations = list(observations or [])
    instance_id = str(graph.get("instance_id", "") or "")
    terminal = _terminal_decision(instance_id, observations)
    if terminal:
        return terminal
    if observations and all(bool(item.get("protocol_blocked", False)) for item in observations):
        action = "no_recovery_protocol_blocked"
        selected = {}
        confidence = "low"
    else:
        selected = _select_hypothesis(hypotheses)
        action = _action_for_hypothesis(selected, observations)
        confidence = str(selected.get("confidence", "") or "low")
    return {
        "schema": "mas_dx_r_recovery_decision_v1",
        "instance_id": instance_id,
        "selected_action": action,
        "selected_run_scope": _run_scope_for_action(action),
        "legacy_recovery_action": ACTION_TO_LEGACY_LABEL.get(action, ""),
        "confirmed_hypothesis_id": str(selected.get("hypothesis_id", "") or ""),
        "failure_type": str(selected.get("failure_type", "") or ""),
        "responsible_stage": str(selected.get("responsible_stage", "") or ""),
        "confidence": confidence,
        "supporting_observations": [
            str(item.get("step_id", "") or "")
            for item in observations
            if str(item.get("status", "") or "") in {"confirmed", "weakened", "contradicted"}
        ],
        "fallback_action": _fallback_action(action),
        "compatibility_mode": "external_controller_only",
        "expected_action_adherence": _expected_adherence(action),
        "do_not_do": _do_not_do(action),
    }


def _terminal_decision(instance_id: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    for observation in observations:
        signal = str(observation.get("observed_signal", "") or "")
        if signal == "candidate_patch_replay_succeeded":
            action = "record_recovered_candidate_patch_state"
            return {
                "schema": "mas_dx_r_recovery_decision_v1",
                "instance_id": instance_id or str(observation.get("instance_id", "") or ""),
                "selected_action": action,
                "selected_run_scope": _run_scope_for_action(action),
                "legacy_recovery_action": ACTION_TO_LEGACY_LABEL.get(action, ""),
                "confirmed_hypothesis_id": "",
                "failure_type": "recovered_by_candidate_patch_replay",
                "responsible_stage": "patcher",
                "confidence": "high",
                "supporting_observations": [str(observation.get("step_id", "") or "")],
                "fallback_action": "",
                "compatibility_mode": "external_controller_only",
                "expected_action_adherence": _expected_adherence(action),
                "do_not_do": _do_not_do(action),
            }
        if signal != "recovery_succeeded":
            continue
        action = "record_recovered_previous_patch_state"
        return {
            "schema": "mas_dx_r_recovery_decision_v1",
            "instance_id": instance_id or str(observation.get("instance_id", "") or ""),
            "selected_action": action,
            "selected_run_scope": _run_scope_for_action(action),
            "legacy_recovery_action": ACTION_TO_LEGACY_LABEL.get(action, ""),
            "confirmed_hypothesis_id": "",
            "failure_type": "recovered_by_intervention",
            "responsible_stage": "verifier",
            "confidence": "high",
            "supporting_observations": [str(observation.get("step_id", "") or "")],
            "fallback_action": "",
            "compatibility_mode": "external_controller_only",
            "expected_action_adherence": _expected_adherence(action),
            "do_not_do": _do_not_do(action),
        }
    return {}


def _select_hypothesis(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    if not hypotheses:
        return {
            "hypothesis_id": "hyp:ambiguous",
            "failure_type": "ambiguous",
            "candidate_recovery_action": "verifier_only_replay",
            "confidence": "low",
        }
    return dict(hypotheses[0])


def _action_for_hypothesis(hypothesis: dict[str, Any], observations: list[dict[str, Any]]) -> str:
    failure_type = str(hypothesis.get("failure_type", "") or "")
    if _has_clean_failed_candidate_patch_replay(observations):
        return "bounded_source_only_retry_after_candidate_replay"
    if _has_graph_static_patch_failure_span(observations):
        return "evidence_conditioned_repatch"
    if _has_replay_blocked_but_patch_failure_evidence(observations):
        return "evidence_conditioned_repatch"
    if _has_clean_environment_preflight_failure(observations):
        return "evidence_conditioned_repatch"
    if (
        failure_type in {"verifier_acceptance_gap", "verifier_protocol_drift", "handoff_information_loss"}
        and _has_clean_failed_verifier_replay(observations)
    ):
        return "evidence_conditioned_repatch"
    if failure_type in {"test_environment_blocker", "test_collection_blocker"}:
        return "environment_preflight_then_verifier"
    if failure_type in {"verifier_acceptance_gap", "verifier_protocol_drift", "handoff_information_loss"}:
        return "verifier_only_replay"
    if failure_type == "shared_fact_contamination":
        return "shared_fact_quarantine_then_repatch"
    if failure_type in {"local_patch_regression", "patch_incomplete"}:
        return "patcher_fixed_localization"
    return str(hypothesis.get("candidate_recovery_action", "") or "verifier_only_replay")


def _has_graph_static_patch_failure_span(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        if str(observation.get("observed_signal", "") or "") == "graph_static_patch_failure_span":
            return True
    return False


def _has_replay_blocked_but_patch_failure_evidence(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        signal = str(observation.get("observed_signal", "") or "")
        if signal not in {"previous_patch_replay_blocked", "candidate_patch_replay_blocked"}:
            continue
        if str(observation.get("status", "") or "") not in {"protocol_blocked", "weakened"}:
            continue
        if str(observation.get("fallback_mutation_basis", "") or "") == "graph_patch_failure_span":
            return True
    return False


def _has_clean_failed_verifier_replay(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        if str(observation.get("intervention_type", "") or "") != "verifier_replay":
            continue
        if bool(observation.get("protocol_blocked", False)):
            continue
        signal = str(observation.get("observed_signal", "") or "")
        status = str(observation.get("status", "") or "")
        if signal in {
            "validated_targets_collectable_but_tests_fail",
            "verifier_replay_clean_failure",
        }:
            return True
        if status in {"weakened", "contradicted"}:
            return True
    return False


def _has_clean_environment_preflight_failure(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        if str(observation.get("intervention_type", "") or "") != "environment_preflight":
            continue
        if bool(observation.get("protocol_blocked", False)):
            continue
        signal = str(observation.get("observed_signal", "") or "")
        status = str(observation.get("status", "") or "")
        if signal == "environment_preflight_clean_but_tests_fail":
            return True
        if status in {"weakened", "contradicted"}:
            return True
    return False


def _has_clean_failed_candidate_patch_replay(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        if str(observation.get("intervention_type", "") or "") != "candidate_patch_replay_verify":
            continue
        if bool(observation.get("protocol_blocked", False)):
            continue
        signal = str(observation.get("observed_signal", "") or "")
        status = str(observation.get("status", "") or "")
        if signal == "candidate_patch_replay_clean_failure":
            return True
        if status in {"weakened", "contradicted"}:
            return True
    return False


def _run_scope_for_action(action: str) -> str:
    return {
        "environment_preflight_then_verifier": "environment_preflight_then_verifier",
        "verifier_only_replay": "verifier_only_replay",
        "patcher_fixed_localization": "patcher_fixed_localization",
        "evidence_conditioned_repatch": "patcher_fixed_localization",
        "shared_fact_quarantine_then_repatch": "patcher_fixed_localization",
        "handoff_correction_then_verifier": "verifier_only_replay",
        "protocol_first_target_validation": "verifier_only_replay",
        "stage_checkpoint_replay": "patcher_fixed_localization",
        "no_recovery_protocol_blocked": "none",
        "record_recovered_previous_patch_state": "none",
        "record_recovered_candidate_patch_state": "none",
        "bounded_source_only_retry_after_candidate_replay": "patcher_fixed_localization",
    }.get(action, "verifier_only_replay")


def _fallback_action(action: str) -> str:
    if action == "environment_preflight_then_verifier":
        return "verifier_only_replay"
    if action == "verifier_only_replay":
        return "patcher_fixed_localization"
    if action == "evidence_conditioned_repatch":
        return "no_recovery_protocol_blocked"
    if action == "bounded_source_only_retry_after_candidate_replay":
        return "no_recovery_protocol_blocked"
    if action == "patcher_fixed_localization":
        return "verifier_only_replay"
    return "verifier_only_replay"


def _expected_adherence(action: str) -> dict[str, Any]:
    return {
        "patch_allowed": action in {
            "patcher_fixed_localization",
            "evidence_conditioned_repatch",
            "shared_fact_quarantine_then_repatch",
            "stage_checkpoint_replay",
            "bounded_source_only_retry_after_candidate_replay",
        },
        "verifier_required": action
        in {
            "environment_preflight_then_verifier",
            "verifier_only_replay",
            "patcher_fixed_localization",
            "evidence_conditioned_repatch",
            "handoff_correction_then_verifier",
            "protocol_first_target_validation",
            "bounded_source_only_retry_after_candidate_replay",
        },
        "environment_preflight_required": action == "environment_preflight_then_verifier",
        "fixed_localization_required": action
        in {
            "patcher_fixed_localization",
            "evidence_conditioned_repatch",
            "shared_fact_quarantine_then_repatch",
            "bounded_source_only_retry_after_candidate_replay",
        },
    }


def _do_not_do(action: str) -> list[str]:
    if action == "verifier_only_replay":
        return ["do_not_repatch_before_oracle_aligned_verifier_replay"]
    if action == "environment_preflight_then_verifier":
        return ["do_not_repatch_before_environment_or_collection_blocker_is_resolved"]
    if action == "patcher_fixed_localization":
        return ["do_not_rerun_locator", "do_not_expand_beyond_existing_localization_without_new_evidence"]
    if action == "evidence_conditioned_repatch":
        return [
            "do_not_repeat_verifier_replay_without_new_evidence",
            "do_not_rerun_locator",
            "do_not_expand_beyond_existing_localization_without_new_evidence",
        ]
    if action == "record_recovered_previous_patch_state":
        return ["do_not_repatch_after_confirmed_recovery"]
    if action == "record_recovered_candidate_patch_state":
        return ["do_not_repatch_after_candidate_patch_replay_recovered"]
    if action == "bounded_source_only_retry_after_candidate_replay":
        return [
            "do_not_repeat_failed_candidate_patch_without_new_evidence",
            "do_not_rerun_locator",
            "do_not_edit_tests",
            "preserve_source_only_patch_contract",
        ]
    return []
