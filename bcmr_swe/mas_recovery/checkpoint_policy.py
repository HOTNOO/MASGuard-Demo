"""Checkpoint-aware recovery planning for MAS-DX-R.

The planner is deliberately conservative: it only proposes a checkpoint when
the propagation graph gives a clear uncontaminated prefix. It never executes a
recovery and never observes post-recovery oracle results.
"""

from __future__ import annotations

from typing import Any


def build_checkpoint_recovery_plan(
    graph: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(graph.get("summary", {}) or {})
    action = str(decision.get("selected_action", "") or "")
    run_scope = str(decision.get("selected_run_scope", "") or "")
    mode = str(decision.get("decision_mode", "") or "")
    operator_gate = str(decision.get("operator_gate", "") or "")
    blockers: list[str] = []

    if action in {"record_recovered_previous_patch_state", "no_recovery_protocol_blocked"}:
        blockers.append("terminal_or_protocol_blocked_decision")
    if action == "confidence_gated_clean_start_fallback" or run_scope == "clean_start_or_swe_repair":
        blockers.append("clean_start_fallback_selected")
    if not summary.get("has_reusable_localization", False) and not summary.get("has_patch_state_artifact", False):
        blockers.append("no_reusable_localization_or_patch_state")

    contamination = _shared_fact_contamination(summary)
    if contamination and operator_gate not in {"shared_fact_quarantine_then_repatch", "handoff_correction_or_ablation"}:
        blockers.append("shared_fact_contamination_crosses_checkpoint_boundary")

    candidate = _candidate_checkpoint(summary, action, run_scope, mode, operator_gate, contamination)
    if not candidate:
        blockers.append("no_supported_checkpoint_for_selected_action")

    eligible = bool(candidate and not blockers)
    if not eligible:
        candidate = candidate or _empty_candidate()

    return {
        "schema": "mas_dx_r_checkpoint_recovery_plan_v1",
        "instance_id": str(graph.get("instance_id", "") or decision.get("instance_id", "") or ""),
        "eligible": eligible,
        "checkpoint_id": candidate["checkpoint_id"] if eligible else "",
        "checkpoint_type": candidate["checkpoint_type"] if eligible else "",
        "resume_stage": candidate["resume_stage"] if eligible else "",
        "clean_prefix_stages": candidate["clean_prefix_stages"] if eligible else [],
        "skipped_stages": candidate["skipped_stages"] if eligible else [],
        "replay_required_stages": candidate["replay_required_stages"] if eligible else [],
        "guard_conditions": candidate["guard_conditions"] if eligible else [],
        "blockers": blockers,
        "expected_savings": (
            _expected_savings(candidate)
            if eligible
            else {"skipped_stage_count": 0, "expected_model_call_reduction": "none"}
        ),
        "claim_boundary": {
            "pre_oracle_decision": True,
            "does_not_count_as_recovery_result": True,
            "requires_runner_checkpoint_support_before_execution": True,
        },
    }


def _candidate_checkpoint(
    summary: dict[str, Any],
    action: str,
    run_scope: str,
    mode: str,
    operator_gate: str,
    contamination: bool,
) -> dict[str, Any]:
    if mode == "protocol_first" or operator_gate == "protocol_first_target_validation":
        if summary.get("has_missing_dependency_signal") or summary.get("has_external_dependency_blocker_signal"):
            return {}
        if summary.get("has_patch_state_artifact") and summary.get("patch_text_replay_shape_ok"):
            return {
                "checkpoint_id": "ckpt:pre_verifier_patch_state",
                "checkpoint_type": "pre_verifier_patch_state",
                "resume_stage": "verifier",
                "clean_prefix_stages": ["locator", "patcher"],
                "skipped_stages": ["locator", "patcher"],
                "replay_required_stages": ["verifier"],
                "guard_conditions": [
                    "patch_state_artifact_available",
                    "validate_test_targets_before_mutation",
                    "do_not_repatch_until_protocol_gate_passes",
                ],
            }
        if summary.get("has_reusable_localization"):
            return {
                "checkpoint_id": "ckpt:post_localization_protocol_gate",
                "checkpoint_type": "post_localization",
                "resume_stage": "verifier",
                "clean_prefix_stages": ["locator"],
                "skipped_stages": ["locator"],
                "replay_required_stages": ["verifier"],
                "guard_conditions": [
                    "localization_artifact_available",
                    "validate_test_targets_before_mutation",
                ],
            }
    if action in {"verifier_only_replay"} and summary.get("has_patch_state_artifact"):
        return {
            "checkpoint_id": "ckpt:pre_verifier_patch_state",
            "checkpoint_type": "pre_verifier_patch_state",
            "resume_stage": "verifier",
            "clean_prefix_stages": ["locator", "patcher"],
            "skipped_stages": ["locator", "patcher"],
            "replay_required_stages": ["verifier"],
            "guard_conditions": [
                "patch_state_artifact_available",
                "verifier_targets_validated_or_collected",
            ],
        }
    if operator_gate == "handoff_correction_or_ablation" and summary.get("has_reusable_localization"):
        return {
            "checkpoint_id": "ckpt:post_localization_pre_handoff",
            "checkpoint_type": "post_localization_pre_handoff",
            "resume_stage": "patcher" if run_scope == "patcher_fixed_localization" else "verifier",
            "clean_prefix_stages": ["locator"],
            "skipped_stages": ["locator"],
            "replay_required_stages": ["patcher", "verifier"] if run_scope == "patcher_fixed_localization" else ["verifier"],
            "guard_conditions": [
                "localization_artifact_available",
                "handoff_artifact_corrected_or_ablated",
                "downstream_stage_must_not_consume_stale_handoff",
            ],
        }
    if operator_gate == "shared_fact_quarantine_then_repatch" and summary.get("has_reusable_localization"):
        return {
            "checkpoint_id": "ckpt:post_localization_pre_shared_fact",
            "checkpoint_type": "post_localization_pre_shared_fact",
            "resume_stage": "patcher",
            "clean_prefix_stages": ["locator"],
            "skipped_stages": ["locator"],
            "replay_required_stages": ["patcher", "verifier"],
            "guard_conditions": [
                "localization_artifact_available",
                "quarantine_suspicious_shared_facts_before_resume",
                "downstream_stage_must_not_consume_stale_shared_fact",
            ],
        }
    if action in {"patcher_fixed_localization", "evidence_conditioned_repatch"} and summary.get("has_reusable_localization"):
        if contamination:
            return {}
        return {
            "checkpoint_id": "ckpt:post_localization",
            "checkpoint_type": "post_localization",
            "resume_stage": "patcher",
            "clean_prefix_stages": ["locator"],
            "skipped_stages": ["locator"],
            "replay_required_stages": ["patcher", "verifier"],
            "guard_conditions": [
                "localization_artifact_available",
                "no_shared_fact_contamination_crosses_boundary",
                "fixed_localization_required",
            ],
        }
    if action == "environment_preflight_then_verifier":
        return {
            "checkpoint_id": "ckpt:environment_preflight",
            "checkpoint_type": "environment_preflight",
            "resume_stage": "verifier",
            "clean_prefix_stages": [],
            "skipped_stages": [],
            "replay_required_stages": ["environment", "verifier"],
            "guard_conditions": [
                "environment_preflight_must_pass_before_mutation",
                "do_not_count_protocol_repair_as_recovery",
            ],
        }
    return {}


def _shared_fact_contamination(summary: dict[str, Any]) -> bool:
    return bool(summary.get("has_shared_fact_artifact") and summary.get("has_suspicious_shared_fact_signal"))


def _expected_savings(candidate: dict[str, Any]) -> dict[str, Any]:
    skipped = list(candidate.get("skipped_stages", []) or [])
    if not skipped:
        reduction = "low"
    elif len(skipped) == 1:
        reduction = "medium"
    else:
        reduction = "high"
    return {
        "skipped_stage_count": len(skipped),
        "skipped_stages": skipped,
        "expected_model_call_reduction": reduction,
        "token_saving_claim_status": "planned_metric_not_yet_measured",
    }


def _empty_candidate() -> dict[str, Any]:
    return {
        "checkpoint_id": "",
        "checkpoint_type": "",
        "resume_stage": "",
        "clean_prefix_stages": [],
        "skipped_stages": [],
        "replay_required_stages": [],
        "guard_conditions": [],
    }
