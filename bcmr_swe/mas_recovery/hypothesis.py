"""Generate structured MAS-DX-R failure hypotheses from propagation graphs."""

from __future__ import annotations

from typing import Any


def generate_failure_hypotheses(graph: dict[str, Any]) -> list[dict[str, Any]]:
    summary = dict(graph.get("summary", {}) or {})
    instance_id = str(graph.get("instance_id", "") or "")
    hypotheses: list[dict[str, Any]] = []

    patch_syntax = bool(summary.get("has_patch_syntax_or_collection_error"))
    contradiction = bool(summary.get("has_oracle_verifier_contradiction"))
    patch_import_regression = bool(summary.get("has_patch_introduced_import_regression_signal"))
    patch_collection_conflict = bool(summary.get("has_patch_collection_causality_conflict"))
    handoff_invalid_target = bool(
        summary.get("has_invalid_test_target_signal")
        and (summary.get("has_handoff_edges") or summary.get("has_shared_fact_artifact"))
        and not contradiction
        and not patch_import_regression
    )
    if (
        (summary.get("has_missing_dependency_signal") or summary.get("has_external_dependency_blocker_signal"))
        and not patch_syntax
        and not contradiction
        and not patch_import_regression
    ):
        hypotheses.append(
            _hypothesis(
                instance_id,
                "test_environment_blocker",
                "environment",
                "artifact:test_targets",
                "",
                "high",
                "Verifier evidence contains missing dependency or unavailable test environment signals.",
                ["span:verifier", "span:oracle"],
                ["pytest_collect_only", "environment_preflight"],
                "environment_preflight_then_verifier",
            )
        )
    elif summary.get("has_test_collection_blocker") and not patch_syntax and not contradiction and not handoff_invalid_target:
        hypotheses.append(
            _hypothesis(
                instance_id,
                "test_collection_blocker",
                "environment",
                "artifact:test_targets",
                "",
                "high",
                "The failure appears before target test execution during collection/import.",
                ["span:verifier", "span:commands"],
                ["pytest_collect_only"],
                "environment_preflight_then_verifier",
            )
        )

    if summary.get("has_invalid_test_target_signal"):
        failure_type = (
            "handoff_information_loss"
            if handoff_invalid_target
            else "verifier_protocol_drift"
        )
        interventions = ["pytest_collect_only", "verifier_replay"]
        if handoff_invalid_target:
            interventions = ["pytest_collect_only", "handoff_ablation_plan", "verifier_replay"]
        hypotheses.append(
            _hypothesis(
                instance_id,
                failure_type,
                "verifier",
                "artifact:test_targets",
                "edge:localization_to_verifier",
                "medium",
                (
                    "Verifier/test-target evidence suggests invalid or uncollectable test selection after upstream handoff."
                    if handoff_invalid_target
                    else "Verifier/test-target evidence suggests invalid or uncollectable test selection."
                ),
                ["span:verifier", "span:commands"],
                interventions,
                "verifier_only_replay",
            )
        )

    if summary.get("has_oracle_verifier_contradiction"):
        hypotheses.append(
            _hypothesis(
                instance_id,
                "verifier_acceptance_gap",
                "verifier",
                "artifact:verifier_verdict",
                "edge:oracle_to_verifier",
                "high",
                "Verifier reported success or accepted weak evidence while oracle/fail-to-pass still failed.",
                ["span:verifier", "span:oracle"],
                ["verifier_replay"],
                "verifier_only_replay",
            )
        )

    if (
        summary.get("has_shared_fact_verifier_dependency")
        and summary.get("has_suspicious_shared_fact_signal")
        and summary.get("has_patch_artifact")
        and not summary.get("oracle_success")
        and not summary.get("has_invalid_test_target_signal")
        and not summary.get("has_oracle_verifier_contradiction")
        and not summary.get("has_missing_dependency_signal")
        and not summary.get("has_patch_syntax_or_collection_error")
    ):
        hypotheses.append(
            _hypothesis(
                instance_id,
                "shared_fact_contamination",
                "verifier",
                "artifact:shared_fact",
                "edge:shared_fact_to_verifier",
                "medium",
                "Shared facts are consumed by downstream verifier/patcher while the propagated patch still fails oracle.",
                ["span:shared_facts", "span:verifier", "span:oracle"],
                ["shared_fact_quarantine_plan", "verifier_replay"],
                "shared_fact_quarantine_then_repatch",
            )
        )

    if summary.get("has_patch_syntax_or_collection_error") or patch_import_regression:
        hypotheses.append(
            _hypothesis(
                instance_id,
                "local_patch_regression",
                "patcher",
                "artifact:patch",
                "",
                "high",
                (
                    "Patch/import evidence conflicts with broad collection failure; run bounded collection verification before mutating."
                    if patch_collection_conflict
                    else
                    "Evidence names a patch-introduced import/runtime regression."
                    if patch_import_regression
                    else "Changed patch/test files appear to introduce syntax/import/collection failure."
                ),
                ["span:diff", "span:verifier"],
                (
                    ["pytest_collect_only", "patch_syntax_check", "fixed_localization_repatch_dry_plan"]
                    if patch_collection_conflict
                    else ["patch_syntax_check", "fixed_localization_repatch_dry_plan"]
                ),
                (
                    "environment_preflight_then_verifier"
                    if patch_collection_conflict
                    else "patcher_fixed_localization"
                ),
            )
        )

    if summary.get("has_patch_artifact") and not summary.get("oracle_success"):
        confidence = "medium" if hypotheses else "high"
        hypotheses.append(
            _hypothesis(
                instance_id,
                "patch_incomplete",
                "patcher",
                "artifact:patch",
                "",
                confidence,
                "A patch artifact exists but oracle evidence still fails.",
                ["span:diff", "span:oracle"],
                ["focused_test_rerun", "fixed_localization_repatch_dry_plan"],
                "patcher_fixed_localization",
            )
        )

    if summary.get("has_selection_only_handoff_failure"):
        confidence = "medium" if hypotheses else "high"
        hypotheses.append(
            _hypothesis(
                instance_id,
                "selection_to_patch_execution_gap",
                "patcher",
                "artifact:localization",
                "edge:localization_to_patcher",
                confidence,
                "Locator evidence reached the patcher, but no patch artifact was produced before the failed trajectory ended.",
                ["span:stage:locator", "span:stage:patcher", "span:oracle"],
                ["fixed_localization_repatch_dry_plan", "verifier_replay"],
                "patcher_fixed_localization",
            )
        )

    if not hypotheses:
        hypotheses.append(
            _hypothesis(
                instance_id,
                "ambiguous",
                "unknown",
                "",
                "",
                "low",
                "No deterministic propagation signal was strong enough for a specific hypothesis.",
                ["span:verifier", "span:oracle"],
                ["pytest_collect_only"],
                "verifier_only_replay",
            )
        )

    return _rank_hypotheses(hypotheses)


def _hypothesis(
    instance_id: str,
    failure_type: str,
    responsible_stage: str,
    faulty_artifact_node: str,
    faulty_edge: str,
    confidence: str,
    rationale: str,
    supporting_spans: list[str],
    interventions: list[str],
    recovery_action: str,
) -> dict[str, Any]:
    return {
        "schema": "mas_dx_r_failure_hypothesis_v1",
        "hypothesis_id": f"hyp:{failure_type}",
        "instance_id": instance_id,
        "failure_type": failure_type,
        "responsible_stage": responsible_stage,
        "faulty_artifact_node": faulty_artifact_node,
        "faulty_edge": faulty_edge,
        "confidence": confidence,
        "rationale": rationale,
        "supporting_evidence_span_ids": supporting_spans,
        "disconfirming_evidence_span_ids": [],
        "recommended_interventions": interventions,
        "candidate_recovery_action": recovery_action,
    }


def _rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "test_environment_blocker": 0,
        "local_patch_regression": 1,
        "verifier_acceptance_gap": 1,
        "handoff_information_loss": 2,
        "test_collection_blocker": 2,
        "verifier_protocol_drift": 2,
        "shared_fact_contamination": 4,
        "patch_incomplete": 5,
        "selection_to_patch_execution_gap": 5,
        "ambiguous": 99,
    }
    return sorted(
        hypotheses,
        key=lambda item: (
            _contextual_priority(item, priority),
            priority.get(str(item.get("failure_type", "")), 50),
        ),
    )


def _contextual_priority(item: dict[str, Any], priority: dict[str, int]) -> int:
    failure_type = str(item.get("failure_type", "") or "")
    action = str(item.get("candidate_recovery_action", "") or "")
    rationale = str(item.get("rationale", "") or "")
    if failure_type == "local_patch_regression" and action == "environment_preflight_then_verifier":
        return 3
    if failure_type == "test_collection_blocker" and "Patch/import evidence conflicts" in rationale:
        return 0
    return priority.get(failure_type, 50)
