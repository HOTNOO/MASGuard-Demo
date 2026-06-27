"""Unified MAS-DX-R recovery blueprint planning.

This module stitches together the graph diagnosis, fusion selector, bounded
minimal intervention plan, operator-template evidence, and checkpoint policy.
It is deliberately pre-oracle and read-only.
"""

from __future__ import annotations

from typing import Any

from bcmr_swe.mas_recovery.checkpoint_policy import build_checkpoint_recovery_plan
from bcmr_swe.mas_recovery.fusion_controller import select_fusion_recovery_decision
from bcmr_swe.mas_recovery.intervention_loop import build_intervention_plan
from bcmr_swe.mas_recovery.recovery_policy_v2 import select_harm_aware_recovery_decision


MUTATING_OPERATORS = {
    "handoff_correction_or_ablation",
    "patch_contract_nonregression_repatch",
    "semantic_span_delta_guarded_repatch",
    "semantic_invariant_guarded_repatch",
    "shared_fact_quarantine_then_repatch",
}

OPERATOR_STEP_LIBRARY = {
    "environment_preflight_or_oracle_target_repair": [
        "environment_preflight",
        "pytest_collect_only",
        "verifier_replay",
    ],
    "protocol_first_target_validation": [
        "pytest_collect_only",
        "verifier_replay",
    ],
    "handoff_correction_or_ablation": [
        "handoff_ablation_plan",
        "verifier_replay",
        "fixed_localization_repatch_dry_plan",
    ],
    "patch_contract_nonregression_repatch": [
        "patch_syntax_check",
        "verifier_replay",
        "fixed_localization_repatch_dry_plan",
    ],
    "semantic_invariant_guarded_repatch": [
        "patch_syntax_check",
        "semantic_invariant_guard",
        "verifier_replay",
        "fixed_localization_repatch_dry_plan",
    ],
    "semantic_span_delta_guarded_repatch": [
        "patch_syntax_check",
        "semantic_invariant_guard",
        "verifier_replay",
        "post_patch_span_delta_guard",
        "fixed_localization_repatch_dry_plan",
    ],
    "shared_fact_quarantine_then_repatch": [
        "shared_fact_quarantine_plan",
        "verifier_replay",
        "fixed_localization_repatch_dry_plan",
    ],
    "confidence_gated_clean_start_fallback": [],
    "uncertainty_gated_minimal_validation_probe": [
        "pytest_collect_only",
    ],
}


def build_recovery_blueprint(
    *,
    graph: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    retrieval_policy_row: dict[str, Any] | None = None,
    preference_profile: dict[str, Any] | None = None,
    template_row: dict[str, Any] | None = None,
    execution_override_row: dict[str, Any] | None = None,
    observations: list[dict[str, Any]] | None = None,
    max_interventions: int = 3,
) -> dict[str, Any]:
    """Build one pre-execution recovery blueprint.

    Runtime inputs must be pre-oracle: propagation graph, hypotheses, optional
    retrieval-policy row, optional template row, and optional bounded
    observations. The returned blueprint is not execute permission and is not a
    recovery metric.
    """

    observations = list(observations or [])
    graph = dict(graph or {})
    hypotheses = [dict(item) for item in list(hypotheses or []) if isinstance(item, dict)]
    retrieval_policy_row = dict(retrieval_policy_row or {})
    template_row = dict(template_row or {})
    execution_override_row = dict(execution_override_row or {})
    instance_id = str(
        graph.get("instance_id", "")
        or retrieval_policy_row.get("instance_id", "")
        or template_row.get("instance_id", "")
        or execution_override_row.get("instance_id", "")
        or ""
    )

    fusion = select_fusion_recovery_decision(
        instance_id=instance_id,
        retrieval_policy_row=retrieval_policy_row or None,
        graph=graph if not retrieval_policy_row else None,
        hypotheses=hypotheses if not retrieval_policy_row else None,
        preference_profile=preference_profile,
    )
    graph_decision = select_harm_aware_recovery_decision(graph, hypotheses, observations)
    selected_operator = _select_operator(fusion=fusion, graph_decision=graph_decision, template_row=template_row)
    selected_run_scope = _run_scope(
        operator=selected_operator,
        fusion=fusion,
        graph_decision=graph_decision,
        template_row=template_row,
    )
    selected_operator, selected_run_scope = _apply_execution_override(
        selected_operator=selected_operator,
        selected_run_scope=selected_run_scope,
        execution_override_row=execution_override_row,
    )
    patch_contract = _patch_contract(
        selected_operator=selected_operator,
        selected_run_scope=selected_run_scope,
        execution_override_row=execution_override_row,
    )
    operator_steps = _operator_steps(
        selected_operator=selected_operator,
        selected_run_scope=selected_run_scope,
        fusion=fusion,
        graph_decision=graph_decision,
    )
    graph_intervention_plan = build_intervention_plan(
        graph,
        hypotheses,
        max_interventions=max_interventions,
    )
    checkpoint_plan = build_checkpoint_recovery_plan(
        graph,
        _decision_for_checkpoint(
            instance_id=instance_id,
            selected_operator=selected_operator,
            selected_run_scope=selected_run_scope,
            fusion=fusion,
            graph_decision=graph_decision,
        ),
    )
    issue_list = _issues(
        selected_operator=selected_operator,
        selected_run_scope=selected_run_scope,
        patch_contract=patch_contract,
        fusion=fusion,
        graph_decision=graph_decision,
        template_row=template_row,
        execution_override_row=execution_override_row,
        checkpoint_plan=checkpoint_plan,
    )
    return {
        "schema": "mas_dx_r_recovery_blueprint_v1",
        "instance_id": instance_id,
        "claim_boundary": {
            "pre_oracle_policy": True,
            "provider_free": True,
            "does_not_execute_recovery": True,
            "does_not_call_models": True,
            "does_not_use_target_outcome": True,
            "does_not_use_expert_label_as_runtime_input": True,
            "does_not_add_recovery_credit": True,
            "blueprint_is_not_execute_permission": True,
        },
        "inputs_used": {
            "graph": bool(graph),
            "hypotheses": bool(hypotheses),
            "retrieval_policy_row": bool(retrieval_policy_row),
            "preference_profile": bool(preference_profile),
            "template_row": bool(template_row),
            "execution_override_row": bool(execution_override_row),
            "observations": bool(observations),
        },
        "graph_signal_digest": _graph_signal_digest(graph),
        "selected_operator": selected_operator,
        "selected_run_scope": selected_run_scope,
        "legacy_recovery_action": str(fusion.get("legacy_recovery_action", "") or graph_decision.get("legacy_recovery_action", "") or ""),
        "decision_mode": _decision_mode(fusion, graph_decision, template_row),
        "fusion_decision": fusion,
        "preference_profile_support": _preference_profile_support(preference_profile),
        "graph_policy_decision": graph_decision,
        "template_support": _template_support(template_row),
        "execution_override": _execution_override_support(execution_override_row),
        "bounded_validation": {
            "required_before_mutation": bool(
                dict(fusion.get("bounded_validation", {}) or {}).get("required_before_mutation", False)
                or str(graph_decision.get("decision_mode", "") or "") == "protocol_first"
                or str(graph_decision.get("decision_mode", "") or "") == "minimal_probe_required"
            ),
            "mutating_operator": selected_operator in MUTATING_OPERATORS
            or selected_run_scope == "candidate_patch_verifier",
            "mutating_operator_allowed_now": bool(
                str(graph_decision.get("decision_mode", "") or "") != "minimal_probe_required"
                and (
                    dict(fusion.get("bounded_validation", {}) or {}).get("mutating_operator_allowed", False)
                    or selected_run_scope == "candidate_patch_verifier"
                )
            ),
            "max_mutating_interventions": (
                1 if selected_operator in MUTATING_OPERATORS or selected_run_scope == "candidate_patch_verifier" else 0
            ),
            "operator_step_count": len(operator_steps),
            "operator_steps": operator_steps,
            "graph_intervention_plan": graph_intervention_plan,
        },
        "checkpoint_recovery_plan": checkpoint_plan,
        "budget_contract": _budget_contract(selected_operator, selected_run_scope, operator_steps),
        "execution_contract": {
            "run_scope": selected_run_scope,
            "patch_contract": patch_contract,
            "operator_gate": selected_operator if selected_operator in MUTATING_OPERATORS else "",
            "evaluation_command_source": "validated_fail_to_pass",
            "action_adherence_required": True,
            "patch_contract_required": bool(patch_contract),
            "checkpoint_resume_allowed": bool(checkpoint_plan.get("eligible", False)),
            "required_post_execution_fields": [
                "oracle_success",
                "fail_to_pass_returncode",
                "oracle_returncode",
                "model_call_count",
                "token_cost",
                "evaluation_protocol_error",
                "action_adherence_observed",
                "candidate_patch_replayed",
            ],
        },
        "issue_count": len(issue_list),
        "issues": issue_list,
        "blueprint_ready_for_predeclared_plan": not issue_list,
    }


def _select_operator(
    *,
    fusion: dict[str, Any],
    graph_decision: dict[str, Any],
    template_row: dict[str, Any],
) -> str:
    template_gate = str(template_row.get("operator_gate", "") or "")
    if template_gate:
        return template_gate
    if str(graph_decision.get("decision_mode", "") or "") == "minimal_probe_required":
        return str(graph_decision.get("operator_gate", "") or "uncertainty_gated_minimal_validation_probe")
    observation_operator = _operator_from_observation_promoted_graph_decision(graph_decision)
    if observation_operator:
        return observation_operator
    fusion_operator = str(fusion.get("selected_policy_action", "") or "")
    if fusion_operator:
        return fusion_operator
    return str(graph_decision.get("operator_gate", "") or graph_decision.get("selected_action", "") or "protocol_first_target_validation")


def _preference_profile_support(preference_profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(preference_profile or {})
    if not profile:
        return {}
    return {
        "schema": str(profile.get("schema", "") or ""),
        "mode": str(profile.get("mode", "") or ""),
        "selected_policy_action": str(profile.get("selected_policy_action", "") or ""),
        "run_scope": str(profile.get("run_scope", "") or ""),
        "preference_score": profile.get("preference_score"),
        "admissible": bool(profile.get("admissible", False)),
        "override_allowed": bool(profile.get("override_allowed", False)),
        "explicit_branch_only": bool(dict(profile.get("claim_boundary", {}) or {}).get("explicit_branch_only", False)),
    }


def _run_scope(
    *,
    operator: str,
    fusion: dict[str, Any],
    graph_decision: dict[str, Any],
    template_row: dict[str, Any],
) -> str:
    template_scope = str(template_row.get("selected_run_scope", "") or "")
    if template_scope:
        return template_scope
    if str(graph_decision.get("decision_mode", "") or "") == "minimal_probe_required":
        return str(graph_decision.get("selected_run_scope", "") or "verifier_only_replay")
    observation_scope = _run_scope_from_observation_promoted_graph_decision(graph_decision)
    if observation_scope:
        return observation_scope
    if str(fusion.get("run_scope", "") or ""):
        return str(fusion.get("run_scope", "") or "")
    if str(graph_decision.get("selected_run_scope", "") or ""):
        return str(graph_decision.get("selected_run_scope", "") or "")
    if operator in MUTATING_OPERATORS:
        return "patcher_fixed_localization"
    if operator == "environment_preflight_or_oracle_target_repair":
        return "environment_preflight_then_verifier"
    return "verifier_only_replay"


def _operator_from_observation_promoted_graph_decision(graph_decision: dict[str, Any]) -> str:
    selected_action = str(graph_decision.get("selected_action", "") or "")
    operator_gate = str(graph_decision.get("operator_gate", "") or "")
    if operator_gate == "semantic_invariant_guarded_repatch":
        return operator_gate
    span_gate = str(
        dict(dict(graph_decision.get("policy_gate", {}) or {}).get("failure_span_family", {}) or {}).get(
            "routed_operator_gate", ""
        )
        or ""
    )
    if span_gate == "semantic_invariant_guarded_repatch" and list(
        graph_decision.get("supporting_observations", []) or []
    ):
        return span_gate
    if selected_action == "bounded_source_only_retry_after_candidate_replay":
        return "patch_contract_nonregression_repatch"
    if selected_action not in {"evidence_conditioned_repatch", "patcher_fixed_localization"}:
        return ""
    if not list(graph_decision.get("supporting_observations", []) or []):
        return ""
    return "patch_contract_nonregression_repatch"


def _run_scope_from_observation_promoted_graph_decision(graph_decision: dict[str, Any]) -> str:
    operator = _operator_from_observation_promoted_graph_decision(graph_decision)
    if not operator:
        return ""
    if operator in MUTATING_OPERATORS:
        return "patcher_fixed_localization"
    return str(graph_decision.get("selected_run_scope", "") or "patcher_fixed_localization")


def _apply_execution_override(
    *,
    selected_operator: str,
    selected_run_scope: str,
    execution_override_row: dict[str, Any],
) -> tuple[str, str]:
    if not execution_override_row:
        return selected_operator, selected_run_scope
    scope = str(
        execution_override_row.get("effective_predeclared_run_scope", "")
        or execution_override_row.get("selected_run_scope", "")
        or execution_override_row.get("run_scope", "")
        or selected_run_scope
        or ""
    )
    override_operator = str(execution_override_row.get("operator_gate", "") or "")
    if scope == "candidate_patch_verifier":
        operator = override_operator or selected_operator or "patch_contract_nonregression_repatch"
    elif not override_operator and scope in {"environment_preflight_then_verifier", "verifier_only_replay"}:
        operator = _operator_for_scope(scope)
    else:
        operator = selected_operator or override_operator
    return operator, scope


def _operator_for_scope(scope: str) -> str:
    if scope == "environment_preflight_then_verifier":
        return "environment_preflight_or_oracle_target_repair"
    if scope == "verifier_only_replay":
        return "protocol_first_target_validation"
    return ""


def _patch_contract(
    *,
    selected_operator: str,
    selected_run_scope: str,
    execution_override_row: dict[str, Any],
) -> str:
    explicit = str(execution_override_row.get("patch_contract", "") or "")
    if explicit:
        return explicit
    if selected_operator in MUTATING_OPERATORS or selected_run_scope == "candidate_patch_verifier":
        return "source_only"
    return ""


def _operator_steps(
    *,
    selected_operator: str,
    selected_run_scope: str,
    fusion: dict[str, Any],
    graph_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    if selected_run_scope == "candidate_patch_verifier":
        return [
            _operator_step(index, step_type)
            for index, step_type in enumerate(
                ["patch_syntax_check", "source_patch_replay", "verifier_replay"],
                start=1,
            )
        ]
    step_types = list(OPERATOR_STEP_LIBRARY.get(selected_operator, []) or [])
    if (
        selected_operator in MUTATING_OPERATORS
        and bool(dict(fusion.get("bounded_validation", {}) or {}).get("required_before_mutation", False))
    ):
        # Keep validation ahead of mutation when the fusion controller deferred
        # mutation behind a bounded protocol check.
        step_types = ["pytest_collect_only", "verifier_replay"] + [
            step for step in step_types if step not in {"pytest_collect_only", "verifier_replay"}
        ]
    if str(graph_decision.get("decision_mode", "") or "") == "protocol_first":
        step_types = ["pytest_collect_only", "verifier_replay"]
    if str(graph_decision.get("decision_mode", "") or "") == "minimal_probe_required":
        probe = dict(dict(graph_decision.get("policy_gate", {}) or {}).get("minimal_probe", {}) or {})
        step_types = [str(probe.get("intervention_type", "") or "pytest_collect_only")]
    return [_operator_step(index, step_type) for index, step_type in enumerate(_dedupe(step_types), start=1)]


def _operator_step(index: int, step_type: str) -> dict[str, Any]:
    mutation_scope = {
        "pytest_collect_only": "none",
        "verifier_replay": "none",
        "environment_preflight": "environment_check_only",
        "patch_syntax_check": "none",
        "semantic_invariant_guard": "none",
        "source_patch_replay": "source_patch_replay",
        "handoff_ablation_plan": "planned_handoff_filter_only",
        "shared_fact_quarantine_plan": "planned_state_filter_only",
        "fixed_localization_repatch_dry_plan": "planned_patch_only",
    }.get(step_type, "unknown")
    return {
        "step_id": f"operator_step:{index:02d}:{step_type}",
        "intervention_type": step_type,
        "mutation_scope": mutation_scope,
        "requires_workspace": step_type not in {"handoff_ablation_plan", "shared_fact_quarantine_plan"},
        "command_source": {
            "pytest_collect_only": "validated_fail_to_pass",
            "verifier_replay": "validated_fail_to_pass",
            "environment_preflight": "manifest",
            "patch_syntax_check": "changed_files",
            "semantic_invariant_guard": "changed_files",
            "source_patch_replay": "candidate_patch_artifact",
            "handoff_ablation_plan": "propagation_graph",
            "shared_fact_quarantine_plan": "trajectory_shared_facts",
            "fixed_localization_repatch_dry_plan": "trajectory_localization",
        }.get(step_type, ""),
        "stop_on": ["confirmed", "contradicted", "protocol_blocked"],
    }


def _decision_for_checkpoint(
    *,
    instance_id: str,
    selected_operator: str,
    selected_run_scope: str,
    fusion: dict[str, Any],
    graph_decision: dict[str, Any],
) -> dict[str, Any]:
    action = str(graph_decision.get("selected_action", "") or "")
    if action == "bounded_source_only_retry_after_candidate_replay":
        action = "patcher_fixed_localization"
    elif selected_operator == "patch_contract_nonregression_repatch":
        action = "patcher_fixed_localization"
    elif selected_operator == "semantic_invariant_guarded_repatch":
        action = "patcher_fixed_localization"
    elif selected_operator == "shared_fact_quarantine_then_repatch":
        action = "shared_fact_quarantine_then_repatch"
    elif selected_operator == "handoff_correction_or_ablation":
        action = "evidence_conditioned_repatch"
    elif selected_operator == "protocol_first_target_validation":
        action = "protocol_first_target_validation"
    elif selected_operator == "environment_preflight_or_oracle_target_repair":
        action = "environment_preflight_then_verifier"
    return {
        "schema": "mas_dx_r_recovery_decision_v2",
        "instance_id": instance_id,
        "selected_action": action,
        "selected_run_scope": selected_run_scope,
        "decision_mode": str(fusion.get("decision_mode", "") or graph_decision.get("decision_mode", "") or ""),
        "operator_gate": selected_operator,
    }


def _graph_signal_digest(graph: dict[str, Any]) -> dict[str, Any]:
    summary = dict(graph.get("summary", {}) or {})
    keys = [
        "has_semantic_edges",
        "semantic_edge_counts",
        "has_handoff_edges",
        "has_shared_fact_artifact",
        "has_shared_fact_verifier_dependency",
        "has_oracle_verifier_contradiction",
        "has_invalid_test_target_signal",
        "has_missing_dependency_signal",
        "has_test_collection_blocker",
        "has_reusable_localization",
        "has_patch_state_artifact",
        "patch_text_replay_shape_ok",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def _template_support(template_row: dict[str, Any]) -> dict[str, Any]:
    if not template_row:
        return {
            "template_selected": False,
            "runtime_oracle_safe": True,
        }
    leakage = dict(template_row.get("runtime_oracle_leakage_check", {}) or {})
    runtime_features = dict(template_row.get("runtime_features", {}) or {})
    if not leakage:
        leakage = {
            "runtime_features_oracle_safe": True,
            "suspicious_feature_keys": [],
        }
    return {
        "template_selected": bool(template_row.get("template_selected", False) or template_row.get("selected_template_id", "")),
        "selected_template_id": str(template_row.get("selected_template_id", "") or template_row.get("template_id", "") or ""),
        "selected_operator_name": str(template_row.get("selected_operator_name", "") or template_row.get("operator_name", "") or ""),
        "template_score": float(template_row.get("template_score", template_row.get("score", 0.0)) or 0.0),
        "runtime_oracle_safe": bool(leakage.get("runtime_features_oracle_safe", True)),
        "suspicious_feature_keys": list(leakage.get("suspicious_feature_keys", []) or []),
        "runtime_feature_keys": sorted(runtime_features.keys()),
    }


def _execution_override_support(execution_override_row: dict[str, Any]) -> dict[str, Any]:
    if not execution_override_row:
        return {
            "override_present": False,
        }
    return {
        "override_present": True,
        "operator_gate": str(execution_override_row.get("operator_gate", "") or ""),
        "effective_predeclared_run_scope": str(
            execution_override_row.get("effective_predeclared_run_scope", "")
            or execution_override_row.get("selected_run_scope", "")
            or execution_override_row.get("run_scope", "")
            or ""
        ),
        "patch_contract": str(execution_override_row.get("patch_contract", "") or ""),
        "candidate_patch_replay_present": bool(execution_override_row.get("candidate_patch_replay_present", False)),
        "candidate_patch_replay_oracle_success": bool(
            execution_override_row.get("candidate_patch_replay_oracle_success", False)
        ),
        "formal_admission_ready": bool(execution_override_row.get("formal_admission_ready", False)),
    }


def _budget_contract(operator: str, run_scope: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    mutating = operator in MUTATING_OPERATORS or run_scope == "candidate_patch_verifier"
    model_free_patch_replay = run_scope == "candidate_patch_verifier"
    return {
        "max_model_calls": 0 if model_free_patch_replay else 12 if mutating else 0,
        "max_completion_tokens": 0 if model_free_patch_replay else 10240 if mutating else 0,
        "max_mutating_interventions": 1 if mutating else 0,
        "max_source_patch_replays": 1 if model_free_patch_replay else 0,
        "zero_model_validation_step_count": sum(
            1 for step in steps if step.get("mutation_scope") in {"none", "environment_check_only"}
        ),
        "token_accounting_required": True,
        "budget_claim_boundary": "budget_contract_is_pre_execution_cap_not_observed_cost",
    }


def _decision_mode(fusion: dict[str, Any], graph_decision: dict[str, Any], template_row: dict[str, Any]) -> str:
    if template_row and str(template_row.get("selected_template_id", "") or template_row.get("template_id", "") or ""):
        return "template_conditioned_" + str(fusion.get("decision_mode", "") or graph_decision.get("decision_mode", "") or "graph")
    return str(fusion.get("decision_mode", "") or graph_decision.get("decision_mode", "") or "graph")


def _issues(
    *,
    selected_operator: str,
    selected_run_scope: str,
    patch_contract: str,
    fusion: dict[str, Any],
    graph_decision: dict[str, Any],
    template_row: dict[str, Any],
    execution_override_row: dict[str, Any],
    checkpoint_plan: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not selected_operator:
        issues.append("missing_selected_operator")
    if not selected_run_scope:
        issues.append("missing_selected_run_scope")
    if template_row:
        support = _template_support(template_row)
        if not support["runtime_oracle_safe"]:
            issues.append("template_runtime_features_not_oracle_safe")
    claim = dict(fusion.get("claim_boundary", {}) or {})
    if fusion and not bool(claim.get("does_not_execute_recovery", False)):
        issues.append("fusion_decision_missing_read_only_boundary")
    graph_claim = bool(graph_decision.get("pre_oracle_policy", False)) or graph_decision.get("schema") == "mas_dx_r_recovery_decision_v2"
    if graph_decision and not graph_claim:
        issues.append("graph_policy_decision_missing_pre_oracle_boundary")
    if (
        selected_operator in MUTATING_OPERATORS
        and selected_run_scope not in {"patcher_fixed_localization", "candidate_patch_verifier"}
    ):
        issues.append("mutating_operator_requires_patcher_fixed_localization")
    if selected_run_scope == "candidate_patch_verifier":
        if patch_contract != "source_only":
            issues.append("candidate_patch_verifier_requires_source_only_patch_contract")
        if not execution_override_row:
            issues.append("candidate_patch_verifier_requires_execution_override_row")
    if selected_operator == "environment_preflight_or_oracle_target_repair" and selected_run_scope != "environment_preflight_then_verifier":
        issues.append("environment_operator_requires_environment_preflight_scope")
    if selected_operator == "protocol_first_target_validation" and selected_run_scope != "verifier_only_replay":
        issues.append("protocol_operator_requires_verifier_only_scope")
    if checkpoint_plan and not dict(checkpoint_plan.get("claim_boundary", {}) or {}).get("does_not_count_as_recovery_result", False):
        issues.append("checkpoint_plan_missing_noncredit_boundary")
    return _dedupe(issues)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
