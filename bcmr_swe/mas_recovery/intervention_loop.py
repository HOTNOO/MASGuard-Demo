"""Build bounded minimal intervention plans for MAS-DX-R."""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_INTERVENTIONS = 3


def build_intervention_plan(
    graph: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    *,
    max_interventions: int = DEFAULT_MAX_INTERVENTIONS,
) -> dict[str, Any]:
    instance_id = str(graph.get("instance_id", "") or "")
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        for intervention_type in list(hypothesis.get("recommended_interventions", []) or []):
            if intervention_type in seen:
                continue
            seen.add(intervention_type)
            steps.append(_step(len(steps) + 1, intervention_type, hypothesis))
            if len(steps) >= max_interventions:
                break
        if len(steps) >= max_interventions:
            break
    return {
        "schema": "mas_dx_r_intervention_plan_v1",
        "instance_id": instance_id,
        "max_interventions": max_interventions,
        "budget": {
            "max_interventions": max_interventions,
            "max_mutating_interventions": 1,
            "default_timeout_seconds": 300,
        },
        "hypotheses": [dict(item) for item in hypotheses],
        "steps": steps,
        "stop_conditions": [
            "stop_after_confirmed_recovery_action",
            "stop_after_all_remaining_steps_protocol_blocked",
            "stop_after_budget_exhausted",
        ],
    }


def planned_observations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-executed observation rows for dry-run evaluation."""

    rows = []
    for step in list(plan.get("steps", []) or []):
        rows.append(
            {
                "schema": "mas_dx_r_intervention_observation_v1",
                "step_id": str(step.get("step_id", "") or ""),
                "instance_id": str(plan.get("instance_id", "") or ""),
                "intervention_type": str(step.get("intervention_type", "") or ""),
                "status": "planned_not_executed",
                "returncode": None,
                "output_excerpt": "",
                "observed_signal": "",
                "hypothesis_update": "inconclusive",
                "protocol_blocked": False,
                "error_type": "",
                "cost": {},
            }
        )
    return rows


def _step(index: int, intervention_type: str, hypothesis: dict[str, Any]) -> dict[str, Any]:
    mutation_scope = {
        "pytest_collect_only": "none",
        "focused_test_rerun": "none",
        "verifier_replay": "none",
        "environment_preflight": "environment_check_only",
        "patch_syntax_check": "none",
        "fixed_localization_repatch_dry_plan": "planned_patch_only",
        "shared_fact_quarantine_plan": "planned_state_filter_only",
        "handoff_ablation_plan": "planned_handoff_filter_only",
    }.get(intervention_type, "unknown")
    return {
        "step_id": f"step:{index:02d}:{intervention_type}",
        "intervention_type": intervention_type,
        "tests_hypothesis_id": str(hypothesis.get("hypothesis_id", "") or ""),
        "expected_signal": _expected_signal(intervention_type),
        "command_source": _command_source(intervention_type),
        "mutation_scope": mutation_scope,
        "requires_workspace": intervention_type not in {"shared_fact_quarantine_plan", "handoff_ablation_plan"},
        "timeout_seconds": 300,
        "stop_on": ["confirmed", "protocol_blocked"],
    }


def _expected_signal(intervention_type: str) -> str:
    return {
        "pytest_collect_only": "fail-to-pass targets are collectable or explicitly blocked",
        "focused_test_rerun": "target failure reproduces without broad search",
        "verifier_replay": "oracle-aligned verifier result resolves verifier contradiction",
        "environment_preflight": "environment can import runner and expose missing dependency blockers",
        "patch_syntax_check": "changed files compile or expose patch-introduced syntax failure",
        "fixed_localization_repatch_dry_plan": "patcher can operate within previous localization boundary",
        "shared_fact_quarantine_plan": "downstream can rerun without suspicious shared fact",
        "handoff_ablation_plan": "downstream can rerun with corrected handoff artifact",
    }.get(intervention_type, "intervention produces diagnostic signal")


def _command_source(intervention_type: str) -> str:
    return {
        "pytest_collect_only": "validated_fail_to_pass",
        "focused_test_rerun": "validated_fail_to_pass",
        "verifier_replay": "validated_fail_to_pass",
        "environment_preflight": "manifest",
        "patch_syntax_check": "changed_files",
        "fixed_localization_repatch_dry_plan": "trajectory_localization",
        "shared_fact_quarantine_plan": "trajectory_shared_facts",
        "handoff_ablation_plan": "propagation_graph",
    }.get(intervention_type, "")
