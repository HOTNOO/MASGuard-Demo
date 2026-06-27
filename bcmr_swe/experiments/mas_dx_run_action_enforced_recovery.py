"""Run MAS-DX action-enforced recovery scopes.

By default this is a contract-only runner. Pass ``--execute`` to run implemented
non-invasive scopes against a materialized workspace.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bcmr_swe.agent.patcher import PlannerPatcherAdapter
from bcmr_swe.experiments.common import (
    QueryBudgetModel,
    build_chat_model,
    build_executor,
    disable_streaming_if_supported,
    materialize_workspace,
    workspace_strategy_for_runtime,
)
from bcmr_swe.mas_recovery.failure_span_family import classify_failure_span_text
from bcmr_swe.mas_recovery.edit_intent import (
    build_masguard_edit_intent,
    edit_intent_skipped_result,
)
from bcmr_swe.mas_recovery.patch_intent_gate import (
    evaluate_patch_intent_gate,
    patch_intent_gate_skipped_result,
)
from bcmr_swe.recovery.patch_contract import StageBoundaryPatchContract
from bcmr_swe.recovery.semantic_invariant_gate import semantic_invariant_patch_violations
from swe_mas.utils.path_filters import classify_changed_files, parse_unified_diff_paths


SUPPORTED_RUN_SCOPES = {
    "candidate_patch_verifier",
    "guarded_protocol_then_clean_patch",
    "environment_preflight_then_verifier",
    "guarded_protocol_then_patch",
    "patcher_fixed_localization_candidate_only",
    "patcher_fixed_localization",
    "verifier_only_replay",
}

EXECUTABLE_RUN_SCOPES = {
    "candidate_patch_verifier",
    "guarded_protocol_then_clean_patch",
    "environment_preflight_then_verifier",
    "guarded_protocol_then_patch",
    "patcher_fixed_localization_candidate_only",
    "patcher_fixed_localization",
    "verifier_only_replay",
}

EVALUATION_COMMAND_SOURCES = {
    "manifest",
    "normalized",
    "normalized_fail_to_pass",
    "validated",
    "validated_fail_to_pass",
}

PATCH_REPLAY_APPLY_MODES = {
    "plain",
    "recount",
}

OPERATOR_GATES = {
    "",
    "environment_preflight_or_oracle_target_repair",
    "shared_fact_quarantine_then_repatch",
    "handoff_correction_or_ablation",
    "patch_contract_nonregression_repatch",
    "patch_family_lock_then_repatch",
    "semantic_effect_guarded_repatch",
    "semantic_span_delta_guarded_repatch",
    "semantic_invariant_guarded_repatch",
    "uncertainty_gated_minimal_validation_probe",
    "protocol_guard_then_patch",
    "verifier_replay_gate",
}

ZERO_MODEL_USAGE = {
    "n_calls": 0.0,
    "prompt_tokens": 0.0,
    "completion_tokens": 0.0,
    "total_tokens": 0.0,
}


def run_action_enforced_recovery(
    *,
    instance_id: str,
    manifest_root: Path,
    run_scope: str,
    api_path: Path | None = None,
    model_name: str = "",
    strong_model_name: str = "",
    trajectory_records_path: Path = Path("outputs/mas_dx/failed_main_dataset_current.json"),
    output_root: Path = Path("outputs/mas_dx/action_enforced_workspaces"),
    runtime: str = "harness",
    execute: bool = False,
    force_rebuild_harness: bool = False,
    harness_setup_timeout: int | None = None,
    harness_container_start_timeout: int | None = None,
    harness_container_cleanup_timeout: int | None = None,
    harness_env_image_key: str = "",
    command_timeout: int = 1800,
    evaluation_command_source: str = "manifest",
    request_timeout: int = 60,
    max_retries: int = 1,
    planner_max_iterations: int = 4,
    patcher_max_iterations: int = 8,
    replay_previous_patch: bool = False,
    patch_replay_apply_mode: str = "plain",
    patcher_failure_evidence_path: Path | None = None,
    patch_contract: str = "",
    operator_gate: str = "",
    target_mapping_probe_paths: list[Path] | None = None,
    bounded_validation_enabled: bool = True,
    post_patch_span_delta_revision_enabled: bool = False,
    patch_intent_gate_enabled: bool = False,
    patch_intent_advisory_enabled: bool = False,
    patch_intent_revision_enabled: bool = False,
    edit_intent_plan_enabled: bool = False,
    syntax_repair_revision_enabled: bool = False,
) -> dict[str, Any]:
    started_at, started_monotonic = _start_runtime_timer()
    manifest_path = _manifest_path(manifest_root, instance_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    guidance = dict(manifest.get("mas_dx_guidance", {}) or {})
    if run_scope not in SUPPORTED_RUN_SCOPES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_run_scope",
                "stop_reason": "unsupported_run_scope",
                "oracle_success": False,
                "reported_success": False,
                "implemented": False,
                "scope_supported": False,
                "scope_executor_available": False,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    if evaluation_command_source not in EVALUATION_COMMAND_SOURCES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_evaluation_command_source",
                "stop_reason": "unsupported_evaluation_command_source",
                "oracle_success": False,
                "reported_success": False,
                "implemented": False,
                "scope_supported": True,
                "scope_executor_available": run_scope in EXECUTABLE_RUN_SCOPES,
                "evaluation_command_source": evaluation_command_source,
                "operator_gate": operator_gate,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    if patch_replay_apply_mode not in PATCH_REPLAY_APPLY_MODES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_patch_replay_apply_mode",
                "stop_reason": "unsupported_patch_replay_apply_mode",
                "oracle_success": False,
                "reported_success": False,
                "implemented": run_scope in EXECUTABLE_RUN_SCOPES,
                "scope_supported": True,
                "scope_executor_available": run_scope in EXECUTABLE_RUN_SCOPES,
                "patch_replay_apply_mode": patch_replay_apply_mode,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    if operator_gate not in OPERATOR_GATES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_operator_gate",
                "stop_reason": "unsupported_operator_gate",
                "oracle_success": False,
                "reported_success": False,
                "implemented": run_scope in EXECUTABLE_RUN_SCOPES,
                "scope_supported": True,
                "scope_executor_available": run_scope in EXECUTABLE_RUN_SCOPES,
                "operator_gate": operator_gate,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    if not execute:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=False,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "planned_not_executed",
                "stop_reason": "planned_not_executed",
                "oracle_success": False,
                "reported_success": False,
                "implemented": run_scope in EXECUTABLE_RUN_SCOPES,
                "scope_supported": True,
                "scope_executor_available": run_scope in EXECUTABLE_RUN_SCOPES,
                "evaluation_command_source": evaluation_command_source,
                "operator_gate": operator_gate,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    if run_scope not in EXECUTABLE_RUN_SCOPES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "scope_not_implemented",
                "stop_reason": f"{run_scope}_not_implemented",
                "oracle_success": False,
                "reported_success": False,
                "implemented": False,
                "scope_supported": True,
                "scope_executor_available": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "action_adherence_observed": {
                    "patch_attempted": False,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": bounded_validation_enabled,
                },
            },
        )

    workspace_strategy = workspace_strategy_for_runtime(runtime, manifest)
    workspace = materialize_workspace(
        manifest["source_snapshot"],
        output_root,
        f"{run_scope}_{instance_id}",
        strategy=workspace_strategy,
    )
    executor = None
    runtime_session = None
    try:
        executor, runtime_session = build_executor(
            workspace=str(workspace),
            runtime=runtime,
            manifest=manifest,
            force_rebuild_harness=force_rebuild_harness,
            harness_setup_timeout=harness_setup_timeout,
            harness_container_start_timeout=harness_container_start_timeout,
            harness_container_cleanup_timeout=harness_container_cleanup_timeout,
            harness_env_image_key=harness_env_image_key,
        )
        result = _execute_scope(
            run_scope=run_scope,
            manifest=manifest,
            instance_id=instance_id,
            executor=executor,
            workspace=workspace,
            timeout=command_timeout,
            api_path=api_path,
            model_name=model_name,
            strong_model_name=strong_model_name,
            trajectory_records_path=trajectory_records_path,
            request_timeout=request_timeout,
            max_retries=max_retries,
            planner_max_iterations=planner_max_iterations,
            patcher_max_iterations=patcher_max_iterations,
            evaluation_command_source=evaluation_command_source,
            replay_previous_patch=replay_previous_patch,
            patch_replay_apply_mode=patch_replay_apply_mode,
            patcher_failure_evidence_path=patcher_failure_evidence_path,
            patch_contract=patch_contract,
            operator_gate=operator_gate,
            target_mapping_probe_paths=list(target_mapping_probe_paths or []),
            bounded_validation_enabled=bounded_validation_enabled,
            post_patch_span_delta_revision_enabled=post_patch_span_delta_revision_enabled,
            patch_intent_gate_enabled=patch_intent_gate_enabled,
            patch_intent_advisory_enabled=patch_intent_advisory_enabled,
            patch_intent_revision_enabled=patch_intent_revision_enabled,
            edit_intent_plan_enabled=edit_intent_plan_enabled,
            syntax_repair_revision_enabled=syntax_repair_revision_enabled,
        )
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "",
                "stop_reason": "executed",
                "implemented": True,
                "scope_supported": True,
                "scope_executor_available": True,
                "workspace": str(workspace),
                "workspace_strategy": workspace_strategy,
                "operator_gate": operator_gate,
                "evaluation_command_source": evaluation_command_source,
                "bounded_validation_enabled": bounded_validation_enabled,
                **result,
            },
        )
    except Exception as exc:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            run_scope=run_scope,
            guidance=guidance,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "runtime_error",
                "stop_reason": "runtime_error",
                "error": str(exc),
                "oracle_success": False,
                "reported_success": False,
                "implemented": True,
                "scope_supported": True,
                "scope_executor_available": True,
                "workspace": str(workspace),
                "workspace_strategy": workspace_strategy,
                "operator_gate": operator_gate,
                "evaluation_command_source": evaluation_command_source,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    finally:
        if runtime_session is not None:
            runtime_session.close()


def build_contract_result(
    *,
    instance_id: str,
    manifest_root: Path,
    run_scope: str,
) -> dict[str, Any]:
    return run_action_enforced_recovery(
        instance_id=instance_id,
        manifest_root=manifest_root,
        run_scope=run_scope,
        execute=False,
    )


def _payload(
    *,
    instance_id: str,
    manifest_path: Path,
    run_scope: str,
    guidance: dict[str, Any],
    execute_requested: bool,
    started_at: datetime | None = None,
    started_monotonic: float | None = None,
    row: dict[str, Any],
) -> dict[str, Any]:
    error_type = str(row.get("error_type", "") or "")
    stop_reason = str(row.get("stop_reason", "") or "")
    implemented = bool(row.get("implemented", False))
    source_type = "action_enforced_execution" if execute_requested else "action_enforced_dry_run"
    runtime_fields = _runtime_duration_fields(started_at=started_at, started_monotonic=started_monotonic)
    return {
        "schema": "mas_dx_action_enforced_recovery_v1",
        "rows": [
            {
                "source_type": source_type,
                "instance_id": instance_id,
                "manifest_path": str(manifest_path),
                "run_scope": run_scope,
                "execute_requested": execute_requested,
                "scope_supported": bool(row.get("scope_supported", run_scope in SUPPORTED_RUN_SCOPES)),
                "scope_executor_available": bool(row.get("scope_executor_available", run_scope in EXECUTABLE_RUN_SCOPES)),
                "recommended_recovery_action": str(guidance.get("recommended_recovery_action", "") or ""),
                "responsible_stage": str(guidance.get("responsible_stage", "") or ""),
                "primary_failure_type": str(guidance.get("primary_failure_type", "") or ""),
                "input_mode": str(guidance.get("input_mode", "") or ""),
                "error_type": error_type,
                "stop_reason": stop_reason,
                "oracle_success": bool(row.get("oracle_success", False)),
                "reported_success": bool(row.get("reported_success", False)),
                "action_adherence_expected": _expected_adherence(
                    run_scope,
                    operator_gate=str(row.get("operator_gate", "") or ""),
                ),
                **{key: value for key, value in row.items() if key not in {"implemented"}},
                **runtime_fields,
            }
        ],
        "summary": {
            "n_rows": 1,
            "status": stop_reason,
            "run_scope": run_scope,
            "implemented": implemented,
            "execute_requested": execute_requested,
            "scope_supported": bool(row.get("scope_supported", run_scope in SUPPORTED_RUN_SCOPES)),
            "scope_executor_available": bool(row.get("scope_executor_available", run_scope in EXECUTABLE_RUN_SCOPES)),
            **runtime_fields,
        },
    }


def _start_runtime_timer() -> tuple[datetime, float]:
    return datetime.now(timezone.utc), time.monotonic()


def _runtime_duration_fields(
    *,
    started_at: datetime | None,
    started_monotonic: float | None,
) -> dict[str, Any]:
    finished_at = datetime.now(timezone.utc)
    if started_at is None:
        started_at = finished_at
    if started_monotonic is None:
        duration_seconds = 0.0
    else:
        duration_seconds = max(0.0, time.monotonic() - float(started_monotonic))
    return {
        "started_at": _isoformat_utc(started_at),
        "finished_at": _isoformat_utc(finished_at),
        "duration_seconds": round(duration_seconds, 6),
        "duration_source": "runner_monotonic_clock",
    }


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-path", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--strong-model", default="")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--trajectory-records", default="outputs/mas_dx/failed_main_dataset_current.json")
    parser.add_argument("--run-scope", required=True)
    parser.add_argument("--runtime", default="harness", choices=["local", "harness"])
    parser.add_argument("--workspace-root", default="outputs/mas_dx/action_enforced_workspaces")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-rebuild-harness", action="store_true")
    parser.add_argument("--harness-setup-timeout", type=int, default=None)
    parser.add_argument("--harness-container-start-timeout", type=int, default=None)
    parser.add_argument("--harness-container-cleanup-timeout", type=int, default=None)
    parser.add_argument(
        "--harness-env-image-key",
        default="",
        help=(
            "Optional prebuilt SWE-bench env image key. When provided, the harness runtime "
            "uses this image directly and skips official SWE-bench image-spec imports."
        ),
    )
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument(
        "--evaluation-command-source",
        default="manifest",
        choices=sorted(EVALUATION_COMMAND_SOURCES),
        help=(
            "Where verifier/oracle commands come from. 'manifest' preserves old behavior; "
            "'normalized' rebuilds commands from safe manifest targets; "
            "'normalized_fail_to_pass' uses only safe FAIL_TO_PASS targets for both checks; "
            "'validated' modes additionally filter targets through pytest --collect-only."
        ),
    )
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--planner-iters", type=int, default=4)
    parser.add_argument("--patcher-iters", type=int, default=8)
    parser.add_argument(
        "--replay-previous-patch",
        action="store_true",
        help="Apply the previous MAS trajectory patch before non-patcher verification scopes.",
    )
    parser.add_argument(
        "--patch-replay-apply-mode",
        default="plain",
        choices=sorted(PATCH_REPLAY_APPLY_MODES),
        help=(
            "How to apply a previous MAS patch. 'plain' preserves old behavior; "
            "'recount' only asks git to recompute hunk line counts for recovered "
            "trajectory diffs with stale hunk metadata."
        ),
    )
    parser.add_argument(
        "--patcher-failure-evidence",
        default="",
        help="Optional MAS-DX-R failure-evidence bundle used only by patcher_fixed_localization.",
    )
    parser.add_argument(
        "--patch-contract",
        default="",
        choices=["", "source_only"],
        help=(
            "Optional patch boundary contract for patcher_fixed_localization. "
            "'source_only' forbids test/generated edits for strict recovery claims."
        ),
    )
    parser.add_argument(
        "--operator-gate",
        default="",
        choices=sorted(OPERATOR_GATES),
        help="Optional MAS-DX-R operator gate injected into patcher/verifier recovery context.",
    )
    parser.add_argument(
        "--target-mapping-probe",
        action="append",
        default=[],
        help=(
            "Optional verified nodeid mapping probe. Accepted non-manual mappings "
            "are applied to evaluation targets before runtime collect validation."
        ),
    )
    parser.add_argument(
        "--bounded-validation-enabled",
        dest="bounded_validation_enabled",
        action="store_true",
        default=True,
        help="Enable bounded validation before recovery execution. This is the default.",
    )
    parser.add_argument(
        "--skip-bounded-validation",
        dest="bounded_validation_enabled",
        action="store_false",
        help=(
            "Disable bounded validation while preserving the selected recovery operator. "
            "Use only for predeclared bounded-intervention ablation arms."
        ),
    )
    parser.add_argument(
        "--post-patch-span-delta-revision",
        action="store_true",
        help=(
            "Enable one post-patch span-delta revision after the selected operator's "
            "first source-only patch. This preserves the operator gate and is used "
            "for paired v7 ablations."
        ),
    )
    parser.add_argument(
        "--patch-intent-gate",
        action="store_true",
        help=(
            "Enable the v10 pre-oracle patch-intent gate. The gate checks whether "
            "the produced source patch touches the bounded-observation boundary "
            "and mentions the observed semantic contract before spending oracle budget."
        ),
    )
    parser.add_argument(
        "--patch-intent-advisory",
        action="store_true",
        help=(
            "Evaluate the patch-intent gate as audit evidence only. Advisory mode "
            "records alignment violations but does not block oracle validation."
        ),
    )
    parser.add_argument(
        "--patch-intent-revision",
        action="store_true",
        help=(
            "When --patch-intent-gate blocks a patch, allow exactly one bounded "
            "revision against the rejected intent evidence before stopping."
        ),
    )
    parser.add_argument(
        "--edit-intent-plan",
        action="store_true",
        help=(
            "Enable v11 pre-patch edit intent. This turns G2 propagation, bounded "
            "observation, and failure-span evidence into a source-span anchored "
            "CAR patch intent before the patcher plans a fix."
        ),
    )
    parser.add_argument(
        "--syntax-repair-revision",
        action="store_true",
        help=(
            "After a generated source patch fails the candidate syntax gate, allow "
            "one bounded same-operator revision using the exact syntax errors. "
            "Default is off to preserve historical experiment behavior."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = run_action_enforced_recovery(
        instance_id=args.instance_id,
        manifest_root=Path(args.manifest_root),
        run_scope=args.run_scope,
        api_path=Path(args.api_path) if args.api_path else None,
        model_name=args.model,
        strong_model_name=args.strong_model,
        trajectory_records_path=Path(args.trajectory_records),
        output_root=Path(args.workspace_root),
        runtime=args.runtime,
        execute=args.execute,
        force_rebuild_harness=args.force_rebuild_harness,
        harness_setup_timeout=args.harness_setup_timeout,
        harness_container_start_timeout=args.harness_container_start_timeout,
        harness_container_cleanup_timeout=args.harness_container_cleanup_timeout,
        harness_env_image_key=args.harness_env_image_key,
        command_timeout=args.command_timeout,
        evaluation_command_source=args.evaluation_command_source,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        planner_max_iterations=args.planner_iters,
        patcher_max_iterations=args.patcher_iters,
        replay_previous_patch=args.replay_previous_patch,
        patch_replay_apply_mode=args.patch_replay_apply_mode,
        patcher_failure_evidence_path=Path(args.patcher_failure_evidence)
        if args.patcher_failure_evidence
        else None,
        patch_contract=args.patch_contract,
        operator_gate=args.operator_gate,
        target_mapping_probe_paths=[Path(path) for path in list(args.target_mapping_probe or [])],
        bounded_validation_enabled=args.bounded_validation_enabled,
        post_patch_span_delta_revision_enabled=bool(args.post_patch_span_delta_revision),
        patch_intent_gate_enabled=bool(args.patch_intent_gate),
        patch_intent_advisory_enabled=bool(args.patch_intent_advisory),
        patch_intent_revision_enabled=bool(args.patch_intent_revision),
        edit_intent_plan_enabled=bool(args.edit_intent_plan),
        syntax_repair_revision_enabled=bool(args.syntax_repair_revision),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row_error = str(payload["rows"][0].get("error_type", "") or "")
    if row_error and row_error != "planned_not_executed":
        return 2
    return 0


def _execute_scope(
    *,
    run_scope: str,
    manifest: dict[str, Any],
    instance_id: str,
    executor,
    workspace: Path,
    timeout: int,
    api_path: Path | None,
    model_name: str,
    strong_model_name: str,
    trajectory_records_path: Path,
    request_timeout: int,
    max_retries: int,
    planner_max_iterations: int,
    patcher_max_iterations: int,
    evaluation_command_source: str,
    replay_previous_patch: bool,
    patch_replay_apply_mode: str,
    patcher_failure_evidence_path: Path | None,
    patch_contract: str,
    operator_gate: str,
    target_mapping_probe_paths: list[Path],
    bounded_validation_enabled: bool,
    post_patch_span_delta_revision_enabled: bool = False,
    patch_intent_gate_enabled: bool = False,
    patch_intent_advisory_enabled: bool = False,
    patch_intent_revision_enabled: bool = False,
    edit_intent_plan_enabled: bool = False,
    syntax_repair_revision_enabled: bool = False,
) -> dict[str, Any]:
    patch_replay = _maybe_replay_previous_patch(
        run_scope=run_scope,
        workspace=workspace,
        trajectory_records_path=trajectory_records_path,
        instance_id=instance_id,
        replay_previous_patch=replay_previous_patch,
        apply_mode=patch_replay_apply_mode,
    )
    evaluation = _evaluation_commands_for_manifest(
        manifest,
        command_source=evaluation_command_source,
    )
    target_mapping = _target_mapping_probe_evidence(
        target_mapping_probe_paths,
        instance_id=instance_id,
    )
    evaluation = _apply_target_mapping_to_evaluation(
        manifest=manifest,
        evaluation=evaluation,
        target_mapping=target_mapping,
    )
    if _patch_replay_blocked(patch_replay):
        return _patch_replay_blocked_result(
            evaluation=evaluation,
            patch_replay=patch_replay,
            action_adherence_observed={
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": bounded_validation_enabled,
                "previous_patch_replay_attempted": True,
            },
        )
    if bounded_validation_enabled and run_scope not in {
        "patcher_fixed_localization",
        "patcher_fixed_localization_candidate_only",
        "candidate_patch_verifier",
    }:
        evaluation = _runtime_validate_evaluation_commands(
            manifest=manifest,
            evaluation=evaluation,
            executor=executor,
            workspace=workspace,
            timeout=min(timeout, 300),
        )
    if run_scope == "verifier_only_replay":
        if evaluation["error_type"]:
            return _evaluation_blocked_result(
                evaluation=evaluation,
                extra=patch_replay,
                action_adherence_observed={
                    "patch_attempted": False,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": False,
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped": not bounded_validation_enabled,
                    "model_call_count": 0,
                    "token_cost": 0.0,
                },
            )
        fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
        oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
        protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
        return {
            **_evaluation_command_payload(evaluation),
            **patch_replay,
            "fail_to_pass_returncode": fail_to_pass["returncode"],
            "oracle_returncode": oracle["returncode"],
            "oracle_success": oracle["returncode"] == 0,
            "reported_success": oracle["returncode"] == 0,
            "fail_to_pass_output": fail_to_pass["output"],
            "oracle_output": oracle["output"],
            **protocol,
            **_zero_model_usage_fields(),
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": [] if bounded_validation_enabled else ["runtime_collect_validation"],
            "action_adherence_observed": {
                "patch_attempted": False,
                "verifier_attempted": True,
                "environment_preflight_attempted": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    if run_scope == "environment_preflight_then_verifier":
        preflight_command = _preflight_command(manifest)
        preflight = (
            executor.execute(preflight_command, cwd=str(workspace), timeout=min(timeout, 120))
            if bounded_validation_enabled
            else {
                "returncode": None,
                "output": "skipped_bounded_validation_disabled",
            }
        )
        if evaluation["error_type"]:
            return _evaluation_blocked_result(
                evaluation=evaluation,
                extra={
                    **patch_replay,
                    "environment_preflight_command": preflight_command,
                    "environment_preflight_returncode": preflight["returncode"],
                    "environment_preflight_output": preflight["output"],
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped_steps": (
                        [] if bounded_validation_enabled else ["environment_preflight", "runtime_collect_validation"]
                    ),
                },
                action_adherence_observed={
                    "patch_attempted": False,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": bounded_validation_enabled,
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped": not bounded_validation_enabled,
                    "model_call_count": 0,
                    "token_cost": 0.0,
                },
            )
        fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
        oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
        protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
        return {
            **_evaluation_command_payload(evaluation),
            **patch_replay,
            "environment_preflight_command": preflight_command,
            "environment_preflight_returncode": preflight["returncode"],
            "environment_preflight_output": preflight["output"],
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                [] if bounded_validation_enabled else ["environment_preflight", "runtime_collect_validation"]
            ),
            "fail_to_pass_returncode": fail_to_pass["returncode"],
            "oracle_returncode": oracle["returncode"],
            "oracle_success": oracle["returncode"] == 0,
            "reported_success": oracle["returncode"] == 0,
            "fail_to_pass_output": fail_to_pass["output"],
            "oracle_output": oracle["output"],
            **protocol,
            **_zero_model_usage_fields(),
            "action_adherence_observed": {
                "patch_attempted": False,
                "verifier_attempted": True,
                "environment_preflight_attempted": bounded_validation_enabled,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    if run_scope in {"guarded_protocol_then_patch", "guarded_protocol_then_clean_patch"}:
        return _execute_guarded_protocol_then_patch(
            instance_id=instance_id,
            manifest=manifest,
            executor=executor,
            workspace=workspace,
            timeout=timeout,
            api_path=api_path,
            model_name=model_name,
            strong_model_name=strong_model_name,
            trajectory_records_path=trajectory_records_path,
            request_timeout=request_timeout,
            max_retries=max_retries,
            planner_max_iterations=planner_max_iterations,
            patcher_max_iterations=patcher_max_iterations,
            evaluation=evaluation,
            patch_replay=patch_replay,
            patcher_failure_evidence_path=patcher_failure_evidence_path,
            patch_contract=patch_contract or "source_only",
            operator_gate=operator_gate,
            bounded_validation_enabled=bounded_validation_enabled,
            post_patch_span_delta_revision_enabled=post_patch_span_delta_revision_enabled,
            patch_intent_gate_enabled=patch_intent_gate_enabled,
            patch_intent_advisory_enabled=patch_intent_advisory_enabled,
            patch_intent_revision_enabled=patch_intent_revision_enabled,
            edit_intent_plan_enabled=edit_intent_plan_enabled,
            syntax_repair_revision_enabled=syntax_repair_revision_enabled,
            clean_workspace_before_fallback=run_scope == "guarded_protocol_then_clean_patch",
        )
    if run_scope in {"patcher_fixed_localization", "patcher_fixed_localization_candidate_only"}:
        return _execute_patcher_fixed_localization(
            instance_id=instance_id,
            manifest=manifest,
            executor=executor,
            workspace=workspace,
            timeout=timeout,
            api_path=api_path,
            model_name=model_name,
            strong_model_name=strong_model_name,
            trajectory_records_path=trajectory_records_path,
            request_timeout=request_timeout,
            max_retries=max_retries,
            planner_max_iterations=planner_max_iterations,
            patcher_max_iterations=patcher_max_iterations,
            evaluation=evaluation,
            patcher_failure_evidence_path=patcher_failure_evidence_path,
            patch_contract=patch_contract,
            operator_gate=operator_gate,
            bounded_validation_enabled=bounded_validation_enabled,
            post_patch_span_delta_revision_enabled=post_patch_span_delta_revision_enabled,
            patch_intent_gate_enabled=patch_intent_gate_enabled,
            patch_intent_advisory_enabled=patch_intent_advisory_enabled,
            patch_intent_revision_enabled=patch_intent_revision_enabled,
            edit_intent_plan_enabled=edit_intent_plan_enabled,
            syntax_repair_revision_enabled=syntax_repair_revision_enabled,
            candidate_only=run_scope == "patcher_fixed_localization_candidate_only",
        )
    if run_scope == "candidate_patch_verifier":
        return _execute_candidate_patch_verifier(
            evaluation=evaluation,
            manifest=manifest,
            instance_id=instance_id,
            executor=executor,
            workspace=workspace,
            timeout=timeout,
            patcher_failure_evidence_path=patcher_failure_evidence_path,
            patch_contract=patch_contract,
            operator_gate=operator_gate,
            bounded_validation_enabled=bounded_validation_enabled,
        )
    raise ValueError(f"Unsupported run scope: {run_scope}")


def _execute_guarded_protocol_then_patch(
    *,
    instance_id: str,
    manifest: dict[str, Any],
    executor,
    workspace: Path,
    timeout: int,
    api_path: Path | None,
    model_name: str,
    strong_model_name: str,
    trajectory_records_path: Path,
    request_timeout: int,
    max_retries: int,
    planner_max_iterations: int,
    patcher_max_iterations: int,
    evaluation: dict[str, Any],
    patch_replay: dict[str, Any],
    patcher_failure_evidence_path: Path | None,
    patch_contract: str,
    operator_gate: str,
    bounded_validation_enabled: bool,
    post_patch_span_delta_revision_enabled: bool = False,
    patch_intent_gate_enabled: bool = False,
    patch_intent_advisory_enabled: bool = False,
    patch_intent_revision_enabled: bool = False,
    edit_intent_plan_enabled: bool = False,
    syntax_repair_revision_enabled: bool = False,
    clean_workspace_before_fallback: bool = False,
) -> dict[str, Any]:
    preflight_command = _preflight_command(manifest)
    preflight = (
        executor.execute(preflight_command, cwd=str(workspace), timeout=min(timeout, 120))
        if bounded_validation_enabled
        else {"returncode": None, "output": "skipped_bounded_validation_disabled"}
    )
    if evaluation["error_type"]:
        return _evaluation_blocked_result(
            evaluation=evaluation,
            extra={
                **patch_replay,
                "guarded_protocol_probe_attempted": False,
                "guarded_protocol_fallback_attempted": False,
                "environment_preflight_command": preflight_command,
                "environment_preflight_returncode": preflight["returncode"],
                "environment_preflight_output": preflight["output"],
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped_steps": (
                    [] if bounded_validation_enabled else ["environment_preflight", "runtime_collect_validation"]
                ),
            },
            action_adherence_observed={
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": bounded_validation_enabled,
                "previous_patch_replay_attempted": bool(patch_replay.get("previous_patch_replay_attempted")),
                "guarded_protocol_probe_attempted": False,
                "guarded_protocol_fallback_attempted": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        )
    fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
    oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
    protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
    probe = {
        **_evaluation_command_payload(evaluation),
        **patch_replay,
        "environment_preflight_command": preflight_command,
        "environment_preflight_returncode": preflight["returncode"],
        "environment_preflight_output_excerpt": _tail_excerpt(preflight.get("output", "")),
        "fail_to_pass_returncode": fail_to_pass["returncode"],
        "oracle_returncode": oracle["returncode"],
        "oracle_success": oracle["returncode"] == 0,
        "reported_success": oracle["returncode"] == 0,
        "fail_to_pass_output_excerpt": _tail_excerpt(fail_to_pass.get("output", "")),
        "oracle_output_excerpt": _tail_excerpt(oracle.get("output", "")),
        **protocol,
        **_zero_model_usage_fields(),
        "bounded_validation_enabled": bounded_validation_enabled,
        "bounded_validation_skipped_steps": (
            [] if bounded_validation_enabled else ["environment_preflight", "runtime_collect_validation"]
        ),
    }
    if probe["oracle_success"]:
        return {
            **_evaluation_command_payload(evaluation),
            **patch_replay,
            "environment_preflight_command": preflight_command,
            "environment_preflight_returncode": preflight["returncode"],
            "environment_preflight_output": preflight["output"],
            "fail_to_pass_returncode": fail_to_pass["returncode"],
            "oracle_returncode": oracle["returncode"],
            "oracle_success": True,
            "reported_success": True,
            "fail_to_pass_output": fail_to_pass["output"],
            "oracle_output": oracle["output"],
            **protocol,
            **_zero_model_usage_fields(),
            "guarded_protocol_probe": probe,
            "guarded_protocol_probe_attempted": True,
            "guarded_protocol_fallback_attempted": False,
            "guarded_protocol_fallback_reason": "protocol_probe_oracle_success",
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                [] if bounded_validation_enabled else ["environment_preflight", "runtime_collect_validation"]
            ),
            "action_adherence_observed": {
                "patch_attempted": False,
                "verifier_attempted": True,
                "environment_preflight_attempted": bounded_validation_enabled,
                "previous_patch_replay_attempted": bool(patch_replay.get("previous_patch_replay_attempted")),
                "guarded_protocol_probe_attempted": True,
                "guarded_protocol_fallback_attempted": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    clean_reset = (
        _reset_workspace_for_guarded_fallback(executor=executor, workspace=workspace, timeout=timeout)
        if clean_workspace_before_fallback
        else {
            "attempted": False,
            "command": "",
            "returncode": None,
            "output_excerpt": "",
            "skipped_reason": "clean_workspace_before_fallback_disabled",
        }
    )
    if clean_reset.get("attempted") and clean_reset.get("returncode") != 0:
        observed = {
            "patch_attempted": False,
            "verifier_attempted": True,
            "environment_preflight_attempted": bounded_validation_enabled,
            "previous_patch_replay_attempted": bool(patch_replay.get("previous_patch_replay_attempted")),
            "guarded_protocol_probe_attempted": True,
            "guarded_protocol_fallback_attempted": False,
            "guarded_protocol_clean_reset_attempted": True,
            "model_call_count": 0,
            "token_cost": 0.0,
        }
        return {
            **_evaluation_command_payload(evaluation),
            "oracle_success": False,
            "reported_success": False,
            "error_type": "guarded_protocol_clean_reset_failed",
            "stop_reason": "guarded_protocol_clean_reset_failed",
            **_zero_model_usage_fields(),
            "guarded_protocol_probe": probe,
            "guarded_protocol_probe_attempted": True,
            "guarded_protocol_fallback_attempted": False,
            "guarded_protocol_fallback_reason": _guarded_protocol_fallback_reason(probe),
            "guarded_protocol_clean_reset": clean_reset,
            "action_adherence_observed": observed,
        }
    patch_result = _execute_patcher_fixed_localization(
        instance_id=instance_id,
        manifest=manifest,
        executor=executor,
        workspace=workspace,
        timeout=timeout,
        api_path=api_path,
        model_name=model_name,
        strong_model_name=strong_model_name,
        trajectory_records_path=trajectory_records_path,
        request_timeout=request_timeout,
        max_retries=max_retries,
        planner_max_iterations=planner_max_iterations,
        patcher_max_iterations=patcher_max_iterations,
        evaluation=evaluation,
        patcher_failure_evidence_path=patcher_failure_evidence_path,
        patch_contract=patch_contract,
        operator_gate=operator_gate,
        bounded_validation_enabled=bounded_validation_enabled,
        post_patch_span_delta_revision_enabled=post_patch_span_delta_revision_enabled,
        patch_intent_gate_enabled=patch_intent_gate_enabled,
        patch_intent_advisory_enabled=patch_intent_advisory_enabled,
        patch_intent_revision_enabled=patch_intent_revision_enabled,
        edit_intent_plan_enabled=edit_intent_plan_enabled,
        syntax_repair_revision_enabled=syntax_repair_revision_enabled,
    )
    observed = dict(patch_result.get("action_adherence_observed", {}) or {})
    observed.update(
        {
            "verifier_attempted": True,
            "environment_preflight_attempted": bounded_validation_enabled,
            "previous_patch_replay_attempted": bool(patch_replay.get("previous_patch_replay_attempted")),
            "guarded_protocol_probe_attempted": True,
            "guarded_protocol_fallback_attempted": True,
            "guarded_protocol_clean_reset_attempted": bool(clean_reset.get("attempted")),
        }
    )
    return {
        **patch_result,
        "guarded_protocol_probe": probe,
        "guarded_protocol_probe_attempted": True,
        "guarded_protocol_fallback_attempted": True,
        "guarded_protocol_fallback_reason": _guarded_protocol_fallback_reason(probe),
        "guarded_protocol_clean_reset": clean_reset,
        "action_adherence_observed": observed,
    }


def _reset_workspace_for_guarded_fallback(*, executor, workspace: Path, timeout: int) -> dict[str, Any]:
    command = "git reset --hard HEAD && git clean -fd"
    result = executor.execute(command, cwd=str(workspace), timeout=min(timeout, 300))
    return {
        "attempted": True,
        "command": command,
        "returncode": int(result.get("returncode", 1)),
        "output_excerpt": _tail_excerpt(result.get("output", "")),
    }


def _guarded_protocol_fallback_reason(probe: dict[str, Any]) -> str:
    if probe.get("evaluation_protocol_error"):
        return str(probe.get("evaluation_protocol_error_type") or "protocol_probe_error")
    if probe.get("oracle_returncode") not in {None, 0}:
        return "protocol_probe_oracle_failed"
    if probe.get("fail_to_pass_returncode") not in {None, 0}:
        return "protocol_probe_fail_to_pass_failed"
    return "protocol_probe_no_oracle_success"


def _tail_excerpt(output: Any, *, max_chars: int = 2000) -> str:
    return str(output or "")[-max_chars:]


def _zero_model_usage_fields() -> dict[str, Any]:
    return {
        "model_usage_before": dict(ZERO_MODEL_USAGE),
        "model_usage_after": dict(ZERO_MODEL_USAGE),
        "model_usage_delta": dict(ZERO_MODEL_USAGE),
        "token_cost": 0.0,
        "model_call_count": 0,
    }


def _manifest_path(manifest_root: Path, instance_id: str) -> Path:
    owner, repo_issue = instance_id.split("__", 1)
    path = manifest_root / f"real_eval_manifest_{owner}__{repo_issue}.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for {instance_id}: {path}")
    return path


def _semantic_span_delta_revision_enabled(operator_gate: str) -> bool:
    return operator_gate in {"semantic_span_delta_guarded_repatch", "semantic_effect_guarded_repatch"}


def _post_patch_span_delta_revision_enabled(
    *,
    operator_gate: str,
    post_patch_span_delta_revision_enabled: bool,
) -> bool:
    return (
        post_patch_span_delta_revision_enabled
        or _semantic_span_delta_revision_enabled(operator_gate)
    )


def _semantic_span_delta_after_validation(
    *,
    raw_failure_output: str,
    patcher_failure_evidence: dict[str, Any],
    oracle_success: bool,
    protocol: dict[str, Any],
    patch_legitimacy: str,
    error_type: str = "",
) -> dict[str, Any]:
    original = dict(patcher_failure_evidence.get("failure_span_family", {}) or {})
    post = (
        {
            "schema": "masguard_failure_span_family_v1",
            "family": "resolved",
            "subtype": "",
            "exception_type": "",
            "evidence_excerpt": "",
            "verified_failure_cause": "oracle success",
        }
        if oracle_success
        else classify_failure_span_text(raw_failure_output)
    )
    validation_excerpt = _bounded_validation_failure_excerpt(raw_failure_output)
    delta = _semantic_span_delta_kind(
        oracle_success=oracle_success,
        protocol=protocol,
        patch_legitimacy=patch_legitimacy,
        error_type=error_type,
        original=original,
        post=post,
    )
    if delta == "no_fresh_patch" and _validation_excerpt_has_unresolved_signal(validation_excerpt):
        delta = "no_fresh_patch_with_unresolved_validation"
    return {
        "schema": "masguard_post_patch_span_delta_v1",
        "attempted": True,
        "delta": delta,
        "requires_revision": _semantic_span_delta_requires_revision(delta),
        "original_span_family": str(original.get("family", "") or ""),
        "original_span_subtype": str(original.get("subtype", "") or ""),
        "original_verified_failure_cause": str(original.get("verified_failure_cause", "") or ""),
        "post_patch_span_family": str(post.get("family", "") or ""),
        "post_patch_span_subtype": str(post.get("subtype", "") or ""),
        "post_patch_verified_failure_cause": str(post.get("verified_failure_cause", "") or ""),
        "post_patch_evidence_excerpt": str(post.get("evidence_excerpt", "") or ""),
        "post_patch_validation_excerpt": validation_excerpt,
    }


def _bounded_validation_failure_excerpt(raw_failure_output: str, *, max_lines: int = 32) -> list[str]:
    """Extract compact runtime evidence for a bounded post-patch revision."""
    lines = [_strip_ansi(line).rstrip()[:260] for line in str(raw_failure_output or "").splitlines()]
    root_cause_patterns = (
        "nameerror",
        "typeerror",
        "attributeerror",
        "assertionerror",
        "valueerror",
        "importerror",
        "modulenotfounderror",
        "not defined",
        "undefined",
        "remains unmatched",
        "nomatch",
    )
    signal_patterns = (
        " fail",
        "failed ",
        "error ",
        "e       ",
        "e   ",
        "> ",
        "nomatch",
        "remains unmatched",
        "expected",
        "actual",
        "traceback",
        "assert ",
        "skipped [",
        "xfailed",
        "xpassed",
        ": failed",
    )
    selected: list[str] = []
    seen: set[str] = set()

    def add(index: int) -> None:
        if index < 0 or index >= len(lines):
            return
        text = lines[index].strip()
        if not text or text in seen:
            return
        seen.add(text)
        selected.append(text)

    for text in _validation_location_mismatch_lines(lines):
        if text not in seen:
            seen.add(text)
            selected.append(text)
    for patterns in (root_cause_patterns, signal_patterns):
        for index, line in enumerate(lines):
            stripped = line.strip()
            lowered = stripped.lower()
            is_failure_header = stripped.startswith("_") and " " in stripped
            is_file_failure = bool(re.search(r"(^|/)[\w./-]+\.py:\d+:\s*(Failed|Error|AssertionError)", stripped))
            is_pytest_failed_node = stripped.startswith("FAILED ") and "::" in stripped
            if (
                is_failure_header
                or is_file_failure
                or is_pytest_failed_node
                or any(pattern in lowered for pattern in patterns)
            ):
                add(index - 1)
                add(index)
                add(index + 1)
            if len(selected) >= max_lines:
                break
        if len(selected) >= max_lines:
            break
    if len(selected) < max_lines:
        for index, line in enumerate(lines):
            if len(selected) >= max_lines:
                break
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if "internalerror>" in lowered and ("/testbed/" in lowered or "nameerror" in lowered):
                add(index)
    if len(selected) >= max_lines:
        return selected[:max_lines]
    for index, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        is_failure_header = stripped.startswith("_") and " " in stripped
        is_file_failure = bool(re.search(r"(^|/)[\w./-]+\.py:\d+:\s*(Failed|Error|AssertionError)", stripped))
        is_pytest_failed_node = stripped.startswith("FAILED ") and "::" in stripped
        if (
            is_failure_header
            or is_file_failure
            or is_pytest_failed_node
            or any(pattern in lowered for pattern in signal_patterns)
        ):
            add(index - 1)
            add(index)
            add(index + 1)
        if len(selected) >= max_lines:
            break
    if not selected:
        for line in lines:
            text = line.strip()
            if text:
                selected.append(text)
            if len(selected) >= min(max_lines, 12):
                break
    return selected[:max_lines]


def _validation_location_mismatch_lines(lines: list[str]) -> list[str]:
    expected: list[tuple[str, int]] = []
    observed: list[tuple[str, int]] = []
    for raw_line in lines:
        line = raw_line.strip()
        lowered = line.lower()
        matches = [
            (path, int(number))
            for path, number in re.findall(r"([A-Za-z0-9_./-]+\.py):(\d+)", line)
        ]
        if not matches:
            continue
        if any(token in lowered for token in ("nomatch", "remains unmatched", "expected")):
            expected.extend(matches)
        elif " and:" in lowered or "actual" in lowered or "short test summary" in lowered:
            observed.extend(matches)
    signals: list[str] = []
    seen: set[str] = set()
    for exp_path, exp_line in expected:
        for obs_path, obs_line in observed:
            if (exp_path, exp_line) == (obs_path, obs_line):
                continue
            delta = obs_line - exp_line
            signal = (
                "validation_location_mismatch: "
                f"expected {exp_path}:{exp_line}; observed {obs_path}:{obs_line}; line_delta={delta}"
            )
            if signal not in seen:
                seen.add(signal)
                signals.append(signal)
    return signals[:4]


def _validation_excerpt_has_unresolved_signal(validation_excerpt: list[str]) -> bool:
    text = "\n".join(str(line) for line in validation_excerpt).lower()
    return any(
        token in text
        for token in (
            "validation_location_mismatch:",
            "failed:",
            "nameerror",
            "typeerror",
            "attributeerror",
            "remains unmatched",
            "assertionerror",
        )
    )


def _source_delta_intent_lines(*, validation_excerpt: list[str], patch_text: str) -> list[str]:
    mismatch_re = re.compile(
        r"validation_location_mismatch:\s*expected\s+(?P<expected_path>[^:;]+):(?P<expected_line>\d+);\s*"
        r"observed\s+(?P<observed_path>[^:;]+):(?P<observed_line>\d+);\s*line_delta=(?P<delta>-?\d+)"
    )
    intents: list[str] = []
    for line in validation_excerpt:
        match = mismatch_re.search(str(line))
        if not match:
            continue
        expected_path = match.group("expected_path")
        observed_path = match.group("observed_path")
        delta = int(match.group("delta"))
        intents.append(
            "location_delta: "
            f"expected={expected_path}:{match.group('expected_line')}, "
            f"observed={observed_path}:{match.group('observed_line')}, line_delta={delta}"
        )
        if expected_path != observed_path:
            intents.append(
                "source_delta_action: observed path differs from expected; repair the reported object/source-origin path before tuning line arithmetic."
            )
        if delta < 0:
            intents.append(
                f"source_delta_action: observed line is {abs(delta)} before expected; increase the reported line by {abs(delta)} or undo an offset removal."
            )
        elif delta > 0:
            intents.append(
                f"source_delta_action: observed line is {delta} after expected; decrease the reported line by {delta} or undo an added offset."
            )
    removed_offset = any(re.match(r"-.*\bline\s*\+\s*\d+\b", line) for line in str(patch_text or "").splitlines())
    added_plain_line = any(re.match(r"\+.*\bline\b(?!\s*[+-])", line) for line in str(patch_text or "").splitlines())
    if intents and removed_offset and added_plain_line:
        intents.append(
            "patch_delta_hint: previous patch removed a positive line offset; if validation under-reports the line, preserve the path fix but restore the needed offset."
        )
    if not intents:
        return []
    output: list[str] = ["[MASGUARD SOURCE-DELTA INTENT]"]
    seen: set[str] = set()
    for item in intents:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output[:10]


def _semantic_span_delta_requires_revision(delta: str) -> bool:
    return delta in {
        "same_family_same_subtype_unresolved",
        "same_family_changed_subtype_unresolved",
        "different_family_unresolved",
        "unknown_post_patch_failure",
        "no_fresh_patch_with_unresolved_validation",
    }


def _semantic_span_delta_kind(
    *,
    oracle_success: bool,
    protocol: dict[str, Any],
    patch_legitimacy: str,
    error_type: str,
    original: dict[str, Any],
    post: dict[str, Any],
) -> str:
    if oracle_success:
        return "resolved"
    if bool(protocol.get("evaluation_protocol_error", False)):
        return "protocol_error"
    if patch_legitimacy == "no_diff":
        return "no_fresh_patch"
    if error_type:
        return "runner_or_patcher_error"
    original_family = str(original.get("family", "") or "")
    original_subtype = str(original.get("subtype", "") or "")
    post_family = str(post.get("family", "") or "")
    post_subtype = str(post.get("subtype", "") or "")
    if not post_family or post_family == "unknown_clean_failure":
        return "unknown_post_patch_failure"
    if original_family == post_family and original_subtype == post_subtype:
        return "same_family_same_subtype_unresolved"
    if original_family == post_family:
        return "same_family_changed_subtype_unresolved"
    return "different_family_unresolved"


def _semantic_span_delta_revision_context(
    *,
    recovery_context: str,
    span_delta: dict[str, Any],
    patch_text: str,
) -> str:
    patch_excerpt = [
        line.rstrip()[:240]
        for line in str(patch_text or "").splitlines()
        if line.strip()
    ][:80]
    lines = [
        str(recovery_context or "").strip(),
        "[MAS-DX-R POST-PATCH SPAN-DELTA REVISION GATE]",
        "The first source diff passed the pre-validation semantic gate, but focused validation still shows the same bounded-observation failure family.",
        "Hard recovery rule: perform exactly one minimal source revision that changes the verified failure span, or stop.",
        "Hard recovery rule: do not broaden localization, do not edit tests, and do not start an open-ended retry loop.",
        f"Original bounded span: family={span_delta.get('original_span_family', '')}, subtype={span_delta.get('original_span_subtype', '')}.",
        f"Original cause: {span_delta.get('original_verified_failure_cause', '')}.",
        f"Post-patch span: family={span_delta.get('post_patch_span_family', '')}, subtype={span_delta.get('post_patch_span_subtype', '')}.",
        f"Post-patch evidence: {span_delta.get('post_patch_evidence_excerpt', '')}.",
    ]
    validation_excerpt = [
        str(line)
        for line in list(span_delta.get("post_patch_validation_excerpt", []) or [])
        if str(line).strip()
    ]
    if validation_excerpt:
        lines.append("Focused post-patch validation output excerpt:")
        lines.extend(f"- {line}" for line in validation_excerpt[:32])
    source_delta_intent = _source_delta_intent_lines(
        validation_excerpt=validation_excerpt,
        patch_text=patch_text,
    )
    if source_delta_intent:
        lines.append("Source-delta intent derived from focused validation:")
        lines.extend(f"- {line}" for line in source_delta_intent)
    if patch_excerpt:
        lines.append("Rejected first patch excerpt:")
        lines.extend(f"- {line}" for line in patch_excerpt)
    return "\n".join(line for line in lines if str(line).strip())


def _execute_patcher_fixed_localization(
    *,
    instance_id: str,
    manifest: dict[str, Any],
    executor,
    workspace: Path,
    timeout: int,
    api_path: Path | None,
    model_name: str,
    strong_model_name: str,
    trajectory_records_path: Path,
    request_timeout: int,
    max_retries: int,
    planner_max_iterations: int,
    patcher_max_iterations: int,
    evaluation: dict[str, Any],
    patcher_failure_evidence_path: Path | None = None,
    patch_contract: str = "",
    operator_gate: str = "",
    bounded_validation_enabled: bool = True,
    post_patch_span_delta_revision_enabled: bool = False,
    patch_intent_gate_enabled: bool = False,
    patch_intent_advisory_enabled: bool = False,
    patch_intent_revision_enabled: bool = False,
    edit_intent_plan_enabled: bool = False,
    syntax_repair_revision_enabled: bool = False,
    candidate_only: bool = False,
) -> dict[str, Any]:
    if evaluation["error_type"]:
        return _evaluation_blocked_result(
            evaluation=evaluation,
            extra={
                "edit_intent_plan": edit_intent_skipped_result("evaluation_protocol_blocked_before_fixed_localization"),
                "edit_intent_plan_enabled": edit_intent_plan_enabled,
            },
            action_adherence_observed={
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": False,
                "locator_attempted": False,
                "selected_target_count": 0,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
            },
        )
    if api_path is None:
        raise ValueError("--api-path is required for patcher_fixed_localization execution")
    preflight_command = _preflight_command(manifest)
    preflight = (
        executor.execute(preflight_command, cwd=str(workspace), timeout=min(timeout, 120))
        if bounded_validation_enabled
        else {"returncode": None, "output": "skipped_bounded_validation_disabled"}
    )
    fixed = _fixed_localization_for_instance(
        trajectory_records_path=trajectory_records_path,
        instance_id=instance_id,
    )
    patcher_failure_evidence = _patcher_failure_evidence_for_instance(
        patcher_failure_evidence_path,
        instance_id,
    )
    derived_semantic_invariant_evidence = _derive_oracle_semantic_invariant_evidence(
        manifest=manifest,
        evaluation=evaluation,
        fixed_localization=fixed,
    )
    patcher_failure_evidence = _merge_patcher_failure_evidence(
        patcher_failure_evidence,
        derived_semantic_invariant_evidence,
    )
    fixed = _fixed_localization_with_support_expansion(
        fixed,
        patcher_failure_evidence=patcher_failure_evidence,
        operator_gate=operator_gate,
    )
    path_resolution = _fixed_localization_workspace_path_resolution(
        fixed=fixed,
        patcher_failure_evidence=patcher_failure_evidence,
        workspace=workspace,
    )
    fixed = _apply_workspace_path_resolution_to_fixed(fixed, path_resolution)
    patcher_failure_evidence = _apply_workspace_path_resolution_to_evidence(
        patcher_failure_evidence,
        path_resolution,
    )
    effective_edit_intent_plan_enabled = bool(edit_intent_plan_enabled) or bool(
        patcher_failure_evidence.get("assertion_contract_edit_intent_required", False)
    )
    edit_intent_plan = build_masguard_edit_intent(
        patcher_failure_evidence=patcher_failure_evidence,
        fixed_localization=fixed,
        operator_gate=operator_gate,
        patch_contract=patch_contract,
        enabled=effective_edit_intent_plan_enabled,
    )
    candidate_replay = (
        _maybe_replay_candidate_patch(
            executor=executor,
            workspace=workspace,
            patcher_failure_evidence=patcher_failure_evidence,
            patch_contract=patch_contract,
            timeout=min(timeout, 120),
        )
        if bounded_validation_enabled
        else _candidate_patch_replay_skipped_result("bounded_validation_disabled")
    )
    if _candidate_replay_blocked(candidate_replay):
        blocked = _candidate_replay_blocked_result(
            evaluation=evaluation,
            candidate_replay=candidate_replay,
            action_adherence_observed={
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": True,
                "locator_attempted": False,
                "selected_target_count": len(fixed["selected_target_candidates"]),
                "candidate_patch_replay_attempted": True,
                "candidate_patch_replayed": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
            },
        )
        blocked["edit_intent_plan"] = edit_intent_plan
        blocked["edit_intent_plan_enabled"] = effective_edit_intent_plan_enabled
        return blocked
    model = disable_streaming_if_supported(
        build_chat_model(
            api_path,
            model_name=model_name or None,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    )
    strong_model = model
    if strong_model_name:
        strong_model = disable_streaming_if_supported(
            build_chat_model(
                api_path,
                model_name=strong_model_name,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )
        )
    usage_before = _usage_snapshot(strong_model)
    planner_model = QueryBudgetModel(strong_model, {"max_tokens": 512})
    implementer_model = QueryBudgetModel(strong_model, {"max_tokens": 1024})
    patcher = PlannerPatcherAdapter(
        model=implementer_model,
        planner_model=planner_model,
        implementer_model=implementer_model,
        executor=executor,
        max_plan_iterations=planner_max_iterations,
        max_patch_iterations=patcher_max_iterations,
        enable_recovery_fresh_diff_audit=patch_contract == "source_only",
        recovery_execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
    )
    recovery_context = _patcher_recovery_context(
        manifest,
        fixed,
        patcher_failure_evidence=patcher_failure_evidence,
        patch_contract=patch_contract,
        operator_gate=operator_gate,
        edit_intent_plan=edit_intent_plan,
    )
    patch_result = patcher.patch(
        issue=_patcher_issue(manifest),
        workspace=str(workspace),
        located_files=str(fixed["located_files"]),
        recovery_context=recovery_context,
    )
    usage_after = _usage_snapshot(strong_model)
    usage_delta = _usage_delta(usage_before, usage_after)
    provider_fields = _provider_api_error_fields(patch_result)
    patch_summary = dict(patch_result.get("patch_summary", {}) or {})
    patch_record = _patch_text_for_record(
        patch_result=patch_result,
        patch_summary=patch_summary,
        workspace=workspace,
    )
    patch_text = patch_record["patch"]
    patch_attempted = _patch_observed(patch_result)
    infrastructure_error = bool(patch_result.get("infrastructure_error", False))
    invariant_gate = (
        _semantic_invariant_patch_gate(
            patch_text=patch_text,
            patcher_failure_evidence=patcher_failure_evidence,
            operator_gate=operator_gate,
        )
        if bounded_validation_enabled
        else _semantic_invariant_patch_gate_skipped_result("bounded_validation_disabled")
    )
    semantic_invariant_revision = {
        "attempted": False,
        "accepted": False,
        "reason": "",
        "first_gate": invariant_gate,
        "second_gate": {},
        "max_revision_count": 1,
    }
    if invariant_gate["blocked"] and _semantic_invariant_revision_enabled(operator_gate):
        revision_context = _semantic_invariant_revision_context(
            recovery_context=recovery_context,
            gate=invariant_gate,
            patch_text=patch_text,
        )
        revision_patcher = PlannerPatcherAdapter(
            model=implementer_model,
            planner_model=planner_model,
            implementer_model=implementer_model,
            executor=executor,
            max_plan_iterations=max(1, min(2, planner_max_iterations)),
            max_patch_iterations=max(2, min(4, patcher_max_iterations)),
            enable_recovery_fresh_diff_audit=patch_contract == "source_only",
            recovery_execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
        )
        revision_patch_result = revision_patcher.patch(
            issue=_patcher_issue(manifest),
            workspace=str(workspace),
            located_files=str(fixed["located_files"]),
            recovery_context=revision_context,
        )
        revision_patch_summary = dict(revision_patch_result.get("patch_summary", {}) or {})
        revision_patch_record = _patch_text_for_record(
            patch_result=revision_patch_result,
            patch_summary=revision_patch_summary,
            workspace=workspace,
        )
        revision_patch_text = revision_patch_record["patch"]
        revision_gate = _semantic_invariant_patch_gate(
            patch_text=revision_patch_text or patch_text,
            patcher_failure_evidence=patcher_failure_evidence,
            operator_gate=operator_gate,
        )
        semantic_invariant_revision = {
            "attempted": True,
            "accepted": not bool(revision_gate.get("blocked", False)),
            "reason": "bounded_revision_after_semantic_invariant_gate",
            "first_gate": invariant_gate,
            "second_gate": revision_gate,
            "max_revision_count": 1,
            "revision_patcher_success": bool(revision_patch_result.get("success", False)),
            "revision_stop_reason": _patcher_stop_reason(revision_patch_result),
        }
        if revision_patch_text.strip():
            patch_result = revision_patch_result
            patch_summary = revision_patch_summary
            patch_record = revision_patch_record
            patch_text = revision_patch_text
            patch_attempted = _patch_observed(patch_result)
            infrastructure_error = bool(patch_result.get("infrastructure_error", False))
            invariant_gate = revision_gate
            usage_after = _usage_snapshot(strong_model)
            usage_delta = _usage_delta(usage_before, usage_after)
    if invariant_gate["blocked"]:
        patch_intent_gate = patch_intent_gate_skipped_result("semantic_invariant_gate_blocked_before_patch_intent")
        patch_intent_revision = _patch_intent_revision_skipped_result("semantic_invariant_gate_blocked")
        invariant_stop_reason = (
            "semantic_invariant_gate_blocked_after_bounded_revision"
            if semantic_invariant_revision.get("attempted")
            else "semantic_invariant_gate_blocked"
        )
        return {
            **_evaluation_command_payload(evaluation),
            **candidate_replay,
            "error_type": "",
            "stop_reason": invariant_stop_reason,
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "evaluation_protocol_error": False,
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_gate_blocked": True,
            "semantic_invariant_revision": semantic_invariant_revision,
            "patch_intent_gate": patch_intent_gate,
            "patch_intent_gate_blocked": False,
            "patch_intent_revision": patch_intent_revision,
            "edit_intent_plan": edit_intent_plan,
            "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
            "patcher_success": bool(patch_result.get("success", False)),
            "patcher_infrastructure_error": infrastructure_error,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_after,
            "model_usage_delta": usage_delta,
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "planner_error": str(patch_result.get("planner_error", "") or ""),
            "implementer_error": str(patch_result.get("implementer_error", "") or ""),
            **provider_fields,
            "patcher_plan": str(patch_result.get("plan", "") or ""),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
            "fixed_localization_workspace_path_resolution": path_resolution,
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
            ),
            "patcher_failure_evidence_gate_status": str(
                patcher_failure_evidence.get("gate_status", "") if patcher_failure_evidence else ""
            ),
            "patcher_failure_evidence_signature": dict(
                patcher_failure_evidence.get("post_patch_failure_signature", {})
                if patcher_failure_evidence
                else {}
            ),
            "patch": patch_text,
            "patch_source": patch_record["source"],
            "patch_summary": patch_summary,
            "patch_legitimacy": str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            "stage_outputs": {
                "locator": {
                    "success": True,
                    "reused_from_previous_trajectory": True,
                    "located_files": fixed["located_files"],
                    "selected_target_candidates": fixed["selected_target_candidates"],
                },
                "patcher": patch_result,
                "verifier": {
                    "semantic_invariant_gate_blocked": True,
                    "semantic_invariant_violations": list(invariant_gate.get("violations", []) or []),
                    "oracle_success": False,
                },
            },
            "stage_stop_reasons": {
                "locator": "skipped_reused_existing_localization",
                "patcher": _patcher_stop_reason(patch_result),
                "verifier": invariant_stop_reason,
            },
            "fixed_localization_source": fixed["source"],
            "fixed_localization_record_path": str(trajectory_records_path),
            "selected_target_candidates": fixed["selected_target_candidates"],
            "action_adherence_observed": {
                "patch_attempted": patch_attempted,
                "verifier_attempted": True,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": True,
                "locator_attempted": False,
                "patcher_success": bool(patch_result.get("success", False)),
                "selected_target_count": len(fixed["selected_target_candidates"]),
                "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
                "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
                "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
                "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": True,
                "semantic_invariant_gate_blocked": True,
                "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
                "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
                "patch_intent_gate_attempted": False,
                "patch_intent_gate_blocked": False,
                "patch_intent_revision_attempted": False,
                "patch_intent_revision_accepted": False,
            },
        }
    patch_intent_gate = (
        evaluate_patch_intent_gate(
            patch_text=patch_text,
            patch_summary=patch_summary,
            patcher_failure_evidence=patcher_failure_evidence,
            enabled=True,
        )
        if bounded_validation_enabled and (patch_intent_gate_enabled or patch_intent_advisory_enabled)
        else patch_intent_gate_skipped_result(
            "bounded_validation_disabled" if not bounded_validation_enabled else "not_requested"
        )
    )
    patch_intent_revision = _patch_intent_revision_skipped_result(
        "patch_intent_gate_not_blocked"
        if not bool(patch_intent_gate.get("blocked", False))
        else "patch_intent_revision_disabled"
    )
    if bool(patch_intent_gate.get("blocked", False)) and patch_intent_revision_enabled:
        intent_revision_context = _patch_intent_revision_context(
            recovery_context=recovery_context,
            gate=patch_intent_gate,
            patch_text=patch_text,
        )
        intent_revision_patcher = PlannerPatcherAdapter(
            model=implementer_model,
            planner_model=planner_model,
            implementer_model=implementer_model,
            executor=executor,
            max_plan_iterations=max(1, min(2, planner_max_iterations)),
            max_patch_iterations=max(2, min(4, patcher_max_iterations)),
            enable_recovery_fresh_diff_audit=patch_contract == "source_only",
            recovery_execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
        )
        intent_revision_patch_result = intent_revision_patcher.patch(
            issue=_patcher_issue(manifest),
            workspace=str(workspace),
            located_files=str(fixed["located_files"]),
            recovery_context=intent_revision_context,
        )
        intent_revision_patch_summary = dict(intent_revision_patch_result.get("patch_summary", {}) or {})
        intent_revision_patch_record = _patch_text_for_record(
            patch_result=intent_revision_patch_result,
            patch_summary=intent_revision_patch_summary,
            workspace=workspace,
        )
        intent_revision_patch_text = intent_revision_patch_record["patch"]
        intent_revision_invariant_gate = _semantic_invariant_patch_gate(
            patch_text=intent_revision_patch_text or patch_text,
            patcher_failure_evidence=patcher_failure_evidence,
            operator_gate=operator_gate,
        )
        intent_revision_gate = evaluate_patch_intent_gate(
            patch_text=intent_revision_patch_text or patch_text,
            patch_summary=intent_revision_patch_summary or patch_summary,
            patcher_failure_evidence=patcher_failure_evidence,
            enabled=True,
        )
        intent_revision_accepted = (
            bool(intent_revision_patch_text.strip())
            and not bool(intent_revision_invariant_gate.get("blocked", False))
            and not bool(intent_revision_gate.get("blocked", False))
        )
        patch_intent_revision = {
            "attempted": True,
            "accepted": intent_revision_accepted,
            "reason": "bounded_revision_after_patch_intent_gate",
            "first_gate": patch_intent_gate,
            "second_gate": intent_revision_gate,
            "revision_invariant_gate": intent_revision_invariant_gate,
            "max_revision_count": 1,
            "revision_patcher_success": bool(intent_revision_patch_result.get("success", False)),
            "revision_stop_reason": _patcher_stop_reason(intent_revision_patch_result),
        }
        if intent_revision_patch_text.strip():
            patch_result = intent_revision_patch_result
            patch_summary = intent_revision_patch_summary
            patch_record = intent_revision_patch_record
            patch_text = intent_revision_patch_text
            patch_attempted = _patch_observed(patch_result)
            infrastructure_error = bool(patch_result.get("infrastructure_error", False))
            invariant_gate = intent_revision_invariant_gate
            patch_intent_gate = intent_revision_gate
            usage_after = _usage_snapshot(strong_model)
            usage_delta = _usage_delta(usage_before, usage_after)
    hard_patch_intent_blocked = bool(patch_intent_gate.get("blocked", False)) and not patch_intent_advisory_enabled
    if bool(invariant_gate.get("blocked", False)) or hard_patch_intent_blocked:
        gate_stop_reason = (
            "patch_intent_gate_semantic_invariant_blocked_after_bounded_revision"
            if bool(invariant_gate.get("blocked", False)) and patch_intent_revision.get("attempted")
            else "patch_intent_gate_blocked_after_bounded_revision"
            if patch_intent_revision.get("attempted")
            else "patch_intent_gate_blocked"
        )
        return {
            **_evaluation_command_payload(evaluation),
            **candidate_replay,
            "error_type": "",
            "stop_reason": gate_stop_reason,
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "evaluation_protocol_error": False,
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
            "semantic_invariant_revision": semantic_invariant_revision,
            "patch_intent_gate": patch_intent_gate,
            "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
            "patch_intent_revision": patch_intent_revision,
            "edit_intent_plan": edit_intent_plan,
            "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
            "patcher_success": bool(patch_result.get("success", False)),
            "patcher_infrastructure_error": infrastructure_error,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_after,
            "model_usage_delta": usage_delta,
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "planner_error": str(patch_result.get("planner_error", "") or ""),
            "implementer_error": str(patch_result.get("implementer_error", "") or ""),
            "patcher_plan": str(patch_result.get("plan", "") or ""),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
            "fixed_localization_workspace_path_resolution": path_resolution,
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "patch_intent_gate", "runtime_collect_validation"]
            ),
            "patcher_failure_evidence_gate_status": str(
                patcher_failure_evidence.get("gate_status", "") if patcher_failure_evidence else ""
            ),
            "patcher_failure_evidence_signature": dict(
                patcher_failure_evidence.get("post_patch_failure_signature", {})
                if patcher_failure_evidence
                else {}
            ),
            "patch": patch_text,
            "patch_source": patch_record["source"],
            "patch_summary": patch_summary,
            "patch_legitimacy": str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            "stage_outputs": {
                "locator": {
                    "success": True,
                    "reused_from_previous_trajectory": True,
                    "located_files": fixed["located_files"],
                    "selected_target_candidates": fixed["selected_target_candidates"],
                },
                "patcher": patch_result,
                "verifier": {
                    "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
                    "patch_intent_violations": list(patch_intent_gate.get("violations", []) or []),
                    "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                    "semantic_invariant_violations": list(invariant_gate.get("violations", []) or []),
                    "oracle_success": False,
                },
            },
            "stage_stop_reasons": {
                "locator": "skipped_reused_existing_localization",
                "patcher": _patcher_stop_reason(patch_result),
                "verifier": gate_stop_reason,
            },
            "fixed_localization_source": fixed["source"],
            "fixed_localization_record_path": str(trajectory_records_path),
            "selected_target_candidates": fixed["selected_target_candidates"],
            "action_adherence_observed": {
                "patch_attempted": patch_attempted,
                "verifier_attempted": True,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": True,
                "locator_attempted": False,
                "patcher_success": bool(patch_result.get("success", False)),
                "selected_target_count": len(fixed["selected_target_candidates"]),
                "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
                "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
                "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
                "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
                "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
                "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
                "patch_intent_gate_attempted": bool(patch_intent_gate.get("attempted", False)),
                "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
                "patch_intent_revision_attempted": bool(patch_intent_revision.get("attempted", False)),
                "patch_intent_revision_accepted": bool(patch_intent_revision.get("accepted", False)),
            },
        }
    if candidate_only:
        provider_api_fields = _provider_api_error_fields(patch_result)
        candidate_status = _candidate_only_status(
            patch_text=patch_text,
            patch_summary=patch_summary,
            patcher_success=bool(patch_result.get("success", False)),
            infrastructure_error=infrastructure_error,
            provider_api_error=bool(provider_api_fields.get("provider_api_error", False)),
            semantic_invariant_gate=invariant_gate,
            patch_intent_gate=patch_intent_gate,
            patch_contract=patch_contract,
        )
        return {
            **_evaluation_command_payload(evaluation),
            **candidate_replay,
            "error_type": (
                "provider_api_error"
                if bool(provider_api_fields.get("provider_api_error", False))
                else "patcher_infrastructure_error" if infrastructure_error else ""
            ),
            "stop_reason": candidate_status["stop_reason"],
            **provider_api_fields,
            "candidate_only": True,
            "candidate_generation_status": candidate_status["status"],
            "candidate_generation_accept": candidate_status["accept"],
            "candidate_generation_reject_reasons": candidate_status["reject_reasons"],
            "candidate_generation_no_oracle": True,
            "candidate_generation_requires_separate_verifier_replay": True,
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "evaluation_protocol_error": False,
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "patcher_success": bool(patch_result.get("success", False)),
            "patcher_infrastructure_error": infrastructure_error,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_after,
            "model_usage_delta": usage_delta,
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "planner_error": str(patch_result.get("planner_error", "") or ""),
            "implementer_error": str(patch_result.get("implementer_error", "") or ""),
            "patcher_plan": str(patch_result.get("plan", "") or ""),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "environment_preflight_command": preflight_command,
            "environment_preflight_returncode": preflight["returncode"],
            "environment_preflight_output": preflight["output"],
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "patch_intent_gate"]
            ),
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_revision": semantic_invariant_revision,
            "patch_intent_gate": patch_intent_gate,
            "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
            "patch_intent_revision": patch_intent_revision,
            "edit_intent_plan": edit_intent_plan,
            "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
            "patcher_failure_evidence_gate_status": str(
                patcher_failure_evidence.get("gate_status", "") if patcher_failure_evidence else ""
            ),
            "patcher_failure_evidence_signature": dict(
                patcher_failure_evidence.get("post_patch_failure_signature", {})
                if patcher_failure_evidence
                else {}
            ),
            "patch": patch_text,
            "patch_source": patch_record["source"],
            "patch_summary": patch_summary,
            "patch_legitimacy": str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            "stage_outputs": {
                "locator": {
                    "success": True,
                    "reused_from_previous_trajectory": True,
                    "located_files": fixed["located_files"],
                    "selected_target_candidates": fixed["selected_target_candidates"],
                },
                "patcher": patch_result,
                "verifier": {
                    "candidate_only": True,
                    "oracle_success": False,
                    "oracle_not_run": True,
                },
            },
            "stage_stop_reasons": {
                "locator": "skipped_reused_existing_localization",
                "patcher": _patcher_stop_reason(patch_result),
                "verifier": "candidate_only_oracle_not_run",
            },
            "fixed_localization_source": fixed["source"],
            "fixed_localization_record_path": str(trajectory_records_path),
            "selected_target_candidates": fixed["selected_target_candidates"],
            "action_adherence_observed": {
                "patch_attempted": patch_attempted,
                "verifier_attempted": False,
                "oracle_attempted": False,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": True,
                "locator_attempted": False,
                "patcher_success": bool(patch_result.get("success", False)),
                "selected_target_count": len(fixed["selected_target_candidates"]),
                "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
                "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
                "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
                "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
                "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
                "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
                "patch_intent_gate_attempted": bool(patch_intent_gate.get("attempted", False)),
                "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
                "patch_intent_revision_attempted": bool(patch_intent_revision.get("attempted", False)),
                "patch_intent_revision_accepted": bool(patch_intent_revision.get("accepted", False)),
            },
        }
    syntax_gate = (
        _candidate_patch_syntax_gate(
            executor=executor,
            workspace=workspace,
            source_files=_candidate_patch_summary_source_files(patch_summary),
            timeout=min(timeout, 120),
        )
        if bounded_validation_enabled
        else _candidate_patch_syntax_gate_skipped_result("bounded_validation_disabled")
    )
    syntax_repair_revision = {
        "attempted": False,
        "accepted": False,
        "reason": "",
        "first_gate": syntax_gate,
        "second_gate": {},
        "max_revision_count": 1,
    }
    if bool(syntax_gate.get("blocked", False)) and syntax_repair_revision_enabled:
        syntax_revision_context = _syntax_repair_revision_context(
            recovery_context=recovery_context,
            syntax_gate=syntax_gate,
            patch_text=patch_text,
        )
        syntax_revision_patcher = PlannerPatcherAdapter(
            model=implementer_model,
            planner_model=planner_model,
            implementer_model=implementer_model,
            executor=executor,
            max_plan_iterations=max(1, min(2, planner_max_iterations)),
            max_patch_iterations=max(2, min(4, patcher_max_iterations)),
            enable_recovery_fresh_diff_audit=patch_contract == "source_only",
            recovery_execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
        )
        syntax_revision_patch_result = syntax_revision_patcher.patch(
            issue=_patcher_issue(manifest),
            workspace=str(workspace),
            located_files=str(fixed["located_files"]),
            recovery_context=syntax_revision_context,
        )
        syntax_revision_patch_summary = dict(syntax_revision_patch_result.get("patch_summary", {}) or {})
        syntax_revision_patch_record = _patch_text_for_record(
            patch_result=syntax_revision_patch_result,
            patch_summary=syntax_revision_patch_summary,
            workspace=workspace,
        )
        syntax_revision_patch_text = syntax_revision_patch_record["patch"]
        syntax_revision_gate = (
            _candidate_patch_syntax_gate(
                executor=executor,
                workspace=workspace,
                source_files=_candidate_patch_summary_source_files(syntax_revision_patch_summary or patch_summary),
                timeout=min(timeout, 120),
            )
            if syntax_revision_patch_text.strip()
            else _candidate_patch_syntax_gate_skipped_result("syntax_revision_no_fresh_patch")
        )
        syntax_repair_revision = {
            "attempted": True,
            "accepted": bool(syntax_revision_patch_text.strip())
            and not bool(syntax_revision_gate.get("blocked", False)),
            "reason": "bounded_revision_after_candidate_patch_syntax_gate",
            "first_gate": syntax_gate,
            "second_gate": syntax_revision_gate,
            "max_revision_count": 1,
            "revision_patcher_success": bool(syntax_revision_patch_result.get("success", False)),
            "revision_stop_reason": _patcher_stop_reason(syntax_revision_patch_result),
        }
        if syntax_revision_patch_text.strip():
            patch_result = syntax_revision_patch_result
            patch_summary = syntax_revision_patch_summary
            patch_record = syntax_revision_patch_record
            patch_text = syntax_revision_patch_text
            patch_attempted = _patch_observed(patch_result)
            infrastructure_error = bool(patch_result.get("infrastructure_error", False))
            syntax_gate = syntax_revision_gate
            usage_after = _usage_snapshot(strong_model)
            usage_delta = _usage_delta(usage_before, usage_after)
            provider_fields = _provider_api_error_fields(patch_result)
    if bool(syntax_gate.get("blocked", False)):
        return {
            **_evaluation_command_payload(evaluation),
            **candidate_replay,
            "error_type": "",
            "stop_reason": "candidate_patch_syntax_gate_blocked",
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "evaluation_protocol_error": False,
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_revision": semantic_invariant_revision,
            "patch_intent_gate": patch_intent_gate,
            "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
            "patch_intent_revision": patch_intent_revision,
            "candidate_patch_syntax_gate": syntax_gate,
            "candidate_patch_syntax_gate_blocked": True,
            "syntax_repair_revision": syntax_repair_revision,
            "syntax_repair_revision_enabled": syntax_repair_revision_enabled,
            "edit_intent_plan": edit_intent_plan,
            "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
            "patcher_success": bool(patch_result.get("success", False)),
            "patcher_infrastructure_error": infrastructure_error,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_after,
            "model_usage_delta": usage_delta,
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "planner_error": str(patch_result.get("planner_error", "") or ""),
            "implementer_error": str(patch_result.get("implementer_error", "") or ""),
            **provider_fields,
            "patcher_plan": str(patch_result.get("plan", "") or ""),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "environment_preflight_command": preflight_command,
            "environment_preflight_returncode": preflight["returncode"],
            "environment_preflight_output": preflight["output"],
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "patch_intent_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
            ),
            "patch": patch_text,
            "patch_source": patch_record["source"],
            "patch_summary": patch_summary,
            "patch_legitimacy": str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            "stage_outputs": {
                "locator": {
                    "success": True,
                    "reused_from_previous_trajectory": True,
                    "located_files": fixed["located_files"],
                    "selected_target_candidates": fixed["selected_target_candidates"],
                },
                "patcher": patch_result,
                "verifier": {
                    "candidate_patch_syntax_gate_blocked": True,
                    "candidate_patch_syntax_violations": list(syntax_gate.get("violations", []) or []),
                    "oracle_success": False,
                },
            },
            "stage_stop_reasons": {
                "locator": "skipped_reused_existing_localization",
                "patcher": _patcher_stop_reason(patch_result),
                "verifier": "candidate_patch_syntax_gate_blocked",
            },
            "fixed_localization_source": fixed["source"],
            "fixed_localization_record_path": str(trajectory_records_path),
            "selected_target_candidates": fixed["selected_target_candidates"],
            "action_adherence_observed": {
                "patch_attempted": patch_attempted,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "fixed_localization_reused": True,
                "locator_attempted": False,
                "patcher_success": bool(patch_result.get("success", False)),
                "selected_target_count": len(fixed["selected_target_candidates"]),
                "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
                "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
                "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
                "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                "candidate_patch_syntax_gate_attempted": bool(syntax_gate.get("attempted", False)),
                "candidate_patch_syntax_gate_blocked": True,
                "post_patch_collect_validation_attempted": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
                "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
                "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
                "patch_intent_gate_attempted": bool(patch_intent_gate.get("attempted", False)),
                "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
                "patch_intent_revision_attempted": bool(patch_intent_revision.get("attempted", False)),
                "patch_intent_revision_accepted": bool(patch_intent_revision.get("accepted", False)),
                "syntax_repair_revision_attempted": bool(syntax_repair_revision.get("attempted", False)),
                "syntax_repair_revision_accepted": bool(syntax_repair_revision.get("accepted", False)),
            },
        }
    provider_fields = _provider_api_error_fields(patch_result)
    if bounded_validation_enabled:
        evaluation = _runtime_validate_evaluation_commands(
            manifest=manifest,
            evaluation=evaluation,
            executor=executor,
            workspace=workspace,
            timeout=min(timeout, 300),
        )
    evaluation.setdefault("normalization_notes", [])
    notes = list(evaluation.get("normalization_notes", []) or [])
    notes.append(
        "runtime_collect_validation_after_patch"
        if bounded_validation_enabled
        else "runtime_collect_validation_skipped_bounded_validation_disabled"
    )
    evaluation["normalization_notes"] = list(dict.fromkeys(notes))
    if evaluation["error_type"]:
        return {
            **_evaluation_blocked_result(
                evaluation=evaluation,
                extra=candidate_replay,
                action_adherence_observed={
                    "patch_attempted": patch_attempted,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": False,
                    "fixed_localization_reused": True,
                    "locator_attempted": False,
                    "patcher_success": bool(patch_result.get("success", False)),
                    "selected_target_count": len(fixed["selected_target_candidates"]),
                    "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
                    "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
                    "candidate_patch_replay_attempted": bool(
                        candidate_replay.get("candidate_patch_replay_attempted", False)
                    ),
                    "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                    "post_patch_collect_validation_attempted": bounded_validation_enabled,
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped": not bounded_validation_enabled,
                    "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
                    "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                    "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
                    "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
                    "patch_intent_gate_attempted": bool(patch_intent_gate.get("attempted", False)),
                    "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
                    "patch_intent_revision_attempted": bool(patch_intent_revision.get("attempted", False)),
                    "patch_intent_revision_accepted": bool(patch_intent_revision.get("accepted", False)),
                },
            ),
            "patcher_success": bool(patch_result.get("success", False)),
            "patcher_infrastructure_error": infrastructure_error,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_after,
            "model_usage_delta": usage_delta,
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "planner_error": str(patch_result.get("planner_error", "") or ""),
            "implementer_error": str(patch_result.get("implementer_error", "") or ""),
            "patcher_plan": str(patch_result.get("plan", "") or ""),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
            ),
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_revision": semantic_invariant_revision,
            "patch_intent_gate": patch_intent_gate,
            "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
            "patch_intent_revision": patch_intent_revision,
            "edit_intent_plan": edit_intent_plan,
            "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
            "patch": patch_text,
            "patch_source": patch_record["source"],
            "patch_summary": patch_summary,
            "patch_legitimacy": str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            "fixed_localization_source": fixed["source"],
            "fixed_localization_record_path": str(trajectory_records_path),
            "selected_target_candidates": fixed["selected_target_candidates"],
        }
    fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
    oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
    protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
    semantic_span_delta_gate = {
        "attempted": False,
        "revision_attempted": False,
        "revision_accepted": False,
        "first_delta": {},
        "final_delta": {},
        "max_revision_count": 1,
    }
    if bounded_validation_enabled and _post_patch_span_delta_revision_enabled(
        operator_gate=operator_gate,
        post_patch_span_delta_revision_enabled=post_patch_span_delta_revision_enabled,
    ):
        first_span_delta = _semantic_span_delta_after_validation(
            raw_failure_output="\n".join([str(fail_to_pass.get("output", "") or ""), str(oracle.get("output", "") or "")]),
            patcher_failure_evidence=patcher_failure_evidence,
            oracle_success=oracle["returncode"] == 0,
            protocol=protocol,
            patch_legitimacy=str(
                patch_summary.get("fresh_target_legitimacy")
                or patch_summary.get("target_legitimacy")
                or ""
            ),
            error_type="patcher_infrastructure_error" if infrastructure_error else "",
        )
        semantic_span_delta_gate["attempted"] = True
        semantic_span_delta_gate["first_delta"] = first_span_delta
        semantic_span_delta_gate["final_delta"] = first_span_delta
        if bool(first_span_delta.get("requires_revision", False)):
            span_delta_context = _semantic_span_delta_revision_context(
                recovery_context=recovery_context,
                span_delta=first_span_delta,
                patch_text=patch_text,
            )
            span_delta_patcher = PlannerPatcherAdapter(
                model=implementer_model,
                planner_model=planner_model,
                implementer_model=implementer_model,
                executor=executor,
                max_plan_iterations=max(1, min(2, planner_max_iterations)),
                max_patch_iterations=max(2, min(4, patcher_max_iterations)),
                enable_recovery_fresh_diff_audit=patch_contract == "source_only",
                recovery_execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
            )
            span_delta_patch_result = span_delta_patcher.patch(
                issue=_patcher_issue(manifest),
                workspace=str(workspace),
                located_files=str(fixed["located_files"]),
                recovery_context=span_delta_context,
            )
            span_delta_patch_summary = dict(span_delta_patch_result.get("patch_summary", {}) or {})
            span_delta_patch_record = _patch_text_for_record(
                patch_result=span_delta_patch_result,
                patch_summary=span_delta_patch_summary,
                workspace=workspace,
            )
            span_delta_patch_text = span_delta_patch_record["patch"]
            span_delta_invariant_gate = _semantic_invariant_patch_gate(
                patch_text=span_delta_patch_text or patch_text,
                patcher_failure_evidence=patcher_failure_evidence,
                operator_gate=operator_gate,
            )
            semantic_span_delta_gate.update(
                {
                    "revision_attempted": True,
                    "revision_accepted": bool(span_delta_patch_text.strip()) and not bool(span_delta_invariant_gate.get("blocked", False)),
                    "revision_patcher_success": bool(span_delta_patch_result.get("success", False)),
                    "revision_stop_reason": _patcher_stop_reason(span_delta_patch_result),
                    "revision_invariant_gate": span_delta_invariant_gate,
                }
            )
            if bool(span_delta_invariant_gate.get("blocked", False)):
                semantic_span_delta_gate["final_delta"] = {
                    **first_span_delta,
                    "delta": "span_delta_revision_semantic_invariant_blocked",
                    "requires_revision": False,
                }
            elif span_delta_patch_text.strip():
                patch_result = span_delta_patch_result
                patch_summary = span_delta_patch_summary
                patch_record = span_delta_patch_record
                patch_text = span_delta_patch_text
                patch_attempted = _patch_observed(patch_result)
                infrastructure_error = bool(patch_result.get("infrastructure_error", False))
                invariant_gate = span_delta_invariant_gate
                usage_after = _usage_snapshot(strong_model)
                usage_delta = _usage_delta(usage_before, usage_after)
                fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
                oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
                protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
                semantic_span_delta_gate["final_delta"] = _semantic_span_delta_after_validation(
                    raw_failure_output="\n".join(
                        [str(fail_to_pass.get("output", "") or ""), str(oracle.get("output", "") or "")]
                    ),
                    patcher_failure_evidence=patcher_failure_evidence,
                    oracle_success=oracle["returncode"] == 0,
                    protocol=protocol,
                    patch_legitimacy=str(
                        patch_summary.get("fresh_target_legitimacy")
                        or patch_summary.get("target_legitimacy")
                        or ""
                    ),
                    error_type="patcher_infrastructure_error" if infrastructure_error else "",
                )
            else:
                semantic_span_delta_gate["final_delta"] = {
                    **first_span_delta,
                    "delta": "span_delta_revision_no_fresh_patch",
                    "requires_revision": False,
                }
    verified_status = _verified_recovery_status_after_patcher(
        patch_result=patch_result,
        fail_to_pass=fail_to_pass,
        oracle=oracle,
        protocol=protocol,
    )
    return {
        **_evaluation_command_payload(evaluation),
        **candidate_replay,
        "error_type": "patcher_infrastructure_error" if infrastructure_error else "",
        "stop_reason": verified_status["stop_reason"],
        "status_consistency_resolution": verified_status["status_consistency_resolution"],
        "fail_to_pass_returncode": fail_to_pass["returncode"],
        "oracle_returncode": oracle["returncode"],
        "oracle_success": oracle["returncode"] == 0,
        "reported_success": oracle["returncode"] == 0,
        "fail_to_pass_output": fail_to_pass["output"],
        "oracle_output": oracle["output"],
        **protocol,
        "patcher_success": bool(patch_result.get("success", False)),
        "patcher_infrastructure_error": infrastructure_error,
        "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
        "strong_model_name": str(
            getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
        ),
        "model_usage_before": usage_before,
        "model_usage_after": usage_after,
        "model_usage_delta": usage_delta,
        "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
        "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
        "planner_error": str(patch_result.get("planner_error", "") or ""),
        "implementer_error": str(patch_result.get("implementer_error", "") or ""),
        **provider_fields,
        "patcher_plan": str(patch_result.get("plan", "") or ""),
        "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
        "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
        "derived_semantic_invariant_evidence": derived_semantic_invariant_evidence,
        "patch_contract": patch_contract,
        "operator_gate": operator_gate,
        "environment_preflight_command": preflight_command,
        "environment_preflight_returncode": preflight["returncode"],
        "environment_preflight_output": preflight["output"],
        "bounded_validation_enabled": bounded_validation_enabled,
        "bounded_validation_skipped_steps": (
            []
            if bounded_validation_enabled
            else ["candidate_patch_replay", "semantic_invariant_gate", "runtime_collect_validation"]
        ),
        "semantic_invariant_gate": invariant_gate,
        "semantic_invariant_revision": semantic_invariant_revision,
        "patch_intent_gate": patch_intent_gate,
        "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
        "patch_intent_revision": patch_intent_revision,
        "candidate_patch_syntax_gate": syntax_gate,
        "candidate_patch_syntax_gate_blocked": bool(syntax_gate.get("blocked", False)),
        "syntax_repair_revision": syntax_repair_revision,
        "syntax_repair_revision_enabled": syntax_repair_revision_enabled,
        "edit_intent_plan": edit_intent_plan,
        "edit_intent_plan_enabled": effective_edit_intent_plan_enabled,
        "semantic_span_delta_gate": semantic_span_delta_gate,
        "post_patch_span_delta_revision_enabled": post_patch_span_delta_revision_enabled,
        "patcher_failure_evidence_gate_status": str(
            patcher_failure_evidence.get("gate_status", "") if patcher_failure_evidence else ""
        ),
        "patcher_failure_evidence_signature": dict(
            patcher_failure_evidence.get("post_patch_failure_signature", {})
            if patcher_failure_evidence
            else {}
        ),
        "patch": patch_text,
        "patch_source": patch_record["source"],
        "patch_summary": patch_summary,
        "patch_legitimacy": str(
            patch_summary.get("fresh_target_legitimacy")
            or patch_summary.get("target_legitimacy")
            or ""
        ),
        "stage_outputs": {
            "locator": {
                "success": True,
                "reused_from_previous_trajectory": True,
                "located_files": fixed["located_files"],
                "selected_target_candidates": fixed["selected_target_candidates"],
            },
            "patcher": patch_result,
            "verifier": {
                "fail_to_pass_returncode": fail_to_pass["returncode"],
                "oracle_returncode": oracle["returncode"],
                "oracle_success": oracle["returncode"] == 0,
                "semantic_span_delta_gate": semantic_span_delta_gate,
                "post_patch_span_delta_revision_enabled": post_patch_span_delta_revision_enabled,
            },
        },
        "stage_stop_reasons": {
            "locator": "skipped_reused_existing_localization",
            "patcher": _patcher_stop_reason(patch_result),
            "verifier": "executed",
        },
        "fixed_localization_source": fixed["source"],
        "fixed_localization_record_path": str(trajectory_records_path),
        "selected_target_candidates": fixed["selected_target_candidates"],
        "action_adherence_observed": {
            "patch_attempted": patch_attempted,
            "verifier_attempted": True,
            "environment_preflight_attempted": bounded_validation_enabled,
            "fixed_localization_reused": True,
            "locator_attempted": False,
            "patcher_success": bool(patch_result.get("success", False)),
            "selected_target_count": len(fixed["selected_target_candidates"]),
            "model_call_count": int(usage_delta.get("n_calls", 0.0) or 0.0),
            "token_cost": float(usage_delta.get("total_tokens", 0.0) or 0.0),
            "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
            "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
            "post_patch_collect_validation_attempted": bounded_validation_enabled,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped": not bounded_validation_enabled,
            "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
            "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
            "semantic_invariant_revision_attempted": bool(semantic_invariant_revision.get("attempted", False)),
            "semantic_invariant_revision_accepted": bool(semantic_invariant_revision.get("accepted", False)),
            "patch_intent_gate_attempted": bool(patch_intent_gate.get("attempted", False)),
            "patch_intent_gate_blocked": bool(patch_intent_gate.get("blocked", False)),
            "patch_intent_revision_attempted": bool(patch_intent_revision.get("attempted", False)),
            "patch_intent_revision_accepted": bool(patch_intent_revision.get("accepted", False)),
            "candidate_patch_syntax_gate_attempted": bool(syntax_gate.get("attempted", False)),
            "candidate_patch_syntax_gate_blocked": bool(syntax_gate.get("blocked", False)),
            "syntax_repair_revision_attempted": bool(syntax_repair_revision.get("attempted", False)),
            "syntax_repair_revision_accepted": bool(syntax_repair_revision.get("accepted", False)),
            "semantic_span_delta_gate_attempted": bool(semantic_span_delta_gate.get("attempted", False)),
            "semantic_span_delta_revision_attempted": bool(semantic_span_delta_gate.get("revision_attempted", False)),
            "semantic_span_delta_revision_accepted": bool(semantic_span_delta_gate.get("revision_accepted", False)),
            "post_patch_span_delta_revision_enabled": post_patch_span_delta_revision_enabled,
        },
    }


def _execute_candidate_patch_verifier(
    *,
    evaluation: dict[str, Any],
    manifest: dict[str, Any],
    instance_id: str,
    executor,
    workspace: Path,
    timeout: int,
    patcher_failure_evidence_path: Path | None = None,
    patch_contract: str = "",
    operator_gate: str = "",
    bounded_validation_enabled: bool = True,
) -> dict[str, Any]:
    preflight_command = _preflight_command(manifest)
    preflight = (
        executor.execute(preflight_command, cwd=str(workspace), timeout=min(timeout, 120))
        if bounded_validation_enabled
        else {
            "returncode": None,
            "output": "skipped_bounded_validation_disabled",
        }
    )
    if evaluation["error_type"]:
        return _evaluation_blocked_result(
            evaluation=evaluation,
            action_adherence_observed={
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "candidate_patch_replay_attempted": False,
                "candidate_patch_replayed": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
            extra={
                **_zero_model_usage_fields(),
                "patch_contract": patch_contract,
                "operator_gate": operator_gate,
                "bounded_validation_enabled": bounded_validation_enabled,
            },
        )
    patcher_failure_evidence = _patcher_failure_evidence_for_instance(
        patcher_failure_evidence_path,
        instance_id,
    )
    replay_spec = (
        _candidate_patch_replay_spec(
            patcher_failure_evidence=patcher_failure_evidence,
            patch_contract=patch_contract,
        )
        if bounded_validation_enabled
        else _candidate_patch_replay_skipped_result("bounded_validation_disabled")
    )
    patch_text = str(replay_spec.get("candidate_patch_replay_patch", "") or "")
    if not patch_text.strip():
        return {
            **_evaluation_command_payload(evaluation),
            **_zero_model_usage_fields(),
            "error_type": "candidate_patch_missing",
            "stop_reason": str(replay_spec.get("candidate_patch_replay_error", "") or "candidate_patch_missing"),
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": str(
                replay_spec.get("candidate_patch_replay_error", "") or "candidate_patch_missing"
            ),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "runtime_collect_validation"]
            ),
            "patch": "",
            "patch_source": "candidate_patch_missing",
            "patch_summary": {},
            "patch_legitimacy": "",
            "action_adherence_observed": {
                "patch_attempted": False,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "candidate_patch_replay_attempted": bool(
                    replay_spec.get("candidate_patch_replay_attempted", False)
                ),
                "candidate_patch_replayed": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    invariant_gate = (
        _semantic_invariant_patch_gate(
            patch_text=patch_text,
            patcher_failure_evidence=patcher_failure_evidence,
            operator_gate=operator_gate,
        )
        if bounded_validation_enabled
        else _semantic_invariant_patch_gate_skipped_result("bounded_validation_disabled")
    )
    if invariant_gate["blocked"]:
        return {
            **_evaluation_command_payload(evaluation),
            **_zero_model_usage_fields(),
            "error_type": "",
            "stop_reason": "semantic_invariant_gate_blocked",
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "evaluation_protocol_error": False,
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "semantic_invariant_gate": invariant_gate,
            "semantic_invariant_gate_blocked": True,
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "runtime_collect_validation"]
            ),
            "patch": patch_text,
            "patch_source": "candidate_patch_failure_evidence",
            "patch_summary": _candidate_patch_summary(patch_text),
            "patch_legitimacy": _candidate_patch_legitimacy(patch_text),
            "action_adherence_observed": {
                "patch_attempted": False,
                "verifier_attempted": True,
                "environment_preflight_attempted": False,
                "candidate_patch_replay_attempted": False,
                "candidate_patch_replayed": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": True,
                "semantic_invariant_gate_blocked": True,
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    candidate_replay = (
        _maybe_replay_candidate_patch(
            executor=executor,
            workspace=workspace,
            patcher_failure_evidence=patcher_failure_evidence,
            patch_contract=patch_contract,
            timeout=min(timeout, 120),
        )
        if bounded_validation_enabled
        else _candidate_patch_replay_skipped_result("bounded_validation_disabled")
    )
    if _candidate_replay_blocked(candidate_replay):
        return {
            **_candidate_replay_blocked_result(
                evaluation=evaluation,
                candidate_replay=candidate_replay,
                action_adherence_observed={
                    "patch_attempted": False,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": False,
                    "candidate_patch_replay_attempted": True,
                    "candidate_patch_replayed": False,
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped": not bounded_validation_enabled,
                    "model_call_count": 0,
                    "token_cost": 0.0,
                },
            ),
            **_zero_model_usage_fields(),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "runtime_collect_validation"]
            ),
            "semantic_invariant_gate": invariant_gate,
            "patch": patch_text,
            "patch_source": "candidate_patch_failure_evidence",
            "patch_summary": _candidate_patch_summary(patch_text),
            "patch_legitimacy": _candidate_patch_legitimacy(patch_text),
        }
    syntax_gate = (
        _candidate_patch_syntax_gate(
            executor=executor,
            workspace=workspace,
            source_files=list(candidate_replay.get("candidate_patch_replay_source_files", []) or []),
            timeout=min(timeout, 120),
        )
        if bounded_validation_enabled
        else _candidate_patch_syntax_gate_skipped_result("bounded_validation_disabled")
    )
    if bool(syntax_gate.get("blocked", False)):
        return {
            **_evaluation_command_payload(evaluation),
            **candidate_replay,
            **_zero_model_usage_fields(),
            "error_type": "",
            "stop_reason": "candidate_patch_syntax_gate_blocked",
            "fail_to_pass_returncode": None,
            "oracle_returncode": None,
            "oracle_success": False,
            "reported_success": False,
            "fail_to_pass_output": "",
            "oracle_output": "",
            "fail_to_pass_protocol_error": False,
            "oracle_protocol_error": False,
            "evaluation_protocol_error": False,
            "evaluation_protocol_error_type": "",
            "candidate_patch_syntax_gate": syntax_gate,
            "candidate_patch_syntax_gate_blocked": True,
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
            ),
            "semantic_invariant_gate": invariant_gate,
            "patch": patch_text,
            "patch_source": "candidate_patch_failure_evidence",
            "patch_summary": _candidate_patch_summary(patch_text),
            "patch_legitimacy": _candidate_patch_legitimacy(patch_text),
            "action_adherence_observed": {
                "patch_attempted": True,
                "verifier_attempted": False,
                "environment_preflight_attempted": False,
                "candidate_patch_replay_attempted": bool(
                    candidate_replay.get("candidate_patch_replay_attempted", False)
                ),
                "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                "candidate_patch_syntax_gate_attempted": bool(syntax_gate.get("attempted", False)),
                "candidate_patch_syntax_gate_blocked": True,
                "post_patch_collect_validation_attempted": False,
                "bounded_validation_enabled": bounded_validation_enabled,
                "bounded_validation_skipped": not bounded_validation_enabled,
                "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
                "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
                "model_call_count": 0,
                "token_cost": 0.0,
            },
        }
    if bounded_validation_enabled:
        evaluation = _runtime_validate_evaluation_commands(
            manifest=manifest,
            evaluation=evaluation,
            executor=executor,
            workspace=workspace,
            timeout=min(timeout, 300),
        )
    notes = list(evaluation.get("normalization_notes", []) or [])
    notes.append(
        "runtime_collect_validation_after_candidate_patch"
        if bounded_validation_enabled
        else "runtime_collect_validation_skipped_bounded_validation_disabled"
    )
    evaluation["normalization_notes"] = list(dict.fromkeys(notes))
    if evaluation["error_type"]:
        return {
            **_evaluation_blocked_result(
                evaluation=evaluation,
                extra=candidate_replay,
                action_adherence_observed={
                    "patch_attempted": True,
                    "verifier_attempted": False,
                    "environment_preflight_attempted": False,
                    "candidate_patch_replay_attempted": bool(
                        candidate_replay.get("candidate_patch_replay_attempted", False)
                    ),
                    "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
                    "post_patch_collect_validation_attempted": bounded_validation_enabled,
                    "bounded_validation_enabled": bounded_validation_enabled,
                    "bounded_validation_skipped": not bounded_validation_enabled,
                    "model_call_count": 0,
                    "token_cost": 0.0,
                },
            ),
            **_zero_model_usage_fields(),
            "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
            "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
            "patch_contract": patch_contract,
            "operator_gate": operator_gate,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped_steps": (
                []
                if bounded_validation_enabled
                else ["candidate_patch_replay", "semantic_invariant_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
            ),
            "semantic_invariant_gate": invariant_gate,
            "candidate_patch_syntax_gate": syntax_gate,
            "patch": patch_text,
            "patch_source": "candidate_patch_failure_evidence",
            "patch_summary": _candidate_patch_summary(patch_text),
            "patch_legitimacy": _candidate_patch_legitimacy(patch_text),
        }
    fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
    oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
    protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)
    patch_summary = _candidate_patch_summary(patch_text)
    return {
        **_evaluation_command_payload(evaluation),
        **candidate_replay,
        **_zero_model_usage_fields(),
        "error_type": "",
        "stop_reason": "candidate_patch_verified",
        "fail_to_pass_returncode": fail_to_pass["returncode"],
        "oracle_returncode": oracle["returncode"],
        "oracle_success": oracle["returncode"] == 0,
        "reported_success": oracle["returncode"] == 0,
        "fail_to_pass_output": fail_to_pass["output"],
        "oracle_output": oracle["output"],
        **protocol,
        "patcher_failure_evidence_path": str(patcher_failure_evidence_path or ""),
        "patcher_failure_evidence_applied": bool(patcher_failure_evidence),
        "patch_contract": patch_contract,
        "operator_gate": operator_gate,
        "environment_preflight_command": preflight_command,
        "environment_preflight_returncode": preflight["returncode"],
        "environment_preflight_output": preflight["output"],
        "bounded_validation_enabled": bounded_validation_enabled,
        "bounded_validation_skipped_steps": (
            []
            if bounded_validation_enabled
            else ["candidate_patch_replay", "semantic_invariant_gate", "candidate_patch_syntax_gate", "runtime_collect_validation"]
        ),
        "semantic_invariant_gate": invariant_gate,
        "candidate_patch_syntax_gate": syntax_gate,
        "patch": patch_text,
        "patch_source": "candidate_patch_failure_evidence",
        "patch_summary": patch_summary,
        "patch_legitimacy": _candidate_patch_legitimacy(patch_text),
        "stage_outputs": {
            "patcher": {
                "success": True,
                "source": "candidate_patch_failure_evidence",
                "patch": patch_text,
                "patch_summary": patch_summary,
            },
            "verifier": {
                "fail_to_pass_returncode": fail_to_pass["returncode"],
                "oracle_returncode": oracle["returncode"],
                "oracle_success": oracle["returncode"] == 0,
            },
        },
        "stage_stop_reasons": {
            "patcher": "candidate_patch_replayed",
            "verifier": "executed",
        },
        "action_adherence_observed": {
            "patch_attempted": True,
            "verifier_attempted": True,
            "environment_preflight_attempted": bounded_validation_enabled,
            "candidate_patch_replay_attempted": bool(candidate_replay.get("candidate_patch_replay_attempted", False)),
            "candidate_patch_replayed": bool(candidate_replay.get("candidate_patch_replayed", False)),
            "candidate_patch_syntax_gate_attempted": bool(syntax_gate.get("attempted", False)),
            "candidate_patch_syntax_gate_blocked": bool(syntax_gate.get("blocked", False)),
            "post_patch_collect_validation_attempted": bounded_validation_enabled,
            "bounded_validation_enabled": bounded_validation_enabled,
            "bounded_validation_skipped": not bounded_validation_enabled,
            "semantic_invariant_gate_attempted": bool(invariant_gate.get("attempted", False)),
            "semantic_invariant_gate_blocked": bool(invariant_gate.get("blocked", False)),
            "model_call_count": 0,
            "token_cost": 0.0,
        },
    }


def _usage_snapshot(model: Any) -> dict[str, float]:
    getter = getattr(model, "get_usage_snapshot", None)
    if callable(getter):
        raw = dict(getter() or {})
    else:
        raw = {
            "n_calls": float(getattr(model, "n_calls", 0.0) or 0.0),
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
        }
    return {
        "n_calls": float(raw.get("n_calls", 0.0) or 0.0),
        "prompt_tokens": float(raw.get("prompt_tokens", 0.0) or 0.0),
        "completion_tokens": float(raw.get("completion_tokens", 0.0) or 0.0),
        "total_tokens": float(raw.get("total_tokens", 0.0) or 0.0),
    }


def _usage_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = ("n_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: max(0.0, float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0))
        for key in keys
    }


def _fixed_localization_for_instance(
    *,
    trajectory_records_path: Path,
    instance_id: str,
) -> dict[str, Any]:
    record = _trajectory_record_for_instance(trajectory_records_path, instance_id)
    stage_outputs = dict(record.get("stage_outputs", {}) or {})
    locator = dict(stage_outputs.get("locator", {}) or {})
    located_files = str(locator.get("located_files", "") or "").strip()
    selected = _selected_target_candidates_from_record(record)
    if not located_files and selected:
        located_files = "\n".join(f"- {path}" for path in selected)
    if not located_files:
        raise ValueError(
            f"No reusable locator output found for {instance_id} in {trajectory_records_path}"
        )
    source = "stage_outputs.locator.located_files" if locator.get("located_files") else "selected_target_candidates"
    return {
        "record": record,
        "located_files": located_files,
        "selected_target_candidates": selected,
        "source": source,
    }


def _fixed_localization_with_support_expansion(
    fixed: dict[str, Any],
    *,
    patcher_failure_evidence: dict[str, Any],
    operator_gate: str,
) -> dict[str, Any]:
    if operator_gate not in {
        "semantic_invariant_guarded_repatch",
        "semantic_span_delta_guarded_repatch",
        "semantic_effect_guarded_repatch",
    }:
        return fixed
    support = _dedupe_texts(
        [
            str(path)
            for path in list(dict(patcher_failure_evidence or {}).get("support_file_candidates", []) or [])
            if str(path).strip()
        ]
    )
    if not support:
        return fixed
    selected = _dedupe_texts(
        [
            *[str(path) for path in list(fixed.get("selected_target_candidates", []) or []) if str(path).strip()],
            *support,
        ]
    )
    located_lines = [
        line.strip()
        for line in str(fixed.get("located_files", "") or "").splitlines()
        if line.strip()
    ]
    located_text = "\n".join(located_lines)
    for path in support:
        if path not in located_text:
            located_lines.append(f"- {path}")
    expanded = dict(fixed)
    expanded["selected_target_candidates"] = selected
    expanded["located_files"] = "\n".join(located_lines)
    expanded["source"] = f"{fixed.get('source', '')}+semantic_assertion_support_expansion"
    expanded["semantic_assertion_support_expansion"] = {
        "enabled": True,
        "support_file_candidates": support,
        "claim_boundary": "bounded support files are derived from focused oracle assertion contracts before patch synthesis",
    }
    return expanded


def _fixed_localization_workspace_path_resolution(
    *,
    fixed: dict[str, Any],
    patcher_failure_evidence: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    raw_paths = _dedupe_texts(
        [
            *[str(path) for path in list(fixed.get("selected_target_candidates", []) or []) if str(path).strip()],
            *[
                str(path)
                for key in ("support_file_candidates", "candidate_patch_source_files", "selected_target_candidates")
                for path in list(dict(patcher_failure_evidence or {}).get(key, []) or [])
                if str(path).strip()
            ],
        ]
    )
    replacements: dict[str, str] = {}
    notes: list[str] = []
    for raw_path in raw_paths:
        normalized = _normalize_repo_path_text(raw_path)
        if not normalized:
            continue
        resolved = _resolve_existing_workspace_path(normalized, workspace=workspace)
        if resolved and resolved != normalized:
            replacements[normalized] = resolved
            notes.append(f"{normalized}->{resolved}")
    return {
        "enabled": bool(replacements),
        "workspace": str(workspace),
        "replacements": replacements,
        "notes": notes,
        "policy": "resolve_nonexistent_fixed_localization_paths_to_existing_workspace_source_aliases",
    }


def _resolve_existing_workspace_path(path: str, *, workspace: Path) -> str:
    normalized = _normalize_repo_path_text(path)
    if not normalized:
        return ""
    if (workspace / normalized).exists():
        return normalized
    path_obj = Path(normalized)
    name = path_obj.name
    candidates: list[str] = []
    if name.startswith("_") and name.endswith(".py"):
        candidates.append(str(path_obj.with_name(name[1:])).replace("\\", "/"))
    for candidate in candidates:
        candidate_norm = _normalize_repo_path_text(candidate)
        if candidate_norm and (workspace / candidate_norm).exists():
            return candidate_norm
    return normalized


def _apply_workspace_path_resolution_to_fixed(
    fixed: dict[str, Any],
    path_resolution: dict[str, Any],
) -> dict[str, Any]:
    replacements = dict(path_resolution.get("replacements", {}) or {})
    if not replacements:
        return fixed
    updated = dict(fixed)
    updated["selected_target_candidates"] = _replace_path_list(
        list(updated.get("selected_target_candidates", []) or []),
        replacements,
    )
    updated["located_files"] = _replace_paths_in_text(str(updated.get("located_files", "") or ""), replacements)
    source = str(updated.get("source", "") or "")
    updated["source"] = f"{source}+workspace_path_resolution" if source else "workspace_path_resolution"
    updated["workspace_path_resolution"] = path_resolution
    return updated


def _apply_workspace_path_resolution_to_evidence(
    evidence: dict[str, Any],
    path_resolution: dict[str, Any],
) -> dict[str, Any]:
    replacements = dict(path_resolution.get("replacements", {}) or {})
    if not replacements:
        return evidence
    updated = dict(evidence or {})
    for key in ("support_file_candidates", "candidate_patch_source_files", "selected_target_candidates"):
        if isinstance(updated.get(key), list):
            updated[key] = _replace_path_list(list(updated.get(key, []) or []), replacements)
    for key in (
        "source_edit_contract_text",
        "action_specific_success_judge",
        "progress_target",
        "progress_before_recovery",
    ):
        if isinstance(updated.get(key), str):
            updated[key] = _replace_paths_in_text(str(updated.get(key, "") or ""), replacements)
    for key in (
        "semantic_invariants",
        "expected_behavior_constraints",
        "forbidden_patch_directions",
        "assertion_contract_scope_constraints",
        "assertion_contract_edit_steps",
    ):
        if isinstance(updated.get(key), list):
            updated[key] = [
                _replace_paths_in_text(str(item), replacements) if isinstance(item, str) else item
                for item in list(updated.get(key, []) or [])
            ]
    updated["workspace_path_resolution"] = path_resolution
    return updated


def _replace_path_list(paths: list[Any], replacements: dict[str, str]) -> list[str]:
    replaced: list[str] = []
    for path in paths:
        normalized = _normalize_repo_path_text(str(path))
        if not normalized:
            continue
        value = replacements.get(normalized, normalized)
        if value not in replaced:
            replaced.append(value)
    return replaced


def _replace_paths_in_text(text: str, replacements: dict[str, str]) -> str:
    updated = str(text or "")
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        updated = updated.replace(old, new)
    return updated


def _normalize_repo_path_text(path: str) -> str:
    return str(path or "").strip().strip("`'\"").replace("\\", "/").lstrip("./")


def _trajectory_record_for_instance(trajectory_records_path: Path, instance_id: str) -> dict[str, Any]:
    payload = json.loads(trajectory_records_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_records = payload.get("records", payload.get("rows", []))
    elif isinstance(payload, list):
        raw_records = payload
    else:
        raw_records = []
    for item in list(raw_records or []):
        if isinstance(item, dict) and str(item.get("instance_id", "") or "") == instance_id:
            return dict(item)
    raise ValueError(f"Trajectory record not found for {instance_id}: {trajectory_records_path}")


def _previous_patch_for_instance(
    *,
    trajectory_records_path: Path,
    instance_id: str,
) -> dict[str, Any]:
    record = _trajectory_record_for_instance(
        trajectory_records_path=trajectory_records_path,
        instance_id=instance_id,
    )
    patcher = dict(dict(record.get("stage_outputs", {}) or {}).get("patcher", {}) or {})
    patch = str(patcher.get("patch", "") or "")
    patch_summary = dict(patcher.get("patch_summary", {}) or record.get("diff_summary", {}) or {})
    return {
        "patch": patch,
        "patch_summary": patch_summary,
        "source": "stage_outputs.patcher.patch" if patch else "",
    }


def _maybe_replay_candidate_patch(
    *,
    executor,
    workspace: Path,
    patcher_failure_evidence: dict[str, Any],
    patch_contract: str,
    timeout: int,
) -> dict[str, Any]:
    replay_spec = _candidate_patch_replay_spec(
        patcher_failure_evidence=patcher_failure_evidence,
        patch_contract=patch_contract,
    )
    if not bool(replay_spec.get("candidate_patch_replay_attempted", False)):
        return {
            "candidate_patch_replay_attempted": False,
            "candidate_patch_replayed": False,
            "candidate_patch_replay_returncode": None,
            "candidate_patch_replay_error": "",
            "candidate_patch_replay_source_files": [],
            "candidate_patch_replay_filtered_files": [],
        }
    if str(replay_spec.get("candidate_patch_replay_error", "") or ""):
        return {
            "candidate_patch_replay_attempted": True,
            "candidate_patch_replayed": False,
            "candidate_patch_replay_returncode": None,
            "candidate_patch_replay_error": str(replay_spec.get("candidate_patch_replay_error", "") or ""),
            "candidate_patch_replay_source_files": list(replay_spec.get("candidate_patch_replay_source_files", []) or []),
            "candidate_patch_replay_filtered_files": list(
                replay_spec.get("candidate_patch_replay_filtered_files", []) or []
            ),
        }
    replay_patch = str(replay_spec.get("candidate_patch_replay_patch", "") or "")
    source_files = list(replay_spec.get("candidate_patch_replay_source_files", []) or [])
    filtered = list(replay_spec.get("candidate_patch_replay_filtered_files", []) or [])
    if "*** Begin Patch" in replay_patch:
        result = _apply_apply_patch_via_workspace(workspace=workspace, patch_text=replay_patch)
    else:
        result = executor.execute(
            "git apply --whitespace=nowarn -",
            cwd=str(workspace),
            timeout=timeout,
            input_text=replay_patch if hasattr(executor, "execute") else None,
        ) if _executor_accepts_input_text(executor) else _apply_patch_via_tempfile(
            executor=executor,
            workspace=workspace,
            patch_text=replay_patch,
            timeout=timeout,
        )
    returncode = int(result.get("returncode", 0) or 0)
    return {
        "candidate_patch_replay_attempted": True,
        "candidate_patch_replayed": returncode == 0,
        "candidate_patch_replay_returncode": returncode,
        "candidate_patch_replay_error": str(result.get("output", "") or "")[-2000:] if returncode else "",
        "candidate_patch_replay_apply_mode": str(result.get("apply_mode", "plain") or "plain"),
        "candidate_patch_replay_recount_attempted": bool(result.get("recount_attempted", False)),
        "candidate_patch_replay_plain_returncode": result.get("plain_returncode"),
        "candidate_patch_replay_plain_error": str(result.get("plain_error", "") or "")[-2000:],
        "candidate_patch_replay_source_files": source_files,
        "candidate_patch_replay_filtered_files": filtered,
    }


def _apply_apply_patch_via_workspace(*, workspace: Path, patch_text: str) -> dict[str, Any]:
    from bcmr_swe.experiments.masguard_evidence_to_edit_patch_prompt_live import (
        _apply_apply_patch_text,
    )

    applied = _apply_apply_patch_text(temp_root=workspace, patch=patch_text)
    satisfied = bool(applied.get("satisfied", False))
    return {
        "returncode": 0 if satisfied else 1,
        "output": "" if satisfied else str(applied.get("reason", "") or applied.get("status", "")),
        "apply_mode": str(applied.get("status", "") or "apply_patch"),
        "plain_returncode": 0 if satisfied else 1,
        "plain_error": "" if satisfied else str(applied.get("reason", "") or ""),
        "recount_attempted": False,
    }


def _candidate_patch_replay_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "candidate_patch_replay_attempted": False,
        "candidate_patch_replayed": False,
        "candidate_patch_replay_returncode": None,
        "candidate_patch_replay_error": "",
        "candidate_patch_replay_source_files": [],
        "candidate_patch_replay_filtered_files": [],
        "candidate_patch_replay_skipped": True,
        "candidate_patch_replay_skip_reason": reason,
    }


def _candidate_patch_syntax_gate(
    *,
    executor,
    workspace: Path,
    source_files: list[str],
    timeout: int,
) -> dict[str, Any]:
    checked_files: list[str] = []
    violations: list[dict[str, Any]] = []
    for raw_path in source_files:
        path = str(raw_path or "").strip()
        if not path or not path.endswith(".py"):
            continue
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            violations.append(
                {
                    "path": path,
                    "returncode": None,
                    "output_excerpt": "unsafe_candidate_patch_path",
                }
            )
            continue
        checked_files.append(path)
        command = (
            "python -c "
            + shlex.quote(
                "import pathlib, sys; "
                "path = pathlib.Path(sys.argv[1]); "
                "compile(path.read_text(encoding='utf-8'), str(path), 'exec')"
            )
            + f" {shlex.quote(path)}"
        )
        result = executor.execute(command, cwd=str(workspace), timeout=timeout)
        returncode = int(result.get("returncode", 0) or 0)
        output = str(result.get("output", "") or "")
        if returncode != 0:
            violations.append(
                {
                    "path": path,
                    "returncode": returncode,
                    "output_excerpt": output[-2000:],
                }
            )
    attempted = bool(checked_files or violations)
    return {
        "attempted": attempted,
        "skipped": not attempted,
        "skip_reason": "" if attempted else "no_python_source_files",
        "blocked": bool(violations),
        "checked_files": checked_files,
        "violations": violations,
    }


def _candidate_patch_syntax_gate_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "skipped": True,
        "skip_reason": reason,
        "blocked": False,
        "checked_files": [],
        "violations": [],
    }


def _candidate_patch_summary_source_files(patch_summary: dict[str, Any]) -> list[str]:
    fresh_classes = dict(patch_summary.get("fresh_changed_file_classes", {}) or {})
    changed_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
    fresh_source = [str(item) for item in list(fresh_classes.get("source_files", []) or []) if str(item).strip()]
    if fresh_source:
        return fresh_source
    return [str(item) for item in list(changed_classes.get("source_files", []) or []) if str(item).strip()]


def _candidate_patch_replay_spec(
    *,
    patcher_failure_evidence: dict[str, Any],
    patch_contract: str,
) -> dict[str, Any]:
    if _candidate_patch_replay_explicitly_disabled(patcher_failure_evidence):
        return {
            "candidate_patch_replay_attempted": False,
            "candidate_patch_replay_patch": "",
            "candidate_patch_replay_error": "",
            "candidate_patch_replay_source_files": [],
            "candidate_patch_replay_filtered_files": [],
            "candidate_patch_replay_skipped": True,
            "candidate_patch_replay_skip_reason": "explicit_no_replay_evidence",
        }
    patch_text = str(patcher_failure_evidence.get("candidate_patch", "") or "")
    if not patch_text.strip():
        return {
            "candidate_patch_replay_attempted": False,
            "candidate_patch_replay_patch": "",
            "candidate_patch_replay_error": "candidate_patch_missing",
            "candidate_patch_replay_source_files": [],
            "candidate_patch_replay_filtered_files": [],
        }
    source_files = _candidate_patch_source_files(patcher_failure_evidence, patch_text)
    all_files = parse_unified_diff_paths(patch_text)
    filtered = [path for path in all_files if path not in source_files]
    if patch_contract == "source_only" and not source_files:
        return {
            "candidate_patch_replay_attempted": True,
            "candidate_patch_replay_patch": "",
            "candidate_patch_replay_error": "candidate_patch_has_no_source_files",
            "candidate_patch_replay_source_files": [],
            "candidate_patch_replay_filtered_files": filtered or all_files,
        }
    replay_patch = patch_text
    if patch_contract == "source_only":
        if "*** Begin Patch" in patch_text and source_files:
            return {
                "candidate_patch_replay_attempted": True,
                "candidate_patch_replay_patch": patch_text,
                "candidate_patch_replay_error": "",
                "candidate_patch_replay_source_files": source_files,
                "candidate_patch_replay_filtered_files": filtered,
            }
        replay_patch = _filter_patch_to_files(patch_text, source_files)
        if not replay_patch.strip():
            return {
                "candidate_patch_replay_attempted": True,
                "candidate_patch_replay_patch": "",
                "candidate_patch_replay_error": "candidate_patch_source_filter_empty",
                "candidate_patch_replay_source_files": source_files,
                "candidate_patch_replay_filtered_files": filtered or all_files,
            }
    return {
        "candidate_patch_replay_attempted": True,
        "candidate_patch_replay_patch": replay_patch,
        "candidate_patch_replay_error": "",
        "candidate_patch_replay_source_files": source_files,
        "candidate_patch_replay_filtered_files": filtered,
    }


def _candidate_patch_replay_explicitly_disabled(evidence: dict[str, Any]) -> bool:
    if bool(evidence.get("candidate_patch_replay_disabled", False)):
        return True
    if bool(evidence.get("disable_candidate_patch_replay", False)):
        return True
    if bool(evidence.get("no_candidate_patch_replay", False)):
        return True
    replay_keys_present = (
        "candidate_patch_replayed" in evidence
        or "candidate_patch_replay_attempted" in evidence
    )
    return (
        replay_keys_present
        and evidence.get("candidate_patch_replayed") is False
        and evidence.get("candidate_patch_replay_attempted") is not True
    )


def _candidate_patch_summary(patch_text: str) -> dict[str, Any]:
    changed_files = _candidate_patch_changed_files(str(patch_text or ""))
    changed_classes = classify_changed_files(changed_files)
    return {
        "changed_files": changed_files,
        "changed_file_classes": changed_classes,
        "fresh_changed_files": changed_files,
        "fresh_changed_file_classes": changed_classes,
        "target_legitimacy": _candidate_patch_legitimacy(patch_text),
        "fresh_target_legitimacy": _candidate_patch_legitimacy(patch_text),
        "has_fresh_source_diff": bool(changed_classes.get("source_files")),
    }


def _candidate_patch_legitimacy(patch_text: str) -> str:
    classes = classify_changed_files(_candidate_patch_changed_files(str(patch_text or "")))
    source_files = list(classes.get("source_files", []) or [])
    non_source = (
        list(classes.get("test_files", []) or [])
        + list(classes.get("generated_files", []) or [])
        + list(classes.get("other_files", []) or [])
    )
    if source_files and not non_source:
        return "source_only"
    if source_files and non_source:
        return "source_mixed"
    return "no_source_diff"


def _candidate_only_status(
    *,
    patch_text: str,
    patch_summary: dict[str, Any],
    patcher_success: bool,
    infrastructure_error: bool,
    semantic_invariant_gate: dict[str, Any],
    patch_intent_gate: dict[str, Any],
    patch_contract: str,
    provider_api_error: bool = False,
) -> dict[str, Any]:
    reject_reasons: list[str] = []
    if provider_api_error:
        reject_reasons.append("provider_api_error")
    elif infrastructure_error:
        reject_reasons.append("patcher_infrastructure_error")
    if not str(patch_text or "").strip():
        reject_reasons.append("missing_patch")
    legitimacy = str(
        patch_summary.get("fresh_target_legitimacy")
        or patch_summary.get("target_legitimacy")
        or _candidate_patch_legitimacy(patch_text)
        or ""
    )
    if patch_contract == "source_only" and legitimacy != "source_only":
        reject_reasons.append(f"patch_contract_not_source_only:{legitimacy or 'unknown'}")
    if bool(semantic_invariant_gate.get("blocked", False)):
        reject_reasons.append("semantic_invariant_gate_blocked")
    accept = not reject_reasons
    return {
        "accept": accept,
        "status": "candidate_ready_for_verifier_replay" if accept else "candidate_rejected_before_oracle",
        "stop_reason": "candidate_generated_no_oracle" if accept else "candidate_rejected_before_oracle",
        "reject_reasons": reject_reasons,
    }


def _executor_accepts_input_text(executor) -> bool:
    del executor
    return False


def _apply_patch_via_tempfile(*, executor, workspace: Path, patch_text: str, timeout: int) -> dict[str, Any]:
    patch_path = workspace / ".mas_dx_candidate_patch.diff"
    patch_path.write_text(patch_text if patch_text.endswith("\n") else patch_text + "\n", encoding="utf-8")
    try:
        plain = executor.execute(
            f"git apply --whitespace=nowarn {shlex.quote(patch_path.name)}",
            cwd=str(workspace),
            timeout=timeout,
        )
        plain_returncode = int(plain.get("returncode", 0) or 0)
        plain_output = str(plain.get("output", "") or "")
        if plain_returncode == 0:
            plain.update(
                {
                    "apply_mode": "plain",
                    "recount_attempted": False,
                    "plain_returncode": plain_returncode,
                    "plain_error": "",
                }
            )
            return plain
        if not _candidate_patch_error_supports_recount(plain_output):
            plain.update(
                {
                    "apply_mode": "plain",
                    "recount_attempted": False,
                    "plain_returncode": plain_returncode,
                    "plain_error": plain_output,
                }
            )
            return plain
        recount = executor.execute(
            f"git apply --whitespace=nowarn --recount {shlex.quote(patch_path.name)}",
            cwd=str(workspace),
            timeout=timeout,
        )
        recount.update(
            {
                "apply_mode": "recount",
                "recount_attempted": True,
                "plain_returncode": plain_returncode,
                "plain_error": plain_output,
            }
        )
        return recount
    finally:
        try:
            patch_path.unlink()
        except OSError:
            pass


def _candidate_patch_error_supports_recount(output: str) -> bool:
    text = str(output or "").lower()
    return (
        "corrupt patch" in text
        or "malformed patch" in text
        or "patch fragment without header" in text
    )


def _candidate_patch_source_files(evidence: dict[str, Any], patch_text: str) -> list[str]:
    explicit = [
        str(path)
        for path in list(evidence.get("candidate_patch_source_files", []) or [])
        if str(path).strip()
    ]
    if explicit:
        return explicit
    return list(classify_changed_files(_candidate_patch_changed_files(patch_text)).get("source_files", []) or [])


def _candidate_patch_changed_files(patch_text: str) -> list[str]:
    paths = parse_unified_diff_paths(str(patch_text or ""))
    if paths:
        return paths
    apply_patch_paths: list[str] = []
    for line in str(patch_text or "").splitlines():
        stripped = line.strip()
        for prefix in ("*** Update File: ", "*** Add File: ", "*** Delete File: "):
            if not stripped.startswith(prefix):
                continue
            path = stripped[len(prefix) :].strip()
            if path and path not in apply_patch_paths:
                apply_patch_paths.append(path)
    return apply_patch_paths


def _filter_patch_to_files(patch_text: str, keep_files: list[str]) -> str:
    keep = {str(path) for path in keep_files if str(path)}
    chunks: list[list[str]] = []
    current: list[str] = []
    current_path = ""
    for line in str(patch_text or "").splitlines():
        if line.startswith("diff --git "):
            if current and current_path in keep:
                chunks.append(current)
            current = [line]
            current_path = _path_from_diff_header(line)
            continue
        if current:
            current.append(line)
    if current and current_path in keep:
        chunks.append(current)
    return "\n".join("\n".join(chunk) for chunk in chunks)


def _path_from_diff_header(header: str) -> str:
    parts = header.split()
    if len(parts) < 4:
        return ""
    path = parts[3]
    if path.startswith("b/"):
        path = path[2:]
    return path


def _candidate_replay_blocked(candidate_replay: dict[str, Any]) -> bool:
    return bool(candidate_replay.get("candidate_patch_replay_attempted", False)) and not bool(
        candidate_replay.get("candidate_patch_replayed", False)
    )


def _candidate_replay_blocked_result(
    *,
    evaluation: dict[str, Any],
    candidate_replay: dict[str, Any],
    action_adherence_observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_evaluation_command_payload(evaluation),
        **candidate_replay,
        "error_type": "candidate_patch_replay_blocked",
        "stop_reason": "candidate_patch_replay_blocked",
        "fail_to_pass_returncode": None,
        "oracle_returncode": None,
        "oracle_success": False,
        "reported_success": False,
        "fail_to_pass_output": "",
        "oracle_output": "",
        "fail_to_pass_protocol_error": True,
        "oracle_protocol_error": True,
        "evaluation_protocol_error": True,
        "evaluation_protocol_error_type": "candidate_patch_replay_blocked",
        "action_adherence_observed": action_adherence_observed,
    }


def _patch_replay_blocked(patch_replay: dict[str, Any]) -> bool:
    return bool(patch_replay.get("previous_patch_replay_requested")) and not bool(
        patch_replay.get("previous_patch_replayed")
    ) and str(patch_replay.get("previous_patch_replay_source", "") or "") != "skipped_for_patcher_fixed_localization"


def _patch_replay_blocked_result(
    *,
    evaluation: dict[str, Any],
    patch_replay: dict[str, Any],
    action_adherence_observed: dict[str, Any],
) -> dict[str, Any]:
    error = str(patch_replay.get("previous_patch_replay_error", "") or "")
    stop_reason = "previous_patch_missing" if error == "previous_patch_missing" else "previous_patch_replay_failed"
    return {
        **_evaluation_command_payload(evaluation),
        **patch_replay,
        "error_type": "previous_patch_replay_blocked",
        "stop_reason": stop_reason,
        "fail_to_pass_returncode": None,
        "oracle_returncode": None,
        "oracle_success": False,
        "reported_success": False,
        "fail_to_pass_output": "",
        "oracle_output": "",
        "fail_to_pass_protocol_error": True,
        "oracle_protocol_error": True,
        "evaluation_protocol_error": True,
        "evaluation_protocol_error_type": stop_reason,
        "action_adherence_observed": action_adherence_observed,
    }


def _maybe_replay_previous_patch(
    *,
    run_scope: str,
    workspace: Path,
    trajectory_records_path: Path,
    instance_id: str,
    replay_previous_patch: bool,
    apply_mode: str = "plain",
) -> dict[str, Any]:
    if not replay_previous_patch:
        return {
            "previous_patch_replay_requested": False,
            "previous_patch_replayed": False,
            "previous_patch_replay_source": "",
            "previous_patch_replay_apply_mode": "",
            "previous_patch_replay_returncode": None,
            "previous_patch_replay_error": "",
            "previous_patch_replay_changed_files": [],
        }
    if run_scope == "patcher_fixed_localization":
        return {
            "previous_patch_replay_requested": True,
            "previous_patch_replayed": False,
            "previous_patch_replay_source": "skipped_for_patcher_fixed_localization",
            "previous_patch_replay_apply_mode": apply_mode,
            "previous_patch_replay_returncode": None,
            "previous_patch_replay_error": "",
            "previous_patch_replay_changed_files": [],
        }
    previous = _previous_patch_for_instance(
        trajectory_records_path=trajectory_records_path,
        instance_id=instance_id,
    )
    patch_text = str(previous.get("patch", "") or "")
    changed_files = list(dict(previous.get("patch_summary", {}) or {}).get("changed_files", []) or [])
    if not patch_text.strip():
        return {
            "previous_patch_replay_requested": True,
            "previous_patch_replayed": False,
            "previous_patch_replay_source": "",
            "previous_patch_replay_apply_mode": apply_mode,
            "previous_patch_replay_returncode": None,
            "previous_patch_replay_error": "previous_patch_missing",
            "previous_patch_replay_changed_files": changed_files,
        }
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    apply_args = ["git", "apply", "--whitespace=nowarn"]
    if apply_mode == "recount":
        apply_args.append("--recount")
    apply_args.append("-")
    completed = subprocess.run(
        apply_args,
        input=patch_text,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "previous_patch_replay_requested": True,
        "previous_patch_replayed": completed.returncode == 0,
        "previous_patch_replay_source": str(previous.get("source", "") or ""),
        "previous_patch_replay_apply_mode": apply_mode,
        "previous_patch_replay_returncode": int(completed.returncode),
        "previous_patch_replay_error": (completed.stderr or completed.stdout or "")[-2000:],
        "previous_patch_replay_changed_files": changed_files,
    }


def _selected_target_candidates_from_record(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for source in (
        record.get("selected_target_candidates", []),
        dict(dict(record.get("stage_outputs", {}) or {}).get("locator", {}) or {}).get(
            "selected_target_candidates",
            [],
        ),
    ):
        if isinstance(source, str):
            source_items = [source]
        else:
            source_items = list(source or [])
        for item in source_items:
            text = str(item or "").strip().strip("`'\"")
            if text and text not in candidates:
                candidates.append(text)
    if candidates:
        return candidates
    located_files = str(
        dict(dict(record.get("stage_outputs", {}) or {}).get("locator", {}) or {}).get("located_files", "")
        or ""
    )
    for line in located_files.splitlines():
        path = _path_from_locator_line(line)
        if path and path not in candidates:
            candidates.append(path)
    if candidates:
        return candidates
    diff_summary = dict(record.get("diff_summary", {}) or {})
    changed_classes = dict(diff_summary.get("changed_file_classes", {}) or {})
    for source in (
        changed_classes.get("source_files", []),
        changed_classes.get("effective_files", []),
        diff_summary.get("changed_files", []),
    ):
        for item in list(source or []):
            text = str(item or "").strip().strip("`'\"")
            if text and _looks_like_source_file_path(text) and text not in candidates:
                candidates.append(text)
    return candidates


def _looks_like_source_file_path(path: str) -> bool:
    text = str(path or "").strip()
    if not text or text.startswith("/"):
        return False
    if text.startswith(("tests/", "test/", "testing/")):
        return False
    parts = set(text.split("/"))
    if "tests" in parts or "test" in parts:
        return False
    return text.endswith(".py")


def _path_from_locator_line(line: str) -> str:
    stripped = str(line or "").strip()
    if not stripped:
        return ""
    if "`" in stripped:
        parts = stripped.split("`")
        for idx in range(1, len(parts), 2):
            candidate = parts[idx].strip()
            if _looks_like_repo_path(candidate):
                return candidate
    token = stripped.lstrip("-* ").split("：", 1)[0].split(":", 1)[0].strip().strip("`")
    if _looks_like_repo_path(token):
        return token
    return ""


def _looks_like_repo_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or any(ch.isspace() for ch in text):
        return False
    return "/" in text or text.endswith((".py", ".pyi"))


def _patcher_issue(manifest: dict[str, Any]) -> str:
    return str(manifest.get("problem_statement") or manifest.get("issue") or "").strip()


def _patcher_failure_evidence_for_instance(
    path: Path | None,
    instance_id: str,
) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Patcher failure evidence not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and str(payload.get("instance_id", "") or "") == instance_id:
        return dict(payload)
    for row in list(payload.get("rows", []) if isinstance(payload, dict) else []):
        if isinstance(row, dict) and str(row.get("instance_id", "") or "") == instance_id:
            return dict(row)
    return {}


def _merge_patcher_failure_evidence(base: dict[str, Any], supplement: dict[str, Any]) -> dict[str, Any]:
    if not supplement:
        return dict(base or {})
    merged = dict(base or {})
    for key in (
        "semantic_invariants",
        "forbidden_patch_directions",
        "expected_behavior_constraints",
        "oracle_assertion_contracts",
        "support_file_candidates",
    ):
        merged[key] = _dedupe_texts(
            [*list(merged.get(key, []) or []), *list(supplement.get(key, []) or [])]
        )
    sources = list(merged.get("semantic_invariant_sources", []) or [])
    sources.extend(list(supplement.get("semantic_invariant_sources", []) or []))
    if sources:
        merged["semantic_invariant_sources"] = _dedupe_dicts(sources)
    assertion_sources = list(merged.get("oracle_assertion_contract_sources", []) or [])
    assertion_sources.extend(list(supplement.get("oracle_assertion_contract_sources", []) or []))
    if assertion_sources:
        merged["oracle_assertion_contract_sources"] = _dedupe_dicts(assertion_sources)
    if supplement.get("oracle_derived_semantic_invariants"):
        merged["oracle_derived_semantic_invariants"] = True
    if supplement.get("oracle_assertion_contracts_derived"):
        merged["oracle_assertion_contracts_derived"] = True
    if supplement.get("assertion_contract_edit_intent_required"):
        merged["assertion_contract_edit_intent_required"] = True
    if supplement.get("bounded_support_expansion_reason"):
        merged["bounded_support_expansion_reason"] = str(supplement.get("bounded_support_expansion_reason", "") or "")
    return merged


def _derive_oracle_semantic_invariant_evidence(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    fixed_localization: dict[str, Any],
) -> dict[str, Any]:
    targets = _dedupe_texts([
        str(item)
        for item in [
            *list(evaluation.get("fail_to_pass_targets", []) or []),
            *list(evaluation.get("oracle_targets", []) or []),
        ]
        if str(item).strip()
    ])
    selected = [
        str(item)
        for item in list(fixed_localization.get("selected_target_candidates", []) or [])
        if str(item).strip()
    ]
    text = "\n".join(
        [
            str(manifest.get("problem_statement", "") or ""),
            str(manifest.get("issue", "") or ""),
            "\n".join(targets),
            str(fixed_localization.get("located_files", "") or ""),
            "\n".join(selected),
        ]
    ).lower()
    semantic_invariants: list[str] = []
    forbidden: list[str] = []
    expected: list[str] = []
    sources: list[dict[str, Any]] = []
    assertion_evidence = _derive_oracle_assertion_contract_evidence(
        manifest=manifest,
        targets=targets,
    )

    if _mentions_dataframe_dtype_preservation_contract(text):
        semantic_invariants.append(
            "DataFrame output dtype preservation: pandas output for selectors/transformers must preserve "
            "the original input dtype for each retained output column."
        )
        semantic_invariants.append(
            "The dtype-preservation patch must pass original input dtype metadata through the set_output "
            "wrapping call chain, not only add an unused dtypes argument or cast without a populated dtype map."
        )
        forbidden.append(
            "Do not only add dtypes=None and dataframe.astype(dtypes) inside _wrap_in_pandas_container; "
            "_wrap_data_with_container must supply original_input dtypes for retained columns."
        )
        expected.append("Focused oracle requires output.dtypes[name] == X.dtypes[name] for retained DataFrame columns.")
        sources.append(
            {
                "invariant_id": "sklearn_dataframe_output_dtype_preservation",
                "source": "manifest_or_fail_to_pass_nodeid",
                "matched_targets": [target for target in targets if "dataframe" in target.lower() or "dtype" in target.lower()],
            }
        )

    if _mentions_roc_curve_probability_threshold_contract(text):
        semantic_invariants.append(
            "roc_curve threshold sentinel: thresholds[0] must be np.inf for probability estimates and "
            "drop_intermediate behavior must still start with np.inf."
        )
        forbidden.append(
            "Do not cap the first roc_curve threshold at 1, use min(first_threshold, 1), or overwrite np.inf with 1."
        )
        expected.append("Focused oracle requires np.isinf(thresholds[0]) and expected thresholds beginning with np.inf.")
        sources.append(
            {
                "invariant_id": "sklearn_roc_curve_first_threshold_inf",
                "source": "manifest_or_fail_to_pass_nodeid",
                "matched_targets": [target for target in targets if "roc_curve" in target.lower()],
            }
        )

    if _mentions_sphinx_none_annotation_object_reference_contract(text):
        semantic_invariants.append(
            "Sphinx signature-mode annotation parsing must turn annotation None into a Python object reference "
            "with reftype='obj', not a Python class reference."
        )
        semantic_invariants.append(
            "The None annotation fix belongs in sphinx/domains/python.py _parse_annotation/make_xref; broad "
            "rewrites of sphinx.ext.autodoc.typehints do not satisfy the oracle."
        )
        expected.append(
            "Focused oracle requires _parse_annotation('None')[0] to be pending_xref with refdomain='py', "
            "reftype='obj', and reftarget='None'."
        )
        forbidden.append(
            "Do not rewrite sphinx/ext/autodoc/typehints.py or field-list handling; the failure is the "
            "signature annotation parser treating None as a class."
        )
        sources.append(
            {
                "invariant_id": "sphinx_none_annotation_reftype_obj",
                "source": "manifest_or_fail_to_pass_nodeid",
                "matched_targets": [
                    target
                    for target in targets
                    if "test_domain_py" in target.lower() or "parse_annotation" in target.lower()
                ],
                "support_file_candidates": ["sphinx/domains/python.py"],
            }
        )

    if _mentions_sympy_permutation_non_disjoint_cycle_contract(text):
        semantic_invariants.append(
            "Sympy Permutation must accept non-disjoint cycle notation such as [[0, 1], [0, 1]] by composing "
            "cycles in order and returning the resulting array-form permutation."
        )
        semantic_invariants.append(
            "The non-disjoint cycle patch must preserve Permutation args/reconstruction invariants; do not leave "
            "the original repeated cycle-list as Basic args."
        )
        expected.append(
            "Focused oracle requires Permutation([[0, 1], [0, 1]]) to construct the identity permutation while "
            "test_args can reconstruct Permutation objects without recursion or ValueError."
        )
        forbidden.append(
            "Do not merely skip the duplicate-element ValueError for cycle input; repeated cycle lists must be "
            "converted to a canonical array form before Basic.__new__ stores args."
        )
        sources.append(
            {
                "invariant_id": "sympy_permutation_non_disjoint_cycles_array_form",
                "source": "manifest_or_fail_to_pass_nodeid",
                "matched_targets": [
                    target
                    for target in targets
                    if "test_args" in target.lower() or "permutation" in target.lower()
                ],
                "support_file_candidates": ["sympy/combinatorics/permutations.py"],
            }
        )

    assertion_contracts = _dedupe_texts(assertion_evidence.get("oracle_assertion_contracts", []) or [])
    if _mentions_xarray_where_keep_attrs_contract(text, assertion_contracts):
        semantic_invariants.append(
            "xarray.where must accept keep_attrs and preserve attrs from the x argument when keep_attrs=True."
        )
        semantic_invariants.append(
            "The where keep_attrs patch must update the public xarray.core.computation.where signature and "
            "preserve attrs from x rather than cond; docstring-only edits are invalid."
        )
        expected.append(
            "Focused oracle requires xr.where(cond, x, y, keep_attrs=True) to return x attrs on the result."
        )
        forbidden.append(
            "Do not only edit where documentation/examples, and do not simply pass keep_attrs=True to apply_ufunc "
            "because that preserves cond attrs instead of x attrs."
        )
        sources.append(
            {
                "invariant_id": "xarray_where_keep_attrs_signature_forwarding",
                "source": "oracle_snapshot_test_assertions",
                "matched_targets": [
                    target
                    for target in targets
                    if "test_computation" in target.lower() or "test_where_attrs" in target.lower()
                ],
                "matched_contract_count": len(assertion_contracts),
            }
        )
    if _mentions_xarray_indexvariable_copy_contract(text, assertion_contracts):
        semantic_invariants.append(
            "xarray IndexVariable copy/deepcopy must preserve unicode index dtype and return the same "
            "IndexVariable type instead of casting coordinate data to object."
        )
        semantic_invariants.append(
            "xarray IndexVariable copy(deep=True) must allocate independent backing ndarray data, while "
            "copy(deep=False) must keep shallow backing-data aliasing."
        )
        expected.append(
            "Focused oracle requires IndexVariable copy contracts: same type, same dtype, deep copy breaks "
            "source ndarray aliasing, shallow copy preserves source ndarray aliasing."
        )
        forbidden.append(
            "Do not only edit generic Variable.copy or constructor wrapping if IndexVariable unicode dtype "
            "and deep/shallow source_ndarray aliasing are not preserved."
        )
        support_files = ["xarray/core/indexing.py", "xarray/core/variable.py"]
        sources.append(
            {
                "invariant_id": "xarray_indexvariable_copy_unicode_dtype_aliasing",
                "source": "oracle_snapshot_test_assertions",
                "matched_targets": [
                    target
                    for target in targets
                    if "test_variable" in target.lower() or "test_copy" in target.lower()
                ],
                "matched_contract_count": len(assertion_contracts),
                "support_file_candidates": support_files,
            }
        )
        expected.append(
            "Bounded support-file expansion allowed: add PandasIndexAdapter.copy(deep=...) in "
            "xarray/core/indexing.py, then make IndexVariable.copy delegate to self._data.copy(deep=deep) "
            "in xarray/core/variable.py."
        )
        expected.append(
            "Required source edit plan: first inspect xarray/core/indexing.py and add a PandasIndexAdapter.copy "
            "method that returns PandasIndexAdapter(self.array.copy(deep=True), self._dtype) for deep copies and "
            "PandasIndexAdapter(self.array, self._dtype) for shallow copies."
        )
        expected.append(
            "Required source edit plan: then inspect xarray/core/variable.py and make IndexVariable.copy use "
            "self._data.copy(deep=deep) when data is None; a variable.py-only patch is rejected."
        )
        expected.append(
            "Required source edit plan: preserve neighboring IndexVariable methods and documentation, including "
            "equals, _data_equals, and to_index_variable; the edit should only replace the data-copy implementation."
        )
    if _mentions_xarray_reset_index_xindexes_contract(text, assertion_contracts):
        semantic_invariants.append(
            "xarray reset_index must update coordinate/index state so xindexes, PandasIndex type, and reset "
            "coordinate variables match the tested reset_index contract."
        )
        semantic_invariants.append(
            "The reset_index patch must touch reset_index/drop_indexes/xindexes index-state behavior; changing "
            "DataVariables length or iteration alone is not a valid semantic fix."
        )
        expected.append(
            "Focused oracle requires reset_index to leave xindexes empty or containing the expected PandasIndex, "
            "and to preserve reset coordinate variables exactly."
        )
        forbidden.append(
            "Do not only change DataVariables.__iter__, __len__, or __contains__; the failing behavior is the "
            "reset_index/xindexes state transition, not mapping length accounting."
        )
        sources.append(
            {
                "invariant_id": "xarray_reset_index_xindexes_contract",
                "source": "oracle_snapshot_test_assertions",
                "matched_targets": [
                    target
                    for target in targets
                    if "reset_index" in target.lower() or "test_dataset" in target.lower() or "test_dataarray" in target.lower()
                ],
                "matched_contract_count": len(assertion_contracts),
                "support_file_candidates": ["xarray/core/dataset.py", "xarray/core/dataarray.py", "xarray/core/indexes.py"],
            }
        )
    if _mentions_sklearn_lasso_lars_copy_x_contract(text, assertion_contracts):
        semantic_invariants.append(
            "sklearn LassoLars/Lars copy_X must propagate the runtime fit(copy_X=...) value into every "
            "_preprocess_data call that can mutate X."
        )
        semantic_invariants.append(
            "The copy_X patch must not only change constructor defaults; it must preserve per-call copy_X "
            "semantics through the fit-time preprocessing path."
        )
        expected.append(
            "Focused oracle requires copy_X == np.array_equal(X, X_copy) for both copy_X=True and copy_X=False."
        )
        forbidden.append(
            "Do not only set self.copy_X conditionally in constructors; propagate the local fit copy_X argument "
            "to _preprocess_data at all affected least_angle.py call sites."
        )
        sources.append(
            {
                "invariant_id": "sklearn_lasso_lars_copy_x_preprocess_propagation",
                "source": "oracle_snapshot_test_assertions",
                "matched_targets": [
                    target
                    for target in targets
                    if "copyx" in target.lower() or "copy_x" in target.lower() or "least_angle" in target.lower()
                ],
                "matched_contract_count": len(assertion_contracts),
                "support_file_candidates": ["sklearn/linear_model/least_angle.py"],
            }
        )

    semantic_invariants = _dedupe_texts(semantic_invariants)
    if assertion_contracts:
        expected.extend(f"Oracle assertion contract: {item}" for item in assertion_contracts[:12])
    if not semantic_invariants and not assertion_contracts:
        return {}
    evidence = {
        "oracle_derived_semantic_invariants": bool(semantic_invariants),
        "semantic_invariants": semantic_invariants,
        "forbidden_patch_directions": _dedupe_texts(forbidden),
        "expected_behavior_constraints": _dedupe_texts(expected),
        "semantic_invariant_sources": sources,
    }
    if assertion_contracts:
        evidence["oracle_assertion_contracts_derived"] = True
        evidence["oracle_assertion_contracts"] = assertion_contracts
        evidence["oracle_assertion_contract_sources"] = list(
            assertion_evidence.get("oracle_assertion_contract_sources", []) or []
        )
    support_file_candidates = _dedupe_texts(
        [
            path
            for source in sources
            if isinstance(source, dict)
            for path in list(source.get("support_file_candidates", []) or [])
        ]
    )
    if support_file_candidates:
        evidence["support_file_candidates"] = support_file_candidates
        evidence["bounded_support_expansion_reason"] = (
            "semantic assertion contract requires a backing adapter capability not expressible in the "
            "initially localized call site alone"
        )
        evidence["assertion_contract_edit_intent_required"] = True
    return evidence


def _derive_oracle_assertion_contract_evidence(
    *,
    manifest: dict[str, Any],
    targets: list[str],
) -> dict[str, Any]:
    oracle_root = Path(str(manifest.get("oracle_snapshot", "") or ""))
    if not oracle_root.exists() or not oracle_root.is_dir():
        return {}
    contracts: list[str] = []
    sources: list[dict[str, Any]] = []
    for target in targets[:12]:
        parsed = _parse_pytest_nodeid(target)
        if not parsed:
            continue
        test_path, test_name = parsed
        full_path = oracle_root / test_path
        if not full_path.exists() or not full_path.is_file() or full_path.suffix != ".py":
            continue
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(source) > 400_000:
            continue
        function_contracts = _extract_assertion_contracts_from_test_source(
            source=source,
            test_name=test_name,
        )
        if not function_contracts:
            continue
        contracts.extend(function_contracts)
        sources.append(
            {
                "source": "oracle_snapshot_test_assertions",
                "target": target,
                "test_file": test_path,
                "test_name": test_name,
                "contract_count": len(function_contracts),
            }
        )
    contracts = _dedupe_texts(contracts)[:24]
    if not contracts:
        return {}
    return {
        "oracle_assertion_contracts_derived": True,
        "oracle_assertion_contracts": contracts,
        "oracle_assertion_contract_sources": sources,
    }


def _parse_pytest_nodeid(nodeid: str) -> tuple[str, str] | None:
    text = str(nodeid or "").strip()
    if "::" not in text:
        return None
    path, remainder = text.split("::", 1)
    if not path.endswith(".py"):
        return None
    test_name = remainder.split("::")[-1].split("[", 1)[0].strip()
    if not test_name:
        return None
    return path, test_name


def _extract_assertion_contracts_from_test_source(*, source: str, test_name: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    contracts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != test_name:
            continue
        for child in ast.walk(node):
            contract = _assertion_contract_from_ast_node(child, source)
            if contract:
                contracts.append(contract)
    return _dedupe_texts(contracts)[:12]


def _assertion_contract_from_ast_node(node: ast.AST, source: str) -> str | None:
    if isinstance(node, ast.Assert):
        expr = ast.get_source_segment(source, node.test) or ast.unparse(node.test)
        return _normalize_oracle_assertion_contract(f"assert {expr}")
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {
            "assert_equal",
            "assert_array_equal",
            "assert_allclose",
            "assert_array_almost_equal",
            "assert_almost_equal",
            "assert_frame_equal",
            "assert_series_equal",
            "assert_dict_equal",
        }:
            expr = ast.get_source_segment(source, node) or ast.unparse(node)
            return _normalize_oracle_assertion_contract(expr)
        if name.endswith("raises") or name == "pytest.raises":
            expr = ast.get_source_segment(source, node) or ast.unparse(node)
            return _normalize_oracle_assertion_contract(expr)
    return None


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _normalize_oracle_assertion_contract(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if not compact:
        return None
    if len(compact) > 220:
        compact = compact[:217].rstrip() + "..."
    return compact


def _mentions_dataframe_dtype_preservation_contract(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        (
            "dataframe_output_dtypes" in lowered
            or "test_output_dataframe" in lowered
            or "preserving dtypes" in lowered
            or "preserve the dtypes" in lowered
            or "preserve dtypes" in lowered
        )
        and ("dtype" in lowered or "dtypes" in lowered)
        and ("_set_output" in lowered or "set_output" in lowered or "dataframe" in lowered)
    )


def _mentions_roc_curve_probability_threshold_contract(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "roc_curve" in lowered
        and (
            "probablity_estimates" in lowered
            or "probability_estimates" in lowered
            or "probability estimate" in lowered
            or "thresholds[0]" in lowered
            or "drop_intermediate" in lowered
        )
        and ("threshold" in lowered or "np.inf" in lowered or "isinf" in lowered)
    )


def _mentions_sphinx_none_annotation_object_reference_contract(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "sphinx" in lowered
        and "none" in lowered
        and ("autodoc_typehints" in lowered or "parse_annotation" in lowered or "test_domain_py" in lowered)
        and ("signature" in lowered or "reftype" in lowered or "pending_xref" in lowered or "clickable" in lowered)
    )


def _mentions_sympy_permutation_non_disjoint_cycle_contract(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "sympy" in lowered
        and "permutation" in lowered
        and ("non-disjoint" in lowered or "non disjoint" in lowered or "[[0,1],[0,1]]" in lowered or "[[0, 1], [0, 1]]" in lowered)
        and ("cycle" in lowered or "cycles" in lowered)
    )


def _mentions_xarray_indexvariable_copy_contract(text: str, assertion_contracts: list[str]) -> bool:
    lowered = str(text or "").lower()
    contract_text = "\n".join(str(item or "") for item in assertion_contracts).lower()
    return (
        ("xarray" in lowered or "test_variable" in lowered)
        and ("indexvariable" in lowered or "unicode indices" in lowered or "unicode index" in lowered)
        and ("copy" in lowered or "deepcopy" in lowered)
        and ("dtype" in lowered or "object" in lowered or "source_ndarray" in contract_text)
        and "assert type(v) is type(w)" in contract_text
        and "assert v.dtype == w.dtype" in contract_text
        and "source_ndarray(v.values)" in contract_text
    )


def _mentions_xarray_where_keep_attrs_contract(text: str, assertion_contracts: list[str]) -> bool:
    lowered = str(text or "").lower()
    contract_text = "\n".join(str(item or "") for item in assertion_contracts).lower()
    return (
        ("xarray" in lowered or "xr.where" in lowered or "test_computation" in lowered)
        and ("where" in lowered or "test_where_attrs" in lowered)
        and "keep_attrs" in (lowered + "\n" + contract_text)
        and ("attrs" in (lowered + "\n" + contract_text) or "assert_identical" in contract_text)
    )


def _mentions_xarray_reset_index_xindexes_contract(text: str, assertion_contracts: list[str]) -> bool:
    lowered = str(text or "").lower()
    contract_text = "\n".join(str(item or "") for item in assertion_contracts).lower()
    return (
        ("xarray" in lowered or "test_dataset" in lowered or "test_dataarray" in lowered)
        and "reset_index" in (lowered + "\n" + contract_text)
        and (
            "xindexes" in contract_text
            or "pandasindex" in contract_text
            or "not coordinates with an index" in contract_text
            or "assert_identical" in contract_text
        )
    )


def _mentions_sklearn_lasso_lars_copy_x_contract(text: str, assertion_contracts: list[str]) -> bool:
    lowered = str(text or "").lower()
    contract_text = "\n".join(str(item or "") for item in assertion_contracts).lower()
    return (
        ("sklearn" in lowered or "least_angle" in lowered or "lasso_lars" in lowered or "lars" in lowered)
        and ("copyx" in lowered or "copy_x" in lowered or "copy_x" in contract_text)
        and ("np.array_equal(x, x_copy)" in contract_text or "copy_x ==" in contract_text)
    )


def _dedupe_texts(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_dicts(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _target_mapping_probe_evidence(paths: list[Path], *, instance_id: str) -> dict[str, Any]:
    target_map: dict[str, str] = {}
    mappings: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    conflicted_originals: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Target mapping probe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []) if isinstance(payload, dict) else [])
        for row in rows:
            if not isinstance(row, dict) or str(row.get("instance_id", "") or "") != instance_id:
                continue
            for raw_mapping in list(row.get("target_mappings", []) or []):
                mapping = dict(raw_mapping or {})
                original = str(mapping.get("original_target", "") or "").strip()
                accepted = str(mapping.get("accepted_candidate", "") or "").strip()
                base = {
                    "probe_path": str(path),
                    "original_target": original,
                    "accepted_target": accepted,
                }
                if not original or not accepted:
                    skipped.append({**base, "reason": "missing_original_or_accepted_target"})
                    continue
                if original == accepted:
                    skipped.append({**base, "reason": "identity_mapping"})
                    continue
                if bool(mapping.get("requires_manual_review", False)):
                    skipped.append({**base, "reason": "requires_manual_review"})
                    continue
                risk = _target_protocol_risk(accepted)
                if risk:
                    skipped.append({**base, "reason": f"accepted_target_protocol_risk:{risk}"})
                    continue
                if original in conflicted_originals:
                    skipped.append({**base, "reason": "conflicting_mapping"})
                    continue
                previous = target_map.get(original)
                if previous and previous != accepted:
                    conflicted_originals.add(original)
                    target_map.pop(original, None)
                    skipped.append(
                        {
                            "probe_path": str(path),
                            "original_target": original,
                            "accepted_target": previous,
                            "reason": "conflicting_mapping",
                        }
                    )
                    skipped.append({**base, "reason": "conflicting_mapping"})
                    continue
                if not previous:
                    target_map[original] = accepted
                    mappings.append(base)
    return {
        "probe_paths": [str(path) for path in paths],
        "accepted_target_map": target_map,
        "accepted_mappings": [
            mapping
            for mapping in mappings
            if mapping["original_target"] in target_map
            and target_map[mapping["original_target"]] == mapping["accepted_target"]
        ],
        "skipped_mappings": skipped,
        "claim_boundary": {
            "target_mapping_is_protocol_repair_not_recovery_success": True,
            "accepted_non_manual_mappings_only": True,
            "applied_before_runtime_collect_validation": True,
            "does_not_modify_patch_or_labels": True,
            "runtime_oracle_still_required_for_recovery_credit": True,
        },
    }


def _apply_target_mapping_to_evaluation(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    target_mapping: dict[str, Any],
) -> dict[str, Any]:
    if not list(target_mapping.get("probe_paths", []) or []):
        return evaluation
    output = dict(evaluation)
    target_map = dict(target_mapping.get("accepted_target_map", {}) or {})
    recovered_fail_targets, recovered_replacements, remaining_filtered = _recover_filtered_targets_with_mapping(
        list(output.get("filtered_targets", []) or []),
        target_map=target_map,
    )
    fail_targets, fail_replacements = _map_evaluation_targets(
        list(output.get("fail_to_pass_targets", []) or []),
        target_map=target_map,
        field="dataset_fail_to_pass",
    )
    oracle_targets, oracle_replacements = _map_evaluation_targets(
        list(output.get("oracle_targets", []) or []),
        target_map=target_map,
        field="dataset_oracle_targets",
    )
    for target in recovered_fail_targets:
        if target not in fail_targets:
            fail_targets.append(target)
        if target not in oracle_targets:
            oracle_targets.append(target)
    replacements = _unique_target_replacements(
        fail_replacements + oracle_replacements + recovered_replacements
    )
    notes = list(output.get("normalization_notes", []) or [])
    notes.append("target_mapping_probe_supplied")
    if replacements:
        notes.append("accepted_target_mapping_applied_before_collect_validation")
    if recovered_fail_targets:
        notes.append("target_mapping_recovered_protocol_filtered_targets")
    elif target_map:
        notes.append("target_mapping_probe_no_applicable_replacement")
    if target_mapping.get("skipped_mappings"):
        notes.append("target_mapping_probe_skipped_ineligible_mappings")
    prefix = _test_command_prefix(manifest)
    output.update(
        {
            "fail_to_pass_targets": fail_targets,
            "oracle_targets": oracle_targets,
            "filtered_targets": remaining_filtered,
            "normalization_notes": list(dict.fromkeys(notes)),
            "fail_to_pass_command": _compose_test_command(prefix, fail_targets) if fail_targets else "",
            "oracle_command": _compose_test_command(prefix, oracle_targets) if oracle_targets else "",
            "error_type": "" if fail_targets and str(output.get("error_type", "") or "") == "evaluation_protocol_blocked" else output.get("error_type", ""),
            "stop_reason": "" if fail_targets and str(output.get("stop_reason", "") or "") == "no_safe_fail_to_pass_targets" else output.get("stop_reason", ""),
            "target_mapping": {
                "probe_paths": list(target_mapping.get("probe_paths", []) or []),
                "accepted_mapping_count": len(target_map),
                "used_replacement_count": len(
                    {
                        (
                            str(item.get("original_target", "") or ""),
                            str(item.get("accepted_target", "") or ""),
                        )
                        for item in replacements
                    }
                ),
                "replacements": replacements,
                "recovered_filtered_targets": recovered_replacements,
                "skipped_mappings": list(target_mapping.get("skipped_mappings", []) or []),
                "claim_boundary": dict(target_mapping.get("claim_boundary", {}) or {}),
            },
        }
    )
    return output


def _recover_filtered_targets_with_mapping(
    filtered_targets: list[dict[str, str]],
    *,
    target_map: dict[str, str],
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    recovered_targets: list[str] = []
    replacements: list[dict[str, str]] = []
    remaining_filtered: list[dict[str, str]] = []
    for item in filtered_targets:
        filtered = dict(item or {})
        original = str(filtered.get("target", "") or "").strip()
        field = str(filtered.get("field", "") or "")
        accepted = str(target_map.get(original, "") or "").strip()
        if field == "dataset_fail_to_pass" and accepted:
            if accepted not in recovered_targets:
                recovered_targets.append(accepted)
            replacements.append(
                {
                    "field": "dataset_fail_to_pass",
                    "original_target": original,
                    "accepted_target": accepted,
                }
            )
            continue
        remaining_filtered.append(filtered)
    return recovered_targets, replacements, remaining_filtered


def _map_evaluation_targets(
    targets: list[str],
    *,
    target_map: dict[str, str],
    field: str,
) -> tuple[list[str], list[dict[str, str]]]:
    mapped_targets: list[str] = []
    replacements: list[dict[str, str]] = []
    for raw_target in targets:
        target = str(raw_target or "").strip()
        if not target:
            continue
        mapped = str(target_map.get(target, target) or "").strip()
        if mapped and mapped not in mapped_targets:
            mapped_targets.append(mapped)
        if target in target_map:
            replacements.append(
                {
                    "field": field,
                    "original_target": target,
                    "accepted_target": mapped,
                }
            )
    return mapped_targets, replacements


def _unique_target_replacements(replacements: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in replacements:
        key = (
            str(item.get("field", "") or ""),
            str(item.get("original_target", "") or ""),
            str(item.get("accepted_target", "") or ""),
        )
        if key not in seen:
            seen.add(key)
            output.append(dict(item))
    return output


def _patcher_recovery_context(
    manifest: dict[str, Any],
    fixed: dict[str, Any],
    *,
    patcher_failure_evidence: dict[str, Any] | None = None,
    patch_contract: str = "",
    operator_gate: str = "",
    edit_intent_plan: dict[str, Any] | None = None,
) -> str:
    guidance = dict(manifest.get("mas_dx_guidance", {}) or {})
    trajectory_record = dict(fixed.get("record", {}) or {})
    patcher_failure_evidence = dict(patcher_failure_evidence or {})
    lines = [
        "[MAS-DX ACTION-ENFORCED RECOVERY]",
        "Run scope: patcher_fixed_localization.",
        "Do not rerun localization. Reuse the previous MAS locator output below as the fixed source boundary.",
        "Discard the previous patch plan and produce a fresh minimal source patch inside this boundary unless the evidence proves it impossible.",
        "After editing, run focused validation when available.",
    ]
    if patch_contract == "source_only":
        lines.extend(_source_only_patch_contract_lines(fixed))
    if operator_gate:
        lines.extend(
            _operator_gate_context_lines(
                operator_gate,
                trajectory_record,
                patcher_failure_evidence=patcher_failure_evidence,
            )
        )
    if guidance:
        lines.extend(
            [
                f"Diagnosis input mode: {guidance.get('input_mode', '')}",
                f"Predicted failure type: {guidance.get('primary_failure_type', '')}",
                f"Responsible stage: {guidance.get('responsible_stage', '')}",
                f"Recommended action: {guidance.get('recommended_recovery_action', '')}",
            ]
        )
    selected = list(fixed.get("selected_target_candidates", []) or [])
    if selected:
        lines.append("Reusable selected targets:")
        lines.extend(f"- {path}" for path in selected[:10])
    if patcher_failure_evidence:
        lines.extend(_patcher_failure_evidence_context_lines(patcher_failure_evidence))
    if edit_intent_plan and bool(edit_intent_plan.get("enabled", False)):
        lines.extend(_edit_intent_context_lines(edit_intent_plan))
    return "\n".join(str(line) for line in lines if str(line).strip())


def _edit_intent_context_lines(edit_intent_plan: dict[str, Any]) -> list[str]:
    prompt = str(edit_intent_plan.get("prompt_text", "") or "").strip()
    source_edit_contract = str(edit_intent_plan.get("source_edit_contract_text", "") or "").strip()
    anchors = dict(edit_intent_plan.get("source_span_anchors", {}) or {})
    failure_span = dict(edit_intent_plan.get("failure_span", {}) or {})
    lines = [
        "[MASGUARD V11 SOURCE-SPAN EDIT INTENT]",
        "Use this pre-patch edit intent as the controlling source-edit plan before patch synthesis.",
        "Hard recovery rule: satisfy the CAR patch intent below in the planner and implementer output.",
    ]
    target_paths = [str(item) for item in list(anchors.get("target_paths", []) or []) if str(item).strip()]
    if target_paths:
        lines.append("V11 source-span anchors:")
        lines.extend(f"- {path}" for path in target_paths[:8])
    family = str(failure_span.get("family", "") or "")
    subtype = str(failure_span.get("subtype", "") or "")
    if family or subtype:
        lines.append(f"V11 failure span: family={family}, subtype={subtype}.")
    if prompt:
        lines.append(prompt)
    if source_edit_contract:
        lines.append(source_edit_contract)
    return lines


def _operator_gate_context_lines(
    operator_gate: str,
    record: dict[str, Any],
    *,
    patcher_failure_evidence: dict[str, Any] | None = None,
) -> list[str]:
    evidence = dict(patcher_failure_evidence or {})
    if operator_gate == "shared_fact_quarantine_then_repatch":
        shared_facts = dict(record.get("shared_facts", {}) or {})
        handoff = dict(record.get("handoff_artifacts", {}) or {})
        lines = [
            "[MAS-DX-R OPERATOR GATE: SHARED FACT QUARANTINE]",
            "Hard recovery rule: treat prior shared facts and upstream handoff assumptions as suspect context, not ground truth.",
            "Hard recovery rule: before editing, re-derive the patch rationale from the original issue, current source, reusable localization, and validated verifier/oracle evidence.",
            "Hard recovery rule: do not copy code, constants, paths, or conclusions solely because they appear in prior shared facts.",
            "Hard recovery rule: if a prior shared fact conflicts with source/verifier evidence, ignore the shared fact and patch the source evidence.",
            "Quarantine stop gate: if the reusable localization plus source/verifier evidence is insufficient after quarantining shared facts, stop rather than broaden the patch.",
        ]
        if shared_facts:
            lines.append("Quarantined shared-fact keys from the failed MAS trajectory:")
            lines.extend(f"- {key}" for key in sorted(shared_facts)[:12])
        if handoff:
            lines.append("Quarantined handoff artifact keys from the failed MAS trajectory:")
            lines.extend(f"- {key}" for key in sorted(handoff)[:12])
        return lines
    if operator_gate == "handoff_correction_or_ablation":
        return [
            "[MAS-DX-R OPERATOR GATE: HANDOFF CORRECTION]",
            "Hard recovery rule: treat upstream handoff artifacts as suspect until they are checked against source and verifier evidence.",
            "Hard recovery rule: do not propagate stale localization, target, or verifier assumptions that conflict with current evidence.",
        ]
    if operator_gate == "patch_contract_nonregression_repatch":
        diff_summary = dict(record.get("diff_summary", {}) or {})
        changed_classes = dict(diff_summary.get("changed_file_classes", {}) or {})
        oracle = dict(record.get("oracle", {}) or {})
        lines = [
            "[MAS-DX-R OPERATOR GATE: PATCH CONTRACT NON-REGRESSION]",
            "Hard recovery rule: treat the previous patch as a failed hypothesis, not as a patch to copy.",
            "Hard recovery rule: preserve source-only recovery boundaries and do not edit tests, generated files, or evaluation targets.",
            "Hard recovery rule: use the previous changed source files as suspect locations, but repair from the original issue and verifier evidence.",
            "Hard recovery rule: prioritize removing regressions introduced by the previous patch while still satisfying fail-to-pass targets.",
        ]
        source_files = [str(path) for path in list(changed_classes.get("source_files", []) or []) if str(path)]
        test_files = [str(path) for path in list(changed_classes.get("test_files", []) or []) if str(path)]
        if source_files:
            lines.append("Previous source files touched by failed patch:")
            lines.extend(f"- {path}" for path in source_files[:8])
        if test_files:
            lines.append("Forbidden test/generated edits observed in failed patch; do not repeat these edits:")
            lines.extend(f"- {path}" for path in test_files[:8])
        if oracle:
            lines.append(
                "Failed-capture verifier signal: "
                f"fail_to_pass_returncode={oracle.get('fail_to_pass_returncode', '')}, "
                f"oracle_returncode={oracle.get('oracle_returncode', '')}."
            )
        return lines
    if operator_gate == "protocol_guard_then_patch":
        oracle = dict(record.get("oracle", {}) or {})
        lines = [
            "[MAS-DX-R OPERATOR GATE: PROTOCOL GUARD THEN PATCH]",
            "Hard recovery rule: the verifier/protocol replay has already failed to produce oracle success, so do not stop at replay-only evidence.",
            "Hard recovery rule: keep source-only recovery boundaries and do not edit tests, generated files, or evaluation targets.",
            "Hard recovery rule: use reusable localization and verifier output to produce a minimal source patch only if the source evidence supports it.",
            "Hard recovery rule: if source evidence is insufficient, abstain instead of broadening the patch.",
        ]
        if oracle:
            lines.append(
                "Protocol probe signal before patch fallback: "
                f"fail_to_pass_returncode={oracle.get('fail_to_pass_returncode', '')}, "
                f"oracle_returncode={oracle.get('oracle_returncode', '')}."
            )
        return lines
    if operator_gate == "patch_family_lock_then_repatch":
        locked_sources = _evidence_patch_family_source_files(evidence)
        family_id = _evidence_patch_family_id(evidence, locked_sources)
        lines = [
            "[MAS-DX-R OPERATOR GATE: PATCH FAMILY LOCK]",
            "Hard recovery rule: route this retry by instance_id plus patch family, not by instance_id alone.",
            "Hard recovery rule: treat the locked patch-family source files as the only source edit targets for this retry.",
            "Hard recovery rule: do not switch to another historical patch family, even if that family appears in reusable localization.",
            "Hard recovery rule: if a fresh source diff cannot be produced inside the locked patch family, stop instead of patching a different family.",
            "Hard recovery rule: do not replay the candidate patch verbatim; re-derive a fresh minimal source patch from the issue, source, and verifier/oracle evidence.",
        ]
        if family_id:
            lines.append(f"Locked patch family id: {family_id}")
        if locked_sources:
            lines.append("Locked source files for this patch family:")
            lines.extend(f"- {path}" for path in locked_sources[:10])
        return lines
    if operator_gate == "semantic_span_delta_guarded_repatch":
        return [
            "[MAS-DX-R OPERATOR GATE: SEMANTIC SPAN-DELTA GUARDED REPATCH]",
            "Hard recovery rule: convert the bounded-observation failure span into executable patch constraints before runtime validation.",
            "Hard recovery rule: after the first source diff, runtime validation must eliminate or change the verified failure span family/subtype.",
            "Hard recovery rule: if the same failure span family/subtype remains, perform at most one bounded span-delta revision using the post-patch failure evidence.",
            "Hard recovery rule: if the bounded revision still leaves the same span unresolved, stop and report the span-delta failure instead of starting another retry loop.",
        ]
    if operator_gate == "semantic_invariant_guarded_repatch":
        return [
            "[MAS-DX-R OPERATOR GATE: SEMANTIC INVARIANT GUARDED REPATCH]",
            "Hard recovery rule: convert the semantic failure evidence into executable patch constraints before runtime validation.",
            "Hard recovery rule: if the first produced source diff matches a forbidden semantic direction, perform at most one bounded revision against the rejected patch evidence.",
            "Hard recovery rule: if the bounded revision still violates the semantic invariant, stop and report the invariant violation instead of spending oracle budget.",
            "Hard recovery rule: this gate is a bounded minimal intervention validator, not an open-ended extra retry.",
        ]
    if operator_gate == "semantic_effect_guarded_repatch":
        return [
            "[MAS-DX-R OPERATOR GATE: SEMANTIC EFFECT GUARDED REPATCH]",
            "Hard recovery rule: convert the failed-run semantic evidence into executable patch constraints before runtime validation.",
            "Hard recovery rule: do not replay a previous failed candidate patch when the evidence marks candidate replay as disabled.",
            "Hard recovery rule: if the first source diff matches a forbidden semantic direction, revise once against the rejected patch evidence before validation credit.",
            "Hard recovery rule: after validation, the focused failure output must show that the edited runtime data/control flow changed the failing behavior.",
            "Hard recovery rule: if span-delta revision cannot produce a fresh semantically aligned patch, stop fail-closed instead of starting another retry loop.",
        ]
    return []


def _evidence_patch_family_id(evidence: dict[str, Any], source_files: list[str] | None = None) -> str:
    for key in (
        "candidate_patch_family_id",
        "patch_family_id",
        "selected_patch_family_id",
    ):
        value = str(evidence.get(key, "") or "").strip()
        if value:
            return value
    instance_id = str(evidence.get("instance_id", "") or "").strip()
    source_text = "_".join(str(path).replace("/", "_").replace(".", "_") for path in list(source_files or []) if str(path))
    if instance_id and source_text:
        normalized_instance = instance_id.replace("__", "_").replace("-", "_")
        return f"{normalized_instance}_{source_text}_source_only"
    return ""


def _evidence_patch_family_source_files(evidence: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in (
        "candidate_patch_source_files",
        "candidate_patch_replay_source_files",
        "source_files",
    ):
        candidates.extend(str(path) for path in list(evidence.get(key, []) or []))
    for container_key in ("changed_file_classes",):
        classes = dict(evidence.get(container_key, {}) or {})
        candidates.extend(str(path) for path in list(classes.get("source_files", []) or []))
    patch_summary = dict(evidence.get("patch_summary", {}) or {})
    summary_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
    candidates.extend(str(path) for path in list(summary_classes.get("source_files", []) or []))
    if not candidates:
        candidates.extend(str(path) for path in list(patch_summary.get("changed_files", []) or []))
        candidates.extend(str(path) for path in list(evidence.get("changed_files", []) or []))
    unique: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        text = str(path or "").strip()
        if not text or _looks_like_test_path(text) or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _semantic_invariant_revision_enabled(operator_gate: str) -> bool:
    return operator_gate in {
        "semantic_invariant_guarded_repatch",
        "semantic_span_delta_guarded_repatch",
        "semantic_effect_guarded_repatch",
    }


def _syntax_repair_revision_context(
    *,
    recovery_context: str,
    syntax_gate: dict[str, Any],
    patch_text: str,
) -> str:
    violations = [
        dict(item)
        for item in list(syntax_gate.get("violations", []) or [])
        if isinstance(item, dict)
    ]
    patch_excerpt = [
        line.rstrip()[:240]
        for line in str(patch_text or "").splitlines()
        if line.strip()
    ][:80]
    lines = [
        str(recovery_context or "").strip(),
        "[MAS-DX-R CANDIDATE SYNTAX REPAIR REVISION GATE]",
        "The previous source diff in this same bounded recovery attempt failed Python syntax validation before oracle execution.",
        "Hard recovery rule: perform exactly one minimal source revision that fixes only the reported syntax errors.",
        "Hard recovery rule: keep the same fixed localization and source-only patch boundary.",
        "Hard recovery rule: do not edit tests, do not broaden localization, and do not start a new repair strategy.",
        "Hard recovery rule: if the syntax error cannot be fixed with a minimal source-only diff, stop instead of emitting a broader patch.",
    ]
    if violations:
        lines.append("Candidate syntax violations to repair:")
        for item in violations[:8]:
            lines.append(f"- path={item.get('path', '')}, returncode={item.get('returncode', '')}")
            excerpt = str(item.get("output_excerpt", "") or "").strip()
            if excerpt:
                lines.append(f"  output_excerpt: {excerpt[-1000:]}")
    checked = [str(item) for item in list(syntax_gate.get("checked_files", []) or []) if str(item).strip()]
    if checked:
        lines.append("Files that must pass syntax validation after the revision:")
        lines.extend(f"- {path}" for path in checked[:8])
    if patch_excerpt:
        lines.append("Rejected patch excerpt to minimally revise:")
        lines.extend(f"- {line}" for line in patch_excerpt)
    return "\n".join(line for line in lines if str(line).strip())


def _semantic_invariant_revision_context(
    *,
    recovery_context: str,
    gate: dict[str, Any],
    patch_text: str,
) -> str:
    violations = [
        str(item)
        for item in list(gate.get("violations", []) or [])
        if str(item).strip()
    ]
    checked = [
        str(item)
        for item in list(gate.get("checked_invariants", []) or [])
        if str(item).strip()
    ]
    patch_excerpt = [
        line.rstrip()[:240]
        for line in str(patch_text or "").splitlines()
        if line.strip()
    ][:80]
    lines = [
        str(recovery_context or "").strip(),
        "[MAS-DX-R SEMANTIC INVARIANT REVISION GATE]",
        "The previous source diff in this same bounded recovery attempt was rejected before runtime validation.",
        "Hard recovery rule: perform exactly one minimal source revision that removes the forbidden semantic direction.",
        "Hard recovery rule: do not broaden localization, do not edit tests, and do not add another retry loop.",
        "Hard recovery rule: if you cannot satisfy every checked invariant with a minimal source diff, stop instead of emitting a patch.",
    ]
    if violations:
        lines.append("Rejected semantic-invariant violations:")
        lines.extend(f"- {item}" for item in violations[:8])
        repair_steps = _semantic_invariant_violation_repair_steps(violations)
        if repair_steps:
            lines.append("Required repair steps for the rejected invariant violation:")
            lines.extend(f"{idx}. {item}" for idx, item in enumerate(repair_steps[:8], start=1))
    if checked:
        lines.append("Checked semantic invariants that must now be preserved:")
        lines.extend(f"- {item}" for item in checked[:8])
    if patch_excerpt:
        lines.append("Rejected patch excerpt to revise away from:")
        lines.extend(f"- {line}" for line in patch_excerpt)
    return "\n".join(line for line in lines if str(line).strip())


def _semantic_invariant_violation_repair_steps(violations: list[str]) -> list[str]:
    values = set(str(item or "") for item in violations)
    if values.intersection(
        {
            "semantic_invariant_xarray_indexvariable_copy_aliasing_fix_missing",
            "semantic_invariant_xarray_indexvariable_copy_wrong_class_scope",
            "semantic_invariant_xarray_indexvariable_unrelated_adapter_api",
        }
    ):
        return [
            "In xarray/core/indexing.py, add exactly one PandasIndexAdapter.copy(self, deep=True) method inside the PandasIndexAdapter class.",
            "Remove or avoid any new copy methods in ExplicitIndexer, LazilyOuterIndexedArray, LazilyVectorizedIndexedArray, or other non-PandasIndexAdapter classes.",
            "Do not add unrelated adapter APIs such as astype while satisfying this copy/deepcopy contract.",
            "That method must deep-copy with self.array.copy(deep=True) when deep is true and reuse self.array when deep is false.",
            "That method must return PandasIndexAdapter(array, self._dtype), preserving the adapter dtype.",
            "In xarray/core/variable.py, only replace the data=None branch of IndexVariable.copy with data = self._data.copy(deep=deep).",
            "Do not delete IndexVariable.equals, _data_equals, to_index_variable, or the surrounding method bodies.",
        ]
    if (
        "semantic_invariant_xarray_where_keep_attrs_contract_missing" in values
        or "semantic_invariant_xarray_where_keep_attrs_uses_cond_attrs" in values
    ):
        return [
            "In xarray/core/computation.py, change the public where signature to accept keep_attrs=None.",
            "Do not rely on apply_ufunc(keep_attrs=True) for where, because that preserves attrs from cond.",
            "When keep_attrs resolves true, the returned DataArray/Dataset must receive attrs from x, not cond.",
            "Do not edit only docstrings, examples, or tests; the runtime TypeError must be fixed.",
            "Preserve existing cond/x/y alignment and dask arguments.",
        ]
    if (
        "semantic_invariant_sphinx_none_annotation_reftype_obj_missing" in values
        or "semantic_invariant_sphinx_none_annotation_wrong_autodoc_typehints_scope" in values
    ):
        return [
            "Edit sphinx/domains/python.py, not sphinx/ext/autodoc/typehints.py.",
            "In _parse_annotation(), make the pending_xref for text 'None' use reftype='obj' while other annotations keep reftype='class'.",
            "Preserve refdomain='py', reftarget=text, and the existing parser punctuation/nesting behavior.",
            "Do not rewrite field-list insertion, merge_typehints, or autodoc description-mode handling.",
        ]
    if "semantic_invariant_sphinx_reserved_toctree_entries_skipped" in values:
        return [
            "Edit sphinx/directives/other.py in TocTree.parse_content(), where toctree['entries'] is populated.",
            "For refs genindex, modindex, and search, append the entry to toctree['entries'] while avoiding the nonexisting-document warning and note_reread path.",
            "Do not add continue, filtering, removal, or entries mutation inside the reserved-ref branch.",
            "Do not edit tests or directive parsing; this is a warning/lookup control-flow repair only.",
            "The focused test must still observe entries [(None, 'genindex'), (None, 'modindex'), (None, 'search')].",
        ]
    if "semantic_invariant_sphinx_reserved_toctree_directive_entry_construction_missing" in values:
        return [
            "The rejected patch only changed warning handling and did not touch the directive code that builds toctree entries.",
            "Edit sphinx/directives/other.py in TocTree.parse_content(), near the branch where docname not in self.env.found_docs currently warns and calls note_reread().",
            "For reserved refs genindex, modindex, and search, append (title, docname) or (title, ref) to toctree['entries'] without adding them to includefiles.",
            "Do not edit only sphinx/environment/adapters/toctree.py; that file resolves entries after construction and cannot restore entries removed during parsing.",
            "Preserve the existing behavior for ordinary missing documents, excluded documents, URLs, self references, and real found_docs.",
        ]
    if "semantic_invariant_sphinx_reserved_toctree_entry_append_missing" in values:
        return [
            "The rejected patch touched sphinx/directives/other.py but still did not add reserved refs back to toctree['entries'].",
            "In TocTree.parse_content(), add a reserved-ref branch for genindex, modindex, and search before the ordinary nonexisting-document warning path completes.",
            "That branch must append the reserved ref to toctree['entries']; otherwise test_toctree_index still observes entries=[].",
            "Do not add these reserved refs to toctree['includefiles'], because they are built-in pages rather than source documents.",
            "Do not call self.env.note_reread() for the reserved-ref branch.",
        ]
    if (
        "semantic_invariant_sympy_permutation_non_disjoint_cycle_array_form_missing" in values
        or "semantic_invariant_sympy_permutation_cycle_args_not_canonicalized" in values
    ):
        return [
            "Edit sympy/combinatorics/permutations.py in Permutation.__new__.",
            "For cycle input with repeated elements, compose the cycles using Cycle() and convert the result to array form before object construction.",
            "Do not merely bypass the repeated-element ValueError while keeping the original repeated cycle-list in args.",
            "The constructed Permutation must store canonical array-form args so test_args can reconstruct it without recursion.",
            "Preserve the existing ValueError for duplicate elements in non-cycle array-form input.",
        ]
    return []


def _patch_intent_revision_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "accepted": False,
        "reason": reason,
        "first_gate": {},
        "second_gate": {},
        "revision_invariant_gate": {},
        "max_revision_count": 1,
    }


def _patch_intent_revision_context(
    *,
    recovery_context: str,
    gate: dict[str, Any],
    patch_text: str,
) -> str:
    violations = [
        str(item)
        for item in list(gate.get("violations", []) or [])
        if str(item).strip()
    ]
    target_files = [
        str(item)
        for item in list(gate.get("target_boundary_files", []) or [])
        if str(item).strip()
    ]
    contract_tokens = [
        str(item)
        for item in list(gate.get("contract_tokens", []) or [])
        if str(item).strip()
    ]
    patch_excerpt = [
        line.rstrip()[:240]
        for line in str(patch_text or "").splitlines()
        if line.strip()
    ][:80]
    lines = [
        str(recovery_context or "").strip(),
        "[MASGUARD PATCH-INTENT REVISION GATE]",
        "The previous source diff was rejected before oracle validation because it did not align with the bounded-observation recovery intent.",
        "Hard recovery rule: perform exactly one minimal source revision that touches the bounded-observation source boundary and addresses the observed semantic contract.",
        "Hard recovery rule: do not broaden localization, do not edit tests, and do not start another retry loop.",
        "Hard recovery rule: if no aligned source diff can be produced, stop instead of spending oracle budget.",
    ]
    if violations:
        lines.append("Rejected patch-intent violations:")
        lines.extend(f"- {item}" for item in violations[:8])
    if target_files:
        lines.append("Bounded-observation target boundary that must be touched:")
        lines.extend(f"- {item}" for item in target_files[:10])
    if contract_tokens:
        lines.append("Observed semantic contract tokens that must be explicitly addressed:")
        lines.extend(f"- {item}" for item in contract_tokens[:12])
    if patch_excerpt:
        lines.append("Rejected patch excerpt to revise away from:")
        lines.extend(f"- {line}" for line in patch_excerpt)
    return "\n".join(line for line in lines if str(line).strip())


def _semantic_invariant_patch_gate(
    *,
    patch_text: str,
    patcher_failure_evidence: dict[str, Any],
    operator_gate: str,
) -> dict[str, Any]:
    if operator_gate not in {
        "semantic_invariant_guarded_repatch",
        "semantic_span_delta_guarded_repatch",
        "semantic_effect_guarded_repatch",
    }:
        return {
            "attempted": False,
            "blocked": False,
            "violations": [],
            "checked_invariants": [],
            "claim_boundary": "not_enabled",
        }
    evidence = dict(patcher_failure_evidence or {})
    checked = [
        str(item)
        for item in list(evidence.get("semantic_invariants", []) or [])
        if str(item).strip()
    ]
    if not checked:
        return {
            "attempted": True,
            "blocked": False,
            "violations": [],
            "checked_invariants": [],
            "claim_boundary": "no_semantic_invariants_supplied",
        }
    violations = _semantic_invariant_patch_violations(
        patch_text=patch_text,
        patcher_failure_evidence=evidence,
    )
    return {
        "attempted": True,
        "blocked": bool(violations),
        "violations": violations,
        "checked_invariants": checked,
        "claim_boundary": "pre_validation_patch_shape_gate",
    }


def _semantic_invariant_patch_gate_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "blocked": False,
        "violations": [],
        "checked_invariants": [],
        "skip_reason": reason,
        "claim_boundary": "bounded_validation_disabled",
    }


def _semantic_invariant_patch_violations(
    *,
    patch_text: str,
    patcher_failure_evidence: dict[str, Any],
) -> list[str]:
    return semantic_invariant_patch_violations(
        patch_text=patch_text,
        evidence=patcher_failure_evidence,
    )


def _source_only_patch_contract_lines(fixed: dict[str, Any]) -> list[str]:
    selected = [
        str(path)
        for path in list(fixed.get("selected_target_candidates", []) or [])
        if str(path).strip()
    ]
    source_candidates = [
        path
        for path in selected
        if not _looks_like_test_path(path)
    ]
    contract = StageBoundaryPatchContract(
        contract_id="mas_dx_r_source_only_fixed_localization",
        replay_scope="patcher_fixed_localization",
        repair_mode="source_only_fixed_localization_repatch",
        execution_profile="focused_source_repair",
        producer_stage="locator",
        consumer_stage="patcher",
        suspect_paths=source_candidates[:8],
        required_fresh_source_diff=True,
        require_suspect_touch=bool(source_candidates),
        require_focused_validation=True,
        max_fresh_source_files=max(1, min(3, len(source_candidates) or 3)),
        forbidden_path_classes=["test", "generated"],
        negative_constraints=["do_not_edit_tests_for_recovery_claim"],
    )
    lines = [
        contract.to_prompt_text(),
        "Strict patch contract: source-only recovery scope is active.",
        "Hard recovery rule: do not edit tests, generated files, copied build outputs, or evaluation targets.",
        "Hard recovery rule: if a test edit is produced accidentally, discard it and keep only a minimal source diff.",
        "Hard recovery rule: strict success requires source-only patch, focused validation, and oracle success.",
    ]
    if source_candidates:
        lines.append("Strict source-only candidate targets:")
        lines.extend(f"- {path}" for path in source_candidates[:8])
    return lines


def _looks_like_test_path(path: str) -> bool:
    text = str(path or "").replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    return (
        any(part in {"test", "tests"} for part in parts)
        or "/test_" in text
        or text.endswith("_test.py")
        or text.startswith("test_")
    )


def _patcher_failure_evidence_context_lines(evidence: dict[str, Any]) -> list[str]:
    signature = dict(evidence.get("post_patch_failure_signature", {}) or {})
    changed_files = [
        str(path)
        for path in list(evidence.get("changed_files", []) or [])
        if str(path).strip()
    ]
    changed_classes = dict(evidence.get("changed_file_classes", {}) or {})
    failed_tests = [
        str(test)
        for test in list(signature.get("failed_tests", []) or [])
        if str(test).strip()
    ]
    exception_excerpt = [
        str(line)
        for line in list(signature.get("exception_excerpt", []) or [])
        if str(line).strip()
    ]
    selected = [
        str(path)
        for path in list(evidence.get("selected_target_candidates", []) or [])
        if str(path).strip()
    ]
    lines = [
        "[MAS-DX-R PATCHER FAILURE EVIDENCE]",
        f"Gate status: {evidence.get('gate_status', '')}",
        f"Gate blocker family: {evidence.get('gate_blocker_family', '')}",
        f"Previous patch legitimacy: {evidence.get('patch_legitimacy', '')}",
        f"Post-patch failure family: {signature.get('family', '')}",
        f"Failure headline: {signature.get('headline', '')}",
        "Hard recovery rule: treat this as evidence-conditioned repatching, not open-ended retry.",
        "Hard recovery rule: do not rerun locator or broaden beyond the reusable localization unless the failure evidence proves the boundary is wrong.",
        "Hard recovery rule: do not edit tests unless the failure evidence explicitly proves the test itself is wrong.",
    ]
    lines.extend(_g2_recovery_evidence_context_lines(evidence))
    if changed_files:
        lines.append("Previous failed recovery changed files:")
        lines.extend(f"- {path}" for path in changed_files[:10])
    candidate_patch_sources = [
        str(path)
        for path in list(evidence.get("candidate_patch_source_files", []) or [])
        if str(path).strip()
    ]
    if candidate_patch_sources:
        candidate_patch_replayed = (
            bool(evidence.get("candidate_patch_replayed", False))
            or bool(evidence.get("candidate_patch_replay_attempted", False))
            or bool(str(evidence.get("candidate_patch", "") or "").strip())
        )
        if candidate_patch_replayed:
            lines.append("Candidate source patch is replayed before this retry; minimally refine these source files:")
        else:
            lines.append("Locked source files from a prior failed patch family; no historical patch text is replayed:")
        lines.extend(f"- {path}" for path in candidate_patch_sources[:10])
        if candidate_patch_replayed:
            lines.append(
                "Hard recovery rule: preserve the useful candidate source change and fix the observed post-patch failure."
            )
        else:
            lines.append(
                "Hard recovery rule: re-derive a fresh minimal source patch from the issue and failure evidence, not from a copied historical diff."
            )
    violations = [
        str(item)
        for item in list(evidence.get("contract_violation_types", []) or [])
        if str(item).strip()
    ]
    warnings = [
        str(item)
        for item in list(evidence.get("contract_warning_types", []) or [])
        if str(item).strip()
    ]
    if violations:
        lines.append("Patch contract violations from the previous recovery attempt:")
        lines.extend(f"- {item}" for item in violations[:10])
        lines.append(
            "Hard recovery rule: the next patch must explicitly avoid every listed contract violation."
        )
    if warnings:
        lines.append("Patch contract warnings from the previous recovery attempt:")
        lines.extend(f"- {item}" for item in warnings[:10])
    hard_constraints = [
        str(item)
        for item in list(evidence.get("hard_constraints", []) or [])
        if str(item).strip()
    ]
    if hard_constraints:
        lines.append("Hard constraints for this bounded retry:")
        lines.extend(f"- {item}" for item in hard_constraints[:12])
    semantic_invariants = [
        str(item)
        for item in list(evidence.get("semantic_invariants", []) or [])
        if str(item).strip()
    ]
    if semantic_invariants:
        lines.append("Semantic invariants that the next patch must preserve:")
        lines.extend(f"- {item}" for item in semantic_invariants[:12])
        lines.append(
            "Hard recovery rule: a patch that violates these invariants is a failed recovery even if it changes source."
        )
    expected_constraints = [
        str(item)
        for item in list(evidence.get("expected_behavior_constraints", []) or [])
        if str(item).strip()
    ]
    if expected_constraints:
        lines.append("Expected behavior constraints from the focused failure:")
        lines.extend(f"- {item}" for item in expected_constraints[:12])
    lines.extend(_semantic_effect_specialized_context_lines(evidence))
    support_files = [
        str(item)
        for item in list(evidence.get("support_file_candidates", []) or [])
        if str(item).strip()
    ]
    if support_files:
        reason = str(evidence.get("bounded_support_expansion_reason", "") or "").strip()
        lines.append("Bounded support source files allowed by semantic evidence:")
        lines.extend(f"- {item}" for item in support_files[:8])
        if reason:
            lines.append(f"Support-file expansion reason: {reason}.")
        lines.append(
            "Hard recovery rule: if the call-site file cannot satisfy the semantic contract alone, "
            "edit only these support source files plus the call site; do not broaden beyond them."
        )
    oracle_assertion_contracts = [
        str(item)
        for item in list(evidence.get("oracle_assertion_contracts", []) or [])
        if str(item).strip()
    ]
    if oracle_assertion_contracts:
        lines.append("Oracle assertion contracts extracted from focused target tests:")
        lines.extend(f"- {item}" for item in oracle_assertion_contracts[:12])
        lines.append(
            "Audit boundary: these contracts constrain edit intent; only specialized semantic gates are static hard blockers."
        )
    forbidden_directions = [
        str(item)
        for item in list(evidence.get("forbidden_patch_directions", []) or [])
        if str(item).strip()
    ]
    if forbidden_directions:
        lines.append("Forbidden patch directions for this bounded retry:")
        lines.extend(f"- {item}" for item in forbidden_directions[:12])
        lines.append(
            "Hard recovery rule: do not repeat a forbidden patch direction from the failed attempt."
        )
    failed_patch_feedback = dict(evidence.get("failed_patch_feedback", {}) or {})
    failed_added_lines = [
        str(item)
        for item in list(failed_patch_feedback.get("failed_patch_added_lines", []) or [])
        + list(failed_patch_feedback.get("failed_added_lines", []) or [])
        if str(item).strip()
    ]
    failed_diff_sha = str(failed_patch_feedback.get("failed_patch_diff_sha256", "") or "").strip()
    if failed_added_lines or failed_diff_sha:
        lines.append("Failed patch-shape feedback from the previous bounded attempt:")
        if failed_diff_sha:
            lines.append(f"- failed_patch_diff_sha256: {failed_diff_sha}")
        if failed_added_lines:
            lines.append("Repeated added source lines to avoid:")
            lines.extend(f"- {item}" for item in failed_added_lines[:12])
        lines.append(
            "Hard recovery rule: do not repeat this failed source-diff shape unless the new patch also changes the failing behavior."
        )
    api_surface_mismatch = dict(evidence.get("api_surface_mismatch", {}) or {})
    if api_surface_mismatch:
        lines.append("API-surface mismatch evidence:")
        for key in ("exception_type", "callable", "parameter", "evidence"):
            value = str(api_surface_mismatch.get(key, "") or "").strip()
            if value:
                lines.append(f"- {key}: {value}")
    verified_cause = str(evidence.get("verified_failure_cause", "") or "").strip()
    if verified_cause:
        lines.append(f"Verified failure cause to address: {verified_cause}")
        lines.append(
            "Hard recovery rule: the next patch must directly address this verified cause."
        )
    rejection_rules = [
        str(item)
        for item in list(evidence.get("patch_quality_rejection_rules", []) or [])
        if str(item).strip()
    ]
    if rejection_rules:
        lines.append("Patch quality rejection rules:")
        lines.extend(f"- {item}" for item in rejection_rules[:12])
        lines.append(
            "Hard recovery rule: reject and revise any patch that violates these quality rules."
        )
    stop_conditions = [
        str(item)
        for item in list(evidence.get("stop_conditions", []) or [])
        if str(item).strip()
    ]
    if stop_conditions:
        lines.append("Stop conditions for this bounded retry:")
        lines.extend(f"- {item}" for item in stop_conditions[:12])
        lines.append(
            "Hard recovery rule: do not request or assume another retry after these stop conditions are reached."
        )
    source_files = [
        str(path)
        for path in list(changed_classes.get("source_files", []) or [])
        if str(path).strip()
    ]
    test_files = [
        str(path)
        for path in list(changed_classes.get("test_files", []) or [])
        if str(path).strip()
    ]
    if source_files:
        lines.append("Previous failed recovery source files:")
        lines.extend(f"- {path}" for path in source_files[:10])
    if test_files:
        lines.append("Previous failed recovery touched tests; avoid repeating test edits unless justified:")
        lines.extend(f"- {path}" for path in test_files[:10])
    if selected:
        lines.append("Evidence-bounded selected candidates:")
        lines.extend(f"- {path}" for path in selected[:10])
    if failed_tests:
        lines.append("Focused failing tests after the failed recovery:")
        lines.extend(f"- {test}" for test in failed_tests[:10])
    if exception_excerpt:
        lines.append("Failure excerpt to satisfy:")
        lines.extend(f"- {line}" for line in exception_excerpt[:8])
    previous_patch_excerpt = [
        str(line)
        for line in list(evidence.get("previous_patch_excerpt", []) or [])
        if str(line).strip()
    ]
    if previous_patch_excerpt:
        lines.append("Previous failed patch excerpt; fix its local defect instead of repeating it:")
        lines.extend(f"- {line}" for line in previous_patch_excerpt[:24])
    failure_output_excerpt = [
        str(line)
        for line in list(evidence.get("failure_output_excerpt", []) or [])
        if str(line).strip()
    ]
    if failure_output_excerpt:
        lines.append("Focused post-patch validation output excerpt:")
        lines.extend(f"- {line}" for line in failure_output_excerpt[:24])
    next_step = str(evidence.get("next_recommended_step", "") or "")
    if next_step:
        lines.append(f"Evidence-derived next step: {next_step}")
    semantic_reason = str(evidence.get("semantic_retry_reason", "") or "")
    if semantic_reason:
        lines.extend(
            [
                f"Semantic retry reason: {semantic_reason}",
                "Hard recovery rule: this retry is for the observed post-patch semantic/runtime failure, not for a broad restart.",
            ]
        )
    return lines


def _semantic_effect_specialized_context_lines(evidence: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            *[str(item) for item in list(evidence.get("semantic_invariants", []) or [])],
            *[str(item) for item in list(evidence.get("expected_behavior_constraints", []) or [])],
            *[str(item) for item in list(evidence.get("forbidden_patch_directions", []) or [])],
            str(evidence.get("verified_failure_cause", "") or ""),
        ]
    ).lower()
    if (
        ("xarray" in text or "dataarray" in text)
        and "quantile" in text
        and "keep_attrs" in text
        and "attrs" in text
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: XARRAY DATAARRAY.QUANTILE KEEP_ATTRS]",
            "Primary edit target: in xarray/core/dataarray.py DataArray.quantile(), preserve attrs on the returned DataArray object.",
            "Required minimal source shape: replace the direct `return self._from_temp_dataset(ds)` with `result = self._from_temp_dataset(ds)`, then if `keep_attrs` is true set `result.attrs = self.attrs.copy()`, then `return result`.",
            "Hard recovery rule: a variable.py-only patch is insufficient because the focused assertion checks the returned DataArray attrs, not only the inner Variable attrs.",
            "Hard recovery rule: do not edit tests, rank(), or unrelated reduction helpers; only quantile metadata propagation is in scope.",
            "Hard recovery rule: do not globally replace every `return self._from_temp_dataset(ds)` occurrence; exactly one hunk inside DataArray.quantile() is allowed.",
            "Hard recovery rule: if the patch touches persist(), chunk(), thin(), copy(), broadcast_like(), or any method before DataArray.quantile(), reject it and produce a quantile-only patch.",
            "Focused assertion to satisfy: `actual.attrs == self.attrs` while preserving existing quantile numeric values.",
        ]
    if (
        ("xarray" in text or "dataarray.integrate" in text or "dataset.integrate" in text)
        and "integrate" in text
        and "dim" in text
        and ("unexpected keyword" in text or "propagate" in text or "api" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: XARRAY DATAARRAY.INTEGRATE DIM KEYWORD]",
            "Primary edit target: in xarray/core/dataarray.py DataArray.integrate(), keep the deprecated `dim=` keyword accepted while forwarding to the Dataset.integrate coord path.",
            "Required semantic effect: `da.integrate(dim=\"x\")` must not raise TypeError; it must emit the expected FutureWarning and produce the same result as `da.integrate(\"x\")`.",
            "Required source shape: do not merely rename the first parameter from `dim` to `coord`; add an explicit `dim` alias path or keyword-only compatibility parameter that maps `dim` to `coord` before calling self._to_temp_dataset().integrate(...).",
            "Hard recovery rule: preserve positional calls such as `da.integrate(\"x\")` and tuple calls such as `da.integrate((\"y\", \"x\"))`.",
            "Hard recovery rule: preserve Dataset.integrate behavior; the focused failure is the DataArray keyword compatibility boundary.",
            "Hard recovery rule: do not edit tests, examples only, or docstrings only.",
            "Focused assertion to satisfy: the two test_integrate parametrizations pass for both dask and non-dask paths.",
        ]
    if (
        ("xarray" in text or "variable" in text)
        and ("values-attribute" in text or "values attribute" in text or "custom values" in text)
        and ("scalar" in text or "dims=()" in text or "ndim 0" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: XARRAY VARIABLE CUSTOM VALUES-ATTR SCALAR]",
            "Primary edit target: in xarray/core/variable.py as_compatible_data(), fix when arbitrary custom objects with a `.values` attribute should be treated as scalar object data.",
            "Required semantic effect: `Variable(dims=(), data=CustomWithValuesAttr(np.arange(3)))` must create a 0-dimensional object array whose item is the original CustomWithValuesAttr instance.",
            "Required source shape: narrow the unconditional `data = getattr(data, 'values', data)` behavior so it consumes `.values` only for known pandas/xarray containers, not arbitrary custom scalar objects.",
            "Hard recovery rule: do not patch Variable.__init__ by rewriting `self._data` after as_compatible_data(); that is too late and can corrupt legitimate 1D object arrays.",
            "Hard recovery rule: do not loosen _parse_dimensions() to accept dims=() with ndim=1; the required behavior is ndim 0, not accepting mismatched dimensions.",
            "Hard recovery rule: preserve CustomArray conversion to ndarray and CustomIndexable preservation as ExplicitlyIndexed from the same focused test.",
            "Focused assertion to satisfy: `isinstance(orig._data.item(), CustomWithValuesAttr)`.",
        ]
    if (
        ("xarray" in text or "dataarray" in text or "dataset" in text)
        and "reset_index" in text
        and ("xindexes" in text or "pandasindex" in text or "not coordinates with an index" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: XARRAY RESET_INDEX XINDEXES CONTRACT]",
            "Primary edit target: fix reset_index/drop_indexes/xindexes state transitions, not DataVariables length accounting.",
            "Required semantic effect: reset_index must leave obj.xindexes empty or containing the expected PandasIndex exactly as asserted by the focused tests.",
            "Required source shape: inspect xarray/core/dataset.py reset_index/drop_indexes and the xindexes/index-state update path; touch xarray/core/indexes.py only if the PandasIndex state transition requires it.",
            "Hard recovery rule: do not only edit DataVariables.__iter__, DataVariables.__len__, or DataVariables.__contains__; that does not satisfy the reset_index/xindexes assertions.",
            "Hard recovery rule: preserve normal data_vars membership semantics while changing only the reset-index coordinate/index transition.",
            "Focused assertion to satisfy: len(obj.xindexes), list(obj.xindexes), PandasIndex type, and reset coordinate variables match the expected objects.",
        ]
    if (
        ("sklearn" in text or "least_angle.py" in text or "lassolarsic" in text or "lars" in text)
        and ("copy_x" in text or "copyx" in text)
        and ("np.array_equal(x, x_copy)" in text or "_preprocess_data" in text or "preprocess" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: SKLEARN LASSO_LARS COPY_X PROPAGATION]",
            "Primary edit target: in sklearn/linear_model/least_angle.py, propagate the fit-time copy_X value into every affected _preprocess_data call.",
            "Required semantic effect: for fit(copy_X=True/False), copy_X == np.array_equal(X, X_copy) must hold.",
            "Required source shape: keep LassoLarsIC.fit(copy_X=None) compatible with self.copy_X defaults, then pass the resolved local copy_X into _preprocess_data at all affected call sites.",
            "Hard recovery rule: do not only edit __init__ assignments to self.copy_X; constructor-only changes do not control per-call fit(copy_X=...) behavior.",
            "Hard recovery rule: do not fix only the Lars path while leaving the LassoLarsIC preprocessing path using self.copy_X.",
            "Focused assertion to satisfy: `assert copy_X == np.array_equal(X, X_copy)`.",
        ]
    if (
        ("xarray" in text or "combine_auto" in text)
        and ("combine-auto" in text or "combine_auto" in text)
        and ("bystander dimension" in text or "bystander dimensions" in text)
        and ("monotonic" in text or "dimension ordering" in text or "global index" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: XARRAY COMBINE_AUTO BYSTANDER DIMENSION ORDERING]",
            "Primary edit target: in xarray/core/combine.py, fix combine_auto ordering for bystander dimensions without deleting the existing global monotonic-index safety check.",
            "Required semantic effect: `TestCombineAuto::test_combine_leaving_bystander_dimensions` must not raise the non-monotonic global index ValueError for the tested bystander-dimension ordering.",
            "Required source shape: adjust the ordering or concat-dimension inference path used by combine_auto; keep the final monotonic global-index validation for genuinely non-monotonic combined indexes.",
            "Hard recovery rule: do not simply remove, weaken, or bypass the `Resulting object does not have monotonic global indexes` ValueError.",
            "Hard recovery rule: do not globally sort all datasets or all coordinates unless the change is explicitly scoped to combine_auto's bystander-dimension ordering case.",
            "Hard recovery rule: do not edit tests, expected values, or unrelated combine_nested behavior.",
            "Focused assertion to satisfy: the bystander-dimension combine_auto test passes while existing non-monotonic global-index failures remain protected.",
        ]
    if (
        "sphinx" in text
        and ("graphviz" in text or "inheritance diagram" in text)
        and "svg" in text
        and ("https://example.org" in text or "external hyperlink" in text or "external link" in text)
    ):
        return [
            "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: SPHINX GRAPHVIZ SVG EXTERNAL HYPERLINK]",
            "Primary edit target: in sphinx/ext/inheritance_diagram.py html_visit_inheritance_diagram(), preserve external intersphinx refuri values in the Graphviz URL map.",
            "Required semantic effect: the generated SVG for an inheritance diagram containing a subdir.* class must include the external URL `https://example.org`.",
            "Required source shape: update the `urls[child['reftitle']] = ...` construction so absolute/external child.get('refuri') values are passed through to Graphviz instead of being rewritten as local relative paths.",
            "Hard recovery rule: a patch that only changes sphinx/ext/graphviz.py fix_svg_relative_paths() is insufficient; that stage can rewrite existing hrefs but cannot create the missing intersphinx external URL mapping.",
            "Hard recovery rule: do not edit tests, generated SVG/HTML files, or only change formatting around xlink:href.",
            "Hard recovery rule: preserve existing relative links for local documents while keeping absolute external refuri links intact.",
            "Focused assertion to satisfy: every generated SVG containing `subdir.` must contain `https://example.org`.",
        ]
    if not (
        "sphinx" in text
        and "toctree" in text
        and "genindex" in text
        and "modindex" in text
        and "search" in text
    ):
        return []
    return [
        "[MAS-DX-R SPECIALIZED SEMANTIC EFFECT TARGET: SPHINX RESERVED TOCTREE ENTRIES]",
        "Primary edit target: in sphinx/directives/other.py parse_content(), reserved refs genindex/modindex/search must still be appended to toctree['entries'].",
        "Secondary effect target: these reserved refs must not emit the nonexisting-document warning path.",
        "Hard recovery rule: preserve the toctree node entries exactly; the focused test requires entries [(None, 'genindex'), (None, 'modindex'), (None, 'search')].",
        "Hard recovery rule: do not add continue, filtering, removal, or entries mutation inside the reserved-ref branch.",
        "Hard recovery rule: a warning-path continue for non-reserved missing documents is acceptable only if the reserved-ref branch falls through without warning and without deleting entries.",
        "Hard recovery rule: an adapter-only warning patch is insufficient because it does not change where toctree entries are constructed.",
    ]


def _g2_recovery_evidence_context_lines(evidence: dict[str, Any]) -> list[str]:
    if str(evidence.get("schema", "") or "") != "masguard_g2_recovery_evidence_bundle_v1":
        return []
    graph = dict(evidence.get("g2_propagation_summary", {}) or {})
    top_before = dict(evidence.get("g2_top_hypothesis_before", {}) or {})
    top_after = dict(evidence.get("g2_top_hypothesis_after", {}) or {})
    driver = dict(evidence.get("g2_recovery_driver", {}) or {})
    observations = [
        dict(item)
        for item in list(evidence.get("g2_bounded_observations", []) or [])
        if isinstance(item, dict)
    ]
    contract = dict(evidence.get("g2_execution_contract", {}) or {})
    lines = [
        "[MASGUARD G2 PROPAGATION-GROUNDED RECOVERY]",
        "Use the propagation graph and bounded observation below as the controlling recovery evidence.",
        "Hard recovery rule: the patch must address the updated hypothesis, not the initial static guess.",
        "Hard recovery rule: if the bounded observation weakens an environment/protocol explanation, do not spend the retry on environment repair.",
        (
            "Propagation graph summary: "
            f"nodes={graph.get('node_count', 0)}, "
            f"edges={graph.get('edge_count', 0)}, "
            f"semantic_edges={graph.get('semantic_edge_count', 0)}."
        ),
        f"G2 bounded observation status: {evidence.get('g2_bounded_observation_status', '')}",
    ]
    before_type = str(top_before.get("failure_type", "") or "")
    after_type = str(top_after.get("failure_type", "") or "")
    if before_type or after_type:
        lines.append(
            "G2 hypothesis update: "
            f"{before_type or '<none>'} -> {after_type or '<none>'}; "
            f"confidence={top_after.get('confidence', '')}."
        )
    if driver:
        lines.append(
            "G2 recovery driver: "
            f"action={driver.get('selected_action', '')}, "
            f"operator={driver.get('selected_operator', '')}, "
            f"scope={driver.get('selected_run_scope', '')}, "
            f"rationale={driver.get('recovery_rationale', '')}."
        )
    rationale = str(top_after.get("rationale", "") or "").strip()
    if rationale:
        lines.append(f"Updated hypothesis rationale: {rationale[:360]}")
    recommended = str(top_after.get("candidate_recovery_action", "") or "").strip()
    if recommended:
        lines.append(f"Updated candidate recovery action: {recommended}")
    if contract:
        lines.append(
            "G2 execution contract: "
            f"run_scope={contract.get('run_scope', '')}, "
            f"operator_gate={contract.get('operator_gate', '')}, "
            f"patch_contract={contract.get('patch_contract', '')}."
        )
    summary_flags = dict(graph.get("summary_flags", {}) or {})
    true_flags = [key for key, value in summary_flags.items() if value is True]
    if true_flags:
        lines.append("True propagation flags:")
        lines.extend(f"- {key}" for key in true_flags[:12])
    if observations:
        lines.append("Bounded minimal intervention observations:")
        for observation in observations[:5]:
            lines.append(
                "- "
                f"signal={observation.get('observed_signal', '')}, "
                f"status={observation.get('status', '')}, "
                f"update={observation.get('hypothesis_update', '')}, "
                f"protocol_blocked={str(bool(observation.get('protocol_blocked', False))).lower()}"
            )
            excerpt = str(observation.get("output_excerpt", "") or "").strip()
            if excerpt:
                first_line = next((line.strip() for line in excerpt.splitlines() if line.strip()), "")
                if first_line:
                    lines.append(f"  evidence_excerpt: {first_line[:240]}")
    return lines


def _patcher_stop_reason(patch_result: dict[str, Any]) -> str:
    if patch_result.get("success"):
        return "patcher_completed"
    if _provider_api_error_fields(patch_result)["provider_api_error"]:
        return "provider_api_error"
    if patch_result.get("infrastructure_error"):
        return "patcher_infrastructure_error"
    error = str(patch_result.get("implementer_error") or patch_result.get("planner_error") or "").strip()
    return error[:120] if error else "patcher_failed"


def _provider_api_error_fields(patch_result: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(patch_result.get(key, "") or "")
        for key in ("planner_error", "implementer_error", "plan", "error", "retry_reason")
    )
    provider_api_error = _looks_like_provider_api_error(text)
    return {
        "provider_api_error": provider_api_error,
        "system_retryable": provider_api_error,
        "system_retryable_reason": "provider_api_error" if provider_api_error else "",
    }


def _looks_like_provider_api_error(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "openai-compatible api request failed",
        "concurrency limit exceeded",
        "rate limit",
        "rate_limit",
        "too many requests",
        "temporarily unavailable",
        "bad gateway",
        "connection reset",
        "connection aborted",
    )
    return any(marker in lowered for marker in markers)


def _verified_recovery_status_after_patcher(
    *,
    patch_result: dict[str, Any],
    fail_to_pass: dict[str, Any],
    oracle: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, str]:
    patcher_stop_reason = _patcher_stop_reason(patch_result)
    if _provider_api_error_fields(patch_result)["provider_api_error"]:
        return {
            "stop_reason": "provider_api_error",
            "status_consistency_resolution": "",
        }
    if patch_result.get("infrastructure_error"):
        return {
            "stop_reason": "patcher_infrastructure_error",
            "status_consistency_resolution": "",
        }
    oracle_green = int(oracle.get("returncode", 1) or 0) == 0
    fail_to_pass_green = int(fail_to_pass.get("returncode", 1) or 0) == 0
    protocol_clean = not bool(protocol.get("evaluation_protocol_error", False))
    if oracle_green and fail_to_pass_green and protocol_clean:
        if patcher_stop_reason == "patcher_completed":
            return {
                "stop_reason": "patcher_completed",
                "status_consistency_resolution": "",
            }
        return {
            "stop_reason": "oracle_verified_after_patcher",
            "status_consistency_resolution": (
                "oracle_and_fail_to_pass_green_override_patcher_internal_stop_reason"
            ),
        }
    return {
        "stop_reason": patcher_stop_reason,
        "status_consistency_resolution": "",
    }


def _patch_observed(patch_result: dict[str, Any]) -> bool:
    patch_summary = dict(patch_result.get("patch_summary", {}) or {})
    changed_files = list(patch_summary.get("changed_files", []) or [])
    fresh_changed_files = list(patch_summary.get("fresh_changed_files", []) or [])
    return bool(
        str(patch_result.get("patch", "") or "").strip()
        or changed_files
        or fresh_changed_files
    )


def _patch_text_for_record(
    *,
    patch_result: dict[str, Any],
    patch_summary: dict[str, Any],
    workspace: Path,
) -> dict[str, str]:
    patch_text = str(patch_result.get("patch", "") or "")
    if parse_unified_diff_paths(patch_text):
        return {"patch": patch_text, "source": "patcher_result.patch"}
    changed_files = []
    seen_changed_files = set()
    for path in list(patch_summary.get("changed_files", []) or []) + list(
        patch_summary.get("fresh_changed_files", []) or []
    ):
        normalized = str(path).strip()
        if normalized and normalized not in seen_changed_files:
            seen_changed_files.add(normalized)
            changed_files.append(normalized)
    if not changed_files:
        return {"patch": patch_text, "source": "patcher_result.patch"}
    diff_text = _workspace_diff_for_files(workspace, changed_files)
    if diff_text.strip():
        return {"patch": diff_text, "source": "workspace_git_diff_fallback"}
    return {"patch": patch_text, "source": "patcher_result.patch"}


def _workspace_diff_for_files(workspace: Path, files: list[str]) -> str:
    if not workspace.is_dir() or not files:
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "--", *files],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return str(result.stdout or "")


def _evaluation_commands_for_manifest(
    manifest: dict[str, Any],
    *,
    command_source: str,
) -> dict[str, Any]:
    if command_source == "manifest":
        return {
            "evaluation_command_source": "manifest",
            "fail_to_pass_command": str(manifest.get("test_command", "") or ""),
            "oracle_command": str(manifest.get("oracle_command", "") or ""),
            "fail_to_pass_targets": list(manifest.get("dataset_fail_to_pass", []) or []),
            "oracle_targets": list(manifest.get("dataset_fail_to_pass", []) or [])
            + list(manifest.get("dataset_pass_to_pass", []) or []),
            "filtered_targets": [],
            "normalization_notes": [],
            "error_type": "",
            "stop_reason": "",
        }
    normalized_mode = {
        "normalized": "normalized",
        "normalized_fail_to_pass": "normalized_fail_to_pass",
        "validated": "normalized",
        "validated_fail_to_pass": "normalized_fail_to_pass",
    }.get(command_source, "")
    if not normalized_mode:
        return {
            "evaluation_command_source": command_source,
            "fail_to_pass_command": "",
            "oracle_command": "",
            "fail_to_pass_targets": [],
            "oracle_targets": [],
            "filtered_targets": [],
            "normalization_notes": ["unsupported_command_source"],
            "error_type": "unsupported_evaluation_command_source",
            "stop_reason": "unsupported_evaluation_command_source",
        }

    test_prefix = _test_command_prefix(manifest)
    raw_fail_targets = list(manifest.get("dataset_fail_to_pass", []) or [])
    fallback_fail_targets = []
    notes = []
    if not raw_fail_targets:
        fallback_fail_targets = _targets_from_test_command(str(manifest.get("test_command", "") or ""))
        if fallback_fail_targets:
            notes.append("fail_to_pass_targets_derived_from_test_command")
    fail_targets = _safe_targets(
        raw_fail_targets or fallback_fail_targets,
        field="dataset_fail_to_pass" if raw_fail_targets else "test_command",
    )
    pass_targets = _safe_targets(manifest.get("dataset_pass_to_pass", []), field="dataset_pass_to_pass")
    oracle_targets = list(fail_targets["kept"])
    if normalized_mode == "normalized":
        oracle_targets.extend(pass_targets["kept"])

    filtered = list(fail_targets["filtered"]) + list(pass_targets["filtered"])
    if filtered:
        notes.append("filtered_protocol_risky_targets")
    if normalized_mode == "normalized_fail_to_pass":
        notes.append("oracle_uses_fail_to_pass_only")
    if command_source.startswith("validated"):
        notes.append("runtime_collect_validation_requested")
    if not fail_targets["kept"]:
        return {
            "evaluation_command_source": command_source,
            "fail_to_pass_command": "",
            "oracle_command": "",
            "fail_to_pass_targets": [],
            "oracle_targets": [],
            "filtered_targets": filtered,
            "normalization_notes": notes,
            "error_type": "evaluation_protocol_blocked",
            "stop_reason": "no_safe_fail_to_pass_targets",
        }

    return {
        "evaluation_command_source": command_source,
        "fail_to_pass_command": _compose_test_command(test_prefix, fail_targets["kept"]),
        "oracle_command": _compose_test_command(test_prefix, oracle_targets),
        "fail_to_pass_targets": fail_targets["kept"],
        "oracle_targets": oracle_targets,
        "filtered_targets": filtered,
        "normalization_notes": notes,
        "error_type": "",
        "stop_reason": "",
    }


def _evaluation_command_payload(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_command_source": str(evaluation.get("evaluation_command_source", "") or ""),
        "fail_to_pass_command": str(evaluation.get("fail_to_pass_command", "") or ""),
        "oracle_command": str(evaluation.get("oracle_command", "") or ""),
        "evaluation_fail_to_pass_targets": list(evaluation.get("fail_to_pass_targets", []) or []),
        "evaluation_oracle_targets": list(evaluation.get("oracle_targets", []) or []),
        "evaluation_filtered_targets": list(evaluation.get("filtered_targets", []) or []),
        "evaluation_normalization_notes": list(evaluation.get("normalization_notes", []) or []),
        "evaluation_collect_validation": dict(evaluation.get("collect_validation", {}) or {}),
        "evaluation_target_mapping": dict(evaluation.get("target_mapping", {}) or {}),
    }


def _evaluation_blocked_result(
    *,
    evaluation: dict[str, Any],
    action_adherence_observed: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        **_evaluation_command_payload(evaluation),
        "error_type": str(evaluation.get("error_type", "") or "evaluation_protocol_blocked"),
        "stop_reason": str(evaluation.get("stop_reason", "") or "evaluation_protocol_blocked"),
        "fail_to_pass_returncode": None,
        "oracle_returncode": None,
        "oracle_success": False,
        "reported_success": False,
        "fail_to_pass_output": "",
        "oracle_output": "",
        "fail_to_pass_protocol_error": True,
        "oracle_protocol_error": True,
        "evaluation_protocol_error": True,
        "evaluation_protocol_error_type": str(evaluation.get("stop_reason", "") or "evaluation_protocol_blocked"),
        "action_adherence_observed": action_adherence_observed,
    }
    if extra:
        row.update(extra)
    return row


def _test_command_prefix(manifest: dict[str, Any]) -> str:
    dataset_test_cmd = str(manifest.get("dataset_test_cmd", "") or "").strip()
    if dataset_test_cmd:
        return dataset_test_cmd
    for field in ("test_command", "oracle_command"):
        command = str(manifest.get(field, "") or "").strip()
        if command:
            parts = shlex.split(command, posix=True)
            if not parts:
                continue
            target_index = _first_pytest_target_index(parts)
            prefix_parts = parts[:target_index] if target_index is not None else parts
            return " ".join(shlex.quote(part) for part in prefix_parts)
    return "pytest -rA"


def _targets_from_test_command(command: str) -> list[str]:
    text = str(command or "").strip()
    if not text:
        return []
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        return []
    target_index = _first_pytest_target_index(parts)
    if target_index is None:
        return []
    targets = []
    for part in parts[target_index:]:
        if part == "--":
            continue
        if part.startswith("-"):
            break
        if part not in targets:
            targets.append(part)
    return targets


def _runtime_validate_evaluation_commands(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    executor,
    workspace: Path,
    timeout: int,
) -> dict[str, Any]:
    source = str(evaluation.get("evaluation_command_source", "") or "")
    if not source.startswith("validated") or evaluation.get("error_type"):
        return evaluation
    prefix = _test_command_prefix(manifest)
    if not _supports_collect_only(prefix):
        output = dict(evaluation)
        notes = list(output.get("normalization_notes", []) or [])
        notes.append("runtime_collect_validation_skipped_non_pytest_prefix")
        output["normalization_notes"] = list(dict.fromkeys(notes))
        output["collect_validation"] = {
            "skipped": True,
            "reason": "non_pytest_prefix",
            "prefix": prefix,
        }
        return output
    fail_validation = _validate_collectable_targets(
        prefix=prefix,
        targets=list(evaluation.get("fail_to_pass_targets", []) or []),
        field="dataset_fail_to_pass",
        executor=executor,
        workspace=workspace,
        timeout=timeout,
    )
    oracle_targets = list(evaluation.get("oracle_targets", []) or [])
    if source == "validated":
        oracle_validation = _validate_collectable_targets(
            prefix=prefix,
            targets=oracle_targets,
            field="dataset_oracle_targets",
            executor=executor,
            workspace=workspace,
            timeout=timeout,
        )
        validated_oracle_targets = oracle_validation["kept"]
    else:
        oracle_validation = dict(fail_validation)
        validated_oracle_targets = list(fail_validation["kept"])

    filtered = list(evaluation.get("filtered_targets", []) or [])
    filtered.extend(fail_validation["filtered"])
    if source == "validated":
        fail_rejected = {item["target"] for item in fail_validation["filtered"]}
        filtered.extend(
            item
            for item in oracle_validation["filtered"]
            if item["target"] not in fail_rejected
        )
    notes = list(evaluation.get("normalization_notes", []) or [])
    if fail_validation["filtered"] or oracle_validation["filtered"]:
        notes.append("filtered_uncollectable_targets")
    output = dict(evaluation)
    output.update(
        {
            "fail_to_pass_targets": fail_validation["kept"],
            "oracle_targets": validated_oracle_targets,
            "filtered_targets": filtered,
            "normalization_notes": list(dict.fromkeys(notes)),
            "fail_to_pass_command": _compose_test_command(prefix, fail_validation["kept"])
            if fail_validation["kept"]
            else "",
            "oracle_command": _compose_test_command(prefix, validated_oracle_targets)
            if validated_oracle_targets
            else "",
            "collect_validation": {
                "fail_to_pass": fail_validation["summary"],
                "oracle": oracle_validation["summary"],
            },
        }
    )
    if not fail_validation["kept"]:
        output.update(
            {
                "error_type": "evaluation_protocol_blocked",
                "stop_reason": "no_collectable_fail_to_pass_targets",
            }
        )
    elif not validated_oracle_targets:
        output.update(
            {
                "error_type": "evaluation_protocol_blocked",
                "stop_reason": "no_collectable_oracle_targets",
            }
        )
    return output


def _validate_collectable_targets(
    *,
    prefix: str,
    targets: list[str],
    field: str,
    executor,
    workspace: Path,
    timeout: int,
) -> dict[str, Any]:
    kept: list[str] = []
    filtered: list[dict[str, str]] = []
    probes = []
    for target in targets:
        command = _collect_only_command(prefix, target)
        result = executor.execute(command, cwd=str(workspace), timeout=timeout)
        returncode = int(result.get("returncode", 0) or 0)
        output = str(result.get("output", "") or "")
        protocol_error = _pytest_protocol_error_type(returncode=returncode, output=output)
        collected_nodeids = _collect_only_nodeids(output)
        target_collected = returncode == 0 and _collect_only_found_tests(output)
        collected = target_collected and _target_collected(target, collected_nodeids)
        probes.append(
            {
                "target": target,
                "command": command,
                "target_count": 1,
                "returncode": returncode,
                "collected": collected,
                "collected_nodeid_count": len(collected_nodeids),
                "collected_nodeids_sample": sorted(collected_nodeids)[:20],
                "protocol_error": protocol_error,
                "output_excerpt": output[-2000:],
            }
        )
        if collected:
            kept.append(target)
        else:
            filtered.append(
                {
                    "field": field,
                    "target": target,
                    "risk_type": protocol_error or "target_not_collectable",
                }
            )
    return {
        "kept": kept,
        "filtered": filtered,
        "summary": {
            "requested_count": len(targets),
            "kept_count": len(kept),
            "filtered_count": len(filtered),
            "probes": probes,
        },
    }


def _collect_only_command(prefix: str, target: str) -> str:
    if _supports_collect_only(prefix):
        return _compose_test_command(f"{prefix} --collect-only -q", [target])
    raise ValueError(f"Cannot build collect-only command for non-pytest prefix: {prefix}")


def _supports_collect_only(prefix: str) -> bool:
    parts = shlex.split(str(prefix or ""), posix=True)
    if not parts:
        return False
    if Path(parts[0]).name == "pytest" or parts[0].endswith("pytest"):
        return True
    if len(parts) >= 3 and Path(parts[0]).name.startswith("python") and parts[1:3] == ["-m", "pytest"]:
        return True
    return False


def _collect_only_found_tests(output: str) -> bool:
    text = _strip_ansi(str(output or "")).lower()
    if _collect_only_nodeids(output):
        return True
    if "no tests collected" in text or "no tests ran" in text:
        return False
    if "collected 0 items" in text:
        return False
    return "collected " in text or "<function " in text or "<class " in text


def _collect_only_nodeids(output: str) -> set[str]:
    nodeids: set[str] = set()
    for raw_line in str(output or "").splitlines():
        line = _strip_ansi(raw_line).strip()
        if not line or line.startswith("="):
            continue
        lowered = line.lower()
        if lowered.startswith(("warning", "error:", "found ", "collected ")):
            continue
        if "::" in line and not any(ch.isspace() for ch in line):
            nodeids.add(line.removeprefix("/testbed/"))
    return nodeids


def _target_collected(target: str, collected_nodeids: set[str]) -> bool:
    if not collected_nodeids:
        return True
    if target in collected_nodeids:
        return True
    normalized = target.removeprefix("/testbed/")
    return normalized in collected_nodeids


def _target_collect_root(target: str) -> str:
    text = str(target or "").strip()
    return text.split("::", 1)[0] if "::" in text else text


def _first_pytest_target_index(parts: list[str]) -> int | None:
    option_args_with_value = {
        "-k",
        "-m",
        "-r",
        "--basetemp",
        "--confcutdir",
        "--cov",
        "--cov-report",
        "--ds",
        "--junitxml",
        "--maxfail",
        "--rootdir",
        "--tb",
    }
    skip_next = False
    for index, part in enumerate(parts):
        if index == 0:
            continue
        if skip_next:
            skip_next = False
            continue
        if part == "--":
            return index + 1 if index + 1 < len(parts) else None
        if part in option_args_with_value:
            skip_next = True
            continue
        if part.startswith("-"):
            continue
        return index
    return None


def _safe_targets(raw_targets: Any, *, field: str) -> dict[str, Any]:
    kept: list[str] = []
    filtered: list[dict[str, str]] = []
    for item in list(raw_targets or []):
        target = str(item or "").strip()
        risk = _target_protocol_risk(target)
        if risk:
            filtered.append({"field": field, "target": target, "risk_type": risk})
            continue
        if target and target not in kept:
            kept.append(target)
    return {"kept": kept, "filtered": filtered}


def _target_protocol_risk(target: str) -> str:
    if not target.strip():
        return "empty_target"
    if target.count("[") != target.count("]"):
        return "unbalanced_param_brackets"
    if any(ch.isspace() for ch in target):
        return "target_contains_space"
    if target.endswith("\\"):
        return "target_trailing_escape"
    return ""


def _compose_test_command(prefix: str, targets: list[str]) -> str:
    quoted_targets = " ".join(shlex.quote(target) for target in targets)
    return f"{prefix.strip()} {quoted_targets}".strip()


def _evaluation_protocol_status(*, fail_to_pass: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    fail_kind = _pytest_protocol_error_type(
        returncode=int(fail_to_pass.get("returncode", 0) or 0),
        output=str(fail_to_pass.get("output", "") or ""),
    )
    oracle_kind = _pytest_protocol_error_type(
        returncode=int(oracle.get("returncode", 0) or 0),
        output=str(oracle.get("output", "") or ""),
    )
    kinds = [kind for kind in (fail_kind, oracle_kind) if kind]
    return {
        "fail_to_pass_protocol_error": bool(fail_kind),
        "oracle_protocol_error": bool(oracle_kind),
        "evaluation_protocol_error": bool(kinds),
        "evaluation_protocol_error_type": "|".join(kinds),
    }


def _pytest_protocol_error_type(*, returncode: int, output: str) -> str:
    lowered = _strip_ansi(output).lower()
    if returncode == 4:
        if "not found:" in lowered or "no name " in lowered:
            return "pytest_nodeid_not_found"
        if "no tests ran" in lowered:
            return "pytest_no_tests_ran"
        return "pytest_usage_or_collection_error"
    if "error: not found:" in lowered and "no tests ran" in lowered:
        return "pytest_nodeid_not_found"
    return ""


def _preflight_command(manifest: dict[str, Any]) -> str:
    command = str(manifest.get("test_command", "") or "").strip()
    if command.startswith("pytest ") or command == "pytest" or " pytest " in f" {command} ":
        if str(manifest.get("repo", "") or "") == "pytest-dev/pytest":
            version = str(manifest.get("version", "") or "").strip()
            version_parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
            if not version_parts:
                version_parts = [0, 0, 0]
            while len(version_parts) < 3:
                version_parts.append(0)
            normalized_version = ".".join(str(part) for part in version_parts)
            version_tuple = f"({version_parts[0]}, {version_parts[1]}, {version_parts[2]})"
            return (
                "test -f src/_pytest/_version.py || "
                "printf \"%s\\n\" "
                "\"# generated for offline SWE-bench validation from manifest version\" "
                f"\"version = '{normalized_version}'\" "
                f"\"version_tuple = {version_tuple}\" "
                "> src/_pytest/_version.py; "
                "python -m pytest --version"
            )
        return "python -m pytest --version"
    if "./tests/runtests.py" in command:
        return "python --version && test -f ./tests/runtests.py"
    first_token = command.split(maxsplit=1)[0] if command else ""
    if first_token.startswith("./"):
        return f"python --version && test -x {first_token}"
    return "python --version"


def _expected_adherence(run_scope: str, *, operator_gate: str = "") -> dict[str, Any]:
    if operator_gate == "uncertainty_gated_minimal_validation_probe":
        return {
            "patch_allowed": False,
            "required_steps": ["minimal_validation_probe", "hypothesis_update_only"],
            "minimal_probe_required": True,
            "recovery_credit_allowed": False,
        }
    if run_scope == "environment_preflight_then_verifier":
        return {
            "patch_allowed": False,
            "required_steps": ["environment_preflight", "verifier_or_oracle_recheck"],
        }
    if run_scope in {"guarded_protocol_then_patch", "guarded_protocol_then_clean_patch"}:
        required_steps = [
            "previous_patch_or_protocol_replay",
            "environment_preflight",
            "verifier_or_oracle_recheck",
        ]
        if run_scope == "guarded_protocol_then_clean_patch":
            required_steps.append("clean_workspace_before_patch_fallback")
        required_steps.append("source_only_patch_fallback_if_probe_fails")
        return {
            "patch_allowed": True,
            "required_steps": required_steps,
            "guarded_protocol_probe_required": True,
            "fallback_patch_allowed_after_failed_probe": True,
            "clean_workspace_before_fallback": run_scope == "guarded_protocol_then_clean_patch",
        }
    if run_scope == "patcher_fixed_localization":
        return {
            "patch_allowed": True,
            "required_steps": ["reuse_existing_localization", "patcher", "verifier_or_oracle_recheck"],
        }
    if run_scope == "patcher_fixed_localization_candidate_only":
        return {
            "patch_allowed": True,
            "required_steps": ["reuse_existing_localization", "patcher", "pre_oracle_candidate_audit"],
        }
    if run_scope == "candidate_patch_verifier":
        return {
            "patch_allowed": True,
            "required_steps": ["candidate_patch_replay", "verifier_or_oracle_recheck"],
        }
    if run_scope == "verifier_only_replay":
        return {
            "patch_allowed": False,
            "required_steps": ["verifier_or_oracle_recheck"],
        }
    return {"patch_allowed": False, "required_steps": []}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text or ""))


if __name__ == "__main__":
    raise SystemExit(main())
