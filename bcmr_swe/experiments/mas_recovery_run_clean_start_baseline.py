"""Run a MAS42 clean-start/freeform repair baseline from source snapshots.

This runner is deliberately separate from the older natural-checkpoint
clean-start comparator. It starts from the admitted MAS-DX manifest source
snapshot, runs the normal locator -> patcher -> verifier MAS once, disables
BCMR/MAS-DX-R recovery, and evaluates with the same manifest oracle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import ast
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from bcmr_swe.agent.coordinator import validation_failure_signature
from bcmr_swe.recovery.patch_contract import PROMPT_BEGIN, PROMPT_END, SCHEMA_VERSION
from swe_mas.utils.path_filters import classify_changed_files, existing_repo_source_paths
from bcmr_swe.experiments.common import (
    build_chat_model,
    build_coordinator,
    build_executor,
    disable_streaming_if_supported,
    materialize_workspace,
    workspace_strategy_for_runtime,
)
from bcmr_swe.experiments.mas_dx_run_action_enforced_recovery import (
    EVALUATION_COMMAND_SOURCES,
    _evaluation_command_payload,
    _evaluation_commands_for_manifest,
    _evaluation_protocol_status,
    _manifest_path,
    _runtime_validate_evaluation_commands,
    _usage_delta,
    _usage_snapshot,
)
from bcmr_swe.experiments.run_recovery_only_benchmark import _bootstrap_run_context
from bcmr_swe.experiments.substrate_ctx import summarize_compact_stage_trace, workspace_patch_summary


PATCH_CONTRACT_CHOICES = {"none", "source_only"}
LOCATOR_FAILURE_SALVAGE_CHOICES = {"none", "source_path_probe"}
SOURCE_EDIT_CONTRACT_CHOICES = {"none", "minimal_targeted", "guarded_fallback"}
HISTORICAL_EXPERIENCE_MODE_CHOICES = {"mas_conditioned", "generic_patch_memory"}


def _clean_start_policy(
    patch_contract: str = "none",
    locator_failure_salvage: str = "none",
    source_edit_contract: str = "none",
    verifier_failure_feedback_retry: bool = False,
    bounded_validation_signal_retry: bool = False,
    failure_class_conditioned_retry: bool = False,
    contract_verified_candidate_retry: bool = False,
    external_validation_feedback_artifact: Path | None = None,
    baseline_success_contract_artifact: Path | None = None,
    historical_experience_artifact: Path | None = None,
    historical_experience_mode: str = "mas_conditioned",
    mas_experience_controller_artifact: Path | None = None,
    historical_action_program_artifact: Path | None = None,
    diverse_repair_hypothesis_retry: bool = False,
    candidate_pre_admission_syntax_guard: bool = False,
    source_edit_pre_oracle_retry: bool = False,
    diff_first_or_abstain: bool = False,
    semantic_patch_correctness_v2: bool = False,
) -> dict[str, Any]:
    return {
        "starts_from": "manifest.source_snapshot",
        "start_stage": "locator",
        "bcmr_recovery_enabled": False,
        "uses_previous_mas_localization": False,
        "uses_expert_labels": False,
        "uses_mas_dx_r_graph": False,
        "patch_contract": patch_contract,
        "locator_failure_salvage": locator_failure_salvage,
        "source_edit_contract": source_edit_contract,
        "verifier_failure_feedback_retry": bool(verifier_failure_feedback_retry),
        "bounded_validation_signal_retry": bool(bounded_validation_signal_retry),
        "failure_class_conditioned_retry": bool(failure_class_conditioned_retry),
        "contract_verified_candidate_retry": bool(contract_verified_candidate_retry),
        "diverse_repair_hypothesis_retry": bool(diverse_repair_hypothesis_retry),
        "candidate_pre_admission_syntax_guard": bool(candidate_pre_admission_syntax_guard),
        "source_edit_pre_oracle_retry": bool(source_edit_pre_oracle_retry),
        "diff_first_or_abstain": bool(diff_first_or_abstain),
        "semantic_patch_correctness_v2": bool(semantic_patch_correctness_v2),
        "external_validation_feedback_enabled": external_validation_feedback_artifact is not None,
        "external_validation_feedback_artifact": str(external_validation_feedback_artifact or ""),
        "baseline_success_contract_enabled": baseline_success_contract_artifact is not None,
        "baseline_success_contract_artifact": str(baseline_success_contract_artifact or ""),
        "historical_experience_enabled": historical_experience_artifact is not None,
        "historical_experience_artifact": str(historical_experience_artifact or ""),
        "historical_experience_mode": historical_experience_mode,
        "mas_experience_controller_enabled": mas_experience_controller_artifact is not None,
        "mas_experience_controller_artifact": str(mas_experience_controller_artifact or ""),
        "historical_action_program_enabled": historical_action_program_artifact is not None,
        "historical_action_program_artifact": str(historical_action_program_artifact or ""),
    }


def run_clean_start_baseline(
    *,
    instance_id: str,
    manifest_root: Path,
    api_path: Path | None = None,
    model_name: str = "",
    strong_model_name: str = "",
    output_root: Path = Path("outputs/mas_recovery/clean_start_mas42_workspaces"),
    runtime: str = "harness",
    execute: bool = False,
    force_rebuild_harness: bool = False,
    harness_setup_timeout: int | None = None,
    harness_container_start_timeout: int | None = None,
    harness_container_cleanup_timeout: int | None = None,
    harness_preflight_before_execute: bool = False,
    harness_preflight_timeout: int = 10,
    harness_preflight_container_smoke_image: str = "",
    command_timeout: int = 1800,
    evaluation_command_source: str = "manifest",
    request_timeout: int = 60,
    max_retries: int = 1,
    locator_max_iterations: int = 8,
    planner_max_iterations: int = 4,
    patcher_max_iterations: int = 8,
    verifier_max_iterations: int = 6,
    patch_contract: str = "none",
    locator_failure_salvage: str = "none",
    source_edit_contract: str = "none",
    verifier_failure_feedback_retry: bool = False,
    bounded_validation_signal_retry: bool = False,
    failure_class_conditioned_retry: bool = False,
    contract_verified_candidate_retry: bool = False,
    external_validation_feedback_artifact: Path | None = None,
    baseline_success_contract_artifact: Path | None = None,
    historical_experience_artifact: Path | None = None,
    historical_experience_mode: str = "mas_conditioned",
    mas_experience_controller_artifact: Path | None = None,
    historical_action_program_artifact: Path | None = None,
    diverse_repair_hypothesis_retry: bool = False,
    candidate_pre_admission_syntax_guard: bool = False,
    source_edit_pre_oracle_retry: bool = False,
    diff_first_or_abstain: bool = False,
    semantic_patch_correctness_v2: bool = False,
) -> dict[str, Any]:
    started_at, started_monotonic = _start_runtime_timer()
    manifest_path = _manifest_path(manifest_root, instance_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if patch_contract not in PATCH_CONTRACT_CHOICES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_patch_contract",
                "stop_reason": "unsupported_patch_contract",
                "oracle_success": False,
                "reported_success": False,
                "patch_contract": patch_contract,
                "locator_failure_salvage": locator_failure_salvage,
                "source_edit_contract": source_edit_contract,
            },
        )
    if locator_failure_salvage not in LOCATOR_FAILURE_SALVAGE_CHOICES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_locator_failure_salvage",
                "stop_reason": "unsupported_locator_failure_salvage",
                "oracle_success": False,
                "reported_success": False,
                "locator_failure_salvage": locator_failure_salvage,
            },
        )
    if source_edit_contract not in SOURCE_EDIT_CONTRACT_CHOICES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_source_edit_contract",
                "stop_reason": "unsupported_source_edit_contract",
                "oracle_success": False,
                "reported_success": False,
                "source_edit_contract": source_edit_contract,
            },
        )
    if evaluation_command_source not in EVALUATION_COMMAND_SOURCES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_evaluation_command_source",
                "stop_reason": "unsupported_evaluation_command_source",
                "oracle_success": False,
                "reported_success": False,
                "evaluation_command_source": evaluation_command_source,
            },
        )
    if historical_experience_mode not in HISTORICAL_EXPERIENCE_MODE_CHOICES:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=execute,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "unsupported_historical_experience_mode",
                "stop_reason": "unsupported_historical_experience_mode",
                "oracle_success": False,
                "reported_success": False,
                "historical_experience_mode": historical_experience_mode,
            },
        )
    if not execute:
        evaluation = _evaluation_commands_for_manifest(manifest, command_source=evaluation_command_source)
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=False,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                **_evaluation_command_payload(evaluation),
                "error_type": "planned_not_executed",
                "stop_reason": "planned_not_executed",
                "oracle_success": False,
                "reported_success": False,
                "workspace_strategy": workspace_strategy_for_runtime(runtime, manifest),
                "runtime": runtime,
                "patch_contract": patch_contract,
                "locator_failure_salvage": locator_failure_salvage,
                "source_edit_contract": source_edit_contract,
                "clean_start_policy": _clean_start_policy(
                    patch_contract,
                    locator_failure_salvage,
                    source_edit_contract,
                    verifier_failure_feedback_retry,
                    bounded_validation_signal_retry,
                    failure_class_conditioned_retry,
                    contract_verified_candidate_retry,
                    external_validation_feedback_artifact,
                    baseline_success_contract_artifact,
                    historical_experience_artifact,
                    historical_experience_mode=historical_experience_mode,
                    diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                    candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                    source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                    diff_first_or_abstain=diff_first_or_abstain,
                    semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                    mas_experience_controller_artifact=mas_experience_controller_artifact,
                    historical_action_program_artifact=historical_action_program_artifact,
                ),
                "method_variant": (
                    _method_variant(
                        verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                        bounded_validation_signal_retry=bounded_validation_signal_retry,
                        failure_class_conditioned_retry=failure_class_conditioned_retry,
                        contract_verified_candidate_retry=contract_verified_candidate_retry,
                        external_validation_feedback_artifact=external_validation_feedback_artifact,
                        baseline_success_contract_artifact=baseline_success_contract_artifact,
                        historical_experience_artifact=historical_experience_artifact,
                        historical_experience_mode=historical_experience_mode,
                        mas_experience_controller_artifact=mas_experience_controller_artifact,
                        diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                        candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                        source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                        diff_first_or_abstain=diff_first_or_abstain,
                        semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                        historical_action_program_artifact=historical_action_program_artifact,
                    )
                ),
            },
        )
    if api_path is None:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "missing_api_path",
                "stop_reason": "missing_api_path",
                "oracle_success": False,
                "reported_success": False,
            },
        )

    workspace_strategy = workspace_strategy_for_runtime(runtime, manifest)
    preflight_audit = _empty_harness_preflight_audit(enabled=bool(harness_preflight_before_execute))
    if runtime == "harness" and harness_preflight_before_execute:
        preflight_audit = _harness_startup_preflight_audit(
            instance_id=instance_id,
            timeout_seconds=harness_preflight_timeout,
            container_smoke_image=harness_preflight_container_smoke_image,
        )
        if preflight_audit["blocks_execution"]:
            return _payload(
                instance_id=instance_id,
                manifest_path=manifest_path,
                execute_requested=True,
                started_at=started_at,
                started_monotonic=started_monotonic,
                row={
                    "error_type": "runtime_error",
                    "stop_reason": "harness_preflight_blocked",
                    "error": str(preflight_audit.get("reason", "") or "harness_preflight_blocked"),
                    "oracle_success": False,
                    "reported_success": False,
                    "workspace": "",
                    "workspace_strategy": workspace_strategy,
                    "runtime": runtime,
                    "patch_contract": patch_contract,
                    "locator_failure_salvage": locator_failure_salvage,
                    "source_edit_contract": source_edit_contract,
                    "clean_start_policy": _clean_start_policy(
                        patch_contract,
                        locator_failure_salvage,
                        source_edit_contract,
                        verifier_failure_feedback_retry,
                        bounded_validation_signal_retry,
                        failure_class_conditioned_retry,
                        contract_verified_candidate_retry,
                        external_validation_feedback_artifact,
                        baseline_success_contract_artifact,
                        historical_experience_artifact,
                        historical_experience_mode=historical_experience_mode,
                        diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                        candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                        source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                        diff_first_or_abstain=diff_first_or_abstain,
                        semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                        mas_experience_controller_artifact=mas_experience_controller_artifact,
                        historical_action_program_artifact=historical_action_program_artifact,
                    ),
                    "method_variant": (
                        _method_variant(
                            verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                            bounded_validation_signal_retry=bounded_validation_signal_retry,
                            failure_class_conditioned_retry=failure_class_conditioned_retry,
                            contract_verified_candidate_retry=contract_verified_candidate_retry,
                            external_validation_feedback_artifact=external_validation_feedback_artifact,
                            baseline_success_contract_artifact=baseline_success_contract_artifact,
                            historical_experience_artifact=historical_experience_artifact,
                            historical_experience_mode=historical_experience_mode,
                            mas_experience_controller_artifact=mas_experience_controller_artifact,
                            diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                            candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                            source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                            diff_first_or_abstain=diff_first_or_abstain,
                            semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                            historical_action_program_artifact=historical_action_program_artifact,
                        )
                    ),
                    "harness_preflight_before_execute": True,
                    "harness_preflight_audit": preflight_audit,
                    "model_call_count": 0,
                    "token_cost": 0.0,
                },
            )
    workspace = materialize_workspace(
        manifest["source_snapshot"],
        output_root,
        f"clean_start_freeform_{instance_id}",
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
        )
        result = _execute_clean_start(
            manifest=manifest,
            executor=executor,
            workspace=workspace,
            timeout=command_timeout,
            api_path=api_path,
            model_name=model_name,
            strong_model_name=strong_model_name,
            request_timeout=request_timeout,
            max_retries=max_retries,
            locator_max_iterations=locator_max_iterations,
            planner_max_iterations=planner_max_iterations,
            patcher_max_iterations=patcher_max_iterations,
            verifier_max_iterations=verifier_max_iterations,
            evaluation_command_source=evaluation_command_source,
            patch_contract=patch_contract,
            locator_failure_salvage=locator_failure_salvage,
            source_edit_contract=source_edit_contract,
            verifier_failure_feedback_retry=verifier_failure_feedback_retry,
            bounded_validation_signal_retry=bounded_validation_signal_retry,
            failure_class_conditioned_retry=failure_class_conditioned_retry,
            contract_verified_candidate_retry=contract_verified_candidate_retry,
            external_validation_feedback_artifact=external_validation_feedback_artifact,
            baseline_success_contract_artifact=baseline_success_contract_artifact,
            historical_experience_artifact=historical_experience_artifact,
            historical_experience_mode=historical_experience_mode,
            mas_experience_controller_artifact=mas_experience_controller_artifact,
            historical_action_program_artifact=historical_action_program_artifact,
            diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
            candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
            source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
            diff_first_or_abstain=diff_first_or_abstain,
            semantic_patch_correctness_v2=semantic_patch_correctness_v2,
        )
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "",
                "stop_reason": "executed",
                "workspace": str(workspace),
                "workspace_strategy": workspace_strategy,
                "runtime": runtime,
                "harness_preflight_before_execute": bool(harness_preflight_before_execute),
                "harness_preflight_audit": preflight_audit,
                **result,
            },
        )
    except Exception as exc:
        return _payload(
            instance_id=instance_id,
            manifest_path=manifest_path,
            execute_requested=True,
            started_at=started_at,
            started_monotonic=started_monotonic,
            row={
                "error_type": "runtime_error",
                "stop_reason": "runtime_error",
                "error": str(exc),
                "oracle_success": False,
                "reported_success": False,
                "workspace": str(workspace),
                "workspace_strategy": workspace_strategy,
                "runtime": runtime,
                "harness_preflight_before_execute": bool(harness_preflight_before_execute),
                "harness_preflight_audit": preflight_audit,
                "patch_contract": patch_contract,
                "locator_failure_salvage": locator_failure_salvage,
                "source_edit_contract": source_edit_contract,
                "clean_start_policy": _clean_start_policy(
                    patch_contract,
                    locator_failure_salvage,
                    source_edit_contract,
                    verifier_failure_feedback_retry,
                    bounded_validation_signal_retry,
                    failure_class_conditioned_retry,
                    contract_verified_candidate_retry,
                    external_validation_feedback_artifact,
                    baseline_success_contract_artifact,
                    historical_experience_artifact,
                    historical_experience_mode=historical_experience_mode,
                    diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                    candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                    source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                    diff_first_or_abstain=diff_first_or_abstain,
                    semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                    mas_experience_controller_artifact=mas_experience_controller_artifact,
                    historical_action_program_artifact=historical_action_program_artifact,
                ),
                "method_variant": (
                    _method_variant(
                        verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                        bounded_validation_signal_retry=bounded_validation_signal_retry,
                        failure_class_conditioned_retry=failure_class_conditioned_retry,
                        contract_verified_candidate_retry=contract_verified_candidate_retry,
                        external_validation_feedback_artifact=external_validation_feedback_artifact,
                        baseline_success_contract_artifact=baseline_success_contract_artifact,
                        historical_experience_artifact=historical_experience_artifact,
                        historical_experience_mode=historical_experience_mode,
                        mas_experience_controller_artifact=mas_experience_controller_artifact,
                        diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                        candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                        source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                        diff_first_or_abstain=diff_first_or_abstain,
                        semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                        historical_action_program_artifact=historical_action_program_artifact,
                    )
                ),
            },
        )
    finally:
        if runtime_session is not None:
            runtime_session.close()


def _execute_clean_start(
    *,
    manifest: dict[str, Any],
    executor,
    workspace: Path,
    timeout: int,
    api_path: Path,
    model_name: str,
    strong_model_name: str,
    request_timeout: int,
    max_retries: int,
    locator_max_iterations: int,
    planner_max_iterations: int,
    patcher_max_iterations: int,
    verifier_max_iterations: int,
    evaluation_command_source: str,
    patch_contract: str,
    locator_failure_salvage: str,
    source_edit_contract: str,
    verifier_failure_feedback_retry: bool,
    bounded_validation_signal_retry: bool,
    failure_class_conditioned_retry: bool,
    contract_verified_candidate_retry: bool,
    external_validation_feedback_artifact: Path | None,
    baseline_success_contract_artifact: Path | None,
    historical_experience_artifact: Path | None,
    historical_experience_mode: str = "mas_conditioned",
    mas_experience_controller_artifact: Path | None = None,
    historical_action_program_artifact: Path | None = None,
    diverse_repair_hypothesis_retry: bool = False,
    candidate_pre_admission_syntax_guard: bool = False,
    source_edit_pre_oracle_retry: bool = False,
    diff_first_or_abstain: bool = False,
    semantic_patch_correctness_v2: bool = False,
) -> dict[str, Any]:
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
    coord = build_coordinator(
        workspace=str(workspace),
        model=model,
        strong_model=strong_model,
        strong_stages=("planner", "implementer"),
        executor=executor,
        recovery_mode="v2_action",
        locator_max_iterations=locator_max_iterations,
        planner_max_iterations=planner_max_iterations,
        patcher_max_iterations=patcher_max_iterations,
        verifier_max_iterations=verifier_max_iterations,
    )
    _bootstrap_run_context(coord, manifest=manifest, workspace=workspace, executor=executor)
    coord._recovery_enabled = False
    recovery_context = _compose_recovery_context(
        patch_contract=patch_contract,
        source_edit_contract=source_edit_contract,
        diff_first_or_abstain=diff_first_or_abstain,
    )
    usage_before = _usage_snapshot(strong_model)
    external_feedback_audit = _external_validation_feedback_admission_audit(
        artifact=external_validation_feedback_artifact,
        instance_id=str(manifest.get("instance_id", "") or ""),
    )
    baseline_contract_audit = _baseline_success_contract_admission_audit(
        artifact=baseline_success_contract_artifact,
        instance_id=str(manifest.get("instance_id", "") or ""),
    )
    historical_experience_audit = _historical_experience_admission_audit(
        artifact=historical_experience_artifact,
        target_instance_id=str(manifest.get("instance_id", "") or ""),
        mode=historical_experience_mode,
    )
    historical_action_program_audit = _historical_action_program_admission_audit(
        artifact=historical_action_program_artifact,
        target_instance_id=str(manifest.get("instance_id", "") or ""),
    )
    if external_validation_feedback_artifact is not None and not external_feedback_audit["admitted"]:
        usage_empty = _usage_delta(usage_before, usage_before)
        return {
            "error_type": "external_validation_feedback_not_admitted",
            "stop_reason": "external_validation_feedback_not_admitted",
            "oracle_success": False,
            "reported_success": False,
            "agent_reported_success": False,
            "initial_agent_reported_success": False,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_before,
            "model_usage_delta": usage_empty,
            "token_cost": 0.0,
            "model_call_count": 0,
            "latency_sec": 0.001,
            "patch_contract": patch_contract,
            "locator_failure_salvage": locator_failure_salvage,
            "source_edit_contract": source_edit_contract,
            "method_variant": _method_variant(
                verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                bounded_validation_signal_retry=bounded_validation_signal_retry,
                failure_class_conditioned_retry=failure_class_conditioned_retry,
                contract_verified_candidate_retry=contract_verified_candidate_retry,
                external_validation_feedback_artifact=external_validation_feedback_artifact,
                baseline_success_contract_artifact=baseline_success_contract_artifact,
                historical_experience_artifact=historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            ),
            "external_validation_feedback_admission_audit": external_feedback_audit,
            "external_validation_feedback_context_attached": False,
            "clean_start_policy": _clean_start_policy(
                patch_contract,
                locator_failure_salvage,
                source_edit_contract,
                verifier_failure_feedback_retry,
                bounded_validation_signal_retry,
                failure_class_conditioned_retry,
                contract_verified_candidate_retry,
                external_validation_feedback_artifact,
                baseline_success_contract_artifact,
                historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
            ),
        }
    if baseline_success_contract_artifact is not None and not baseline_contract_audit["admitted"]:
        usage_empty = _usage_delta(usage_before, usage_before)
        return {
            "error_type": "baseline_success_contract_not_admitted",
            "stop_reason": "baseline_success_contract_not_admitted",
            "oracle_success": False,
            "reported_success": False,
            "agent_reported_success": False,
            "initial_agent_reported_success": False,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_before,
            "model_usage_delta": usage_empty,
            "token_cost": 0.0,
            "model_call_count": 0,
            "latency_sec": 0.001,
            "patch_contract": patch_contract,
            "locator_failure_salvage": locator_failure_salvage,
            "source_edit_contract": source_edit_contract,
            "method_variant": _method_variant(
                verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                bounded_validation_signal_retry=bounded_validation_signal_retry,
                failure_class_conditioned_retry=failure_class_conditioned_retry,
                contract_verified_candidate_retry=contract_verified_candidate_retry,
                external_validation_feedback_artifact=external_validation_feedback_artifact,
                baseline_success_contract_artifact=baseline_success_contract_artifact,
                historical_experience_artifact=historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            ),
            "external_validation_feedback_admission_audit": external_feedback_audit,
            "external_validation_feedback_context_attached": False,
            "baseline_success_contract_admission_audit": baseline_contract_audit,
            "baseline_success_contract_context_attached": False,
            "clean_start_policy": _clean_start_policy(
                patch_contract,
                locator_failure_salvage,
                source_edit_contract,
                verifier_failure_feedback_retry,
                bounded_validation_signal_retry,
                failure_class_conditioned_retry,
                contract_verified_candidate_retry,
                external_validation_feedback_artifact,
                baseline_success_contract_artifact,
                historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
            ),
        }
    if historical_experience_artifact is not None and not historical_experience_audit["admitted"]:
        usage_empty = _usage_delta(usage_before, usage_before)
        return {
            "error_type": "historical_experience_not_admitted",
            "stop_reason": "historical_experience_not_admitted",
            "oracle_success": False,
            "reported_success": False,
            "agent_reported_success": False,
            "initial_agent_reported_success": False,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_before,
            "model_usage_delta": usage_empty,
            "token_cost": 0.0,
            "model_call_count": 0,
            "latency_sec": 0.001,
            "patch_contract": patch_contract,
            "locator_failure_salvage": locator_failure_salvage,
            "source_edit_contract": source_edit_contract,
            "method_variant": _method_variant(
                verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                bounded_validation_signal_retry=bounded_validation_signal_retry,
                failure_class_conditioned_retry=failure_class_conditioned_retry,
                contract_verified_candidate_retry=contract_verified_candidate_retry,
                external_validation_feedback_artifact=external_validation_feedback_artifact,
                baseline_success_contract_artifact=baseline_success_contract_artifact,
                historical_experience_artifact=historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            ),
            "external_validation_feedback_admission_audit": external_feedback_audit,
            "external_validation_feedback_context_attached": False,
            "baseline_success_contract_admission_audit": baseline_contract_audit,
            "baseline_success_contract_context_attached": False,
            "historical_experience_admission_audit": historical_experience_audit,
            "historical_experience_context_attached": False,
            "clean_start_policy": _clean_start_policy(
                patch_contract,
                locator_failure_salvage,
                source_edit_contract,
                verifier_failure_feedback_retry,
                bounded_validation_signal_retry,
                failure_class_conditioned_retry,
                contract_verified_candidate_retry,
                external_validation_feedback_artifact,
                baseline_success_contract_artifact,
                historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
            ),
        }
    if historical_action_program_artifact is not None and not historical_action_program_audit["admitted"]:
        usage_empty = _usage_delta(usage_before, usage_before)
        return {
            "error_type": "historical_action_program_not_admitted",
            "stop_reason": "historical_action_program_not_admitted",
            "oracle_success": False,
            "reported_success": False,
            "agent_reported_success": False,
            "initial_agent_reported_success": False,
            "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
            "strong_model_name": str(
                getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
            ),
            "model_usage_before": usage_before,
            "model_usage_after": usage_before,
            "model_usage_delta": usage_empty,
            "token_cost": 0.0,
            "model_call_count": 0,
            "latency_sec": 0.001,
            "patch_contract": patch_contract,
            "locator_failure_salvage": locator_failure_salvage,
            "source_edit_contract": source_edit_contract,
            "method_variant": _method_variant(
                verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                bounded_validation_signal_retry=bounded_validation_signal_retry,
                failure_class_conditioned_retry=failure_class_conditioned_retry,
                contract_verified_candidate_retry=contract_verified_candidate_retry,
                external_validation_feedback_artifact=external_validation_feedback_artifact,
                baseline_success_contract_artifact=baseline_success_contract_artifact,
                historical_experience_artifact=historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            ),
            "external_validation_feedback_admission_audit": external_feedback_audit,
            "external_validation_feedback_context_attached": False,
            "baseline_success_contract_admission_audit": baseline_contract_audit,
            "baseline_success_contract_context_attached": False,
            "historical_experience_admission_audit": historical_experience_audit,
            "historical_experience_context_attached": False,
            "historical_action_program_admission_audit": historical_action_program_audit,
            "historical_action_program_context_attached": False,
            "clean_start_policy": _clean_start_policy(
                patch_contract,
                locator_failure_salvage,
                source_edit_contract,
                verifier_failure_feedback_retry,
                bounded_validation_signal_retry,
                failure_class_conditioned_retry,
                contract_verified_candidate_retry,
                external_validation_feedback_artifact,
                baseline_success_contract_artifact,
                historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                diff_first_or_abstain=diff_first_or_abstain,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
            ),
        }
    external_feedback_context = _external_validation_feedback_context(external_feedback_audit)
    if external_feedback_context:
        recovery_context = f"{recovery_context.strip()}\n\n{external_feedback_context}".strip()
    baseline_contract_context = _baseline_success_contract_context(baseline_contract_audit)
    if baseline_contract_context:
        recovery_context = f"{recovery_context.strip()}\n\n{baseline_contract_context}".strip()
    historical_experience_context = _historical_experience_context(
        historical_experience_audit,
        mode=historical_experience_mode,
    )
    if historical_experience_context:
        recovery_context = f"{recovery_context.strip()}\n\n{historical_experience_context}".strip()
    mas_experience_controller_audit = _mas_experience_controller_admission_audit(
        artifact=mas_experience_controller_artifact,
        target_instance_id=str(manifest.get("instance_id", "") or ""),
    )
    mas_experience_controller_context = _mas_experience_controller_context(mas_experience_controller_audit)
    if mas_experience_controller_context:
        recovery_context = f"{recovery_context.strip()}\n\n{mas_experience_controller_context}".strip()
    historical_action_program_context = _historical_action_program_context(historical_action_program_audit)
    if historical_action_program_context:
        recovery_context = f"{recovery_context.strip()}\n\n{historical_action_program_context}".strip()
    active_recovery_context = recovery_context
    feedback_retry_audit: dict[str, Any] = {
        "enabled": bool(verifier_failure_feedback_retry),
        "attempted": False,
        "executed": False,
        "reason": "disabled" if not verifier_failure_feedback_retry else "not_applicable",
        "uses_verifier_failure_feedback": False,
        "retry_recovery_context_contains_feedback": False,
        "agent_reported_success_after_retry": False,
        "claim_boundary": (
            "This optional branch is a MASGuard method-change probe. It is not part of the clean-start baseline "
            "and must be audited separately before recovery credit."
        ),
    }
    bounded_signal_retry_audit: dict[str, Any] = {
        "schema": "masguard.bounded_validation_signal_retry.v2",
        "enabled": bool(bounded_validation_signal_retry),
        "failure_class_conditioned_retry_enabled": bool(failure_class_conditioned_retry),
        "diverse_repair_hypothesis_retry_enabled": bool(diverse_repair_hypothesis_retry),
        "diverse_repair_hypothesis_audit": _empty_diverse_repair_hypothesis_audit(
            enabled=bool(diverse_repair_hypothesis_retry)
        ),
        "attempted": False,
        "executed": False,
        "reason": "disabled" if not bounded_validation_signal_retry else "not_applicable",
        "minimal_validation_executed": False,
        "minimal_validation_feedback_ready": False,
        "minimal_validation_source_repair_feedback_ready": False,
        "minimal_validation_signature_present": False,
        "minimal_validation_failure_observed": False,
        "minimal_validation_failure_class": "unknown",
        "minimal_validation_retry_action": "abstain",
        "minimal_validation_failure_classification": _minimal_validation_failure_classification(
            protocol={},
            signature=_empty_validation_failure_signature(),
            failure_sources=[],
            patch_summary={},
        ),
        "minimal_validation_class_specific_edit_contract": _failure_class_edit_contract({}),
        "retry_recovery_context_contains_feedback": False,
        "retry_recovery_context_contains_failure_class": False,
        "retry_recovery_context_contains_class_specific_contract": False,
        "contract_verified_candidate_retry_enabled": bool(contract_verified_candidate_retry),
        "contract_verified_candidate_retry_audit": _empty_contract_verified_candidate_retry_audit(
            enabled=bool(contract_verified_candidate_retry),
            candidate_pre_admission_syntax_guard=bool(candidate_pre_admission_syntax_guard),
        ),
        "agent_reported_success_after_retry": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_grant_recovery_credit": True,
            "uses_minimal_fail_to_pass_feedback_before_retry": True,
            "fail_closed_when_feedback_missing": True,
            "failure_class_conditioning_requires_explicit_flag": True,
            "contract_verified_candidate_retry_requires_explicit_flag": True,
        },
    }
    source_edit_pre_oracle_retry_audit = _empty_source_edit_pre_oracle_retry_audit(
        enabled=bool(source_edit_pre_oracle_retry)
    )
    diff_first_or_abstain_audit = _empty_diff_first_or_abstain_audit(
        enabled=bool(diff_first_or_abstain)
    )
    historical_action_program_direct_script_enabled = bool(
        historical_action_program_audit.get("admitted", False)
        and str(historical_action_program_audit.get("execution_mode", "") or "") == "direct_script"
    )
    historical_action_program_direct_script_audit = _empty_historical_action_program_direct_script_audit(
        enabled=historical_action_program_direct_script_enabled
    )
    started_at = time.monotonic()
    if historical_action_program_direct_script_enabled:
        historical_action_program_direct_script_audit, direct_script_stage_output = (
            _historical_action_program_direct_script_audit(
                enabled=True,
                admission=historical_action_program_audit,
                executor=executor,
                workspace=workspace,
                timeout=timeout,
                manifest=manifest,
                source_edit_contract=source_edit_contract,
                patch_contract=patch_contract,
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            )
        )
        coord.stage_outputs["patcher"] = direct_script_stage_output
        initial_agent_reported_success = bool(direct_script_stage_output.get("success", False))
        agent_reported_success = initial_agent_reported_success
        salvage_audit = _source_path_probe_locator_salvage(
            locator_result={},
            workspace=workspace,
            recovery_context=recovery_context,
            patch_contract=patch_contract,
            mode="none",
            initial_agent_reported_success=initial_agent_reported_success,
        )
        salvage_audit["reason"] = "disabled_for_historical_action_program_direct_script"
    else:
        initial_agent_reported_success = bool(
            coord._resume_from("locator", recovery_context=recovery_context, escalation_level=0, deep_verify=False)
        )
        salvage_audit = _source_path_probe_locator_salvage(
            locator_result=dict(coord.stage_outputs.get("locator", {}) or {}),
            workspace=workspace,
            recovery_context=recovery_context,
            patch_contract=patch_contract,
            mode=locator_failure_salvage,
            initial_agent_reported_success=initial_agent_reported_success,
        )
        agent_reported_success = initial_agent_reported_success
    if salvage_audit["accepted"]:
        locate_result = dict(coord.stage_outputs.get("locator", {}) or {})
        locate_result.update(
            {
                "located_files": salvage_audit["located_files"],
                "success": False,
                "best_effort_only": True,
                "failure_mode": "locator_source_path_probe_salvage",
                "locator_failure_salvage": locator_failure_salvage,
                "selected_target_candidates": list(salvage_audit["candidate_paths"]),
            }
        )
        coord.stage_outputs["locator"] = locate_result
        coord._accept_locator_best_effort_for_recovery(locate_result)
        patcher_recovery_context = _compose_recovery_context(
            patch_contract=patch_contract,
            source_edit_contract=source_edit_contract,
            locator_target_paths=list(salvage_audit["candidate_paths"]),
            locator_salvage_accepted=True,
            diff_first_or_abstain=diff_first_or_abstain,
        )
        if external_feedback_context:
            patcher_recovery_context = f"{patcher_recovery_context.strip()}\n\n{external_feedback_context}".strip()
        active_recovery_context = patcher_recovery_context
        agent_reported_success = bool(
            coord._resume_from("patcher", recovery_context=patcher_recovery_context, escalation_level=0, deep_verify=False)
        )
        salvage_audit["agent_reported_success_after_salvage"] = agent_reported_success
    if bounded_validation_signal_retry:
        bounded_signal_retry_audit["attempted"] = True
        if agent_reported_success:
            bounded_signal_retry_audit["reason"] = "pipeline_already_reported_success"
        else:
            evaluation = _evaluation_commands_for_manifest(manifest, command_source=evaluation_command_source)
            evaluation = _runtime_validate_evaluation_commands(
                manifest=manifest,
                evaluation=evaluation,
                executor=executor,
                workspace=workspace,
                timeout=min(timeout, 300),
            )
            if evaluation.get("error_type"):
                bounded_signal_retry_audit["reason"] = str(
                    evaluation.get("stop_reason", "") or "evaluation_protocol_blocked"
                )
            else:
                pre_retry_patch_summary = workspace_patch_summary(
                    workspace,
                    list(manifest.get("oracle_patch_files", []) or []),
                )
                fail_to_pass_probe = executor.execute(
                    evaluation["fail_to_pass_command"],
                    cwd=str(workspace),
                    timeout=min(timeout, 300),
                )
                minimal_protocol = _evaluation_protocol_status(
                    fail_to_pass=fail_to_pass_probe,
                    oracle={"returncode": 0, "output": ""},
                )
                minimal_failure_audit = _external_validation_failure_audit(
                    evaluation=evaluation,
                    fail_to_pass=fail_to_pass_probe,
                    oracle={"returncode": 0, "output": ""},
                    protocol=minimal_protocol,
                    patch_summary=pre_retry_patch_summary,
                )
                controlled_no_diff_retry_ready = _bounded_validation_no_diff_source_target_retry_allowed(
                    minimal_failure_audit
                )
                bounded_signal_retry_audit.update(
                    {
                        "minimal_validation_executed": True,
                        "minimal_validation_returncode": fail_to_pass_probe.get("returncode"),
                        "minimal_validation_protocol": minimal_protocol,
                        "minimal_validation_failure_audit": minimal_failure_audit,
                        "minimal_validation_failure_classification": dict(
                            minimal_failure_audit["failure_classification"]
                        ),
                        "minimal_validation_class_specific_edit_contract": _failure_class_edit_contract(
                            dict(minimal_failure_audit["failure_classification"])
                        ),
                        "minimal_validation_failure_class": str(minimal_failure_audit["failure_class"]),
                        "minimal_validation_retry_action": str(
                            minimal_failure_audit["failure_classification"]["retry_action"]
                        ),
                        "minimal_validation_failure_observed": bool(
                            minimal_failure_audit["failure_observed"]
                        ),
                        "minimal_validation_signature_present": bool(
                            minimal_failure_audit["signature_present"]
                        ),
                        "minimal_validation_feedback_ready": bool(
                            minimal_failure_audit["feedback_ready"]
                        ),
                        "minimal_validation_source_repair_feedback_ready": bool(
                            minimal_failure_audit["source_repair_feedback_ready"]
                        ),
                        "minimal_validation_controlled_no_diff_retry_ready": controlled_no_diff_retry_ready,
                    }
                )
                if minimal_protocol.get("evaluation_protocol_error"):
                    bounded_signal_retry_audit["reason"] = str(
                        minimal_protocol.get("evaluation_protocol_error_type", "")
                        or "minimal_validation_protocol_error"
                    )
                elif not minimal_failure_audit["failure_observed"] and not controlled_no_diff_retry_ready:
                    bounded_signal_retry_audit["reason"] = "minimal_validation_passed_no_retry_signal"
                elif not _bounded_validation_retry_allowed(
                    minimal_failure_audit,
                    failure_class_conditioned_retry=failure_class_conditioned_retry,
                ):
                    bounded_signal_retry_audit["reason"] = "minimal_validation_signal_not_source_repair_ready"
                else:
                    synthetic_admission = {
                        "admitted": True,
                        "previous_source_files": list(pre_retry_patch_summary.get("non_test_files", []) or []),
                        "previous_diff_excerpt": _workspace_diff_for_files(
                            workspace,
                            list(pre_retry_patch_summary.get("changed_files", []) or []),
                        )[:4000],
                        "failure_sources": list(minimal_failure_audit["failure_sources"]),
                        "signature": dict(minimal_failure_audit["signature"]),
                        "failure_classification": dict(minimal_failure_audit["failure_classification"]),
                    }
                    retry_context = (
                        f"{active_recovery_context.strip()}\n\n"
                        f"{_external_validation_feedback_context(synthetic_admission)}"
                    ).strip()
                    diverse_hypothesis_audit = _empty_diverse_repair_hypothesis_audit(
                        enabled=bool(diverse_repair_hypothesis_retry)
                    )
                    if diverse_repair_hypothesis_retry:
                        diverse_hypothesis_context, diverse_hypothesis_audit = (
                            _diverse_repair_hypothesis_context(
                                failure_classification=dict(
                                    minimal_failure_audit.get("failure_classification", {}) or {}
                                ),
                                signature=dict(minimal_failure_audit.get("signature", {}) or {}),
                                previous_source_files=list(
                                    pre_retry_patch_summary.get("non_test_files", []) or []
                                ),
                                previous_diff_excerpt=synthetic_admission["previous_diff_excerpt"],
                                workspace=workspace,
                            )
                        )
                        retry_context = f"{retry_context.strip()}\n\n{diverse_hypothesis_context}".strip()
                    bounded_signal_retry_audit.update(
                        {
                            "reason": "executed_after_minimal_validation_failure_signal",
                            "executed": True,
                            "diverse_repair_hypothesis_audit": diverse_hypothesis_audit,
                            "retry_recovery_context_contains_diverse_hypotheses": (
                                "[MASGUARD DIVERSE REPAIR HYPOTHESES]" in retry_context
                            ),
                            "retry_recovery_context_contains_feedback": (
                                "[MASGUARD EXTERNAL VALIDATION FEEDBACK]" in retry_context
                            ),
                            "retry_recovery_context_contains_failure_class": (
                                "minimal_validation_failure_classification" in retry_context
                            ),
                            "retry_recovery_context_contains_class_specific_contract": (
                                "class_specific_edit_contract" in retry_context
                            ),
                        }
                    )
                    coord._clear_stage_outputs_from("patcher")
                    agent_reported_success = bool(
                        coord._resume_from(
                            "patcher",
                            recovery_context=retry_context,
                            escalation_level=1,
                            deep_verify=False,
                        )
                    )
                    active_recovery_context = retry_context
                    bounded_signal_retry_audit["agent_reported_success_after_retry"] = agent_reported_success
                    if contract_verified_candidate_retry:
                        first_candidate_summary = workspace_patch_summary(
                            workspace,
                            list(manifest.get("oracle_patch_files", []) or []),
                        )
                        first_candidate_source_audit = _source_edit_contract_audit(
                            source_edit_contract=source_edit_contract,
                            patch_summary=first_candidate_summary,
                            stage_outputs=dict(coord.stage_outputs or {}),
                            workspace=workspace,
                        )
                        candidate_audit = _contract_verified_candidate_retry_audit(
                            enabled=True,
                            workspace=workspace,
                            patch_summary=first_candidate_summary,
                            source_edit_contract_audit=first_candidate_source_audit,
                            failure_classification=dict(
                                minimal_failure_audit.get("failure_classification", {}) or {}
                            ),
                            signature=dict(minimal_failure_audit.get("signature", {}) or {}),
                            candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                        )
                        if candidate_audit["retry_allowed"]:
                            rejection_context = _contract_verified_candidate_rejection_context(
                                candidate_audit,
                                workspace=workspace,
                            )
                            retry_context = f"{retry_context.strip()}\n\n{rejection_context}".strip()
                            candidate_audit["rejection_context_attached"] = (
                                "[MASGUARD CONTRACT-VERIFIED CANDIDATE REJECTION]" in retry_context
                            )
                            coord._clear_stage_outputs_from("patcher")
                            agent_reported_success = bool(
                                coord._resume_from(
                                    "patcher",
                                    recovery_context=retry_context,
                                    escalation_level=2,
                                    deep_verify=False,
                                )
                            )
                            active_recovery_context = retry_context
                            candidate_audit["second_candidate_executed"] = True
                            candidate_audit["agent_reported_success_after_second_candidate"] = agent_reported_success
                            second_candidate_summary = workspace_patch_summary(
                                workspace,
                                list(manifest.get("oracle_patch_files", []) or []),
                            )
                            second_candidate_source_audit = _source_edit_contract_audit(
                                source_edit_contract=source_edit_contract,
                                patch_summary=second_candidate_summary,
                                stage_outputs=dict(coord.stage_outputs or {}),
                                workspace=workspace,
                            )
                            second_candidate_contract_audit = _failure_class_contract_candidate_audit(
                                contract=dict(candidate_audit.get("class_specific_edit_contract", {}) or {}),
                                patch_summary=second_candidate_summary,
                                source_edit_contract_audit=second_candidate_source_audit,
                                workspace=workspace,
                            )
                            second_candidate_syntax_guard_audit = _candidate_pre_admission_syntax_guard_audit(
                                enabled=candidate_pre_admission_syntax_guard,
                                workspace=workspace,
                                patch_summary=second_candidate_summary,
                            )
                            candidate_audit["second_candidate_patch_summary"] = second_candidate_summary
                            candidate_audit["second_candidate_source_edit_contract_audit"] = second_candidate_source_audit
                            candidate_audit["second_candidate_class_contract_audit"] = second_candidate_contract_audit
                            candidate_audit[
                                "second_candidate_pre_admission_syntax_guard_audit"
                            ] = second_candidate_syntax_guard_audit
                            candidate_audit["final_candidate_admitted"] = bool(
                                second_candidate_source_audit.get("satisfied", True)
                                and second_candidate_contract_audit.get("satisfied", False)
                                and second_candidate_syntax_guard_audit.get("satisfied", True)
                            )
                            candidate_audit["final_reason"] = (
                                "second_candidate_contract_admitted"
                                if candidate_audit["final_candidate_admitted"]
                                else (
                                    "second_candidate_syntax_guard_rejected"
                                    if not second_candidate_syntax_guard_audit.get("satisfied", True)
                                    else "second_candidate_still_contract_rejected"
                                )
                            )
                        bounded_signal_retry_audit["contract_verified_candidate_retry_audit"] = candidate_audit
    if verifier_failure_feedback_retry:
        feedback_retry_audit["attempted"] = True
        verifier_result = dict(coord.stage_outputs.get("verifier", {}) or {})
        patcher_result = dict(coord.stage_outputs.get("patcher", {}) or {})
        pending_feedback = str(getattr(coord, "_pending_verifier_failure_feedback", "") or "").strip()
        if agent_reported_success:
            feedback_retry_audit["reason"] = "pipeline_already_reported_success"
        elif not verifier_result:
            feedback_retry_audit["reason"] = "no_verifier_failure_available"
        elif not patcher_result.get("patch"):
            feedback_retry_audit["reason"] = "no_candidate_patch_to_revise"
        elif not pending_feedback:
            feedback_retry_audit["reason"] = "verifier_failure_feedback_missing"
        else:
            retry_context = coord._attach_pending_verifier_failure_feedback(
                resume_from="patcher",
                recovery_context=active_recovery_context,
            )
            feedback_retry_audit.update(
                {
                    "reason": "executed_after_verifier_failure",
                    "executed": True,
                    "uses_verifier_failure_feedback": True,
                    "retry_recovery_context_contains_feedback": "[VERIFIER FAILURE FEEDBACK]" in retry_context,
                }
            )
            agent_reported_success = bool(
                coord._resume_from("patcher", recovery_context=retry_context, escalation_level=1, deep_verify=False)
            )
            feedback_retry_audit["agent_reported_success_after_retry"] = agent_reported_success
    forced_revision_audit: dict[str, Any] = {
        "schema": "masguard.external_validation_forced_revision_gate.v1",
        "enabled": bool(external_feedback_audit.get("admitted", False)),
        "attempted": False,
        "executed": False,
        "reason": "disabled" if not external_feedback_audit.get("admitted", False) else "not_applicable",
        "repeated_previous_diff_before_retry": False,
        "repeated_previous_diff_after_retry": False,
        "semantic_target_covered_before_retry": True,
        "semantic_target_covered_after_retry": True,
        "agent_reported_success_after_retry": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_grant_recovery_credit": True,
            "blocks_repeated_failed_diff_from_promotion": True,
            "semantic_target_coverage_is_pre_oracle_audit": True,
        },
    }
    if external_feedback_audit.get("admitted", False):
        forced_revision_audit["attempted"] = True
        pre_force_patch_summary = workspace_patch_summary(workspace, list(manifest.get("oracle_patch_files", []) or []))
        pre_force_repeat_audit = _external_validation_feedback_repeat_diff_audit(
            admission=external_feedback_audit,
            workspace=workspace,
            changed_files=list(pre_force_patch_summary.get("changed_files", []) or []),
        )
        forced_revision_audit["pre_retry_repeat_diff_audit"] = pre_force_repeat_audit
        forced_revision_audit["repeated_previous_diff_before_retry"] = bool(
            pre_force_repeat_audit["repeated_previous_diff"]
        )
        pre_force_semantic_target_audit = _external_validation_semantic_target_audit(
            admission=external_feedback_audit,
            changed_files=list(pre_force_patch_summary.get("changed_files", []) or []),
        )
        forced_revision_audit["pre_retry_semantic_target_audit"] = pre_force_semantic_target_audit
        forced_revision_audit["semantic_target_covered_before_retry"] = bool(
            pre_force_semantic_target_audit["satisfied"]
        )
        forced_revision_reasons: list[str] = []
        if pre_force_repeat_audit["repeated_previous_diff"]:
            forced_revision_reasons.append("repeated_previous_diff")
        if not pre_force_semantic_target_audit["satisfied"]:
            forced_revision_reasons.append(str(pre_force_semantic_target_audit["reason"]))
        if not forced_revision_reasons:
            forced_revision_audit["reason"] = (
                f"{pre_force_repeat_audit['reason']};{pre_force_semantic_target_audit['reason']}"
            )
        else:
            forced_context = _external_validation_forced_revision_context(
                admission=external_feedback_audit,
                repeat_audit=pre_force_repeat_audit,
                semantic_target_audit=pre_force_semantic_target_audit,
                trigger_reasons=forced_revision_reasons,
            )
            retry_context = f"{active_recovery_context.strip()}\n\n{forced_context}".strip()
            forced_revision_audit.update(
                {
                    "reason": "forced_retry_after_" + "+".join(forced_revision_reasons),
                    "executed": True,
                    "forced_revision_context_attached": "[MASGUARD FORCED REVISION GATE]" in retry_context,
                }
            )
            coord._clear_stage_outputs_from("patcher")
            agent_reported_success = bool(
                coord._resume_from("patcher", recovery_context=retry_context, escalation_level=2, deep_verify=False)
            )
            active_recovery_context = retry_context
            forced_revision_audit["agent_reported_success_after_retry"] = agent_reported_success
            post_force_patch_summary = workspace_patch_summary(workspace, list(manifest.get("oracle_patch_files", []) or []))
            post_force_repeat_audit = _external_validation_feedback_repeat_diff_audit(
                admission=external_feedback_audit,
                workspace=workspace,
                changed_files=list(post_force_patch_summary.get("changed_files", []) or []),
            )
            forced_revision_audit["post_retry_repeat_diff_audit"] = post_force_repeat_audit
            forced_revision_audit["repeated_previous_diff_after_retry"] = bool(
                post_force_repeat_audit["repeated_previous_diff"]
            )
            post_force_semantic_target_audit = _external_validation_semantic_target_audit(
                admission=external_feedback_audit,
                changed_files=list(post_force_patch_summary.get("changed_files", []) or []),
            )
            forced_revision_audit["post_retry_semantic_target_audit"] = post_force_semantic_target_audit
            forced_revision_audit["semantic_target_covered_after_retry"] = bool(
                post_force_semantic_target_audit["satisfied"]
            )
    if source_edit_pre_oracle_retry and not historical_action_program_direct_script_enabled:
        source_edit_pre_oracle_retry_audit["attempted"] = True
        semantic_v2_classification = dict(
            bounded_signal_retry_audit.get("minimal_validation_failure_classification", {}) or {}
        )
        if not list(semantic_v2_classification.get("semantic_effect_cues", []) or []) and external_feedback_audit.get(
            "admitted", False
        ):
            semantic_v2_classification = dict(external_feedback_audit.get("failure_classification", {}) or {})
        semantic_v2_contract = dict(
            bounded_signal_retry_audit.get("minimal_validation_class_specific_edit_contract", {}) or {}
        )
        if not list(semantic_v2_contract.get("target_source_candidates", []) or []):
            semantic_v2_contract = _failure_class_edit_contract(semantic_v2_classification)
        first_candidate_summary = workspace_patch_summary(
            workspace,
            list(manifest.get("oracle_patch_files", []) or []),
        )
        first_candidate_source_audit = _source_edit_contract_audit(
            source_edit_contract=source_edit_contract,
            patch_summary=first_candidate_summary,
            stage_outputs=dict(coord.stage_outputs or {}),
            workspace=workspace,
        )
        first_candidate_gate = _source_edit_pre_oracle_candidate_gate(
            workspace=workspace,
            patch_summary=first_candidate_summary,
            source_edit_contract_audit=first_candidate_source_audit,
            patch_contract=patch_contract,
            stage_outputs=dict(coord.stage_outputs or {}),
            semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            semantic_effect_cues=list(semantic_v2_classification.get("semantic_effect_cues", []) or []),
            target_source_candidates=list(semantic_v2_contract.get("target_source_candidates", []) or []),
        )
        first_admitted = bool(first_candidate_gate["admitted"])
        source_edit_pre_oracle_retry_audit.update(
            {
                "first_candidate_patch_summary": first_candidate_summary,
                "first_candidate_source_edit_contract_audit": first_candidate_source_audit,
                "first_candidate_gate": first_candidate_gate,
                "first_candidate_admitted": first_admitted,
                "retry_allowed": not first_admitted,
                "final_candidate_admitted": first_admitted,
                "reason": (
                    "first_candidate_pre_oracle_admitted"
                    if first_admitted
                    else "first_candidate_pre_oracle_rejected_retry_allowed"
                ),
                "final_reason": (
                    "first_candidate_pre_oracle_admitted"
                    if first_admitted
                    else "no_second_candidate_yet"
                ),
            }
        )
        if not first_admitted:
            retry_context = (
                f"{active_recovery_context.strip()}\n\n"
                f"{_source_edit_pre_oracle_rejection_context(source_edit_pre_oracle_retry_audit)}"
            ).strip()
            source_edit_pre_oracle_retry_audit["rejection_context_attached"] = (
                "[MASGUARD SOURCE-EDIT PRE-ORACLE REJECTION]" in retry_context
            )
            coord._clear_stage_outputs_from("patcher")
            agent_reported_success = bool(
                coord._resume_from(
                    "patcher",
                    recovery_context=retry_context,
                    escalation_level=2,
                    deep_verify=False,
                )
            )
            active_recovery_context = retry_context
            source_edit_pre_oracle_retry_audit["second_candidate_executed"] = True
            source_edit_pre_oracle_retry_audit["agent_reported_success_after_second_candidate"] = agent_reported_success
            second_candidate_summary = workspace_patch_summary(
                workspace,
                list(manifest.get("oracle_patch_files", []) or []),
            )
            second_candidate_source_audit = _source_edit_contract_audit(
                source_edit_contract=source_edit_contract,
                patch_summary=second_candidate_summary,
                stage_outputs=dict(coord.stage_outputs or {}),
                workspace=workspace,
            )
            second_candidate_gate = _source_edit_pre_oracle_candidate_gate(
                workspace=workspace,
                patch_summary=second_candidate_summary,
                source_edit_contract_audit=second_candidate_source_audit,
                patch_contract=patch_contract,
                stage_outputs=dict(coord.stage_outputs or {}),
                semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                semantic_effect_cues=list(semantic_v2_classification.get("semantic_effect_cues", []) or []),
                target_source_candidates=list(semantic_v2_contract.get("target_source_candidates", []) or []),
            )
            second_admitted = bool(second_candidate_gate["admitted"])
            source_edit_pre_oracle_retry_audit.update(
                {
                    "second_candidate_patch_summary": second_candidate_summary,
                    "second_candidate_source_edit_contract_audit": second_candidate_source_audit,
                    "second_candidate_gate": second_candidate_gate,
                    "final_candidate_admitted": second_admitted,
                    "reason": (
                        "second_candidate_pre_oracle_admitted"
                        if second_admitted
                        else "second_candidate_pre_oracle_rejected_fail_closed"
                    ),
                    "final_reason": (
                        "second_candidate_pre_oracle_admitted"
                        if second_admitted
                        else "second_candidate_pre_oracle_rejected_fail_closed"
                    ),
                }
            )
    if diff_first_or_abstain:
        diff_first_or_abstain_audit = _diff_first_or_abstain_audit(
            enabled=True,
            stage_outputs=dict(coord.stage_outputs or {}),
            patch_summary=workspace_patch_summary(
                workspace,
                list(manifest.get("oracle_patch_files", []) or []),
            ),
        )
    latency_sec = time.monotonic() - started_at
    usage_after = _usage_snapshot(strong_model)
    usage = _usage_delta(usage_before, usage_after)

    evaluation = _evaluation_commands_for_manifest(manifest, command_source=evaluation_command_source)
    external_evaluation_blocked_by_repeat_diff = _external_validation_feedback_blocks_external_evaluation(
        external_feedback_audit=external_feedback_audit,
        forced_revision_audit=forced_revision_audit,
    )
    contract_verified_candidate_blocks_final_evaluation = _contract_verified_candidate_blocks_final_evaluation(
        bounded_signal_retry_audit
    )
    source_edit_pre_oracle_blocks_final_evaluation = _source_edit_pre_oracle_blocks_final_evaluation(
        source_edit_pre_oracle_retry_audit
    )
    historical_action_program_direct_script_blocks_final_evaluation = (
        _historical_action_program_direct_script_blocks_final_evaluation(
            historical_action_program_direct_script_audit
        )
    )
    diff_first_or_abstain_blocks_final_evaluation = _diff_first_or_abstain_blocks_final_evaluation(
        diff_first_or_abstain_audit
    )
    if external_evaluation_blocked_by_repeat_diff:
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": "external_validation_feedback_repeated_diff_blocked",
        }
    elif contract_verified_candidate_blocks_final_evaluation:
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": "contract_verified_candidate_rejected_pre_oracle",
        }
    elif source_edit_pre_oracle_blocks_final_evaluation:
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": "source_edit_pre_oracle_candidate_rejected",
        }
    elif historical_action_program_direct_script_blocks_final_evaluation:
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": "historical_action_program_direct_script_rejected",
        }
    elif diff_first_or_abstain_blocks_final_evaluation:
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": "diff_first_or_abstain_not_satisfied",
        }
    else:
        evaluation = _runtime_validate_evaluation_commands(
            manifest=manifest,
            evaluation=evaluation,
            executor=executor,
            workspace=workspace,
            timeout=min(timeout, 300),
        )
    if (
        not external_evaluation_blocked_by_repeat_diff
        and not contract_verified_candidate_blocks_final_evaluation
        and not source_edit_pre_oracle_blocks_final_evaluation
        and not historical_action_program_direct_script_blocks_final_evaluation
        and not diff_first_or_abstain_blocks_final_evaluation
        and evaluation.get("error_type")
    ):
        fail_to_pass = {"returncode": None, "output": ""}
        oracle = {"returncode": None, "output": ""}
        protocol = {
            "fail_to_pass_protocol_error": True,
            "oracle_protocol_error": True,
            "evaluation_protocol_error": True,
            "evaluation_protocol_error_type": str(evaluation.get("stop_reason", "") or "evaluation_protocol_blocked"),
        }
    elif (
        not external_evaluation_blocked_by_repeat_diff
        and not contract_verified_candidate_blocks_final_evaluation
        and not source_edit_pre_oracle_blocks_final_evaluation
        and not historical_action_program_direct_script_blocks_final_evaluation
        and not diff_first_or_abstain_blocks_final_evaluation
    ):
        fail_to_pass = executor.execute(evaluation["fail_to_pass_command"], cwd=str(workspace), timeout=timeout)
        oracle = executor.execute(evaluation["oracle_command"], cwd=str(workspace), timeout=timeout)
        protocol = _evaluation_protocol_status(fail_to_pass=fail_to_pass, oracle=oracle)

    patch_summary = workspace_patch_summary(workspace, list(manifest.get("oracle_patch_files", []) or []))
    external_validation_failure_audit = _external_validation_failure_audit(
        evaluation=evaluation,
        fail_to_pass=fail_to_pass,
        oracle=oracle,
        protocol=protocol,
        patch_summary=patch_summary,
    )
    patch_classification = _classify_patch_summary(patch_summary)
    external_feedback_repeat_diff_audit = _external_validation_feedback_repeat_diff_audit(
        admission=external_feedback_audit,
        workspace=workspace,
        changed_files=list(patch_summary.get("changed_files", []) or []),
    )
    source_edit_contract_audit = _source_edit_contract_audit(
        source_edit_contract=source_edit_contract,
        patch_summary=patch_summary,
        stage_outputs=dict(coord.stage_outputs or {}),
        workspace=workspace,
    )
    return {
        **_evaluation_command_payload(evaluation),
        "fail_to_pass_returncode": fail_to_pass["returncode"],
        "oracle_returncode": oracle["returncode"],
        "oracle_success": oracle["returncode"] == 0,
        "reported_success": oracle["returncode"] == 0,
        "agent_reported_success": agent_reported_success,
        "initial_agent_reported_success": initial_agent_reported_success,
        "fail_to_pass_output": fail_to_pass["output"],
        "oracle_output": oracle["output"],
        **protocol,
        "external_validation_failure_audit": external_validation_failure_audit,
        "external_validation_failure_observed": bool(external_validation_failure_audit["failure_observed"]),
        "external_validation_failure_signature": dict(external_validation_failure_audit["signature"]),
        "external_validation_failure_signature_present": bool(
            external_validation_failure_audit["signature_present"]
        ),
        "external_validation_failure_exception_classes": list(
            external_validation_failure_audit["signature"].get("exception_classes", [])
        ),
        "external_validation_failure_failing_tests": list(
            external_validation_failure_audit["signature"].get("failing_tests", [])
        ),
        "external_validation_failure_traceback_source_files": list(
            external_validation_failure_audit["signature"].get("traceback_source_files", [])
        ),
        "external_validation_tool_missing": bool(
            external_validation_failure_audit["signature"].get("validation_tool_missing", False)
        ),
        "external_validation_failure_sources": list(external_validation_failure_audit["failure_sources"]),
        "external_validation_feedback_ready": bool(external_validation_failure_audit["feedback_ready"]),
        "external_validation_source_repair_feedback_ready": bool(
            external_validation_failure_audit["source_repair_feedback_ready"]
        ),
        "external_validation_feedback_repeat_diff_audit": external_feedback_repeat_diff_audit,
        "external_validation_feedback_repeated_previous_diff": bool(
            external_feedback_repeat_diff_audit["repeated_previous_diff"]
        ),
        "external_validation_forced_revision_audit": forced_revision_audit,
        "external_validation_forced_revision_executed": bool(forced_revision_audit["executed"]),
        "external_validation_forced_revision_repeated_after_retry": bool(
            forced_revision_audit["repeated_previous_diff_after_retry"]
        ),
        "external_validation_forced_revision_semantic_target_covered_after_retry": bool(
            forced_revision_audit["semantic_target_covered_after_retry"]
        ),
        "external_validation_feedback_external_evaluation_blocked_by_repeat_diff": bool(
            external_evaluation_blocked_by_repeat_diff
        ),
        "contract_verified_candidate_final_evaluation_blocked": bool(
            contract_verified_candidate_blocks_final_evaluation
        ),
        "external_validation_feedback_promotion_blocked_by_repeat_diff": bool(
            external_feedback_repeat_diff_audit["repeated_previous_diff"]
            or forced_revision_audit["repeated_previous_diff_after_retry"]
            or not forced_revision_audit["semantic_target_covered_after_retry"]
        ),
        "external_validation_feedback_live_credit_ready": bool(
            oracle["returncode"] == 0
            and not external_feedback_repeat_diff_audit["repeated_previous_diff"]
            and not forced_revision_audit["repeated_previous_diff_after_retry"]
            and forced_revision_audit["semantic_target_covered_after_retry"]
        ),
        "model_name": str(getattr(getattr(model, "config", None), "model", "") or model_name or ""),
        "strong_model_name": str(
            getattr(getattr(strong_model, "config", None), "model", "") or strong_model_name or model_name or ""
        ),
        "model_usage_before": usage_before,
        "model_usage_after": usage_after,
        "model_usage_delta": usage,
        "token_cost": float(usage.get("total_tokens", 0.0) or 0.0),
        "model_call_count": int(usage.get("n_calls", 0.0) or 0.0),
        "latency_sec": max(0.001, float(latency_sec)),
        "patch_legitimacy": patch_classification["patch_legitimacy"],
        "patch_contract": patch_contract,
        "locator_failure_salvage": locator_failure_salvage,
        "source_edit_contract": source_edit_contract,
        "method_variant": (
            _method_variant(
                verifier_failure_feedback_retry=verifier_failure_feedback_retry,
                bounded_validation_signal_retry=bounded_validation_signal_retry,
                failure_class_conditioned_retry=failure_class_conditioned_retry,
                contract_verified_candidate_retry=contract_verified_candidate_retry,
                external_validation_feedback_artifact=external_validation_feedback_artifact,
                baseline_success_contract_artifact=baseline_success_contract_artifact,
                historical_experience_artifact=historical_experience_artifact,
                historical_experience_mode=historical_experience_mode,
                mas_experience_controller_artifact=mas_experience_controller_artifact,
                historical_action_program_artifact=historical_action_program_artifact,
                diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
                    candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
                    source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
                    diff_first_or_abstain=diff_first_or_abstain,
                    semantic_patch_correctness_v2=semantic_patch_correctness_v2,
                )
        ),
        "external_validation_feedback_admission_audit": external_feedback_audit,
        "external_validation_feedback_context_attached": bool(external_feedback_context),
        "baseline_success_contract_admission_audit": baseline_contract_audit,
        "baseline_success_contract_context_attached": bool(baseline_contract_context),
        "historical_experience_admission_audit": historical_experience_audit,
        "historical_experience_context_attached": bool(historical_experience_context),
        "mas_experience_controller_admission_audit": mas_experience_controller_audit,
        "mas_experience_controller_context_attached": bool(mas_experience_controller_context),
        "historical_action_program_admission_audit": historical_action_program_audit,
        "historical_action_program_context_attached": bool(historical_action_program_context),
        "historical_action_program_direct_script_enabled": bool(
            historical_action_program_direct_script_enabled
        ),
        "historical_action_program_direct_script_audit": historical_action_program_direct_script_audit,
        "historical_action_program_direct_script_executed": bool(
            historical_action_program_direct_script_audit.get("executed", False)
        ),
        "historical_action_program_direct_script_final_candidate_admitted": bool(
            historical_action_program_direct_script_audit.get("final_candidate_admitted", False)
        ),
        "historical_action_program_direct_script_final_evaluation_blocked": bool(
            historical_action_program_direct_script_blocks_final_evaluation
        ),
        "bounded_validation_signal_retry_audit": bounded_signal_retry_audit,
        "bounded_validation_signal_retry_enabled": bool(bounded_validation_signal_retry),
        "bounded_validation_signal_retry_executed": bool(bounded_signal_retry_audit["executed"]),
        "bounded_validation_signal_retry_feedback_ready": bool(
            bounded_signal_retry_audit["minimal_validation_feedback_ready"]
        ),
        "bounded_validation_signal_retry_source_repair_feedback_ready": bool(
            bounded_signal_retry_audit["minimal_validation_source_repair_feedback_ready"]
        ),
        "failure_class_conditioned_retry_enabled": bool(failure_class_conditioned_retry),
        "contract_verified_candidate_retry_enabled": bool(contract_verified_candidate_retry),
        "candidate_pre_admission_syntax_guard_enabled": bool(candidate_pre_admission_syntax_guard),
        "diverse_repair_hypothesis_retry_enabled": bool(diverse_repair_hypothesis_retry),
        "diverse_repair_hypothesis_audit": dict(
            bounded_signal_retry_audit.get("diverse_repair_hypothesis_audit", {}) or {}
        ),
        "diverse_repair_hypothesis_context_attached": bool(
            dict(bounded_signal_retry_audit.get("diverse_repair_hypothesis_audit", {}) or {}).get(
                "context_attached", False
            )
        ),
        "contract_verified_candidate_retry_audit": dict(
            bounded_signal_retry_audit.get("contract_verified_candidate_retry_audit", {}) or {}
        ),
        "contract_verified_candidate_retry_executed": bool(
            dict(bounded_signal_retry_audit.get("contract_verified_candidate_retry_audit", {}) or {}).get(
                "second_candidate_executed", False
            )
        ),
        "contract_verified_candidate_retry_final_candidate_admitted": bool(
            dict(bounded_signal_retry_audit.get("contract_verified_candidate_retry_audit", {}) or {}).get(
                "final_candidate_admitted", False
            )
        ),
        "source_edit_pre_oracle_retry_enabled": bool(source_edit_pre_oracle_retry),
        "semantic_patch_correctness_v2_enabled": bool(semantic_patch_correctness_v2),
        "source_edit_pre_oracle_retry_audit": source_edit_pre_oracle_retry_audit,
        "source_edit_pre_oracle_retry_executed": bool(
            source_edit_pre_oracle_retry_audit.get("second_candidate_executed", False)
        ),
        "source_edit_pre_oracle_retry_final_candidate_admitted": bool(
            source_edit_pre_oracle_retry_audit.get("final_candidate_admitted", False)
        ),
        "source_edit_pre_oracle_final_evaluation_blocked": bool(
            source_edit_pre_oracle_blocks_final_evaluation
        ),
        "source_edit_pre_oracle_semantic_patch_correctness_v2_attempted": bool(
            dict(source_edit_pre_oracle_retry_audit.get("first_candidate_gate", {}) or {})
            .get("semantic_patch_correctness_v2", {})
            .get("attempted", False)
        ),
        "source_edit_pre_oracle_semantic_patch_correctness_v2_final_satisfied": bool(
            dict(
                dict(source_edit_pre_oracle_retry_audit.get("second_candidate_gate", {}) or {})
                .get(
                    "semantic_patch_correctness_v2",
                    dict(source_edit_pre_oracle_retry_audit.get("first_candidate_gate", {}) or {}).get(
                        "semantic_patch_correctness_v2", {}
                    ),
                )
                or {}
            ).get("satisfied", True)
        ),
        "diff_first_or_abstain_enabled": bool(diff_first_or_abstain),
        "diff_first_or_abstain_audit": diff_first_or_abstain_audit,
        "diff_first_or_abstain_intent_present": bool(
            diff_first_or_abstain_audit.get("intent_present", False)
        ),
        "diff_first_or_abstain_abstained": bool(
            diff_first_or_abstain_audit.get("abstained", False)
        ),
        "diff_first_or_abstain_final_evaluation_blocked": bool(
            diff_first_or_abstain_blocks_final_evaluation
        ),
        "minimal_validation_failure_class": str(
            bounded_signal_retry_audit.get("minimal_validation_failure_class", "unknown") or "unknown"
        ),
        "minimal_validation_retry_action": str(
            bounded_signal_retry_audit.get("minimal_validation_retry_action", "abstain") or "abstain"
        ),
        "minimal_validation_controlled_no_diff_retry_ready": bool(
            bounded_signal_retry_audit.get("minimal_validation_controlled_no_diff_retry_ready", False)
        ),
        "minimal_validation_failure_classification": dict(
            bounded_signal_retry_audit.get("minimal_validation_failure_classification", {}) or {}
        ),
        "minimal_validation_class_specific_edit_contract": dict(
            bounded_signal_retry_audit.get("minimal_validation_class_specific_edit_contract", {}) or {}
        ),
        "verifier_failure_feedback_retry_audit": feedback_retry_audit,
        "verifier_failure_feedback_retry_enabled": bool(verifier_failure_feedback_retry),
        "verifier_failure_feedback_retry_executed": bool(feedback_retry_audit["executed"]),
        "source_edit_contract_audit": source_edit_contract_audit,
        "source_edit_contract_adhered": bool(source_edit_contract_audit["satisfied"]),
        "source_edit_contract_violation_types": list(source_edit_contract_audit["violations"]),
        "locator_failure_salvage_audit": salvage_audit,
        "locator_failure_salvage_attempted": bool(salvage_audit["attempted"]),
        "locator_failure_salvage_accepted": bool(salvage_audit["accepted"]),
        "locator_failure_salvage_candidate_paths": list(salvage_audit["candidate_paths"]),
        "locator_failure_salvage_reason": str(salvage_audit["reason"]),
        "patch_contract_prompted": bool(recovery_context.strip()),
        "patch_contract_adhered": _patch_contract_adhered(
            patch_contract=patch_contract,
            patch_legitimacy=patch_classification["patch_legitimacy"],
        ),
        "patch_contract_violation_types": _patch_contract_violations(
            patch_contract=patch_contract,
            patch_legitimacy=patch_classification["patch_legitimacy"],
        ),
        "patch_summary": {
            **patch_summary,
            "changed_file_classes": patch_classification["changed_file_classes"],
        },
        "stage_outputs": dict(coord.stage_outputs or {}),
        "stage_trace": summarize_compact_stage_trace(coord.stage_outputs or {}),
        "clean_start_policy": _clean_start_policy(
            patch_contract,
            locator_failure_salvage,
            source_edit_contract,
            verifier_failure_feedback_retry,
            bounded_validation_signal_retry,
            failure_class_conditioned_retry,
            contract_verified_candidate_retry,
            external_validation_feedback_artifact,
            baseline_success_contract_artifact,
            historical_experience_artifact,
            historical_experience_mode=historical_experience_mode,
            diverse_repair_hypothesis_retry=diverse_repair_hypothesis_retry,
            candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
            source_edit_pre_oracle_retry=source_edit_pre_oracle_retry,
            diff_first_or_abstain=diff_first_or_abstain,
            semantic_patch_correctness_v2=semantic_patch_correctness_v2,
            mas_experience_controller_artifact=mas_experience_controller_artifact,
            historical_action_program_artifact=historical_action_program_artifact,
        ),
    }


def _external_validation_failure_audit(
    *,
    evaluation: dict[str, Any],
    fail_to_pass: dict[str, Any],
    oracle: dict[str, Any],
    protocol: dict[str, Any],
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "schema": "masguard.external_validation_failure_audit.v1",
        "failure_observed": False,
        "signature_present": False,
        "feedback_ready": False,
        "failure_sources": [],
        "signature": _empty_validation_failure_signature(),
        "failure_classification": _minimal_validation_failure_classification(
            protocol=protocol,
            signature=_empty_validation_failure_signature(),
            failure_sources=[],
            patch_summary=patch_summary,
        ),
        "failure_class": "unknown",
        "source_repair_feedback_ready": False,
        "reason": "",
        "claim_boundary": {
            "post_action_audit_only": True,
            "does_not_grant_recovery_credit": True,
            "intended_use": "condition a future bounded retry or minimal-probe decision on external validation evidence",
        },
    }
    if protocol.get("evaluation_protocol_error"):
        base["reason"] = str(protocol.get("evaluation_protocol_error_type", "") or "evaluation_protocol_error")
        return base

    source_rows = [
        ("fail_to_pass", str(evaluation.get("fail_to_pass_command", "") or ""), fail_to_pass),
        ("oracle", str(evaluation.get("oracle_command", "") or ""), oracle),
    ]
    signatures: list[dict[str, object]] = []
    failure_sources: list[dict[str, Any]] = []
    for source_name, command, result in source_rows:
        returncode = result.get("returncode")
        if returncode in (None, 0):
            continue
        output = str(result.get("output", "") or "")
        failure_sources.append(
            {
                "source": source_name,
                "returncode": returncode,
                "command": command,
                "output_excerpt": output[:1200],
            }
        )
        signatures.append(validation_failure_signature(command=command, output=output))

    merged_signature = _merge_validation_failure_signatures(signatures)
    signature_present = _validation_failure_signature_has_signal(merged_signature)
    failure_observed = bool(failure_sources)
    source_diff_present = bool(list(patch_summary.get("non_test_files", []) or []))
    failure_classification = _minimal_validation_failure_classification(
        protocol=protocol,
        signature=merged_signature,
        failure_sources=failure_sources,
        patch_summary=patch_summary,
    )
    base.update(
        {
            "failure_observed": failure_observed,
            "signature_present": signature_present,
            "feedback_ready": bool(failure_observed and (signature_present or any(row["output_excerpt"] for row in failure_sources))),
            "source_repair_feedback_ready": bool(failure_observed and source_diff_present and signature_present),
            "failure_sources": failure_sources,
            "signature": merged_signature,
            "failure_classification": failure_classification,
            "failure_class": failure_classification["failure_class"],
            "reason": "external_validation_failed" if failure_observed else "external_validation_passed",
        }
    )
    return base


def _minimal_validation_failure_classification(
    *,
    protocol: dict[str, Any],
    signature: dict[str, object],
    failure_sources: list[dict[str, Any]],
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    output_text = "\n".join(
        str(item.get("output_excerpt", "") or "") for item in failure_sources if isinstance(item, dict)
    )
    output_lower = output_text.lower()
    semantic_effect_cues = _semantic_effect_cues_from_output(output_text)
    exception_classes = [str(item) for item in list(signature.get("exception_classes", []) or [])]
    exception_names = {item.rsplit(".", 1)[-1] for item in exception_classes}
    traceback_files = [str(item) for item in list(signature.get("traceback_source_files", []) or [])]
    production_traceback_targets = _production_traceback_target_suffixes(traceback_files)
    changed_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("non_test_files", []) or patch_summary.get("changed_files", []) or [])
        if _normalize_repo_path(path)
    ]
    suspect_paths = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("suspect_paths", []) or [])
        if _normalize_repo_path(path)
    ]
    traceback_overlap = _path_suffix_overlap(changed_files, production_traceback_targets)
    suspect_overlap = _path_suffix_overlap(changed_files, suspect_paths)
    no_source_diff = not changed_files
    protocol_blocked = bool(protocol.get("evaluation_protocol_error")) or bool(signature.get("validation_tool_missing"))
    env_markers = (
        "unicodeencodeerror",
        "setup_databases",
        "create_test_db",
        "command not found",
        "no module named pytest",
        "validation_tool_missing",
        "migrate.py",
    )
    reasons: list[str] = []
    failure_class = "unknown"
    retry_action = "abstain"
    prompt_guidance = "Evidence is insufficient for a safe source edit; abstain unless a minimal targeted edit is justified."
    confidence = "low"

    if protocol_blocked or any(marker in output_lower for marker in env_markers):
        failure_class = "environment_or_protocol"
        retry_action = "abstain"
        confidence = "medium"
        prompt_guidance = (
            "Treat this as environment/protocol evidence, not source-repair evidence. Do not widen the source edit."
        )
        reasons.append("protocol_or_environment_marker")
    elif no_source_diff:
        failure_class = "no_diff"
        retry_action = "retry_from_suspect_source"
        confidence = "high" if suspect_paths or production_traceback_targets else "medium"
        prompt_guidance = (
            "The previous attempt produced no source diff. Build one minimal source-only edit using suspect paths, "
            "failing tests, and exception evidence; do not edit tests or commands."
        )
        reasons.append("no_source_diff")
    elif production_traceback_targets and not traceback_overlap:
        failure_class = "source_location"
        retry_action = "retarget_source_span"
        confidence = "high"
        prompt_guidance = (
            "The failing traceback points outside the edited source span. Retarget the source edit toward the "
            "production traceback path or an adjacent caller/callee."
        )
        reasons.append("production_traceback_not_covered_by_diff")
    elif suspect_paths and not suspect_overlap:
        failure_class = "source_location"
        retry_action = "retarget_suspect_path"
        confidence = "medium"
        prompt_guidance = (
            "The previous diff missed the suspected source path. Retarget to the suspect path before changing "
            "broader behavior."
        )
        reasons.append("suspect_path_not_touched")
    elif {"TypeError", "ValueError", "AttributeError", "DatabaseError"} & exception_names:
        failure_class = "wrong_abstraction"
        retry_action = "revise_semantic_mechanism"
        confidence = "medium"
        prompt_guidance = (
            "The previous edit likely used the wrong abstraction or API boundary. Revise the mechanism to explain "
            "the exception class and failing test, not just the local symptom."
        )
        reasons.append("semantic_exception_class")
    elif (
        "assertionerror" in output_lower
        or "!=" in output_text
        or "not equal" in output_lower
        or bool(semantic_effect_cues)
    ):
        failure_class = "missing_condition"
        retry_action = "add_narrow_condition"
        confidence = "medium"
        prompt_guidance = (
            "The validation failure is value/condition-oriented. Add the narrow missing condition or edge-case "
            "handling needed by the failing tests."
        )
        reasons.append("assertion_or_value_mismatch")
        if semantic_effect_cues:
            reasons.append("semantic_effect_cues_extracted")
    elif failure_sources:
        failure_class = "unknown"
        retry_action = "guarded_retry"
        confidence = "low"
        reasons.append("failure_observed_without_stable_class")
    else:
        reasons.append("no_failure_observed")

    return {
        "schema": "masguard.minimal_validation_failure_classification.v1",
        "failure_class": failure_class,
        "retry_action": retry_action,
        "confidence": confidence,
        "prompt_guidance": prompt_guidance,
        "reasons": reasons,
        "changed_files": changed_files[:8],
        "suspect_paths": suspect_paths[:8],
        "production_traceback_targets": production_traceback_targets[:8],
        "covered_traceback_targets": traceback_overlap[:8],
        "covered_suspect_paths": suspect_overlap[:8],
        "exception_classes": exception_classes[:8],
        "semantic_effect_cues": semantic_effect_cues[:8],
        "claim_boundary": {
            "does_not_use_oracle_success": True,
            "heuristic_signal_only": True,
            "requires_fresh_execution_for_credit": True,
        },
    }


def _semantic_effect_cues_from_output(output_text: str) -> list[dict[str, str]]:
    """Extract deterministic expected-vs-actual cues from validation output."""

    text = str(output_text or "")
    cues: list[dict[str, str]] = []
    pending_call_target: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        call_match = re.search(r"=\s*([A-Za-z_][\w.]*)\([^)]{0,240}\)", stripped)
        if call_match:
            call_target = call_match.group(1).strip()
            if call_target not in {"array", "list", "tuple", "dict", "set"}:
                pending_call_target = call_target
        assert_match = re.search(r"assert\s+(.{1,160}?)\s*==\s*(.{1,160})$", stripped)
        if assert_match:
            cues.append(
                {
                    "kind": "assert_equal",
                    "actual": assert_match.group(1).strip(),
                    "expected": assert_match.group(2).strip(),
                    "source": stripped[:240],
                }
            )
            boundary_match = re.search(
                r"assert\s+(.+?\[\s*0\s*\])\s*==\s*([^\s]+)|assert\s+(.+?\[\s*-1\s*\])\s*==\s*([^\s]+)",
                stripped,
            )
            if boundary_match:
                actual = str(boundary_match.group(1) or boundary_match.group(3) or "").strip()
                expected = str(boundary_match.group(2) or boundary_match.group(4) or "").strip()
                cues.append(
                    {
                        "kind": "boundary_value",
                        "actual": actual,
                        "expected": expected,
                        "source": stripped[:240],
                    }
                )
            continue
        reverse_match = re.search(r"assert\s+(.{1,160}?)\s*!=\s*(.{1,160})$", stripped)
        if reverse_match:
            cues.append(
                {
                    "kind": "assert_not_equal",
                    "left": reverse_match.group(1).strip(),
                    "right": reverse_match.group(2).strip(),
                    "source": stripped[:240],
                }
            )
            continue
        pytest_match = re.search(r"(?i)\b(expected|actual|obtained)\b\s*[:=]\s*(.{1,160})", stripped)
        if pytest_match:
            cues.append(
                {
                    "kind": pytest_match.group(1).lower(),
                    "value": pytest_match.group(2).strip(),
                    "source": stripped[:240],
                }
            )
            continue
        unpack_match = re.search(r"([A-Za-z_][\w]*)\s*,\s*([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\s*$", stripped)
        if unpack_match:
            cues.append(
                {
                    "kind": "unpack_source",
                    "targets": f"{unpack_match.group(1)}, {unpack_match.group(2)}",
                    "source_value": unpack_match.group(3),
                    "source": stripped[:240],
                }
            )
            continue
        class_value_match = re.search(r"([A-Za-z_][\w]*)\s*=\s*<class '([^']+)'>", stripped)
        if class_value_match:
            value_name = class_value_match.group(1)
            value_class = class_value_match.group(2)
            cue = {
                "kind": "unexpected_class_value",
                "name": value_name,
                "class": value_class,
                "source": stripped[:240],
            }
            if pending_call_target:
                cue["call_target"] = pending_call_target
            if value_name == "range" and value_class == "range":
                cue["required_behavior"] = "pass the user supplied range tuple through the repair path, not the Python built-in range class"
            cues.append(cue)
            continue
    deduped: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for cue in cues:
        key = tuple(sorted(cue.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cue)
        if len(deduped) >= 8:
            break
    return deduped


def _external_validation_feedback_admission_audit(
    *,
    artifact: Path | None,
    instance_id: str,
) -> dict[str, Any]:
    base = {
        "schema": "masguard.external_validation_feedback_admission.v1",
        "enabled": artifact is not None,
        "artifact": str(artifact or ""),
        "admitted": False,
        "reason": "disabled" if artifact is None else "",
        "instance_id": instance_id,
        "artifact_instance_id": "",
        "previous_source_files": [],
        "previous_diff_excerpt": "",
        "signature": _empty_validation_failure_signature(),
        "failure_sources": [],
        "failure_classification": _minimal_validation_failure_classification(
            protocol={},
            signature=_empty_validation_failure_signature(),
            failure_sources=[],
            patch_summary={},
        ),
        "previous_oracle_success": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_use_gold_patch": True,
            "does_not_grant_recovery_credit": True,
            "requires_fresh_oracle_audit": True,
        },
    }
    if artifact is None:
        return base
    if not artifact.is_file():
        base["reason"] = "artifact_missing"
        return base
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = "artifact_unreadable"
        base["error"] = str(exc)
        return base
    row = _row_for_instance(payload, instance_id)
    if not row:
        base["reason"] = "instance_row_missing"
        return base
    artifact_instance_id = str(row.get("instance_id", "") or payload.get("instance_id", "") or "")
    base["artifact_instance_id"] = artifact_instance_id
    if artifact_instance_id and artifact_instance_id != instance_id:
        base["reason"] = "instance_mismatch"
        return base
    if bool(row.get("oracle_success", False)):
        base["previous_oracle_success"] = True
        base["reason"] = "previous_row_already_oracle_success"
        return base
    patch_summary = dict(row.get("patch_summary", {}) or {})
    source_files = [str(path) for path in list(patch_summary.get("non_test_files", []) or []) if str(path).strip()]
    base["previous_source_files"] = source_files[:8]
    if not source_files:
        base["reason"] = "previous_row_has_no_source_diff"
        return base
    base["previous_diff_excerpt"] = _previous_source_diff_excerpt(row, source_files)
    signature = _external_validation_signature_from_row(row)
    base["signature"] = signature
    if not _validation_failure_signature_has_signal(signature):
        base["reason"] = "external_validation_signature_missing"
        return base
    failure_sources = _external_validation_failure_sources_from_row(row)
    base["failure_sources"] = failure_sources
    if not failure_sources:
        base["reason"] = "external_validation_failure_source_missing"
        return base
    base["failure_classification"] = _minimal_validation_failure_classification(
        protocol={},
        signature=signature,
        failure_sources=failure_sources,
        patch_summary=patch_summary,
    )
    base["admitted"] = True
    base["reason"] = "admitted_source_diff_with_structured_external_validation_failure"
    return base


def _baseline_success_contract_admission_audit(
    *,
    artifact: Path | None,
    instance_id: str,
) -> dict[str, Any]:
    base = {
        "schema": "masguard.baseline_success_contract_admission.v1",
        "enabled": artifact is not None,
        "artifact": str(artifact or ""),
        "admitted": False,
        "reason": "disabled" if artifact is None else "",
        "instance_id": instance_id,
        "artifact_instance_id": "",
        "baseline_oracle_success": False,
        "source_files": [],
        "test_files": [],
        "patch_legitimacy": "",
        "diff_excerpt": "",
        "claim_boundary": {
            "uses_baseline_success_patch_as_contract_only": True,
            "does_not_use_gold_patch": True,
            "does_not_grant_recovery_credit": True,
            "requires_fresh_oracle_audit": True,
            "must_be_reported_as_transfer_probe": True,
        },
    }
    if artifact is None:
        return base
    if not artifact.is_file():
        base["reason"] = "artifact_missing"
        return base
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = "artifact_unreadable"
        base["error"] = str(exc)
        return base
    row = _row_for_instance(payload, instance_id)
    if not row:
        base["reason"] = "instance_row_missing"
        return base
    artifact_instance_id = str(row.get("instance_id", "") or payload.get("instance_id", "") or "")
    base["artifact_instance_id"] = artifact_instance_id
    if artifact_instance_id and artifact_instance_id != instance_id:
        base["reason"] = "instance_mismatch"
        return base
    if not bool(row.get("oracle_success", False)):
        base["reason"] = "baseline_row_not_oracle_success"
        return base
    patch_summary = dict(row.get("patch_summary", {}) or {})
    source_files = _patch_summary_source_files(patch_summary)
    test_files = _patch_summary_test_files(patch_summary)
    patch_legitimacy = str(row.get("patch_legitimacy", "") or _classify_patch_summary(patch_summary)["patch_legitimacy"])
    base.update(
        {
            "baseline_oracle_success": True,
            "source_files": source_files[:8],
            "test_files": test_files[:8],
            "patch_legitimacy": patch_legitimacy,
        }
    )
    if not source_files:
        base["reason"] = "baseline_row_has_no_source_diff"
        return base
    if patch_legitimacy not in {"source_only", "source_mixed"}:
        base["reason"] = "baseline_patch_not_source_based"
        return base
    if test_files:
        base["reason"] = "admitted_source_mixed_baseline_contract_tests_ignored"
    else:
        base["reason"] = "admitted_source_only_baseline_contract"
    base["diff_excerpt"] = _previous_source_diff_excerpt(row, source_files)[:3000]
    if not str(base["diff_excerpt"]).strip():
        base["reason"] = "baseline_diff_excerpt_missing"
        return base
    base["admitted"] = True
    return base


def _baseline_success_contract_context(admission: dict[str, Any]) -> str:
    if not admission.get("admitted"):
        return ""
    payload = {
        "schema": "masguard.baseline_success_contract_context.v1",
        "source_files": list(admission.get("source_files", []) or [])[:8],
        "test_files_ignored": list(admission.get("test_files", []) or [])[:8],
        "patch_legitimacy": str(admission.get("patch_legitimacy", "") or ""),
        "diff_excerpt": str(admission.get("diff_excerpt", "") or "")[:2200],
        "selection_policy": {
            "use_as_patch_contract_not_as_answer_copy": True,
            "derive_a_fresh_source_only_patch": True,
            "prefer_same_source_files_when_supported_by_mas_failure_evidence": True,
            "do_not_edit_tests_even_if_baseline_did": True,
            "fresh_oracle_required_for_credit": True,
        },
        "hard_rules": [
            "This is a successful baseline patch contract, not gold ground truth and not direct recovery credit.",
            "Use the listed source files and diff shape to constrain a fresh source-only MASGuard repair.",
            "Do not edit tests, generated files, evaluation commands, or broad runtime setup.",
            "If the baseline artifact changed tests, ignore those test edits and implement only a source-code repair.",
            "State the MAS failure mechanism before editing and explain why the selected source file is sufficient.",
            "The final patch must pass the same oracle/action/source/protocol/runtime/cost audit as other branches.",
        ],
    }
    return (
        "[MASGUARD BASELINE SUCCESS CONTRACT]\n"
        "A successful strong-baseline artifact is available for this same instance. Use it only as a bounded "
        "source-diff contract to synthesize a fresh source-only recovery from the failed MAS trajectory; do not "
        "treat the baseline success itself as MASGuard recovery credit.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD BASELINE SUCCESS CONTRACT]"
    )


def _historical_experience_admission_audit(
    *,
    artifact: Path | None,
    target_instance_id: str,
    mode: str = "mas_conditioned",
) -> dict[str, Any]:
    base = {
        "schema": "masguard.historical_experience_admission.v1",
        "enabled": artifact is not None,
        "artifact": str(artifact or ""),
        "mode": mode,
        "admitted": False,
        "reason": "disabled" if artifact is None else "",
        "target_instance_id": target_instance_id,
        "case_instance_id": "",
        "case_family": "",
        "case_oracle_success": False,
        "source_files": [],
        "test_files": [],
        "patch_legitimacy": "",
        "diff_excerpt": "",
        "abstracted_experience": {},
        "claim_boundary": {
            "cross_instance_case_memory_only": True,
            "rejects_same_instance_cases": True,
            "does_not_use_gold_patch": True,
            "does_not_grant_recovery_credit": True,
            "requires_fresh_oracle_audit": True,
            "mas_specific_when_combined_with_graph_or_validation_evidence": True,
            "generic_patch_memory_mode_is_baseline_comparator_not_masguard_credit": (
                mode == "generic_patch_memory"
            ),
        },
    }
    if artifact is None:
        return base
    if not artifact.is_file():
        base["reason"] = "artifact_missing"
        return base
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = "artifact_unreadable"
        base["error"] = str(exc)
        return base
    row = _first_historical_experience_row(payload)
    if not row:
        base["reason"] = "case_row_missing"
        return base
    case_instance_id = str(row.get("instance_id", "") or payload.get("instance_id", "") or "")
    base["case_instance_id"] = case_instance_id
    base["case_family"] = _instance_family(case_instance_id)
    if not case_instance_id:
        base["reason"] = "case_instance_id_missing"
        return base
    if case_instance_id == target_instance_id:
        base["reason"] = "same_instance_case_rejected_use_baseline_contract_instead"
        return base
    if not bool(row.get("oracle_success", False)):
        base["reason"] = "case_row_not_oracle_success"
        return base
    patch_summary = dict(row.get("patch_summary", {}) or {})
    source_files = _patch_summary_source_files(patch_summary)
    test_files = _patch_summary_test_files(patch_summary)
    patch_legitimacy = str(row.get("patch_legitimacy", "") or _classify_patch_summary(patch_summary)["patch_legitimacy"])
    base.update(
        {
            "case_oracle_success": True,
            "source_files": source_files[:8],
            "test_files": test_files[:8],
            "patch_legitimacy": patch_legitimacy,
        }
    )
    if patch_legitimacy != "source_only":
        base["reason"] = "case_patch_not_strict_source_only"
        return base
    if not source_files:
        base["reason"] = "case_row_has_no_source_diff"
        return base
    diff_excerpt = _previous_source_diff_excerpt(row, source_files)[:3000]
    if not diff_excerpt.strip():
        base["reason"] = "case_diff_excerpt_missing"
        return base
    base["diff_excerpt"] = diff_excerpt
    base["abstracted_experience"] = _abstract_historical_experience(
        row=row,
        case_instance_id=case_instance_id,
        source_files=source_files,
        diff_excerpt=diff_excerpt,
    )
    base["admitted"] = True
    base["reason"] = "admitted_cross_instance_source_only_success_case"
    return base


def _first_historical_experience_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and bool(row.get("oracle_success", False)):
                return row
        for row in rows:
            if isinstance(row, dict):
                return row
    if isinstance(payload, dict):
        return payload
    return {}


def _patch_summary_source_files(patch_summary: dict[str, Any]) -> list[str]:
    direct = [str(path) for path in list(patch_summary.get("non_test_files", []) or []) if str(path).strip()]
    if direct:
        return direct
    for class_key in ("changed_file_classes", "fresh_changed_file_classes"):
        classes = dict(patch_summary.get(class_key, {}) or {})
        source_files = [str(path) for path in list(classes.get("source_files", []) or []) if str(path).strip()]
        if source_files:
            return source_files
    changed = [str(path) for path in list(patch_summary.get("changed_files", []) or []) if str(path).strip()]
    return [path for path in changed if not _looks_like_test_or_runner_path(path)]


def _patch_summary_test_files(patch_summary: dict[str, Any]) -> list[str]:
    direct = [str(path) for path in list(patch_summary.get("test_files", []) or []) if str(path).strip()]
    if direct:
        return direct
    for class_key in ("changed_file_classes", "fresh_changed_file_classes"):
        classes = dict(patch_summary.get(class_key, {}) or {})
        test_files = [str(path) for path in list(classes.get("test_files", []) or []) if str(path).strip()]
        if test_files:
            return test_files
    changed = [str(path) for path in list(patch_summary.get("changed_files", []) or []) if str(path).strip()]
    return [path for path in changed if _looks_like_test_or_runner_path(path)]


def _instance_family(instance_id: str) -> str:
    if "__" in instance_id:
        return instance_id.split("__", 1)[0]
    if "-" in instance_id:
        return instance_id.split("-", 1)[0]
    return instance_id


def _abstract_historical_experience(
    *,
    row: dict[str, Any],
    case_instance_id: str,
    source_files: list[str],
    diff_excerpt: str,
) -> dict[str, Any]:
    signature = dict(row.get("external_validation_failure_signature", {}) or {})
    failure_classes = list(signature.get("exception_classes", []) or [])
    changed_lines = [
        line[:220]
        for line in str(diff_excerpt).splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ][:12]
    return {
        "case_instance_id": case_instance_id,
        "case_family": _instance_family(case_instance_id),
        "source_file_count": len(source_files),
        "source_files": source_files[:8],
        "failure_exception_classes": [str(item) for item in failure_classes[:6]],
        "changed_line_shape": changed_lines,
        "model_call_count": int(row.get("model_call_count", 0) or 0),
        "token_cost": float(row.get("token_cost", 0.0) or 0.0),
        "patch_effect_hint": "Use only the abstract edit pattern and source-boundary signal; do not copy this case patch.",
    }


def _historical_experience_context(admission: dict[str, Any], *, mode: str = "mas_conditioned") -> str:
    if not admission.get("admitted"):
        return ""
    generic_mode = mode == "generic_patch_memory"
    payload = {
        "schema": "masguard.historical_experience_context.v1",
        "mode": mode,
        "case_instance_id": str(admission.get("case_instance_id", "") or ""),
        "case_family": str(admission.get("case_family", "") or ""),
        "source_files": list(admission.get("source_files", []) or [])[:8],
        "abstracted_experience": dict(admission.get("abstracted_experience", {}) or {}),
        "diff_excerpt_for_shape_only": str(admission.get("diff_excerpt", "") or "")[:1800],
        "selection_policy": {
            "use_as_cross_instance_experience_not_answer": True,
            "combine_with_mas_failure_graph_or_minimal_validation_evidence": not generic_mode,
            "generic_patch_memory_baseline_comparator": generic_mode,
            "derive_a_fresh_source_only_patch": True,
            "prefer_shared_failure_mechanism_over_repository_similarity": not generic_mode,
            "fresh_oracle_required_for_credit": True,
        },
        "hard_rules": [
            "This is a different-instance historical success case, not gold truth and not direct recovery credit.",
            "Use it only to infer an abstract repair mechanism, likely source-boundary, or probe/action preference.",
            "Do not copy identifiers, tests, evaluation commands, or repository-specific edits unless independently supported by the current MAS evidence.",
            (
                "For this generic-memory comparator, do not claim MASGuard credit from the memory itself; "
                "the run must still derive a fresh source-only patch and pass the same audit."
                if generic_mode
                else "State how the current MAS graph, handoff/shared-fact evidence, or minimal validation signal matches the historical case before editing."
            ),
            (
                "This comparator intentionally omits the MAS-conditioning gate so it can quantify generic retrieval risk."
                if generic_mode
                else "If the current evidence does not support the historical mechanism, ignore the case and abstain or use the current evidence."
            ),
        ],
    }
    header = (
        "[MASGUARD GENERIC HISTORICAL PATCH MEMORY COMPARATOR]"
        if generic_mode
        else "[MASGUARD HISTORICAL EXPERIENCE]"
    )
    footer = (
        "[/MASGUARD GENERIC HISTORICAL PATCH MEMORY COMPARATOR]"
        if generic_mode
        else "[/MASGUARD HISTORICAL EXPERIENCE]"
    )
    message = (
        "A different-instance source-only recovery success is available as generic patch-memory comparator. "
        "Use it only as an abstract source-diff cue; it is not MAS-conditioned evidence and not MASGuard credit."
        if generic_mode
        else "A different-instance source-only recovery success is available as bounded case memory. Use it only as abstract MAS recovery experience, conditioned on the current failed trajectory evidence."
    )
    return (
        f"{header}\n"
        f"{message}\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        f"{footer}"
    )


def _mas_experience_controller_admission_audit(
    *,
    artifact: Path | None,
    target_instance_id: str,
) -> dict[str, Any]:
    base = {
        "schema": "masguard.mas_experience_controller_admission.v1",
        "enabled": artifact is not None,
        "artifact": str(artifact or ""),
        "admitted": False,
        "reason": "disabled" if artifact is None else "",
        "target_instance_id": target_instance_id,
        "decision_action": "",
        "selected_arm": "",
        "evidence_mode": "",
        "controller_reason": "",
        "mas_memory_case_instance_id": "",
        "mas_memory_case_arm": "",
        "mas_evidence_match_count": 0,
        "generic_only_risk": False,
        "preaction_mas_features": {},
        "uses_oracle_success_as_runtime_feature": False,
        "uses_selected_action_outcome_as_runtime_feature": False,
        "leakage_excluded_fields": [
            "selected_success",
            "fixed_source_only_success",
            "clean_start_success",
            "probe_flat_success",
            "reflexion_k_live_success",
            "baseline_union_success",
            "mas_current_source_credit_success",
            "oracle_success",
        ],
        "claim_boundary": {
            "controller_context_only": True,
            "does_not_grant_recovery_credit": True,
            "uses_preaction_mas_features_only": True,
            "requires_fresh_live_execution_for_credit": True,
            "does_not_copy_historical_patch": True,
        },
    }
    if artifact is None:
        return base
    if not artifact.is_file():
        base["reason"] = "artifact_missing"
        return base
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = "artifact_unreadable"
        base["error"] = str(exc)
        return base
    audit = dict(
        dict(payload.get("current88_matched_execution_matrix", {}) or {}).get(
            "current88_mas_experience_controller_audit", {}
        )
        or {}
    )
    if not audit:
        audit = dict(payload.get("current88_mas_experience_controller_audit", {}) or {})
    summary = dict(audit.get("summary", {}) or {})
    base["uses_oracle_success_as_runtime_feature"] = bool(
        summary.get("uses_oracle_success_as_runtime_feature", False)
    )
    base["uses_selected_action_outcome_as_runtime_feature"] = bool(
        summary.get("uses_selected_action_outcome_as_runtime_feature", False)
    )
    if base["uses_oracle_success_as_runtime_feature"] or base["uses_selected_action_outcome_as_runtime_feature"]:
        base["reason"] = "controller_artifact_uses_outcome_leakage"
        return base
    rows = list(audit.get("rows", []) or [])
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and str(item.get("instance_id", "") or "") == target_instance_id
        ),
        {},
    )
    if not row:
        base["reason"] = "target_instance_not_in_controller_rows"
        return base
    decision_action = str(row.get("decision_action", "") or "")
    base.update(
        {
            "admitted": bool(decision_action),
            "reason": "admitted_preaction_mas_controller_row" if decision_action else "decision_action_missing",
            "decision_action": decision_action,
            "selected_arm": str(row.get("selected_arm", "") or ""),
            "evidence_mode": str(row.get("evidence_mode", "") or ""),
            "controller_reason": str(row.get("reason", "") or ""),
            "mas_memory_case_instance_id": str(row.get("mas_memory_case_instance_id", "") or ""),
            "mas_memory_case_arm": str(row.get("mas_memory_case_arm", "") or ""),
            "mas_evidence_match_count": int(row.get("mas_evidence_match_count", 0) or 0),
            "generic_only_risk": bool(row.get("generic_only_risk", False)),
            "preaction_mas_features": {
                str(key): str(value)
                for key, value in dict(row.get("preaction_mas_features", {}) or {}).items()
            },
        }
    )
    return base


def _mas_experience_controller_context(admission: dict[str, Any]) -> str:
    if not admission.get("admitted"):
        return ""
    payload = {
        "schema": "masguard.mas_experience_controller_context.v1",
        "target_instance_id": str(admission.get("target_instance_id", "") or ""),
        "decision_action": str(admission.get("decision_action", "") or ""),
        "selected_arm": str(admission.get("selected_arm", "") or ""),
        "evidence_mode": str(admission.get("evidence_mode", "") or ""),
        "controller_reason": str(admission.get("controller_reason", "") or ""),
        "mas_memory": {
            "case_instance_id": str(admission.get("mas_memory_case_instance_id", "") or ""),
            "case_arm": str(admission.get("mas_memory_case_arm", "") or ""),
            "mas_evidence_match_count": int(admission.get("mas_evidence_match_count", 0) or 0),
        },
        "generic_only_risk": bool(admission.get("generic_only_risk", False)),
        "preaction_mas_features": dict(admission.get("preaction_mas_features", {}) or {}),
        "hard_rules": [
            "Use this controller row only as a pre-action MAS control signal.",
            "Do not use any baseline, selected-action, or oracle outcome field as runtime evidence.",
            "If decision_action is probe, run or respect a bounded minimal validation step before a mutating edit.",
            "If decision_action is source-action, derive a fresh source-only edit from the current issue and MAS evidence.",
            "If decision_action is fallback, avoid generic historical memory and use the conservative fallback path.",
            "If current evidence conflicts with the controller row, abstain rather than widening the patch.",
        ],
    }
    return (
        "[MASGUARD MAS EXPERIENCE CONTROLLER]\n"
        "A frozen MAS experience controller selected a pre-action control decision for this row. "
        "It is not recovery credit and it excludes outcome leakage; use it to choose the recovery path, "
        "minimal probe need, and source-edit constraints.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD MAS EXPERIENCE CONTROLLER]"
    )


def _historical_action_program_admission_audit(
    *,
    artifact: Path | None,
    target_instance_id: str,
) -> dict[str, Any]:
    base = {
        "schema": "masguard.historical_action_program_admission.v1",
        "enabled": artifact is not None,
        "artifact": str(artifact or ""),
        "admitted": False,
        "reason": "disabled" if artifact is None else "",
        "target_instance_id": target_instance_id,
        "program_id": "",
        "decision_action": "",
        "execution_mode": "",
        "selected_history_case_ids": [],
        "mas_failure_category": "",
        "target_file_patterns": [],
        "edit_invariants": [],
        "forbidden_edit_patterns": [],
        "validation_templates": [],
        "early_stop_rules": [],
        "patch_script_present": False,
        "patch_script_sha256": "",
        "patch_script_char_count": 0,
        "uses_oracle_success_as_runtime_feature": False,
        "uses_selected_action_outcome_as_runtime_feature": False,
        "claim_boundary": {
            "cross_instance_action_program_only": True,
            "does_not_copy_historical_patch": True,
            "does_not_grant_recovery_credit": True,
            "requires_fresh_live_execution_for_credit": True,
            "uses_preaction_mas_features_only": True,
            "may_force_abstention_or_validation_but_not_success": True,
        },
    }
    if artifact is None:
        return base
    if not artifact.is_file():
        base["reason"] = "artifact_missing"
        return base
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["reason"] = "artifact_unreadable"
        base["error"] = str(exc)
        return base
    audit = dict(
        dict(payload.get("current88_matched_execution_matrix", {}) or {}).get(
            "current88_mas_experience_action_program_audit", {}
        )
        or {}
    )
    if not audit:
        audit = dict(payload.get("mas_experience_action_program_audit", {}) or {})
    summary = dict(audit.get("summary", {}) or {})
    base["uses_oracle_success_as_runtime_feature"] = bool(
        summary.get("uses_oracle_success_as_runtime_feature", False)
    )
    base["uses_selected_action_outcome_as_runtime_feature"] = bool(
        summary.get("uses_selected_action_outcome_as_runtime_feature", False)
    )
    if base["uses_oracle_success_as_runtime_feature"] or base["uses_selected_action_outcome_as_runtime_feature"]:
        base["reason"] = "action_program_artifact_uses_outcome_leakage"
        return base
    rows = list(audit.get("rows", []) or payload.get("action_program_rows", []) or [])
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and str(item.get("instance_id", "") or "") == target_instance_id
        ),
        {},
    )
    if not row:
        base["reason"] = "target_instance_not_in_action_program_rows"
        return base
    program = dict(row.get("action_program", {}) or {})
    target_file_patterns = [str(item) for item in list(program.get("target_file_patterns", []) or []) if str(item)]
    edit_invariants = [str(item) for item in list(program.get("edit_invariants", []) or []) if str(item)]
    validation_templates = [str(item) for item in list(program.get("validation_templates", []) or []) if str(item)]
    early_stop_rules = [str(item) for item in list(program.get("early_stop_rules", []) or []) if str(item)]
    decision_action = str(row.get("decision_action", "") or "")
    execution_mode = str(program.get("execution_mode", "") or row.get("execution_mode", "") or "prompt_context")
    patch_script = str(program.get("patch_script", "") or program.get("patch_script_skeleton", "") or "")
    if decision_action not in {"source-action", "probe", "fallback", "abstain"}:
        base["reason"] = "unsupported_decision_action"
        return base
    if execution_mode not in {"prompt_context", "direct_script"}:
        base["reason"] = "unsupported_execution_mode"
        return base
    if decision_action == "source-action" and not target_file_patterns:
        base["reason"] = "source_action_missing_target_file_patterns"
        return base
    if decision_action == "source-action" and not edit_invariants:
        base["reason"] = "source_action_missing_edit_invariants"
        return base
    if execution_mode == "direct_script" and decision_action != "source-action":
        base["reason"] = "direct_script_requires_source_action"
        return base
    if execution_mode == "direct_script" and not patch_script.strip():
        base["reason"] = "direct_script_missing_patch_script"
        return base
    if not validation_templates and not early_stop_rules:
        base["reason"] = "missing_validation_or_early_stop_rules"
        return base
    base.update(
        {
            "admitted": True,
            "reason": "admitted_preaction_historical_action_program",
            "program_id": str(row.get("program_id", "") or program.get("program_id", "") or ""),
            "decision_action": decision_action,
            "execution_mode": execution_mode,
            "selected_history_case_ids": [
                str(item)
                for item in list(row.get("selected_history_case_ids", []) or [])
                if str(item)
            ][:8],
            "mas_failure_category": str(row.get("mas_failure_category", "") or ""),
            "target_file_patterns": target_file_patterns[:8],
            "edit_invariants": edit_invariants[:12],
            "forbidden_edit_patterns": [
                str(item)
                for item in list(program.get("forbidden_edit_patterns", []) or [])
                if str(item)
            ][:12],
            "validation_templates": validation_templates[:8],
            "early_stop_rules": early_stop_rules[:12],
            "patch_script": patch_script if execution_mode == "direct_script" else "",
            "patch_script_present": bool(patch_script.strip()),
            "patch_script_sha256": (
                hashlib.sha256(patch_script.encode("utf-8")).hexdigest()
                if patch_script.strip()
                else ""
            ),
            "patch_script_char_count": len(patch_script),
        }
    )
    return base


def _historical_action_program_context(admission: dict[str, Any]) -> str:
    if not admission.get("admitted"):
        return ""
    payload = {
        "schema": "masguard.historical_action_program_context.v1",
        "target_instance_id": str(admission.get("target_instance_id", "") or ""),
        "program_id": str(admission.get("program_id", "") or ""),
        "decision_action": str(admission.get("decision_action", "") or ""),
        "execution_mode": str(admission.get("execution_mode", "") or "prompt_context"),
        "selected_history_case_ids": list(admission.get("selected_history_case_ids", []) or [])[:8],
        "mas_failure_category": str(admission.get("mas_failure_category", "") or ""),
        "action_program": {
            "target_file_patterns": list(admission.get("target_file_patterns", []) or [])[:8],
            "edit_invariants": list(admission.get("edit_invariants", []) or [])[:12],
            "forbidden_edit_patterns": list(admission.get("forbidden_edit_patterns", []) or [])[:12],
            "validation_templates": list(admission.get("validation_templates", []) or [])[:8],
            "early_stop_rules": list(admission.get("early_stop_rules", []) or [])[:12],
            "patch_script_sha256": str(admission.get("patch_script_sha256", "") or ""),
            "patch_script_char_count": int(admission.get("patch_script_char_count", 0) or 0),
        },
        "hard_rules": [
            "Use this as a cross-instance historical action program, not a patch to copy.",
            "Before editing, print one MASGUARD_DIFF_INTENT JSON object that names the target file, edit kind, expected changed symbol, and validation command.",
            "If the current MAS evidence or source anchor conflicts with the action program, print MASGUARD_STRICT_ABSTAIN_NO_EDIT and a concrete MASGUARD_ABSTAIN_REASON instead of widening the patch.",
            "Do not edit tests, generated files, evaluation commands, or repository setup.",
            "Run the listed validation template when feasible; otherwise run a focused syntax check on the edited source file.",
            "Fresh oracle execution is still required for credit; this program only controls action selection, validation, and abstention.",
        ],
    }
    return (
        "[MASGUARD HISTORICAL ACTION PROGRAM]\n"
        "A frozen no-leakage historical-learning controller selected an executable pre-action recovery program. "
        "It distills repeated MAS failure categories into target-file patterns, edit invariants, validation templates, "
        "and early-stop rules; it is not recovery credit and it is not a historical patch answer.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD HISTORICAL ACTION PROGRAM]"
    )


def _empty_historical_action_program_direct_script_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.historical_action_program_direct_script.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "executed": False,
        "returncode": None,
        "output_preview": "",
        "patch_script_sha256": "",
        "patch_script_char_count": 0,
        "patch_summary": {},
        "source_edit_contract_audit": {},
        "candidate_gate": {},
        "final_candidate_admitted": False,
        "reason": "disabled" if not enabled else "not_attempted",
        "claim_boundary": {
            "historical_action_program_execution_only": True,
            "does_not_call_models": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit_without_fresh_oracle": True,
            "pre_oracle_source_candidate_gate": True,
            "fail_closed_on_script_error_or_invalid_candidate": True,
        },
    }


def _historical_action_program_direct_script_audit(
    *,
    enabled: bool,
    admission: dict[str, Any],
    executor,
    workspace: Path,
    timeout: int,
    manifest: dict[str, Any],
    source_edit_contract: str,
    patch_contract: str,
    semantic_patch_correctness_v2: bool = False,
    semantic_effect_cues: list[object] | None = None,
    target_source_candidates: list[object] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = _empty_historical_action_program_direct_script_audit(enabled=enabled)
    empty_stage = {"success": False, "direct_script": bool(enabled), "commands": []}
    if not enabled:
        return audit, empty_stage
    audit["attempted"] = True
    if str(admission.get("execution_mode", "") or "") != "direct_script":
        audit["reason"] = "execution_mode_not_direct_script"
        return audit, empty_stage
    script = str(admission.get("patch_script", "") or "")
    if not script.strip():
        audit["reason"] = "missing_patch_script"
        return audit, empty_stage
    script_hash = str(admission.get("patch_script_sha256", "") or "") or hashlib.sha256(
        script.encode("utf-8")
    ).hexdigest()
    result = executor.execute(script, cwd=str(workspace), timeout=min(timeout, 600))
    output = str(result.get("output", "") or "")
    returncode = result.get("returncode")
    stage_output = {
        "success": returncode == 0,
        "direct_script": True,
        "commands": [
            {
                "command": script,
                "output": output,
                "returncode": returncode,
                "historical_action_program_direct_script": True,
                "patch_script_sha256": script_hash,
            }
        ],
    }
    patch_summary = workspace_patch_summary(workspace, list(manifest.get("oracle_patch_files", []) or []))
    source_audit = _source_edit_contract_audit(
        source_edit_contract=source_edit_contract,
        patch_summary=patch_summary,
        stage_outputs={"patcher": stage_output},
        workspace=workspace,
    )
    candidate_gate = _source_edit_pre_oracle_candidate_gate(
        workspace=workspace,
        patch_summary=patch_summary,
        source_edit_contract_audit=source_audit,
        patch_contract=patch_contract,
        stage_outputs={"patcher": stage_output},
        semantic_patch_correctness_v2=semantic_patch_correctness_v2,
        semantic_effect_cues=list(semantic_effect_cues or []),
        target_source_candidates=list(target_source_candidates or []),
    )
    changed_source_files = [
        str(path)
        for path in list(patch_summary.get("non_test_files", []) or [])
        if str(path).strip()
    ]
    final_admitted = bool(returncode == 0 and changed_source_files and candidate_gate.get("admitted", False))
    reason = "direct_script_candidate_admitted"
    if returncode != 0:
        reason = "direct_script_returncode_nonzero"
    elif not changed_source_files:
        reason = "direct_script_no_source_diff"
    elif not candidate_gate.get("admitted", False):
        reason = "direct_script_candidate_gate_rejected"
    audit.update(
        {
            "executed": True,
            "returncode": returncode,
            "output_preview": output[:2000],
            "patch_script_sha256": script_hash,
            "patch_script_char_count": int(admission.get("patch_script_char_count", 0) or len(script)),
            "patch_summary": patch_summary,
            "source_edit_contract_audit": source_audit,
            "candidate_gate": candidate_gate,
            "final_candidate_admitted": final_admitted,
            "reason": reason,
        }
    )
    return audit, stage_output


def _historical_action_program_direct_script_blocks_final_evaluation(audit: dict[str, Any]) -> bool:
    if not bool(audit.get("enabled", False)) or not bool(audit.get("attempted", False)):
        return False
    return not bool(audit.get("final_candidate_admitted", False))


def _row_for_instance(payload: dict[str, Any], instance_id: str) -> dict[str, Any]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("instance_id", "") or "") == instance_id:
                return row
        for row in rows:
            if isinstance(row, dict):
                return row
    if isinstance(payload, dict):
        return payload
    return {}


def _external_validation_signature_from_row(row: dict[str, Any]) -> dict[str, object]:
    existing = row.get("external_validation_failure_signature")
    if isinstance(existing, dict):
        return _merge_validation_failure_signatures([existing])
    signatures: list[dict[str, object]] = []
    for prefix in ("fail_to_pass", "oracle"):
        returncode = row.get(f"{prefix}_returncode")
        if returncode in (None, 0):
            continue
        signatures.append(
            validation_failure_signature(
                command=str(row.get(f"{prefix}_command", "") or ""),
                output=str(row.get(f"{prefix}_output", "") or ""),
            )
        )
    return _merge_validation_failure_signatures(signatures)


def _external_validation_failure_sources_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    full_outputs: dict[str, str] = {}
    for prefix in ("fail_to_pass", "oracle"):
        returncode = row.get(f"{prefix}_returncode")
        if returncode in (None, 0):
            continue
        full_outputs[prefix] = str(row.get(f"{prefix}_output", "") or "")
    existing = row.get("external_validation_failure_sources")
    if isinstance(existing, list) and existing:
        enriched: list[dict[str, Any]] = []
        for item in existing:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "") or "")
            copied = dict(item)
            full_output = full_outputs.get(source, "")
            if full_output:
                copied["output_excerpt"] = full_output[:3000]
            enriched.append(copied)
        return enriched
    sources: list[dict[str, Any]] = []
    for prefix in ("fail_to_pass", "oracle"):
        returncode = row.get(f"{prefix}_returncode")
        if returncode in (None, 0):
            continue
        sources.append(
            {
                "source": prefix,
                "returncode": returncode,
                "command": str(row.get(f"{prefix}_command", "") or ""),
                "output_excerpt": str(row.get(f"{prefix}_output", "") or "")[:3000],
            }
        )
    return sources


def _external_validation_feedback_context(admission: dict[str, Any]) -> str:
    if not admission.get("admitted"):
        return ""
    signature = dict(admission.get("signature", {}) or {})
    failure_classification = dict(admission.get("failure_classification", {}) or {})
    class_specific_contract = _failure_class_edit_contract(failure_classification, signature=signature)
    guarded_fallback_plan = _guarded_fallback_repair_plan_from_signature(
        signature=signature,
        failure_sources=list(admission.get("failure_sources", []) or []),
    )
    payload = {
        "schema": "masguard.external_validation_feedback_context.v2",
        "previous_source_files": list(admission.get("previous_source_files", []) or []),
        "previous_diff_excerpt": str(admission.get("previous_diff_excerpt", "") or "")[:2000],
        "minimal_validation_failure_classification": {
            "failure_class": str(failure_classification.get("failure_class", "") or "unknown"),
            "retry_action": str(failure_classification.get("retry_action", "") or "abstain"),
            "confidence": str(failure_classification.get("confidence", "") or "low"),
            "prompt_guidance": str(failure_classification.get("prompt_guidance", "") or "")[:500],
            "reasons": list(failure_classification.get("reasons", []) or [])[:6],
            "semantic_effect_cues": list(failure_classification.get("semantic_effect_cues", []) or [])[:8],
        },
        "class_specific_edit_contract": class_specific_contract,
        "failure_sources": [
            {
                "source": item.get("source", ""),
                "returncode": item.get("returncode"),
                "command": str(item.get("command", "") or "")[:300],
                "output_excerpt": str(item.get("output_excerpt", "") or "")[:2000],
            }
            for item in list(admission.get("failure_sources", []) or [])[:2]
            if isinstance(item, dict)
        ],
        "signature": {
            "validation_tool_missing": bool(signature.get("validation_tool_missing", False)),
            "exception_classes": list(signature.get("exception_classes", []) or []),
            "failing_tests": list(signature.get("failing_tests", []) or []),
            "traceback_source_files": list(signature.get("traceback_source_files", []) or []),
        },
        "guarded_fallback_repair_plan": guarded_fallback_plan,
        "hard_rules": [
            "Treat the previous source diff as insufficient, not as a solution to repeat.",
            "Do not reproduce the same previous_diff_excerpt unless the new validation evidence justifies it.",
            "Revise the semantic repair hypothesis to explain the exception class, failing test, and traceback source evidence.",
            "If guarded_fallback_repair_plan.enabled is true, use one listed narrow_exception_candidate or explicit_guard_candidate before any broad fallback.",
            "When semantic_effect_cues are present, the revised patch must make the edited source span produce the expected value or boundary named in those cues.",
            "Follow minimal_validation_failure_classification.retry_action and prompt_guidance when they are present.",
            "Follow class_specific_edit_contract.required_edit_behavior and class_specific_edit_contract.forbidden_behavior.",
            "Keep the patch source-only and minimal; do not edit tests, generated files, or evaluation commands.",
            "If evidence is insufficient for a safe source edit, abstain rather than widening behavior.",
            *list(class_specific_contract.get("hard_rules", []) or []),
        ],
    }
    return (
        "[MASGUARD EXTERNAL VALIDATION FEEDBACK]\n"
        "A previous source-only repair attempt failed external fail-to-pass/oracle validation. "
        "Use this bounded validation evidence and the previous diff excerpt to revise the repair mechanism; "
        "do not repeat the same edit.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD EXTERNAL VALIDATION FEEDBACK]"
    )


def _guarded_fallback_repair_plan_from_signature(
    *,
    signature: dict[str, Any],
    failure_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    exception_candidates = _guarded_fallback_narrow_exception_candidates(signature)
    explicit_guard_candidates = _guarded_fallback_explicit_guard_candidates(
        signature=signature,
        failure_sources=list(failure_sources or []),
    )
    enabled = bool(exception_candidates or explicit_guard_candidates)
    return {
        "schema": "masguard.guarded_fallback_repair_plan.v1",
        "enabled": enabled,
        "narrow_exception_candidates": exception_candidates[:6],
        "explicit_guard_candidates": explicit_guard_candidates[:6],
        "required_behavior": (
            "Choose exactly one narrow exception candidate or explicit guard candidate before any compatibility "
            "fallback. If none is applicable to the edited boundary, abstain instead of using broad except Exception."
            if enabled
            else "No signature-grounded narrow exception or explicit guard candidate was found; abstain from guarded fallback unless the source evidence supplies one."
        ),
        "forbidden_behavior": [
            "broad except Exception as the first guard",
            "catch-all except without a preceding narrow exception or explicit condition",
            "silent return of a safe default before attempting the compatibility fallback",
        ],
        "claim_boundary": {
            "provider_free_signature_to_control_signal": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _guarded_fallback_narrow_exception_candidates(signature: dict[str, Any]) -> list[str]:
    broad_names = {"Exception", "BaseException", "AssertionError"}
    candidates: list[str] = []
    for raw in list(signature.get("exception_classes", []) or []):
        value = str(raw).strip()
        if not value:
            continue
        simple = value.rsplit(".", 1)[-1]
        if simple in broad_names or value in broad_names:
            continue
        if simple.endswith(("Error", "Exception")) or "." in value:
            candidates.append(value)
    return _dedupe_preserve_order(candidates)


def _guarded_fallback_explicit_guard_candidates(
    *,
    signature: dict[str, Any],
    failure_sources: list[dict[str, Any]],
) -> list[str]:
    text = "\n".join(
        [
            " ".join(str(item) for item in list(signature.get("exception_classes", []) or [])),
            " ".join(str(item) for item in list(signature.get("traceback_source_files", []) or [])),
            *[
                str(item.get("output_excerpt", "") or "")
                for item in list(failure_sources or [])
                if isinstance(item, dict)
            ],
        ]
    ).lower()
    candidates: list[str] = []
    if any(marker in text for marker in ("incorrect padding", "base64", "binascii")):
        candidates.append("guard encoded/base64 session payload before legacy decode")
    if any(marker in text for marker in ("bad signature", "badsignature", "signature")):
        candidates.append("guard signed payload failure before legacy compatibility fallback")
    if any(marker in text for marker in ("jsondecodeerror", "json decode", "invalid json")):
        candidates.append("guard invalid JSON payload before compatibility fallback")
    if any(marker in text for marker in ("unicode", "decodeerror", "encoding")):
        candidates.append("guard text decoding failure before compatibility fallback")
    if any(marker in text for marker in ("keyerror", "missing key")):
        candidates.append("guard missing key before compatibility fallback")
    if any(marker in text for marker in ("valueerror", "invalid literal", "invalid value")):
        candidates.append("guard invalid value before compatibility fallback")
    return _dedupe_preserve_order(candidates)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _empty_diverse_repair_hypothesis_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.diverse_repair_hypothesis_retry.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "context_attached": False,
        "failure_class": "unknown",
        "hypothesis_count": 0,
        "hypothesis_families": [],
        "selection_policy": "disabled" if not enabled else "not_attempted",
        "uses_oracle_success": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit_without_fresh_execution": True,
            "bounded_to_single_selected_candidate": True,
        },
    }


def _diverse_repair_hypothesis_context(
    *,
    failure_classification: dict[str, Any],
    signature: dict[str, Any],
    previous_source_files: list[str],
    previous_diff_excerpt: str,
    workspace: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    failure_class = str(failure_classification.get("failure_class", "") or "unknown")
    retry_action = str(failure_classification.get("retry_action", "") or "abstain")
    contract = _failure_class_edit_contract(failure_classification, signature=signature)
    hypotheses = _diverse_repair_hypotheses_for_class(
        failure_class=failure_class,
        retry_action=retry_action,
    )
    payload = {
        "schema": "masguard.diverse_repair_hypothesis_context.v1",
        "failure_class": failure_class,
        "retry_action": retry_action,
        "target_source_candidates": list(contract.get("target_source_candidates", []) or []),
        "target_source_candidate_summaries": _target_source_candidate_summaries(
            workspace=workspace,
            target_source_candidates=list(contract.get("target_source_candidates", []) or []),
        ),
        "previous_source_files": list(previous_source_files or [])[:8],
        "previous_diff_excerpt": str(previous_diff_excerpt or "")[:1800],
        "semantic_effect_cues": list(failure_classification.get("semantic_effect_cues", []) or [])[:8],
        "hypotheses": hypotheses,
        "selection_policy": {
            "select_exactly_one_hypothesis_before_editing": True,
            "anchor_selected_hypothesis_to_target_source_summary": bool(
                workspace and list(contract.get("target_source_candidates", []) or [])
            ),
            "prefer_smallest_source_only_patch_that_satisfies_contract": True,
            "do_not_blend_unrelated_hypotheses": True,
            "abstain_if_no_hypothesis_is_supported": True,
        },
        "class_specific_edit_contract": contract,
        "hard_rules": [
            "Propose at least two distinct repair mechanisms internally before editing.",
            "Choose exactly one mechanism and make one minimal source-only patch for it.",
            "Use semantic_effect_cues as the required patch effect when they are present.",
            "Do not repeat the previous failed diff unless the selected hypothesis explains why it was incomplete.",
            "Do not edit tests, generated files, evaluation commands, or broad runtime setup.",
            *list(contract.get("hard_rules", []) or []),
        ],
    }
    context = (
        "[MASGUARD DIVERSE REPAIR HYPOTHESES]\n"
        "Before producing a replacement patch, compare the listed mutually distinct repair mechanisms. "
        "Select exactly one supported hypothesis, implement the minimal source-only patch for it, and abstain "
        "if none is supported by the bounded validation evidence.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD DIVERSE REPAIR HYPOTHESES]"
    )
    audit = {
        **_empty_diverse_repair_hypothesis_audit(enabled=True),
        "attempted": True,
        "context_attached": True,
        "failure_class": failure_class,
        "retry_action": retry_action,
        "hypothesis_count": len(hypotheses),
        "hypothesis_families": [str(item.get("family", "") or "") for item in hypotheses],
        "target_source_candidate_count": len(list(contract.get("target_source_candidates", []) or [])),
        "target_source_candidate_summary_count": len(
            _target_source_candidate_summaries(
                workspace=workspace,
                target_source_candidates=list(contract.get("target_source_candidates", []) or []),
            )
        ),
        "selection_policy": "select_one_supported_hypothesis_then_contract_gate",
    }
    return context, audit


def _diverse_repair_hypotheses_for_class(*, failure_class: str, retry_action: str) -> list[dict[str, str]]:
    if failure_class == "missing_condition":
        return [
            {
                "family": "narrow_guard_or_boundary_condition",
                "intent": "Add the smallest guard, boundary check, or branch that explains the failing test.",
            },
            {
                "family": "input_normalization_or_compatibility_condition",
                "intent": "Normalize or special-case the failing input shape before the existing mechanism runs.",
            },
        ]
    if failure_class == "wrong_abstraction":
        return [
            {
                "family": "replace_wrong_api_or_semantic_boundary",
                "intent": "Move the fix to the API boundary or abstraction layer responsible for the exception.",
            },
            {
                "family": "change_internal_representation_mechanism",
                "intent": "Revise the representation/conversion mechanism rather than masking the symptom locally.",
            },
        ]
    if failure_class == "source_location":
        return [
            {
                "family": "retarget_traceback_source_span",
                "intent": "Move the patch to the traceback or suspect source span.",
            },
            {
                "family": "retarget_call_boundary",
                "intent": "Patch the caller/callee boundary that feeds the failing source span.",
            },
        ]
    if failure_class == "no_diff":
        return [
            {
                "family": "source_span_reacquisition",
                "intent": "Reacquire a concrete production source span and edit it minimally.",
            },
            {
                "family": "abstain_if_no_supported_source_span",
                "intent": "Avoid a cosmetic or empty retry when no defensible source target exists.",
            },
        ]
    return [
        {
            "family": "minimal_supported_source_edit",
            "intent": f"Apply only a directly supported source edit for retry_action={retry_action or 'abstain'}.",
        },
        {
            "family": "fail_closed_abstention",
            "intent": "Abstain when the bounded validation evidence does not identify a safe source mechanism.",
        },
    ]


def _failure_class_edit_contract(
    classification: dict[str, Any],
    *,
    signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure_class = str(classification.get("failure_class", "") or "unknown")
    retry_action = str(classification.get("retry_action", "") or "abstain")
    target_source_candidates = list(
        dict.fromkeys(
            [
                str(path)
                for path in (
                    list(classification.get("production_traceback_targets", []) or [])
                    + list(classification.get("suspect_paths", []) or [])
                    + _production_traceback_target_suffixes(
                        list((signature or {}).get("traceback_source_files", []) or [])
                    )
                    + list(classification.get("changed_files", []) or [])
                )
                if str(path).strip()
            ]
        )
    )[:8]
    base = {
        "schema": "masguard.failure_class_specific_edit_contract.v1",
        "failure_class": failure_class,
        "retry_action": retry_action,
        "target_source_candidates": target_source_candidates,
        "must_change_source": False,
        "must_retarget_source": False,
        "must_add_narrow_condition": False,
        "must_revise_semantic_mechanism": False,
        "required_edit_behavior": "",
        "forbidden_behavior": [
            "editing tests or evaluation commands",
            "broad exception swallowing",
            "repeating the previous failed diff without new source evidence",
        ],
        "hard_rules": [],
        "claim_boundary": {
            "does_not_use_oracle_success": True,
            "pre_oracle_edit_constraint_only": True,
            "requires_execution_audit_for_credit": True,
        },
    }
    if failure_class == "no_diff":
        base.update(
            {
                "must_change_source": True,
                "must_retarget_source": True,
                "required_edit_behavior": (
                    "The previous attempt produced no source diff. Reacquire a source span from the target "
                    "candidates and submit exactly one minimal production-source edit, or abstain if no target "
                    "candidate is defensible."
                ),
                "hard_rules": [
                    "A second no-diff patch is not admissible for this branch.",
                    "Touch one target_source_candidate when candidates exist.",
                ],
            }
        )
    elif failure_class == "source_location":
        base.update(
            {
                "must_change_source": True,
                "must_retarget_source": True,
                "required_edit_behavior": (
                    "The previous diff missed the failing source location. Retarget the patch to a production "
                    "traceback or suspect source candidate before changing surrounding behavior."
                ),
                "hard_rules": [
                    "Do not keep editing only the previous missed source span.",
                    "The revised diff must cover one target_source_candidate when candidates exist.",
                ],
            }
        )
    elif failure_class == "missing_condition":
        base.update(
            {
                "must_change_source": True,
                "must_add_narrow_condition": True,
                "required_edit_behavior": (
                    "The validation signal is condition or edge-case oriented. Add the narrow missing guard, "
                    "branch, normalization, or boundary check needed by the failing test."
                ),
                "forbidden_behavior": base["forbidden_behavior"] + [
                    "large rewrites unrelated to the failing condition",
                    "global behavior changes without a failing-test condition",
                ],
                "hard_rules": [
                    "Prefer the smallest condition that explains the failure output.",
                    "Do not rewrite unrelated abstractions when a narrow condition is sufficient.",
                ],
            }
        )
    elif failure_class == "wrong_abstraction":
        base.update(
            {
                "must_change_source": True,
                "must_revise_semantic_mechanism": True,
                "required_edit_behavior": (
                    "The previous edit likely used the wrong API boundary or semantic mechanism. Replace the "
                    "mechanism with one that directly explains the exception class, failing test, and traceback."
                ),
                "forbidden_behavior": base["forbidden_behavior"] + [
                    "local symptom patching that leaves the exception mechanism unexplained",
                    "type coercions or fallbacks that only mask the observed exception",
                ],
                "hard_rules": [
                    "State and implement the semantic mechanism change, not only a local guard.",
                    "Keep the mechanism change within the source targets unless evidence justifies retargeting.",
                ],
            }
        )
    elif failure_class == "environment_or_protocol":
        base.update(
            {
                "required_edit_behavior": (
                    "This is environment/protocol evidence. Do not perform a source-repair retry under this branch."
                ),
                "forbidden_behavior": base["forbidden_behavior"] + ["source edits based only on environment setup failure"],
                "hard_rules": ["Abstain from source repair when the only signal is protocol or environment failure."],
            }
        )
    else:
        base.update(
            {
                "required_edit_behavior": (
                    "Evidence is weak. A retry is allowed only if a minimal source edit is directly supported by "
                    "the failure output; otherwise abstain."
                ),
                "hard_rules": ["Do not widen behavior when the failure class is unknown."],
            }
        )
    return base


def _empty_contract_verified_candidate_retry_audit(
    *, enabled: bool, candidate_pre_admission_syntax_guard: bool = False
) -> dict[str, Any]:
    return {
        "schema": "masguard.contract_verified_candidate_retry.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "first_candidate_admitted": False,
        "candidate_pre_admission_syntax_guard_enabled": bool(candidate_pre_admission_syntax_guard),
        "first_candidate_pre_admission_syntax_guard_audit": _empty_candidate_pre_admission_syntax_guard_audit(
            enabled=bool(candidate_pre_admission_syntax_guard)
        ),
        "retry_allowed": False,
        "second_candidate_executed": False,
        "second_candidate_pre_admission_syntax_guard_audit": _empty_candidate_pre_admission_syntax_guard_audit(
            enabled=bool(candidate_pre_admission_syntax_guard)
        ),
        "final_candidate_admitted": False,
        "reason": "disabled" if not enabled else "not_attempted",
        "rejection_context_attached": False,
        "agent_reported_success_after_second_candidate": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit_without_fresh_execution": True,
            "reject_option_before_extra_model_call": True,
        },
    }


def _contract_verified_candidate_retry_audit(
    *,
    enabled: bool,
    workspace: Path,
    patch_summary: dict[str, Any],
    source_edit_contract_audit: dict[str, Any],
    failure_classification: dict[str, Any],
    signature: dict[str, Any],
    candidate_pre_admission_syntax_guard: bool = False,
) -> dict[str, Any]:
    audit = _empty_contract_verified_candidate_retry_audit(
        enabled=enabled,
        candidate_pre_admission_syntax_guard=candidate_pre_admission_syntax_guard,
    )
    if not enabled:
        return audit
    contract = _failure_class_edit_contract(failure_classification, signature=signature)
    class_contract_audit = _failure_class_contract_candidate_audit(
        contract=contract,
        patch_summary=patch_summary,
        source_edit_contract_audit=source_edit_contract_audit,
        workspace=workspace,
    )
    syntax_guard_audit = _candidate_pre_admission_syntax_guard_audit(
        enabled=candidate_pre_admission_syntax_guard,
        workspace=workspace,
        patch_summary=patch_summary,
    )
    first_admitted = bool(
        source_edit_contract_audit.get("satisfied", True)
        and class_contract_audit.get("satisfied", False)
        and syntax_guard_audit.get("satisfied", True)
    )
    first_rejection_reason = (
        "first_candidate_syntax_guard_rejected"
        if not syntax_guard_audit.get("satisfied", True)
        else "first_candidate_contract_rejected"
    )
    retryable_class = str(contract.get("failure_class", "") or "") in {
        "no_diff",
        "source_location",
        "missing_condition",
        "wrong_abstraction",
    }
    retry_allowed = bool(not first_admitted and retryable_class)
    audit.update(
        {
            "attempted": True,
            "class_specific_edit_contract": contract,
            "candidate_pre_admission_syntax_guard_enabled": bool(candidate_pre_admission_syntax_guard),
            "first_candidate_patch_summary": patch_summary,
            "first_candidate_source_edit_contract_audit": source_edit_contract_audit,
            "first_candidate_class_contract_audit": class_contract_audit,
            "first_candidate_pre_admission_syntax_guard_audit": syntax_guard_audit,
            "first_candidate_admitted": first_admitted,
            "retry_allowed": retry_allowed,
            "final_candidate_admitted": first_admitted,
            "reason": (
                "first_candidate_contract_admitted"
                if first_admitted
                else (
                    f"{first_rejection_reason}_second_candidate_allowed"
                    if retry_allowed
                    else f"{first_rejection_reason}_fail_closed"
                )
            ),
            "final_reason": "first_candidate_contract_admitted" if first_admitted else "no_second_candidate_yet",
        }
    )
    return audit


def _empty_candidate_pre_admission_syntax_guard_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.candidate_pre_admission_syntax_guard.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "satisfied": True,
        "reason": "disabled" if not enabled else "not_attempted",
        "target_files": [],
        "python_source_files": [],
        "compiled_file_count": 0,
        "syntax_error_count": 0,
        "missing_file_count": 0,
        "violations": [],
        "syntax_errors": [],
        "claim_boundary": {
            "does_not_call_models": True,
            "syntax_only_not_oracle": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _candidate_pre_admission_syntax_guard_audit(
    *,
    enabled: bool,
    workspace: Path,
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    audit = _empty_candidate_pre_admission_syntax_guard_audit(enabled=enabled)
    if not enabled:
        return audit
    target_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("non_test_files", []) or [])
        if _normalize_repo_path(path)
    ]
    python_files = [path for path in target_files if path.endswith(".py")]
    violations: list[str] = []
    syntax_errors: list[dict[str, Any]] = []
    missing_count = 0
    compiled_count = 0
    for rel_path in python_files:
        abs_path = workspace / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            missing_count += 1
            violations.append("candidate_syntax_guard_source_file_missing")
            syntax_errors.append(
                {
                    "path": rel_path,
                    "error_type": "missing_file",
                    "message": "candidate source file is missing before admission",
                }
            )
            continue
        try:
            source = abs_path.read_text(encoding="utf-8")
            compile(source, str(abs_path), "exec")
            compiled_count += 1
        except SyntaxError as exc:
            violations.append("candidate_syntax_guard_syntax_error")
            syntax_errors.append(
                {
                    "path": rel_path,
                    "error_type": "syntax_error",
                    "message": str(exc.msg),
                    "lineno": exc.lineno,
                    "offset": exc.offset,
                    "text": (exc.text or "").strip(),
                }
            )
        except UnicodeDecodeError as exc:
            violations.append("candidate_syntax_guard_decode_error")
            syntax_errors.append(
                {
                    "path": rel_path,
                    "error_type": "decode_error",
                    "message": str(exc),
                }
            )
    satisfied = not violations
    audit.update(
        {
            "attempted": True,
            "target_files": target_files,
            "python_source_files": python_files,
            "compiled_file_count": compiled_count,
            "syntax_error_count": sum(1 for item in syntax_errors if item["error_type"] == "syntax_error"),
            "missing_file_count": missing_count,
            "violations": sorted(set(violations)),
            "syntax_errors": syntax_errors,
            "satisfied": satisfied,
            "reason": (
                "no_python_source_files"
                if not python_files
                else ("candidate_syntax_guard_passed" if satisfied else "candidate_syntax_guard_failed")
            ),
        }
    )
    return audit


def _empty_semantic_patch_correctness_v2_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.semantic_patch_correctness_v2_candidate_gate.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "satisfied": True,
        "reason": "disabled" if not enabled else "not_attempted",
        "violations": [],
        "semantic_effect_cues": [],
        "expected_runtime_effects": [],
        "expected_runtime_effect_declared": False,
        "source_span_coverage": {
            "target_source_candidates": [],
            "changed_source_files": [],
            "covered_target_source_candidates": [],
            "satisfied": True,
        },
        "final_target_normalization": {
            "protocol_target_issue_signal": False,
            "normalization_declared": False,
            "satisfied": True,
        },
        "claim_boundary": {
            "pre_oracle_candidate_gate": True,
            "does_not_call_models": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _semantic_patch_correctness_v2_candidate_audit(
    *,
    enabled: bool,
    workspace: Path,
    patch_summary: dict[str, Any],
    stage_outputs: dict[str, Any] | None = None,
    semantic_effect_cues: list[object] | None = None,
    target_source_candidates: list[object] | None = None,
) -> dict[str, Any]:
    audit = _empty_semantic_patch_correctness_v2_audit(enabled=enabled)
    if not enabled:
        return audit
    normalized_cues = [
        dict(cue)
        for cue in list(semantic_effect_cues or [])
        if isinstance(cue, dict) and any(str(value).strip() for value in cue.values())
    ]
    changed_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("changed_files", []) or [])
        if _normalize_repo_path(path)
    ]
    changed_source_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("non_test_files", []) or [])
        if _normalize_repo_path(path)
    ]
    targets = [
        _normalize_repo_path(path)
        for path in list(target_source_candidates or [])
        if _normalize_repo_path(path)
    ]
    diff_text = _workspace_diff_for_files(workspace, changed_files)
    stage_text = _stage_outputs_text(stage_outputs or {})
    effect_tokens = _semantic_patch_v2_effect_tokens(normalized_cues)
    effect_declared = bool(
        "MASGUARD_SEMANTIC_EFFECT" in stage_text
        or "expected_runtime_effect" in stage_text
        or "expected runtime effect" in stage_text.lower()
        or _semantic_patch_v2_text_mentions_effect(
            text=f"{stage_text}\n{diff_text}",
            tokens=effect_tokens,
        )
    )
    covered_targets = _path_suffix_overlap(changed_source_files, targets)
    protocol_signal = _semantic_patch_v2_protocol_target_issue_signal(normalized_cues)
    final_target_normalized = bool(
        not protocol_signal
        or "MASGUARD_FINAL_TARGET_NORMALIZED" in stage_text
        or "final target normalized" in stage_text.lower()
        or "pytest nodeid normalized" in stage_text.lower()
    )
    violations: list[str] = []
    if not normalized_cues:
        violations.append("semantic_v2_missing_semantic_effect_cue_or_explicit_abstain")
    elif not effect_declared:
        violations.append("semantic_v2_expected_runtime_effect_not_declared")
    if targets and not covered_targets:
        violations.append("semantic_v2_source_span_target_not_covered")
    if not final_target_normalized:
        violations.append("semantic_v2_final_target_not_normalized")
    satisfied = not violations
    audit.update(
        {
            "attempted": True,
            "semantic_effect_cues": normalized_cues[:8],
            "expected_runtime_effects": effect_tokens[:16],
            "expected_runtime_effect_declared": effect_declared,
            "source_span_coverage": {
                "target_source_candidates": targets[:16],
                "changed_source_files": changed_source_files[:16],
                "covered_target_source_candidates": covered_targets[:16],
                "satisfied": bool(not targets or covered_targets),
            },
            "final_target_normalization": {
                "protocol_target_issue_signal": protocol_signal,
                "normalization_declared": final_target_normalized,
                "satisfied": final_target_normalized,
            },
            "violations": violations,
            "satisfied": satisfied,
            "reason": (
                "semantic_patch_correctness_v2_passed"
                if satisfied
                else "semantic_patch_correctness_v2_rejected"
            ),
        }
    )
    return audit


def _semantic_patch_v2_effect_tokens(cues: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for cue in cues:
        for key in ("value", "call_target", "targets", "expected", "source", "kind"):
            raw = str(cue.get(key, "") or "")
            if not raw:
                continue
            tokens.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", raw))
            tokens.extend(re.findall(r"\b\d+(?:\.\d+)?\b", raw))
    stopwords = {"expected", "actual", "source", "value", "kind", "assert", "assertion"}
    return list(
        dict.fromkeys(
            token
            for token in tokens
            if len(token) >= 2 and token.lower() not in stopwords
        )
    )


def _semantic_patch_v2_text_mentions_effect(*, text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    lowered = str(text or "").lower()
    return any(str(token).lower() in lowered for token in tokens[:16])


def _semantic_patch_v2_protocol_target_issue_signal(cues: list[dict[str, Any]]) -> bool:
    text = "\n".join(
        " ".join(str(value) for value in cue.values())
        for cue in cues
    ).lower()
    return bool(
        "error: not found:" in text
        or "pytest_nodeid_not_found" in text
        or "nodeid not found" in text
    )


def _stage_outputs_text(stage_outputs: dict[str, Any]) -> str:
    chunks: list[str] = []
    for value in stage_outputs.values():
        chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return "\n".join(chunks)


def _empty_source_edit_pre_oracle_retry_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.source_edit_pre_oracle_retry.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "first_candidate_admitted": False,
        "first_candidate_gate": {},
        "retry_allowed": False,
        "second_candidate_executed": False,
        "second_candidate_gate": {},
        "final_candidate_admitted": False,
        "reason": "disabled" if not enabled else "not_attempted",
        "final_reason": "disabled" if not enabled else "not_attempted",
        "rejection_context_attached": False,
        "agent_reported_success_after_second_candidate": False,
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit_without_fresh_execution": True,
            "pre_oracle_source_candidate_gate": True,
            "one_retry_max": True,
        },
    }


def _source_edit_pre_oracle_candidate_gate(
    *,
    workspace: Path,
    patch_summary: dict[str, Any],
    source_edit_contract_audit: dict[str, Any],
    patch_contract: str,
    stage_outputs: dict[str, Any] | None = None,
    semantic_patch_correctness_v2: bool = False,
    semantic_effect_cues: list[object] | None = None,
    target_source_candidates: list[object] | None = None,
) -> dict[str, Any]:
    patch_classification = _classify_patch_summary(patch_summary)
    patch_contract_violations = _patch_contract_violations(
        patch_contract=patch_contract,
        patch_legitimacy=patch_classification["patch_legitimacy"],
    )
    syntax_guard = _candidate_pre_admission_syntax_guard_audit(
        enabled=True,
        workspace=workspace,
        patch_summary=patch_summary,
    )
    semantic_v2_audit = _semantic_patch_correctness_v2_candidate_audit(
        enabled=bool(semantic_patch_correctness_v2),
        workspace=workspace,
        patch_summary=patch_summary,
        stage_outputs=stage_outputs or {},
        semantic_effect_cues=list(semantic_effect_cues or []),
        target_source_candidates=list(target_source_candidates or []),
    )
    violations: list[str] = []
    violations.extend(str(item) for item in list(source_edit_contract_audit.get("violations", []) or []))
    violations.extend(str(item) for item in patch_contract_violations)
    violations.extend(str(item) for item in list(syntax_guard.get("violations", []) or []))
    violations.extend(str(item) for item in list(semantic_v2_audit.get("violations", []) or []))
    admitted = not violations
    return {
        "schema": "masguard.source_edit_pre_oracle_candidate_gate.v1",
        "admitted": admitted,
        "reason": "candidate_pre_oracle_admitted" if admitted else "candidate_pre_oracle_rejected",
        "violations": sorted(set(violations)),
        "patch_legitimacy": patch_classification["patch_legitimacy"],
        "patch_contract_violations": patch_contract_violations,
        "source_edit_contract_satisfied": bool(source_edit_contract_audit.get("satisfied", True)),
        "source_edit_contract_violations": list(source_edit_contract_audit.get("violations", []) or []),
        "syntax_guard": syntax_guard,
        "semantic_patch_correctness_v2": semantic_v2_audit,
        "changed_files": list(patch_summary.get("changed_files", []) or []),
        "source_files": list(patch_summary.get("non_test_files", []) or []),
        "uses_oracle_success": False,
        "claim_boundary": {
            "pre_oracle_candidate_gate": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _source_edit_pre_oracle_rejection_context(audit: dict[str, Any]) -> str:
    first_gate = dict(audit.get("first_candidate_gate", {}) or {})
    syntax_guard = dict(first_gate.get("syntax_guard", {}) or {})
    semantic_v2 = dict(first_gate.get("semantic_patch_correctness_v2", {}) or {})
    payload = {
        "schema": "masguard.source_edit_pre_oracle_rejection_context.v1",
        "violations": list(first_gate.get("violations", []) or []),
        "patch_legitimacy": str(first_gate.get("patch_legitimacy", "") or ""),
        "changed_files": list(first_gate.get("changed_files", []) or []),
        "source_files": list(first_gate.get("source_files", []) or []),
        "syntax_errors": list(syntax_guard.get("syntax_errors", []) or []),
        "semantic_patch_correctness_v2": semantic_v2,
        "required_behavior": [
            "produce exactly one source-only candidate that passes the source-edit contract",
            "ensure every changed Python source file compiles before claiming completion",
            "if semantic_patch_correctness_v2 is enabled, state MASGUARD_SEMANTIC_EFFECT or expected_runtime_effect before the patch",
            "if semantic_patch_correctness_v2 is enabled, cover the listed source_span_coverage target source candidates",
            "if semantic_patch_correctness_v2 reports a target/protocol issue, state MASGUARD_FINAL_TARGET_NORMALIZED or abstain",
            "do not repeat the rejected no-diff or syntax-broken candidate",
            "abstain explicitly if the bounded source evidence is insufficient",
        ],
        "candidate_policy": {
            "previous_candidate_rejected_pre_oracle": True,
            "produce_one_new_candidate_or_abstain": True,
            "must_pass_pre_oracle_gate_before_validation": True,
            "one_retry_max": True,
        },
    }
    return (
        "[MASGUARD SOURCE-EDIT PRE-ORACLE REJECTION]\n"
        "The previous candidate was rejected before oracle evaluation by a deterministic source-edit gate. "
        "Revise it once into a minimal source-only patch that passes syntax and source-contract checks, or abstain.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD SOURCE-EDIT PRE-ORACLE REJECTION]"
    )


def _source_edit_pre_oracle_blocks_final_evaluation(audit: dict[str, Any]) -> bool:
    if not bool(audit.get("enabled", False)) or not bool(audit.get("attempted", False)):
        return False
    return not bool(audit.get("final_candidate_admitted", False))


def _empty_diff_first_or_abstain_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.diff_first_or_abstain.v1",
        "enabled": bool(enabled),
        "attempted": False,
        "intent_present": False,
        "intent_valid": False,
        "abstain_marker_present": False,
        "abstain_ignored_due_to_source_diff": False,
        "abstained": False,
        "satisfied": False,
        "reason": "disabled" if not enabled else "not_attempted",
        "intent": {},
        "abstain_reason": "",
        "marker_source": "",
        "claim_boundary": {
            "method_change_probe_only": True,
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit_without_fresh_execution": True,
            "candidate_generation_interface": True,
        },
    }


def _diff_first_or_abstain_audit(
    *,
    enabled: bool,
    stage_outputs: dict[str, Any],
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    audit = _empty_diff_first_or_abstain_audit(enabled=enabled)
    if not enabled:
        return audit
    text = _stage_output_text(dict(stage_outputs.get("patcher", {}) or {}))
    intent = _extract_diff_first_intent(text)
    abstain_reason = _extract_diff_first_abstain_reason(text)
    source_files = [str(path) for path in list(patch_summary.get("non_test_files", []) or []) if str(path).strip()]
    intent_present = bool(intent)
    intent_valid = intent_present and not bool(intent.get("parse_error"))
    abstain_marker_present = bool(abstain_reason) or "MASGUARD_STRICT_ABSTAIN_NO_EDIT" in text
    diff_intent_with_source_diff = bool(intent_valid and source_files)
    abstained = bool(abstain_marker_present and not diff_intent_with_source_diff)
    satisfied = bool(diff_intent_with_source_diff or abstained)
    reason = "diff_intent_with_source_diff" if diff_intent_with_source_diff else (
        "explicit_abstention" if abstained else "missing_diff_intent_or_abstain"
    )
    audit.update(
        {
            "attempted": True,
            "intent_present": intent_present,
            "intent_valid": intent_valid,
            "abstain_marker_present": abstain_marker_present,
            "abstain_ignored_due_to_source_diff": bool(abstain_marker_present and diff_intent_with_source_diff),
            "abstained": abstained,
            "satisfied": satisfied,
            "reason": reason,
            "intent": intent,
            "abstain_reason": abstain_reason,
            "source_files": source_files,
            "changed_files": list(patch_summary.get("changed_files", []) or []),
            "marker_source": "patcher_stage_output",
        }
    )
    return audit


def _stage_output_text(stage_output: dict[str, Any]) -> str:
    chunks: list[str] = []
    for command in list(stage_output.get("commands", []) or []):
        if isinstance(command, dict):
            chunks.append(str(command.get("command", "") or ""))
            chunks.append(str(command.get("output", "") or ""))
    for message in list(stage_output.get("messages", []) or []):
        if isinstance(message, dict):
            chunks.append(str(message.get("content", "") or ""))
    chunks.append(str(stage_output.get("patch", "") or ""))
    return "\n".join(item for item in chunks if item)


def _extract_diff_first_intent(text: str) -> dict[str, Any]:
    candidates = _extract_diff_first_intent_candidates(str(text or ""))
    if not candidates:
        return {}
    last_error: dict[str, Any] = {}
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last_error = {"parse_error": "invalid_json", "raw": raw[:500]}
            continue
        if not isinstance(parsed, dict):
            last_error = {"parse_error": "non_object_json", "raw": raw[:500]}
            continue
        return parsed
    return last_error


def _extract_diff_first_intent_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"MASGUARD_DIFF_INTENT\s*[:=]\s*", str(text or "")):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text) or text[start] != "{":
            continue
        raw = _balanced_json_object_at(text, start)
        if raw:
            candidates.append(raw)
    return candidates


def _balanced_json_object_at(text: str, start: int) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:].splitlines()[0][:500]


def _extract_diff_first_abstain_reason(text: str) -> str:
    match = re.search(r"MASGUARD_ABSTAIN_REASON\s*[:=]\s*(.+)", str(text or ""))
    return match.group(1).strip()[:500] if match else ""


def _diff_first_or_abstain_blocks_final_evaluation(audit: dict[str, Any]) -> bool:
    if not bool(audit.get("enabled", False)) or not bool(audit.get("attempted", False)):
        return False
    if bool(audit.get("abstained", False)):
        return True
    return not bool(audit.get("satisfied", False))


def _failure_class_contract_candidate_audit(
    *,
    contract: dict[str, Any],
    patch_summary: dict[str, Any],
    source_edit_contract_audit: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    source_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("non_test_files", []) or [])
        if _normalize_repo_path(path)
    ]
    changed_files = [
        _normalize_repo_path(path)
        for path in list(patch_summary.get("changed_files", []) or [])
        if _normalize_repo_path(path)
    ]
    target_candidates = [
        _normalize_repo_path(path)
        for path in list(contract.get("target_source_candidates", []) or [])
        if _normalize_repo_path(path)
    ]
    diff_text = _workspace_diff_for_files(workspace, changed_files)
    added_lines = [
        line[1:].strip()
        for line in str(diff_text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    violations = []
    if not bool(source_edit_contract_audit.get("satisfied", True)):
        violations.append("source_edit_contract_not_satisfied")
    if bool(contract.get("must_change_source", False)) and not source_files:
        violations.append("failure_class_contract_no_source_diff")
    if bool(contract.get("must_retarget_source", False)) and target_candidates:
        if not _paths_overlap(source_files, target_candidates):
            violations.append("failure_class_contract_target_source_not_covered")
    if bool(contract.get("must_add_narrow_condition", False)):
        condition_markers = ("if ", "elif ", "else:", "except ", "return ", "raise ")
        if not any(line.startswith(condition_markers) or " if " in line for line in added_lines):
            violations.append("failure_class_contract_missing_narrow_condition_signal")
    if bool(contract.get("must_revise_semantic_mechanism", False)):
        guarded_fallback_active = (
            str(source_edit_contract_audit.get("contract", "") or "") == "guarded_fallback"
            and bool(
                dict(source_edit_contract_audit.get("guarded_fallback_audit", {}) or {}).get(
                    "satisfied", False
                )
            )
        )
        if _diff_has_broad_exception_swallowing(diff_text) and not guarded_fallback_active:
            violations.append("failure_class_contract_broad_exception_swallowing")
        if not added_lines:
            violations.append("failure_class_contract_missing_semantic_diff")
    if str(contract.get("failure_class", "") or "") == "environment_or_protocol" and source_files:
        violations.append("failure_class_contract_source_edit_on_protocol_signal")
    return {
        "schema": "masguard.failure_class_contract_candidate_audit.v1",
        "satisfied": not violations,
        "violations": violations,
        "failure_class": str(contract.get("failure_class", "") or "unknown"),
        "retry_action": str(contract.get("retry_action", "") or "abstain"),
        "source_files": source_files,
        "changed_files": changed_files,
        "target_source_candidates": target_candidates,
        "covered_target_source_candidates": _path_suffix_overlap(source_files, target_candidates),
        "added_line_count": len(added_lines),
        "uses_oracle_success": False,
        "claim_boundary": {
            "pre_oracle_candidate_gate": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _contract_verified_candidate_rejection_context(
    candidate_audit: dict[str, Any],
    *,
    workspace: Path | None = None,
) -> str:
    class_contract = dict(candidate_audit.get("class_specific_edit_contract", {}) or {})
    first_audit = dict(candidate_audit.get("first_candidate_class_contract_audit", {}) or {})
    first_syntax_guard = dict(
        candidate_audit.get("first_candidate_pre_admission_syntax_guard_audit", {}) or {}
    )
    first_source_audit = dict(candidate_audit.get("first_candidate_source_edit_contract_audit", {}) or {})
    first_patch_summary = dict(candidate_audit.get("first_candidate_patch_summary", {}) or {})
    target_source_candidates = list(class_contract.get("target_source_candidates", []) or [])
    syntax_errors = list(first_syntax_guard.get("syntax_errors", []) or [])
    syntax_violations = list(first_syntax_guard.get("violations", []) or [])
    source_violations = list(first_source_audit.get("violations", []) or [])
    contract_violations = list(first_audit.get("violations", []) or [])
    payload = {
        "schema": "masguard.contract_verified_candidate_rejection_context.v1",
        "failure_class": str(class_contract.get("failure_class", "") or "unknown"),
        "retry_action": str(class_contract.get("retry_action", "") or "abstain"),
        "violations": list(dict.fromkeys(contract_violations + source_violations + syntax_violations)),
        "contract_violations": contract_violations,
        "source_edit_contract_violations": source_violations,
        "syntax_guard": {
            "attempted": bool(first_syntax_guard.get("attempted", False)),
            "satisfied": bool(first_syntax_guard.get("satisfied", True)),
            "reason": str(first_syntax_guard.get("reason", "") or ""),
            "syntax_errors": syntax_errors[:5],
        },
        "rejected_candidate_patch_summary": {
            "changed_files": list(first_patch_summary.get("changed_files", []) or [])[:8],
            "source_files": list(first_patch_summary.get("non_test_files", []) or [])[:8],
        },
        "target_source_candidates": target_source_candidates,
        "target_source_candidate_summaries": _target_source_candidate_summaries(
            workspace=workspace,
            target_source_candidates=target_source_candidates,
        ),
        "required_edit_behavior": str(class_contract.get("required_edit_behavior", "") or ""),
        "hard_rules": list(class_contract.get("hard_rules", []) or []),
        "candidate_policy": {
            "previous_candidate_rejected_pre_oracle": True,
            "previous_candidate_syntax_rejected": bool(
                first_syntax_guard.get("attempted", False)
                and not first_syntax_guard.get("satisfied", True)
            ),
            "produce_one_new_candidate_or_abstain": True,
            "do_not_repeat_rejected_candidate": True,
            "must_touch_one_target_source_candidate": True,
            "must_anchor_candidate_to_target_source_summary": bool(workspace and target_source_candidates),
            "must_pass_contract_before_oracle": True,
            "must_fix_listed_syntax_errors_before_semantic_validation": bool(syntax_errors),
        },
    }
    return (
        "[MASGUARD CONTRACT-VERIFIED CANDIDATE REJECTION]\n"
        "The previous candidate patch was rejected before oracle evaluation because it violated the "
        "failure-class edit contract, source-edit contract, or syntax guard. Produce one new minimal "
        "source-only candidate that satisfies the listed contract and fixes any listed syntax errors, "
        "or abstain if the evidence is insufficient.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD CONTRACT-VERIFIED CANDIDATE REJECTION]"
    )


def _target_source_candidate_summaries(
    *,
    workspace: Path | None,
    target_source_candidates: list[object],
    limit: int = 3,
) -> list[dict[str, Any]]:
    if workspace is None:
        return []
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in target_source_candidates:
        rel_path = _normalize_repo_path(raw_path)
        if not rel_path or rel_path in seen or _looks_like_test_or_runner_path(rel_path):
            continue
        abs_path = workspace / rel_path
        if not abs_path.exists():
            matches = [
                candidate
                for candidate in workspace.rglob(Path(rel_path).name)
                if candidate.is_file()
                and (
                    candidate.relative_to(workspace).as_posix().endswith(rel_path)
                    or rel_path.endswith(candidate.relative_to(workspace).as_posix())
                )
            ]
            if len(matches) == 1:
                try:
                    rel_path = matches[0].relative_to(workspace).as_posix()
                    abs_path = matches[0]
                except ValueError:
                    pass
        if rel_path in seen:
            continue
        seen.add(rel_path)
        summary = {
            "path": rel_path,
            "exists": abs_path.exists() and abs_path.is_file(),
            "line_count": 0,
            "symbols": [],
            "excerpt": "",
        }
        if summary["exists"]:
            try:
                text = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            lines = text.splitlines()
            summary["line_count"] = len(lines)
            summary["symbols"] = _python_symbol_outline(text)[:12] if rel_path.endswith(".py") else []
            excerpt_lines = lines[:80]
            summary["excerpt"] = "\n".join(excerpt_lines)[:2400]
        summaries.append(summary)
        if len(summaries) >= limit:
            break
    return summaries


def _python_symbol_outline(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    symbols: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(
                {
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "lineno": int(getattr(node, "lineno", 0) or 0),
                }
            )
    return sorted(symbols, key=lambda item: int(item["lineno"]))


def _contract_verified_candidate_blocks_final_evaluation(bounded_signal_retry_audit: dict[str, Any]) -> bool:
    candidate_audit = dict(bounded_signal_retry_audit.get("contract_verified_candidate_retry_audit", {}) or {})
    if not bool(candidate_audit.get("attempted", False)):
        return False
    return not bool(candidate_audit.get("final_candidate_admitted", False))


def _bounded_validation_retry_allowed(
    failure_audit: dict[str, Any],
    *,
    failure_class_conditioned_retry: bool,
) -> bool:
    if bool(failure_audit.get("source_repair_feedback_ready", False)):
        return True
    if not failure_class_conditioned_retry:
        return False
    if _bounded_validation_no_diff_source_target_retry_allowed(failure_audit):
        return True
    if not bool(failure_audit.get("feedback_ready", False)):
        return False
    classification = dict(failure_audit.get("failure_classification", {}) or {})
    failure_class = str(classification.get("failure_class", "") or "")
    retry_action = str(classification.get("retry_action", "") or "")
    return failure_class in {"no_diff", "source_location", "missing_condition", "wrong_abstraction"} and retry_action not in {
        "",
        "abstain",
    }


def _bounded_validation_no_diff_source_target_retry_allowed(failure_audit: dict[str, Any]) -> bool:
    classification = dict(failure_audit.get("failure_classification", {}) or {})
    failure_class = str(classification.get("failure_class", "") or "")
    retry_action = str(classification.get("retry_action", "") or "")
    suspect_paths = [
        _normalize_repo_path(path)
        for path in list(classification.get("suspect_paths", []) or [])
        if _normalize_repo_path(path)
    ]
    traceback_targets = [
        _normalize_repo_path(path)
        for path in list(classification.get("production_traceback_targets", []) or [])
        if _normalize_repo_path(path)
    ]
    if failure_class != "no_diff":
        return False
    if retry_action in {"", "abstain"}:
        return False
    return bool(suspect_paths or traceback_targets)


def _external_validation_feedback_repeat_diff_audit(
    *,
    admission: dict[str, Any],
    workspace: Path,
    changed_files: list[str],
) -> dict[str, Any]:
    base = {
        "schema": "masguard.external_validation_feedback_repeat_diff_audit.v1",
        "enabled": bool(admission.get("admitted", False)),
        "repeated_previous_diff": False,
        "previous_diff_present": bool(str(admission.get("previous_diff_excerpt", "") or "").strip()),
        "current_diff_present": False,
        "reason": "disabled" if not admission.get("admitted", False) else "",
        "claim_boundary": {
            "does_not_grant_recovery_credit": True,
            "repeated_previous_diff_blocks_method_promotion": True,
        },
    }
    if not admission.get("admitted", False):
        return base
    previous = _normalize_diff_for_repeat_check(str(admission.get("previous_diff_excerpt", "") or ""))
    current = _normalize_diff_for_repeat_check(_workspace_diff_for_files(workspace, changed_files))
    base["current_diff_present"] = bool(current)
    if not previous:
        base["reason"] = "previous_diff_missing"
        return base
    if not current:
        base["reason"] = "current_diff_missing"
        return base
    base["repeated_previous_diff"] = previous == current
    base["reason"] = "repeated_previous_diff" if base["repeated_previous_diff"] else "new_diff_relative_to_previous"
    return base


def _external_validation_semantic_target_audit(
    *,
    admission: dict[str, Any],
    changed_files: list[str],
) -> dict[str, Any]:
    signature = dict(admission.get("signature", {}) or {})
    changed = [_normalize_repo_path(path) for path in changed_files if _normalize_repo_path(path)]
    previous_targets = [
        _normalize_repo_path(path) for path in list(admission.get("previous_source_files", []) or [])
        if _normalize_repo_path(path)
    ]
    traceback_targets = _production_traceback_target_suffixes(
        list(signature.get("traceback_source_files", []) or [])
    )
    all_targets = list(dict.fromkeys(previous_targets + traceback_targets))
    previous_overlap = _path_suffix_overlap(changed, previous_targets)
    traceback_overlap = _path_suffix_overlap(changed, traceback_targets)
    satisfied = bool(all_targets and (previous_overlap or traceback_overlap))
    reason = "semantic_target_covered_by_current_diff" if satisfied else "semantic_targets_not_covered_by_current_diff"
    if not admission.get("admitted", False):
        reason = "disabled"
        satisfied = True
    elif not all_targets:
        reason = "no_semantic_targets_available"
        satisfied = True
    elif not changed:
        reason = "current_source_diff_missing"
    return {
        "schema": "masguard.external_validation_semantic_target_audit.v1",
        "enabled": bool(admission.get("admitted", False)),
        "satisfied": bool(satisfied),
        "reason": reason,
        "changed_files": changed[:16],
        "previous_source_targets": previous_targets[:16],
        "production_traceback_source_targets": traceback_targets[:16],
        "covered_previous_source_targets": previous_overlap[:16],
        "covered_production_traceback_source_targets": traceback_overlap[:16],
        "claim_boundary": {
            "does_not_use_oracle_success": True,
            "does_not_grant_recovery_credit": True,
            "semantic_target_miss_triggers_one_forced_revision": True,
        },
    }


def _production_traceback_target_suffixes(paths: list[object]) -> list[str]:
    targets: list[str] = []
    for raw in paths:
        path = _normalize_repo_path(str(raw))
        if not path or _looks_like_test_or_runner_path(path):
            continue
        parts = path.split("/")
        for index in range(max(0, len(parts) - 6), len(parts)):
            suffix = "/".join(parts[index:])
            if "/" in suffix and suffix.endswith(".py"):
                targets.append(suffix)
    return list(dict.fromkeys(targets))


def _normalize_repo_path(path: object) -> str:
    value = str(path or "").strip().replace("\\", "/")
    if not value:
        return ""
    value = re.sub(r"^[ab]/", "", value)
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


def _looks_like_test_or_runner_path(path: str) -> bool:
    parts = set(_normalize_repo_path(path).split("/"))
    basename = Path(path).name
    return bool(
        "tests" in parts
        or "test" in parts
        or basename.startswith("test_")
        or basename in {"runtests.py", "runner.py"}
    )


def _path_suffix_overlap(changed: list[str], targets: list[str]) -> list[str]:
    covered: list[str] = []
    for target in targets:
        for changed_file in changed:
            if changed_file == target or changed_file.endswith(f"/{target}") or target.endswith(f"/{changed_file}"):
                covered.append(target)
                break
    return list(dict.fromkeys(covered))


def _normalize_diff_for_repeat_check(diff_text: str) -> str:
    lines = []
    for line in str(diff_text or "").splitlines():
        if line.startswith("index ") or line.startswith("\\ No newline"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _external_validation_feedback_blocks_external_evaluation(
    *,
    external_feedback_audit: dict[str, Any],
    forced_revision_audit: dict[str, Any],
) -> bool:
    return bool(
        external_feedback_audit.get("admitted", False)
        and forced_revision_audit.get("repeated_previous_diff_after_retry", False)
    )


def _external_validation_forced_revision_context(
    *,
    admission: dict[str, Any],
    repeat_audit: dict[str, Any],
    semantic_target_audit: dict[str, Any] | None = None,
    trigger_reasons: list[str] | None = None,
) -> str:
    signature = dict(admission.get("signature", {}) or {})
    semantic_target_audit = dict(semantic_target_audit or {})
    trigger_reasons = list(trigger_reasons or [str(repeat_audit.get("reason", "") or "forced_revision")])
    payload = {
        "schema": "masguard.external_validation_forced_revision_context.v1",
        "trigger": "+".join(trigger_reasons),
        "repeat_audit_reason": str(repeat_audit.get("reason", "") or ""),
        "semantic_target_audit_reason": str(semantic_target_audit.get("reason", "") or ""),
        "previous_source_files": list(admission.get("previous_source_files", []) or []),
        "previous_diff_excerpt": str(admission.get("previous_diff_excerpt", "") or "")[:2000],
        "required_semantic_target_evidence": {
            "changed_files": list(semantic_target_audit.get("changed_files", []) or []),
            "previous_source_targets": list(semantic_target_audit.get("previous_source_targets", []) or []),
            "production_traceback_source_targets": list(
                semantic_target_audit.get("production_traceback_source_targets", []) or []
            ),
            "covered_previous_source_targets": list(
                semantic_target_audit.get("covered_previous_source_targets", []) or []
            ),
            "covered_production_traceback_source_targets": list(
                semantic_target_audit.get("covered_production_traceback_source_targets", []) or []
            ),
        },
        "external_validation_signature": {
            "exception_classes": list(signature.get("exception_classes", []) or []),
            "failing_tests": list(signature.get("failing_tests", []) or []),
            "traceback_source_files": list(signature.get("traceback_source_files", []) or []),
            "validation_tool_missing": bool(signature.get("validation_tool_missing", False)),
        },
        "hard_rules": [
            "The current source diff is equivalent to the previous failed diff and is rejected.",
            "Do not submit another patch with the same added and removed source lines.",
            "The revised source diff must cover at least one prior failed source target or production traceback source target when such targets exist.",
            "Change the repair mechanism so the listed failing tests and exception evidence are directly explained.",
            "Keep the patch source-only, minimal, and within the same source target unless the locator evidence justifies abstaining.",
            "If a genuinely different semantic repair is not supported by source evidence, abstain instead of repeating the rejected diff.",
        ],
    }
    return (
        "[MASGUARD FORCED REVISION GATE]\n"
        "The current patch repeats a previous source diff that already failed external validation. "
        "This repeated diff is rejected for this method branch; produce a different semantic repair or abstain.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD FORCED REVISION GATE]"
    )


def _method_variant(
    *,
    verifier_failure_feedback_retry: bool,
    bounded_validation_signal_retry: bool,
    failure_class_conditioned_retry: bool,
    contract_verified_candidate_retry: bool = False,
    external_validation_feedback_artifact: Path | None,
    baseline_success_contract_artifact: Path | None = None,
    historical_experience_artifact: Path | None = None,
    historical_experience_mode: str = "mas_conditioned",
    mas_experience_controller_artifact: Path | None = None,
    historical_action_program_artifact: Path | None = None,
    diverse_repair_hypothesis_retry: bool = False,
    candidate_pre_admission_syntax_guard: bool = False,
    source_edit_pre_oracle_retry: bool = False,
    diff_first_or_abstain: bool = False,
    semantic_patch_correctness_v2: bool = False,
) -> str:
    if historical_action_program_artifact is not None:
        if source_edit_pre_oracle_retry:
            if semantic_patch_correctness_v2:
                if diff_first_or_abstain:
                    return "historical_action_program_semantic_patch_v2_diff_first_source_edit_pre_oracle"
                return "historical_action_program_semantic_patch_v2_source_edit_pre_oracle"
            if diff_first_or_abstain:
                return "historical_action_program_diff_first_source_edit_pre_oracle"
            return "historical_action_program_source_edit_pre_oracle"
        if diff_first_or_abstain:
            return "historical_action_program_diff_first_or_abstain"
        return "historical_action_program"
    if mas_experience_controller_artifact is not None:
        if historical_experience_artifact is not None:
            return "mas_experience_controller_with_historical_case_memory"
        if source_edit_pre_oracle_retry:
            if semantic_patch_correctness_v2:
                return "mas_experience_controller_semantic_patch_v2_source_edit_pre_oracle"
            return "mas_experience_controller_source_edit_pre_oracle"
        if bounded_validation_signal_retry:
            return "mas_experience_controller_bounded_validation"
        return "mas_experience_controller"
    if baseline_success_contract_artifact is not None:
        if source_edit_pre_oracle_retry:
            return "baseline_success_contract_source_edit_pre_oracle_transfer"
        if diff_first_or_abstain:
            return "baseline_success_contract_diff_first_transfer"
        return "baseline_success_contract_transfer"
    if historical_experience_artifact is not None:
        prefix = "historical_experience_generic_patch_memory" if historical_experience_mode == "generic_patch_memory" else "historical_experience_mas_conditioned"
        if source_edit_pre_oracle_retry:
            if semantic_patch_correctness_v2:
                return f"{prefix}_semantic_patch_v2_source_edit_pre_oracle_case_memory"
            return f"{prefix}_source_edit_pre_oracle_case_memory"
        if bounded_validation_signal_retry:
            return f"{prefix}_bounded_validation_case_memory"
        return f"{prefix}_case_memory"
    if diff_first_or_abstain:
        if source_edit_pre_oracle_retry:
            if semantic_patch_correctness_v2:
                return "semantic_patch_v2_diff_first_or_abstain_with_source_edit_pre_oracle_retry"
            return "diff_first_or_abstain_with_source_edit_pre_oracle_retry"
        if external_validation_feedback_artifact is not None:
            return "diff_first_or_abstain_with_external_feedback"
        return "diff_first_or_abstain"
    if source_edit_pre_oracle_retry:
        if semantic_patch_correctness_v2:
            return "semantic_patch_correctness_v2_source_edit_pre_oracle_retry"
        if external_validation_feedback_artifact is not None:
            return "source_edit_pre_oracle_retry_with_external_feedback"
        if verifier_failure_feedback_retry:
            return "source_edit_pre_oracle_retry_with_verifier_retry"
        return "source_edit_pre_oracle_retry"
    if bounded_validation_signal_retry:
        if diverse_repair_hypothesis_retry:
            if contract_verified_candidate_retry:
                if candidate_pre_admission_syntax_guard:
                    return "diverse_repair_hypothesis_contract_verified_syntax_guard_retry"
                return "diverse_repair_hypothesis_contract_verified_retry"
            if failure_class_conditioned_retry:
                return "diverse_repair_hypothesis_retry"
            return "diverse_repair_hypothesis_retry_without_failure_class"
        if contract_verified_candidate_retry:
            if candidate_pre_admission_syntax_guard:
                return "contract_verified_candidate_syntax_guard_retry"
            if external_validation_feedback_artifact is not None:
                return "contract_verified_candidate_retry_with_external_feedback"
            if verifier_failure_feedback_retry:
                return "contract_verified_candidate_retry_with_verifier_retry"
            return "contract_verified_candidate_retry"
        if failure_class_conditioned_retry:
            if external_validation_feedback_artifact is not None:
                return "failure_class_conditioned_bounded_validation_retry_with_external_feedback"
            if verifier_failure_feedback_retry:
                return "failure_class_conditioned_bounded_validation_retry_with_verifier_retry"
            return "failure_class_conditioned_bounded_validation_retry"
        if external_validation_feedback_artifact is not None:
            return "bounded_validation_signal_retry_with_external_feedback"
        if verifier_failure_feedback_retry:
            return "bounded_validation_signal_retry_with_verifier_retry"
        return "bounded_validation_signal_retry"
    if external_validation_feedback_artifact is not None:
        if verifier_failure_feedback_retry:
            return "external_validation_feedback_second_pass_with_verifier_retry"
        return "external_validation_feedback_second_pass"
    if verifier_failure_feedback_retry:
        return "verifier_failure_feedback_retry"
    return "clean_start_baseline"


def _previous_source_diff_excerpt(row: dict[str, Any], source_files: list[str]) -> str:
    workspace = Path(str(row.get("workspace", "") or ""))
    if not workspace.is_dir():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--", *source_files[:8]],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return str(result.stdout or "")[:4000]


def _empty_validation_failure_signature() -> dict[str, object]:
    return {
        "validation_tool_missing": False,
        "exception_classes": [],
        "failing_tests": [],
        "traceback_source_files": [],
    }


def _merge_validation_failure_signatures(signatures: list[dict[str, object]]) -> dict[str, object]:
    merged = _empty_validation_failure_signature()
    for signature in signatures:
        merged["validation_tool_missing"] = bool(
            merged["validation_tool_missing"] or signature.get("validation_tool_missing")
        )
        for key in ("exception_classes", "failing_tests", "traceback_source_files"):
            values = list(merged[key])
            values.extend(str(item) for item in list(signature.get(key, []) or []) if str(item).strip())
            merged[key] = list(dict.fromkeys(values))[:8]
    return merged


def _validation_failure_signature_has_signal(signature: dict[str, object]) -> bool:
    return bool(
        signature.get("validation_tool_missing")
        or signature.get("exception_classes")
        or signature.get("failing_tests")
        or signature.get("traceback_source_files")
    )


def _source_path_probe_locator_salvage(
    *,
    locator_result: dict[str, Any],
    workspace: Path,
    recovery_context: str,
    patch_contract: str,
    mode: str,
    initial_agent_reported_success: bool,
) -> dict[str, Any]:
    audit = {
        "mode": mode,
        "attempted": False,
        "accepted": False,
        "reason": "not_attempted",
        "candidate_paths": [],
        "located_files": "",
        "initial_agent_reported_success": bool(initial_agent_reported_success),
        "agent_reported_success_after_salvage": False,
        "uses_oracle_or_test_outcome": False,
        "uses_prior_failed_mas_graph": False,
        "uses_only_locator_readonly_probe_paths": True,
    }
    if mode == "none":
        audit["reason"] = "disabled"
        return audit
    audit["attempted"] = True
    if mode != "source_path_probe":
        audit["reason"] = "unsupported_mode"
        return audit
    if initial_agent_reported_success:
        audit["reason"] = "locator_pipeline_already_succeeded"
        return audit
    if patch_contract != "source_only":
        audit["reason"] = "requires_source_only_patch_contract"
        return audit
    if not str(recovery_context or "").strip():
        audit["reason"] = "requires_recovery_context"
        return audit
    if bool(locator_result.get("success")):
        audit["reason"] = "locator_success_not_salvaged"
        return audit

    candidate_paths = _source_probe_candidate_paths(locator_result=locator_result, workspace=workspace, limit=3)
    audit["candidate_paths"] = list(candidate_paths)
    if not candidate_paths:
        audit["reason"] = "no_existing_source_probe_candidates"
        return audit

    audit["located_files"] = _format_source_probe_located_files(candidate_paths)
    audit["accepted"] = True
    audit["reason"] = "accepted_existing_source_paths_from_locator_readonly_probes"
    return audit


def _source_probe_candidate_paths(*, locator_result: dict[str, Any], workspace: Path, limit: int = 3) -> list[str]:
    scored_paths: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_index = 0

    def add(value: Any, *, score: float) -> None:
        nonlocal seen_index
        text = str(value or "").strip().lstrip("./")
        if not text:
            return
        if text not in first_seen:
            first_seen[text] = seen_index
            seen_index += 1
        scored_paths[text] = max(scored_paths.get(text, 0.0), score)

    for value in list(locator_result.get("suspect_paths", []) or []):
        add(value, score=80.0)
    for value in list(locator_result.get("selected_target_candidates", []) or []):
        add(value, score=30.0)
    for match in re.findall(r"[A-Za-z0-9_./-]+\.py", str(locator_result.get("located_files", "") or "")):
        add(match, score=25.0)
    inspected_symbols = _locator_inspected_symbol_terms(locator_result)
    for command in list(locator_result.get("commands", []) or []):
        if not isinstance(command, dict):
            continue
        command_text = str(command.get("command", "") or "")
        output_text = str(command.get("output", "") or "")
        command_score = _locator_command_specificity_score(command_text, inspected_symbols=inspected_symbols)
        signature = dict(command.get("probe_signature", {}) or {})
        for value in list(signature.get("paths", []) or []):
            add(value, score=command_score)
        for match in re.findall(r"[A-Za-z0-9_./-]+\.py", command_text):
            add(match, score=command_score)
        output_score = max(command_score - 5.0, 5.0)
        for match in re.findall(r"[A-Za-z0-9_./-]+\.py", output_text):
            add(match, score=output_score)

    existing = existing_repo_source_paths(scored_paths.keys(), workspace)
    return sorted(
        existing,
        key=lambda path: (-scored_paths.get(path, 0.0), first_seen.get(path, 10**9), path),
    )[:limit]


def _locator_inspected_symbol_terms(locator_result: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    raw_values = list(locator_result.get("inspected_regions_or_symbols", []) or [])
    for command in list(locator_result.get("commands", []) or []):
        if isinstance(command, dict):
            raw_values.extend(list(command.get("inspected_regions_or_symbols", []) or []))
    for value in raw_values:
        for term in re.split(r"[^A-Za-z0-9_]+", str(value or "")):
            if len(term) >= 4 and term not in terms:
                terms.append(term)
    return terms


def _locator_command_specificity_score(command_text: str, *, inspected_symbols: list[str]) -> float:
    text = str(command_text or "")
    lowered = text.lower()
    if not text.strip():
        return 10.0
    score = 20.0
    if "grep" in lowered or "rg " in lowered or lowered.startswith("rg"):
        score += 20.0
    if "find " in lowered:
        score -= 10.0
    if any(term and term in text for term in inspected_symbols):
        score += 30.0
    if re.search(r"\b(class|def)\s+[A-Za-z_]", text):
        score += 15.0
    if "|" in text or "\\" in text:
        score += 5.0
    return score


def _format_source_probe_located_files(candidate_paths: list[str]) -> str:
    files = [
        {
            "path": path,
            "reason": "extracted from existing source path observed by locator read-only probes after locator failure",
        }
        for path in candidate_paths
    ]
    lines = [
        "source-path-probe salvage localization: locator failed after bounded read-only probes; "
        "use these existing source files as low-confidence patch targets.",
    ]
    for item in files:
        lines.append(f"- {item['path']}: {item['reason']}")
    payload = {
        "files": files,
        "entry_points": [],
        "confidence": "low",
    }
    return "\n".join(
        [
            *lines,
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
        ]
    )


def _compose_recovery_context(
    *,
    patch_contract: str,
    source_edit_contract: str,
    locator_target_paths: list[str] | None = None,
    locator_salvage_accepted: bool = False,
    diff_first_or_abstain: bool = False,
) -> str:
    parts = [
        text
        for text in (
            _recovery_context_for_patch_contract(patch_contract),
            _source_edit_contract_context(
                source_edit_contract,
                locator_target_paths=locator_target_paths,
                locator_salvage_accepted=locator_salvage_accepted,
                diff_first_or_abstain=diff_first_or_abstain,
            ),
        )
        if text.strip()
    ]
    return "\n\n".join(parts)


def _source_edit_contract_context(
    source_edit_contract: str,
    *,
    locator_target_paths: list[str] | None = None,
    locator_salvage_accepted: bool = False,
    diff_first_or_abstain: bool = False,
) -> str:
    if source_edit_contract not in {"minimal_targeted", "guarded_fallback"}:
        return ""
    target_paths = [
        str(path).strip().lstrip("./")
        for path in list(locator_target_paths or [])
        if str(path).strip()
    ][:3]
    primary_target = target_paths[0] if target_paths else ""
    guarded_fallback = source_edit_contract == "guarded_fallback"
    payload = {
        "schema_version": "masguard.source_edit_contract.v4",
        "contract_id": (
            "clean_start_guarded_fallback_source_edit_v4"
            if guarded_fallback
            else "clean_start_minimal_targeted_source_edit_v2"
        ),
        "requirements": {
            "max_source_files": 1,
            "prefer_first_locator_source_target": True,
            "primary_source_target": primary_target,
            "allowed_source_files": target_paths,
            "locator_salvage_accepted": bool(locator_salvage_accepted),
            "state_failure_mechanism_before_edit": True,
            "produce_minimal_diff_or_abstain": True,
            "focused_validation_required": True,
            "diff_first_or_abstain": bool(diff_first_or_abstain),
            "guarded_fallback_mode": guarded_fallback,
            "compatibility_boundary_required": guarded_fallback,
            "narrow_exception_or_explicit_guard_required": guarded_fallback,
            "observable_fallback_required": guarded_fallback,
            "guarded_fallback_plan_required": guarded_fallback,
            "duplicate_try_except_shape_forbidden": guarded_fallback,
            "signature_grounded_narrow_exception_or_guard_required": guarded_fallback,
            "guarded_fallback_selection_required": guarded_fallback,
            "guarded_fallback_selection_must_match_diff": guarded_fallback,
        },
        "forbidden": {
            "path_classes": ["test", "generated", "build_artifact", "installed_copy"],
            "path_prefixes": [
                "build/lib/",
                "dist/",
                "target/",
                "generated/",
                "site-packages/",
                "__pycache__/",
            ],
            "patch_patterns": [
                "unguarded_broad_except_exception_return_empty",
                "broad_except_baseexception",
                "catch_all_except_pass",
                "unrelated_cleanup",
            ],
        },
        "credit_boundary": {
            "separate_from_source_only_contract": True,
            "oracle_still_required": True,
        },
    }
    return (
        "[MASGUARD SOURCE EDIT CONTRACT]\n"
        "Clean-start strict source-edit contract. Use the locator result as the source boundary. "
        "Prefer one minimal canonical source-file edit in the first concrete locator target. "
        f"{_target_contract_sentence(primary_target=primary_target, target_paths=target_paths)}"
        "Do not edit build/lib, dist, target, generated, site-packages, cache, or other copied/generated paths; "
        "map any such path back to the canonical repository source file before editing. "
        "Before editing, state the failure mechanism and why this file is sufficient. "
        f"{_diff_first_or_abstain_sentence(enabled=diff_first_or_abstain)}"
        f"{_guarded_fallback_sentence(enabled=guarded_fallback)}"
        "After reading enough context, either produce a minimal source diff in the allowed target set or explicitly "
        "abstain because the evidence is insufficient; do not continue with read-only exploration as the final action. "
        "Do not repair by adding broad exception swallowing such as `except Exception: return {}`, "
        "`except BaseException`, catch-all `except: pass`, or other dummy fallback behavior unless the "
        "issue explicitly requires that exact public API behavior and the guarded fallback contract is active. "
        "Run focused validation after the edit.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD SOURCE EDIT CONTRACT]"
    )


def _guarded_fallback_sentence(*, enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "Guarded fallback mode is active: fallback-style edits are allowed only at a decode, parse, load, "
        "deserialize, compatibility, or legacy boundary named by the evidence. The patch must first handle a "
        "narrow exception or explicit guard, attempt the compatibility fallback before any safe default, and make "
        "the fallback observable with logging, warning, raise, or another non-silent signal. Before editing, print "
        "exactly one MASGUARD_GUARDED_FALLBACK_PLAN={...} JSON object with boundary, primary_attempt, "
        "narrow_exception_or_guard, fallback_attempt, observable_signal, safe_default, and target_file. The edit "
        "must derive that narrow_exception_or_guard from the validation signature when a "
        "guarded_fallback_repair_plan is present; otherwise abstain instead of inventing a catch-all fallback. "
        "Before editing, also print exactly one MASGUARD_GUARDED_FALLBACK_SELECTION={...} JSON object with "
        "selected_type, selected_value, source, target_file, and reason; selected_type must be narrow_exception "
        "or explicit_guard, selected_value must name the concrete exception/guard that will appear in the diff, "
        "and the source-edit gate will reject the patch if the selected value is missing from the diff. "
        "The edit must use one coherent try/except structure; do not emit duplicate adjacent try: blocks, dangling "
        "except blocks, or nested fallback scaffolding that would not parse. "
    )


def _diff_first_or_abstain_sentence(*, enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "Before running focused validation or claiming completion, the patch command must print exactly one "
        "`MASGUARD_DIFF_INTENT={...}` JSON object naming target_file, edit_kind, expected_changed_symbol, "
        "and validation_command; if no safe edit is possible, print MASGUARD_STRICT_ABSTAIN_NO_EDIT and "
        "MASGUARD_ABSTAIN_REASON=<one concrete reason> instead. "
    )


def _target_contract_sentence(*, primary_target: str, target_paths: list[str]) -> str:
    if not target_paths:
        return ""
    allowed = ", ".join(target_paths)
    return (
        f"Allowed source targets for this strict attempt are: {allowed}. "
        f"Treat `{primary_target}` as the primary target. "
    )


def _source_edit_contract_audit(
    *,
    source_edit_contract: str,
    patch_summary: dict[str, Any],
    stage_outputs: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    changed_files = [
        str(path)
        for path in list(patch_summary.get("changed_files", []) or [])
        if str(path).strip()
    ]
    candidate_files = changed_files or [
        str(path)
        for path in (
            list(patch_summary.get("non_test_files", []) or [])
            + list(patch_summary.get("test_files", []) or [])
        )
        if str(path).strip()
    ]
    canonical_classes = classify_changed_files(candidate_files)
    source_files = list(canonical_classes.get("source_files", []) or [])
    generated_files = list(canonical_classes.get("generated_files", []) or [])
    test_files = list(canonical_classes.get("test_files", []) or [])
    other_files = list(canonical_classes.get("other_files", []) or [])
    locator_result = dict(stage_outputs.get("locator", {}) or {})
    target_paths = _source_probe_candidate_paths(locator_result=locator_result, workspace=workspace, limit=3)
    diff_text = _workspace_diff_for_files(workspace, changed_files)
    broad_swallowing = _diff_has_broad_exception_swallowing(diff_text)
    guarded_fallback_audit = _guarded_fallback_diff_audit(
        diff_text,
        stage_outputs=stage_outputs,
    )
    audit = {
        "schema": "masguard.source_edit_contract_audit.v1",
        "contract": source_edit_contract,
        "enabled": source_edit_contract != "none",
        "satisfied": True,
        "violations": [],
        "source_files": source_files,
        "changed_files": changed_files,
        "generated_files": generated_files,
        "test_files": test_files,
        "other_files": other_files,
        "locator_target_paths": target_paths,
        "broad_exception_swallowing_detected": broad_swallowing,
        "guarded_fallback_audit": guarded_fallback_audit,
        "diff_source": "workspace_git_diff",
        "claim_boundary": {
            "separate_from_source_only_contract": True,
            "does_not_use_oracle_to_route": True,
            "does_not_grant_recovery_credit": True,
        },
    }
    if source_edit_contract == "none":
        return audit
    violations: list[str] = []
    if source_edit_contract not in {"minimal_targeted", "guarded_fallback"}:
        violations.append("unsupported_source_edit_contract")
    violation_prefix = (
        "guarded_fallback"
        if source_edit_contract == "guarded_fallback"
        else "minimal_targeted"
    )
    if generated_files:
        violations.append(f"{violation_prefix}_generated_or_build_path_changed")
    if test_files:
        violations.append(f"{violation_prefix}_test_path_changed")
    if not source_files:
        violations.append(f"{violation_prefix}_no_source_diff")
    if len(source_files) > 1:
        violations.append(f"{violation_prefix}_multiple_source_files")
    if target_paths and source_files and not _paths_overlap(source_files, target_paths):
        violations.append(f"{violation_prefix}_changed_source_outside_locator_targets")
    if source_edit_contract == "guarded_fallback":
        if not bool(guarded_fallback_audit["satisfied"]):
            violations.extend(
                f"guarded_fallback_{violation}"
                for violation in list(guarded_fallback_audit["violations"])
            )
    elif broad_swallowing:
        violations.append("minimal_targeted_broad_exception_swallowing")
    audit["violations"] = violations
    audit["satisfied"] = not violations
    return audit


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


def _guarded_fallback_diff_audit(
    diff_text: str,
    *,
    stage_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_added_lines = [
        line[1:]
        for line in str(diff_text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_lines = [
        line.strip()
        for line in raw_added_lines
    ]
    full_text = str(diff_text or "").lower()
    added_text = "\n".join(added_lines).lower()
    selection_audit = _guarded_fallback_selection_audit(
        stage_outputs=dict(stage_outputs or {}),
        added_text=added_text,
    )
    compatibility_boundary_markers = (
        "decode",
        "encode",
        "parse",
        "load",
        "loads",
        "deserialize",
        "serializer",
        "legacy",
        "compat",
        "compatibility",
        "bad signature",
        "badsignature",
    )
    fallback_attempt_markers = (
        "_legacy",
        "legacy_",
        "fallback",
        "default",
        "compat",
        "loads(",
        "load(",
        "parse",
        "deserialize",
    )
    observable_markers = (
        "logger.",
        "logging.",
        "warnings.warn",
        "warn(",
        "print(",
        "raise ",
        "exc_info",
    )
    broad_swallowing = _diff_has_broad_exception_swallowing(diff_text)
    checks = {
        "compatibility_boundary_present": any(marker in full_text for marker in compatibility_boundary_markers),
        "narrow_exception_or_guard_present": _guarded_fallback_has_narrow_exception_or_guard(added_text),
        "fallback_attempt_present": any(marker in added_text for marker in fallback_attempt_markers),
        "observable_fallback_present": any(marker in added_text for marker in observable_markers),
        "fallback_patch_shape_valid": not _guarded_fallback_has_malformed_try_shape(raw_added_lines),
        "guarded_fallback_selection_present": bool(selection_audit["present"]),
        "guarded_fallback_selection_matches_diff": bool(selection_audit["matches_diff"]),
    }
    violations = [key for key, value in checks.items() if not value]
    return {
        "schema": "masguard.guarded_fallback_diff_audit.v1",
        "enabled_when_contract_is_guarded_fallback": True,
        "broad_exception_swallowing_detected": broad_swallowing,
        "selection_audit": selection_audit,
        **checks,
        "satisfied": not violations,
        "violations": violations,
        "claim_boundary": {
            "general_compatibility_boundary_rule": True,
            "does_not_use_oracle_to_route": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _guarded_fallback_has_malformed_try_shape(raw_added_lines: list[str]) -> bool:
    code_lines = [
        line
        for line in list(raw_added_lines or [])
        if line.strip() and not line.lstrip().startswith("#")
    ]
    previous_was_try = False
    for line in code_lines:
        stripped = line.strip()
        if stripped == "try:":
            if previous_was_try:
                return True
            previous_was_try = True
            continue
        if stripped.startswith(("except ", "except:", "finally:", "else:")):
            previous_was_try = False
            continue
        if previous_was_try and not line.startswith((" ", "\t")):
            return True
        previous_was_try = False
    return False


def _guarded_fallback_selection_audit(
    *,
    stage_outputs: dict[str, Any],
    added_text: str,
) -> dict[str, Any]:
    text = _stage_output_text(dict(stage_outputs.get("patcher", {}) or {}))
    selection = _extract_guarded_fallback_selection(text)
    present = bool(selection) and not bool(selection.get("parse_error"))
    selected_type = str(selection.get("selected_type", "") or "")
    selected_value = str(selection.get("selected_value", "") or "")
    matches_diff = False
    reason = "missing_selection"
    if present:
        if selected_type == "narrow_exception":
            matches_diff = _guarded_fallback_diff_contains_exception(added_text, selected_value)
            reason = "selection_matches_diff" if matches_diff else "selected_exception_missing_from_diff"
        elif selected_type == "explicit_guard":
            matches_diff = _guarded_fallback_diff_contains_explicit_guard(added_text, selected_value)
            reason = "selection_matches_diff" if matches_diff else "selected_guard_missing_from_diff"
        else:
            reason = "unsupported_selected_type"
    return {
        "schema": "masguard.guarded_fallback_selection_audit.v1",
        "present": present,
        "selection": selection,
        "selected_type": selected_type,
        "selected_value": selected_value,
        "matches_diff": matches_diff,
        "reason": reason,
        "uses_oracle_success": False,
        "claim_boundary": {
            "pre_oracle_guard_selection_gate": True,
            "does_not_grant_recovery_credit": True,
        },
    }


def _extract_guarded_fallback_selection(text: str) -> dict[str, Any]:
    candidates = _extract_marker_json_candidates(
        str(text or ""),
        marker="MASGUARD_GUARDED_FALLBACK_SELECTION",
    )
    if not candidates:
        return {}
    last_error: dict[str, Any] = {}
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last_error = {"parse_error": "invalid_json", "raw": raw[:500]}
            continue
        if not isinstance(parsed, dict):
            last_error = {"parse_error": "non_object_json", "raw": raw[:500]}
            continue
        return parsed
    return last_error


def _extract_marker_json_candidates(text: str, *, marker: str) -> list[str]:
    candidates: list[str] = []
    pattern = re.compile(rf"{re.escape(marker)}\s*[:=]\s*")
    for match in pattern.finditer(str(text or "")):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text) or text[start] != "{":
            continue
        raw = _balanced_json_object_at(text, start)
        if raw:
            candidates.append(raw)
    return candidates


def _guarded_fallback_diff_contains_exception(added_text: str, selected_value: str) -> bool:
    selected = str(selected_value or "").strip()
    if not selected:
        return False
    selected_simple = selected.rsplit(".", 1)[-1].lower()
    text = str(added_text or "").lower()
    for match in re.finditer(r"except\s+([^:\n]+):", text):
        clause = match.group(1).strip().split(" as ", 1)[0].strip()
        alternatives = [part.strip(" ()").rsplit(".", 1)[-1].lower() for part in clause.split(",")]
        if selected_simple in alternatives:
            return True
    return False


def _guarded_fallback_diff_contains_explicit_guard(added_text: str, selected_value: str) -> bool:
    if not re.search(r"(^|\n)\s*if\s+[^:\n]+:", str(added_text or "").lower()):
        return False
    selected_terms = [
        term.lower()
        for term in re.split(r"[^A-Za-z0-9_]+", str(selected_value or ""))
        if len(term) >= 4
    ]
    if not selected_terms:
        return True
    text = str(added_text or "").lower()
    return any(term in text for term in selected_terms)


def _guarded_fallback_has_narrow_exception_or_guard(added_text: str) -> bool:
    text = str(added_text or "").lower()
    if re.search(r"(^|\n)\s*if\s+[^:\n]+:", text) or re.search(r"\sif\s+[^:\n]+\s+else\s+", text):
        return True
    for match in re.finditer(r"except\s+([^:\n]+):", text):
        clause = match.group(1).strip()
        if not clause:
            continue
        clause = clause.split(" as ", 1)[0].strip()
        alternatives = [part.strip(" ()") for part in clause.split(",")]
        for candidate in alternatives:
            normalized = candidate.rsplit(".", 1)[-1].lower()
            if normalized in {"exception", "baseexception"}:
                continue
            if normalized.endswith(("error", "exception")):
                return True
    return False


def _paths_overlap(changed_files: list[str], target_paths: list[str]) -> bool:
    changed = {str(path).strip().lstrip("./") for path in changed_files if str(path).strip()}
    targets = {str(path).strip().lstrip("./") for path in target_paths if str(path).strip()}
    if not changed or not targets:
        return False
    return any(
        item in targets
        or any(item.startswith(f"{target}/") or target.startswith(f"{item}/") for target in targets)
        for item in changed
    )


def _diff_has_broad_exception_swallowing(diff_text: str) -> bool:
    added_lines = [
        line[1:].strip()
        for line in str(diff_text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return False
    broad_except = any(
        re.match(r"except\s*(Exception|BaseException)?\s*(?:as\s+\w+)?\s*:", line)
        or re.match(r"except\s*:\s*$", line)
        for line in added_lines
    )
    if not broad_except:
        return False
    swallowing_fallback = any(
        re.match(r"(return\s+(\{\}|\[\]|None|False|True|['\"]{2})|pass)\s*(#.*)?$", line)
        for line in added_lines
    )
    return swallowing_fallback


def _recovery_context_for_patch_contract(patch_contract: str) -> str:
    if patch_contract == "source_only":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": "mas42_clean_start_source_only_v1",
            "boundary": {
                "producer": "<STAGE:LOCATOR>",
                "consumer": "<STAGE:PATCHER>",
                "scope": "<SCOPE:CLEAN_START>",
            },
            "active_object": {
                "type": "<OBJ_TYPE:SOURCE_PATCH>",
                "id_hash": "",
            },
            "requirements": {
                "fresh_source_diff": "<REQ:TRUE>",
                "touch_suspect_path": "<REQ:FALSE>",
                "focused_validation": "<REQ:TRUE>",
                "max_fresh_source_files": 3,
            },
            "forbidden": {
                "path_classes": ["<PATH_CLASS:TEST>", "<PATH_CLASS:GENERATED>"],
            },
            "paths": {
                "suspect_paths": [],
            },
            "negative_constraints": ["<NEG:DO_NOT_EDIT_TESTS_TO_MAKE_MANIFEST_NODEIDS_PASS>"],
        }
        return (
            "Clean-start baseline source-only contract. Do not edit tests, generated files, "
            "or evaluation targets. Produce a minimal source-code diff and validate it.\n"
            f"{PROMPT_BEGIN}\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n{PROMPT_END}"
        )
    return ""


def _classify_patch_summary(patch_summary: dict[str, Any]) -> dict[str, Any]:
    source_files = list(patch_summary.get("non_test_files", []) or [])
    test_files = list(patch_summary.get("test_files", []) or [])
    changed_files = list(patch_summary.get("changed_files", []) or [])
    other_files: list[str] = []
    if not changed_files:
        patch_legitimacy = "no_diff"
    elif source_files and test_files:
        patch_legitimacy = "source_mixed"
    elif source_files:
        patch_legitimacy = "source_only"
    elif test_files:
        patch_legitimacy = "tests_only"
    else:
        patch_legitimacy = "non_source_only"
        other_files = changed_files
    return {
        "patch_legitimacy": patch_legitimacy,
        "changed_file_classes": {
            "source_files": source_files,
            "test_files": test_files,
            "other_files": other_files,
        },
    }


def _patch_contract_adhered(*, patch_contract: str, patch_legitimacy: str) -> bool:
    return not _patch_contract_violations(
        patch_contract=patch_contract,
        patch_legitimacy=patch_legitimacy,
    )


def _patch_contract_violations(*, patch_contract: str, patch_legitimacy: str) -> list[str]:
    if patch_contract != "source_only":
        return []
    if patch_legitimacy in {"source_mixed", "tests_only"}:
        return ["source_only_contract_test_files_changed"]
    if patch_legitimacy in {"no_diff", "non_source_only"}:
        return ["source_only_contract_no_source_diff"]
    return []


def _empty_harness_preflight_audit(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema": "masguard.clean_start_harness_preflight.v1",
        "enabled": bool(enabled),
        "status": "not_run" if enabled else "disabled",
        "blocks_execution": False,
        "reason": "",
        "timeout_seconds": 0,
        "docker_info": {"status": "not_checked"},
        "container_smoke": {"status": "not_configured"},
        "container_inventory": {
            "status": "not_checked",
            "same_instance_created_count": 0,
            "same_instance_created_containers": [],
            "running_bcmr_tail_count": 0,
            "running_bcmr_tail_containers": [],
            "created_bcmr_count": 0,
            "created_bcmr_containers": [],
            "parse_error_count": 0,
        },
        "claim_boundary": (
            "This is a cheap runtime preflight only. It executes no recovery, calls no models, "
            "and grants no recovery credit."
        ),
    }


def _harness_startup_preflight_audit(
    *,
    instance_id: str,
    timeout_seconds: int,
    container_smoke_image: str = "",
) -> dict[str, Any]:
    timeout = max(1, int(timeout_seconds or 1))
    audit = _empty_harness_preflight_audit(enabled=True)
    audit["timeout_seconds"] = timeout
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "docker_binary_missing",
                "docker_info": {"status": "missing_binary"},
            }
        )
        return audit
    except subprocess.TimeoutExpired as exc:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "docker_info_timeout",
                "docker_info": {
                    "status": "timeout",
                    "stdout_tail": _text_tail(exc.stdout, 500),
                    "stderr_tail": _text_tail(exc.stderr, 500),
                },
            }
        )
        return audit
    audit["docker_info"] = {
        "status": "info_ok" if info.returncode == 0 else "blocked",
        "returncode": int(info.returncode),
        "stdout_tail": _text_tail(info.stdout, 500),
        "stderr_tail": _text_tail(info.stderr, 500),
    }
    if info.returncode != 0:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "docker_info_failed",
            }
        )
        return audit

    try:
        ps = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{json .}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "docker_ps_timeout",
                "container_inventory": {
                    **audit["container_inventory"],
                    "status": "timeout",
                    "stdout_tail": _text_tail(exc.stdout, 500),
                    "stderr_tail": _text_tail(exc.stderr, 500),
                },
            }
        )
        return audit
    if ps.returncode != 0:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "docker_ps_failed",
                "container_inventory": {
                    **audit["container_inventory"],
                    "status": "blocked",
                    "returncode": int(ps.returncode),
                    "stdout_tail": _text_tail(ps.stdout, 500),
                    "stderr_tail": _text_tail(ps.stderr, 500),
                },
            }
        )
        return audit

    inventory = _harness_container_inventory(
        ps.stdout,
        instance_id=instance_id,
    )
    audit["container_inventory"] = inventory
    if inventory["same_instance_created_count"]:
        audit.update(
            {
                "status": "blocked",
                "blocks_execution": True,
                "reason": "same_instance_created_harness_container_present",
            }
        )
    elif container_smoke_image:
        smoke = _harness_container_smoke_audit(
            instance_id=instance_id,
            image=container_smoke_image,
            timeout_seconds=timeout,
        )
        audit["container_smoke"] = smoke
        if smoke["blocks_execution"]:
            audit.update(
                {
                    "status": "blocked",
                    "blocks_execution": True,
                    "reason": str(smoke.get("reason", "") or "container_smoke_failed"),
                }
            )
        else:
            audit.update({"status": "ok", "blocks_execution": False, "reason": ""})
    else:
        audit.update({"status": "ok", "blocks_execution": False, "reason": ""})
    return audit


def _harness_container_smoke_audit(
    *,
    instance_id: str,
    image: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    timeout = max(1, int(timeout_seconds or 1))
    name = f"masguard.cleanstart.preflight.{_safe_container_token(instance_id)}.{int(time.time())}"
    base = {
        "status": "not_run",
        "image": str(image or ""),
        "container_name": name,
        "timeout_seconds": timeout,
        "blocks_execution": False,
        "reason": "",
    }
    if not str(image or "").strip():
        return {**base, "status": "not_configured"}
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "blocked",
            "blocks_execution": True,
            "reason": "container_smoke_image_inspect_timeout",
            "stdout_tail": _text_tail(exc.stdout, 500),
            "stderr_tail": _text_tail(exc.stderr, 500),
        }
    if inspect.returncode != 0:
        return {
            **base,
            "status": "blocked",
            "blocks_execution": True,
            "reason": "container_smoke_image_missing",
            "returncode": int(inspect.returncode),
            "stdout_tail": _text_tail(inspect.stdout, 500),
            "stderr_tail": _text_tail(inspect.stderr, 500),
        }
    try:
        started_at = time.monotonic()
        run_proc = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                image,
                "tail",
                "-f",
                "/dev/null",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        cleanup = _docker_rm_force(name, timeout_seconds=timeout)
        post_timeout_cleanup = _docker_post_timeout_cleanup(
            name,
            timeout_seconds=timeout,
            prior_cleanup=cleanup,
        )
        return {
            **base,
            "status": "blocked",
            "blocks_execution": True,
            "reason": "container_smoke_run_timeout",
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "stdout_tail": _text_tail(exc.stdout, 500),
            "stderr_tail": _text_tail(exc.stderr, 500),
            "cleanup": cleanup,
            "post_timeout_cleanup": post_timeout_cleanup,
        }
    if run_proc.returncode != 0:
        return {
            **base,
            "status": "blocked",
            "blocks_execution": True,
            "reason": "container_smoke_run_failed",
            "returncode": int(run_proc.returncode),
            "stdout_tail": _text_tail(run_proc.stdout, 500),
            "stderr_tail": _text_tail(run_proc.stderr, 500),
        }
    cleanup = _docker_rm_force(name, timeout_seconds=timeout)
    return {
        **base,
        "status": "ok" if cleanup.get("returncode", 1) == 0 else "blocked",
        "blocks_execution": cleanup.get("returncode", 1) != 0,
        "reason": "" if cleanup.get("returncode", 1) == 0 else "container_smoke_cleanup_failed",
        "returncode": int(run_proc.returncode),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "stdout_tail": _text_tail(run_proc.stdout, 500),
        "stderr_tail": _text_tail(run_proc.stderr, 500),
        "cleanup": cleanup,
    }


def _docker_rm_force(container_name: str, *, timeout_seconds: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["docker", "rm", "-f", container_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(1, int(timeout_seconds or 1)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": 124,
            "stdout_tail": _text_tail(exc.stdout, 300),
            "stderr_tail": _text_tail(exc.stderr, 300),
        }
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "stdout_tail": _text_tail(proc.stdout, 300),
        "stderr_tail": _text_tail(proc.stderr, 300),
    }


def _docker_post_timeout_cleanup(
    container_name: str,
    *,
    timeout_seconds: int,
    prior_cleanup: dict[str, Any],
) -> dict[str, Any]:
    """Handle a Docker race where a timed-out run creates the container late."""
    inspect_timeout = max(1, min(5, int(timeout_seconds or 1)))
    try:
        inspect = subprocess.run(
            ["docker", "container", "inspect", container_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=inspect_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "inspect_timeout",
            "container_present_after_timeout": None,
            "prior_cleanup_returncode": int(prior_cleanup.get("returncode", 0) or 0),
            "stdout_tail": _text_tail(exc.stdout, 300),
            "stderr_tail": _text_tail(exc.stderr, 300),
        }
    if inspect.returncode != 0:
        return {
            "status": "not_present_after_timeout",
            "container_present_after_timeout": False,
            "prior_cleanup_returncode": int(prior_cleanup.get("returncode", 0) or 0),
            "stdout_tail": _text_tail(inspect.stdout, 300),
            "stderr_tail": _text_tail(inspect.stderr, 300),
        }
    retry_cleanup = _docker_rm_force(container_name, timeout_seconds=inspect_timeout)
    return {
        "status": "cleanup_retried_after_timeout",
        "container_present_after_timeout": True,
        "prior_cleanup_returncode": int(prior_cleanup.get("returncode", 0) or 0),
        "inspect_returncode": int(inspect.returncode),
        "inspect_stdout_tail": _text_tail(inspect.stdout, 300),
        "inspect_stderr_tail": _text_tail(inspect.stderr, 300),
        "retry_cleanup": retry_cleanup,
    }


def _safe_container_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").lower())
    token = token.strip(".-")
    return token[:48] or "unknown"


def _harness_container_inventory(ps_json_lines: str, *, instance_id: str) -> dict[str, Any]:
    prefix = f"bcmr.{str(instance_id or '').lower()}."
    same_instance_created: list[dict[str, Any]] = []
    running_bcmr_tail: list[dict[str, Any]] = []
    created_bcmr: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for raw in str(ps_json_lines or "").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        name = str(row.get("Names", "") or row.get("Name", "") or "")
        status = str(row.get("Status", "") or "")
        command = str(row.get("Command", "") or "")
        normalized = {
            "id": str(row.get("ID", "") or ""),
            "name": name,
            "status": status,
            "image": str(row.get("Image", "") or ""),
            "command": command,
        }
        lower_name = name.lower()
        lower_status = status.lower()
        lower_command = command.lower()
        if lower_name.startswith(prefix) and lower_status.startswith("created"):
            same_instance_created.append(normalized)
        if lower_name.startswith("bcmr.") and lower_status.startswith("created"):
            created_bcmr.append(normalized)
        if lower_name.startswith("bcmr.") and lower_status.startswith("up ") and "tail -f /dev/null" in lower_command:
            running_bcmr_tail.append(normalized)
    return {
        "status": "inventory_ok",
        "same_instance_created_count": len(same_instance_created),
        "same_instance_created_containers": same_instance_created[:8],
        "running_bcmr_tail_count": len(running_bcmr_tail),
        "running_bcmr_tail_containers": running_bcmr_tail[:8],
        "created_bcmr_count": len(created_bcmr),
        "created_bcmr_containers": created_bcmr[:8],
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
    }


def _text_tail(value: object, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-int(limit or 0) :] if limit else text


def _payload(
    *,
    instance_id: str,
    manifest_path: Path,
    execute_requested: bool,
    started_at: datetime | None = None,
    started_monotonic: float | None = None,
    row: dict[str, Any],
) -> dict[str, Any]:
    stop_reason = str(row.get("stop_reason", "") or "")
    runtime_fields = _runtime_duration_fields(started_at=started_at, started_monotonic=started_monotonic)
    return {
        "schema": "mas_dx_r_clean_start_baseline_v1",
        "rows": [
            {
                "source_type": "clean_start_baseline_execution" if execute_requested else "clean_start_baseline_dry_run",
                "instance_id": instance_id,
                "manifest_path": str(manifest_path),
                "baseline_role": "swe_repair_or_clean_restart",
                "baseline_name": "mas42_clean_start_freeform",
                "execute_requested": execute_requested,
                "error_type": str(row.get("error_type", "") or ""),
                "stop_reason": stop_reason,
                "oracle_success": bool(row.get("oracle_success", False)),
                "reported_success": bool(row.get("reported_success", False)),
                **row,
                **runtime_fields,
            }
        ],
        "summary": {
            "n_rows": 1,
            "status": stop_reason,
            "execute_requested": execute_requested,
            "baseline_name": "mas42_clean_start_freeform",
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
    parser.add_argument("--runtime", default="harness", choices=["local", "harness"])
    parser.add_argument("--workspace-root", default="outputs/mas_recovery/clean_start_mas42_workspaces")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-rebuild-harness", action="store_true")
    parser.add_argument("--harness-setup-timeout", type=int, default=None)
    parser.add_argument("--harness-container-start-timeout", type=int, default=None)
    parser.add_argument("--harness-container-cleanup-timeout", type=int, default=None)
    parser.add_argument(
        "--harness-preflight-before-execute",
        action="store_true",
        help=(
            "Before materializing/running a harness workspace, run a cheap Docker health and stale-container "
            "preflight. If it fails, record a runtime blocker with zero model calls instead of entering recovery."
        ),
    )
    parser.add_argument("--harness-preflight-timeout", type=int, default=10)
    parser.add_argument(
        "--harness-preflight-container-smoke-image",
        default="",
        help=(
            "Optional already-local image used by --harness-preflight-before-execute for a short detached "
            "docker-run smoke test. Leave empty to check docker info/ps only."
        ),
    )
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument(
        "--evaluation-command-source",
        default="manifest",
        choices=sorted(EVALUATION_COMMAND_SOURCES),
    )
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--locator-max-iterations", type=int, default=8)
    parser.add_argument("--planner-max-iterations", type=int, default=4)
    parser.add_argument("--patcher-max-iterations", type=int, default=8)
    parser.add_argument("--verifier-max-iterations", type=int, default=6)
    parser.add_argument("--patch-contract", default="none", choices=sorted(PATCH_CONTRACT_CHOICES))
    parser.add_argument(
        "--locator-failure-salvage",
        default="none",
        choices=sorted(LOCATOR_FAILURE_SALVAGE_CHOICES),
    )
    parser.add_argument(
        "--source-edit-contract",
        default="none",
        choices=sorted(SOURCE_EDIT_CONTRACT_CHOICES),
    )
    parser.add_argument(
        "--verifier-failure-feedback-retry",
        action="store_true",
        help=(
            "Enable one method-change retry after verifier failure by feeding the failed validation evidence "
            "back into the patcher. This is not part of the clean-start baseline credit."
        ),
    )
    parser.add_argument(
        "--bounded-validation-signal-retry",
        action="store_true",
        help=(
            "Enable one method-change retry only after a minimal fail-to-pass validation run produces "
            "structured source-repair feedback. This is a bounded-validation controller branch, not clean-start "
            "baseline credit."
        ),
    )
    parser.add_argument(
        "--failure-class-conditioned-retry",
        action="store_true",
        help=(
            "With --bounded-validation-signal-retry, classify the minimal-validation failure and condition the "
            "single retry on that class. This is a new audited method branch and does not alter legacy bounded "
            "validation retry results."
        ),
    )
    parser.add_argument(
        "--contract-verified-candidate-retry",
        action="store_true",
        help=(
            "With --bounded-validation-signal-retry, reject a candidate that violates the failure-class edit "
            "contract and allow one audited replacement candidate. This is a method-change probe, not clean-start "
            "baseline credit."
        ),
    )
    parser.add_argument(
        "--diverse-repair-hypothesis-retry",
        action="store_true",
        help=(
            "With --bounded-validation-signal-retry, attach a bounded multi-hypothesis repair context before "
            "the retry so the patcher must select one supported source-only mechanism before the contract gate. "
            "This is a method-change probe, not clean-start baseline credit."
        ),
    )
    parser.add_argument(
        "--candidate-pre-admission-syntax-guard",
        action="store_true",
        help=(
            "With --contract-verified-candidate-retry, require Python source candidates to pass a cheap syntax "
            "compile guard before final candidate admission. This guard calls no models and grants no recovery credit."
        ),
    )
    parser.add_argument(
        "--source-edit-pre-oracle-retry",
        action="store_true",
        help=(
            "Enable a default-off method-change branch that applies a deterministic source/syntax gate before "
            "oracle-adjacent evaluation, allows one constrained rewrite, and grants no credit by itself."
        ),
    )
    parser.add_argument(
        "--diff-first-or-abstain",
        action="store_true",
        help=(
            "Enable a default-off method-change branch that requires a structured MASGUARD_DIFF_INTENT marker "
            "or explicit MASGUARD abstention before final evaluation. This audits candidate generation quality "
            "and grants no credit by itself."
        ),
    )
    parser.add_argument(
        "--semantic-patch-correctness-v2",
        action="store_true",
        help=(
            "With --source-edit-pre-oracle-retry, require a provider-free semantic correctness gate before final "
            "evaluation: candidate output must declare the expected runtime effect, cover the target source span, "
            "and normalize target/protocol issues when signaled. Default off; no recovery credit by itself."
        ),
    )
    parser.add_argument(
        "--external-validation-feedback-artifact",
        default="",
        help=(
            "Optional previous raw artifact whose source-only diff plus external fail-to-pass/oracle failure "
            "signature is fed into a fresh source-only second-pass repair. This is a method-change probe, not "
            "clean-start baseline credit."
        ),
    )
    parser.add_argument(
        "--baseline-success-contract-artifact",
        default="",
        help=(
            "Optional successful strong-baseline raw artifact for the same instance. The runner extracts only "
            "its source-diff contract as a bounded transfer cue; it does not count the baseline success as "
            "MASGuard recovery credit."
        ),
    )
    parser.add_argument(
        "--historical-experience-artifact",
        default="",
        help=(
            "Optional different-instance successful source-only raw artifact used as bounded historical case "
            "memory. Same-instance artifacts are rejected; fresh oracle audit is still required."
        ),
    )
    parser.add_argument(
        "--historical-experience-mode",
        choices=sorted(HISTORICAL_EXPERIENCE_MODE_CHOICES),
        default="mas_conditioned",
        help=(
            "How to expose the historical success artifact. mas_conditioned is the MASGuard branch; "
            "generic_patch_memory is a comparator that intentionally omits MAS-conditioning credit."
        ),
    )
    parser.add_argument(
        "--mas-experience-controller-artifact",
        default="",
        help=(
            "Optional frozen dataset-expansion/controller artifact. The runner extracts only the pre-action "
            "MAS controller decision and excludes outcome fields; fresh execution is still required for credit."
        ),
    )
    parser.add_argument(
        "--historical-action-program-artifact",
        default="",
        help=(
            "Optional frozen no-leakage historical action-program artifact. The runner extracts only "
            "pre-action target-file patterns, edit invariants, validation templates, and early-stop rules; "
            "fresh execution is still required for credit."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = run_clean_start_baseline(
        instance_id=args.instance_id,
        manifest_root=Path(args.manifest_root),
        api_path=Path(args.api_path) if args.api_path else None,
        model_name=args.model,
        strong_model_name=args.strong_model,
        output_root=Path(args.workspace_root),
        runtime=args.runtime,
        execute=args.execute,
        force_rebuild_harness=args.force_rebuild_harness,
        harness_setup_timeout=args.harness_setup_timeout,
        harness_container_start_timeout=args.harness_container_start_timeout,
        harness_container_cleanup_timeout=args.harness_container_cleanup_timeout,
        harness_preflight_before_execute=args.harness_preflight_before_execute,
        harness_preflight_timeout=args.harness_preflight_timeout,
        harness_preflight_container_smoke_image=args.harness_preflight_container_smoke_image,
        command_timeout=args.command_timeout,
        evaluation_command_source=args.evaluation_command_source,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        locator_max_iterations=args.locator_max_iterations,
        planner_max_iterations=args.planner_max_iterations,
        patcher_max_iterations=args.patcher_max_iterations,
        verifier_max_iterations=args.verifier_max_iterations,
        patch_contract=args.patch_contract,
        locator_failure_salvage=args.locator_failure_salvage,
        source_edit_contract=args.source_edit_contract,
        verifier_failure_feedback_retry=args.verifier_failure_feedback_retry,
        bounded_validation_signal_retry=args.bounded_validation_signal_retry,
        failure_class_conditioned_retry=args.failure_class_conditioned_retry,
        contract_verified_candidate_retry=args.contract_verified_candidate_retry,
        external_validation_feedback_artifact=(
            Path(args.external_validation_feedback_artifact) if args.external_validation_feedback_artifact else None
        ),
        baseline_success_contract_artifact=(
            Path(args.baseline_success_contract_artifact) if args.baseline_success_contract_artifact else None
        ),
        historical_experience_artifact=(
            Path(args.historical_experience_artifact) if args.historical_experience_artifact else None
        ),
        historical_experience_mode=args.historical_experience_mode,
        mas_experience_controller_artifact=(
            Path(args.mas_experience_controller_artifact) if args.mas_experience_controller_artifact else None
        ),
        historical_action_program_artifact=(
            Path(args.historical_action_program_artifact) if args.historical_action_program_artifact else None
        ),
        diverse_repair_hypothesis_retry=args.diverse_repair_hypothesis_retry,
        candidate_pre_admission_syntax_guard=args.candidate_pre_admission_syntax_guard,
        source_edit_pre_oracle_retry=args.source_edit_pre_oracle_retry,
        diff_first_or_abstain=args.diff_first_or_abstain,
        semantic_patch_correctness_v2=args.semantic_patch_correctness_v2,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "instance_id": args.instance_id,
                "status": payload["summary"]["status"],
                "execute_requested": payload["summary"]["execute_requested"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
