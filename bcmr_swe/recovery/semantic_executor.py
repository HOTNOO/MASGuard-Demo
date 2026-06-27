"""Executor for the redesigned dual-layer semantic recovery language."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from bcmr_swe.recovery.action_guidance_gate import (
    audit_car_action_guidance_packet,
    build_car_action_guidance_packet,
)
from bcmr_swe.recovery.budgeted_controller import suggest_next_action
from bcmr_swe.recovery.car_controller import remember_car_episode, select_car_action
from bcmr_swe.recovery.episode_memory import CAR_METHOD_VERSION
from bcmr_swe.recovery.episode_memory import finalize_episodes as finalize_car_episodes
from bcmr_swe.recovery.object_lifecycle import apply_lifecycle_transition_to_ledger, lifecycle_summary
from bcmr_swe.recovery.patch_contract import (
    audit_stage_boundary_patch_contract,
    build_stage_boundary_patch_contract,
)
from bcmr_swe.recovery.patch_intent import audit_car_patch_intent, build_car_patch_intent
from bcmr_swe.recovery.program_executor import CoordinatorProtocol, ProgramExecutor
from bcmr_swe.recovery.semantic_language import (
    bootstrap_recovery_ledger,
    compile_semantic_program,
)
from bcmr_swe.recovery.stage_boundary_delta import (
    invalidation_delta_for_targets,
    object_chain_from_failed_state,
)
from bcmr_swe.recovery.structured_state import build_structured_recovery_state_from_failed_state
from bcmr_swe.types import (
    ActionObservation,
    FailedState,
    PrimitiveOpType,
    PrimitiveProgram,
    PrimitiveStep,
    ProgramOutcome,
    RecoveryBudget,
    RecoveryLedger,
    SemanticActionType,
    SemanticRecoveryProgram,
    SemanticRecoveryStep,
    StateDelta,
    StructuredRecoveryState,
)
from swe_mas.utils.command_classification import looks_like_validation_command
from swe_mas.utils.path_filters import classify_changed_files, normalize_repo_path


class SemanticProgramExecutor(ProgramExecutor):
    """Execute upper-layer semantic programs via lower-layer primitives."""

    def execute_semantic(
        self,
        coordinator: CoordinatorProtocol,
        program: SemanticRecoveryProgram,
        *,
        failed_state: FailedState,
        budget: RecoveryBudget,
        checkpoint_ids: dict[str, str] | None = None,
        closed_loop: bool = False,
        official_eval_commands: dict[str, str] | None = None,
    ) -> ProgramOutcome:
        started = time.time()
        usage_before = coordinator._usage_snapshot()
        progress_before = coordinator._progress_score()
        program_space_version = str(program.metadata.get("program_space_version", "") or "").strip()
        parc_controller_enforced = bool(program.metadata.get("parc_controller_enforced", False))
        car_enabled = bool(program.metadata.get("car_enabled", False))
        parc_contract_enabled = bool(program.metadata.get("parc_contract_enabled", False))
        closed_loop_enabled = bool(closed_loop) or program_space_version in {
            "semantic_closed_loop_v1",
            "semantic_action_loop_v1",
        }
        execution_mode = (
            program_space_version
            if program_space_version in {"semantic_closed_loop_v1", "semantic_action_loop_v1"}
            else ("semantic_closed_loop_v1" if closed_loop_enabled else "semantic_dual_v1")
        )
        closed_loop_unit = "fragment"
        if execution_mode == "semantic_action_loop_v1":
            closed_loop_unit = "action"
        elif closed_loop_enabled:
            closed_loop_unit = str(program.metadata.get("closed_loop_unit", "fragment") or "fragment")
        step_budget = max(1, len(program.steps))
        if closed_loop_enabled:
            step_budget = max(
                int(program.metadata.get("closed_loop_step_budget", 4) or 4),
                step_budget,
            )
        ledger = bootstrap_recovery_ledger(
            failed_state,
            budget,
            step_budget=step_budget,
        )
        ledger.metadata["car_enabled"] = car_enabled
        if car_enabled:
            ledger.metadata["car_method_version"] = str(
                program.metadata.get("car_method_version", CAR_METHOD_VERSION)
                or CAR_METHOD_VERSION
            )
            ledger.metadata["car_source_run_id"] = str(
                program.metadata.get("car_source_run_id", program.program_id)
                or program.program_id
            )
            ledger.metadata["car_output_episodes_path"] = str(
                program.metadata.get("car_output_episodes_path", "")
                or ""
            )
            ledger.metadata["car_prior_episodes"] = [
                dict(item)
                for item in list(program.metadata.get("car_prior_episodes", []) or [])
                if isinstance(item, dict)
            ]
            ledger.metadata["car_prior_episode_count"] = len(
                ledger.metadata["car_prior_episodes"]
            )
            ledger.metadata["car_cross_sample_prior_enabled"] = bool(
                program.metadata.get("car_cross_sample_prior_enabled", False)
            )
            ledger.metadata["car_fair_replay_policy"] = str(
                program.metadata.get("car_fair_replay_policy", "")
                or ""
            )
        ledger.metadata["parc_contract_enabled"] = parc_contract_enabled
        if parc_contract_enabled:
            ledger.metadata["parc_contract_profile"] = str(
                program.metadata.get("parc_contract_profile", "stage_boundary_patch_contract_v1")
                or "stage_boundary_patch_contract_v1"
            )
        ledger.metadata["initial_token_budget"] = float(budget.token_budget)
        ledger.metadata["initial_latency_budget_sec"] = float(budget.latency_budget_sec)
        ledger.metadata["expensive_replay_budget"] = int(
            program.metadata.get(
                "closed_loop_max_replays",
                program.metadata.get("closed_loop_max_fragments", 3),
            )
            or 3
        )
        ledger.metadata["unproductive_replay_budget"] = int(
            program.metadata.get(
                "closed_loop_max_unproductive_replays",
                program.metadata.get("closed_loop_max_fragments", 2),
            )
            or 2
        )
        ledger.metadata["initial_strong_stage_set"] = sorted(
            str(item)
            for item in set(getattr(coordinator, "_bcmr_strong_stage_set", set()) or set())
            if str(item).strip()
        )
        ledger_before = ledger.to_dict()
        semantic_trace: list[dict[str, Any]] = []
        primitive_outcomes: list[dict[str, Any]] = []
        actual_compiled_steps: list[PrimitiveStep] = []
        fragment_trace: list[dict[str, Any]] = []
        propagation_objects = object_chain_from_failed_state(failed_state)
        initial_step_budget = step_budget
        success = False
        budget_stop_reason = ""
        replay_completed = False
        official_success_deferred = False
        escalation_level = 0
        max_fragments = max(
            1,
            int(
                program.metadata.get(
                    "closed_loop_max_units",
                    program.metadata.get("closed_loop_max_fragments", 2),
                )
                or 2
            ),
        )
        if (parc_controller_enforced or car_enabled) and execution_mode == "semantic_action_loop_v1":
            parc_first = self._parc_controlled_first_action_program(
                program=program,
                ledger=ledger,
                checkpoint_ids=checkpoint_ids or {},
            )
            fragment_queue: list[SemanticRecoveryProgram] = [parc_first or program]
        else:
            fragment_queue = [program]
        fragment_index = 0

        while fragment_queue:
            current_fragment = fragment_queue.pop(0)
            if parc_controller_enforced and self._parc_recovery_goal_satisfied(ledger):
                ledger.metadata["parc_recovery_goal_stop"] = True
                break
            if self._hard_budget_exhausted(ledger):
                budget_stop_reason = self._budget_stop_reason(ledger)
                self._record_budget_event(
                    ledger,
                    reason=budget_stop_reason,
                    action="",
                    primitive_op="",
                    required_tokens=0.0,
                )
                break
            fragment_record = {
                "fragment_index": fragment_index,
                "program_id": current_fragment.program_id,
                "skeleton": current_fragment.skeleton,
                "start_remaining_step_budget": ledger.remaining_step_budget,
            }
            fragment_semantic_start = len(semantic_trace)
            for semantic_step in current_fragment.steps:
                semantic_started = time.time()
                semantic_ledger_before = ledger.to_dict()
                semantic_step_guard: dict[str, Any] = {}
                semantic_step_replay_completed = False
                semantic_step_official_success_deferred = False
                semantic_step_budget_stopped = False
                action_gate = self._budget_gate_for_action(semantic_step.action, ledger)
                if action_gate:
                    budget_stop_reason = str(action_gate["reason"])
                    self._record_budget_event(
                        ledger,
                        reason=budget_stop_reason,
                        action=semantic_step.action.value,
                        primitive_op="",
                        required_tokens=float(action_gate.get("required_tokens", 0.0) or 0.0),
                    )
                    semantic_step_guard = self._semantic_budget_stop_guard(
                        ledger=ledger,
                        reason=budget_stop_reason,
                        action=semantic_step.action.value,
                        primitive_op="",
                        required_tokens=float(action_gate.get("required_tokens", 0.0) or 0.0),
                    )
                    semantic_step_budget_stopped = True
                    compiled = PrimitiveProgram(
                        program_id=f"{current_fragment.program_id}__budget_stop",
                        steps=[],
                        rationale=current_fragment.rationale,
                        metadata=dict(current_fragment.metadata),
                    )
                    semantic_primitive_trace = [
                        {
                            "step_index": len(primitive_outcomes),
                            "semantic_step_index": len(semantic_trace),
                            "semantic_action": semantic_step.action.value,
                            "primitive_step_index": -1,
                            "op": "BUDGET_STOP",
                            "args": {
                                "action": semantic_step.action.value,
                                "reason": budget_stop_reason,
                            },
                            "result": semantic_step_guard,
                            "latency_sec": 0.001,
                            "token_cost": 0.0,
                            "ledger_after": ledger.to_dict(),
                        }
                    ]
                    primitive_outcomes.extend(semantic_primitive_trace)
                else:
                    compiled = compile_semantic_program(
                        SemanticRecoveryProgram(
                            program_id=f"{current_fragment.program_id}__s{len(semantic_trace) + 1}",
                            steps=[semantic_step],
                            rationale=current_fragment.rationale,
                            metadata=dict(current_fragment.metadata),
                        ),
                        failed_state=failed_state,
                        ledger=ledger,
                    )
                    semantic_primitive_trace = []
                    actual_compiled_steps.extend(compiled.steps)

                    for primitive_index, primitive_step in enumerate(compiled.steps):
                        primitive_gate = self._budget_gate_for_primitive(
                            primitive_step,
                            semantic_step.action,
                            ledger,
                        )
                        if primitive_gate:
                            budget_stop_reason = str(primitive_gate["reason"])
                            self._record_budget_event(
                                ledger,
                                reason=budget_stop_reason,
                                action=semantic_step.action.value,
                                primitive_op=primitive_step.op.value,
                                required_tokens=float(primitive_gate.get("required_tokens", 0.0) or 0.0),
                            )
                            semantic_step_guard = self._semantic_budget_stop_guard(
                                ledger=ledger,
                                reason=budget_stop_reason,
                                action=semantic_step.action.value,
                                primitive_op=primitive_step.op.value,
                                required_tokens=float(primitive_gate.get("required_tokens", 0.0) or 0.0),
                            )
                            primitive_entry = {
                                "step_index": len(primitive_outcomes),
                                "semantic_step_index": len(semantic_trace),
                                "semantic_action": semantic_step.action.value,
                                "primitive_step_index": primitive_index,
                                "op": "BUDGET_STOP",
                                "args": {
                                    "action": semantic_step.action.value,
                                    "blocked_primitive_op": primitive_step.op.value,
                                    "reason": budget_stop_reason,
                                },
                                "result": semantic_step_guard,
                                "latency_sec": 0.001,
                                "token_cost": 0.0,
                                "ledger_after": ledger.to_dict(),
                            }
                            primitive_outcomes.append(primitive_entry)
                            semantic_primitive_trace.append(primitive_entry)
                            semantic_step_budget_stopped = True
                            break

                        step_usage_before = coordinator._usage_snapshot()
                        step_started = time.time()
                        primitive_result, ledger, escalation_level = self._execute_primitive_step(
                            coordinator,
                            primitive_step,
                            ledger=ledger,
                            escalation_level=escalation_level,
                            semantic_action=semantic_step.action,
                            propagation_objects=propagation_objects,
                        )
                        step_usage_after = coordinator._usage_snapshot()
                        step_latency = max(0.001, time.time() - step_started)
                        step_tokens = max(
                            0.0,
                            float(step_usage_after.get("total_tokens", 0.0))
                            - float(step_usage_before.get("total_tokens", 0.0)),
                        )
                        if primitive_step.op == PrimitiveOpType.CONSTRAINED_REPLAY:
                            self._record_replay_cost_observation(
                                ledger=ledger,
                                primitive_step=primitive_step,
                                action=semantic_step.action,
                                token_cost=step_tokens,
                                latency_sec=step_latency,
                            )
                        primitive_entry = {
                            "step_index": len(primitive_outcomes),
                            "semantic_step_index": len(semantic_trace),
                            "semantic_action": semantic_step.action.value,
                            "primitive_step_index": primitive_index,
                            "op": primitive_step.op.value,
                            "args": dict(primitive_step.args),
                            "result": primitive_result,
                            "latency_sec": step_latency,
                            "token_cost": step_tokens,
                            "ledger_after": ledger.to_dict(),
                        }
                        primitive_outcomes.append(primitive_entry)
                        semantic_primitive_trace.append(primitive_entry)
                        ledger.remaining_token_budget = max(
                            0.0,
                            float(ledger.remaining_token_budget) - float(step_tokens),
                        )
                        ledger.remaining_latency_budget_sec = max(
                            0.0,
                            float(ledger.remaining_latency_budget_sec) - float(step_latency),
                        )

                        if primitive_step.op == PrimitiveOpType.CONSTRAINED_REPLAY:
                            semantic_step_replay_completed = bool(primitive_result.get("success", False))
                            semantic_step_official_success_deferred = bool(primitive_result.get("internal_verifier_skipped", False))
                            semantic_step_guard = dict(primitive_result.get("semantic_guard", {}) or {})
                            if (
                                closed_loop_enabled
                                and semantic_step_official_success_deferred
                                and semantic_step_replay_completed
                                and official_eval_commands
                            ):
                                eval_started = time.time()
                                semantic_step_guard = self._resolve_pending_guard_with_official_eval(
                                    coordinator=coordinator,
                                    guard=semantic_step_guard,
                                    official_eval_commands=official_eval_commands,
                                )
                                eval_latency = max(0.001, time.time() - eval_started)
                                ledger.remaining_latency_budget_sec = max(
                                    0.0,
                                    float(ledger.remaining_latency_budget_sec) - float(eval_latency),
                                )
                                primitive_result["semantic_guard"] = semantic_step_guard
                                primitive_result["closed_loop_official_eval_latency_sec"] = eval_latency
                                guard_history = list(ledger.metadata.get("guard_history", []) or [])
                                if guard_history:
                                    guard_history[-1] = dict(semantic_step_guard)
                                    ledger.metadata["guard_history"] = guard_history
                                ledger.last_source_edit_summary = {
                                    **dict(ledger.last_source_edit_summary or {}),
                                    "semantic_guard": dict(semantic_step_guard),
                                }
                                semantic_step_official_success_deferred = False
                            elif (
                                closed_loop_enabled
                                and official_eval_commands
                                and self._should_probe_official_after_source_diff(semantic_step_guard)
                            ):
                                eval_started = time.time()
                                semantic_step_guard = self._resolve_source_diff_candidate_with_focused_eval(
                                    coordinator=coordinator,
                                    guard=semantic_step_guard,
                                    official_eval_commands=official_eval_commands,
                                )
                                eval_latency = max(0.001, time.time() - eval_started)
                                ledger.remaining_latency_budget_sec = max(
                                    0.0,
                                    float(ledger.remaining_latency_budget_sec) - float(eval_latency),
                                )
                                primitive_result["semantic_guard"] = semantic_step_guard
                                primitive_result["closed_loop_official_eval_latency_sec"] = eval_latency
                                guard_history = list(ledger.metadata.get("guard_history", []) or [])
                                if guard_history:
                                    guard_history[-1] = dict(semantic_step_guard)
                                    ledger.metadata["guard_history"] = guard_history
                                ledger.last_source_edit_summary = {
                                    **dict(ledger.last_source_edit_summary or {}),
                                    "semantic_guard": dict(semantic_step_guard),
                                }
                                semantic_step_replay_completed = bool(
                                    semantic_step_guard.get("closed_loop_focused_eval_passed", False)
                                )
                                semantic_step_official_success_deferred = bool(
                                    semantic_step_guard.get("official_success_deferred", False)
                                )
                            self._record_replay_execution_contract(
                                ledger=ledger,
                                primitive_step=primitive_step,
                                action=semantic_step.action,
                                guard=semantic_step_guard,
                            )
                            self._remember_source_candidate(
                                coordinator=coordinator,
                                ledger=ledger,
                                semantic_step=semantic_step,
                                guard=semantic_step_guard,
                            )
                            if bool(ledger.metadata.get("car_enabled", False)):
                                provider_error_excerpt = self._provider_error_excerpt(primitive_result)
                                previous_provider_flag = bool(
                                    ledger.metadata.get("provider_error_observed", False)
                                )
                                if provider_error_excerpt:
                                    ledger.metadata["provider_error_observed"] = True
                                    provider_errors = [
                                        str(item)
                                        for item in list(
                                            ledger.metadata.get("provider_error_excerpts", [])
                                            or []
                                        )
                                        if str(item).strip()
                                    ]
                                    provider_errors.append(provider_error_excerpt[:500])
                                    ledger.metadata["provider_error_excerpts"] = provider_errors[-6:]
                                remember_car_episode(
                                    ledger,
                                    action_taken=semantic_step.action.value,
                                    result_mode=str(semantic_step_guard.get("result_mode", "") or ""),
                                    token_cost=step_tokens,
                                    latency_sec=step_latency,
                                    state_changed=bool(semantic_step_guard.get("fresh_changed_files", []))
                                    or bool(
                                        dict(
                                            semantic_step_guard.get("fresh_changed_file_classes", {})
                                            or {}
                                        ).get("source_files", [])
                                    ),
                                )
                                if provider_error_excerpt and not previous_provider_flag:
                                    ledger.metadata["provider_error_observed"] = False
                            if closed_loop_enabled:
                                success = str(semantic_step_guard.get("result_mode", "") or "") == "strong_source_success"
                            else:
                                success = semantic_step_replay_completed and not semantic_step_official_success_deferred

                        if not success and self._hard_budget_exhausted(ledger):
                            budget_stop_reason = self._budget_stop_reason(ledger)
                            self._record_budget_event(
                                ledger,
                                reason=budget_stop_reason,
                                action=semantic_step.action.value,
                                primitive_op=primitive_step.op.value,
                                required_tokens=0.0,
                            )
                            semantic_step_guard = self._annotate_guard_budget_stop(
                                dict(semantic_step_guard or {}),
                                ledger=ledger,
                                reason=budget_stop_reason,
                                action=semantic_step.action.value,
                                primitive_op=primitive_step.op.value,
                            )
                            primitive_entry["result"] = {
                                **dict(primitive_entry.get("result", {}) or {}),
                                "semantic_guard": dict(semantic_step_guard),
                            }
                            semantic_step_budget_stopped = True
                            break

                ledger.last_action = semantic_step.action.value
                semantic_action_observation = self._build_action_observation(
                    semantic_step=semantic_step,
                    ledger_before=RecoveryLedger.from_dict(semantic_ledger_before),
                    ledger_after=ledger,
                    primitive_trace=semantic_primitive_trace,
                    semantic_guard=semantic_step_guard,
                )
                semantic_state_delta = self._build_state_delta(
                    semantic_step=semantic_step,
                    ledger_before=RecoveryLedger.from_dict(semantic_ledger_before),
                    ledger_after=ledger,
                    primitive_trace=semantic_primitive_trace,
                    semantic_guard=semantic_step_guard,
                )
                self._apply_state_delta(
                    ledger=ledger,
                    state_delta=semantic_state_delta,
                    action_observation=semantic_action_observation,
                )
                if (
                    bool(ledger.metadata.get("car_enabled", False))
                    and semantic_step.action
                    not in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL, SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}
                    and not semantic_step_budget_stopped
                ):
                    remember_car_episode(
                        ledger,
                        action_taken=semantic_step.action.value,
                        result_mode=self._non_replay_episode_result_mode(
                            action_observation=semantic_action_observation,
                            state_delta=semantic_state_delta,
                        ),
                        token_cost=sum(
                            float(item.get("token_cost", 0.0) or 0.0)
                            for item in semantic_primitive_trace
                        ),
                        latency_sec=max(0.001, time.time() - semantic_started),
                        state_changed=self._non_replay_episode_state_changed(
                            action_observation=semantic_action_observation,
                            state_delta=semantic_state_delta,
                        ),
                    )
                if parc_controller_enforced and self._parc_recovery_goal_satisfied(ledger):
                    ledger.metadata["parc_recovery_goal_stop"] = True
                    semantic_step_guard = {
                        **dict(semantic_step_guard or {}),
                        "next_action_suggestion": "STOP",
                        "parc_recovery_goal_satisfied": True,
                    }
                ledger.last_action_result = {
                    "compiled_primitive_count": len(compiled.steps),
                    "semantic_step_latency_sec": max(0.001, time.time() - semantic_started),
                    "replay_completed_after_step": semantic_step_replay_completed,
                    "official_success_deferred": semantic_step_official_success_deferred,
                    "success_after_step": success,
                    "semantic_guard": semantic_step_guard,
                    "guard_flags": list(semantic_step_guard.get("guard_flags", []) or []),
                    "guard_next_action_suggestion": str(
                        semantic_step_guard.get("next_action_suggestion", "") or ""
                    ),
                    "action_observation": semantic_action_observation.to_dict(),
                    "state_delta": semantic_state_delta.to_dict(),
                }
                ledger.tried_actions.append(semantic_step.action.value)
                if not semantic_step_budget_stopped:
                    ledger.remaining_step_budget = max(0, ledger.remaining_step_budget - 1)

                semantic_trace.append(
                    {
                        "semantic_step_index": len(semantic_trace),
                        "fragment_index": fragment_index,
                        "action": semantic_step.action.value,
                        "args": dict(semantic_step.args),
                        "action_observation": semantic_action_observation.to_dict(),
                        "state_delta": semantic_state_delta.to_dict(),
                        "compiled_primitives": [step.to_dict() for step in compiled.steps],
                        "ledger_before": semantic_ledger_before,
                        "ledger_after": ledger.to_dict(),
                        "primitive_trace": semantic_primitive_trace,
                    }
                )
                if bool(ledger.metadata.get("parc_recovery_goal_stop", False)):
                    break
                if success or ledger.remaining_step_budget <= 0 or semantic_step_budget_stopped:
                    break

            fragment_record["end_remaining_step_budget"] = ledger.remaining_step_budget
            fragment_record["executed_semantic_steps"] = len(semantic_trace) - fragment_semantic_start
            executed_entries = semantic_trace[fragment_semantic_start:]
            fragment_record["executed_actions"] = [
                str(entry.get("action", "") or "")
                for entry in executed_entries
                if str(entry.get("action", "") or "")
            ]
            fragment_record["unit_name"] = (
                fragment_record["executed_actions"][0]
                if fragment_record["executed_actions"]
                else ""
            )
            fragment_guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
            fragment_record["guard_result_mode"] = str(fragment_guard.get("result_mode", "") or "")
            fragment_record["next_action_suggestion"] = str(
                fragment_guard.get("next_action_suggestion", "") or ""
            )
            fragment_record["last_guard_suggestion"] = str(
                ledger.last_action_result.get("guard_next_action_suggestion", "") or ""
            )
            fragment_trace.append(fragment_record)

            if not closed_loop_enabled or success or budget_stop_reason:
                break
            if parc_controller_enforced and self._parc_recovery_goal_satisfied(ledger):
                ledger.metadata["parc_recovery_goal_stop"] = True
                break
            if fragment_index + 1 >= max_fragments:
                break
            last_semantic_entry = semantic_trace[-1] if semantic_trace else None
            if closed_loop_unit == "action":
                next_fragment = self._next_closed_loop_action_program(
                    failed_state=failed_state,
                    ledger=ledger,
                    checkpoint_ids=checkpoint_ids or {},
                    last_semantic_entry=last_semantic_entry,
                )
            else:
                next_fragment = self._next_closed_loop_fragment(
                    failed_state=failed_state,
                    ledger=ledger,
                    checkpoint_ids=checkpoint_ids or {},
                )
            if next_fragment is None:
                break
            fragment_queue.append(next_fragment)
            fragment_index += 1

        usage_after = coordinator._usage_snapshot()
        token_cost = max(
            0.0,
            float(usage_after.get("total_tokens", 0.0))
            - float(usage_before.get("total_tokens", 0.0)),
        )
        progress_after = coordinator._progress_score()
        compiled_program = PrimitiveProgram(
            program_id=f"{program.program_id}__actual_compiled",
            steps=actual_compiled_steps,
            rationale=program.rationale,
            metadata={
                **dict(program.metadata),
                "semantic_program_id": program.program_id,
                "semantic_skeleton": program.skeleton,
                "compiler_version": execution_mode,
                "compile_mode": "dynamic_execution_trace",
            },
        )
        guard_summary = self._summarize_guard_history(ledger)
        if bool(ledger.metadata.get("car_enabled", False)):
            finalize_car_episodes(
                ledger,
                cell_succeeded=bool(success and guard_summary.get("clean_final_success", False)),
                provider_clean=not bool(ledger.metadata.get("provider_error_observed", False)),
                oracle_clean=not bool(ledger.metadata.get("oracle_infra_error", False)),
                infra_clean=not bool(ledger.metadata.get("substrate_error", False)),
                append_path=str(ledger.metadata.get("car_output_episodes_path", "") or "") or None,
            )
        return ProgramOutcome(
            program_id=program.program_id,
            recover_success=success,
            official_resolved=success,
            token_cost=token_cost,
            latency_sec=max(0.001, time.time() - started),
            secondary_risk=program.estimated_risk,
            milestone_gain=max(0.0, progress_after - progress_before),
            step_outcomes=primitive_outcomes,
            notes=f"Executed semantic program: {program.skeleton}",
            metadata={
                "execution_mode": execution_mode,
                "guard_mode": (
                    "semantic_action_loop_v1_ledger_guards"
                    if execution_mode == "semantic_action_loop_v1"
                    else (
                        "semantic_closed_loop_v1_ledger_guards"
                        if closed_loop_enabled
                        else "semantic_dual_v1_1_audit_guards"
                    )
                ),
                "loop_unit": closed_loop_unit,
                "guard_summary": guard_summary,
                "guarded_recover_success": bool(success and guard_summary.get("clean_final_success", False)),
                "budget_summary": {
                    "initial_step_budget": initial_step_budget,
                    "remaining_step_budget": ledger.remaining_step_budget,
                    "expensive_replay_budget": int(ledger.metadata.get("expensive_replay_budget", 0) or 0),
                    "remaining_expensive_replay_budget": self._remaining_replay_budget(ledger),
                    "expensive_replay_count": len(self._replay_history(ledger)),
                    "unproductive_replay_budget": int(ledger.metadata.get("unproductive_replay_budget", 0) or 0),
                    "unproductive_replay_count": self._unproductive_replay_count(ledger),
                    "initial_token_budget": float(budget.token_budget),
                    "remaining_token_budget": ledger.remaining_token_budget,
                    "spent_token_budget": token_cost,
                    "token_budget_overrun": max(0.0, token_cost - float(budget.token_budget)),
                    "initial_latency_budget_sec": float(budget.latency_budget_sec),
                    "remaining_latency_budget_sec": ledger.remaining_latency_budget_sec,
                    "spent_latency_budget_sec": max(0.001, time.time() - started),
                    "budget_exhausted": bool(
                        budget_stop_reason
                        or token_cost >= float(budget.token_budget)
                        or max(0.001, time.time() - started) >= float(budget.latency_budget_sec)
                    ),
                    "budget_stop_reason": budget_stop_reason,
                    "budget_events": list(ledger.metadata.get("budget_events", []) or []),
                },
                "semantic_program": program.to_dict(),
                "compiled_primitive_program": compiled_program.to_dict(),
                "semantic_trace": semantic_trace,
                "closed_loop_summary": {
                    "enabled": closed_loop_enabled,
                    "unit_granularity": closed_loop_unit,
                    "unit_count": len(fragment_trace),
                    "followup_unit_count": max(0, len(fragment_trace) - 1),
                    "fragment_count": len(fragment_trace),
                    "followup_fragment_count": max(0, len(fragment_trace) - 1),
                    "unit_trace": fragment_trace,
                    "fragment_trace": fragment_trace,
                },
                "ledger_before": ledger_before,
                "ledger_after": ledger.to_dict(),
                "family": str(program.metadata.get("family", "")),
                "scope": str(program.metadata.get("scope", "")),
                "strategy": str(program.metadata.get("strategy", program.program_id)),
            },
        )

    def _next_closed_loop_fragment(
        self,
        *,
        failed_state: FailedState,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
    ) -> SemanticRecoveryProgram | None:
        suggestion = str(
            ledger.last_action_result.get("guard_next_action_suggestion", "") or ""
        ).strip()
        if suggestion in {"", "STOP", "STOP_FOR_OFFICIAL_EVAL"}:
            return None

        post_locate_id = str(checkpoint_ids.get("post_locate", "") or "")
        initial_id = str(checkpoint_ids.get("initial", "") or "")
        if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            return SemanticRecoveryProgram(
                program_id="semantic_followup_evidence_local_repair",
                steps=[
                    SemanticRecoveryStep(
                        action=SemanticActionType.EVIDENCE_RECHECK,
                        args={"target": "test_output", "depth": "deep", "focused_verify": True},
                    ),
                    SemanticRecoveryStep(
                        action=SemanticActionType.LOCAL_REPAIR,
                        args={
                            "scope": "patcher+verifier",
                            "repair_mode": "local",
                            "respect_active_target": True,
                            "execution_profile": "normal",
                        },
                    ),
                ],
                metadata={"family": "semantic_evidence_recheck", "scope": "patcher+verifier"},
            )
        if suggestion == SemanticActionType.TARGET_RESET.value:
            return SemanticRecoveryProgram(
                program_id="semantic_followup_target_reset_local_repair",
                steps=[
                    SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    SemanticRecoveryStep(
                        action=SemanticActionType.LOCAL_REPAIR,
                        args={
                            "scope": "patcher+verifier",
                            "repair_mode": "rebuild_from_local_anchor",
                            "respect_active_target": True,
                            "execution_profile": "normal",
                        },
                    ),
                ],
                metadata={"family": "semantic_target_reset", "scope": "patcher+verifier"},
            )
        if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            current_level = 0
            for action in ledger.tried_actions:
                if action == SemanticActionType.CAPABILITY_BOOST.value:
                    current_level += 1
            return SemanticRecoveryProgram(
                program_id="semantic_followup_capability_boost_local_repair",
                steps=[
                    SemanticRecoveryStep(
                        action=SemanticActionType.CAPABILITY_BOOST,
                        args={
                            "scope": "patcher",
                            "strategy": "stronger_prompt",
                            "level": max(1, current_level + 1),
                        },
                    ),
                    SemanticRecoveryStep(
                        action=SemanticActionType.LOCAL_REPAIR,
                        args={
                            "scope": "patcher+verifier",
                            "repair_mode": "local_boosted",
                            "respect_active_target": True,
                        },
                    ),
                ],
                metadata={"family": "semantic_capability_boost", "scope": "patcher+verifier"},
            )
        if suggestion == SemanticActionType.SCOPE_EXPAND.value:
            return SemanticRecoveryProgram(
                program_id="semantic_followup_scope_expand",
                steps=[
                    SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["localized_path", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args=self._scope_expand_replay_args(ledger),
                    ),
                ],
                metadata={"family": "semantic_scope_expand", "scope": "locator+patcher+verifier"},
            )
        if initial_id:
            return SemanticRecoveryProgram(
                program_id="semantic_followup_global_reset",
                steps=[
                    SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "initial",
                            "checkpoint_id": initial_id,
                            "checkpoint_label": "initial",
                            "invalidate": ["all"],
                            "reset_active_target": True,
                        },
                    ),
                    SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args={
                            "scope": "full",
                            "strategy": "global_restart",
                            "expansion_level": "global",
                        },
                    ),
                ],
                metadata={"family": "semantic_global_reset", "scope": "full"},
            )
        return None

    def _parc_recovery_goal_satisfied(self, ledger: RecoveryLedger) -> bool:
        """Return true when C1 has driven all tracked objects to terminal states."""

        structured_state = dict(ledger.structured_state or {})
        if not structured_state:
            return False
        try:
            state = StructuredRecoveryState.from_dict(structured_state)
        except Exception:
            return False
        summary = lifecycle_summary(state)
        ledger.structured_state = state.to_dict()
        ledger.metadata["latest_lifecycle_summary"] = summary
        return bool(summary.get("terminal_objects")) and bool(summary.get("recovery_done"))

    def _next_closed_loop_action_program(
        self,
        *,
        failed_state: FailedState,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
        last_semantic_entry: dict[str, Any] | None,
    ) -> SemanticRecoveryProgram | None:
        if not last_semantic_entry:
            return None

        last_action = str(last_semantic_entry.get("action", "") or "").strip()
        last_args = dict(last_semantic_entry.get("args", {}) or {})
        suggestion = str(
            ledger.last_action_result.get("guard_next_action_suggestion", "") or ""
        ).strip()
        parc_forced_action = False

        if suggestion in {"STOP", "STOP_FOR_OFFICIAL_EVAL"}:
            return None

        post_locate_id = str(checkpoint_ids.get("post_locate", "") or "")

        if bool(ledger.metadata.get("car_enabled", False)):
            controller_decision = select_car_action(
                ledger,
                current_suggestion=suggestion,
            )
            if controller_decision is not None:
                ledger.metadata["latest_car_controller_decision"] = dict(
                    ledger.metadata.get("latest_car_controller_decision", {})
                    or controller_decision.to_dict()
                )
                if (
                    str(controller_decision.reason or "") == "recovery_goal_satisfied"
                    or bool(controller_decision.frontier.recovery_done)
                ):
                    ledger.metadata["parc_recovery_goal_stop"] = True
                    return None
                controller_action = str(controller_decision.action or "")
                if not controller_action and str(controller_decision.reason or "") == "no_viable_action":
                    ledger.metadata["budget_stop_reason"] = "car_no_viable_action"
                    ledger.metadata["car_no_viable_action_stop"] = True
                    return None
                if controller_action:
                    if controller_action != suggestion:
                        self._record_policy_redirect(
                            ledger=ledger,
                            from_suggestion=suggestion,
                            to_suggestion=controller_action,
                            last_action=last_action,
                            reason="car_controller_enforced",
                        )
                    suggestion = controller_action
                    parc_forced_action = True
                    if (
                        bool(ledger.metadata.get("car_enabled", False))
                        and suggestion
                        in {
                            SemanticActionType.LOCAL_REPAIR.value,
                            SemanticActionType.SCOPE_EXPAND.value,
                        }
                        and self._source_candidate_should_converge(ledger)
                        and (
                            str(self._best_source_candidate(ledger).get("result_mode", "") or "")
                            in {
                                "oracle_failed_after_source_edit",
                                "contract_violation_after_source_edit",
                                "intent_violation_missed_target",
                                "intent_violation_revoked_target",
                                "intent_violation_too_broad",
                                "intent_violation_after_source_edit",
                                "source_edit_pending_official",
                            }
                            or
                            str(self._latest_guard(ledger).get("result_mode", "") or "")
                            in {
                                "oracle_failed_after_source_edit",
                                "contract_violation_after_source_edit",
                                "intent_violation_missed_target",
                                "intent_violation_revoked_target",
                                "intent_violation_too_broad",
                                "intent_violation_after_source_edit",
                                "source_edit_pending_official",
                            }
                            or bool(self._validation_evidence_excerpt_from_guard(self._latest_guard(ledger)))
                            or str(self._latest_guard(ledger).get("closed_loop_focused_eval_status", "") or "")
                        )
                    ):
                        self._record_policy_redirect(
                            ledger=ledger,
                            from_suggestion=suggestion,
                            to_suggestion=SemanticActionType.LOCAL_REPAIR.value,
                            last_action=last_action,
                            reason="car_forced_action_converges_on_source_candidate",
                        )
                        return self._candidate_preserving_local_repair_program(
                            program_id="semantic_action_followup_car_candidate_convergence_refine",
                            repair_mode="candidate_preserving_car_convergence",
                            ledger=ledger,
                            execution_profile=self._candidate_refine_execution_profile(ledger),
                        )
                    if (
                        bool(ledger.metadata.get("car_enabled", False))
                        and suggestion == SemanticActionType.CAPABILITY_BOOST.value
                        and self._source_candidate_should_converge(ledger)
                    ):
                        self._record_policy_redirect(
                            ledger=ledger,
                            from_suggestion=suggestion,
                            to_suggestion=SemanticActionType.LOCAL_REPAIR.value,
                            last_action=last_action,
                            reason="car_capability_boost_compiles_candidate_refine",
                        )
                        return self._candidate_preserving_local_repair_program(
                            program_id="semantic_action_followup_car_boosted_candidate_refine",
                            repair_mode="candidate_preserving_boosted_refine",
                            ledger=ledger,
                            execution_profile="boosted",
                        )
                    forced_program = self._car_forced_action_program(
                        suggestion=suggestion,
                        ledger=ledger,
                        checkpoint_ids=checkpoint_ids,
                        post_locate_id=post_locate_id,
                    )
                    if forced_program is not None:
                        return forced_program
        elif bool(ledger.metadata.get("parc_controller_enforced", False)):
            controller_decision = suggest_next_action(
                ledger,
                current_suggestion=suggestion,
            )
            if controller_decision is not None:
                ledger.metadata["latest_parc_controller_decision"] = controller_decision.to_dict()
                if (
                    str(controller_decision.reason or "") == "recovery_goal_satisfied"
                    or bool(controller_decision.frontier.recovery_done)
                ):
                    ledger.metadata["parc_recovery_goal_stop"] = True
                    return None
                controller_action = str(controller_decision.action or "")
                if controller_action:
                    if controller_action != suggestion:
                        self._record_policy_redirect(
                            ledger=ledger,
                            from_suggestion=suggestion,
                            to_suggestion=controller_action,
                            last_action=last_action,
                            reason="parc_controller_enforced",
                        )
                    suggestion = controller_action
                    parc_forced_action = True
                    forced_program = self._car_forced_action_program(
                        suggestion=suggestion,
                        ledger=ledger,
                        checkpoint_ids=checkpoint_ids,
                        post_locate_id=post_locate_id,
                    )
                    if forced_program is not None:
                        return forced_program
        elif bool(ledger.metadata.get("parc_controller_hint_enabled", False)):
            controller_decision = suggest_next_action(
                ledger,
                current_suggestion=suggestion,
            )
            if controller_decision is not None:
                ledger.metadata["latest_parc_controller_decision"] = controller_decision.to_dict()
                controller_action = str(controller_decision.action or "")
                if (
                    controller_action
                    and controller_action != suggestion
                    and suggestion not in {"", "STOP", "STOP_FOR_OFFICIAL_EVAL"}
                ):
                    self._record_policy_redirect(
                        ledger=ledger,
                        from_suggestion=suggestion,
                        to_suggestion=controller_action,
                        last_action=last_action,
                        reason="parc_controller_hint_recorded",
                    )

        latest_guard = self._latest_guard(ledger)
        latest_mode = str(latest_guard.get("result_mode", "") or "")
        latest_flags = {
            str(item)
            for item in list(latest_guard.get("guard_flags", []) or [])
            if str(item).strip()
        }

        if (
            parc_forced_action
            and bool(ledger.metadata.get("car_enabled", False))
            and suggestion == SemanticActionType.LOCAL_REPAIR.value
            and self._source_candidate_should_converge(ledger)
            and latest_mode
            in {
                "oracle_failed_after_source_edit",
                "contract_violation_after_source_edit",
                "intent_violation_missed_target",
                "intent_violation_revoked_target",
                "intent_violation_too_broad",
                "intent_violation_after_source_edit",
                "source_edit_pending_official",
            }
        ):
            self._record_policy_redirect(
                ledger=ledger,
                from_suggestion=suggestion,
                to_suggestion=SemanticActionType.LOCAL_REPAIR.value,
                last_action=last_action,
                reason="car_source_candidate_convergence_refine",
            )
            return self._candidate_preserving_local_repair_program(
                program_id="semantic_action_followup_car_candidate_convergence_refine",
                repair_mode="candidate_preserving_car_convergence",
                ledger=ledger,
                execution_profile=self._candidate_refine_execution_profile(ledger),
            )

        if (
            not parc_forced_action
            and
            self._source_candidate_should_converge(ledger)
            and str(self._best_source_candidate(ledger).get("checkpoint_id", "") or "")
            and latest_mode in {
                "oracle_failed_after_source_edit",
                "intent_violation_missed_target",
                "intent_violation_revoked_target",
                "intent_violation_too_broad",
                "intent_violation_after_source_edit",
            }
            and (
                suggestion
                in {
                    SemanticActionType.CAPABILITY_BOOST.value,
                    SemanticActionType.SCOPE_EXPAND.value,
                    SemanticActionType.EVIDENCE_RECHECK.value,
                    "",
                }
                or "test_edit_present" in latest_flags
                or "source_edit_without_focused_validation" in latest_flags
                or self._validation_evidence_excerpt_from_guard(latest_guard)
            )
            and last_action
            in {
                SemanticActionType.LOCAL_REPAIR.value,
                SemanticActionType.REPAIR_LOCAL.value,
                SemanticActionType.SCOPE_EXPAND.value,
                SemanticActionType.EXPAND_SCOPE.value,
            }
        ):
            self._record_policy_redirect(
                ledger=ledger,
                from_suggestion=suggestion,
                to_suggestion=SemanticActionType.LOCAL_REPAIR.value,
                last_action=last_action,
                reason="candidate_preserving_refine_after_source_edit_failure",
            )
            profile = "boosted" if SemanticActionType.CAPABILITY_BOOST.value in list(ledger.tried_actions or []) else self._candidate_refine_execution_profile(ledger)
            return self._candidate_preserving_local_repair_program(
                program_id="semantic_action_followup_candidate_preserving_refine_after_source_edit_failure",
                repair_mode="candidate_preserving_refine",
                ledger=ledger,
                execution_profile=profile,
            )

        if (
            not parc_forced_action
            and
            latest_mode == "no_diff"
            and not self._has_source_candidate(ledger)
            and SemanticActionType.CAPABILITY_BOOST.value in list(ledger.tried_actions or [])
            and self._unproductive_replay_count(ledger) >= 1
            and last_action
            in {
                SemanticActionType.LOCAL_REPAIR.value,
                SemanticActionType.REPAIR_LOCAL.value,
                SemanticActionType.SCOPE_EXPAND.value,
                SemanticActionType.EXPAND_SCOPE.value,
            }
            and (
                "readonly_replay_exhausted" in latest_flags
                or "source_diff_unchanged_during_replay" in latest_flags
                or "stale_diff_removed_without_fresh_source_edit" in latest_flags
            )
        ):
            self._record_policy_redirect(
                ledger=ledger,
                from_suggestion=suggestion,
                to_suggestion="STOP",
                last_action=last_action,
                reason="stop_after_unproductive_no_diff_without_candidate",
            )
            ledger.metadata["budget_stop_reason"] = "unproductive_no_diff_without_candidate"
            return None

        if not parc_forced_action and self._is_repeated_failure_pattern(ledger):
            redirected = self._next_action_after_stagnation(
                ledger=ledger,
                current_suggestion=suggestion,
                last_action=last_action,
            )
            stagnation_events = list(ledger.metadata.get("stagnation_events", []) or [])
            stagnation_events.append(
                {
                    "last_action": last_action,
                    "from_suggestion": suggestion,
                    "redirected_suggestion": redirected,
                    "active_object_type": str(ledger.active_object_type or ""),
                    "active_target": str(ledger.active_target or ""),
                    "reason": "repeated_failure_pattern",
                    "policy": "ledger_adaptive_untried_action_v1",
                }
            )
            ledger.metadata["stagnation_events"] = stagnation_events
            suggestion = redirected
            if suggestion in {"", "STOP", "STOP_FOR_OFFICIAL_EVAL"}:
                return None

        if not parc_forced_action:
            if suggestion == SemanticActionType.SCOPE_EXPAND.value and self._source_candidate_should_converge(ledger):
                tried = list(ledger.tried_actions or [])
                candidate_replays = self._candidate_preserving_replay_count(ledger)
                if candidate_replays == 0:
                    if self._remaining_replay_budget(ledger) <= 1:
                        return self._candidate_preserving_local_repair_program(
                            program_id="semantic_action_followup_candidate_preserving_repair_low_replay_budget",
                            repair_mode="candidate_preserving_local",
                            ledger=ledger,
                            execution_profile=self._candidate_refine_execution_profile(ledger),
                        )
                    if SemanticActionType.CAPABILITY_BOOST.value not in tried:
                        suggestion = SemanticActionType.CAPABILITY_BOOST.value
                    elif SemanticActionType.EVIDENCE_RECHECK.value not in tried:
                        suggestion = SemanticActionType.EVIDENCE_RECHECK.value
                    else:
                        return self._candidate_preserving_local_repair_program(
                            program_id="semantic_action_followup_candidate_preserving_repair_after_source_candidate",
                            repair_mode="candidate_preserving_local",
                            ledger=ledger,
                            execution_profile=self._candidate_refine_execution_profile(ledger),
                        )
            elif suggestion == SemanticActionType.SCOPE_EXPAND.value and self._should_try_local_boost_before_expand(
                ledger=ledger,
                last_action=last_action,
            ):
                self._record_policy_redirect(
                    ledger=ledger,
                    from_suggestion=suggestion,
                    to_suggestion=SemanticActionType.CAPABILITY_BOOST.value,
                    last_action=last_action,
                    reason="try_local_capability_before_scope_expand",
                )
                suggestion = SemanticActionType.CAPABILITY_BOOST.value

            if suggestion == SemanticActionType.CAPABILITY_BOOST.value and not self._capability_boost_has_expected_value(
                ledger
            ):
                redirected = self._fallback_after_low_value_capability_boost(
                    ledger=ledger,
                    last_action=last_action,
                )
                self._record_policy_redirect(
                    ledger=ledger,
                    from_suggestion=suggestion,
                    to_suggestion=redirected,
                    last_action=last_action,
                    reason="skip_low_value_capability_boost",
                )
                suggestion = redirected
                if suggestion in {"", "STOP", "STOP_FOR_OFFICIAL_EVAL"}:
                    return None

        if ledger.remaining_step_budget == 1:
            compact = self._compact_final_budget_action_program(
                suggestion=suggestion,
                ledger=ledger,
                checkpoint_ids=checkpoint_ids,
                last_action=last_action,
            )
            if compact is not None:
                return compact

        if last_action == SemanticActionType.EVIDENCE_RECHECK.value:
            if suggestion == SemanticActionType.REVOKE_OBJECT.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_object_revoke_after_evidence",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.REVOKE_OBJECT,
                        args={
                            "checkpoint_id": str(checkpoint_ids.get("post_locate", "") or ""),
                            "checkpoint_label": "post_locate",
                            "anchor": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_object_revoke",
                    scope="object_reset",
                )
            if suggestion == SemanticActionType.TARGET_RESET.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_target_reset_after_evidence",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_target_reset",
                    scope="patcher+verifier",
                )
            if suggestion == SemanticActionType.SCOPE_EXPAND.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_scope_expand_after_evidence",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args=self._scope_expand_replay_args(ledger),
                    ),
                    family="semantic_scope_expand",
                    scope="locator+patcher+verifier",
                )
            return self._single_action_program(
                program_id="semantic_action_followup_local_repair_after_evidence",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "evidence_rechecked_local",
                        "respect_active_target": True,
                        "replay_precondition": "post_evidence_source_repair",
                        "require_focused_validation_after_edit": True,
                        "execution_profile": "compact",
                    },
                ),
                family="semantic_evidence_recheck",
                scope="patcher+verifier",
            )

        if last_action == SemanticActionType.RECHECK_OBJECT.value:
            if suggestion == SemanticActionType.REVOKE_OBJECT.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_object_revoke_after_recheck",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.REVOKE_OBJECT,
                        args={
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "anchor": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_object_revoke",
                    scope="object_reset",
                )
            if suggestion == SemanticActionType.TARGET_RESET.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_target_reset_after_recheck",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_target_reset",
                    scope="patcher+verifier",
                )
            if suggestion == SemanticActionType.SCOPE_EXPAND.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_scope_expand_after_recheck",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args=self._scope_expand_replay_args(ledger),
                    ),
                    family="semantic_scope_expand",
                    scope="locator+patcher+verifier",
                )
            return self._single_action_program(
                program_id="semantic_action_followup_local_repair_after_object_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "object_rechecked_local",
                        "respect_active_target": True,
                    },
                ),
                family="semantic_object_recheck",
                scope="patcher+verifier",
            )

        if last_action == SemanticActionType.REVOKE_OBJECT.value:
            last_guard = self._latest_guard(ledger)
            last_guard_suggestion = str(last_guard.get("next_action_suggestion", "") or "")
            if (
                not suggestion
                and last_guard_suggestion == SemanticActionType.REVOKE_OBJECT.value
                and self._revoked_active_object_type(ledger) == "shared_fact"
            ):
                return self._single_action_program(
                    program_id="semantic_action_followup_local_repair_after_shared_fact_revoke",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.LOCAL_REPAIR,
                        args={
                            "scope": "patcher+verifier",
                            "repair_mode": "object_revoked_local",
                            "respect_active_target": True,
                        },
                    ),
                    family="semantic_object_revoke",
                    scope="patcher+verifier",
                )
            if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_evidence_recheck_after_object_revoke",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.EVIDENCE_RECHECK,
                        args={"target": "test_output", "depth": "deep", "focused_verify": True},
                    ),
                    family="semantic_evidence_recheck",
                    scope="read+verify",
                )
            if suggestion == SemanticActionType.TARGET_RESET.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_target_reset_after_object_revoke",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_target_reset",
                    scope="patcher+verifier",
                )
            if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
                current_level = sum(
                    1
                    for action in ledger.tried_actions
                    if action == SemanticActionType.CAPABILITY_BOOST.value
                )
                return self._single_action_program(
                    program_id="semantic_action_followup_capability_boost_after_object_revoke",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.CAPABILITY_BOOST,
                        args={
                            "scope": "patcher",
                            "strategy": "stronger_prompt",
                            "level": max(1, current_level + 1),
                        },
                    ),
                    family="semantic_capability_boost",
                    scope="patcher",
                )
            if suggestion == SemanticActionType.SCOPE_EXPAND.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_scope_expand_after_object_revoke",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args=self._scope_expand_replay_args(ledger),
                    ),
                    family="semantic_scope_expand",
                    scope="locator+patcher+verifier",
                )
            return self._single_action_program(
                program_id="semantic_action_followup_local_repair_after_object_revoke",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "object_revoked_local",
                        "respect_active_target": True,
                    },
                ),
                family="semantic_object_revoke",
                scope="patcher+verifier",
            )

        if last_action == SemanticActionType.TARGET_RESET.value:
            anchor = str(last_args.get("anchor", "") or "")
            if anchor == "initial":
                return self._single_action_program(
                    program_id="semantic_action_followup_global_scope_expand",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args={
                            "scope": "full",
                            "strategy": "global_restart",
                            "expansion_level": "global",
                        },
                    ),
                    family="semantic_global_reset",
                    scope="full",
                )
            return self._single_action_program(
                program_id="semantic_action_followup_local_repair_after_reset",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "rebuild_from_local_anchor",
                        "respect_active_target": True,
                    },
                ),
                family="semantic_target_reset",
                scope="patcher+verifier",
            )

        if last_action == SemanticActionType.CAPABILITY_BOOST.value:
            if self._source_candidate_should_converge(ledger):
                return self._candidate_preserving_local_repair_program(
                    program_id="semantic_action_followup_candidate_preserving_repair_after_boost",
                    repair_mode="candidate_preserving_boosted_local",
                    ledger=ledger,
                    execution_profile=self._candidate_refine_execution_profile(ledger),
                )
            return self._single_action_program(
                program_id="semantic_action_followup_local_repair_after_boost",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "local_boosted",
                        "respect_active_target": True,
                        "execution_profile": ledger.execution_profile,
                    },
                ),
                family="semantic_capability_boost",
                scope="patcher+verifier",
            )

        if last_action in {SemanticActionType.LOCAL_REPAIR.value, SemanticActionType.REPAIR_LOCAL.value}:
            if suggestion == SemanticActionType.REVOKE_OBJECT.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_object_revoke_after_local_repair",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.REVOKE_OBJECT,
                        args={
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "anchor": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_object_revoke",
                    scope="object_reset",
                )
            if suggestion == SemanticActionType.TARGET_RESET.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_target_reset_after_local_repair",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.TARGET_RESET,
                        args={
                            "anchor": "post_locate",
                            "checkpoint_id": post_locate_id,
                            "checkpoint_label": "post_locate",
                            "invalidate": ["latest_patch", "current_target"],
                            "reset_active_target": True,
                        },
                    ),
                    family="semantic_target_reset",
                    scope="reset_only",
                )
            if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
                current_level = sum(
                    1
                    for action in ledger.tried_actions
                    if action == SemanticActionType.CAPABILITY_BOOST.value
                )
                return self._single_action_program(
                    program_id="semantic_action_followup_capability_boost_after_local_repair",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.CAPABILITY_BOOST,
                        args={
                            "scope": "patcher",
                            "strategy": "stronger_prompt",
                            "level": max(1, current_level + 1),
                        },
                    ),
                    family="semantic_capability_boost",
                    scope="patcher",
                )
            if suggestion == SemanticActionType.SCOPE_EXPAND.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_scope_expand_after_local_repair",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.SCOPE_EXPAND,
                        args=self._scope_expand_replay_args(ledger),
                    ),
                    family="semantic_scope_expand",
                    scope="locator+patcher+verifier",
                )
            if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
                return self._single_action_program(
                    program_id="semantic_action_followup_evidence_recheck_after_local_repair",
                    step=SemanticRecoveryStep(
                        action=SemanticActionType.EVIDENCE_RECHECK,
                        args={"target": "test_output", "depth": "deep", "focused_verify": True},
                    ),
                    family="semantic_evidence_recheck",
                    scope="read+verify",
                )

        if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            return self._single_action_program(
                program_id="semantic_action_followup_evidence_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.EVIDENCE_RECHECK,
                    args={"target": "test_output", "depth": "deep", "focused_verify": True},
                ),
                family="semantic_evidence_recheck",
                scope="read+verify",
            )
        if suggestion == SemanticActionType.RECHECK_OBJECT.value:
            return self._single_action_program(
                program_id="semantic_action_followup_object_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.RECHECK_OBJECT,
                    args={"depth": "deep", "focused_verify": True},
                ),
                family="semantic_object_recheck",
                scope="object+verify",
            )
        if suggestion == SemanticActionType.REVOKE_OBJECT.value:
            return self._single_action_program(
                program_id="semantic_action_followup_object_revoke",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_object_revoke",
                scope="object_reset",
            )
        if suggestion == SemanticActionType.TARGET_RESET.value:
            return self._single_action_program(
                program_id="semantic_action_followup_target_reset",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_target_reset",
                scope="reset_only",
            )
        if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            current_level = sum(
                1
                for action in ledger.tried_actions
                if action == SemanticActionType.CAPABILITY_BOOST.value
            )
            return self._single_action_program(
                program_id="semantic_action_followup_capability_boost",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.CAPABILITY_BOOST,
                    args={
                        "scope": "patcher",
                        "strategy": "stronger_prompt",
                        "level": max(1, current_level + 1),
                    },
                ),
                family="semantic_capability_boost",
                scope="patcher",
            )
        if suggestion == SemanticActionType.SCOPE_EXPAND.value:
            return self._single_action_program(
                program_id="semantic_action_followup_scope_expand",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.SCOPE_EXPAND,
                    args=self._scope_expand_replay_args(ledger),
                ),
                family="semantic_scope_expand",
                scope="locator+patcher+verifier",
            )
        return None

    def _car_forced_action_program(
        self,
        *,
        suggestion: str,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
        post_locate_id: str,
    ) -> SemanticRecoveryProgram | None:
        """Compile an enforced controller action without context fallback rewrites.

        CAR's method claim depends on typed actions actually reaching the
        executor.  Generic follow-up defaults may still map an empty or heuristic
        suggestion to a repair action, but an enforced CAR action is already a
        controller decision and should stay faithful to that action.
        """

        if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_evidence_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.EVIDENCE_RECHECK,
                    args={"target": "test_output", "depth": "deep", "focused_verify": True},
                ),
                family="semantic_evidence_recheck",
                scope="read+verify",
            )
        if suggestion == SemanticActionType.RECHECK_OBJECT.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_object_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.RECHECK_OBJECT,
                    args={"depth": "deep", "focused_verify": True},
                ),
                family="semantic_object_recheck",
                scope="object+verify",
            )
        if suggestion == SemanticActionType.REVOKE_OBJECT.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_object_revoke",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_object_revoke",
                scope="object_reset",
            )
        if suggestion == SemanticActionType.TARGET_RESET.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_target_reset",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_target_reset",
                scope="reset_only",
            )
        if suggestion == SemanticActionType.SCOPE_EXPAND.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_scope_expand",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.SCOPE_EXPAND,
                    args=self._scope_expand_replay_args(ledger),
                ),
                family="semantic_scope_expand",
                scope="locator+patcher+verifier",
            )
        if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            return self._single_action_program(
                program_id="semantic_action_followup_car_forced_capability_boost",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.CAPABILITY_BOOST,
                    args={
                        "scope": "patcher",
                        "strategy": "stronger_prompt",
                        "level": 1,
                    },
                ),
                family="semantic_capability_boost",
                scope="patcher",
            )
        return None

    def _parc_controlled_first_action_program(
        self,
        *,
        program: SemanticRecoveryProgram,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
    ) -> SemanticRecoveryProgram | None:
        if bool(program.metadata.get("car_enabled", False)):
            ledger.metadata["car_enabled"] = True
            decision = select_car_action(ledger)
        else:
            decision = suggest_next_action(ledger)
        if decision is None or not str(decision.action or ""):
            return None
        ledger.metadata["parc_controller_enforced"] = True
        ledger.metadata["car_enabled"] = bool(program.metadata.get("car_enabled", False))
        ledger.metadata["parc_controller_mode"] = str(
            program.metadata.get("parc_controller_mode", "") or "lifecycle_only"
        )
        if bool(program.metadata.get("car_enabled", False)):
            ledger.metadata["latest_car_controller_decision"] = dict(
                ledger.metadata.get("latest_car_controller_decision", {})
                or decision.to_dict()
            )
        else:
            ledger.metadata["latest_parc_controller_decision"] = decision.to_dict()
        action_value = str(decision.action or "")
        original = ""
        if program.steps:
            original = str(getattr(program.steps[0].action, "value", program.steps[0].action) or "")
        if original and original != action_value:
            self._record_policy_redirect(
                ledger=ledger,
                from_suggestion=original,
                to_suggestion=action_value,
                last_action="",
                reason=(
                    "car_controller_first_action_enforced"
                    if bool(program.metadata.get("car_enabled", False))
                    else "parc_controller_first_action_enforced"
                ),
            )
        return self._program_for_action_suggestion(
            suggestion=action_value,
            ledger=ledger,
            checkpoint_ids=checkpoint_ids,
            last_action="",
            program_id_prefix="semantic_action_parc_first",
        )

    def _program_for_action_suggestion(
        self,
        *,
        suggestion: str,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
        last_action: str,
        program_id_prefix: str,
    ) -> SemanticRecoveryProgram | None:
        post_locate_id = str(checkpoint_ids.get("post_locate", "") or "")
        if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            return self._single_action_program(
                program_id=f"{program_id_prefix}_evidence_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.EVIDENCE_RECHECK,
                    args={"target": "test_output", "depth": "deep", "focused_verify": True},
                ),
                family="semantic_evidence_recheck",
                scope="read+verify",
            )
        if suggestion == SemanticActionType.RECHECK_OBJECT.value:
            return self._single_action_program(
                program_id=f"{program_id_prefix}_object_recheck",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.RECHECK_OBJECT,
                    args={"depth": "deep", "focused_verify": True},
                ),
                family="semantic_object_recheck",
                scope="object+verify",
            )
        if suggestion == SemanticActionType.REVOKE_OBJECT.value:
            return self._single_action_program(
                program_id=f"{program_id_prefix}_object_revoke",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_object_revoke",
                scope="object_reset",
            )
        if suggestion == SemanticActionType.TARGET_RESET.value:
            return self._single_action_program(
                program_id=f"{program_id_prefix}_target_reset",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                family="semantic_target_reset",
                scope="reset_only",
            )
        if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            current_level = sum(
                1
                for action in ledger.tried_actions
                if action == SemanticActionType.CAPABILITY_BOOST.value
            )
            return self._single_action_program(
                program_id=f"{program_id_prefix}_capability_boost",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.CAPABILITY_BOOST,
                    args={
                        "scope": "patcher",
                        "strategy": "stronger_prompt",
                        "level": max(1, current_level + 1),
                    },
                ),
                family="semantic_capability_boost",
                scope="patcher",
            )
        if suggestion == SemanticActionType.SCOPE_EXPAND.value:
            return self._single_action_program(
                program_id=f"{program_id_prefix}_scope_expand",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.SCOPE_EXPAND,
                    args=self._scope_expand_replay_args(ledger),
                ),
                family="semantic_scope_expand",
                scope="locator+patcher+verifier",
            )
        if suggestion == SemanticActionType.LOCAL_REPAIR.value:
            repair_mode = "local"
            if last_action == SemanticActionType.REVOKE_OBJECT.value:
                repair_mode = "object_revoked_local"
            elif last_action == SemanticActionType.RECHECK_OBJECT.value:
                repair_mode = "object_rechecked_local"
            elif last_action == SemanticActionType.TARGET_RESET.value:
                repair_mode = "rebuild_from_local_anchor"
            elif last_action == SemanticActionType.CAPABILITY_BOOST.value:
                repair_mode = "local_boosted"
            return self._single_action_program(
                program_id=f"{program_id_prefix}_local_repair",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": repair_mode,
                        "respect_active_target": True,
                        "execution_profile": ledger.execution_profile,
                    },
                ),
                family="semantic_local_repair",
                scope="patcher+verifier",
            )
        return None

    @staticmethod
    def _single_action_program(
        *,
        program_id: str,
        step: SemanticRecoveryStep,
        family: str,
        scope: str,
        ledger: RecoveryLedger | None = None,
    ) -> SemanticRecoveryProgram:
        step_args = dict(step.args)
        if step.action in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL}:
            step_args.setdefault("execution_profile", "normal")
        if step.action in {SemanticActionType.REVOKE_OBJECT, SemanticActionType.TARGET_RESET} and bool(
            step_args.get("repair_after_clear", False)
        ):
            step_args.setdefault("execution_profile", "normal")
        normalized_step = SemanticRecoveryStep(action=step.action, args=step_args)
        return SemanticRecoveryProgram(
            program_id=program_id,
            steps=[normalized_step],
            metadata={
                "family": family,
                "scope": scope,
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": (
                    max(1, int(ledger.metadata.get("expensive_replay_budget", 3) or 3))
                    if ledger is not None
                    else 3
                ),
                "closed_loop_max_unproductive_replays": (
                    max(1, int(ledger.metadata.get("unproductive_replay_budget", 2) or 2))
                    if ledger is not None
                    else 2
                ),
            },
        )

    def _candidate_preserving_local_repair_program(
        self,
        *,
        program_id: str,
        repair_mode: str,
        ledger: RecoveryLedger,
        execution_profile: str = "normal",
    ) -> SemanticRecoveryProgram:
        candidate = self._best_source_candidate(ledger)
        candidate_files = list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or [])
        candidate_checkpoint_id = str(candidate.get("checkpoint_id", "") or "")
        args: dict[str, Any] = {
            "scope": "patcher+verifier",
            "repair_mode": repair_mode,
            "respect_active_target": True,
            "preserve_candidate_source_diff": True,
            "candidate_source_files": candidate_files[:6],
            "replay_precondition": "source_candidate_refine",
        }
        latest_guard = self._latest_guard(ledger)
        if self._validation_evidence_excerpt_from_guard(latest_guard):
            args["pre_evidence_recheck"] = True
            args["focused_verify"] = True
            args["target"] = "test_output"
            args["depth"] = "deep"
            args["refine_from_validation_failure"] = True
            args["require_focused_validation_after_edit"] = True
        if candidate_checkpoint_id:
            args["candidate_checkpoint_id"] = candidate_checkpoint_id
            args["candidate_checkpoint_label"] = str(candidate.get("checkpoint_label", "bcmr_source_candidate") or "")
        args["execution_profile"] = execution_profile or "normal"
        return self._single_action_program(
            program_id=program_id,
            step=SemanticRecoveryStep(
                action=SemanticActionType.LOCAL_REPAIR,
                args=args,
            ),
            family="semantic_candidate_preserving_repair",
            scope="patcher+verifier",
        )


    
    def _candidate_refine_execution_profile(ledger: RecoveryLedger) -> str:
        """Use a compact replay once CAR has a source candidate to refine."""

        if str(ledger.execution_profile or "").strip().lower() == "boosted":
            return "boosted"
        return "compact"

    def _scope_expand_replay_args(self, ledger: RecoveryLedger) -> dict[str, Any]:
        """Compile SCOPE_EXPAND into bounded relocalization plus repair.

        The mixed-project traces showed an important method failure: expansion
        sometimes became broad read-only probing.  This keeps expansion as a
        MAS-level action over locator+patcher+verifier, but feeds it the current
        failure evidence and a compact edit-and-validate profile.
        """

        args: dict[str, Any] = {
            "scope": "locator+patcher+verifier",
            "strategy": "broader_search",
            "expansion_level": "regional",
            "execution_profile": "compact",
            "replay_precondition": "evidence_bounded_scope_expand",
            "pre_evidence_recheck": True,
            "focused_verify": True,
            "target": "test_output",
            "depth": "deep",
        }
        candidate = self._best_source_candidate(ledger)
        if candidate:
            candidate_files = list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or [])
            args["preserve_candidate_source_diff"] = True
            args["candidate_source_files"] = [str(path) for path in candidate_files[:6] if str(path).strip()]
        if self._validation_evidence_excerpt_from_guard(self._latest_guard(ledger)):
            args["refine_from_validation_failure"] = True
            args["require_focused_validation_after_edit"] = True
        return args

    def _compact_final_budget_action_program(
        self,
        *,
        suggestion: str,
        ledger: RecoveryLedger,
        checkpoint_ids: dict[str, str],
        last_action: str,
    ) -> SemanticRecoveryProgram | None:
        """Fuse preparatory recovery actions with replay when only one action slot remains.

        Action-level closed-loop recovery normally spends one semantic step on
        object/state preparation and another on the downstream replay. When the
        final remaining slot would be a preparation-only action, the loop would
        stop immediately after making the ledger more correct but before testing
        the repair. This compact form preserves the same MAS recovery intent
        while making the last slot end in a replayable patcher/verifier action.
        """

        if suggestion in {"", "STOP", "STOP_FOR_OFFICIAL_EVAL"}:
            return None

        post_locate_id = str(checkpoint_ids.get("post_locate", "") or "")
        active_object_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
        object_id = str(ledger.active_object_id or "")
        car_enabled = bool(ledger.metadata.get("car_enabled", False))

        if suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            if car_enabled and not self._final_budget_should_force_expensive_replay(
                ledger,
                suggestion=suggestion,
                last_action=last_action,
            ):
                return self._program_for_action_suggestion(
                    suggestion=suggestion,
                    ledger=ledger,
                    checkpoint_ids=checkpoint_ids,
                    last_action=last_action,
                    program_id_prefix="semantic_action_followup_final_budget_car",
                )
            current_level = sum(
                1
                for action in ledger.tried_actions
                if action == SemanticActionType.CAPABILITY_BOOST.value
            )
            repair_mode = "local_boosted"
            if last_action == SemanticActionType.REVOKE_OBJECT.value:
                repair_mode = "object_revoked_boosted_local"
            elif last_action == SemanticActionType.TARGET_RESET.value:
                repair_mode = "rebuild_from_local_anchor"
            return self._single_action_program(
                program_id="semantic_action_followup_compact_boosted_repair_final_budget",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": repair_mode,
                        "respect_active_target": True,
                        "execution_profile": "boosted",
                        "boost_level": max(1, current_level + 1),
                    },
                ),
                family="semantic_capability_boost",
                scope="patcher+verifier",
            )

        if suggestion == SemanticActionType.REVOKE_OBJECT.value:
            return self._single_action_program(
                program_id="semantic_action_followup_compact_object_revoke_repair_final_budget",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                        "object_type": active_object_type or ledger.active_object_type,
                        "object_id": object_id,
                        "repair_after_clear": True,
                        "replay_scope": "patcher+verifier",
                        "repair_mode": "object_revoked_local",
                        "respect_active_target": True,
                        "execution_profile": "normal",
                    },
                ),
                family="semantic_object_revoke",
                scope="patcher+verifier",
            )

        if suggestion == SemanticActionType.TARGET_RESET.value:
            return self._single_action_program(
                program_id="semantic_action_followup_compact_target_reset_repair_final_budget",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                        "repair_after_clear": True,
                        "replay_scope": "patcher+verifier",
                        "repair_mode": "rebuild_from_local_anchor",
                        "respect_active_target": True,
                        "execution_profile": "normal",
                    },
                ),
                family="semantic_target_reset",
                scope="patcher+verifier",
            )

        if suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            if car_enabled and not self._final_budget_should_force_expensive_replay(
                ledger,
                suggestion=suggestion,
                last_action=last_action,
            ):
                return self._program_for_action_suggestion(
                    suggestion=suggestion,
                    ledger=ledger,
                    checkpoint_ids=checkpoint_ids,
                    last_action=last_action,
                    program_id_prefix="semantic_action_followup_final_budget_car",
                )
            return self._single_action_program(
                program_id="semantic_action_followup_compact_evidence_repair_final_budget",
                step=SemanticRecoveryStep(
                    action=SemanticActionType.LOCAL_REPAIR,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "evidence_rechecked_local",
                        "respect_active_target": True,
                        "pre_evidence_recheck": True,
                        "target": "test_output",
                        "depth": "deep",
                        "focused_verify": True,
                        "execution_profile": "normal",
                    },
                ),
                family="semantic_evidence_recheck",
                scope="patcher+verifier",
            )

        if (
            suggestion == SemanticActionType.SCOPE_EXPAND.value
            and self._has_source_candidate(ledger)
            and self._candidate_preserving_replay_count(ledger) == 0
        ):
            return self._candidate_preserving_local_repair_program(
                program_id="semantic_action_followup_compact_candidate_preserving_repair_final_budget",
                repair_mode="candidate_preserving_local",
                ledger=ledger,
            )

        return None

    @staticmethod
    def _final_budget_should_force_expensive_replay(
        ledger: RecoveryLedger,
        *,
        suggestion: str,
        last_action: str,
    ) -> bool:
        """Decide whether a final cheap CAR action may be fused into replay.

        CAR's mainline is a small action loop over structured state.  When the
        controller explicitly chooses a cheap evidence/capability action at the
        final slot, silently replacing it with another expensive replay can
        erase the budgeted-controller signal.  We only fuse when prior evidence
        says a source candidate exists and has not yet received a candidate
        preserving refinement.
        """

        if suggestion not in {
            SemanticActionType.EVIDENCE_RECHECK.value,
            SemanticActionType.CAPABILITY_BOOST.value,
        }:
            return True
        candidate = SemanticProgramExecutor._best_source_candidate(ledger)
        fresh_source_files = [
            str(item)
            for item in list(candidate.get("fresh_source_files", []) or [])
            if str(item).strip()
        ]
        candidate_mode = str(candidate.get("result_mode", "") or "")
        if fresh_source_files and candidate_mode in {
            "oracle_failed_after_source_edit",
            "source_edit_pending_official",
            "contract_violation_after_source_edit",
            "intent_violation_missed_target",
            "intent_violation_revoked_target",
            "intent_violation_too_broad",
            "intent_violation_after_source_edit",
        }:
            return SemanticProgramExecutor._candidate_preserving_replay_count(ledger) == 0
        latest_guard = SemanticProgramExecutor._latest_guard(ledger)
        latest_mode = str(latest_guard.get("result_mode", "") or "")
        if latest_mode in {
            "oracle_failed_after_source_edit",
            "contract_violation_after_source_edit",
            "intent_violation_missed_target",
            "intent_violation_revoked_target",
            "intent_violation_too_broad",
            "intent_violation_after_source_edit",
        }:
            return False
        return False

    @staticmethod
    def _is_repeated_failure_pattern(ledger: RecoveryLedger) -> bool:
        guards = [dict(item) for item in list(ledger.metadata.get("guard_history", []) or [])]
        if len(guards) < 2:
            return False
        last = guards[-1]
        prev = guards[-2]
        if str(last.get("result_mode", "") or "") != str(prev.get("result_mode", "") or ""):
            return False
        if str(last.get("failure_mode", "") or "") != str(prev.get("failure_mode", "") or ""):
            return False
        if str(last.get("patch_scope", "") or "") != str(prev.get("patch_scope", "") or ""):
            return False
        if bool(last.get("touches_suspect_path", False)) != bool(prev.get("touches_suspect_path", False)):
            return False
        last_files = sorted(str(item) for item in (last.get("changed_files", []) or []))
        prev_files = sorted(str(item) for item in (prev.get("changed_files", []) or []))
        last_set = set(last_files)
        prev_set = set(prev_files)
        if last_set != prev_set and not (last_set.issubset(prev_set) or prev_set.issubset(last_set)):
            return False
        return str(last.get("result_mode", "") or "") not in {
            "",
            "strong_source_success",
            "mixed_source_test_success",
            "checkpoint_or_no_diff_success",
        }

    @staticmethod
    def _has_source_candidate(ledger: RecoveryLedger) -> bool:
        return bool(SemanticProgramExecutor._best_source_candidate(ledger))

    @staticmethod
    def _source_candidate_should_converge(ledger: RecoveryLedger) -> bool:
        candidate = SemanticProgramExecutor._best_source_candidate(ledger)
        if not candidate:
            return False
        status = str(candidate.get("candidate_status", "") or "")
        if status in {
            "off_target",
            "validation_not_target_related",
            "validation_missing_result",
            "validation_pending",
        }:
            return False
        if SemanticProgramExecutor._candidate_preserving_replay_count(ledger) > 0:
            latest_mode = str(SemanticProgramExecutor._latest_guard(ledger).get("result_mode", "") or "")
            exhausted_modes = {
                "oracle_failed_after_source_edit",
                "contract_violation_after_source_edit",
                "intent_violation_after_source_edit",
                "intent_violation_missed_target",
                "intent_violation_revoked_target",
                "intent_violation_too_broad",
                "no_diff",
                "intent_violation_no_fresh_source",
                "contract_violation_no_fresh_source",
                "budget_exhausted",
            }
            if latest_mode in exhausted_modes:
                return False
        candidate_files = [
            str(item)
            for item in list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or [])
            if str(item).strip()
        ]
        if not candidate_files:
            return False
        suspect_paths = [
            str(item)
            for item in list(ledger.suspect_paths or [])
            if str(item).strip()
        ]
        if suspect_paths and not bool(candidate.get("touches_suspect_path", False)):
            return False
        return True

    @staticmethod
    def _candidate_preserving_replay_count(ledger: RecoveryLedger) -> int:
        count = 0
        for item in list(ledger.metadata.get("replay_cost_history", []) or []):
            if not isinstance(item, dict):
                continue
            repair_mode = str(item.get("repair_mode", "") or "")
            replay_precondition = str(item.get("replay_precondition", "") or "")
            if repair_mode.startswith("candidate_preserving") or replay_precondition == "source_candidate_refine":
                count += 1
        return count

    @staticmethod
    def _best_source_candidate(ledger: RecoveryLedger) -> dict[str, Any]:
        candidates = [
            dict(item)
            for item in list(ledger.metadata.get("source_candidate_memory", []) or [])
            if isinstance(item, dict)
        ]
        if not candidates:
            return {}

        def _score(candidate: dict[str, Any]) -> tuple[int, int, float]:
            mode = str(candidate.get("result_mode", "") or "")
            fresh_source_files = list(candidate.get("fresh_source_files", []) or [])
            touches = bool(candidate.get("touches_suspect_path", False))
            status = str(candidate.get("candidate_status", "") or "")
            mode_score = 0
            if mode == "source_edit_pending_official":
                mode_score = 6
            elif mode == "oracle_failed_after_source_edit":
                mode_score = 4
            elif mode == "source_edit_but_not_suspect":
                mode_score = 2
            if status in {
                "off_target",
                "validation_not_target_related",
                "validation_missing_result",
                "validation_pending",
            }:
                mode_score -= 3
            return (
                mode_score,
                1 if touches else 0,
                float(candidate.get("created_at", 0.0) or 0.0),
            )

        return max(candidates, key=_score)

    @staticmethod
    def _remember_source_candidate(
        *,
        coordinator: CoordinatorProtocol | None = None,
        ledger: RecoveryLedger,
        semantic_step: SemanticRecoveryStep,
        guard: dict[str, Any],
    ) -> None:
        fresh_classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
        changed_classes = dict(guard.get("changed_file_classes", {}) or {})
        fresh_source_files = [
            str(item)
            for item in list(fresh_classes.get("source_files", []) or [])
            if str(item).strip()
        ]
        source_files = [
            str(item)
            for item in list(changed_classes.get("source_files", []) or [])
            if str(item).strip()
        ]
        if not fresh_source_files and not source_files:
            return
        result_mode = str(guard.get("result_mode", "") or "")
        if result_mode in {"", "no_diff", "intent_violation_no_fresh_source", "wrong_edit_target", "budget_exhausted"}:
            return
        candidate = {
            "schema": "source_candidate_memory_v1",
            "created_at": time.time(),
            "after_action": semantic_step.action.value,
            "result_mode": result_mode,
            "success_legitimacy": str(guard.get("success_legitimacy", "") or ""),
            "fresh_source_files": fresh_source_files,
            "source_files": source_files,
            "fresh_changed_files": [
                str(item)
                for item in list(guard.get("fresh_changed_files", []) or [])
                if str(item).strip()
            ],
            "touches_suspect_path": bool(
                guard.get("decision_touches_suspect_path", guard.get("touches_suspect_path", False))
            ),
            "suspect_overlap": [
                str(item)
                for item in list(guard.get("fresh_suspect_path_overlap", []) or guard.get("suspect_path_overlap", []) or [])
                if str(item).strip()
            ],
            "guard_flags": [
                str(item)
                for item in list(guard.get("guard_flags", []) or [])
                if str(item).strip()
            ],
        }
        candidate["candidate_status"] = SemanticProgramExecutor._source_candidate_status_from_guard(
            ledger=ledger,
            guard=guard,
            candidate=candidate,
        )
        checkpoint = SemanticProgramExecutor._create_source_candidate_checkpoint(
            coordinator=coordinator,
            candidate=candidate,
        )
        if checkpoint:
            candidate.update(checkpoint)
        candidates = [
            dict(item)
            for item in list(ledger.metadata.get("source_candidate_memory", []) or [])
            if isinstance(item, dict)
        ]
        fingerprint = (
            tuple(candidate["fresh_source_files"] or candidate["source_files"]),
            candidate["result_mode"],
        )
        deduped = []
        for item in candidates:
            item_fingerprint = (
                tuple(list(item.get("fresh_source_files", []) or item.get("source_files", []) or [])),
                str(item.get("result_mode", "") or ""),
            )
            if item_fingerprint != fingerprint:
                deduped.append(item)
        deduped.append(candidate)
        ledger.metadata["source_candidate_memory"] = deduped[-6:]

    @staticmethod
    def _source_candidate_status_from_guard(
        *,
        ledger: RecoveryLedger,
        guard: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        if str(guard.get("closed_loop_focused_eval_status", "") or "") == "passed":
            return "focused_validated"
        if str(guard.get("closed_loop_focused_eval_status", "") or "") == "failed":
            return "focused_failed"
        flags = {
            str(item)
            for item in list(guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        focused_validation = dict(guard.get("focused_validation", {}) or {})
        if "focused_validation_not_target_related" in flags:
            return "validation_not_target_related"
        if (
            "focused_validation_missing_result" in flags
            or bool(focused_validation)
            and not bool(focused_validation.get("has_result", False))
        ):
            return "validation_missing_result"
        if "source_edit_without_focused_validation" in flags:
            return "validation_pending"
        if str(candidate.get("result_mode", "") or "") == "source_edit_pending_official":
            return "pending_official"
        suspect_paths = [
            str(item)
            for item in list(ledger.suspect_paths or [])
            if str(item).strip()
        ]
        if suspect_paths and not bool(candidate.get("touches_suspect_path", False)):
            return "off_target"
        if str(candidate.get("result_mode", "") or "") == "oracle_failed_after_source_edit":
            return "needs_refine"
        return "unresolved"

    @staticmethod
    def _create_source_candidate_checkpoint(
        *,
        coordinator: CoordinatorProtocol | None,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if coordinator is None:
            return {}
        recorder = getattr(coordinator, "recorder", None)
        create_checkpoint = getattr(recorder, "create_checkpoint", None)
        if not callable(create_checkpoint):
            return {}
        try:
            checkpoint = create_checkpoint(
                label="bcmr_source_candidate",
                metadata={
                    "resume_from": "patcher",
                    "stage": "bcmr_recovery",
                    "candidate_kind": "source_diff",
                    "result_mode": str(candidate.get("result_mode", "") or ""),
                    "source_files": list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or [])[:8],
                    "touches_suspect_path": bool(candidate.get("touches_suspect_path", False)),
                },
            )
        except Exception as exc:
            return {
                "checkpoint_error": str(exc)[:300],
            }
        return {
            "checkpoint_id": str(getattr(checkpoint, "checkpoint_id", "") or ""),
            "checkpoint_label": str(getattr(checkpoint, "label", "bcmr_source_candidate") or "bcmr_source_candidate"),
        }

    @staticmethod
    def _should_try_local_boost_before_expand(
        *,
        ledger: RecoveryLedger,
        last_action: str,
    ) -> bool:
        tried = list(ledger.tried_actions or [])
        if SemanticActionType.CAPABILITY_BOOST.value in tried:
            return False
        if int(ledger.remaining_step_budget or 0) <= 1:
            return False
        if last_action not in {
            SemanticActionType.LOCAL_REPAIR.value,
            SemanticActionType.REPAIR_LOCAL.value,
            SemanticActionType.RECHECK_OBJECT.value,
            SemanticActionType.REVOKE_OBJECT.value,
            SemanticActionType.TARGET_RESET.value,
        }:
            return False
        if SemanticProgramExecutor._ambiguous_local_repair_target(ledger):
            return False
        guard = SemanticProgramExecutor._latest_guard(ledger)
        flags = {
            str(item)
            for item in list(guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        mode = str(guard.get("result_mode", "") or "")
        if mode == "no_diff" and (
            "readonly_replay_exhausted" in flags
            or "stale_diff_removed_without_fresh_source_edit" in flags
            or "source_diff_unchanged_during_replay" in flags
            or "no_diff" in flags
        ):
            return True
        return False

    @staticmethod
    def _ambiguous_local_repair_target(ledger: RecoveryLedger) -> bool:
        source_suspects = [
            str(item)
            for item in list(ledger.suspect_paths or [])
            if str(item).strip()
        ]
        if len(source_suspects) >= 2:
            return True
        structured_state = dict(ledger.structured_state or {})
        object_chain = structured_state.get("object_chain_view") or []
        if isinstance(object_chain, dict):
            object_chain = object_chain.get("objects") or []
        selection_count = 0
        for item in list(object_chain or []):
            if not isinstance(item, dict):
                continue
            object_type = str(item.get("object_type", "") or "").strip().lower()
            if object_type in {"selection", "selectionobject"}:
                selection_count += 1
        return selection_count >= 2

    @staticmethod
    def _capability_boost_has_expected_value(ledger: RecoveryLedger) -> bool:
        """Whether CAPABILITY_BOOST can plausibly change the next replay.

        A boost is a MAS recovery action, not a magic retry token. It should
        only lead to another expensive replay when it changes model routing or
        a still-untried execution condition. If the batch already runs base and
        strong stages on the same model, repeated boost actions have low
        expected value and should yield to evidence/candidate refinement or
        scope changes instead.
        """

        tried = [str(item) for item in list(ledger.tried_actions or []) if str(item).strip()]
        if tried.count(SemanticActionType.CAPABILITY_BOOST.value) > 0:
            return False
        strong_stage_set = {
            str(item).strip().lower()
            for item in list(ledger.metadata.get("initial_strong_stage_set", []) or [])
            if str(item).strip()
        }
        if {"planner", "implementer"} & strong_stage_set:
            return True
        latest_guard = SemanticProgramExecutor._latest_guard(ledger)
        flags = {
            str(item)
            for item in list(latest_guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        return bool(
            "readonly_replay_exhausted" in flags
            or "source_diff_without_observed_write_command" in flags
        )

    @staticmethod
    def _fallback_after_low_value_capability_boost(
        *,
        ledger: RecoveryLedger,
        last_action: str,
    ) -> str:
        latest_guard = SemanticProgramExecutor._latest_guard(ledger)
        latest_mode = str(latest_guard.get("result_mode", "") or "")
        flags = {
            str(item)
            for item in list(latest_guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        if (
            SemanticProgramExecutor._has_source_candidate(ledger)
            and latest_mode == "oracle_failed_after_source_edit"
        ):
            return SemanticActionType.LOCAL_REPAIR.value
        if (
            "source_edit_without_focused_validation" in flags
            or SemanticProgramExecutor._validation_evidence_excerpt_from_guard(latest_guard)
        ) and SemanticActionType.EVIDENCE_RECHECK.value not in list(ledger.tried_actions or []):
            return SemanticActionType.EVIDENCE_RECHECK.value
        active_object_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
        if (
            active_object_type in {"shared_fact", "selection"}
            and SemanticActionType.TARGET_RESET.value not in list(ledger.tried_actions or [])
        ):
            return SemanticActionType.TARGET_RESET.value
        if last_action in {SemanticActionType.LOCAL_REPAIR.value, SemanticActionType.REPAIR_LOCAL.value}:
            return SemanticActionType.SCOPE_EXPAND.value
        return SemanticActionType.EVIDENCE_RECHECK.value

    @staticmethod
    def _record_policy_redirect(
        *,
        ledger: RecoveryLedger,
        from_suggestion: str,
        to_suggestion: str,
        last_action: str,
        reason: str,
    ) -> None:
        events = [
            dict(item)
            for item in list(ledger.metadata.get("action_policy_events", []) or [])
            if isinstance(item, dict)
        ]
        events.append(
            {
                "schema": "action_policy_redirect_v1",
                "from_suggestion": str(from_suggestion or ""),
                "to_suggestion": str(to_suggestion or ""),
                "last_action": str(last_action or ""),
                "reason": str(reason or ""),
            }
        )
        ledger.metadata["action_policy_events"] = events[-8:]

    @staticmethod
    def _latest_guard(ledger: RecoveryLedger) -> dict[str, Any]:
        guards = [dict(item) for item in list(ledger.metadata.get("guard_history", []) or [])]
        return guards[-1] if guards else {}

    @staticmethod
    def _hard_budget_exhausted(ledger: RecoveryLedger) -> bool:
        return float(ledger.remaining_token_budget or 0.0) <= 0.0 or float(
            ledger.remaining_latency_budget_sec or 0.0
        ) <= 0.0

    @staticmethod
    def _budget_stop_reason(ledger: RecoveryLedger) -> str:
        if float(ledger.remaining_token_budget or 0.0) <= 0.0:
            return "token_budget_exhausted"
        if float(ledger.remaining_latency_budget_sec or 0.0) <= 0.0:
            return "latency_budget_exhausted"
        return "budget_exhausted"

    @staticmethod
    def _semantic_action_min_token_reserve(action: SemanticActionType) -> float:
        """Conservative MAS-call reserve before starting another action.

        These are action-family costs, not benchmark-case rules. They keep a
        bounded recovery loop from silently turning into repeated full MAS
        retries under a fixed token envelope.
        """

        if action in {
            SemanticActionType.LOCAL_REPAIR,
            SemanticActionType.REPAIR_LOCAL,
        }:
            return 30000.0
        if action in {
            SemanticActionType.SCOPE_EXPAND,
            SemanticActionType.EXPAND_SCOPE,
        }:
            return 45000.0
        if action == SemanticActionType.CAPABILITY_BOOST:
            return 1000.0
        if action in {
            SemanticActionType.EVIDENCE_RECHECK,
            SemanticActionType.RECHECK_OBJECT,
        }:
            return 1000.0
        if action in {
            SemanticActionType.REVOKE_OBJECT,
            SemanticActionType.TARGET_RESET,
        }:
            return 500.0
        return 1000.0

    @classmethod
    def _semantic_action_min_token_reserve_for_ledger(
        cls,
        action: SemanticActionType,
        ledger: RecoveryLedger,
    ) -> float:
        return cls._semantic_action_min_token_reserve(action)

    def _budget_gate_for_action(
        self,
        action: SemanticActionType,
        ledger: RecoveryLedger,
    ) -> dict[str, Any]:
        if self._hard_budget_exhausted(ledger):
            return {"reason": self._budget_stop_reason(ledger), "required_tokens": 0.0}
        # Action selection itself is cheap: object revocation, target reset,
        # evidence bookkeeping, and capability routing should not be blocked by
        # the expected price of a later replay. The replay primitive has its
        # own gate below.
        required_tokens = 0.0
        remaining_tokens = float(ledger.remaining_token_budget or 0.0)
        if remaining_tokens < required_tokens:
            return {
                "reason": "insufficient_token_budget_for_next_action",
                "required_tokens": required_tokens,
                "remaining_token_budget": remaining_tokens,
            }
        return {}

    def _budget_gate_for_primitive(
        self,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
        ledger: RecoveryLedger,
    ) -> dict[str, Any]:
        if self._hard_budget_exhausted(ledger):
            return {"reason": self._budget_stop_reason(ledger), "required_tokens": 0.0}
        if primitive_step.op != PrimitiveOpType.CONSTRAINED_REPLAY:
            return {}
        replay_gate = self._replay_budget_gate(ledger=ledger, primitive_step=primitive_step, action=action)
        if replay_gate:
            return replay_gate
        return {}

    def _remaining_replay_budget(self, ledger: RecoveryLedger) -> int:
        max_replays = max(1, int(ledger.metadata.get("expensive_replay_budget", 3) or 3))
        return max(0, max_replays - len(self._replay_history(ledger)))

    @staticmethod
    def _replay_history(ledger: RecoveryLedger) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in list(ledger.metadata.get("replay_cost_history", []) or [])
            if isinstance(item, dict)
        ]

    @classmethod
    def _observed_replay_token_costs(cls, ledger: RecoveryLedger) -> list[float]:
        costs: list[float] = []
        for item in cls._replay_history(ledger):
            try:
                value = float(item.get("token_cost", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                costs.append(value)
        return costs

    @staticmethod
    def _scope_base_replay_cost(scope: str) -> float:
        normalized = str(scope or "").strip().lower()
        if normalized == "full":
            return 65000.0
        if "locator" in normalized:
            return 50000.0
        if "patcher" in normalized:
            return 25000.0
        if "verifier" in normalized:
            return 8000.0
        return 22000.0

    @classmethod
    def _estimated_next_replay_token_cost(
        cls,
        ledger: RecoveryLedger,
        *,
        action: SemanticActionType,
        scope: str,
        replay_precondition: str = "",
    ) -> float:
        initial_budget = float(
            dict(ledger.metadata or {}).get(
                "initial_token_budget",
                float(ledger.remaining_token_budget or 0.0),
            )
            or 0.0
        )
        observed = cls._observed_replay_token_costs(ledger)
        base = cls._scope_base_replay_cost(scope)
        precondition = str(replay_precondition or "").strip().lower()
        if precondition == "source_candidate_refine":
            base = min(base, 32000.0)
            if observed:
                observed_high = max(observed)
                return max(28000.0, min(42000.0, observed_high * 0.75))
        if precondition == "evidence_bounded_scope_expand":
            base = min(base, 38000.0)
            if observed:
                observed_high = max(observed)
                return max(30000.0, min(46000.0, observed_high * 0.85))
        if action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}:
            base = max(base, 50000.0)
        if observed:
            # Use the observed high-water mark: repeated MAS replays tend to be
            # at least as expensive as the last failed one because the ledger
            # carries extra reflection and constraints.
            observed_high = max(observed)
            if 0.0 < initial_budget < base:
                base = max(1000.0, observed_high * 1.25)
            else:
                base = max(base, observed_high)
        elif 0.0 < initial_budget < base:
            # Unit and diagnostic tests often use tiny synthetic budgets with
            # synthetic replay costs. Keep the gate meaningful without making the
            # production live budget policy impossible to test locally.
            base = max(1000.0, initial_budget * 0.1)
        return float(base)

    @classmethod
    def _unproductive_replay_count(cls, ledger: RecoveryLedger) -> int:
        count = 0
        for guard in list(ledger.metadata.get("guard_history", []) or []):
            if not isinstance(guard, dict):
                continue
            result_mode = str(guard.get("result_mode", "") or "")
            fresh_classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
            fresh_source_files = [
                str(item)
                for item in list(fresh_classes.get("source_files", []) or [])
                if str(item).strip()
            ]
            if result_mode in {
                "no_diff",
                "intent_violation_no_fresh_source",
                "wrong_edit_target",
                "budget_exhausted",
            } and not fresh_source_files:
                count += 1
        return count

    @classmethod
    def _conditioning_actions_since_last_replay(cls, ledger: RecoveryLedger) -> list[str]:
        """Cheap MAS-state actions that changed conditions after the last replay."""

        history = cls._replay_history(ledger)
        if not history:
            return []
        last = dict(history[-1])
        try:
            replay_action_count = int(last.get("tried_action_count_after_replay", -1))
        except (TypeError, ValueError):
            return []
        if replay_action_count < 0:
            return []
        tried = [str(item) for item in list(ledger.tried_actions or []) if str(item).strip()]
        after_replay = tried[min(len(tried), replay_action_count):]
        conditioning_actions = {
            SemanticActionType.CAPABILITY_BOOST.value,
            SemanticActionType.EVIDENCE_RECHECK.value,
            SemanticActionType.RECHECK_OBJECT.value,
            SemanticActionType.REVOKE_OBJECT.value,
            SemanticActionType.TARGET_RESET.value,
        }
        return [action for action in after_replay if action in conditioning_actions]

    def _replay_budget_gate(
        self,
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
    ) -> dict[str, Any]:
        scope = str(primitive_step.args.get("scope", "") or "")
        history = self._replay_history(ledger)
        max_replays = max(1, int(ledger.metadata.get("expensive_replay_budget", 3) or 3))
        fair_policy = str(ledger.metadata.get("car_fair_replay_policy", "") or "")
        if fair_policy == "single_expensive_attempt_v1" and history:
            return {
                "reason": "car_single_expensive_attempt_exhausted",
                "required_tokens": 0.0,
                "remaining_replay_budget": 0,
            }
        if fair_policy == "evidence_gated_retry_v1" and history and not self._car_retry_evidence_gate(
            ledger=ledger,
            primitive_step=primitive_step,
            action=action,
        ):
            return {
                "reason": "car_retry_requires_new_evidence",
                "required_tokens": 0.0,
                "remaining_replay_budget": max(0, max_replays - len(history)),
            }
        if len(history) >= max_replays:
            return {
                "reason": "expensive_replay_budget_exhausted",
                "required_tokens": 0.0,
                "remaining_replay_budget": 0,
            }

        signature_gate = self._replay_signature_gate(
            ledger=ledger,
            primitive_step=primitive_step,
            action=action,
        )
        if signature_gate:
            return {
                **signature_gate,
                "remaining_replay_budget": max(0, max_replays - len(history)),
            }

        max_unproductive = max(1, int(ledger.metadata.get("unproductive_replay_budget", 2) or 2))
        conditioned_retry = bool(self._conditioning_actions_since_last_replay(ledger))
        if (
            self._unproductive_replay_count(ledger) >= max_unproductive
            and not self._has_source_candidate(ledger)
            and not conditioned_retry
            and action in {
                SemanticActionType.LOCAL_REPAIR,
                SemanticActionType.REPAIR_LOCAL,
                SemanticActionType.SCOPE_EXPAND,
                SemanticActionType.EXPAND_SCOPE,
            }
        ):
            return {
                "reason": "unproductive_replay_budget_exhausted",
                "required_tokens": 0.0,
                "remaining_replay_budget": max(0, max_replays - len(history)),
            }

        scope_expand_already_replayed = any(
            str(item.get("action", "") or "")
            in {
                SemanticActionType.SCOPE_EXPAND.value,
                SemanticActionType.EXPAND_SCOPE.value,
            }
            for item in history
            if isinstance(item, dict)
        )
        first_structural_scope_expand = (
            action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}
            and not scope_expand_already_replayed
        )

        if (
            self._unproductive_replay_count(ledger) >= 1
            and not self._has_source_candidate(ledger)
            and not conditioned_retry
            and action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}
            and not first_structural_scope_expand
        ):
            latest_guard = self._latest_guard(ledger)
            flags = {
                str(item)
                for item in list(latest_guard.get("guard_flags", []) or [])
                if str(item).strip()
            }
            if (
                str(latest_guard.get("result_mode", "") or "") == "no_diff"
                and (
                    "readonly_replay_exhausted" in flags
                    or "source_diff_unchanged_during_replay" in flags
                    or "stale_diff_removed_without_fresh_source_edit" in flags
                )
            ):
                return {
                    "reason": "unproductive_no_diff_without_candidate",
                    "required_tokens": 0.0,
                    "remaining_replay_budget": max(0, max_replays - len(history)),
                }

        required_tokens = self._estimated_next_replay_token_cost(
            ledger,
            action=action,
            scope=scope,
            replay_precondition=str(primitive_step.args.get("replay_precondition", "") or ""),
        )
        remaining_tokens = float(ledger.remaining_token_budget or 0.0)
        if remaining_tokens < required_tokens:
            return {
                "reason": "insufficient_token_budget_for_replay",
                "required_tokens": required_tokens,
                "remaining_token_budget": remaining_tokens,
                "remaining_replay_budget": max(0, max_replays - len(history)),
            }
        return {}

    @classmethod
    def _replay_signature_gate(
        cls,
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
    ) -> dict[str, Any]:
        signature = cls._replay_execution_signature(
            ledger=ledger,
            primitive_step=primitive_step,
            action=action,
        )
        if not signature:
            return {}
        history = [
            dict(item)
            for item in list(ledger.metadata.get("replay_execution_contract_history", []) or [])
            if isinstance(item, dict)
        ]
        if not history:
            return {}
        current_sequence = cls._conditioning_sequence(ledger)
        for item in reversed(history):
            if str(item.get("signature", "") or "") != signature:
                continue
            if not bool(item.get("blocked_if_repeated", False)):
                continue
            if str(item.get("conditioning_sequence", "") or "") == current_sequence:
                return {
                    "reason": "replay_signature_repeated_without_new_evidence",
                    "required_tokens": 0.0,
                    "replay_signature": signature,
                    "prior_result_mode": str(item.get("result_mode", "") or ""),
                    "prior_action": str(item.get("action", "") or ""),
                }
        return {}

    @classmethod
    def _record_replay_execution_contract(
        cls,
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
        guard: dict[str, Any],
    ) -> None:
        signature = cls._replay_execution_signature(
            ledger=ledger,
            primitive_step=primitive_step,
            action=action,
        )
        if not signature:
            return
        result_mode = str(guard.get("result_mode", "") or "")
        fresh_classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
        fresh_source_files = [
            str(item)
            for item in list(fresh_classes.get("source_files", []) or [])
            if str(item).strip()
        ]
        flags = {
            str(item)
            for item in list(guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        blocked_if_repeated = (
            result_mode
            in {
                "no_diff",
                "contract_violation_no_fresh_source",
                "intent_violation_no_fresh_source",
                "wrong_edit_target",
                "budget_exhausted",
            }
            and not fresh_source_files
        ) or bool(
            flags
            & {
                "readonly_replay_exhausted",
                "source_diff_unchanged_during_replay",
                "stale_diff_removed_without_fresh_source_edit",
                "intent_no_fresh_source_diff",
                "contract_no_fresh_source_diff",
            }
        )
        event = {
            "schema": "replay_execution_contract_v1",
            "signature": signature,
            "action": action.value,
            "scope": str(primitive_step.args.get("scope", "") or ""),
            "repair_mode": str(primitive_step.args.get("repair_mode", "") or ""),
            "replay_precondition": str(primitive_step.args.get("replay_precondition", "") or ""),
            "target_paths": cls._replay_signature_target_paths(ledger, primitive_step=primitive_step),
            "result_mode": result_mode,
            "fresh_source_files": fresh_source_files[:8],
            "guard_flags": sorted(flags),
            "blocked_if_repeated": bool(blocked_if_repeated),
            "conditioning_sequence": cls._conditioning_sequence(ledger, include_action=action),
            "created_at": time.time(),
        }
        history = [
            dict(item)
            for item in list(ledger.metadata.get("replay_execution_contract_history", []) or [])
            if isinstance(item, dict)
        ]
        history.append(event)
        ledger.metadata["replay_execution_contract_history"] = history[-10:]
        ledger.metadata["latest_replay_execution_contract"] = event

    @classmethod
    def _replay_execution_signature(
        cls,
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
    ) -> str:
        action_value = action.value
        scope = str(primitive_step.args.get("scope", "") or "")
        repair_mode = str(primitive_step.args.get("repair_mode", "") or "")
        precondition = str(primitive_step.args.get("replay_precondition", "") or "")
        targets = cls._replay_signature_target_paths(ledger, primitive_step=primitive_step)
        payload = "|".join([action_value, scope, repair_mode, precondition, ",".join(targets)])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _replay_signature_target_paths(
        ledger: RecoveryLedger,
        *,
        primitive_step: PrimitiveStep | None = None,
    ) -> list[str]:
        values: list[str] = []
        primitive_args = dict(getattr(primitive_step, "args", {}) or {}) if primitive_step is not None else {}
        values.extend(
            str(item)
            for item in list(primitive_args.get("candidate_source_files", []) or [])
            if str(item).strip()
        )
        if ledger.active_target:
            values.append(str(ledger.active_target))
        values.extend(str(item) for item in list(ledger.suspect_paths or [])[:4] if str(item).strip())
        patch_intent = dict(ledger.metadata.get("latest_patch_intent", {}) or {})
        paths = dict(patch_intent.get("paths", {}) or {})
        for key in ("candidate_source_paths", "target_paths", "suspect_paths"):
            values.extend(str(item) for item in list(paths.get(key, []) or []) if str(item).strip())
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            normalized = normalize_repo_path(str(value))
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
        return out[:8]

    @staticmethod
    def _conditioning_sequence(
        ledger: RecoveryLedger,
        *,
        include_action: SemanticActionType | None = None,
    ) -> str:
        conditioning_actions = {
            SemanticActionType.EVIDENCE_RECHECK.value,
            SemanticActionType.RECHECK_OBJECT.value,
            SemanticActionType.REVOKE_OBJECT.value,
            SemanticActionType.TARGET_RESET.value,
            SemanticActionType.CAPABILITY_BOOST.value,
        }
        meaningful = [
            str(item)
            for item in list(ledger.tried_actions or [])
            if str(item) in conditioning_actions
        ]
        if include_action is not None and include_action.value in conditioning_actions:
            meaningful.append(include_action.value)
        return "|".join(meaningful[-6:])

    @classmethod
    def _car_retry_evidence_gate(
        cls,
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
    ) -> bool:
        """Allow a second CAR replay only after the run has new repair evidence.

        This keeps the comparison close to Reflexion's two attempts while
        preserving CAR's claim: the second expensive call must be justified by
        structured in-run evidence, not by an unconditional retry budget.
        """

        replay_precondition = str(primitive_step.args.get("replay_precondition", "") or "")
        if cls._source_candidate_should_converge(ledger):
            return True
        if replay_precondition == "source_candidate_refine":
            return cls._source_candidate_should_converge(ledger)
        if replay_precondition == "evidence_bounded_scope_expand":
            return True
        latest_guard = cls._latest_guard(ledger)
        latest_mode = str(latest_guard.get("result_mode", "") or "")
        flags = {
            str(item)
            for item in list(latest_guard.get("guard_flags", []) or [])
            if str(item).strip()
        }
        conditioned = cls._conditioning_actions_since_last_replay(ledger)
        if not conditioned:
            return False
        if action in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL}:
            meaningful = cls._meaningful_retry_conditioning_actions(ledger)
            if not meaningful:
                return False
        if action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}:
            return latest_mode in {
                "no_diff",
                "contract_violation_no_fresh_source",
                "intent_violation_no_fresh_source",
                "wrong_edit_target",
            } or bool(flags & {
                "readonly_replay_exhausted",
                "source_diff_unchanged_during_replay",
                "intent_no_fresh_source_diff",
                "contract_no_fresh_source_diff",
            })
        if action in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL}:
            return bool(
                {
                    SemanticActionType.EVIDENCE_RECHECK.value,
                    SemanticActionType.RECHECK_OBJECT.value,
                    SemanticActionType.TARGET_RESET.value,
                    SemanticActionType.REVOKE_OBJECT.value,
                }
                & set(conditioned)
            )
        return False

    @staticmethod
    def _meaningful_retry_conditioning_actions(ledger: RecoveryLedger) -> list[str]:
        events = [
            dict(item)
            for item in list(ledger.metadata.get("belief_revision_history", []) or [])
            if isinstance(item, dict)
        ]
        meaningful_revisions = {
            "observed_evidence",
            "belief_retargeted",
            "revoked",
            "reinforced_source_candidate",
            "candidate_preserved_unverified",
        }
        return [
            str(item.get("action", "") or "")
            for item in events
            if str(item.get("action", "") or "") in {
                SemanticActionType.EVIDENCE_RECHECK.value,
                SemanticActionType.RECHECK_OBJECT.value,
                SemanticActionType.REVOKE_OBJECT.value,
                SemanticActionType.TARGET_RESET.value,
            }
            and (
                str(item.get("revision_type", "") or "") in meaningful_revisions
                or bool(item.get("state_changed", False))
            )
        ]

    @staticmethod
    def _record_replay_cost_observation(
        *,
        ledger: RecoveryLedger,
        primitive_step: PrimitiveStep,
        action: SemanticActionType,
        token_cost: float,
        latency_sec: float,
    ) -> None:
        history = [
            dict(item)
            for item in list(ledger.metadata.get("replay_cost_history", []) or [])
            if isinstance(item, dict)
        ]
        history.append(
            {
                "schema": "replay_cost_observation_v1",
                "action": action.value,
                "scope": str(primitive_step.args.get("scope", "") or ""),
                "repair_mode": str(primitive_step.args.get("repair_mode", "") or ""),
                "execution_profile": str(primitive_step.args.get("execution_profile", "") or ""),
                "replay_precondition": str(primitive_step.args.get("replay_precondition", "") or ""),
                "token_cost": float(token_cost or 0.0),
                "latency_sec": float(latency_sec or 0.0),
                "remaining_token_budget_after": float(ledger.remaining_token_budget or 0.0),
                "tried_action_count_after_replay": len(list(ledger.tried_actions or [])) + 1,
                "created_at": time.time(),
            }
        )
        ledger.metadata["replay_cost_history"] = history[-8:]

    @staticmethod
    def _record_budget_event(
        ledger: RecoveryLedger,
        *,
        reason: str,
        action: str,
        primitive_op: str,
        required_tokens: float,
    ) -> None:
        event = {
            "reason": str(reason or "budget_exhausted"),
            "action": str(action or ""),
            "primitive_op": str(primitive_op or ""),
            "required_tokens": float(required_tokens or 0.0),
            "remaining_token_budget": float(ledger.remaining_token_budget or 0.0),
            "remaining_latency_budget_sec": float(ledger.remaining_latency_budget_sec or 0.0),
        }
        events = [
            dict(item)
            for item in list(ledger.metadata.get("budget_events", []) or [])
            if isinstance(item, dict)
        ]
        events.append(event)
        ledger.metadata["budget_events"] = events[-8:]
        ledger.metadata["budget_stop_reason"] = event["reason"]

    @staticmethod
    def _semantic_budget_stop_guard(
        *,
        ledger: RecoveryLedger,
        reason: str,
        action: str,
        primitive_op: str,
        required_tokens: float,
    ) -> dict[str, Any]:
        return {
            "patch_scope": "no_diff",
            "changed_files": [],
            "decision_patch_scope": "no_diff",
            "fresh_changed_files": [],
            "changed_file_classes": {
                "source_files": [],
                "test_files": [],
                "generated_files": [],
                "other_files": [],
                "effective_files": [],
            },
            "fresh_changed_file_classes": {
                "source_files": [],
                "test_files": [],
                "generated_files": [],
                "other_files": [],
                "effective_files": [],
            },
            "changed_file_class_counts": {
                "source_files": 0,
                "test_files": 0,
                "generated_files": 0,
                "other_files": 0,
                "effective_files": 0,
            },
            "suspect_paths": list(ledger.suspect_paths[:6]),
            "suspect_path_overlap": [],
            "fresh_suspect_path_overlap": [],
            "touches_suspect_path": False,
            "decision_touches_suspect_path": False,
            "active_target": str(ledger.active_target or ""),
            "active_object_type": str(ledger.active_object_type or ""),
            "active_object_id": str(ledger.active_object_id or ""),
            "replay_success": False,
            "official_success_deferred": False,
            "patcher_command_count": 0,
            "patcher_write_command_count": 0,
            "patcher_validation_command_count": 0,
            "replay_diff_changed": False,
            "result_mode": "budget_exhausted",
            "failure_mode": "budget_exhausted",
            "success_legitimacy": "not_resolved",
            "guard_flags": [
                "budget_exhausted",
                str(reason or "budget_exhausted"),
            ],
            "next_action_suggestion": "STOP",
            "budget_stop": {
                "reason": str(reason or "budget_exhausted"),
                "action": str(action or ""),
                "primitive_op": str(primitive_op or ""),
                "required_tokens": float(required_tokens or 0.0),
                "remaining_token_budget": float(ledger.remaining_token_budget or 0.0),
                "remaining_latency_budget_sec": float(ledger.remaining_latency_budget_sec or 0.0),
            },
        }

    def _annotate_guard_budget_stop(
        self,
        guard: dict[str, Any],
        *,
        ledger: RecoveryLedger,
        reason: str,
        action: str,
        primitive_op: str,
    ) -> dict[str, Any]:
        if not guard:
            return self._semantic_budget_stop_guard(
                ledger=ledger,
                reason=reason,
                action=action,
                primitive_op=primitive_op,
                required_tokens=0.0,
            )
        flags = [
            str(item)
            for item in list(guard.get("guard_flags", []) or [])
            if str(item).strip()
        ]
        for flag in ("budget_exhausted", str(reason or "budget_exhausted")):
            if flag and flag not in flags:
                flags.append(flag)
        guard["guard_flags"] = flags
        guard["next_action_suggestion"] = "STOP"
        guard["budget_stop"] = {
            "reason": str(reason or "budget_exhausted"),
            "action": str(action or ""),
            "primitive_op": str(primitive_op or ""),
            "remaining_token_budget": float(ledger.remaining_token_budget or 0.0),
            "remaining_latency_budget_sec": float(ledger.remaining_latency_budget_sec or 0.0),
        }
        return guard

    @staticmethod
    def _revoked_active_object_type(ledger: RecoveryLedger) -> str:
        revoked = dict(ledger.metadata.get("revoked_active_object", {}) or {})
        return str(revoked.get("object_type", "") or "").strip().lower().replace("-", "_")

    @staticmethod
    def _first_noninvalidated_object_id(
        propagation_objects: list[Any] | None,
        *,
        object_type: str,
        invalidated_ids: list[str],
    ) -> str:
        normalized_target = object_type.strip().lower().replace("-", "_")
        invalidated = {str(item) for item in list(invalidated_ids or []) if str(item).strip()}
        for item in list(propagation_objects or []):
            item_type = str(getattr(item, "object_type", "") or "").strip().lower().replace("-", "_")
            item_id = str(getattr(item, "object_id", "") or "").strip()
            if item_type == normalized_target and item_id and item_id not in invalidated:
                return item_id
        return ""

    @staticmethod
    def _next_action_after_stagnation(
        *,
        ledger: RecoveryLedger,
        current_suggestion: str,
        last_action: str = "",
    ) -> str:
        tried = list(ledger.tried_actions or [])
        if current_suggestion == SemanticActionType.CAPABILITY_BOOST.value:
            if tried.count(SemanticActionType.EVIDENCE_RECHECK.value) == 0:
                return SemanticActionType.EVIDENCE_RECHECK.value
            return SemanticActionType.SCOPE_EXPAND.value
        if current_suggestion == SemanticActionType.EVIDENCE_RECHECK.value:
            return SemanticActionType.TARGET_RESET.value
        if current_suggestion == SemanticActionType.TARGET_RESET.value:
            return SemanticActionType.SCOPE_EXPAND.value
        if current_suggestion == SemanticActionType.REVOKE_OBJECT.value:
            active_object_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
            if active_object_type == "shared_fact":
                if last_action in {
                    SemanticActionType.LOCAL_REPAIR.value,
                    SemanticActionType.REPAIR_LOCAL.value,
                }:
                    return SemanticActionType.SCOPE_EXPAND.value
                if SemanticActionType.LOCAL_REPAIR.value in tried or SemanticActionType.REPAIR_LOCAL.value in tried:
                    return SemanticActionType.SCOPE_EXPAND.value
                return SemanticActionType.LOCAL_REPAIR.value
            if active_object_type == "selection":
                return SemanticActionType.CAPABILITY_BOOST.value
            if active_object_type == "verifier_verdict":
                return SemanticActionType.EVIDENCE_RECHECK.value
            return SemanticActionType.EVIDENCE_RECHECK.value
        if current_suggestion == SemanticActionType.SCOPE_EXPAND.value:
            if last_action not in {
                SemanticActionType.SCOPE_EXPAND.value,
                SemanticActionType.EXPAND_SCOPE.value,
            }:
                return SemanticActionType.SCOPE_EXPAND.value
            guard = SemanticProgramExecutor._latest_guard(ledger)
            flags = {
                str(item)
                for item in list(guard.get("guard_flags", []) or [])
                if str(item).strip()
            }
            if (
                "readonly_replay_exhausted" in flags
                or "stale_diff_removed_without_fresh_source_edit" in flags
                or "no_diff" in flags
            ) and SemanticActionType.CAPABILITY_BOOST.value not in tried:
                return SemanticActionType.CAPABILITY_BOOST.value
            if (
                "source_diff_without_observed_write_command" in flags
                or "source_diff_unchanged_during_replay" in flags
            ) and SemanticActionType.CAPABILITY_BOOST.value not in tried:
                return SemanticActionType.CAPABILITY_BOOST.value
            if int(ledger.remaining_step_budget or 0) > 1:
                active_object_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
                if (
                    active_object_type in {"selection", "shared_fact"}
                    and SemanticActionType.TARGET_RESET.value not in tried
                ):
                    return SemanticActionType.TARGET_RESET.value
                if SemanticActionType.EVIDENCE_RECHECK.value not in tried:
                    return SemanticActionType.EVIDENCE_RECHECK.value
                if SemanticActionType.CAPABILITY_BOOST.value not in tried:
                    return SemanticActionType.CAPABILITY_BOOST.value
            return "STOP"
        return current_suggestion

    def _execute_primitive_step(
        self,
        coordinator: CoordinatorProtocol,
        primitive_step: PrimitiveStep,
        *,
        ledger: RecoveryLedger,
        escalation_level: int,
        semantic_action: SemanticActionType | None = None,
        propagation_objects: list[Any] | None = None,
    ) -> tuple[dict[str, Any], RecoveryLedger, int]:
        op = primitive_step.op
        args = dict(primitive_step.args)

        if op == PrimitiveOpType.READ_EVIDENCE:
            result = self._exec_inspect(
                coordinator,
                {"target": args.get("target", "test_output"), "depth": args.get("depth", "deep")},
            )
            findings = str(result.get("findings", "") or "").strip()
            if findings:
                ledger.key_evidence = [findings[:300]] + ledger.key_evidence[:2]
            return result, ledger, escalation_level

        if op == PrimitiveOpType.FOCUSED_VERIFY:
            result = self._run_workspace_tests(coordinator)
            output = str(result.get("output", "") or "").strip()
            if output:
                ledger.key_evidence = [output[:300]] + ledger.key_evidence[:2]
            return {
                "mode": str(args.get("mode", "failure_probe")),
                "output": output[:800],
                "returncode": result.get("returncode", -1),
            }, ledger, escalation_level

        if op == PrimitiveOpType.CLEAR_LOCAL_STATE:
            invalidate = [str(item) for item in args.get("invalidate", []) or []]
            object_id = str(args.get("object_id", "") or "").strip()
            object_type = str(args.get("object_type", "") or "").strip()
            reset_active_target = bool(args.get("reset_active_target", True))
            if "all" in invalidate:
                coordinator.shared_facts.clear()
                ledger.invalidated_targets = list(dict.fromkeys(ledger.invalidated_targets + ["all"]))
                if object_id:
                    ledger.invalidated_object_ids = list(dict.fromkeys(ledger.invalidated_object_ids + [object_id]))
                if reset_active_target:
                    ledger.active_target = ""
                    ledger.active_object_id = ""
                    ledger.active_object_type = ""
            else:
                for key in invalidate:
                    if key in {"latest_patch", "localized_path"}:
                        coordinator.shared_facts.pop(key, None)
                    ledger.invalidated_targets.append(key)
                ledger.invalidated_targets = list(dict.fromkeys(ledger.invalidated_targets))
                if object_id:
                    ledger.invalidated_object_ids = list(dict.fromkeys(ledger.invalidated_object_ids + [object_id]))
                if reset_active_target:
                    ledger.active_target = ""
                if object_type:
                    ledger.active_object_type = object_type
                if object_id:
                    ledger.active_object_id = object_id
                normalized_object_type = object_type.strip().lower().replace("-", "_")
                if normalized_object_type in {"shared_fact", "sharedfact"} and object_id:
                    ledger.metadata["revoked_active_object"] = {
                        "object_type": object_type,
                        "object_id": object_id,
                    }
                    ledger.active_object_type = "selection"
                    ledger.active_object_id = self._first_noninvalidated_object_id(
                        propagation_objects,
                        object_type="selection",
                        invalidated_ids=ledger.invalidated_object_ids,
                    )
                    ledger.active_object_excerpt = ledger.active_target
                    ledger.latest_shared_fact_key = ""
            delta = invalidation_delta_for_targets(
                list(propagation_objects or []),
                invalidated_targets=invalidate,
                invalidated_object_ids=[object_id] if object_id else [],
                active_object_id=object_id or ledger.active_object_id,
                include_all=("all" in invalidate),
            )
            if delta["invalidated_object_ids"]:
                ledger.invalidated_object_ids = list(
                    dict.fromkeys(
                        list(ledger.invalidated_object_ids)
                        + list(delta["invalidated_object_ids"])
                    )
                )
            return {
                "cleared": True,
                "invalidated": invalidate,
                "invalidated_object_id": object_id,
                "invalidated_object_type": object_type,
                **delta,
                "active_target_after_clear": ledger.active_target,
            }, ledger, escalation_level

        if op == PrimitiveOpType.ROLLBACK_ANCHOR:
            result = self._exec_rollback(
                coordinator,
                {
                    "checkpoint_id": args.get("checkpoint_id", ""),
                    "checkpoint_label": args.get("checkpoint_label", ""),
                },
            )
            anchor = str(args.get("anchor", "") or args.get("checkpoint_label", "") or "")
            if result.get("restored"):
                ledger.metadata["last_anchor"] = anchor
                if anchor != "initial" and ledger.active_target == "" and ledger.suspect_paths:
                    ledger.active_target = ledger.suspect_paths[0]
                    result["active_target_after_rollback"] = ledger.active_target
                if bool(ledger.metadata.get("car_enabled", False)):
                    constraints = list(ledger.negative_constraints or [])
                    constraints.append(
                        "CAR target reset completed; inspect focused evidence before replaying a local patch"
                    )
                    ledger.negative_constraints = list(dict.fromkeys(constraints))
            return result, ledger, escalation_level

        if op == PrimitiveOpType.BOOST_EXECUTION:
            level = int(args.get("level", 1))
            before_profile = str(ledger.execution_profile or "normal")
            result = self._exec_escalate(
                coordinator,
                {
                    "scope": args.get("scope", "patcher"),
                    "strategy": args.get("strategy", "stronger_prompt"),
                    "escalation_level": level,
                },
            )
            capability_changed = bool(result.get("capability_changed", False))
            if str(args.get("strategy", "") or "") == "stronger_prompt" and capability_changed:
                ledger.execution_profile = "boosted"
                constraints = list(ledger.negative_constraints or [])
                constraints.append(
                    "previous replay exhausted read-only probing; next repair must produce a fresh source diff and focused validation"
                )
                ledger.negative_constraints = list(dict.fromkeys(constraints))
            elif str(args.get("strategy", "") or "") == "stronger_prompt":
                ledger.execution_profile = before_profile
                result["capability_boost_noop"] = True
                result["capability_noop_reason"] = "no_distinct_strong_model_or_already_boosted"
            return result, ledger, max(escalation_level, level)

        if op == PrimitiveOpType.CONSTRAINED_REPLAY:
            scope = str(args.get("scope", "patcher+verifier"))
            execution_profile = str(args.get("execution_profile", "") or "").strip() or str(
                ledger.execution_profile or "normal"
            )
            if execution_profile == "compact" and str(ledger.execution_profile or "") == "boosted":
                execution_profile = "boosted"
            previous_precondition = str(ledger.metadata.get("current_replay_precondition", "") or "")
            replay_precondition = str(args.get("replay_precondition", "") or "")
            if replay_precondition:
                ledger.metadata["current_replay_precondition"] = replay_precondition
            replay_hint = self._compose_semantic_replay_hint(
                ledger=ledger,
                scope=scope,
                repair_mode=str(args.get("repair_mode", "local")),
                execution_profile=execution_profile,
                respect_active_target=bool(args.get("respect_active_target", True)),
            )
            patch_contract = None
            if bool(ledger.metadata.get("parc_contract_enabled", False)):
                patch_contract = build_stage_boundary_patch_contract(
                    ledger,
                    scope=scope,
                    repair_mode=str(args.get("repair_mode", "local")),
                    execution_profile=execution_profile,
                )
                replay_hint = f"{replay_hint}\n\n{patch_contract.to_prompt_text()}"
                patch_contract_history = [
                    dict(item)
                    for item in list(ledger.metadata.get("patch_contract_history", []) or [])
                    if isinstance(item, dict)
                ]
                patch_contract_history.append(patch_contract.to_dict())
                ledger.metadata["latest_patch_contract"] = patch_contract.to_dict()
                ledger.metadata["patch_contract_history"] = patch_contract_history[-6:]
            patch_intent = None
            action_guidance = None
            if bool(ledger.metadata.get("car_enabled", False)):
                action_guidance = build_car_action_guidance_packet(
                    ledger,
                    selected_action=str(
                        getattr(semantic_action, "value", semantic_action)
                        or ledger.last_action
                        or ""
                    ),
                    scope=scope,
                    repair_mode=str(args.get("repair_mode", "local")),
                    execution_profile=execution_profile,
                )
                replay_hint = f"{replay_hint}\n\n{action_guidance.to_prompt_text()}"
                action_guidance_history = [
                    dict(item)
                    for item in list(ledger.metadata.get("action_guidance_gate_history", []) or [])
                    if isinstance(item, dict)
                ]
                action_guidance_history.append(action_guidance.to_dict())
                ledger.metadata["latest_action_guidance_gate"] = action_guidance.to_dict()
                ledger.metadata["action_guidance_gate_history"] = action_guidance_history[-8:]
                patch_intent = build_car_patch_intent(
                    ledger,
                    selected_action=str(
                        getattr(semantic_action, "value", semantic_action)
                        or ledger.last_action
                        or ""
                    ),
                    scope=scope,
                    repair_mode=str(args.get("repair_mode", "local")),
                    execution_profile=execution_profile,
                )
                replay_hint = f"{replay_hint}\n\n{patch_intent.to_prompt_text()}"
                patch_intent_history = [
                    dict(item)
                    for item in list(ledger.metadata.get("patch_intent_history", []) or [])
                    if isinstance(item, dict)
                ]
                patch_intent_history.append(patch_intent.to_dict())
                ledger.metadata["latest_patch_intent"] = patch_intent.to_dict()
                ledger.metadata["patch_intent_history"] = patch_intent_history[-8:]
            if replay_precondition:
                replay_hint = f"{replay_hint}\nReplay precondition: {replay_precondition}"
            try:
                result = self._exec_replay(
                    coordinator,
                    {
                        "scope": scope,
                        "context_hint": replay_hint,
                        "execution_profile": execution_profile,
                    },
                    escalation_level=escalation_level,
                    context_hints=[],
                )
            finally:
                if previous_precondition:
                    ledger.metadata["current_replay_precondition"] = previous_precondition
                else:
                    ledger.metadata.pop("current_replay_precondition", None)
            patcher_trace = dict(result.get("patcher_trace", {}) or {})
            patch_summary = dict(patcher_trace.get("patcher_patch_summary") or {})
            ledger.last_source_edit_summary = patch_summary
            guard = self._build_semantic_guard(
                patch_summary=patch_summary,
                ledger=ledger,
                replay_result=result,
                patch_contract=patch_contract.to_dict() if patch_contract is not None else None,
                patch_intent=patch_intent.to_dict() if patch_intent is not None else None,
                action_guidance=action_guidance.to_dict() if action_guidance is not None else None,
            )
            budget_stop = dict(guard.get("budget_stop", {}) or {})
            if budget_stop:
                self._record_budget_event(
                    ledger,
                    reason=str(budget_stop.get("reason", "") or "budget_exhausted"),
                    action=str(ledger.last_action or ""),
                    primitive_op=PrimitiveOpType.CONSTRAINED_REPLAY.value,
                    required_tokens=0.0,
                )
                if str(budget_stop.get("reason", "") or "") == "token_budget_exhausted":
                    ledger.remaining_token_budget = 0.0
            ledger.last_source_edit_summary = {
                **patch_summary,
                "semantic_guard": guard,
            }
            evidence_excerpt = self._validation_evidence_excerpt_from_guard(guard)
            if evidence_excerpt:
                ledger.key_evidence = [evidence_excerpt[:300]] + ledger.key_evidence[:2]
            ledger.touches_suspect_path = bool(guard.get("touches_suspect_path", False))
            if ledger.active_target == "" and ledger.suspect_paths:
                ledger.active_target = ledger.suspect_paths[0]
            guard_history = list(ledger.metadata.get("guard_history", []) or [])
            guard_history.append(guard)
            ledger.metadata["guard_history"] = guard_history
            result["semantic_guard"] = guard
            return result, ledger, escalation_level

        return {"error": f"unknown primitive operator {op.value}"}, ledger, escalation_level

    @staticmethod
    def _non_replay_episode_state_changed(
        *,
        action_observation: ActionObservation,
        state_delta: StateDelta,
    ) -> bool:
        return bool(
            getattr(action_observation, "evidence_delta_kind", "")
            or action_observation.verifier_excerpt
            or action_observation.touched_paths
            or action_observation.invalidated_targets
            or action_observation.invalidated_object_ids
            or action_observation.negative_constraints
            or state_delta.added_invalidated_targets
            or state_delta.added_invalidated_object_ids
            or state_delta.added_negative_constraints
            or state_delta.latest_verifier_verdict
            or state_delta.latest_shared_fact_key
            or dict(state_delta.metadata or {}).get("lifecycle_transition")
        )

    @staticmethod
    def _non_replay_episode_result_mode(
        *,
        action_observation: ActionObservation,
        state_delta: StateDelta,
    ) -> str:
        if not SemanticProgramExecutor._non_replay_episode_state_changed(
            action_observation=action_observation,
            state_delta=state_delta,
        ):
            return "no_state_change"
        status = str(action_observation.status or "").strip()
        if status and status != "observed":
            return status
        action = str(getattr(action_observation, "action_type", "") or "")
        if action == SemanticActionType.REVOKE_OBJECT.value:
            return "belief_invalidated"
        if action == SemanticActionType.TARGET_RESET.value:
            return "anchor_restored"
        if action == SemanticActionType.EVIDENCE_RECHECK.value:
            return "evidence_observed"
        if action == SemanticActionType.RECHECK_OBJECT.value:
            return "object_rechecked"
        if action == SemanticActionType.CAPABILITY_BOOST.value:
            return "capability_changed"
        return "state_changed"

    def _build_action_observation(
        self,
        *,
        semantic_step: SemanticRecoveryStep,
        ledger_before: RecoveryLedger,
        ledger_after: RecoveryLedger,
        primitive_trace: list[dict[str, Any]],
        semantic_guard: dict[str, Any],
    ) -> ActionObservation:
        changed_files = [
            str(item)
            for item in list(semantic_guard.get("changed_files", []) or [])
            if str(item).strip()
        ]
        target_legitimacy = self._target_legitimacy_from_guard(semantic_guard)
        patch_legitimacy = str(
            semantic_guard.get("success_legitimacy", "")
            or semantic_guard.get("failure_mode", "")
            or semantic_guard.get("result_mode", "")
            or ""
        )
        replay_scope = next(
            (
                str(step.get("args", {}).get("scope", "") or "")
                for step in primitive_trace
                if str(step.get("op", "") or "") == PrimitiveOpType.CONSTRAINED_REPLAY.value
            ),
            "",
        )
        verifier_excerpt = next(
            (
                str((step.get("result", {}) or {}).get("output", "") or "")[:800]
                for step in primitive_trace
                if str(step.get("op", "") or "") == PrimitiveOpType.FOCUSED_VERIFY.value
            ),
            "",
        )
        if not verifier_excerpt:
            verifier_excerpt = str(
                semantic_guard.get("closed_loop_fail_to_pass_output_excerpt", "")
                or semantic_guard.get("closed_loop_oracle_output_excerpt", "")
                or semantic_guard.get("patcher_error_excerpt", "")
                or semantic_guard.get("verification_excerpt", "")
                or ""
            )[:800]
        return ActionObservation(
            action_type=semantic_step.action.value,
            status=str(semantic_guard.get("result_mode", "") or "observed"),
            active_target_before=ledger_before.active_target,
            active_target_after=ledger_after.active_target,
            active_object_type_before=ledger_before.active_object_type,
            active_object_type_after=ledger_after.active_object_type,
            active_object_id_before=ledger_before.active_object_id,
            active_object_id_after=ledger_after.active_object_id,
            replay_scope=replay_scope,
            touched_paths=changed_files,
            invalidated_targets=[
                item
                for item in ledger_after.invalidated_targets
                if item not in set(ledger_before.invalidated_targets)
            ],
            invalidated_object_ids=[
                item
                for item in ledger_after.invalidated_object_ids
                if item not in set(ledger_before.invalidated_object_ids)
            ],
            verifier_excerpt=verifier_excerpt,
            target_legitimacy=target_legitimacy,
            patch_legitimacy=patch_legitimacy,
            negative_constraints=list(
                dict.fromkeys(
                    list(ledger_after.negative_constraints)
                    + self._negative_constraints_from_guard(semantic_guard)
                )
            ),
            notes=str(semantic_guard.get("next_action_suggestion", "") or ""),
            metadata={
                "canonical_action_family": self._canonical_action_family(semantic_step.action),
                "guard_flags": list(semantic_guard.get("guard_flags", []) or []),
                "patch_scope": str(semantic_guard.get("patch_scope", "") or ""),
                "touches_suspect_path": bool(semantic_guard.get("touches_suspect_path", False)),
                "patch_contract_audit": dict(semantic_guard.get("patch_contract_audit", {}) or {}),
                "patch_intent_audit": dict(semantic_guard.get("patch_intent_audit", {}) or {}),
                "action_guidance_audit": dict(semantic_guard.get("action_guidance_audit", {}) or {}),
            },
        )

    def _build_state_delta(
        self,
        *,
        semantic_step: SemanticRecoveryStep,
        ledger_before: RecoveryLedger,
        ledger_after: RecoveryLedger,
        primitive_trace: list[dict[str, Any]],
        semantic_guard: dict[str, Any],
    ) -> StateDelta:
        consumed_token_budget = sum(float(step.get("token_cost", 0.0) or 0.0) for step in primitive_trace)
        consumed_latency_budget = sum(float(step.get("latency_sec", 0.0) or 0.0) for step in primitive_trace)
        latest_shared_fact_key = ledger_after.latest_shared_fact_key
        if semantic_step.action == SemanticActionType.REVOKE_OBJECT:
            latest_shared_fact_key = ""
        latest_verifier_verdict = ledger_after.latest_verifier_verdict
        if semantic_guard:
            latest_verifier_verdict = str(
                semantic_guard.get("success_legitimacy", "")
                or semantic_guard.get("result_mode", "")
                or latest_verifier_verdict
                or ""
            )
        lifecycle_target_object_id = ledger_after.active_object_id
        lifecycle_target_object_type = ledger_after.active_object_type
        if semantic_step.action in {
            SemanticActionType.REVOKE_OBJECT,
            SemanticActionType.TARGET_RESET,
        }:
            lifecycle_target_object_id = (
                str(semantic_step.args.get("object_id", "") or "")
                or ledger_before.active_object_id
                or ledger_after.active_object_id
            )
            lifecycle_target_object_type = (
                str(semantic_step.args.get("object_type", "") or "")
                or ledger_before.active_object_type
                or ledger_after.active_object_type
            )
        return StateDelta(
            active_target=ledger_after.active_target,
            active_object_type=ledger_after.active_object_type,
            active_object_id=ledger_after.active_object_id,
            added_invalidated_targets=[
                item
                for item in ledger_after.invalidated_targets
                if item not in set(ledger_before.invalidated_targets)
            ],
            added_invalidated_object_ids=[
                item
                for item in ledger_after.invalidated_object_ids
                if item not in set(ledger_before.invalidated_object_ids)
            ],
            added_negative_constraints=self._negative_constraints_from_guard(semantic_guard),
            latest_verifier_verdict=latest_verifier_verdict,
            latest_shared_fact_key=latest_shared_fact_key,
            touches_suspect_path=bool(semantic_guard.get("touches_suspect_path", ledger_after.touches_suspect_path)),
            consumed_step_budget=1,
            consumed_token_budget=consumed_token_budget,
            consumed_latency_budget_sec=consumed_latency_budget,
            metadata={
                "semantic_action": semantic_step.action.value,
                "canonical_action_family": self._canonical_action_family(semantic_step.action),
                "result_mode": str(semantic_guard.get("result_mode", "") or ""),
                "patch_contract_audit": dict(semantic_guard.get("patch_contract_audit", {}) or {}),
                "patch_intent_audit": dict(semantic_guard.get("patch_intent_audit", {}) or {}),
                "action_guidance_audit": dict(semantic_guard.get("action_guidance_audit", {}) or {}),
                "lifecycle_target_object_id": lifecycle_target_object_id,
                "lifecycle_target_object_type": lifecycle_target_object_type,
                "invalidated_object_stages": self._invalidated_object_stages_from_trace(
                    primitive_trace
                ),
            },
        )

    @staticmethod
    def _invalidated_object_stages_from_trace(primitive_trace: list[dict[str, Any]]) -> list[tuple[str, str]]:
        stages: list[tuple[str, str]] = []
        for step in primitive_trace:
            result = dict(step.get("result", {}) or {})
            for pair in list(result.get("invalidated_object_stages", []) or []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                normalized = (str(pair[0]), str(pair[1]))
                if normalized not in stages:
                    stages.append(normalized)
        return stages

    def _apply_state_delta(
        self,
        *,
        ledger: RecoveryLedger,
        state_delta: StateDelta,
        action_observation: ActionObservation,
    ) -> None:
        if state_delta.active_target:
            ledger.active_target = state_delta.active_target
        if state_delta.active_object_type:
            ledger.active_object_type = state_delta.active_object_type
        if state_delta.active_object_id:
            ledger.active_object_id = state_delta.active_object_id
        ledger.invalidated_targets = list(
            dict.fromkeys(list(ledger.invalidated_targets) + list(state_delta.added_invalidated_targets))
        )
        ledger.invalidated_object_ids = list(
            dict.fromkeys(list(ledger.invalidated_object_ids) + list(state_delta.added_invalidated_object_ids))
        )
        ledger.negative_constraints = list(
            dict.fromkeys(list(ledger.negative_constraints) + list(state_delta.added_negative_constraints))
        )
        reflection = self._structured_reflection_from_observation(
            action_observation=action_observation,
            state_delta=state_delta,
        )
        if reflection:
            reflection_memory = [
                dict(item)
                for item in list(ledger.metadata.get("reflection_memory", []) or [])
                if isinstance(item, dict)
            ]
            reflection_memory.append(reflection)
            ledger.metadata["reflection_memory"] = reflection_memory[-6:]
            ledger.metadata["latest_structured_reflection"] = reflection
            avoid_rule = str(reflection.get("avoid_rule", "") or "").strip()
            if avoid_rule:
                ledger.negative_constraints = list(
                    dict.fromkeys(list(ledger.negative_constraints) + [avoid_rule])
                )
        if state_delta.latest_verifier_verdict:
            ledger.latest_verifier_verdict = state_delta.latest_verifier_verdict
        ledger.latest_shared_fact_key = state_delta.latest_shared_fact_key
        ledger.touches_suspect_path = state_delta.touches_suspect_path
        if ledger.structured_state:
            local_region = dict(ledger.structured_state.get("local_region_view", {}) or {})
            evidence_pack = dict(ledger.structured_state.get("evidence_pack", {}) or {})
            local_region["active_target"] = ledger.active_target
            local_region["active_object_type"] = ledger.active_object_type
            local_region["active_object_id"] = ledger.active_object_id
            local_region["negative_constraints"] = list(ledger.negative_constraints)
            evidence_pack["negative_constraints"] = list(ledger.negative_constraints)
            if action_observation.target_legitimacy:
                evidence_pack["target_legitimacy"] = action_observation.target_legitimacy
            if action_observation.patch_legitimacy:
                evidence_pack["patch_legitimacy"] = action_observation.patch_legitimacy
            if action_observation.verifier_excerpt:
                evidence_pack["verifier_excerpt"] = action_observation.verifier_excerpt
            reflection_memory = list(ledger.metadata.get("reflection_memory", []) or [])
            if reflection_memory:
                local_region["reflection_memory"] = reflection_memory[-6:]
                evidence_pack["latest_structured_reflection"] = reflection_memory[-1]
            source_candidate_memory = list(ledger.metadata.get("source_candidate_memory", []) or [])
            if source_candidate_memory:
                local_region["source_candidate_memory"] = source_candidate_memory[-4:]
                evidence_pack["best_source_candidate"] = self._best_source_candidate(ledger)
            patch_contract = dict(ledger.metadata.get("latest_patch_contract", {}) or {})
            patch_contract_audit = dict(state_delta.metadata.get("patch_contract_audit", {}) or {})
            action_guidance = dict(ledger.metadata.get("latest_action_guidance_gate", {}) or {})
            action_guidance_audit = dict(state_delta.metadata.get("action_guidance_audit", {}) or {})
            patch_intent = dict(ledger.metadata.get("latest_patch_intent", {}) or {})
            patch_intent_audit = dict(state_delta.metadata.get("patch_intent_audit", {}) or {})
            if patch_contract:
                local_region["latest_patch_contract"] = patch_contract
                evidence_pack["latest_patch_contract"] = patch_contract
            if patch_contract_audit:
                local_region["latest_patch_contract_audit"] = patch_contract_audit
                evidence_pack["latest_patch_contract_audit"] = patch_contract_audit
            if action_guidance:
                local_region["latest_action_guidance_gate"] = action_guidance
                evidence_pack["latest_action_guidance_gate"] = action_guidance
            if action_guidance_audit:
                local_region["latest_action_guidance_gate_audit"] = action_guidance_audit
                evidence_pack["latest_action_guidance_gate_audit"] = action_guidance_audit
            if patch_intent:
                local_region["latest_patch_intent"] = patch_intent
                evidence_pack["latest_patch_intent"] = patch_intent
            if patch_intent_audit:
                local_region["latest_patch_intent_audit"] = patch_intent_audit
                evidence_pack["latest_patch_intent_audit"] = patch_intent_audit
            ledger.structured_state["local_region_view"] = local_region
            ledger.structured_state["evidence_pack"] = evidence_pack
            ledger.structured_state["last_action_observation"] = action_observation.to_dict()
            ledger.structured_state["last_state_delta"] = state_delta.to_dict()
        lifecycle_transition = apply_lifecycle_transition_to_ledger(
            ledger,
            action_observation=action_observation,
            state_delta=state_delta,
            unproductive_threshold=int(ledger.metadata.get("object_unproductive_threshold", 2) or 2),
        )
        if lifecycle_transition is not None and ledger.structured_state:
            local_region = dict(ledger.structured_state.get("local_region_view", {}) or {})
            evidence_pack = dict(ledger.structured_state.get("evidence_pack", {}) or {})
            local_region["latest_lifecycle_transition"] = lifecycle_transition.to_dict()
            evidence_pack["latest_lifecycle_transition"] = lifecycle_transition.to_dict()
            ledger.structured_state["local_region_view"] = local_region
            ledger.structured_state["evidence_pack"] = evidence_pack

    @staticmethod
    def _structured_reflection_from_observation(
        *,
        action_observation: ActionObservation,
        state_delta: StateDelta,
    ) -> dict[str, Any]:
        status = str(action_observation.status or "").strip()
        if status in {
            "",
            "observed",
            "strong_source_success",
            "mixed_source_test_success",
            "checkpoint_or_no_diff_success",
        }:
            return {}

        flags = [
            str(item)
            for item in list(action_observation.metadata.get("guard_flags", []) or [])
            if str(item).strip()
        ]
        action_type = str(action_observation.action_type or "")
        replay_scope = str(action_observation.replay_scope or "")
        active_target = str(action_observation.active_target_after or action_observation.active_target_before or "")
        active_object_type = str(
            action_observation.active_object_type_after
            or action_observation.active_object_type_before
            or ""
        )
        prefer_next = str(action_observation.notes or "").strip()
        avoid_rule = ""
        reflection_kind = "unresolved_recovery_attempt"

        if status == "no_diff" or "no_diff" in flags:
            reflection_kind = "ineffective_replay"
            avoid_rule = (
                "avoid repeating the same replay scope without changing target, evidence, capability, "
                "or invalidating the polluted propagation object"
            )
        elif status == "wrong_edit_target" or "generated_or_build_edit" in flags or "tests_only_edit" in flags:
            reflection_kind = "invalid_edit_target"
            avoid_rule = "avoid tests-only, generated, or off-target edits when source repair is required"
        elif status == "source_edit_but_not_suspect" or "source_edit_but_not_suspect" in flags:
            reflection_kind = "missed_suspect_region"
            avoid_rule = "avoid trusting a source edit that misses the current suspicious region"
        elif status == "oracle_failed_after_source_edit":
            reflection_kind = "unresolved_source_edit"
            avoid_rule = "avoid treating a source diff as sufficient until focused and official verification pass"
        elif status.startswith("contract_violation"):
            reflection_kind = "stage_boundary_contract_violation"
            avoid_rule = "avoid replaying patcher without satisfying fresh source diff and suspect-boundary contract"
        elif status.startswith("intent_violation"):
            reflection_kind = "car_patch_intent_violation"
            avoid_rule = "avoid replaying patcher without satisfying the CAR patch intent or refreshing evidence"
        elif status.endswith("_pending_official"):
            reflection_kind = "pending_official_validation"
            avoid_rule = "avoid declaring success before official verification resolves the replay"
        else:
            avoid_rule = "avoid repeating the same failed recovery pattern without new evidence"

        if "readonly_replay_exhausted" in flags:
            prefer_next = prefer_next or SemanticActionType.CAPABILITY_BOOST.value
        if "source_diff_unchanged_during_replay" in flags:
            avoid_rule = (
                avoid_rule
                + "; do not reuse a stale pre-existing diff as a fresh recovery patch"
            )
        if "contract_missing_focused_validation" in flags:
            avoid_rule = avoid_rule + "; do not leave fresh source edits without focused validation"

        return {
            "schema": "structured_reflection_v1",
            "kind": reflection_kind,
            "after_action": action_type,
            "result_mode": status,
            "guard_flags": flags,
            "active_target": active_target,
            "active_object_type": active_object_type,
            "replay_scope": replay_scope,
            "touched_paths": list(action_observation.touched_paths[:6]),
            "avoid_rule": avoid_rule,
            "prefer_next_action": prefer_next,
            "consumed_step_budget": int(state_delta.consumed_step_budget or 0),
            "consumed_token_budget": float(state_delta.consumed_token_budget or 0.0),
            "consumed_latency_budget_sec": float(state_delta.consumed_latency_budget_sec or 0.0),
        }

    @staticmethod
    def _canonical_action_family(action: SemanticActionType) -> str:
        if action in {SemanticActionType.EVIDENCE_RECHECK, SemanticActionType.RECHECK_OBJECT}:
            return "RECHECK_OBJECT"
        if action == SemanticActionType.REVOKE_OBJECT:
            return "REVOKE_OBJECT"
        if action == SemanticActionType.TARGET_RESET:
            return "ROLLBACK_TO_ANCHOR"
        if action in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL}:
            return "REPAIR_LOCAL"
        if action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE, SemanticActionType.CAPABILITY_BOOST}:
            return "EXPAND_SCOPE"
        return action.value

    def _target_legitimacy_from_guard(self, guard: dict[str, Any]) -> str:
        patch_scope = str(guard.get("patch_scope", "") or "").strip()
        classes = dict(guard.get("changed_file_classes", {}) or {})
        source_files = list(classes.get("source_files", []) or [])
        test_files = list(classes.get("test_files", []) or [])
        generated_files = list(classes.get("generated_files", []) or [])
        other_files = list(classes.get("other_files", []) or [])
        if patch_scope == "no_diff":
            return "no_diff"
        if source_files and not test_files and not generated_files and not other_files:
            return "source_only"
        if source_files and (test_files or other_files):
            return "source_mixed"
        if test_files and not source_files:
            return "tests_only"
        if generated_files and not source_files:
            return "generated_only"
        if other_files and not source_files and not test_files:
            return "non_source_only"
        return ""

    @staticmethod
    def _negative_constraints_from_guard(guard: dict[str, Any]) -> list[str]:
        constraints: list[str] = []
        result_mode = str(guard.get("result_mode", "") or "")
        flags = [str(item) for item in list(guard.get("guard_flags", []) or [])]
        if "no_diff" in flags or result_mode == "no_diff":
            constraints.append("previous replay produced no effective diff")
        if result_mode == "wrong_edit_target" or "generated_or_build_edit" in flags:
            constraints.append("avoid off-target edits outside canonical suspect source paths")
        if result_mode.startswith("contract_violation"):
            constraints.append("previous replay violated the stage-boundary patch contract")
        if result_mode.startswith("intent_violation"):
            constraints.append("previous replay violated the CAR patch intent")
        if result_mode == "source_edit_but_not_suspect" or "source_edit_but_not_suspect" in flags:
            constraints.append("previous source edit missed the current suspicious region")
        if result_mode == "oracle_failed_after_source_edit":
            constraints.append("previous source edit did not resolve the downstream verifier failure")
        if "tests_only_edit" in flags:
            constraints.append("avoid tests-only recovery when source repair is still required")
        if "contract_no_fresh_source_diff" in flags:
            constraints.append("next replay must create a fresh source diff at the stage boundary")
        if "contract_missed_suspect_path" in flags:
            constraints.append("next replay must touch the current suspect source boundary or refresh localization first")
        if "contract_missing_focused_validation" in flags:
            constraints.append("next source edit must be followed by focused validation")
        if "intent_no_fresh_source_diff" in flags:
            constraints.append("next CAR replay must create a fresh source diff")
        if "intent_missed_target_path" in flags:
            constraints.append("next CAR replay must touch the intended source target or refresh evidence first")
        if "intent_touched_revoked_path" in flags:
            constraints.append("avoid returning to revoked or stale CAR target paths without new evidence")
        return list(dict.fromkeys(constraints))

    def _build_semantic_guard(
        self,
        *,
        patch_summary: dict[str, Any],
        ledger: RecoveryLedger,
        replay_result: dict[str, Any],
        patch_contract: dict[str, Any] | None = None,
        patch_intent: dict[str, Any] | None = None,
        action_guidance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed_files = self._extract_changed_files(patch_summary)
        file_classes = classify_changed_files(changed_files)
        source_files = self._merge_path_lists(
            file_classes.get("source_files", []),
            patch_summary.get("source_files", []),
            patch_summary.get("non_test_source_files", []),
        )
        test_files = self._merge_path_lists(file_classes.get("test_files", []), patch_summary.get("test_files", []))
        generated_files = self._merge_path_lists(
            file_classes.get("generated_files", []),
            patch_summary.get("generated_files", []),
        )
        other_files = self._merge_path_lists(file_classes.get("other_files", []), patch_summary.get("other_files", []))
        non_test_files = self._merge_path_lists(
            [path for path in changed_files if path not in set(test_files)],
            patch_summary.get("non_test_files", []),
        )
        normalized_classes = {
            "source_files": source_files,
            "test_files": test_files,
            "generated_files": generated_files,
            "other_files": other_files,
            "effective_files": self._merge_path_lists(
                file_classes.get("effective_files", []),
                source_files,
                test_files,
                other_files,
            ),
        }

        explicit_scope = str(patch_summary.get("patch_scope", "") or "").strip()
        patch_scope = explicit_scope or self._infer_patch_scope(changed_files, test_files, non_test_files)
        suspect_paths = self._merge_path_lists(ledger.suspect_paths, patch_summary.get("suspect_paths", []))
        suspect_overlap = self._suspect_overlap(changed_files, suspect_paths)
        explicit_touches = patch_summary.get("touches_suspect_path")
        touches_suspect_path = bool(explicit_touches) if explicit_touches is not None else bool(suspect_overlap)
        replay_success = bool(replay_result.get("success", False))
        official_success_deferred = bool(replay_result.get("internal_verifier_skipped", False))
        patcher_trace = dict(replay_result.get("patcher_trace", {}) or {})
        patcher_command_count = int(patcher_trace.get("patcher_command_count", 0) or 0)
        patcher_write_command_count = int(patcher_trace.get("patcher_write_command_count", 0) or 0)
        patcher_validation_command_count = int(patcher_trace.get("patcher_validation_command_count", 0) or 0)
        patcher_readonly_command_count = int(patcher_trace.get("patcher_readonly_command_count", 0) or 0)
        focused_validation = self._focused_validation_evidence(
            patcher_trace=patcher_trace,
            suspect_paths=suspect_paths,
            target_paths=self._patch_intent_target_paths(patch_intent or {}),
            failure_tests=self._failed_state_test_refs(ledger),
        )
        source_patch_risk = dict(patch_summary.get("source_patch_risk", {}) or {})
        patcher_error_excerpt = self._compact_error_excerpt(
            patcher_trace.get("patcher_error", ""),
            patcher_trace.get("patcher_planner_error", ""),
            patcher_trace.get("patcher_implementer_error", ""),
        )
        replay_diff_changed = bool(replay_result.get("replay_diff_changed", False))
        pre_replay_diff = dict(replay_result.get("pre_replay_diff", {}) or {})
        post_replay_diff = dict(replay_result.get("post_replay_diff", {}) or {})
        has_replay_diff_metadata = any(
            key in replay_result
            for key in ("pre_replay_diff", "post_replay_diff", "replay_diff_changed")
        )
        if has_replay_diff_metadata:
            fresh_changed_files = self._fresh_changed_files(
                pre_replay_diff=pre_replay_diff,
                post_replay_diff=post_replay_diff,
            )
            removed_stale_diff_files = self._removed_stale_diff_files(
                pre_replay_diff=pre_replay_diff,
                post_replay_diff=post_replay_diff,
            )
            decision_changed_files = list(fresh_changed_files) if replay_diff_changed else []
        else:
            fresh_changed_files = list(changed_files)
            removed_stale_diff_files = []
            decision_changed_files = list(changed_files)
        decision_file_classes = classify_changed_files(decision_changed_files)
        decision_source_files = self._merge_path_lists(decision_file_classes.get("source_files", []))
        decision_test_files = self._merge_path_lists(decision_file_classes.get("test_files", []))
        decision_generated_files = self._merge_path_lists(decision_file_classes.get("generated_files", []))
        decision_other_files = self._merge_path_lists(decision_file_classes.get("other_files", []))
        decision_non_test_files = self._merge_path_lists(
            [path for path in decision_changed_files if path not in set(decision_test_files)]
        )
        decision_patch_scope = self._infer_patch_scope(
            decision_changed_files,
            decision_test_files,
            decision_non_test_files,
        )
        if replay_diff_changed and not decision_changed_files:
            decision_patch_scope = "no_diff" if removed_stale_diff_files else "unknown_diff"
        decision_suspect_overlap = self._suspect_overlap(decision_changed_files, suspect_paths)
        decision_touches_suspect_path = bool(decision_suspect_overlap)
        if replay_diff_changed and explicit_touches is not None and set(decision_changed_files) == set(changed_files):
            decision_touches_suspect_path = bool(explicit_touches)
        result_mode = self._guard_result_mode(
            replay_success=replay_success,
            official_success_deferred=official_success_deferred,
            patch_scope=decision_patch_scope,
            source_files=decision_source_files,
            test_files=decision_test_files,
            generated_files=decision_generated_files,
            other_files=decision_other_files,
            touches_suspect_path=decision_touches_suspect_path,
        )
        guard_flags = self._guard_flags(
            replay_success=replay_success,
            official_success_deferred=official_success_deferred,
            patch_scope=decision_patch_scope,
            source_files=decision_source_files,
            test_files=decision_test_files,
            generated_files=decision_generated_files,
            other_files=decision_other_files,
            touches_suspect_path=decision_touches_suspect_path,
        )
        guard_flags.extend(
            self._execution_quality_flags(
                observed_source_files=source_files,
                fresh_source_files=decision_source_files,
                replay_success=replay_success,
                patcher_command_count=patcher_command_count,
                patcher_write_command_count=patcher_write_command_count,
                patcher_validation_command_count=patcher_validation_command_count,
                patcher_readonly_command_count=patcher_readonly_command_count,
                replay_diff_changed=replay_diff_changed,
                removed_stale_diff_files=removed_stale_diff_files,
            )
        )
        patch_contract_audit = audit_stage_boundary_patch_contract(
            patch_contract or {},
            changed_files=changed_files,
            fresh_changed_files=decision_changed_files,
            changed_file_classes=normalized_classes,
            fresh_changed_file_classes={
                "source_files": decision_source_files,
                "test_files": decision_test_files,
                "generated_files": decision_generated_files,
                "other_files": decision_other_files,
                "effective_files": self._merge_path_lists(
                    decision_file_classes.get("effective_files", []),
                    decision_source_files,
                    decision_test_files,
                    decision_other_files,
                ),
            },
            patcher_validation_command_count=patcher_validation_command_count,
            replay_diff_changed=replay_diff_changed,
        )
        contract_flags = [
            str(item)
            for item in list(patch_contract_audit.get("flags", []) or [])
            if str(item).strip()
        ]
        guard_flags.extend(contract_flags)
        if patch_contract_audit.get("contract_present") and not patch_contract_audit.get("hard_satisfied", True):
            if decision_source_files:
                result_mode = "contract_violation_after_source_edit"
            else:
                result_mode = "contract_violation_no_fresh_source"
            decision_patch_scope = decision_patch_scope or "no_diff"
        patch_intent_audit = audit_car_patch_intent(
            patch_intent or {},
            changed_files=changed_files,
            fresh_changed_files=decision_changed_files,
            changed_file_classes=normalized_classes,
            fresh_changed_file_classes={
                "source_files": decision_source_files,
                "test_files": decision_test_files,
                "generated_files": decision_generated_files,
                "other_files": decision_other_files,
                "effective_files": self._merge_path_lists(
                    decision_file_classes.get("effective_files", []),
                    decision_source_files,
                    decision_test_files,
                    decision_other_files,
                ),
            },
            patcher_validation_command_count=patcher_validation_command_count,
            replay_diff_changed=replay_diff_changed,
        )
        intent_flags = [
            str(item)
            for item in list(patch_intent_audit.get("flags", []) or [])
            if str(item).strip()
        ]
        guard_flags.extend(intent_flags)
        if decision_source_files and patcher_validation_command_count > 0:
            if not bool(focused_validation.get("target_related", False)):
                guard_flags.append("focused_validation_not_target_related")
            if not bool(focused_validation.get("has_result", False)):
                guard_flags.append("focused_validation_missing_result")
            elif bool(focused_validation.get("passed", False)):
                guard_flags.append("focused_validation_passed")
            else:
                guard_flags.append("focused_validation_failed")
        source_patch_risk_level = str(source_patch_risk.get("risk_level", "") or "")
        if source_patch_risk_level == "large_source_rewrite":
            guard_flags.append("large_source_rewrite")
        elif source_patch_risk_level == "broad_source_change":
            guard_flags.append("broad_source_change")
        if patch_intent_audit.get("intent_present") and not patch_intent_audit.get("hard_satisfied", True):
            hard_flags = set(str(item) for item in list(patch_intent_audit.get("hard_flags", []) or []))
            if "intent_no_fresh_source_diff" in hard_flags:
                result_mode = "intent_violation_no_fresh_source"
            elif "intent_missed_target_path" in hard_flags:
                result_mode = "intent_violation_missed_target"
            elif "intent_touched_revoked_path" in hard_flags:
                result_mode = "intent_violation_revoked_target"
            elif "intent_too_many_source_files" in hard_flags:
                result_mode = "intent_violation_too_broad"
            else:
                result_mode = "intent_violation_after_source_edit"
            decision_patch_scope = decision_patch_scope or "no_diff"
        action_guidance_audit = audit_car_action_guidance_packet(
            action_guidance or {},
            guard={
                "fresh_changed_file_classes": {
                    "source_files": decision_source_files,
                    "test_files": decision_test_files,
                    "generated_files": decision_generated_files,
                    "other_files": decision_other_files,
                    "effective_files": self._merge_path_lists(
                        decision_file_classes.get("effective_files", []),
                        decision_source_files,
                        decision_test_files,
                        decision_other_files,
                    ),
                },
                "focused_validation": focused_validation,
            },
        )
        guidance_flags = [
            str(item)
            for item in list(action_guidance_audit.get("flags", []) or [])
            if str(item).strip()
        ]
        guard_flags.extend(guidance_flags)
        budget_error_excerpt = self._budget_exceeded_excerpt(replay_result)
        if budget_error_excerpt:
            guard_flags.extend(["budget_exhausted", "token_budget_exhausted"])
            result_mode = "budget_exhausted"
            decision_patch_scope = "no_diff"
        guard_flags = list(dict.fromkeys(guard_flags))
        return {
            "patch_scope": patch_scope,
            "changed_files": changed_files,
            "decision_patch_scope": decision_patch_scope,
            "fresh_changed_files": decision_changed_files,
            "removed_stale_diff_files": removed_stale_diff_files,
            "changed_file_classes": normalized_classes,
            "fresh_changed_file_classes": {
                "source_files": decision_source_files,
                "test_files": decision_test_files,
                "generated_files": decision_generated_files,
                "other_files": decision_other_files,
                "effective_files": self._merge_path_lists(
                    decision_file_classes.get("effective_files", []),
                    decision_source_files,
                    decision_test_files,
                    decision_other_files,
                ),
            },
            "changed_file_class_counts": {
                key: len(value)
                for key, value in normalized_classes.items()
            },
            "suspect_paths": suspect_paths,
            "suspect_path_overlap": suspect_overlap,
            "fresh_suspect_path_overlap": decision_suspect_overlap,
            "touches_suspect_path": touches_suspect_path,
            "decision_touches_suspect_path": decision_touches_suspect_path,
            "active_target": str(ledger.active_target or ""),
            "active_object_type": str(ledger.active_object_type or ""),
            "active_object_id": str(ledger.active_object_id or ""),
            "replay_success": replay_success,
            "official_success_deferred": official_success_deferred,
            "patcher_command_count": patcher_command_count,
            "patcher_write_command_count": patcher_write_command_count,
            "patcher_validation_command_count": patcher_validation_command_count,
            "focused_validation": focused_validation,
            "source_patch_risk": source_patch_risk,
            "patcher_readonly_command_count": patcher_readonly_command_count,
            "patcher_error_excerpt": patcher_error_excerpt,
            "replay_diff_changed": replay_diff_changed,
            "pre_replay_diff": pre_replay_diff,
            "post_replay_diff": post_replay_diff,
            "result_mode": result_mode,
            "failure_mode": "" if replay_success else result_mode,
            "success_legitimacy": (
                "pending_official_validation"
                if replay_success and official_success_deferred
                else result_mode if replay_success
                else "not_resolved"
            ),
            "guard_flags": guard_flags,
            "next_action_suggestion": (
                "STOP"
                if budget_error_excerpt
                else self._suggest_next_semantic_action(
                    result_mode=result_mode,
                    patch_scope=decision_patch_scope,
                    source_files=decision_source_files,
                    touches_suspect_path=decision_touches_suspect_path,
                    active_object_type=str(ledger.active_object_type or ""),
                )
            ),
            "budget_stop": (
                {
                    "reason": "token_budget_exhausted",
                    "excerpt": budget_error_excerpt[:500],
                }
                if budget_error_excerpt
                else {}
            ),
            "patch_contract": dict(patch_contract or {}),
            "patch_contract_audit": patch_contract_audit,
            "patch_intent": dict(patch_intent or {}),
            "patch_intent_audit": patch_intent_audit,
            "action_guidance": dict(action_guidance or {}),
            "action_guidance_audit": action_guidance_audit,
        }

    @staticmethod
    def _budget_exceeded_excerpt(payload: Any) -> str:
        if isinstance(payload, dict):
            skip_keys = {
                "pre_replay_diff",
                "post_replay_diff",
                "file_digests",
                "changed_files",
                "changed_file_classes",
                "fresh_changed_file_classes",
            }
            for key, value in payload.items():
                if str(key) in skip_keys:
                    continue
                found = SemanticProgramExecutor._budget_exceeded_excerpt(value)
                if found:
                    return found
            return ""
        if isinstance(payload, (list, tuple)):
            for item in payload:
                found = SemanticProgramExecutor._budget_exceeded_excerpt(item)
                if found:
                    return found
            return ""
        text = str(payload or "")
        if "budget_exceeded" in text.lower():
            return text
        return ""

    @staticmethod
    def _patch_intent_target_paths(patch_intent: dict[str, Any]) -> list[str]:
        paths = dict(patch_intent.get("paths", {}) or {})
        values: list[str] = []
        for key in ("target_paths", "candidate_source_paths", "suspect_paths"):
            values.extend(
                normalize_repo_path(str(item))
                for item in list(paths.get(key, []) or [])
                if normalize_repo_path(str(item))
            )
        return sorted(dict.fromkeys(values))

    @staticmethod
    def _failed_state_test_refs(ledger: RecoveryLedger) -> list[str]:
        failed_state = dict(ledger.metadata.get("failed_state", {}) or {})
        metadata = dict(failed_state.get("metadata", {}) or {})
        observation = dict(
            failed_state.get("failure_observation", {})
            or metadata.get("failure_observation", {})
            or {}
        )
        values: list[str] = []
        for key in ("test_command", "failing_tests", "tests"):
            value = observation.get(key, "")
            if isinstance(value, str):
                values.append(value)
            else:
                values.extend(str(item) for item in list(value or []))
        return [item for item in values if item.strip()][:8]

    @classmethod
    def _focused_validation_evidence(
        cls,
        *,
        patcher_trace: dict[str, Any],
        suspect_paths: list[str],
        target_paths: list[str],
        failure_tests: list[str],
    ) -> dict[str, Any]:
        commands = [
            dict(item)
            for item in list(patcher_trace.get("patcher_commands_excerpt", []) or [])
            if isinstance(item, dict)
        ]
        if not commands:
            commands = [
                dict(item)
                for item in list(patcher_trace.get("commands", []) or [])
                if isinstance(item, dict)
            ]
        validation_events = [
            item
            for item in commands
            if cls._looks_like_validation_command_text(str(item.get("command", "") or ""))
        ]
        targets = [normalize_repo_path(path) for path in list(target_paths or []) + list(suspect_paths or [])]
        targets = [path for path in targets if path]
        test_refs = [str(item).strip() for item in list(failure_tests or []) if str(item).strip()]
        evidence_terms = set(targets)
        generic_test_tokens = {
            "pytest",
            "tests",
            "test",
            "python",
            "unittest",
            "nosetests",
            "tox",
        }
        for ref in test_refs:
            evidence_terms.add(ref)
            for token in ref.replace("::", " ").replace("/", " ").split():
                normalized_token = token.strip().lower()
                if len(normalized_token) >= 4 and normalized_token not in generic_test_tokens:
                    evidence_terms.add(token)
        latest = validation_events[-1] if validation_events else {}
        latest_command = str(latest.get("command", "") or "")
        latest_output = str(latest.get("output", "") or "")
        haystack = f"{latest_command}\n{latest_output}".lower()
        target_related = bool(
            validation_events
            and (
                not evidence_terms
                or any(term.lower() in haystack for term in evidence_terms if term)
            )
        )
        has_result = bool(validation_events and "returncode" in latest)
        latest_returncode = latest.get("returncode", None)
        passed = bool(has_result and int(latest_returncode if latest_returncode is not None else -1) == 0)
        return {
            "schema": "focused_validation_evidence_v1",
            "count": len(validation_events),
            "has_result": has_result,
            "passed": passed,
            "target_related": target_related,
            "latest_command": latest_command[:500],
            "latest_returncode": latest.get("returncode", None),
            "target_paths": targets[:8],
            "failure_tests": test_refs[:8],
            "output_excerpt": cls._failure_focused_excerpt(latest_output, limit=500),
        }

    @staticmethod
    def _looks_like_validation_command_text(command: str) -> bool:
        return looks_like_validation_command(command)

    @staticmethod
    def _provider_error_excerpt(payload: Any) -> str:
        patterns = (
            "openai-compatible api request failed",
            "operation timed out",
            "upstream_error",
            "chat upstream returned",
            "rate_limit",
            "model request failed",
            "non-json upstream response",
            "heartbeat stream connected",
        )
        if isinstance(payload, dict):
            skip_keys = {
                "pre_replay_diff",
                "post_replay_diff",
                "file_digests",
                "changed_files",
                "changed_file_classes",
                "fresh_changed_file_classes",
            }
            for key, value in payload.items():
                if str(key) in skip_keys:
                    continue
                found = SemanticProgramExecutor._provider_error_excerpt(value)
                if found:
                    return found
            return ""
        if isinstance(payload, (list, tuple)):
            for item in payload:
                found = SemanticProgramExecutor._provider_error_excerpt(item)
                if found:
                    return found
            return ""
        text = " ".join(str(payload or "").split())
        lowered = text.lower()
        if any(pattern in lowered for pattern in patterns):
            return text[:500]
        return ""

    @staticmethod
    def _compact_error_excerpt(*values: Any, limit: int = 900) -> str:
        parts: list[str] = []
        for value in values:
            text = " ".join(str(value or "").split())
            if text and text not in parts:
                parts.append(text)
        return " | ".join(parts)[:limit]

    @classmethod
    def _failure_focused_excerpt(cls, value: Any, limit: int = 900) -> str:
        text = str(value or "")
        if not text.strip():
            return ""
        lines = [line.rstrip() for line in text.splitlines()]
        markers = (
            "traceback",
            "error",
            "failed",
            "assert",
            "exception",
            "mismatch",
            "returncode",
            "syntaxerror",
            "valueerror",
            "typeerror",
            "indexerror",
            "keyerror",
        )
        selected: list[str] = []
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if not any(marker in lowered for marker in markers):
                continue
            start = max(0, idx - 2)
            end = min(len(lines), idx + 5)
            for nearby in lines[start:end]:
                compact = " ".join(nearby.split())
                if compact and compact not in selected:
                    selected.append(compact)
                if len(" | ".join(selected)) >= limit:
                    break
            if len(" | ".join(selected)) >= limit:
                break
        if not selected:
            for line in lines[-18:]:
                compact = " ".join(line.split())
                if compact and compact not in selected:
                    selected.append(compact)
                if len(" | ".join(selected)) >= limit:
                    break
        return " | ".join(selected)[:limit]

    @classmethod
    def _validation_evidence_excerpt_from_guard(cls, guard: dict[str, Any], limit: int = 900) -> str:
        return cls._compact_error_excerpt(
            cls._failure_focused_excerpt(guard.get("closed_loop_fail_to_pass_output_excerpt", "")),
            cls._failure_focused_excerpt(guard.get("closed_loop_oracle_output_excerpt", "")),
            guard.get("patcher_error_excerpt", ""),
            limit=limit,
        )

    @staticmethod
    def _fresh_changed_files(
        *,
        pre_replay_diff: dict[str, Any],
        post_replay_diff: dict[str, Any],
    ) -> list[str]:
        pre_file_digests = dict(pre_replay_diff.get("file_digests", {}) or {})
        post_file_digests = dict(post_replay_diff.get("file_digests", {}) or {})
        if pre_file_digests or post_file_digests:
            return sorted(
                path
                for path in set(post_file_digests)
                if str(post_file_digests.get(path, "") or "") != str(pre_file_digests.get(path, "") or "")
            )
        pre_files = set(str(item) for item in list(pre_replay_diff.get("changed_files", []) or []))
        post_files = set(str(item) for item in list(post_replay_diff.get("changed_files", []) or []))
        return sorted(path for path in post_files if path and path not in pre_files)

    @staticmethod
    def _removed_stale_diff_files(
        *,
        pre_replay_diff: dict[str, Any],
        post_replay_diff: dict[str, Any],
    ) -> list[str]:
        pre_file_digests = dict(pre_replay_diff.get("file_digests", {}) or {})
        post_file_digests = dict(post_replay_diff.get("file_digests", {}) or {})
        if pre_file_digests or post_file_digests:
            return sorted(path for path in set(pre_file_digests) if path and path not in set(post_file_digests))
        pre_files = set(str(item) for item in list(pre_replay_diff.get("changed_files", []) or []))
        post_files = set(str(item) for item in list(post_replay_diff.get("changed_files", []) or []))
        return sorted(path for path in pre_files if path and path not in post_files)

    @staticmethod
    def _extract_changed_files(patch_summary: dict[str, Any]) -> list[str]:
        raw_values: list[Any] = []
        for key in (
            "changed_files",
            "source_files",
            "non_test_files",
            "test_files",
            "generated_files",
            "other_files",
        ):
            value = patch_summary.get(key, [])
            if isinstance(value, (list, tuple, set)):
                raw_values.extend(value)
        return sorted(
            {
                normalize_repo_path(str(path))
                for path in raw_values
                if str(path).strip()
            }
        )

    @staticmethod
    def _merge_path_lists(*path_lists: Any) -> list[str]:
        merged: list[str] = []
        for paths in path_lists:
            if isinstance(paths, str):
                values = [paths]
            else:
                values = list(paths or [])
            for path in values:
                normalized = normalize_repo_path(str(path))
                if normalized:
                    merged.append(normalized)
        return sorted(dict.fromkeys(merged))

    @staticmethod
    def _infer_patch_scope(
        changed_files: list[str],
        test_files: list[str],
        non_test_files: list[str],
    ) -> str:
        if not changed_files:
            return "no_diff"
        if changed_files and not non_test_files:
            return "tests_only"
        if non_test_files and not test_files:
            return "non_test_only"
        return "mixed"

    @staticmethod
    def _suspect_overlap(changed_files: list[str], suspect_paths: list[str]) -> list[str]:
        overlap: list[str] = []
        for changed in changed_files:
            for suspect in suspect_paths:
                if (
                    changed == suspect
                    or changed.endswith(f"/{suspect}")
                    or suspect.endswith(f"/{changed}")
                ):
                    overlap.append(changed)
                    break
        return sorted(dict.fromkeys(overlap))

    @staticmethod
    def _guard_result_mode(
        *,
        replay_success: bool,
        official_success_deferred: bool,
        patch_scope: str,
        source_files: list[str],
        test_files: list[str],
        generated_files: list[str],
        other_files: list[str],
        touches_suspect_path: bool,
    ) -> str:
        if replay_success:
            if official_success_deferred:
                if patch_scope == "no_diff":
                    return "no_diff_pending_official"
                if test_files and not source_files:
                    return "tests_only_edit_pending_official"
                if source_files and touches_suspect_path and test_files:
                    return "mixed_source_test_pending_official"
                if source_files and touches_suspect_path:
                    return "source_edit_pending_official"
                if source_files:
                    return "off_suspect_source_edit_pending_official"
                if generated_files or other_files:
                    return "wrong_target_edit_pending_official"
                return "unknown_pending_official"
            if patch_scope == "no_diff":
                return "checkpoint_or_no_diff_success"
            if test_files and not source_files:
                return "weak_tests_only_success"
            if source_files and touches_suspect_path and test_files:
                return "mixed_source_test_success"
            if source_files and touches_suspect_path:
                return "strong_source_success"
            if source_files:
                return "weak_off_suspect_source_success"
            if generated_files or other_files:
                return "weak_wrong_target_success"
            return "unknown_success"

        if patch_scope == "no_diff":
            return "no_diff"
        if (generated_files or other_files or test_files) and not source_files and not touches_suspect_path:
            return "wrong_edit_target"
        if source_files and not touches_suspect_path:
            return "source_edit_but_not_suspect"
        if source_files:
            return "oracle_failed_after_source_edit"
        return "wrong_edit_target"

    @staticmethod
    def _guard_flags(
        *,
        replay_success: bool,
        official_success_deferred: bool,
        patch_scope: str,
        source_files: list[str],
        test_files: list[str],
        generated_files: list[str],
        other_files: list[str],
        touches_suspect_path: bool,
    ) -> list[str]:
        flags: list[str] = []
        if patch_scope == "no_diff":
            flags.append("no_diff")
        if test_files:
            flags.append("test_edit_present")
        if test_files and not source_files:
            flags.append("tests_only_edit")
        if generated_files and not source_files:
            flags.append("generated_or_build_edit")
        if other_files and not source_files and not test_files:
            flags.append("other_only_edit")
        if source_files and not touches_suspect_path:
            flags.append("source_edit_but_not_suspect")
        if replay_success and official_success_deferred:
            flags.append("pending_official_validation")
        if replay_success and patch_scope == "no_diff":
            flags.append("weak_no_diff_success")
        if replay_success and test_files:
            flags.append("success_touches_tests")
        return flags

    @staticmethod
    def _execution_quality_flags(
        *,
        observed_source_files: list[str],
        fresh_source_files: list[str],
        replay_success: bool,
        patcher_command_count: int,
        patcher_write_command_count: int,
        patcher_validation_command_count: int,
        patcher_readonly_command_count: int,
        replay_diff_changed: bool,
        removed_stale_diff_files: list[str],
    ) -> list[str]:
        flags: list[str] = []
        if observed_source_files and not replay_diff_changed:
            flags.append("source_diff_unchanged_during_replay")
        if removed_stale_diff_files:
            flags.append("stale_diff_removed_during_replay")
        if removed_stale_diff_files and not fresh_source_files:
            flags.append("stale_diff_removed_without_fresh_source_edit")
        if fresh_source_files and patcher_validation_command_count <= 0:
            flags.append("source_edit_without_focused_validation")
        if fresh_source_files and patcher_write_command_count <= 0:
            flags.append("source_diff_without_observed_write_command")
        if (
            not replay_success
            and patcher_command_count > 0
            and patcher_readonly_command_count >= patcher_command_count
        ):
            flags.append("readonly_replay_exhausted")
        return flags

    @staticmethod
    def _suggest_next_semantic_action(
        *,
        result_mode: str,
        patch_scope: str,
        source_files: list[str],
        touches_suspect_path: bool,
        active_object_type: str = "",
    ) -> str:
        active_object_type = str(active_object_type or "").strip().lower().replace("-", "_")
        if result_mode.endswith("_pending_official") or result_mode == "unknown_pending_official":
            return "STOP_FOR_OFFICIAL_EVAL"
        if result_mode in {
            "strong_source_success",
            "mixed_source_test_success",
            "checkpoint_or_no_diff_success",
        }:
            return "STOP"
        if patch_scope == "no_diff":
            return SemanticActionType.SCOPE_EXPAND.value
        if result_mode == "wrong_edit_target":
            return SemanticActionType.TARGET_RESET.value
        if result_mode == "source_edit_but_not_suspect":
            return SemanticActionType.TARGET_RESET.value
        if result_mode == "oracle_failed_after_source_edit":
            if active_object_type == "shared_fact":
                return SemanticActionType.REVOKE_OBJECT.value
            if active_object_type == "selection":
                return SemanticActionType.CAPABILITY_BOOST.value
            if active_object_type == "verifier_verdict":
                return SemanticActionType.EVIDENCE_RECHECK.value
        if source_files and touches_suspect_path:
            if active_object_type == "shared_fact":
                return SemanticActionType.REVOKE_OBJECT.value
            if active_object_type == "selection":
                return SemanticActionType.CAPABILITY_BOOST.value
            return SemanticActionType.CAPABILITY_BOOST.value
        return SemanticActionType.EVIDENCE_RECHECK.value

    @staticmethod
    def _summarize_guard_history(ledger: RecoveryLedger) -> dict[str, Any]:
        guards = [dict(item) for item in list(ledger.metadata.get("guard_history", []) or [])]
        result_mode_counts: dict[str, int] = {}
        suggestion_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        for guard in guards:
            result_mode = str(guard.get("result_mode", "") or "")
            if result_mode:
                result_mode_counts[result_mode] = result_mode_counts.get(result_mode, 0) + 1
            suggestion = str(guard.get("next_action_suggestion", "") or "")
            if suggestion:
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
            for flag in guard.get("guard_flags", []) or []:
                normalized_flag = str(flag)
                flag_counts[normalized_flag] = flag_counts.get(normalized_flag, 0) + 1
        final_guard = guards[-1] if guards else {}
        final_mode = str(final_guard.get("result_mode", "") or "")
        return {
            "guard_count": len(guards),
            "expensive_replay_count": len(SemanticProgramExecutor._replay_history(ledger)),
            "unproductive_replay_count": SemanticProgramExecutor._unproductive_replay_count(ledger),
            "replay_cost_history": [
                {
                    "action": str(item.get("action", "") or ""),
                    "scope": str(item.get("scope", "") or ""),
                    "execution_profile": str(item.get("execution_profile", "") or ""),
                    "token_cost": float(item.get("token_cost", 0.0) or 0.0),
                    "latency_sec": float(item.get("latency_sec", 0.0) or 0.0),
                }
                for item in SemanticProgramExecutor._replay_history(ledger)[-4:]
            ],
            "conditioning_actions_since_last_replay": SemanticProgramExecutor._conditioning_actions_since_last_replay(
                ledger
            ),
            "result_mode_counts": result_mode_counts,
            "suggestion_counts": suggestion_counts,
            "flag_counts": flag_counts,
            "no_diff_count": flag_counts.get("no_diff", 0),
            "test_edit_count": flag_counts.get("test_edit_present", 0),
            "weak_success_count": sum(
                result_mode_counts.get(mode, 0)
                for mode in (
                    "checkpoint_or_no_diff_success",
                    "weak_tests_only_success",
                    "mixed_source_test_success",
                    "weak_off_suspect_source_success",
                    "weak_wrong_target_success",
                )
            ),
            "pending_official_count": sum(
                result_mode_counts.get(mode, 0)
                for mode in (
                    "source_edit_pending_official",
                    "mixed_source_test_pending_official",
                    "tests_only_edit_pending_official",
                    "no_diff_pending_official",
                    "off_suspect_source_edit_pending_official",
                    "wrong_target_edit_pending_official",
                    "unknown_pending_official",
                )
            ),
            "final_result_mode": final_mode,
            "final_next_action_suggestion": str(final_guard.get("next_action_suggestion", "") or ""),
            "clean_final_success": final_mode == "strong_source_success",
        }

    @staticmethod
    def _should_probe_official_after_source_diff(guard: dict[str, Any]) -> bool:
        """Return whether a fresh source candidate deserves a terminal probe.

        This is a generic closed-loop control rule: when the MAS has produced a
        fresh source edit on the suspected region but the internal verifier says
        "still failing", a pass-only focused test probe can stop wasted follow-up
        actions. A failing probe does not feed new text back into the model; the
        normal action policy continues from the same guard mode.
        """

        result_mode = str(guard.get("result_mode", "") or "")
        if result_mode != "oracle_failed_after_source_edit":
            return False
        fresh_classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
        fresh_source_files = [
            str(item)
            for item in list(fresh_classes.get("source_files", []) or [])
            if str(item).strip()
        ]
        if not fresh_source_files:
            return False
        if not bool(guard.get("decision_touches_suspect_path", guard.get("touches_suspect_path", False))):
            return False
        budget_stop = dict(guard.get("budget_stop", {}) or {})
        return not budget_stop

    def _resolve_source_diff_candidate_with_focused_eval(
        self,
        *,
        coordinator: CoordinatorProtocol,
        guard: dict[str, Any],
        official_eval_commands: dict[str, str],
    ) -> dict[str, Any]:
        updated = dict(guard)
        executor = getattr(coordinator, "executor", None)
        if executor is None:
            updated["closed_loop_focused_eval_status"] = "missing_executor"
            return updated

        test_command = str(official_eval_commands.get("test_command", "") or "").strip()
        if not test_command:
            updated["closed_loop_focused_eval_status"] = "missing_test_command"
            return updated

        workspace = str(getattr(coordinator, "workspace", ""))
        fail_to_pass = executor.execute(test_command, cwd=workspace, timeout=1800)
        focused_passed = int(fail_to_pass.get("returncode", -1)) == 0
        fail_excerpt = self._compact_error_excerpt(fail_to_pass.get("output", ""))
        updated.update(
            {
                "closed_loop_focused_eval_status": "passed" if focused_passed else "failed",
                "closed_loop_candidate_probe_reason": "source_diff_internal_verifier_disagreement",
                "closed_loop_fail_to_pass_returncode": int(fail_to_pass.get("returncode", -1)),
                "closed_loop_focused_eval_passed": focused_passed,
                "closed_loop_fail_to_pass_output_excerpt": fail_excerpt,
            }
        )
        if not focused_passed:
            return updated

        patch_scope = str(updated.get("decision_patch_scope", "") or updated.get("patch_scope", "") or "")
        changed_classes = dict(
            updated.get("fresh_changed_file_classes", {})
            or updated.get("changed_file_classes", {})
            or {}
        )
        source_files = list(changed_classes.get("source_files", []) or [])
        test_files = list(changed_classes.get("test_files", []) or [])
        generated_files = list(changed_classes.get("generated_files", []) or [])
        other_files = list(changed_classes.get("other_files", []) or [])
        touches_suspect_path = bool(
            updated.get("decision_touches_suspect_path", updated.get("touches_suspect_path", False))
        )
        result_mode = self._guard_result_mode(
            replay_success=True,
            official_success_deferred=True,
            patch_scope=patch_scope,
            source_files=source_files,
            test_files=test_files,
            generated_files=generated_files,
            other_files=other_files,
            touches_suspect_path=touches_suspect_path,
        )
        flags = self._guard_flags(
            replay_success=True,
            official_success_deferred=True,
            patch_scope=patch_scope,
            source_files=source_files,
            test_files=test_files,
            generated_files=generated_files,
            other_files=other_files,
            touches_suspect_path=touches_suspect_path,
        )
        updated.update(
            {
                "replay_success": True,
                "official_success_deferred": True,
                "result_mode": result_mode,
                "failure_mode": result_mode,
                "success_legitimacy": "pending_official_validation",
                "guard_flags": flags,
                "next_action_suggestion": "STOP_FOR_OFFICIAL_EVAL",
            }
        )
        return updated

    def _resolve_pending_guard_with_official_eval(
        self,
        *,
        coordinator: CoordinatorProtocol,
        guard: dict[str, Any],
        official_eval_commands: dict[str, str],
        reason: str = "pending_internal_verifier",
    ) -> dict[str, Any]:
        updated = dict(guard)
        executor = getattr(coordinator, "executor", None)
        if executor is None:
            updated["closed_loop_official_eval_status"] = "missing_executor"
            return updated

        test_command = str(official_eval_commands.get("test_command", "") or "").strip()
        oracle_command = str(official_eval_commands.get("oracle_command", "") or "").strip()
        workspace = str(getattr(coordinator, "workspace", ""))
        fail_to_pass = {"returncode": -1, "output": ""}
        oracle = {"returncode": -1, "output": ""}
        if test_command:
            fail_to_pass = executor.execute(test_command, cwd=workspace, timeout=1800)
        if oracle_command:
            oracle = executor.execute(oracle_command, cwd=workspace, timeout=1800)

        actual_success = int(oracle.get("returncode", -1)) == 0
        fail_excerpt = self._compact_error_excerpt(fail_to_pass.get("output", ""))
        oracle_excerpt = self._compact_error_excerpt(oracle.get("output", ""))
        patch_scope = str(updated.get("decision_patch_scope", "") or updated.get("patch_scope", "") or "")
        changed_classes = dict(
            updated.get("fresh_changed_file_classes", {})
            or updated.get("changed_file_classes", {})
            or {}
        )
        source_files = list(changed_classes.get("source_files", []) or [])
        test_files = list(changed_classes.get("test_files", []) or [])
        generated_files = list(changed_classes.get("generated_files", []) or [])
        other_files = list(changed_classes.get("other_files", []) or [])
        touches_suspect_path = bool(
            updated.get("decision_touches_suspect_path", updated.get("touches_suspect_path", False))
        )

        result_mode = self._guard_result_mode(
            replay_success=actual_success,
            official_success_deferred=False,
            patch_scope=patch_scope,
            source_files=source_files,
            test_files=test_files,
            generated_files=generated_files,
            other_files=other_files,
            touches_suspect_path=touches_suspect_path,
        )
        flags = self._guard_flags(
            replay_success=actual_success,
            official_success_deferred=False,
            patch_scope=patch_scope,
            source_files=source_files,
            test_files=test_files,
            generated_files=generated_files,
            other_files=other_files,
            touches_suspect_path=touches_suspect_path,
        )
        updated.update(
            {
                "replay_success": actual_success,
                "official_success_deferred": False,
                "result_mode": result_mode,
                "failure_mode": "" if actual_success else result_mode,
                "success_legitimacy": result_mode if actual_success else "not_resolved",
                "guard_flags": flags,
                "next_action_suggestion": self._suggest_next_semantic_action(
                    result_mode=result_mode,
                    patch_scope=patch_scope,
                    source_files=source_files,
                    touches_suspect_path=touches_suspect_path,
                ),
                "closed_loop_official_eval_status": "resolved",
                "closed_loop_official_eval_reason": str(reason or "pending_internal_verifier"),
                "closed_loop_fail_to_pass_returncode": int(fail_to_pass.get("returncode", -1)),
                "closed_loop_oracle_returncode": int(oracle.get("returncode", -1)),
                "closed_loop_fail_to_pass_output_excerpt": fail_excerpt,
                "closed_loop_oracle_output_excerpt": oracle_excerpt,
            }
        )
        return updated

    def _compose_semantic_replay_hint(
        self,
        *,
        ledger: RecoveryLedger,
        scope: str,
        repair_mode: str,
        execution_profile: str = "normal",
        respect_active_target: bool,
    ) -> str:
        lines = [
            "This is a semantic recovery replay, not a clean-start solve.",
            f"Repair mode: {repair_mode}",
            f"Replay scope: {scope}",
            f"Execution profile: {execution_profile}",
        ]
        if str(execution_profile or "").strip().lower() == "compact":
            lines.extend(
                [
                    "Use the smallest useful repair loop: inspect the active target briefly, edit canonical source, run the focused test if available, then stop.",
                    "Do not spend the compact pass on broad repository exploration or repeated read-only probes.",
                ]
            )
        replay_precondition = str(ledger.metadata.get("current_replay_precondition", "") or "").strip()
        if replay_precondition == "evidence_bounded_scope_expand":
            lines.extend(
                [
                    "Evidence-bounded scope expansion: use the latest failing assertion and suspect paths to relocalize only the smallest adjacent source boundary.",
                    "The expanded pass must end in a fresh source diff or a focused validation-backed explanation for why no source edit is possible.",
                ]
            )
        if replay_precondition == "post_evidence_source_repair":
            lines.extend(
                [
                    "Post-evidence source repair: CAR already refreshed the failure evidence and selected LOCAL_REPAIR, so this replay should not spend the attempt on broad read-only diagnosis.",
                    "Use the latest evidence to make a fresh source diff on the intended source boundary, then run focused validation when available.",
                    "If the intended boundary is impossible to edit, say which evidence blocks the edit instead of silently returning no diff.",
                ]
            )
        if ledger.failing_tests_summary:
            lines.append(f"Current failing tests: {', '.join(ledger.failing_tests_summary[:4])}")
        test_command = str(ledger.metadata.get("test_command", "") or "").strip()
        if test_command:
            lines.append(f"Focused verification command: {test_command}")
        if ledger.suspect_paths:
            lines.append(f"Suspect source paths: {', '.join(ledger.suspect_paths[:5])}")
            lines.append(
                "Patch boundary: modify only the smallest necessary subset of these suspect source paths. "
                "Do not touch tests, generated outputs, documentation, or broad repository files unless the focused evidence makes them the source boundary."
            )
        if respect_active_target and ledger.active_target:
            lines.append(f"Current active repair target: {ledger.active_target}")
            lines.append(
                "Active-target discipline: prefer a fresh source diff on the current active repair target before broadening to another file."
            )
        if ledger.active_object_type:
            active_object = ledger.active_object_type
            if ledger.active_object_id:
                active_object = f"{active_object} ({ledger.active_object_id})"
            lines.append(f"Current active recovery object: {active_object}")
        if ledger.invalidated_targets:
            lines.append(f"Invalidated targets or beliefs: {', '.join(ledger.invalidated_targets[:6])}")
        if ledger.invalidated_object_ids:
            lines.append(f"Invalidated object ids: {', '.join(ledger.invalidated_object_ids[:6])}")
        best_candidate = self._best_source_candidate(ledger)
        if best_candidate:
            candidate_files = [
                str(item)
                for item in list(best_candidate.get("fresh_source_files", []) or best_candidate.get("source_files", []) or [])
                if str(item).strip()
            ]
            if candidate_files:
                if "locator" in str(scope or "").lower() or str(repair_mode or "") == "expanded":
                    lines.append(
                        "Earlier source candidate to audit during expanded localization: "
                        + ", ".join(candidate_files[:4])
                    )
                else:
                    lines.append(
                        "Best preserved source candidate from earlier recovery: "
                        + ", ".join(candidate_files[:4])
                    )
            candidate_mode = str(best_candidate.get("result_mode", "") or "").strip()
            if candidate_mode:
                if "locator" in str(scope or "").lower() or str(repair_mode or "") == "expanded":
                    lines.append(
                        f"Candidate status: {candidate_mode}; treat it as evidence, not a lock. "
                        "Keep, revise, or replace it according to refreshed localization and failing-test evidence."
                    )
                else:
                    lines.append(f"Candidate status: {candidate_mode}; revise or preserve it instead of discarding it silently.")
            candidate_guard_flags = [
                str(item)
                for item in list(best_candidate.get("guard_flags", []) or [])
                if str(item).strip()
            ]
            if candidate_guard_flags:
                lines.append(
                    "Candidate guard feedback: "
                    + "; ".join(candidate_guard_flags[:6])
                )
        compact_constraints = [
            " ".join(str(item or "").split())[:220]
            for item in list(ledger.negative_constraints or [])[:5]
            if str(item or "").strip()
        ]
        if compact_constraints:
            lines.append(f"Negative constraints from previous failed recovery: {'; '.join(compact_constraints)}")
        reflection_memory = [
            dict(item)
            for item in list(ledger.metadata.get("reflection_memory", []) or [])
            if isinstance(item, dict)
        ][-4:]
        if reflection_memory:
            lines.append("Structured recovery reflections from previous failed actions:")
            for idx, reflection in enumerate(reflection_memory, start=1):
                after_action = str(reflection.get("after_action", "") or "unknown")
                result_mode = str(reflection.get("result_mode", "") or "unknown")
                avoid_rule = str(reflection.get("avoid_rule", "") or "").strip()
                prefer_next = str(reflection.get("prefer_next_action", "") or "").strip()
                target = str(reflection.get("active_target", "") or "").strip()
                entry = f"  [{idx}] {after_action} -> {result_mode}"
                if target:
                    entry += f" on {target}"
                if avoid_rule:
                    entry += f"; avoid: {avoid_rule}"
                if prefer_next:
                    entry += f"; prefer next: {prefer_next}"
                lines.append(entry[:360])
        if ledger.key_evidence:
            lines.append(f"Most recent evidence: {' '.join(str(ledger.key_evidence[0]).split())[:600]}")
        if str(repair_mode or "").startswith("candidate_preserving"):
            latest_guard = self._latest_guard(ledger)
            validation_excerpt = self._validation_evidence_excerpt_from_guard(latest_guard, limit=1100)
            lines.extend(
                [
                    "Candidate-preserving refine: start from the preserved candidate diff, keep useful source edits, and make the smallest revision needed for the latest focused/official failure.",
                    "Do not restart broad exploration unless the candidate file is impossible to validate from the current evidence.",
                    "Candidate-preserving refine is edit-or-validate driven: after at most one brief inspection, either edit the preserved candidate source or run the focused failing test.",
                ]
            )
            if validation_excerpt:
                lines.append(
                    "Validation failure to repair now: "
                    + " ".join(validation_excerpt.split())[:1100]
                )
        lines.extend(
            [
                "Prioritize canonical source files before tests or generated outputs.",
                "After a source edit, run the focused failing test command when available; do not stop at syntax-only validation unless tests are unavailable.",
                "Do not stop without a concrete source diff when recovery still targets a source bug.",
                "If the previous local target was contradicted, rebuild it from the current ledger instead of trusting stale conclusions.",
                "Keep the recovery patch small: normally one source file, at most three fresh source files, and no fresh test/generated file edits.",
            ]
        )
        return "\n".join(lines)
