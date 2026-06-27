"""Dual-layer recovery action language for BCMR.

This module defines:

1. Upper-layer semantic recovery actions
2. Lower-layer execution primitives
3. The compiler from semantic programs to primitive programs
4. Bootstrap logic for the structured recovery ledger
"""

from __future__ import annotations

from typing import Any

from bcmr_swe.recovery.object_lifecycle import initialize_lifecycle_state
from bcmr_swe.recovery.structured_state import build_structured_recovery_state_from_failed_state
from bcmr_swe.types import (
    FailedState,
    PrimitiveOpType,
    PrimitiveProgram,
    PrimitiveStep,
    RecoveryBudget,
    RecoveryLedger,
    SemanticActionType,
    SemanticRecoveryProgram,
    SemanticRecoveryStep,
)
from swe_mas.utils.path_filters import existing_repo_source_paths, ordered_canonical_source_paths


def _normalize_object_type(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "sharedfact": "shared_fact",
        "shared_fact": "shared_fact",
        "selection": "selection",
        "verifierverdict": "verifier_verdict",
        "verifier_verdict": "verifier_verdict",
    }
    return aliases.get(token, token)


def _ledger_object_chain(ledger: RecoveryLedger) -> list[dict[str, Any]]:
    structured_state = dict(ledger.structured_state or {})
    raw_chain = structured_state.get("object_chain_view") or []
    if isinstance(raw_chain, dict):
        raw_chain = raw_chain.get("objects") or []
    return [dict(item) for item in list(raw_chain or []) if isinstance(item, dict)]


def _active_object_payload(ledger: RecoveryLedger) -> dict[str, Any]:
    active_id = str(ledger.active_object_id or "").strip()
    active_type = _normalize_object_type(ledger.active_object_type)
    fallback: dict[str, Any] = {}
    for item in _ledger_object_chain(ledger):
        item_type = _normalize_object_type(item.get("object_type", ""))
        if active_id and str(item.get("object_id", "") or "").strip() == active_id:
            return dict(item.get("payload", {}) or {})
        if active_type and item_type == active_type and not fallback:
            fallback = dict(item.get("payload", {}) or {})
    return fallback


def _inspect_target_for_active_object(
    ledger: RecoveryLedger,
    *,
    explicit_target: Any = "",
) -> str:
    explicit = str(explicit_target or "").strip()
    if explicit and not explicit.startswith("object:"):
        return explicit

    active_type = _normalize_object_type(ledger.active_object_type)
    payload = _active_object_payload(ledger)
    if active_type == "shared_fact":
        fact_key = (
            str(payload.get("fact_key", "") or "").strip()
            or str(ledger.latest_shared_fact_key or "").strip()
            or str(payload.get("node_id", "") or "").strip()
            or "latest_patch"
        )
        if fact_key.startswith("object:"):
            fact_key = "latest_patch"
        return f"fact:{fact_key}" if fact_key else "test_output"
    if active_type == "selection":
        return "localization"
    if active_type == "verifier_verdict":
        return "test_output"
    if explicit.startswith("object:"):
        return "test_output"
    return explicit or "test_output"


def _first_suspicious_object(objects: list[Any]) -> Any | None:
    preferred = {"SharedFact": 0, "Selection": 1, "VerifierVerdict": 2}
    candidates = []
    for index, item in enumerate(objects):
        status = str(getattr(item, "contamination_status", "") or "").strip().lower()
        object_type = str(getattr(item, "object_type", "") or "")
        if status in {"resolved", "revalidated", "observed"}:
            continue
        candidates.append((preferred.get(object_type, 99), index, item))
    if not candidates:
        return objects[0] if objects else None
    return sorted(candidates, key=lambda row: (row[0], row[1]))[0][2]


def _object_excerpt(obj: Any) -> str:
    payload = dict(getattr(obj, "payload", {}) or {})
    for key in ("fact_value_excerpt", "fact_key", "path", "output_excerpt", "verdict"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value[:160]
    return str(getattr(obj, "evidence_anchor", "") or "")[:160]


def bootstrap_recovery_ledger(
    failed_state: FailedState,
    budget: RecoveryBudget,
    *,
    step_budget: int = 4,
) -> RecoveryLedger:
    structured_state = build_structured_recovery_state_from_failed_state(failed_state)
    initialize_lifecycle_state(structured_state)
    observation = dict(failed_state.metadata.get("failure_observation", {}) or {})
    failing_tests = [
        str(item)
        for item in observation.get("failing_tests", []) or []
        if str(item).strip()
    ]
    touched_paths = [
        str(item).replace("\\", "/")
        for item in list(failed_state.metadata.get("touched_paths", []) or [])
        if str(item).strip()
    ]
    verifier_excerpt = str(observation.get("verifier_output_excerpt", "") or "").strip()
    trigger_reason = str(failed_state.trigger.reason or failed_state.trigger.trigger_type.value)
    summary = dict(failed_state.suspect_region.summary or {})
    replay_anchor_role = str(summary.get("replay_anchor_role", "") or "").strip().lower()
    trigger_type = failed_state.trigger.trigger_type
    object_chain = list(structured_state.object_chain_view or [])
    local_region = dict(structured_state.local_region_view or {})
    evidence_pack = dict(structured_state.evidence_pack or {})
    state_target_candidates = [
        str(item).replace("\\", "/")
        for item in list(local_region.get("selected_target_candidates", []) or [])
        if str(item).strip()
    ]
    candidate_paths = state_target_candidates + touched_paths
    workspace = str(failed_state.checkpoint.workspace or "") if failed_state.checkpoint else ""
    suspect_paths = existing_repo_source_paths(candidate_paths, workspace) or ordered_canonical_source_paths(candidate_paths)
    active_target = suspect_paths[0] if suspect_paths else ""

    active_object_type = ""
    if trigger_type.value == "fact_conflict":
        active_object_type = "shared_fact"
    elif trigger_type.value == "verifier_contradiction":
        active_object_type = "verifier_verdict"
    elif bool(failed_state.metadata.get("wrong_localized_path")) or replay_anchor_role == "locator":
        active_object_type = "selection"

    active_object_excerpt = ""
    active_object_id = ""
    if active_object_type == "selection":
        active_object_excerpt = active_target
    elif active_object_type == "verifier_verdict":
        active_object_excerpt = verifier_excerpt[:160]
    elif active_object_type == "shared_fact":
        active_object_excerpt = str(summary.get("conflicting_fact_key", "") or "")
    for item in object_chain:
        normalized_type = str(item.object_type or "").strip().lower()
        if active_object_type == "selection" and normalized_type == "selection":
            active_object_id = item.object_id
            active_object_excerpt = active_object_excerpt or str(item.payload.get("fact_value_excerpt", "") or "")
            break
        if active_object_type == "verifier_verdict" and normalized_type == "verifierverdict":
            active_object_id = item.object_id
            active_object_excerpt = active_object_excerpt or str(item.payload.get("output_excerpt", "") or "")
            break
        if active_object_type == "shared_fact" and normalized_type == "sharedfact":
            active_object_id = item.object_id
            active_object_excerpt = active_object_excerpt or str(item.payload.get("fact_key", "") or "")
            break

    if not active_object_id and object_chain:
        fallback = _first_suspicious_object(object_chain)
        if fallback is not None:
            active_object_id = fallback.object_id
            active_object_type = _normalize_object_type(fallback.object_type)
            active_object_excerpt = _object_excerpt(fallback)

    active_object_payload: dict[str, Any] = {}
    if active_object_id:
        for item in object_chain:
            if str(item.object_id or "") == active_object_id:
                active_object_payload = dict(item.payload or {})
                break

    latest_shared_fact_key = ""
    if active_object_type == "shared_fact":
        latest_shared_fact_key = (
            str(active_object_payload.get("fact_key", "") or "").strip()
            or str(summary.get("conflicting_fact_key", "") or "").strip()
        )
    if not latest_shared_fact_key and "latest_patch" in dict(failed_state.metadata.get("phase_outputs", {}) or {}):
        latest_shared_fact_key = "latest_patch"

    latest_verifier_verdict = str(
        dict(failed_state.metadata.get("latest_test_status", {}) or {}).get("status", "")
        or evidence_pack.get("verifier_excerpt", "")
        or ""
    )
    negative_constraints = [
        str(item)
        for item in list(local_region.get("negative_constraints", []) or [])
        if str(item).strip()
    ]

    return RecoveryLedger(
        trigger_reason=trigger_reason,
        failing_tests_summary=failing_tests[:6],
        active_target=active_target,
        active_object_type=active_object_type,
        active_object_id=active_object_id,
        active_object_excerpt=active_object_excerpt,
        suspect_paths=suspect_paths[:8],
        invalidated_targets=[],
        invalidated_object_ids=[],
        key_evidence=[verifier_excerpt[:300]] if verifier_excerpt else [],
        latest_verifier_verdict=latest_verifier_verdict,
        latest_shared_fact_key=latest_shared_fact_key,
        negative_constraints=negative_constraints,
        execution_profile="normal",
        last_action="",
        last_action_result={},
        last_source_edit_summary={},
        touches_suspect_path=False,
        tried_actions=[],
        remaining_step_budget=max(0, int(step_budget)),
        remaining_token_budget=float(budget.token_budget),
        remaining_latency_budget_sec=float(budget.latency_budget_sec),
        structured_state=structured_state.to_dict(),
        metadata={
            "instance_id": failed_state.instance_id,
            "fault_type": str(failed_state.metadata.get("fault_type", "")),
            "test_command": str(observation.get("test_command", "") or ""),
            "checkpoint_candidates": list(failed_state.metadata.get("checkpoint_candidates", []) or []),
            "phase_outputs": dict(failed_state.metadata.get("phase_outputs", {}) or {}),
        },
    )


def semantic_programs_v1(
    checkpoint_ids: dict[str, str],
    *,
    profile: str = "full_matrix",
    fault_family: str | None = None,
) -> list[SemanticRecoveryProgram]:
    """Discovery-time semantic candidate language.

    The set is intentionally compact:
    - a minimal restore
    - evidence-first local repair
    - target reset + local repair
    - scope expansion
    - capability boost
    - global reset rebuild
    """

    normalized_fault_family = str(fault_family or "").strip().lower()
    programs = [
        SemanticRecoveryProgram(
            program_id="semantic_local_restore",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_patch",
                        "checkpoint_id": checkpoint_ids.get("post_patch", ""),
                        "checkpoint_label": "post_patch",
                        "invalidate": ["latest_patch"],
                        "reset_active_target": False,
                    },
                ),
            ],
            rationale="Restore the last healthy patch anchor as the cheapest recovery when the latest failure looks like a local regression.",
            estimated_total_cost=100.0,
            estimated_recover_prob=0.94,
            estimated_risk=0.02,
            metadata={
                "family": "semantic_local_restore",
                "strategy": "semantic_target_reset_post_patch",
                "scope": "restore_only",
                "program_space_version": "semantic_dual_v1",
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_evidence_local_repair",
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
            rationale="Refresh the failure evidence first, then perform a constrained local repair.",
            estimated_total_cost=1900.0,
            estimated_recover_prob=0.72,
            estimated_risk=0.08,
            metadata={
                "family": "semantic_evidence_recheck",
                "strategy": "semantic_evidence_then_local_repair",
                "scope": "patcher+verifier",
                "program_space_version": "semantic_dual_v1",
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_target_reset_local_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": checkpoint_ids.get("post_locate", ""),
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
            rationale="Reset the local target and rebuild the downstream patch-and-verify region from the localization anchor.",
            estimated_total_cost=2300.0,
            estimated_recover_prob=0.79,
            estimated_risk=0.10,
            metadata={
                "family": "semantic_target_reset",
                "strategy": "semantic_reset_then_local_repair",
                "scope": "patcher+verifier",
                "program_space_version": "semantic_dual_v1",
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_scope_expand_rebuild",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "post_locate",
                        "checkpoint_id": checkpoint_ids.get("post_locate", ""),
                        "checkpoint_label": "post_locate",
                        "invalidate": ["localized_path", "current_target"],
                        "reset_active_target": True,
                    },
                ),
                SemanticRecoveryStep(
                    action=SemanticActionType.SCOPE_EXPAND,
                    args={
                        "scope": "locator+patcher+verifier",
                        "strategy": "broader_search",
                        "expansion_level": "regional",
                        "execution_profile": "compact",
                        "replay_precondition": "evidence_bounded_scope_expand",
                        "pre_evidence_recheck": True,
                        "focused_verify": True,
                        "target": "test_output",
                        "depth": "deep",
                    },
                ),
            ],
            rationale="Discard the stale target and expand recovery scope to relocalize before rebuilding.",
            estimated_total_cost=3200.0,
            estimated_recover_prob=0.70,
            estimated_risk=0.14,
            metadata={
                "family": "semantic_scope_expand",
                "strategy": "semantic_reset_then_scope_expand",
                "scope": "locator+patcher+verifier",
                "program_space_version": "semantic_dual_v1",
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_capability_boost_local_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.CAPABILITY_BOOST,
                    args={
                        "scope": "patcher",
                        "strategy": "stronger_prompt",
                        "level": 1,
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
            rationale="Keep the local repair target but increase repair capability before retrying.",
            estimated_total_cost=2600.0,
            estimated_recover_prob=0.74,
            estimated_risk=0.12,
            metadata={
                "family": "semantic_capability_boost",
                "strategy": "semantic_boost_then_local_repair",
                "scope": "patcher+verifier",
                "program_space_version": "semantic_dual_v1",
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_global_reset_rebuild",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "initial",
                        "checkpoint_id": checkpoint_ids.get("initial", ""),
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
            rationale="Reset to the clean initial state and restart the full pipeline when local recovery is no longer trustworthy.",
            estimated_total_cost=6200.0,
            estimated_recover_prob=0.73,
            estimated_risk=0.24,
            metadata={
                "family": "semantic_global_reset",
                "strategy": "semantic_global_reset_then_rebuild",
                "scope": "full",
                "program_space_version": "semantic_dual_v1",
            },
        ),
    ]

    normalized_profile = str(profile or "full_matrix").strip().lower()

    if normalized_profile == "anchor_restore":
        if normalized_fault_family == "contaminated_post_patch":
            return [programs[5]]
        return [programs[0], programs[5]]

    if normalized_profile == "local_rebuild":
        local_subset = [programs[1], programs[2], programs[4], programs[5]]
        if normalized_fault_family == "post_patch_regression":
            return [programs[1], programs[2], programs[5]]
        return local_subset

    if normalized_fault_family == "post_patch_regression":
        return [programs[0], programs[1], programs[2], programs[4], programs[5]]
    if normalized_fault_family == "contaminated_post_patch":
        return [programs[1], programs[2], programs[3], programs[4], programs[5]]
    return programs


def semantic_closed_loop_programs_v1(
    checkpoint_ids: dict[str, str],
    *,
    profile: str = "full_matrix",
    fault_family: str | None = None,
) -> list[SemanticRecoveryProgram]:
    """Closed-loop semantic starters.

    We intentionally reuse the validated v1 semantic fragments as starting
    points, but mark them as closed-loop seeds so the executor can keep
    selecting the next fragment from the recovery ledger instead of stopping
    after the first fragment fails.
    """

    programs = semantic_programs_v1(
        checkpoint_ids,
        profile=profile,
        fault_family=fault_family,
    )
    closed_loop: list[SemanticRecoveryProgram] = []
    for program in programs:
        metadata = dict(program.metadata)
        metadata.update(
            {
                "program_space_version": "semantic_closed_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_step_budget": max(4, len(program.steps) + 2),
                "closed_loop_max_fragments": 2,
            }
        )
        closed_loop.append(
            SemanticRecoveryProgram(
                program_id=program.program_id,
                steps=list(program.steps),
                rationale=program.rationale,
                estimated_total_cost=program.estimated_total_cost,
                estimated_recover_prob=program.estimated_recover_prob,
                estimated_risk=program.estimated_risk,
                metadata=metadata,
            )
        )
    return closed_loop


def semantic_action_loop_programs_v1(
    checkpoint_ids: dict[str, str],
    *,
    profile: str = "full_matrix",
    fault_family: str | None = None,
) -> list[SemanticRecoveryProgram]:
    """Action-level closed-loop starters.

    Unlike ``semantic_closed_loop_v1``, which reuses validated two-step
    fragments as closed-loop seeds, this version makes the decision unit a
    single upper-layer semantic action. The executor is then responsible for
    reading the recovery ledger after each action and selecting the next
    action.
    """

    normalized_profile = str(profile or "full_matrix").strip().lower()
    normalized_fault_family = str(fault_family or "").strip().lower()
    if normalized_profile not in {"anchor_restore", "local_rebuild", "full_matrix"}:
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    target_reset_anchor = "post_patch" if normalized_fault_family == "post_patch_regression" else "post_locate"
    target_reset_checkpoint_id = str(checkpoint_ids.get(target_reset_anchor, "") or "")

    starters = [
        SemanticRecoveryProgram(
            program_id="semantic_action_object_recheck_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.RECHECK_OBJECT,
                    args={"depth": "deep", "focused_verify": True},
                ),
            ],
            rationale="Start by rechecking the active polluted propagation object from the structured ledger.",
            estimated_total_cost=850.0,
            estimated_recover_prob=0.68,
            estimated_risk=0.06,
            metadata={
                "family": "semantic_object_recheck",
                "strategy": "semantic_action_loop_object_recheck_start",
                "scope": "object+verify",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_object_revoke_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "invalidate": ["latest_patch", "current_target"],
                        "checkpoint_id": target_reset_checkpoint_id,
                        "checkpoint_label": target_reset_anchor,
                        "anchor": target_reset_anchor,
                        "reset_active_target": True,
                    },
                ),
            ],
            rationale="Revoke the active suspicious propagation object before any new local repair.",
            estimated_total_cost=1050.0,
            estimated_recover_prob=0.73,
            estimated_risk=0.08,
            metadata={
                "family": "semantic_object_revoke",
                "strategy": "semantic_action_loop_object_revoke_start",
                "scope": "object_reset",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_evidence_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.EVIDENCE_RECHECK,
                    args={"target": "test_output", "depth": "deep", "focused_verify": True},
                ),
            ],
            rationale="Start with one evidence-refresh action, then let the recovery ledger choose the next action.",
            estimated_total_cost=900.0,
            estimated_recover_prob=0.66,
            estimated_risk=0.07,
            metadata={
                "family": "semantic_evidence_recheck",
                "strategy": "semantic_action_loop_evidence_start",
                "scope": "read+verify",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_target_reset_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": target_reset_anchor,
                        "checkpoint_id": target_reset_checkpoint_id,
                        "checkpoint_label": target_reset_anchor,
                        "invalidate": ["latest_patch", "current_target"],
                        "reset_active_target": True,
                    },
                ),
            ],
            rationale="Reset one local target first, then let the recovery ledger decide whether to repair, expand, or boost.",
            estimated_total_cost=1100.0,
            estimated_recover_prob=0.74,
            estimated_risk=0.08,
            metadata={
                "family": "semantic_target_reset",
                "strategy": "semantic_action_loop_target_reset_start",
                "scope": "reset_only",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_scope_expand_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.SCOPE_EXPAND,
                    args={
                        "scope": "locator+patcher+verifier",
                        "strategy": "broader_search",
                        "expansion_level": "regional",
                        "execution_profile": "compact",
                        "replay_precondition": "evidence_bounded_scope_expand",
                        "pre_evidence_recheck": True,
                        "focused_verify": True,
                        "target": "test_output",
                        "depth": "deep",
                    },
                ),
            ],
            rationale="Use one explicit scope-expansion action when local repair looks too narrow.",
            estimated_total_cost=2500.0,
            estimated_recover_prob=0.64,
            estimated_risk=0.14,
            metadata={
                "family": "semantic_scope_expand",
                "strategy": "semantic_action_loop_scope_expand_start",
                "scope": "locator+patcher+verifier",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_capability_boost_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.CAPABILITY_BOOST,
                    args={
                        "scope": "patcher",
                        "strategy": "stronger_prompt",
                        "level": 1,
                    },
                ),
            ],
            rationale="Spend one explicit action on local capability before deciding whether a repair retry is worthwhile.",
            estimated_total_cost=1300.0,
            estimated_recover_prob=0.62,
            estimated_risk=0.11,
            metadata={
                "family": "semantic_capability_boost",
                "strategy": "semantic_action_loop_capability_start",
                "scope": "patcher",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_action_global_reset_start",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.TARGET_RESET,
                    args={
                        "anchor": "initial",
                        "checkpoint_id": str(checkpoint_ids.get("initial", "") or ""),
                        "checkpoint_label": "initial",
                        "invalidate": ["all"],
                        "reset_active_target": True,
                    },
                ),
            ],
            rationale="Reset to the global anchor first, then allow the ledger to decide whether a full restart is worth paying for.",
            estimated_total_cost=1800.0,
            estimated_recover_prob=0.58,
            estimated_risk=0.18,
            metadata={
                "family": "semantic_global_reset",
                "strategy": "semantic_action_loop_global_reset_start",
                "scope": "global_reset_only",
                "program_space_version": "semantic_action_loop_v1",
                "closed_loop_enabled": True,
                "closed_loop_unit": "action",
                "closed_loop_step_budget": 4,
                "closed_loop_max_units": 4,
                "closed_loop_max_replays": 3,
                "closed_loop_max_unproductive_replays": 2,
            },
        ),
    ]

    if normalized_profile == "anchor_restore":
        return [starters[6]]
    if normalized_profile == "local_rebuild":
        return [starters[0], starters[1], starters[2], starters[3], starters[5]]
    if normalized_fault_family == "post_patch_regression":
        return [
            starters[0],
            starters[1],
            starters[2],
            starters[3],
            starters[5],
            starters[6],
        ]
    if normalized_fault_family == "contaminated_post_patch":
        return starters
    return starters


def semantic_object_loop_programs_v2(
    checkpoint_ids: dict[str, str],
    *,
    profile: str = "full_matrix",
    fault_family: str | None = None,
) -> list[SemanticRecoveryProgram]:
    """Object-centric macro-action candidates for the next closed-loop line.

    This is intentionally not a replacement for the frozen open-loop baseline.
    It keeps the inner execution as short programs while making the outer
    decision unit an object-aware recovery intent.
    """

    normalized_profile = str(profile or "full_matrix").strip().lower()
    normalized_fault_family = str(fault_family or "").strip().lower()
    if normalized_profile not in {"anchor_restore", "local_rebuild", "full_matrix"}:
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    object_type = "shared_fact"
    object_id = "fact:latest_patch"
    if normalized_fault_family == "localization_pollution":
        object_type = "selection"
        object_id = "fact:localized_path"
    elif normalized_fault_family == "post_patch_regression":
        object_type = "verifier_verdict"
        object_id = "verifier:latest"

    post_locate_id = str(checkpoint_ids.get("post_locate", "") or "")
    initial_id = str(checkpoint_ids.get("initial", "") or "")
    common_metadata = {
        "program_space_version": "semantic_object_loop_v2",
        "macro_action_level": "object",
        "closed_loop_candidate": True,
        "closed_loop_enabled": False,
        "object_type": object_type,
        "object_id": object_id,
    }

    programs = [
        SemanticRecoveryProgram(
            program_id="semantic_object_recheck_then_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.RECHECK_OBJECT,
                    args={
                        "object_type": object_type,
                        "object_id": object_id,
                        "focused_verify": True,
                    },
                ),
                SemanticRecoveryStep(
                    action=SemanticActionType.REPAIR_LOCAL,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "object_rechecked_local",
                        "respect_active_target": True,
                        "execution_profile": "normal",
                    },
                ),
            ],
            rationale="Recheck the active propagation object, then run one constrained local repair macro.",
            estimated_total_cost=2200.0,
            estimated_recover_prob=0.75,
            estimated_risk=0.08,
            metadata={
                **common_metadata,
                "family": "semantic_object_recheck",
                "strategy": "object_recheck_then_repair",
                "scope": "patcher+verifier",
                "postcondition_contract": {
                    "object_status": "supported|refuted|unchanged|inconclusive",
                    "repair_result": "source_edit|no_diff|test_edit|oracle_passed|oracle_failed",
                },
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_object_revoke_then_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "object_type": object_type,
                        "object_id": object_id,
                        "invalidate": [object_id, "latest_patch", "current_target"],
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                    },
                ),
                SemanticRecoveryStep(
                    action=SemanticActionType.REPAIR_LOCAL,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "object_revoked_local",
                        "respect_active_target": True,
                        "execution_profile": "normal",
                    },
                ),
            ],
            rationale="Revoke the suspicious propagated object and rebuild the affected local cone.",
            estimated_total_cost=2450.0,
            estimated_recover_prob=0.78,
            estimated_risk=0.09,
            metadata={
                **common_metadata,
                "family": "semantic_object_revoke",
                "strategy": "object_revoke_then_repair",
                "scope": "patcher+verifier",
                "postcondition_contract": {
                    "revoked_object_id": object_id,
                    "affected_cone": "local_downstream",
                    "repair_result": "source_edit|no_diff|test_edit|oracle_passed|oracle_failed",
                },
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_object_expand_then_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.EXPAND_SCOPE,
                    args={
                        "object_type": object_type,
                        "object_id": object_id,
                        "scope": "locator+patcher+verifier",
                        "strategy": "broader_search",
                        "expansion_level": "regional",
                        "execution_profile": "compact",
                        "replay_precondition": "evidence_bounded_scope_expand",
                        "pre_evidence_recheck": True,
                        "focused_verify": True,
                        "target": "test_output",
                        "depth": "deep",
                    },
                ),
            ],
            rationale="Expand around the active object when the current local cone is too narrow.",
            estimated_total_cost=3300.0,
            estimated_recover_prob=0.68,
            estimated_risk=0.14,
            metadata={
                **common_metadata,
                "family": "semantic_object_expand",
                "strategy": "object_expand_then_repair",
                "scope": "locator+patcher+verifier",
                "postcondition_contract": {
                    "expanded_from": object_id,
                    "expanded_to": "regional_cone",
                    "new_candidate_objects": "selection|shared_fact|verifier_verdict",
                },
            },
        ),
        SemanticRecoveryProgram(
            program_id="semantic_object_revoke_then_boosted_repair",
            steps=[
                SemanticRecoveryStep(
                    action=SemanticActionType.REVOKE_OBJECT,
                    args={
                        "object_type": object_type,
                        "object_id": object_id,
                        "invalidate": [object_id, "latest_patch", "current_target"],
                        "checkpoint_id": post_locate_id,
                        "checkpoint_label": "post_locate",
                        "anchor": "post_locate",
                    },
                ),
                SemanticRecoveryStep(
                    action=SemanticActionType.REPAIR_LOCAL,
                    args={
                        "scope": "patcher+verifier",
                        "repair_mode": "object_revoked_boosted_local",
                        "respect_active_target": True,
                        "execution_profile": "boosted",
                    },
                ),
            ],
            rationale="Revoke the object and spend a boosted local repair only when the object-level path is plausible but hard.",
            estimated_total_cost=3100.0,
            estimated_recover_prob=0.76,
            estimated_risk=0.12,
            metadata={
                **common_metadata,
                "family": "semantic_object_revoke_boosted",
                "strategy": "object_revoke_then_boosted_repair",
                "scope": "patcher+verifier",
                "execution_profile": "boosted",
                "postcondition_contract": {
                    "revoked_object_id": object_id,
                    "execution_profile": "boosted",
                    "repair_result": "source_edit|no_diff|test_edit|oracle_passed|oracle_failed",
                },
            },
        ),
    ]

    if normalized_profile == "anchor_restore":
        return [programs[1]]
    if normalized_profile == "local_rebuild":
        return [programs[0], programs[1], programs[3]]
    if normalized_fault_family == "post_patch_regression":
        return [programs[0], programs[1]]
    return programs


def compile_semantic_program(
    program: SemanticRecoveryProgram,
    *,
    failed_state: FailedState,
    ledger: RecoveryLedger,
) -> PrimitiveProgram:
    """Compile upper-layer semantic actions into lower-layer primitives."""

    steps: list[PrimitiveStep] = []
    for semantic_step in program.steps:
        action = semantic_step.action
        args = dict(semantic_step.args)
        if action in {SemanticActionType.EVIDENCE_RECHECK, SemanticActionType.RECHECK_OBJECT}:
            object_type = str(args.get("object_type", ledger.active_object_type) or "")
            object_id = str(args.get("object_id", ledger.active_object_id) or "")
            target = _inspect_target_for_active_object(
                ledger,
                explicit_target=args.get("target", ""),
            )
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.READ_EVIDENCE,
                    args={
                        "target": target,
                        "depth": args.get("depth", "deep"),
                        "object_type": object_type,
                        "object_id": object_id,
                    },
                )
            )
            if bool(args.get("focused_verify", True)):
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.FOCUSED_VERIFY,
                        args={"mode": "failure_probe"},
                    )
                )
        elif action in {SemanticActionType.TARGET_RESET, SemanticActionType.REVOKE_OBJECT}:
            object_type = str(args.get("object_type", ledger.active_object_type) or "")
            object_id = str(args.get("object_id", ledger.active_object_id) or "")
            invalidate = list(args.get("invalidate", []) or [])
            if object_id and object_id not in invalidate:
                invalidate.append(object_id)
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.CLEAR_LOCAL_STATE,
                    args={
                        "invalidate": invalidate,
                        "reset_active_target": bool(args.get("reset_active_target", True)),
                        "object_type": object_type,
                        "object_id": object_id,
                    },
                )
            )
            checkpoint_id = str(args.get("checkpoint_id", "") or "")
            if checkpoint_id:
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.ROLLBACK_ANCHOR,
                        args={
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_label": str(args.get("checkpoint_label", "")),
                            "anchor": str(args.get("anchor", "")),
                        },
                    )
                )
            if bool(args.get("repair_after_clear", False)):
                execution_profile = str(args.get("execution_profile", ledger.execution_profile or "normal") or "normal")
                if execution_profile == "boosted":
                    steps.append(
                        PrimitiveStep(
                            op=PrimitiveOpType.BOOST_EXECUTION,
                            args={
                                "scope": "patcher",
                                "strategy": "stronger_prompt",
                                "level": 1,
                                "execution_profile": execution_profile,
                            },
                        )
                    )
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.CONSTRAINED_REPLAY,
                        args={
                            "scope": str(args.get("replay_scope", args.get("scope", "patcher+verifier"))),
                            "repair_mode": str(args.get("repair_mode", "rebuild_from_local_anchor")),
                            "respect_active_target": bool(args.get("respect_active_target", True)),
                            "execution_profile": execution_profile,
                            "object_type": object_type,
                            "object_id": object_id,
                        },
                    )
                )
        elif action in {SemanticActionType.LOCAL_REPAIR, SemanticActionType.REPAIR_LOCAL}:
            execution_profile = str(args.get("execution_profile", ledger.execution_profile or "normal") or "normal")
            candidate_checkpoint_id = str(args.get("candidate_checkpoint_id", "") or "")
            if candidate_checkpoint_id:
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.ROLLBACK_ANCHOR,
                        args={
                            "checkpoint_id": candidate_checkpoint_id,
                            "checkpoint_label": str(args.get("candidate_checkpoint_label", "bcmr_source_candidate")),
                            "anchor": "source_candidate",
                            "candidate_preserving": True,
                        },
                    )
                )
            if bool(args.get("pre_evidence_recheck", False)):
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.READ_EVIDENCE,
                        args={
                            "target": _inspect_target_for_active_object(
                                ledger,
                                explicit_target=args.get("target", "test_output"),
                            ),
                            "depth": args.get("depth", "deep"),
                            "object_type": str(args.get("object_type", ledger.active_object_type) or ""),
                            "object_id": str(args.get("object_id", ledger.active_object_id) or ""),
                        },
                    )
                )
                if bool(args.get("focused_verify", True)):
                    steps.append(
                        PrimitiveStep(
                            op=PrimitiveOpType.FOCUSED_VERIFY,
                            args={"mode": "failure_probe"},
                        )
                    )
            if execution_profile == "boosted":
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.BOOST_EXECUTION,
                        args={
                            "scope": "patcher",
                            "strategy": "stronger_prompt",
                            "level": 1,
                            "execution_profile": execution_profile,
                        },
                    )
                )
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.CONSTRAINED_REPLAY,
                    args={
                        "scope": str(args.get("scope", "patcher+verifier")),
                        "repair_mode": str(args.get("repair_mode", "local")),
                        "respect_active_target": bool(args.get("respect_active_target", True)),
                        "execution_profile": execution_profile,
                        "object_type": str(args.get("object_type", ledger.active_object_type) or ""),
                        "object_id": str(args.get("object_id", ledger.active_object_id) or ""),
                        "replay_precondition": str(args.get("replay_precondition", "") or ""),
                    },
                )
            )
        elif action in {SemanticActionType.SCOPE_EXPAND, SemanticActionType.EXPAND_SCOPE}:
            strategy = str(args.get("strategy", "broader_search"))
            replay_scope = str(args.get("scope", "locator+patcher+verifier"))
            boost_scope = "locator+patcher+verifier" if "locator" in replay_scope else "patcher"
            execution_profile = str(args.get("execution_profile", "normal") or "normal")
            pre_evidence_recheck = bool(args.get("pre_evidence_recheck", False))
            focused_verify = bool(args.get("focused_verify", False))
            if pre_evidence_recheck:
                steps.append(
                    PrimitiveStep(
                        op=PrimitiveOpType.READ_EVIDENCE,
                        args={
                            "target": _inspect_target_for_active_object(
                                ledger,
                                explicit_target=args.get("target", "test_output"),
                            ),
                            "depth": args.get("depth", "deep"),
                            "object_type": str(args.get("object_type", ledger.active_object_type) or ""),
                            "object_id": str(args.get("object_id", ledger.active_object_id) or ""),
                        },
                    )
                )
                if focused_verify:
                    steps.append(
                        PrimitiveStep(
                            op=PrimitiveOpType.FOCUSED_VERIFY,
                            args={"mode": "failure_probe"},
                        )
                    )
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.BOOST_EXECUTION,
                    args={
                        "scope": boost_scope,
                        "strategy": strategy if strategy in {"broader_search", "stronger_prompt"} else "broader_search",
                        "level": 1 if str(args.get("expansion_level", "regional")) == "regional" else 2,
                    },
                )
            )
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.CONSTRAINED_REPLAY,
                    args={
                        "scope": replay_scope,
                        "repair_mode": "expanded",
                        "respect_active_target": False,
                        "execution_profile": execution_profile,
                        "replay_precondition": str(args.get("replay_precondition", "") or ""),
                    },
                )
            )
        elif action == SemanticActionType.CAPABILITY_BOOST:
            steps.append(
                PrimitiveStep(
                    op=PrimitiveOpType.BOOST_EXECUTION,
                    args={
                        "scope": str(args.get("scope", "patcher")),
                        "strategy": str(args.get("strategy", "stronger_prompt")),
                        "level": int(args.get("level", 1)),
                    },
                )
            )

    return PrimitiveProgram(
        program_id=f"{program.program_id}__compiled",
        steps=steps,
        rationale=program.rationale,
        metadata={
            **dict(program.metadata),
            "semantic_program_id": program.program_id,
            "semantic_skeleton": program.skeleton,
            "compiler_version": "semantic_dual_v1",
            "scope": str(program.metadata.get("scope", "")),
            "ledger_active_target_before_compile": ledger.active_target,
            "failed_instance_id": failed_state.instance_id,
        },
    )
