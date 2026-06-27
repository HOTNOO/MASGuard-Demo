"""Run a recovery-only benchmark from structured polluted late-stage failures.

This benchmark is designed to quickly validate the BCMR core claim:

1. Start from a real SWE task that already has a healthy post-patch anchor.
2. Inject a structured late-stage regression into the oracle workspace.
3. Evaluate a fixed recovery program set with different recovery scopes:
   - minimal local rollback -> post_patch restore
   - broader rollback -> patcher+verifier
   - full restart -> locator+patcher+verifier
4. Score success using the real manifest test/oracle commands, not only the
   agent's self-reported success.

This isolates recovery quality from clean-start task-solving variance.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any

from bcmr_swe import DATA_ROOT, OUTPUT_ROOT
from bcmr_swe.experiments.common import (
    build_coordinator,
    build_executor,
    build_gemini_model,
    materialize_workspace,
    resolve_model_names,
    resolve_runtime,
    run_model_preflight,
    workspace_strategy_for_runtime,
)
from bcmr_swe.recovery.episode_memory import finalize_episodes as finalize_car_episodes
from bcmr_swe.experiments.governance import benchmark_row_governance_fields, summarize_governance_rows
from bcmr_swe.provenance import ProvenanceRecorder
from bcmr_swe.recovery.case_memory import CaseMemory
from bcmr_swe.recovery.action_schema import (
    discovery_programs_v1,
    intent_programs_v2,
    intent_programs_v3,
    rollback_only_programs_v1,
)
from bcmr_swe.recovery.semantic_executor import SemanticProgramExecutor
from bcmr_swe.recovery.semantic_language import (
    bootstrap_recovery_ledger,
    semantic_action_loop_programs_v1,
    semantic_closed_loop_programs_v1,
    semantic_object_loop_programs_v2,
    semantic_programs_v1,
)
from bcmr_swe.recovery.budgeted_controller import suggest_next_action
from bcmr_swe.recovery.car_controller import select_car_action
from bcmr_swe.substrate import load_manifest_catalog, select_manifest_entry
from bcmr_swe.types import (
    FailedState,
    OpType,
    PrimitiveOpType,
    ProgramOutcome,
    RecoveryBudget,
    RecoveryLedger,
    RecoveryProgram,
    RecoveryStep,
    SemanticActionType,
    SemanticRecoveryProgram,
    TriggerType,
)
from swe_mas.utils.path_filters import canonical_source_paths, classify_changed_files, existing_repo_source_paths, normalize_repo_path

DEFAULT_MANUAL_FAULT_SPECS = DATA_ROOT / "artifacts" / "starter_benchmark_manual_fault_specs_v2.json"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _patch_paths(manifest: dict[str, Any]) -> list[str]:
    paths = [str(item) for item in manifest.get("oracle_patch_files", [])]
    non_tests = [path for path in paths if "test" not in path.lower()]
    return non_tests or paths[:1]


def _suspicious_manifest_test_targets(manifest: dict[str, Any]) -> list[str]:
    """Return obviously malformed pytest node ids embedded in the manifest."""
    raw_targets = list(manifest.get("dataset_fail_to_pass") or []) + list(manifest.get("dataset_pass_to_pass") or [])
    suspicious: list[str] = []
    for target in raw_targets:
        text = str(target).strip()
        if not text:
            continue
        if text.count("[") != text.count("]"):
            suspicious.append(text)
    return suspicious


def _classify_changed_files(changed_files: list[str]) -> dict[str, list[str]]:
    return classify_changed_files(changed_files)


def _patch_legality_diagnostic(coord, failed_state: FailedState) -> dict[str, Any]:
    """Record whether a recovery succeeded by touching source or only tests."""
    changed_result = coord.executor.execute(
        "git diff --name-only && git diff --cached --name-only",
        cwd=str(coord.workspace),
        timeout=30,
    )
    changed_files = sorted(
        {
            normalize_repo_path(line.strip())
            for line in str(changed_result.get("output", "")).splitlines()
            if line.strip()
        }
    )
    file_classes = _classify_changed_files(changed_files)
    test_files = list(file_classes["test_files"])
    generated_files = list(file_classes["generated_files"])
    source_files = list(file_classes["source_files"])
    other_files = list(file_classes["other_files"])
    non_test_files = [path for path in changed_files if path not in test_files]
    raw_suspect_paths = [
        str(path).replace("\\", "/")
        for path in list(failed_state.metadata.get("touched_paths", []) or [])
        if str(path).strip()
    ]
    suspect_paths = set(
        existing_repo_source_paths(
            raw_suspect_paths,
            str(getattr(coord, "workspace", "") or ""),
        )
        or canonical_source_paths(raw_suspect_paths)
    )
    if not changed_files:
        patch_scope = "no_diff"
    elif changed_files and not non_test_files:
        patch_scope = "tests_only"
    elif non_test_files and not test_files:
        patch_scope = "non_test_only"
    else:
        patch_scope = "mixed"
    overlap = sorted(path for path in changed_files if path in suspect_paths)
    return {
        "patch_scope": patch_scope,
        "changed_files": changed_files,
        "test_files": test_files,
        "non_test_files": non_test_files,
        "generated_files": generated_files,
        "source_files": source_files,
        "other_files": other_files,
        "changed_file_classes": file_classes,
        "changed_file_class_counts": {
            key: len(value)
            for key, value in file_classes.items()
        },
        "suspect_paths": sorted(suspect_paths),
        "suspect_path_overlap": overlap,
        "touches_suspect_path": bool(overlap),
        "diff_probe_returncode": changed_result.get("returncode"),
    }


def _compute_patch_text(source_snapshot: Path, oracle_snapshot: Path, paths: list[str]) -> str:
    chunks: list[str] = []
    for rel_path in paths:
        source_path = source_snapshot / rel_path
        oracle_path = oracle_snapshot / rel_path
        source_text = _read_text(source_path) if source_path.exists() else ""
        oracle_text = _read_text(oracle_path) if oracle_path.exists() else ""
        diff = difflib.unified_diff(
            source_text.splitlines(keepends=True),
            oracle_text.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
        chunks.append("".join(diff))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _copy_paths_from_source(source_snapshot: Path, workspace: Path, paths: list[str]) -> list[str]:
    changed: list[str] = []
    for rel_path in paths:
        src = source_snapshot / rel_path
        dst = workspace / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(_read_text(src), encoding="utf-8")
            changed.append(rel_path)
        elif dst.exists():
            dst.unlink()
            changed.append(rel_path)
    return changed


def _copy_paths_from_snapshot(snapshot: Path, workspace: Path, paths: list[str]) -> list[str]:
    changed: list[str] = []
    for rel_path in paths:
        src = snapshot / rel_path
        dst = workspace / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(_read_text(src), encoding="utf-8")
            changed.append(rel_path)
        elif dst.exists():
            dst.unlink()
            changed.append(rel_path)
    return changed


def _paths_match_snapshot(workspace: Path, snapshot: Path, paths: list[str]) -> bool:
    for rel_path in paths:
        left = workspace / rel_path
        right = snapshot / rel_path
        if left.exists() != right.exists():
            return False
        if left.exists() and _read_text(left) != _read_text(right):
            return False
    return True


def _apply_first_changed_region_only(source_snapshot: Path, oracle_snapshot: Path, workspace: Path, paths: list[str]) -> list[str]:
    changed = _copy_paths_from_source(source_snapshot, workspace, paths)
    for rel_path in paths:
        source_path = source_snapshot / rel_path
        oracle_path = oracle_snapshot / rel_path
        workspace_path = workspace / rel_path
        if not source_path.exists() or not oracle_path.exists():
            continue
        source_lines = _read_text(source_path).splitlines(keepends=True)
        oracle_lines = _read_text(oracle_path).splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(a=source_lines, b=oracle_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            partial_lines = source_lines[:i1] + oracle_lines[j1:j2] + source_lines[i2:]
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.write_text("".join(partial_lines), encoding="utf-8")
            if rel_path not in changed:
                changed.append(rel_path)
            return changed
    return changed


def _apply_selected_changed_regions(
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    paths: list[str],
    *,
    selected_region_indices: list[int],
    selected_regions_by_path: dict[str, list[int]] | None = None,
) -> list[str]:
    changed = _copy_paths_from_source(source_snapshot, workspace, paths)
    by_path = selected_regions_by_path or {}
    default_indices = {int(idx) for idx in selected_region_indices if int(idx) > 0}
    applied_any = False
    for rel_path in paths:
        path_indices = by_path.get(rel_path)
        selected = {int(idx) for idx in (path_indices if path_indices is not None else default_indices) if int(idx) > 0}
        if not selected:
            continue
        source_path = source_snapshot / rel_path
        oracle_path = oracle_snapshot / rel_path
        workspace_path = workspace / rel_path
        if not source_path.exists() or not oracle_path.exists():
            continue
        source_lines = _read_text(source_path).splitlines(keepends=True)
        oracle_lines = _read_text(oracle_path).splitlines(keepends=True)
        changed_regions = [
            (idx, tag, i1, i2, j1, j2)
            for idx, (tag, i1, i2, j1, j2) in enumerate(
                (op for op in difflib.SequenceMatcher(a=source_lines, b=oracle_lines).get_opcodes() if op[0] != "equal"),
                start=1,
            )
        ]
        selected_regions = [region for region in changed_regions if region[0] in selected]
        if not selected_regions:
            continue
        partial_lines = list(source_lines)
        for _idx, _tag, i1, i2, j1, j2 in sorted(selected_regions, key=lambda region: region[2], reverse=True):
            partial_lines = partial_lines[:i1] + oracle_lines[j1:j2] + partial_lines[i2:]
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text("".join(partial_lines), encoding="utf-8")
        if rel_path not in changed:
            changed.append(rel_path)
        applied_any = True
    if not applied_any:
        raise RuntimeError("Invalid perturbation: selected changed regions did not match any oracle edit.")
    return changed


def _apply_changed_region_oracle_prefix(
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    paths: list[str],
    *,
    region_index: int,
    oracle_line_count: int,
) -> list[str]:
    changed = _copy_paths_from_source(source_snapshot, workspace, paths)
    if region_index <= 0 or oracle_line_count <= 0:
        raise RuntimeError("Invalid perturbation: region_index and oracle_line_count must be positive.")
    for rel_path in paths:
        source_path = source_snapshot / rel_path
        oracle_path = oracle_snapshot / rel_path
        workspace_path = workspace / rel_path
        if not source_path.exists() or not oracle_path.exists():
            continue
        source_lines = _read_text(source_path).splitlines(keepends=True)
        oracle_lines = _read_text(oracle_path).splitlines(keepends=True)
        changed_regions = [
            (idx, tag, i1, i2, j1, j2)
            for idx, (tag, i1, i2, j1, j2) in enumerate(
                (op for op in difflib.SequenceMatcher(a=source_lines, b=oracle_lines).get_opcodes() if op[0] != "equal"),
                start=1,
            )
        ]
        for idx, _tag, i1, i2, j1, j2 in changed_regions:
            if idx != region_index:
                continue
            prefix_end = min(j1 + oracle_line_count, j2)
            if prefix_end <= j1:
                break
            partial_lines = source_lines[:i1] + oracle_lines[j1:prefix_end] + source_lines[i2:]
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.write_text("".join(partial_lines), encoding="utf-8")
            if rel_path not in changed:
                changed.append(rel_path)
            return changed
    raise RuntimeError("Invalid perturbation: requested changed region prefix did not match any oracle edit.")


def _insert_changed_region_oracle_prefix_before_source(
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    paths: list[str],
    *,
    region_index: int,
    oracle_line_count: int,
) -> list[str]:
    changed = _copy_paths_from_source(source_snapshot, workspace, paths)
    if region_index <= 0 or oracle_line_count <= 0:
        raise RuntimeError("Invalid perturbation: region_index and oracle_line_count must be positive.")
    for rel_path in paths:
        source_path = source_snapshot / rel_path
        oracle_path = oracle_snapshot / rel_path
        workspace_path = workspace / rel_path
        if not source_path.exists() or not oracle_path.exists():
            continue
        source_lines = _read_text(source_path).splitlines(keepends=True)
        oracle_lines = _read_text(oracle_path).splitlines(keepends=True)
        changed_regions = [
            (idx, tag, i1, i2, j1, j2)
            for idx, (tag, i1, i2, j1, j2) in enumerate(
                (op for op in difflib.SequenceMatcher(a=source_lines, b=oracle_lines).get_opcodes() if op[0] != "equal"),
                start=1,
            )
        ]
        for idx, _tag, i1, _i2, j1, j2 in changed_regions:
            if idx != region_index:
                continue
            prefix_end = min(j1 + oracle_line_count, j2)
            if prefix_end <= j1:
                break
            partial_lines = source_lines[:i1] + oracle_lines[j1:prefix_end] + source_lines[i1:]
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.write_text("".join(partial_lines), encoding="utf-8")
            if rel_path not in changed:
                changed.append(rel_path)
            return changed
    raise RuntimeError("Invalid perturbation: requested changed region prefix did not match any oracle edit.")


def _apply_primary_patch_file_only(source_snapshot: Path, oracle_snapshot: Path, workspace: Path, paths: list[str]) -> list[str]:
    changed = _copy_paths_from_source(source_snapshot, workspace, paths)
    primary = paths[0] if paths else ""
    if primary:
        changed.extend(path for path in _copy_paths_from_snapshot(oracle_snapshot, workspace, [primary]) if path not in changed)
    return changed


def _apply_partial_oracle_patch(
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    paths: list[str],
    *,
    mode: str,
    selected_region_indices: list[int] | None = None,
    selected_regions_by_path: dict[str, list[int]] | None = None,
    region_index: int | None = None,
    oracle_line_count: int | None = None,
) -> list[str]:
    normalized = str(mode or "apply_first_changed_region_only").strip().lower()
    if normalized == "apply_primary_patch_file_only":
        return _apply_primary_patch_file_only(source_snapshot, oracle_snapshot, workspace, paths)
    if normalized == "apply_selected_changed_regions":
        return _apply_selected_changed_regions(
            source_snapshot,
            oracle_snapshot,
            workspace,
            paths,
            selected_region_indices=selected_region_indices or [],
            selected_regions_by_path=selected_regions_by_path,
        )
    if normalized == "apply_changed_region_oracle_prefix":
        return _apply_changed_region_oracle_prefix(
            source_snapshot,
            oracle_snapshot,
            workspace,
            paths,
            region_index=int(region_index or 0),
            oracle_line_count=int(oracle_line_count or 0),
        )
    if normalized == "insert_changed_region_oracle_prefix_before_source":
        return _insert_changed_region_oracle_prefix_before_source(
            source_snapshot,
            oracle_snapshot,
            workspace,
            paths,
            region_index=int(region_index or 0),
            oracle_line_count=int(oracle_line_count or 0),
        )
    if normalized == "apply_first_changed_region_only":
        return _apply_first_changed_region_only(source_snapshot, oracle_snapshot, workspace, paths)
    raise RuntimeError(f"Invalid perturbation: unsupported partial patch mode {mode!r}")


def _manual_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0:
            result.append(number)
    return result


def _manual_selected_regions_by_path(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[int]] = {}
    for path, indices in value.items():
        normalized_path = str(path).strip()
        if not normalized_path:
            continue
        numbers = _manual_int_list(indices)
        if numbers:
            result[normalized_path] = numbers
    return result


def _load_manual_fault_specs(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    families = payload.get("families", {})
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(families, dict):
        return specs
    for family, cases in families.items():
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            instance_id = str(case.get("instance_id", "")).strip()
            fault_family = str(case.get("fault_family") or family).strip()
            if instance_id and fault_family:
                specs[(fault_family, instance_id)] = case
    return specs


def _manual_injection_spec(
    specs: dict[tuple[str, str], dict[str, Any]],
    *,
    fault_family: str,
    instance_id: str,
) -> dict[str, Any]:
    case = specs.get((fault_family, instance_id), {})
    spec = case.get("manual_injection_spec", {}) if isinstance(case, dict) else {}
    return dict(spec) if isinstance(spec, dict) else {}


def _usage_delta(before: dict[str, float], after: dict[str, float]) -> float:
    return max(0.0, float(after.get("total_tokens", 0.0)) - float(before.get("total_tokens", 0.0)))


def _program_scope(program) -> str:
    metadata = dict(getattr(program, "metadata", {}) or {})
    metadata_scope = str(metadata.get("scope", "") or "").strip().lower()
    if metadata_scope:
        return metadata_scope
    for step in reversed(program.steps):
        step_op = getattr(step, "op", None)
        op_value = getattr(step_op, "value", "")
        if step_op == OpType.REPLAY or op_value == PrimitiveOpType.CONSTRAINED_REPLAY.value:
            return str(step.args.get("scope", "")).strip().lower()
    return ""


def _program_strategy(program) -> str:
    return str(program.metadata.get("strategy", program.program_id))


def _program_family(program) -> str:
    return str(program.metadata.get("family", "unknown"))


def _program_to_dict(program) -> dict[str, Any]:
    if hasattr(program, "to_dict"):
        return dict(program.to_dict())
    return {
        "program_id": str(getattr(program, "program_id", "")),
        "metadata": dict(getattr(program, "metadata", {}) or {}),
    }


def _is_semantic_program(program) -> bool:
    return isinstance(program, SemanticRecoveryProgram)


def _program_from_payload(payload: dict[str, Any]):
    steps = list(payload.get("steps", []) or [])
    if steps and isinstance(steps[0], dict) and "action" in steps[0]:
        return SemanticRecoveryProgram.from_dict(payload)
    return RecoveryProgram.from_dict(payload)


def _is_global_family(family: str) -> bool:
    normalized = str(family or "").strip().lower()
    return normalized in {"global", "semantic_global_reset"}


def _program_has_redundant_belief_cleanup(program: RecoveryProgram) -> bool:
    """Detect revoke-before-rollback programs whose cleanup is weakly consumed.

    In the current executor, `REVOKE` has two effects:
    1. remove the conflicted shared fact from `coord.shared_facts`
    2. append a short textual hint into the next replay context

    If the same program then restores an earlier checkpoint and replays
    `patcher+verifier` / `locator+...` / `full`, the rollback already discards
    the polluted patch workspace and stage outputs.  In that setting the revoke
    step is often redundant or even distracting, because it is no longer
    consumed as structured state by later steps.
    """

    if _is_semantic_program(program):
        return False

    saw_revoke = False
    saw_rollback = False
    replay_scope = _program_scope(program)
    for step in program.steps:
        if step.op == OpType.REVOKE:
            saw_revoke = True
        elif step.op == OpType.ROLLBACK and saw_revoke:
            saw_rollback = True
    if not (saw_revoke and saw_rollback):
        return False
    return replay_scope in {"patcher+verifier", "locator+patcher+verifier", "full"}


def _checkpoint_id_for_label(
    failed_state: FailedState,
    *,
    label: str,
) -> str:
    candidates = failed_state.metadata.get("checkpoint_candidates", [])
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("label", "")).strip() != label:
            continue
        return str(candidate.get("checkpoint_id", "")).strip()
    return ""


def _checkpoint_label_for_id(
    failed_state: FailedState,
    checkpoint_id: str,
) -> str:
    target = str(checkpoint_id or "").strip()
    if not target:
        return ""
    candidates = failed_state.metadata.get("checkpoint_candidates", [])
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("checkpoint_id", "")).strip() == target:
            return str(candidate.get("label", "")).strip()
    return ""


def _checkpoint_anchor_health(
    failed_state: FailedState,
    *,
    label: str,
) -> str:
    candidates = failed_state.metadata.get("checkpoint_candidates", [])
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("label", "")).strip() != label:
            continue
        metadata = candidate.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return str(metadata.get("anchor_health", "")).strip().lower()
    return ""


def _program_online_priority(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> float:
    utility = (
        float(program.estimated_recover_prob)
        - float(budget.lambda_token) * float(program.estimated_total_cost)
        - float(budget.lambda_risk) * float(program.estimated_risk)
    )
    family = _program_family(program)
    scope = _program_scope(program)
    post_patch_health = _checkpoint_anchor_health(failed_state, label="post_patch")

    if post_patch_health in {"healthy_oracle_patch", "healthy_patch", "verified_healthy"}:
        if family == "local_minimal":
            utility += 100.0
        if family == "semantic_local_restore":
            utility += 100.0
        elif family == "global":
            utility -= 10.0
        elif scope in {"patcher+verifier", "locator+patcher+verifier", "full"}:
            utility -= 2.0

    if post_patch_health in {"contaminated", "wrong_localization", "partial_patch"}:
        if family == "local_minimal":
            utility -= 100.0
        if family == "semantic_local_restore":
            utility -= 100.0
        if family in {
            "local_replan",
            "local_broader",
            "belief_cleanup",
            "evidence_recheck",
            "capability_boost",
            "relocalize",
            "semantic_target_reset",
            "semantic_evidence_recheck",
            "semantic_object_recheck",
            "semantic_object_revoke",
            "semantic_object_revoke_boosted",
            "semantic_scope_expand",
            "semantic_capability_boost",
        }:
            utility += 5.0
        if family == "global":
            utility += 1.0
        if family == "semantic_global_reset":
            utility += 1.0

    if _program_has_redundant_belief_cleanup(program):
        utility -= 1.0

    if scope == "verifier" and family != "local_minimal":
        utility -= 20.0

    return utility


def _semantic_selector_state_signals(failed_state: FailedState) -> dict[str, Any]:
    """Extract selector signals from structured MAS artifacts only.

    These signals intentionally avoid oracle patch files and manual failure
    labels. The selector may use the historical role outputs and typed
    propagation objects, because those are the polluted MAS state BCMR is
    supposed to recover from.
    """
    trigger_type = failed_state.trigger.trigger_type
    summary = dict(failed_state.suspect_region.summary or {})
    metadata = dict(failed_state.metadata or {})
    phase_outputs = dict(metadata.get("phase_outputs", {}) or {})
    patcher = dict(phase_outputs.get("patcher", {}) or {})
    verifier = dict(phase_outputs.get("verifier", {}) or {})
    patch_summary = dict(patcher.get("patch_summary", {}) or {})
    changed_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
    source_files = [str(item) for item in list(changed_classes.get("source_files", []) or []) if str(item)]
    test_files = [str(item) for item in list(changed_classes.get("test_files", []) or []) if str(item)]
    effective_files = [str(item) for item in list(changed_classes.get("effective_files", []) or []) if str(item)]
    patch_text = str(patcher.get("patch", "") or "").strip()
    patch_failure_mode = str(patch_summary.get("failure_mode", "") or "").strip().lower()
    patcher_failed = not bool(patcher.get("success", False))
    no_effective_patch = (
        patch_failure_mode == "no_effective_patch"
        or (patcher_failed and not patch_text)
        or (bool(patch_summary) and not effective_files)
    )
    effective_source_patch = bool(source_files)
    mixed_source_test_patch = bool(source_files and test_files)
    source_only_patch = bool(source_files and not test_files)
    infrastructure_error = bool(patcher.get("infrastructure_error", False))
    if not infrastructure_error:
        infra_text = "\n".join(
            str(patcher.get(key, "") or "")
            for key in ("planner_error", "implementer_error", "patcher_error")
        ).lower()
        infrastructure_error = any(
            marker in infra_text
            for marker in ("upstream_error", "heartbeat stream", "operation timed out", "api request failed")
        )

    object_types = [
        str(dict(item).get("object_type", "") or "").strip().lower()
        for item in list(metadata.get("mas_object_chain", []) or [])
        if isinstance(item, dict)
    ]
    selection_count = sum(1 for item in object_types if item == "selection")
    shared_fact_count = sum(1 for item in object_types if item == "sharedfact")
    verifier_object_count = sum(1 for item in object_types if item == "verifierverdict")
    wrong_localized_path = bool(metadata.get("wrong_localized_path"))
    replay_anchor_role = str(summary.get("replay_anchor_role", "") or "").strip().lower()
    active_object_type = ""
    if trigger_type == TriggerType.FACT_CONFLICT and shared_fact_count:
        active_object_type = "shared_fact"
    elif trigger_type == TriggerType.VERIFIER_CONTRADICTION and verifier_object_count:
        active_object_type = "verifier_verdict"
    elif (wrong_localized_path or replay_anchor_role == "locator") and selection_count:
        active_object_type = "selection"
    latest = metadata.get("latest_test_status", "")
    if isinstance(latest, dict):
        latest_test_status = str(latest.get("status", "") or latest.get("verification", "") or "").strip().lower()
    else:
        latest_test_status = str(latest or "").strip().lower()
    verifier_excerpt = str(verifier.get("verification", "") or "").strip()
    role_chain = [str(item).strip().lower() for item in list(summary.get("role_chain", []) or [])]
    touched_paths = [str(item).strip() for item in list(metadata.get("touched_paths", []) or []) if str(item).strip()]
    selected_targets = [
        str(item).strip()
        for item in list(patcher.get("selected_target_candidates", []) or [])
        if str(item).strip()
    ]
    return {
        "replay_anchor_role": replay_anchor_role,
        "role_chain": role_chain,
        "has_conflicting_fact": bool(summary.get("has_conflicting_fact", False)),
        "wrong_localized_path": wrong_localized_path,
        "touched_path_count": len(touched_paths),
        "selected_target_count": len(selected_targets),
        "latest_test_status": latest_test_status,
        "verifier_has_output": bool(verifier_excerpt),
        "recovery_invocations": int(metadata.get("recovery_invocations", 0) or 0),
        "anchor_health": _checkpoint_anchor_health(failed_state, label="post_patch"),
        "patcher_failed": patcher_failed,
        "patch_plan_present": bool(str(patcher.get("plan", "") or "").strip()),
        "no_effective_patch": no_effective_patch,
        "effective_source_patch": effective_source_patch,
        "source_only_patch": source_only_patch,
        "mixed_source_test_patch": mixed_source_test_patch,
        "source_file_count": len(source_files),
        "test_file_count": len(test_files),
        "effective_file_count": len(effective_files),
        "infrastructure_error": infrastructure_error,
        "selection_object_count": selection_count,
        "shared_fact_object_count": shared_fact_count,
        "verifier_object_count": verifier_object_count,
        "active_object_type": active_object_type,
        "selection_only_objects": bool(selection_count and not shared_fact_count and not verifier_object_count),
        "has_object_chain": bool(object_types),
    }


def _semantic_action_starter_priority_with_debug(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> float:
    """State-aware starter priority for action-level semantic closed loop.

    This selector is intentionally narrow: it only decides the *first* semantic
    action when we are validating action-level execution loops. The goal is to
    avoid collapsing every case into the same generic target reset starter.
    """
    utility = _program_online_priority(program, failed_state=failed_state, budget=budget)
    strategy = str(getattr(program, "metadata", {}).get("strategy", "") or "")
    trigger = failed_state.trigger.trigger_type
    signals = _semantic_selector_state_signals(failed_state)
    replay_anchor_role = str(signals["replay_anchor_role"])
    role_chain = list(signals["role_chain"])
    has_conflicting_fact = bool(signals["has_conflicting_fact"])
    wrong_localized_path = bool(signals["wrong_localized_path"])
    latest_test_status = str(signals["latest_test_status"])
    recovery_invocations = int(signals["recovery_invocations"])
    anchor_health = str(signals["anchor_health"])
    active_object_type = str(signals["active_object_type"])

    bonus = 0.0
    if strategy == "semantic_action_loop_object_recheck_start":
        if signals["effective_source_patch"]:
            bonus += 8.0
        if signals["source_only_patch"]:
            bonus += 2.0
        if signals["mixed_source_test_patch"]:
            bonus -= 2.0
        if signals["verifier_has_output"] or trigger == TriggerType.VERIFIER_CONTRADICTION:
            bonus += 1.0
        if has_conflicting_fact and signals["effective_source_patch"]:
            bonus += 4.0
        if active_object_type == "shared_fact":
            bonus -= 1.0
        elif active_object_type == "verifier_verdict":
            bonus += 2.0
        elif active_object_type == "selection":
            bonus += 1.0
        if signals["no_effective_patch"]:
            bonus -= 3.0
    elif strategy == "semantic_action_loop_object_revoke_start":
        if signals["no_effective_patch"] and (
            signals["shared_fact_object_count"] or signals["selection_object_count"]
        ):
            bonus += 7.0
        if signals["shared_fact_object_count"] and not signals["effective_source_patch"]:
            bonus += 3.0
        if has_conflicting_fact and signals["no_effective_patch"]:
            bonus += 2.0
        if signals["effective_source_patch"] and has_conflicting_fact:
            bonus += 2.0
        if signals["mixed_source_test_patch"]:
            bonus += 1.0
        if active_object_type == "shared_fact":
            bonus += 5.0
        elif active_object_type == "selection":
            bonus += 1.5
        elif active_object_type == "verifier_verdict":
            bonus -= 1.0
    elif strategy == "semantic_action_loop_evidence_start":
        if not signals["has_object_chain"]:
            bonus += 4.0
        if signals["verifier_has_output"] and not signals["effective_source_patch"]:
            bonus += 2.0
        if signals["effective_source_patch"]:
            bonus += 1.0
        if signals["no_effective_patch"] and signals["has_object_chain"]:
            bonus -= 2.0
        if has_conflicting_fact:
            bonus += 1.0
        if replay_anchor_role == "verifier":
            bonus += 0.5
        if latest_test_status == "fail":
            bonus += 1.0
    elif strategy == "semantic_action_loop_target_reset_start":
        if signals["mixed_source_test_patch"]:
            bonus += 8.0
        if signals["effective_source_patch"] and has_conflicting_fact:
            bonus += 2.0
        if anchor_health in {"contaminated", "partial_patch", "wrong_localization"}:
            bonus += 3.0
        if signals["touched_path_count"]:
            bonus += 1.0
        if replay_anchor_role in {"patcher", "verifier"}:
            bonus += 1.5
        if has_conflicting_fact:
            bonus += 0.5
    elif strategy == "semantic_action_loop_scope_expand_start":
        if signals["selection_only_objects"] and signals["no_effective_patch"] and not signals["infrastructure_error"]:
            bonus += 11.0
        if (
            signals["patcher_failed"]
            and signals["selection_object_count"] >= 2
            and not signals["infrastructure_error"]
        ):
            bonus += 5.0
        if wrong_localized_path:
            bonus += 7.0
        if replay_anchor_role == "locator":
            bonus += 3.0
        if "locator" in role_chain:
            bonus += 1.5
        if not signals["touched_path_count"]:
            bonus += 2.0
    elif strategy == "semantic_action_loop_capability_start":
        if signals["infrastructure_error"]:
            bonus += 12.0
        if recovery_invocations > 0:
            bonus += min(4.0, float(recovery_invocations))
        if latest_test_status == "fail" and signals["touched_path_count"]:
            bonus += 2.0
        if signals["patcher_failed"] and signals["patch_plan_present"] and signals["no_effective_patch"]:
            bonus += 2.0
        if trigger == TriggerType.NO_PROGRESS_LOOP:
            bonus += 1.5
    elif strategy == "semantic_action_loop_global_reset_start":
        if anchor_health in {"partial_patch"}:
            bonus += 1.5
        else:
            bonus -= 2.0

    score = utility + bonus
    debug = {
        "selector": "semantic_action_state_top1",
        "strategy": strategy,
        "base_utility": utility,
        "bonus": bonus,
        "final_score": score,
        "signals": {
            "trigger_type": trigger.value if hasattr(trigger, "value") else str(trigger),
            "replay_anchor_role": replay_anchor_role,
            "role_chain": role_chain,
            "has_conflicting_fact": has_conflicting_fact,
            "active_object_type": active_object_type,
            **signals,
        },
    }
    return score, debug


def _semantic_action_starter_priority(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> float:
    score, _ = _semantic_action_starter_priority_with_debug(
        program,
        failed_state=failed_state,
        budget=budget,
    )
    return score


def _parc_lifecycle_starter_priority_with_debug(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> tuple[float, dict[str, Any]]:
    """Select the first action from PARC's lifecycle frontier.

    This selector only reads the structured failed state and the deterministic
    PARC controller. It keeps the same seven-action semantic space as the
    existing action-loop cell, but changes the first action choice from
    hand-weighted starter heuristics to object-lifecycle convergence.
    """

    base_score, base_debug = _semantic_action_starter_priority_with_debug(
        program,
        failed_state=failed_state,
        budget=budget,
    )
    ledger = bootstrap_recovery_ledger(failed_state, budget)
    decision = suggest_next_action(ledger)
    controller_action = str(getattr(decision, "action", "") or "")
    program_action = _first_semantic_action_value(program)
    score = float(base_score) * 0.01
    action_match = bool(controller_action and controller_action == program_action)
    if action_match:
        score += 1000.0
    if controller_action:
        score -= abs(float(getattr(program, "estimated_total_cost", 0.0) or 0.0)) / 100000.0
    else:
        score = base_score
    debug = {
        "selector": "parc_lifecycle_frontier_top1",
        "base_selector": base_debug,
        "program_action": program_action,
        "controller_decision": decision.to_dict() if decision is not None else {},
        "action_match": action_match,
        "base_score": base_score,
        "final_score": score,
    }
    return score, debug


def _car_counterexample_starter_priority_with_debug(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> tuple[float, dict[str, Any]]:
    base_score, base_debug = _semantic_action_starter_priority_with_debug(
        program,
        failed_state=failed_state,
        budget=budget,
    )
    ledger = bootstrap_recovery_ledger(failed_state, budget)
    decision = select_car_action(ledger)
    controller_action = str(getattr(decision, "action", "") or "")
    program_action = _first_semantic_action_value(program)
    score = float(base_score) * 0.01
    action_match = bool(controller_action and controller_action == program_action)
    if action_match:
        score += 1200.0
    if controller_action:
        score -= abs(float(getattr(program, "estimated_total_cost", 0.0) or 0.0)) / 100000.0
    else:
        score = base_score
    debug = {
        "selector": "car_counterexample_top1",
        "base_selector": base_debug,
        "program_action": program_action,
        "controller_decision": decision.to_dict() if decision is not None else {},
        "car_counterexample": dict(ledger.metadata.get("latest_car_counterexample", {}) or {}),
        "car_controller_decision": dict(ledger.metadata.get("latest_car_controller_decision", {}) or {}),
        "action_match": action_match,
        "base_score": base_score,
        "final_score": score,
    }
    return score, debug


def _first_semantic_action_value(program: Any) -> str:
    steps = list(getattr(program, "steps", []) or [])
    if not steps:
        return ""
    action = getattr(steps[0], "action", "")
    return str(getattr(action, "value", action) or "")


def _semantic_object_macro_priority_with_debug(
    program,
    *,
    failed_state: FailedState,
    budget,
) -> tuple[float, dict[str, Any]]:
    """Object-aware priority for semantic_object_loop_v2 macro actions."""
    utility = _program_online_priority(program, failed_state=failed_state, budget=budget)
    strategy = str(getattr(program, "metadata", {}).get("strategy", "") or "")
    family = str(getattr(program, "metadata", {}).get("family", "") or "")
    object_type = str(getattr(program, "metadata", {}).get("object_type", "") or "")
    trigger = failed_state.trigger.trigger_type
    trigger_value = trigger.value if hasattr(trigger, "value") else str(trigger)
    signals = _semantic_selector_state_signals(failed_state)
    replay_anchor_role = str(signals["replay_anchor_role"])
    has_conflicting_fact = bool(signals["has_conflicting_fact"])
    wrong_localized_path = bool(signals["wrong_localized_path"])
    anchor_health = str(signals["anchor_health"])

    bonus = 0.0
    if family == "semantic_object_recheck":
        if signals["effective_source_patch"]:
            bonus += 8.0
        if signals["source_only_patch"]:
            bonus += 2.0
        if signals["mixed_source_test_patch"]:
            bonus -= 2.0
        if signals["verifier_has_output"]:
            bonus += 1.0
    elif family == "semantic_object_revoke":
        if signals["no_effective_patch"] and (
            signals["shared_fact_object_count"] or signals["selection_object_count"]
        ):
            bonus += 7.0
        if signals["shared_fact_object_count"] and not signals["effective_source_patch"]:
            bonus += 3.0
        if has_conflicting_fact:
            bonus += 2.0
        if signals["mixed_source_test_patch"]:
            bonus += 1.0
    elif family == "semantic_object_expand":
        if signals["selection_only_objects"] and signals["no_effective_patch"] and not signals["infrastructure_error"]:
            bonus += 11.0
        if signals["patcher_failed"] and not signals["infrastructure_error"]:
            bonus += 3.0
        if wrong_localized_path or replay_anchor_role == "locator":
            bonus += 3.0
    elif family == "semantic_object_revoke_boosted":
        if signals["infrastructure_error"]:
            bonus += 12.0
        if signals["patcher_failed"] and signals["patch_plan_present"]:
            bonus += 2.0

    if object_type == "shared_fact":
        if has_conflicting_fact or trigger == TriggerType.FACT_CONFLICT:
            bonus += 1.0
        if anchor_health in {"contaminated", "partial_patch"}:
            bonus += 3.0
    elif object_type == "selection":
        if wrong_localized_path or replay_anchor_role == "locator":
            bonus += 3.0
    elif object_type == "verifier_verdict":
        if trigger == TriggerType.VERIFIER_CONTRADICTION or replay_anchor_role == "verifier":
            bonus += 2.0

    if family == "semantic_object_revoke_boosted":
        bonus -= 1.0
        if signals["touched_path_count"] and anchor_health in {"contaminated", "partial_patch"}:
            bonus += 0.5

    score = utility + bonus
    debug = {
        "selector": "semantic_object_state_top1",
        "strategy": strategy,
        "family": family,
        "object_type": object_type,
        "base_utility": utility,
        "bonus": bonus,
        "final_score": score,
        "signals": {
            "trigger_type": trigger_value,
            "replay_anchor_role": replay_anchor_role,
            "has_conflicting_fact": has_conflicting_fact,
            "wrong_localized_path": wrong_localized_path,
            "anchor_health": anchor_health,
            **signals,
        },
    }
    return score, debug


def _contaminated_diagnosis_programs(
    failed_state: FailedState,
) -> list[RecoveryProgram]:
    post_locate_id = _checkpoint_id_for_label(failed_state, label="post_locate")
    if not post_locate_id:
        raise RuntimeError("contaminated diagnosis mode requires a post_locate checkpoint.")

    base_context = (
        "This is a contaminated recovery case. Rebuild from the post-locate anchor, "
        "prioritize the suspect source path, and avoid generated directories, copied build outputs, and test-only edits."
    )

    return [
        RecoveryProgram(
            program_id="diagnostic_post_locate_patch_replay",
            steps=[
                RecoveryStep(
                    op=OpType.ROLLBACK,
                    args={"checkpoint_id": post_locate_id, "checkpoint_label": "post_locate"},
                ),
                RecoveryStep(
                    op=OpType.REPLAY,
                    args={
                        "scope": "patcher+verifier",
                        "context_hint": base_context,
                    },
                ),
            ],
            rationale="Contaminated diagnosis path: restore the clean post-locate anchor, then locally rebuild patch and verification.",
            estimated_total_cost=1400.0,
            estimated_recover_prob=0.55,
            estimated_risk=0.10,
            metadata={
                "family": "local_broader",
                "strategy": "diagnostic_post_locate_patch_replay",
                "program_space_version": "atomic_actions_v1",
                "diagnosis_mode": "contaminated_post_patch_v1",
            },
        ),
        RecoveryProgram(
            program_id="diagnostic_revoke_then_post_locate_patch_replay",
            steps=[
                RecoveryStep(
                    op=OpType.REVOKE,
                    args={"fact_id": "fact:latest_patch"},
                ),
                RecoveryStep(
                    op=OpType.ROLLBACK,
                    args={"checkpoint_id": post_locate_id, "checkpoint_label": "post_locate"},
                ),
                RecoveryStep(
                    op=OpType.REPLAY,
                    args={
                        "scope": "patcher+verifier",
                        "context_hint": (
                            base_context
                            + " Also treat the previous patch hypothesis as stale and rebuild from scratch on the same localized target."
                        ),
                    },
                ),
            ],
            rationale="Contaminated diagnosis path: revoke the stale patch belief, restore post-locate, then rebuild patch and verify.",
            estimated_total_cost=1550.0,
            estimated_recover_prob=0.58,
            estimated_risk=0.11,
            metadata={
                "family": "belief_cleanup",
                "strategy": "diagnostic_revoke_then_post_locate_patch_replay",
                "program_space_version": "atomic_actions_v1",
                "diagnosis_mode": "contaminated_post_patch_v1",
            },
        ),
    ]


def _actual_utility(outcome: ProgramOutcome, budget) -> float:
    return (
        (1.0 if outcome.recover_success else 0.0)
        - budget.lambda_token * outcome.token_cost
        - budget.lambda_latency * outcome.latency_sec
        - budget.lambda_risk * outcome.secondary_risk
    )


def _best_program_for_pool(
    programs: list[RecoveryProgram],
    outcomes: list[ProgramOutcome],
    budget,
) -> tuple[RecoveryProgram, ProgramOutcome, str]:
    by_id = {outcome.program_id: outcome for outcome in outcomes}
    successful = [
        (program, by_id[program.program_id])
        for program in programs
        if by_id[program.program_id].recover_success
    ]
    if successful:
        best_program, best_outcome = min(
            successful,
            key=lambda item: (
                float(item[1].token_cost),
                float(item[1].latency_sec),
                float(item[1].secondary_risk),
                -float(item[1].milestone_gain),
                item[0].program_id,
            ),
        )
        return best_program, best_outcome, "success_first_then_cost"

    ranked = sorted(
        programs,
        key=lambda program: _actual_utility(by_id[program.program_id], budget),
        reverse=True,
    )
    best_program = ranked[0]
    return best_program, by_id[best_program.program_id], "utility_fallback"


def _bootstrap_run_context(coord, *, manifest: dict[str, Any], workspace: Path, executor) -> dict[str, str]:
    run_id = f"bcmr_recovery_bench_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    coord.issue = str(manifest["problem_statement"])
    coord.workspace = str(workspace)
    coord.instance_id = str(manifest["instance_id"])
    coord.run_id = run_id
    coord.run_dir = run_dir
    coord.executor = executor
    coord.test_command = str(manifest["test_command"])
    coord._test_command = str(manifest["test_command"])
    coord.recorder = ProvenanceRecorder(run_dir=run_dir, workspace=workspace)
    coord.shared_facts = {}
    coord.stage_outputs = {}
    coord.stage_nodes = {}
    coord.recovery_calls = 0
    coord.captured_failed_state_groups = 0
    coord.selected_actions = []
    coord.executed_programs = []

    initial = coord.recorder.create_checkpoint(
        label="initial",
        metadata={"resume_from": "locator", "stage": "initial"},
    )
    return {"initial": initial.checkpoint_id}


def _seed_healthy_anchor(
    coord,
    *,
    manifest: dict[str, Any],
    source_snapshot: Path,
    oracle_snapshot: Path,
    patch_paths: list[str],
    checkpoint_ids: dict[str, str],
) -> None:
    located_summary = "\n".join(patch_paths)
    locator_node = coord.recorder.record_agent_step(
        role="locator",
        phase="locate",
        content=located_summary or "oracle_patch_files",
        payload={"success": True, "synthetic_seed": True},
    )
    coord.stage_nodes["locator"] = locator_node.node_id
    coord.stage_outputs["locator"] = {
        "success": True,
        "located_files": located_summary,
        "synthetic_seed": True,
    }

    if patch_paths:
        localized_fact = coord.recorder.record_shared_fact(
            key="localized_path",
            value=patch_paths[0],
            role="locator",
            phase="locate",
            source_node_id=locator_node.node_id,
            confidence=0.95,
            payload={"synthetic_seed": True},
        )
        coord.shared_facts["localized_path"] = {
            "value": patch_paths[0],
            "node_id": localized_fact.node_id,
        }

    post_locate = coord.recorder.create_checkpoint(
        label="post_locate",
        metadata={"resume_from": "patcher", "stage": "locator", "synthetic_seed": True},
        source_node_id=locator_node.node_id,
    )
    checkpoint_ids["post_locate"] = post_locate.checkpoint_id

    applied_paths = _copy_paths_from_snapshot(oracle_snapshot, Path(str(coord.workspace)), patch_paths)
    patch_text = _compute_patch_text(source_snapshot, oracle_snapshot, patch_paths)
    patch_node = coord.recorder.record_agent_step(
        role="patcher",
        phase="patch",
        content=patch_text[:4000] or "oracle_patch",
        payload={
            "success": True,
            "patch": patch_text[:4000],
            "synthetic_seed": True,
            "applied_paths": list(applied_paths),
        },
        depends_on=[locator_node.node_id],
        reads=[coord.shared_facts["localized_path"]["node_id"]] if "localized_path" in coord.shared_facts else None,
    )
    coord.stage_nodes["patcher"] = patch_node.node_id
    coord.stage_outputs["patcher"] = {
        "success": True,
        "plan": "Synthetic oracle patch seed.",
        "patch": patch_text,
        "commands": [],
        "synthetic_seed": True,
        "applied_paths": list(applied_paths),
    }
    patch_fact = coord.recorder.record_shared_fact(
        key="latest_patch",
        value=patch_text[:400] if patch_text else "\n".join(patch_paths),
        role="patcher",
        phase="patch",
        source_node_id=patch_node.node_id,
        confidence=0.95,
        payload={"synthetic_seed": True},
    )
    coord.shared_facts["latest_patch"] = {
        "value": patch_text,
        "node_id": patch_fact.node_id,
    }

    post_patch = coord.recorder.create_checkpoint(
        label="post_patch",
        metadata={
            "resume_from": "verifier",
            "stage": "patcher",
            "synthetic_seed": True,
            "anchor_health": "healthy_oracle_patch",
            "touched_paths": list(applied_paths),
        },
        source_node_id=patch_node.node_id,
    )
    checkpoint_ids["post_patch"] = post_patch.checkpoint_id
    coord.recorder.save()


def _seed_contaminated_post_patch_and_build_failed_state(
    coord,
    *,
    manifest: dict[str, Any],
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    patch_paths: list[str],
    checkpoint_ids: dict[str, str],
) -> FailedState:
    located_summary = "\n".join(patch_paths)
    locator_node = coord.recorder.record_agent_step(
        role="locator",
        phase="locate",
        content=located_summary or "oracle_patch_files",
        payload={"success": True, "synthetic_seed": True},
    )
    coord.stage_nodes["locator"] = locator_node.node_id
    coord.stage_outputs["locator"] = {
        "success": True,
        "located_files": located_summary,
        "synthetic_seed": True,
    }

    if patch_paths:
        localized_fact = coord.recorder.record_shared_fact(
            key="localized_path",
            value=patch_paths[0],
            role="locator",
            phase="locate",
            source_node_id=locator_node.node_id,
            confidence=0.95,
            payload={"synthetic_seed": True},
        )
        coord.shared_facts["localized_path"] = {
            "value": patch_paths[0],
            "node_id": localized_fact.node_id,
        }

    post_locate = coord.recorder.create_checkpoint(
        label="post_locate",
        metadata={"resume_from": "patcher", "stage": "locator", "synthetic_seed": True},
        source_node_id=locator_node.node_id,
    )
    checkpoint_ids["post_locate"] = post_locate.checkpoint_id

    patch_text = _compute_patch_text(source_snapshot, oracle_snapshot, patch_paths)
    changed = _copy_paths_from_source(source_snapshot, workspace, patch_paths)
    patch_node = coord.recorder.record_agent_step(
        role="patcher",
        phase="patch",
        content=patch_text[:4000] or "claimed_oracle_patch_but_anchor_is_contaminated",
        payload={
            "success": True,
            "patch": patch_text[:4000],
            "synthetic_seed": True,
            "fault_type": "contaminated_post_patch",
            "touched_paths": changed,
        },
        depends_on=[locator_node.node_id],
        reads=[coord.shared_facts["localized_path"]["node_id"]] if "localized_path" in coord.shared_facts else None,
    )
    coord.stage_nodes["patcher"] = patch_node.node_id
    coord.stage_outputs["patcher"] = {
        "success": True,
        "plan": "Synthetic contaminated post_patch anchor.",
        "patch": patch_text,
        "commands": [],
        "synthetic_seed": True,
    }
    patch_fact = coord.recorder.record_shared_fact(
        key="latest_patch",
        value=patch_text[:400] if patch_text else "\n".join(patch_paths),
        role="patcher",
        phase="patch",
        source_node_id=patch_node.node_id,
        confidence=0.95,
        payload={
            "synthetic_seed": True,
            "fault_type": "contaminated_post_patch",
            "anchor_health": "contaminated",
        },
    )
    coord.shared_facts["latest_patch"] = {
        "value": patch_text,
        "node_id": patch_fact.node_id,
    }

    post_patch = coord.recorder.create_checkpoint(
        label="post_patch",
        metadata={
            "resume_from": "verifier",
            "stage": "patcher",
            "synthetic_seed": True,
            "fault_type": "contaminated_post_patch",
            "anchor_health": "contaminated",
            "touched_paths": changed,
        },
        source_node_id=patch_node.node_id,
    )
    checkpoint_ids["post_patch"] = post_patch.checkpoint_id
    checkpoint_ids["post_fault"] = post_patch.checkpoint_id

    verify_exec = coord.executor.execute(
        str(manifest["test_command"]),
        cwd=str(workspace),
        timeout=1800,
    )
    verification_text = str(verify_exec.get("output", ""))
    if int(verify_exec.get("returncode", 1)) == 0:
        raise RuntimeError("Contaminated post_patch anchor unexpectedly passed verifier.")

    contradicted_fact_ids: list[str] = [coord.shared_facts["latest_patch"]["node_id"]]
    coord.recorder.mark_fact_conflict(
        coord.shared_facts["latest_patch"]["node_id"],
        reason="Synthetic contaminated post_patch anchor contradicted the promoted patch fact.",
    )
    failing_tests = coord._extract_failing_tests(verification_text)
    verifier_node = coord.recorder.record_verifier_result(
        role="verifier",
        phase="verify",
        verdict="fail",
        test_status="fail",
        failing_tests=failing_tests,
        output=verification_text,
        depends_on=[coord.stage_nodes["patcher"]] if "patcher" in coord.stage_nodes else None,
        contradicted_fact_ids=contradicted_fact_ids,
        failure_signature="|".join(failing_tests[:3]),
    )
    coord.stage_nodes["verifier"] = verifier_node.node_id
    coord.stage_outputs["verifier"] = {
        "success": False,
        "verification": verification_text,
        "status": "fail",
        "test_command": str(manifest["test_command"]),
        "synthetic_seed": True,
    }
    coord.recorder.save()

    trigger = coord.trigger_detector.detect(coord.recorder.graph)
    if trigger is None:
        raise RuntimeError("Failed to derive a recovery trigger from the contaminated post_patch state.")
    suspect_region = coord.region_extractor.extract(coord.recorder.graph, trigger)
    checkpoint = coord.recorder.latest_checkpoint()
    failed_state_metadata = coord._build_failed_state_metadata(suspect_region, checkpoint)
    failed_state_metadata["fault_type"] = "contaminated_post_patch"
    failed_state_metadata["touched_paths"] = list(changed)
    failed_state_metadata["anchor_health"] = "contaminated"
    failed_state_metadata["failure_observation"] = {
        "source_type": "controlled_perturbation",
        "fault_type": "contaminated_post_patch",
        "test_command": str(manifest["test_command"]),
        "failing_tests": list(failing_tests),
        "verifier_output_excerpt": verification_text[:4000],
    }
    failed_state = FailedState(
        group_id=f"bench_fs_{uuid.uuid4().hex[:10]}",
        run_id=coord.run_id,
        instance_id=coord.instance_id,
        trigger=trigger,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
        checkpoint=checkpoint,
        suspect_region=suspect_region,
        state_features=coord.state_encoder.encode(
            coord.recorder.graph,
            FailedState(
                group_id="tmp",
                run_id=coord.run_id,
                instance_id=coord.instance_id,
                trigger=trigger,
                checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
                checkpoint=checkpoint,
                suspect_region=suspect_region,
                state_features=None,  # type: ignore[arg-type]
                metadata=failed_state_metadata,
            ),
        ),
        metadata=failed_state_metadata,
    )
    return failed_state


def _inject_fault_and_build_failed_state(
    coord,
    *,
    manifest: dict[str, Any],
    source_snapshot: Path,
    workspace: Path,
    patch_paths: list[str],
    checkpoint_ids: dict[str, str],
) -> FailedState:
    changed = _copy_paths_from_source(source_snapshot, workspace, patch_paths)
    fault_node = coord.recorder.record_tool_call(
        role="system",
        phase="fault_injection",
        command="inject post_patch_regression",
        output="\n".join(changed),
        returncode=0,
        depends_on=[coord.stage_nodes["patcher"]] if "patcher" in coord.stage_nodes else None,
        files_touched=changed,
    )
    post_fault = coord.recorder.create_checkpoint(
        label="post_fault_regression",
        metadata={
            "resume_from": "verifier",
            "stage": "fault_injection",
            "fault_type": "post_patch_regression",
            "touched_paths": changed,
        },
        source_node_id=fault_node.node_id,
    )
    checkpoint_ids["post_fault"] = post_fault.checkpoint_id

    verify_exec = coord.executor.execute(
        str(manifest["test_command"]),
        cwd=str(workspace),
        timeout=1800,
    )
    verification_text = str(verify_exec.get("output", ""))
    if int(verify_exec.get("returncode", 1)) == 0:
        raise RuntimeError("Synthetic polluted workspace unexpectedly passed verifier.")

    contradicted_fact_ids: list[str] = []
    if "latest_patch" in coord.shared_facts:
        contradicted_fact_ids.append(coord.shared_facts["latest_patch"]["node_id"])
        coord.recorder.mark_fact_conflict(
            coord.shared_facts["latest_patch"]["node_id"],
            reason="Synthetic polluted benchmark contradicted the promoted patch fact.",
        )
    failing_tests = coord._extract_failing_tests(verification_text)
    verifier_node = coord.recorder.record_verifier_result(
        role="verifier",
        phase="verify",
        verdict="fail",
        test_status="fail",
        failing_tests=failing_tests,
        output=verification_text,
        depends_on=[coord.stage_nodes["patcher"]] if "patcher" in coord.stage_nodes else None,
        contradicted_fact_ids=contradicted_fact_ids,
        failure_signature="|".join(failing_tests[:3]),
    )
    coord.stage_nodes["verifier"] = verifier_node.node_id
    coord.stage_outputs["verifier"] = {
        "success": False,
        "verification": verification_text,
        "status": "fail",
        "test_command": str(manifest["test_command"]),
        "synthetic_seed": True,
    }
    coord.recorder.save()

    trigger = coord.trigger_detector.detect(coord.recorder.graph)
    if trigger is None:
        raise RuntimeError("Failed to derive a recovery trigger from the synthetic polluted state.")
    suspect_region = coord.region_extractor.extract(coord.recorder.graph, trigger)
    checkpoint = coord.recorder.latest_checkpoint()
    failed_state_metadata = coord._build_failed_state_metadata(suspect_region, checkpoint)
    failed_state_metadata["fault_type"] = "post_patch_regression"
    failed_state_metadata["touched_paths"] = list(changed)
    failed_state_metadata["failure_observation"] = {
        "source_type": "controlled_perturbation",
        "fault_type": "post_patch_regression",
        "test_command": str(manifest["test_command"]),
        "failing_tests": list(failing_tests),
        "verifier_output_excerpt": verification_text[:4000],
    }
    failed_state = FailedState(
        group_id=f"bench_fs_{uuid.uuid4().hex[:10]}",
        run_id=coord.run_id,
        instance_id=coord.instance_id,
        trigger=trigger,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
        checkpoint=checkpoint,
        suspect_region=suspect_region,
        state_features=coord.state_encoder.encode(
            coord.recorder.graph,
            FailedState(
                group_id="tmp",
                run_id=coord.run_id,
                instance_id=coord.instance_id,
                trigger=trigger,
                checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
                checkpoint=checkpoint,
                suspect_region=suspect_region,
                state_features=None,  # type: ignore[arg-type]
                metadata=failed_state_metadata,
            ),
        ),
        metadata=failed_state_metadata,
    )
    return failed_state


def _verify_failure_and_build_failed_state(
    coord,
    *,
    manifest: dict[str, Any],
    fault_type: str,
    touched_paths: list[str],
    contradicted_fact_keys: list[str],
    verifier_dep_stage: str,
    extra_metadata: dict[str, Any] | None = None,
) -> FailedState:
    verify_exec = coord.executor.execute(
        str(manifest["test_command"]),
        cwd=str(coord.workspace),
        timeout=1800,
    )
    verification_text = str(verify_exec.get("output", ""))
    if int(verify_exec.get("returncode", 1)) == 0:
        raise RuntimeError(f"Invalid perturbation: {fault_type} unexpectedly passed verifier.")

    contradicted_fact_ids: list[str] = []
    for fact_key in contradicted_fact_keys:
        fact = coord.shared_facts.get(fact_key)
        if not fact:
            continue
        fact_node_id = str(fact.get("node_id", ""))
        if not fact_node_id:
            continue
        contradicted_fact_ids.append(fact_node_id)
        coord.recorder.mark_fact_conflict(
            fact_node_id,
            reason=f"Synthetic {fault_type} benchmark contradicted the promoted {fact_key} fact.",
        )

    failing_tests = coord._extract_failing_tests(verification_text)
    verifier_dep = coord.stage_nodes.get(verifier_dep_stage) or coord.stage_nodes.get("patcher") or coord.stage_nodes.get("locator")
    verifier_node = coord.recorder.record_verifier_result(
        role="verifier",
        phase="verify",
        verdict="fail",
        test_status="fail",
        failing_tests=failing_tests,
        output=verification_text,
        depends_on=[verifier_dep] if verifier_dep else None,
        contradicted_fact_ids=contradicted_fact_ids,
        failure_signature="|".join(failing_tests[:3]),
    )
    coord.stage_nodes["verifier"] = verifier_node.node_id
    coord.stage_outputs["verifier"] = {
        "success": False,
        "verification": verification_text,
        "status": "fail",
        "test_command": str(manifest["test_command"]),
        "synthetic_seed": True,
    }
    coord.recorder.save()

    trigger = coord.trigger_detector.detect(coord.recorder.graph)
    if trigger is None:
        raise RuntimeError(f"Failed to derive a recovery trigger from the {fault_type} state.")
    suspect_region = coord.region_extractor.extract(coord.recorder.graph, trigger)
    checkpoint = coord.recorder.latest_checkpoint()
    failed_state_metadata = coord._build_failed_state_metadata(suspect_region, checkpoint)
    failed_state_metadata["fault_type"] = fault_type
    failed_state_metadata["touched_paths"] = list(touched_paths)
    if extra_metadata:
        failed_state_metadata.update(extra_metadata)
    failed_state_metadata["failure_observation"] = {
        "source_type": "controlled_perturbation",
        "fault_type": fault_type,
        "test_command": str(manifest["test_command"]),
        "failing_tests": list(failing_tests),
        "verifier_output_excerpt": verification_text[:4000],
        **(dict(extra_metadata or {})),
    }
    failed_state = FailedState(
        group_id=f"bench_fs_{uuid.uuid4().hex[:10]}",
        run_id=coord.run_id,
        instance_id=coord.instance_id,
        trigger=trigger,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
        checkpoint=checkpoint,
        suspect_region=suspect_region,
        state_features=coord.state_encoder.encode(
            coord.recorder.graph,
            FailedState(
                group_id="tmp",
                run_id=coord.run_id,
                instance_id=coord.instance_id,
                trigger=trigger,
                checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
                checkpoint=checkpoint,
                suspect_region=suspect_region,
                state_features=None,  # type: ignore[arg-type]
                metadata=failed_state_metadata,
            ),
        ),
        metadata=failed_state_metadata,
    )
    return failed_state


def _seed_localization_pollution_and_build_failed_state(
    coord,
    *,
    manifest: dict[str, Any],
    source_snapshot: Path,
    workspace: Path,
    patch_paths: list[str],
    checkpoint_ids: dict[str, str],
    manual_spec: dict[str, Any],
) -> FailedState:
    wrong_path = str(manual_spec.get("wrong_localized_path", "")).strip()
    if not wrong_path:
        raise RuntimeError("manual_fault_spec_not_found: localization_pollution requires wrong_localized_path")
    if wrong_path in set(patch_paths):
        raise RuntimeError("Invalid perturbation: wrong localized path equals an oracle patch path.")
    if not (source_snapshot / wrong_path).exists():
        raise RuntimeError(f"Invalid perturbation: wrong localized path is missing: {wrong_path}")

    changed = _copy_paths_from_source(source_snapshot, workspace, patch_paths)
    locator_node = coord.recorder.record_agent_step(
        role="locator",
        phase="locate",
        content=wrong_path,
        payload={
            "success": True,
            "synthetic_seed": True,
            "fault_type": "localization_pollution",
            "correct_paths": list(patch_paths),
            "wrong_localized_path": wrong_path,
        },
    )
    coord.stage_nodes["locator"] = locator_node.node_id
    coord.stage_outputs["locator"] = {
        "success": True,
        "located_files": wrong_path,
        "synthetic_seed": True,
        "fault_type": "localization_pollution",
    }
    localized_fact = coord.recorder.record_shared_fact(
        key="localized_path",
        value=wrong_path,
        role="locator",
        phase="locate",
        source_node_id=locator_node.node_id,
        confidence=0.92,
        payload={
            "synthetic_seed": True,
            "fault_type": "localization_pollution",
            "is_polluted": True,
            "correct_paths": list(patch_paths),
        },
    )
    coord.shared_facts["localized_path"] = {
        "value": wrong_path,
        "node_id": localized_fact.node_id,
    }

    post_locate = coord.recorder.create_checkpoint(
        label="post_locate",
        metadata={
            "resume_from": "patcher",
            "stage": "locator",
            "synthetic_seed": True,
            "fault_type": "localization_pollution",
            "wrong_localized_path": wrong_path,
            "correct_paths": list(patch_paths),
            "touched_paths": list(changed),
        },
        source_node_id=locator_node.node_id,
    )
    checkpoint_ids["post_locate"] = post_locate.checkpoint_id

    patch_node = coord.recorder.record_agent_step(
        role="patcher",
        phase="patch",
        content=f"plausible patch attempt focused on wrong localized path: {wrong_path}",
        payload={
            "success": True,
            "patch": "",
            "synthetic_seed": True,
            "fault_type": "localization_pollution",
            "wrong_localized_path": wrong_path,
            "correct_paths": list(patch_paths),
        },
        depends_on=[locator_node.node_id],
        reads=[localized_fact.node_id],
    )
    coord.stage_nodes["patcher"] = patch_node.node_id
    coord.stage_outputs["patcher"] = {
        "success": True,
        "plan": f"Synthetic wrong-localization patch attempt on {wrong_path}.",
        "patch": "",
        "commands": [],
        "synthetic_seed": True,
        "fault_type": "localization_pollution",
    }
    post_patch = coord.recorder.create_checkpoint(
        label="post_patch",
        metadata={
            "resume_from": "verifier",
            "stage": "patcher",
            "synthetic_seed": True,
            "fault_type": "localization_pollution",
            "anchor_health": "wrong_localization",
            "wrong_localized_path": wrong_path,
            "correct_paths": list(patch_paths),
            "touched_paths": list(changed),
        },
        source_node_id=patch_node.node_id,
    )
    checkpoint_ids["post_patch"] = post_patch.checkpoint_id
    checkpoint_ids["post_fault"] = post_patch.checkpoint_id

    return _verify_failure_and_build_failed_state(
        coord,
        manifest=manifest,
        fault_type="localization_pollution",
        touched_paths=changed,
        contradicted_fact_keys=["localized_path"],
        verifier_dep_stage="patcher",
        extra_metadata={
            "wrong_localized_path": wrong_path,
            "correct_paths": list(patch_paths),
            "anchor_health": "wrong_localization",
        },
    )


def _seed_partial_patch_and_build_failed_state(
    coord,
    *,
    manifest: dict[str, Any],
    source_snapshot: Path,
    oracle_snapshot: Path,
    workspace: Path,
    patch_paths: list[str],
    checkpoint_ids: dict[str, str],
    manual_spec: dict[str, Any],
) -> FailedState:
    mode = str(manual_spec.get("mode", "apply_first_changed_region_only")).strip()
    source_reset_paths = _copy_paths_from_source(source_snapshot, workspace, patch_paths)

    located_summary = "\n".join(patch_paths)
    locator_node = coord.recorder.record_agent_step(
        role="locator",
        phase="locate",
        content=located_summary or "oracle_patch_files",
        payload={
            "success": True,
            "synthetic_seed": True,
            "fault_type": "partial_patch",
        },
    )
    coord.stage_nodes["locator"] = locator_node.node_id
    coord.stage_outputs["locator"] = {
        "success": True,
        "located_files": located_summary,
        "synthetic_seed": True,
        "fault_type": "partial_patch",
    }
    localized_fact = coord.recorder.record_shared_fact(
        key="localized_path",
        value=patch_paths[0],
        role="locator",
        phase="locate",
        source_node_id=locator_node.node_id,
        confidence=0.95,
        payload={"synthetic_seed": True, "fault_type": "partial_patch"},
    )
    coord.shared_facts["localized_path"] = {
        "value": patch_paths[0],
        "node_id": localized_fact.node_id,
    }
    post_locate = coord.recorder.create_checkpoint(
        label="post_locate",
        metadata={
            "resume_from": "patcher",
            "stage": "locator",
            "synthetic_seed": True,
            "fault_type": "partial_patch",
            "touched_paths": list(source_reset_paths),
        },
        source_node_id=locator_node.node_id,
    )
    checkpoint_ids["post_locate"] = post_locate.checkpoint_id

    changed = _apply_partial_oracle_patch(
        source_snapshot,
        oracle_snapshot,
        workspace,
        patch_paths,
        mode=mode,
        selected_region_indices=_manual_int_list(manual_spec.get("selected_region_indices")),
        selected_regions_by_path=_manual_selected_regions_by_path(manual_spec.get("selected_regions_by_path")),
        region_index=int(manual_spec.get("region_index") or 0),
        oracle_line_count=int(manual_spec.get("oracle_line_count") or 0),
    )
    if _paths_match_snapshot(workspace, source_snapshot, patch_paths):
        raise RuntimeError("Invalid perturbation: partial patch is indistinguishable from source snapshot.")
    if _paths_match_snapshot(workspace, oracle_snapshot, patch_paths):
        raise RuntimeError("Invalid perturbation: partial patch is equivalent to the full oracle patch.")

    patch_text = _compute_patch_text(source_snapshot, workspace, patch_paths)
    patch_node = coord.recorder.record_agent_step(
        role="patcher",
        phase="patch",
        content=patch_text[:4000] or f"partial patch via {mode}",
        payload={
            "success": True,
            "patch": patch_text[:4000],
            "synthetic_seed": True,
            "fault_type": "partial_patch",
            "partial_mode": mode,
            "touched_paths": list(changed),
        },
        depends_on=[locator_node.node_id],
        reads=[localized_fact.node_id],
    )
    coord.stage_nodes["patcher"] = patch_node.node_id
    coord.stage_outputs["patcher"] = {
        "success": True,
        "plan": f"Synthetic partial oracle patch using mode={mode}.",
        "patch": patch_text,
        "commands": [],
        "synthetic_seed": True,
        "fault_type": "partial_patch",
        "partial_mode": mode,
    }
    patch_fact = coord.recorder.record_shared_fact(
        key="latest_patch",
        value=patch_text[:400] if patch_text else "\n".join(patch_paths),
        role="patcher",
        phase="patch",
        source_node_id=patch_node.node_id,
        confidence=0.80,
        payload={
            "synthetic_seed": True,
            "fault_type": "partial_patch",
            "partial_mode": mode,
            "anchor_health": "partial_patch",
        },
    )
    coord.shared_facts["latest_patch"] = {
        "value": patch_text,
        "node_id": patch_fact.node_id,
    }
    post_patch = coord.recorder.create_checkpoint(
        label="post_patch",
        metadata={
            "resume_from": "verifier",
            "stage": "patcher",
            "synthetic_seed": True,
            "fault_type": "partial_patch",
            "anchor_health": "partial_patch",
            "partial_mode": mode,
            "touched_paths": list(changed),
        },
        source_node_id=patch_node.node_id,
    )
    checkpoint_ids["post_patch"] = post_patch.checkpoint_id
    checkpoint_ids["post_fault"] = post_patch.checkpoint_id

    return _verify_failure_and_build_failed_state(
        coord,
        manifest=manifest,
        fault_type="partial_patch",
        touched_paths=changed,
        contradicted_fact_keys=["latest_patch"],
        verifier_dep_stage="patcher",
        extra_metadata={
            "partial_mode": mode,
            "anchor_health": "partial_patch",
        },
    )


def _candidate_programs(
    checkpoint_ids: dict[str, str],
    *,
    program_space_version: str = "legacy",
    fault_family: str | None = None,
) -> list[RecoveryProgram]:
    return _candidate_programs_for_profile(
        checkpoint_ids,
        profile="full_matrix",
        program_space_version=program_space_version,
        fault_family=fault_family,
    )


def _candidate_programs_for_profile(
    checkpoint_ids: dict[str, str],
    *,
    profile: str,
    program_space_version: str = "legacy",
    fault_family: str | None = None,
) -> list[Any]:
    normalized = profile.strip().lower()
    version = str(program_space_version).strip().lower()
    normalized_fault_family = str(fault_family or "").strip().lower()
    if version not in {
        "legacy",
        "intent_discovery_v1",
        "intent_discovery_v2",
        "intent_schema_v1",
        "intent_schema_v2",
        "intent_schema_v3",
        "rollback_only_v1",
        "atomic_actions_v1",
        "semantic_dual_v1",
        "semantic_closed_loop_v1",
        "semantic_action_loop_v1",
        "semantic_object_loop_v2",
    }:
        raise ValueError(f"Unsupported program_space_version: {program_space_version}")

    # Early-return newer program spaces before constructing legacy templates:
    # some natural failures stop at post_locate and legitimately have no
    # post_patch checkpoint.
    if version == "atomic_actions_v1":
        raise ValueError("atomic_actions_v1 must be synthesized from the live failed state.")

    if version == "rollback_only_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return rollback_only_programs_v1(checkpoint_ids)
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_dual_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_closed_loop_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_closed_loop_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_action_loop_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_action_loop_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_object_loop_v2":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_object_loop_programs_v2(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v1":
        if normalized in {"local_rebuild", "full_matrix"}:
            return discovery_programs_v1(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
        if normalized == "anchor_restore":
            schema_programs = discovery_programs_v1(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
            return [program for program in schema_programs if str(program.metadata.get("family")) in {"local_minimal", "global"}]
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v2":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return intent_programs_v2(checkpoint_ids)
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v3":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return intent_programs_v3(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    minimal = RecoveryProgram(
        program_id="local_anchor_restore",
        steps=[
            RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_ids["post_patch"]}),
        ],
        rationale="Minimal recovery: restore the last known healthy patch checkpoint and let the official tests judge the outcome.",
        estimated_total_cost=80.0,
        estimated_recover_prob=0.95,
        estimated_risk=0.01,
        metadata={"family": "local_minimal", "strategy": "rollback_post_patch_restore"},
    )
    local_rebuild = RecoveryProgram(
        program_id="local_patch_replay",
        steps=[
            RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_ids["post_locate"]}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "patcher+verifier",
                    "context_hint": "Rebuild only the downstream patch-and-verify region from the localization anchor.",
                },
            ),
        ],
        rationale="Broader local recovery from the localization checkpoint.",
        estimated_total_cost=2200.0,
        estimated_recover_prob=0.80,
        estimated_risk=0.10,
        metadata={"family": "local_broader", "strategy": "rollback_post_locate_patch_verify"},
    )
    global_restart = RecoveryProgram(
        program_id="global_full_restart",
        steps=[
            RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_ids["initial"]}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "full",
                    "context_hint": "Discard all intermediate progress and restart the full SWE pipeline.",
                },
            ),
        ],
        rationale="Coarse baseline: restart the full pipeline.",
        estimated_total_cost=6000.0,
        estimated_recover_prob=0.75,
        estimated_risk=0.25,
        metadata={"family": "global", "strategy": "rollback_initial_full_replay"},
    )

    evidence_recheck_local = RecoveryProgram(
        program_id="evidence_recheck_patch_replay",
        steps=[
            RecoveryStep(op=OpType.INSPECT, args={"target": "test_output", "depth": "deep"}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "patcher+verifier",
                    "context_hint": "Re-check the failing evidence and rebuild only the patch/verify region without trusting the previous local conclusion.",
                },
            ),
        ],
        rationale="Evidence-oriented local recovery: re-check the failure signal before a local patch replay.",
        estimated_total_cost=1800.0,
        estimated_recover_prob=0.68,
        estimated_risk=0.08,
        metadata={"family": "evidence_recheck", "strategy": "inspect_test_then_patch_replay"},
    )

    belief_cleanup_local = RecoveryProgram(
        program_id="belief_cleanup_patch_replay",
        steps=[
            RecoveryStep(op=OpType.REVOKE, args={"fact_id": "fact:latest_patch"}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "patcher+verifier",
                    "context_hint": "The promoted patch fact was contradicted; discard it and derive a fresh local repair from the current failing workspace.",
                },
            ),
        ],
        rationale="Belief-cleanup recovery: remove the stale promoted patch fact before local replay.",
        estimated_total_cost=1700.0,
        estimated_recover_prob=0.70,
        estimated_risk=0.07,
        metadata={"family": "belief_cleanup", "strategy": "revoke_latest_patch_then_patch_replay"},
    )

    capability_boost_local = RecoveryProgram(
        program_id="capability_boost_patch_replay",
        steps=[
            RecoveryStep(op=OpType.ESCALATE, args={"scope": "patcher", "strategy": "stronger_prompt", "escalation_level": 1}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "patcher+verifier",
                    "context_hint": "Keep the local target but retry with stronger local repair reasoning.",
                },
            ),
        ],
        rationale="Capability-boost recovery: improve local repair capability before replaying patch/verify.",
        estimated_total_cost=2400.0,
        estimated_recover_prob=0.72,
        estimated_risk=0.11,
        metadata={"family": "capability_boost", "strategy": "escalate_patcher_then_patch_replay"},
    )

    relocalize_rebuild = RecoveryProgram(
        program_id="relocalize_and_rebuild",
        steps=[
            RecoveryStep(op=OpType.REVOKE, args={"fact_id": "fact:localized_path"}),
            RecoveryStep(op=OpType.ESCALATE, args={"scope": "locator", "strategy": "broader_search", "escalation_level": 1}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "locator+patcher+verifier",
                    "context_hint": "The previous target hypothesis may be stale; re-localize more broadly, then rebuild patch and verify.",
                },
            ),
        ],
        rationale="Re-localize then rebuild when the localized target itself may be polluted or stale.",
        estimated_total_cost=3200.0,
        estimated_recover_prob=0.66,
        estimated_risk=0.14,
        metadata={"family": "relocalize", "strategy": "revoke_localization_escalate_locator_then_rebuild"},
    )

    normalized = profile.strip().lower()
    version = str(program_space_version).strip().lower()
    normalized_fault_family = str(fault_family or "").strip().lower()
    if version not in {
        "legacy",
        "intent_discovery_v1",
        "intent_discovery_v2",
        "intent_schema_v1",
        "intent_schema_v2",
        "intent_schema_v3",
        "rollback_only_v1",
        "atomic_actions_v1",
        "semantic_dual_v1",
        "semantic_closed_loop_v1",
        "semantic_action_loop_v1",
        "semantic_object_loop_v2",
    }:
        raise ValueError(f"Unsupported program_space_version: {program_space_version}")

    if version == "atomic_actions_v1":
        raise ValueError("atomic_actions_v1 must be synthesized from the live failed state.")

    if version == "rollback_only_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return rollback_only_programs_v1(checkpoint_ids)
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_dual_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_closed_loop_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_closed_loop_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_action_loop_v1":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_action_loop_programs_v1(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "semantic_object_loop_v2":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return semantic_object_loop_programs_v2(
                checkpoint_ids,
                profile=normalized,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v1":
        if normalized in {"local_rebuild", "full_matrix"}:
            return discovery_programs_v1(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
        if normalized == "anchor_restore":
            schema_programs = discovery_programs_v1(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
            return [program for program in schema_programs if str(program.metadata.get("family")) in {"local_minimal", "global"}]
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v2":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return intent_programs_v2(checkpoint_ids)
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if version == "intent_schema_v3":
        if normalized in {"anchor_restore", "local_rebuild", "full_matrix"}:
            return intent_programs_v3(
                checkpoint_ids,
                fault_family=normalized_fault_family,
            )
        raise ValueError(f"Unsupported recovery benchmark profile: {profile}")

    if normalized == "anchor_restore":
        return [minimal, global_restart]
    if normalized == "local_rebuild":
        if version == "intent_discovery_v1":
            return [local_rebuild, evidence_recheck_local, capability_boost_local, global_restart]
        if version == "intent_discovery_v2":
            if normalized_fault_family == "contaminated_post_patch":
                return [local_rebuild, belief_cleanup_local, evidence_recheck_local, capability_boost_local, global_restart]
            return [local_rebuild, belief_cleanup_local, evidence_recheck_local, global_restart]
        return [local_rebuild, global_restart]
    if normalized == "full_matrix":
        if version == "intent_discovery_v1":
            return [
                minimal,
                belief_cleanup_local,
                evidence_recheck_local,
                capability_boost_local,
                relocalize_rebuild,
                local_rebuild,
                global_restart,
            ]
        if version == "intent_discovery_v2":
            base_programs = [
                minimal,
                belief_cleanup_local,
                evidence_recheck_local,
                local_rebuild,
                global_restart,
            ]
            if normalized_fault_family == "contaminated_post_patch":
                return [
                    minimal,
                    belief_cleanup_local,
                    evidence_recheck_local,
                    capability_boost_local,
                    local_rebuild,
                    global_restart,
                ]
            return base_programs
        return [minimal, local_rebuild, global_restart]
    raise ValueError(f"Unsupported recovery benchmark profile: {profile}")


def _infer_family_from_program(program: RecoveryProgram) -> str:
    explicit = str(program.metadata.get("family", "")).strip().lower()
    if explicit:
        return explicit
    ops = [step.op for step in program.steps]
    if OpType.ESCALATE in ops:
        return "capability_boost"
    if OpType.REVOKE in ops:
        return "belief_cleanup"
    if OpType.INSPECT in ops:
        return "evidence_recheck"
    if OpType.ROLLBACK in ops:
        return "global" if _program_scope(program) == "full" else "local_replan"
    if OpType.REPLAY in ops:
        return "direct_replay"
    return "unknown"


def _synthesized_atomic_programs(
    coord,
    *,
    failed_state: FailedState,
    max_candidate_programs: int,
    program_online_selector: str,
) -> list[RecoveryProgram]:
    synthesizer = getattr(coord, "synthesizer", None)
    if synthesizer is None:
        raise RuntimeError("Coordinator has no synthesizer for atomic action composition.")
    synthesizer.max_candidate_programs = max(1, int(max_candidate_programs))
    similar_cases = None
    if getattr(coord, "case_memory", None) is not None:
        try:
            similar_cases = coord.case_memory.retrieve(failed_state, top_k=3)
        except Exception:
            similar_cases = None
    programs = synthesizer.synthesize(
        coord.recorder.graph,
        failed_state,
        coord.config.budget,
        similar_cases=similar_cases,
        used_recovery_calls=0,
        used_tokens=0.0,
        preserve_recommended=program_online_selector == "llm_recommended_top1",
    )
    for idx, program in enumerate(programs, start=1):
        program.metadata.setdefault("family", _infer_family_from_program(program))
        program.metadata.setdefault("strategy", f"atomic_compose_{idx}")
        program.metadata.setdefault("program_space_version", "atomic_actions_v1")
        program.metadata.setdefault("online_selection_mode", program_online_selector)
    coord.stage_outputs["atomic_program_synthesis"] = {
        **synthesizer.get_synthesis_record(),
        "returned_candidate_programs": [program.to_dict() for program in programs],
    }
    return programs


def _limit_programs_for_online_execution(
    programs: list[RecoveryProgram],
    *,
    failed_state: FailedState,
    budget,
    execute_candidate_limit: int,
    program_online_selector: str = "heuristic_rerank",
) -> list[RecoveryProgram]:
    if execute_candidate_limit <= 0:
        return programs
    limit = max(1, int(execute_candidate_limit))
    selector = str(program_online_selector or "heuristic_rerank").strip().lower()
    if selector == "raw_order_top1":
        return programs[:limit]
    if selector == "llm_recommended_top1":
        recommended = next(
            (
                program
                for program in programs
                if bool(program.metadata.get("llm_recommended"))
            ),
            None,
        )
        if recommended is None:
            return programs[:limit]
        selected = [recommended] + [
            program for program in programs if program.program_id != recommended.program_id
        ]
        return selected[:limit]
    if selector == "semantic_action_state_top1":
        scored_programs = []
        for idx, program in enumerate(programs):
            score, debug = _semantic_action_starter_priority_with_debug(
                program,
                failed_state=failed_state,
                budget=budget,
            )
            program.metadata["online_selector_debug"] = {
                **debug,
                "estimated_total_cost": float(program.estimated_total_cost),
                "estimated_recover_prob": float(program.estimated_recover_prob),
                "rank_tiebreak_cost": -float(program.estimated_total_cost),
                "candidate_index": idx,
            }
            scored_programs.append((score, program))
        ranked = [
            program
            for _, program in sorted(
                scored_programs,
                key=lambda item: (
                    item[0],
                    -float(item[1].estimated_total_cost),
                ),
                reverse=True,
            )
        ]
        return ranked[:limit]
    if selector == "parc_lifecycle_frontier_top1":
        scored_programs = []
        for idx, program in enumerate(programs):
            score, debug = _parc_lifecycle_starter_priority_with_debug(
                program,
                failed_state=failed_state,
                budget=budget,
            )
            program.metadata["online_selector_debug"] = {
                **debug,
                "estimated_total_cost": float(program.estimated_total_cost),
                "estimated_recover_prob": float(program.estimated_recover_prob),
                "rank_tiebreak_cost": -float(program.estimated_total_cost),
                "candidate_index": idx,
            }
            scored_programs.append((score, program))
        ranked = [
            program
            for _, program in sorted(
                scored_programs,
                key=lambda item: (
                    item[0],
                    -float(item[1].estimated_total_cost),
                ),
                reverse=True,
            )
        ]
        return ranked[:limit]
    if selector == "car_counterexample_top1":
        scored_programs = []
        for idx, program in enumerate(programs):
            score, debug = _car_counterexample_starter_priority_with_debug(
                program,
                failed_state=failed_state,
                budget=budget,
            )
            program.metadata["online_selector_debug"] = {
                **debug,
                "estimated_total_cost": float(program.estimated_total_cost),
                "estimated_recover_prob": float(program.estimated_recover_prob),
                "rank_tiebreak_cost": -float(program.estimated_total_cost),
                "candidate_index": idx,
            }
            scored_programs.append((score, program))
        ranked = [
            program
            for _, program in sorted(
                scored_programs,
                key=lambda item: (
                    item[0],
                    -float(item[1].estimated_total_cost),
                ),
                reverse=True,
            )
        ]
        return ranked[:limit]
    if selector == "semantic_object_state_top1":
        scored_programs = []
        for idx, program in enumerate(programs):
            score, debug = _semantic_object_macro_priority_with_debug(
                program,
                failed_state=failed_state,
                budget=budget,
            )
            program.metadata["online_selector_debug"] = {
                **debug,
                "estimated_total_cost": float(program.estimated_total_cost),
                "estimated_recover_prob": float(program.estimated_recover_prob),
                "rank_tiebreak_cost": -float(program.estimated_total_cost),
                "candidate_index": idx,
            }
            scored_programs.append((score, program))
        ranked = [
            program
            for _, program in sorted(
                scored_programs,
                key=lambda item: (
                    item[0],
                    -float(item[1].estimated_total_cost),
                ),
                reverse=True,
            )
        ]
        return ranked[:limit]
    ranked = sorted(
        programs,
        key=lambda program: (
            _program_online_priority(program, failed_state=failed_state, budget=budget),
            -float(program.estimated_total_cost),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _filter_programs_by_id(
    programs: list[Any],
    *,
    program_id_filter: list[str] | None,
) -> list[Any]:
    wanted = {
        str(item).strip()
        for item in (program_id_filter or [])
        if str(item).strip()
    }
    if not wanted:
        return programs
    filtered = [
        program
        for program in programs
        if str(getattr(program, "program_id", "")).strip() in wanted
    ]
    missing = sorted(
        wanted
        - {
            str(getattr(program, "program_id", "")).strip()
            for program in filtered
        }
    )
    if missing:
        available = ", ".join(str(getattr(program, "program_id", "")) for program in programs)
        raise ValueError(
            f"Unknown program_id_filter entries: {', '.join(missing)}. "
            f"Available programs: {available}"
        )
    return filtered


def _seed_case_memory_from_dir(coord, *, seed_dir: Path) -> dict[str, Any]:
    target_dir = Path(coord.run_dir) / "case_memory"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(seed_dir.glob("case_*.json")):
        shutil.copy2(path, target_dir / path.name)
        copied += 1
    skills_path = seed_dir / "skills.json"
    if skills_path.exists():
        shutil.copy2(skills_path, target_dir / "skills.json")
    coord.case_memory = CaseMemory(target_dir)
    return {
        "seed_dir": str(seed_dir),
        "copied_cases": copied,
        "loaded_cases": len(coord.case_memory.cases),
    }


def _derive_failure_mode(
    *,
    actual_success: bool,
    legal_success: bool,
    patch_legality: dict[str, Any],
    patcher_trace: dict[str, Any] | None = None,
    oracle_infra_error: bool = False,
) -> str:
    if oracle_infra_error:
        return "oracle_infra_error"
    trace = dict(patcher_trace or {})
    if bool(trace.get("patcher_infrastructure_error", False)):
        return "replay_model_infra_error"
    if legal_success:
        return "resolved"
    if actual_success and not legal_success:
        return "illegal_success"

    patch_scope = str(patch_legality.get("patch_scope", "") or "")
    source_files = list((patch_legality.get("changed_file_classes", {}) or {}).get("source_files", []) or [])
    generated_files = list((patch_legality.get("changed_file_classes", {}) or {}).get("generated_files", []) or [])
    other_files = list((patch_legality.get("changed_file_classes", {}) or {}).get("other_files", []) or [])
    test_files = list((patch_legality.get("changed_file_classes", {}) or {}).get("test_files", []) or [])
    touches_suspect = bool(patch_legality.get("touches_suspect_path", False))
    patcher_command_count = trace.get("patcher_command_count")
    patcher_message_count = trace.get("patcher_message_count")

    if patch_scope == "no_diff":
        if patcher_command_count is not None and int(patcher_command_count or 0) == 0:
            return "replay_no_patcher_activity"
        if patcher_message_count is not None and int(patcher_message_count or 0) == 0:
            return "replay_no_patcher_activity"
        return "no_diff"
    if (generated_files or other_files or test_files) and not source_files and not touches_suspect:
        return "wrong_edit_target"
    if source_files and not touches_suspect:
        return "source_edit_but_not_suspect"
    if source_files:
        return "oracle_failed_after_source_edit"
    return "wrong_edit_target"


def _is_oracle_infra_error(returncode: int, output: str) -> bool:
    """Detect evaluation-command failures that should not be counted as repair failures."""
    normalized = str(output or "").lower()
    patch_breakage_markers = (
        "importerror",
        "modulenotfounderror",
        "cannot import name",
        "syntaxerror",
        "indentationerror",
        "no module named",
    )
    if any(marker in normalized for marker in patch_breakage_markers):
        return False
    if int(returncode) == 127:
        return True
    if "command not found" in normalized or "not found" in normalized and "pytest" in normalized:
        return True
    if "pytest: not found" in normalized or "py.test: not found" in normalized:
        return True
    if int(returncode) == 4 and "error: not found" in normalized:
        return True
    if "no name " in normalized and " in any of " in normalized:
        return True
    return False


def _evaluation_status(
    *,
    actual_success: bool,
    oracle_infra_error: bool,
) -> str:
    if oracle_infra_error:
        return "invalid_oracle_command"
    if actual_success:
        return "oracle_passed"
    return "oracle_failed"


def _success_legitimacy_tier(
    *,
    actual_success: bool,
    patch_legality: dict[str, Any],
) -> str:
    if not actual_success:
        return "not_resolved"

    patch_scope = str(patch_legality.get("patch_scope", "") or "")
    file_classes = patch_legality.get("changed_file_classes", {}) or {}
    source_files = list(file_classes.get("source_files", []) or [])
    test_files = list(file_classes.get("test_files", []) or [])
    generated_files = list(file_classes.get("generated_files", []) or [])
    other_files = list(file_classes.get("other_files", []) or [])
    touches_suspect = bool(patch_legality.get("touches_suspect_path", False))

    if patch_scope == "no_diff":
        return "checkpoint_or_no_diff_success"
    if test_files and not source_files:
        return "weak_tests_only_success"
    if source_files and touches_suspect and test_files:
        return "mixed_source_test_success"
    if source_files and touches_suspect:
        return "strong_source_success"
    if source_files:
        return "weak_off_suspect_source_success"
    if generated_files or other_files:
        return "weak_wrong_target_success"
    return "unknown_success"


def _guarded_recover_success_from_legitimacy(success_legitimacy: str) -> bool:
    return str(success_legitimacy or "") == "strong_source_success"


def _is_legal_recovery_success(
    *,
    actual_success: bool,
    patch_legality: dict[str, Any],
    oracle_infra_error: bool = False,
) -> bool:
    """Return whether an oracle pass should count as a valid recovery.

    Checkpoint/no-diff success is still valid for rollback-style recovery, but
    oracle passes that only touch tests, generated outputs, or unrelated files
    should not be counted as strong method success.
    """
    if not actual_success or oracle_infra_error:
        return False

    patch_scope = str(patch_legality.get("patch_scope", "") or "")
    if patch_scope == "tests_only":
        return False

    file_classes = patch_legality.get("changed_file_classes", {}) or {}
    source_files = list(file_classes.get("source_files", []) or [])
    test_files = list(file_classes.get("test_files", []) or [])
    generated_files = list(file_classes.get("generated_files", []) or [])
    other_files = list(file_classes.get("other_files", []) or [])
    wrong_target_only = bool(test_files or generated_files or other_files) and not source_files
    if patch_scope != "no_diff" and wrong_target_only:
        return False

    return True


def _selected_rollback_anchor_label(
    *,
    failed_state: FailedState,
    program,
    outcome: ProgramOutcome,
) -> str | None:
    for step_outcome in outcome.step_outcomes:
        if str(step_outcome.get("op", "")).upper() not in {"ROLLBACK", "ROLLBACK_ANCHOR"}:
            continue
        result = dict(step_outcome.get("result") or {})
        label = str(result.get("checkpoint_label", "")).strip()
        if label:
            return label
        checkpoint_id = str(result.get("checkpoint_id", "")).strip()
        if checkpoint_id:
            resolved = _checkpoint_label_for_id(failed_state, checkpoint_id)
            if resolved:
                return resolved
    for step in program.steps:
        step_op = getattr(step, "op", None)
        op_value = getattr(step_op, "value", "")
        if step_op != OpType.ROLLBACK and op_value != PrimitiveOpType.ROLLBACK_ANCHOR.value:
            continue
        label = str(step.args.get("checkpoint_label", "")).strip()
        if label:
            return label
        checkpoint_id = str(step.args.get("checkpoint_id", "")).strip()
        if checkpoint_id:
            resolved = _checkpoint_label_for_id(failed_state, checkpoint_id)
            if resolved:
                return resolved
    return None


def _selected_replay_resume_stage(
    program,
    outcome: ProgramOutcome,
) -> str | None:
    for step_outcome in outcome.step_outcomes:
        if str(step_outcome.get("op", "")).upper() not in {"REPLAY", "CONSTRAINED_REPLAY"}:
            continue
        result = dict(step_outcome.get("result") or {})
        resumed = str(result.get("resumed_from", "")).strip()
        if resumed:
            return resumed
    for step in program.steps:
        step_op = getattr(step, "op", None)
        op_value = getattr(step_op, "value", "")
        if step_op != OpType.REPLAY and op_value != PrimitiveOpType.CONSTRAINED_REPLAY.value:
            continue
        scope = str(step.args.get("scope", "")).strip().lower()
        if "+" in scope:
            return scope.split("+")[0].strip()
        if scope == "full":
            return "locator"
        if scope:
            return scope
    return None


def _selected_replay_patcher_trace(outcome: ProgramOutcome) -> dict[str, Any]:
    for step_outcome in outcome.step_outcomes:
        if str(step_outcome.get("op", "")).upper() not in {"REPLAY", "CONSTRAINED_REPLAY"}:
            continue
        result = dict(step_outcome.get("result") or {})
        trace = result.get("patcher_trace")
        if isinstance(trace, dict):
            return trace
    return {}


def _evaluate_programs(
    coord,
    *,
    failed_state: FailedState,
    programs: list[Any],
    manifest: dict[str, Any],
    polluted_checkpoint_id: str,
) -> list[ProgramOutcome]:
    baseline_shared_facts = copy.deepcopy(coord.shared_facts)
    baseline_stage_outputs = copy.deepcopy(coord.stage_outputs)
    baseline_stage_nodes = copy.deepcopy(coord.stage_nodes)
    baseline_graph = copy.deepcopy(coord.recorder.graph)
    baseline_checkpoints = copy.deepcopy(coord.recorder._checkpoints)
    baseline_recovery_calls = coord.recovery_calls
    baseline_recovery_enabled = coord._recovery_enabled
    baseline_agent_configs = coord._snapshot_agent_configs()
    polluted_checkpoint = coord.recorder.get_checkpoint(polluted_checkpoint_id)
    if polluted_checkpoint is None:
        raise RuntimeError(f"Missing polluted checkpoint: {polluted_checkpoint_id}")

    outcomes: list[ProgramOutcome] = []
    semantic_executor = SemanticProgramExecutor()
    for program in programs:
        coord.recorder.checkpoint_store.restore(polluted_checkpoint, coord.workspace)
        coord.shared_facts = copy.deepcopy(baseline_shared_facts)
        coord.stage_outputs = copy.deepcopy(baseline_stage_outputs)
        coord.stage_nodes = copy.deepcopy(baseline_stage_nodes)
        coord.recorder.graph = copy.deepcopy(baseline_graph)
        coord.recorder._checkpoints = copy.deepcopy(baseline_checkpoints)
        coord.recovery_calls = baseline_recovery_calls
        coord._recovery_enabled = False
        coord._restore_agent_configs(baseline_agent_configs)

        overall_start = time.time()
        usage_before = coord._usage_snapshot()
        if _is_semantic_program(program):
            raw_outcome = semantic_executor.execute_semantic(
                coord,
                program,
                failed_state=failed_state,
                budget=coord.config.budget,
                checkpoint_ids={
                    "initial": _checkpoint_id_for_label(failed_state, label="initial"),
                    "post_locate": _checkpoint_id_for_label(failed_state, label="post_locate"),
                    "post_patch": _checkpoint_id_for_label(failed_state, label="post_patch"),
                },
                closed_loop=str(manifest.get("_program_space_version", "")).strip().lower()
                in {"semantic_closed_loop_v1", "semantic_action_loop_v1"},
                official_eval_commands={
                    "test_command": str(manifest.get("test_command", "") or ""),
                    "oracle_command": str(manifest.get("oracle_command", "") or ""),
                },
            )
        else:
            raw_outcome = coord.program_executor.execute(coord, program)
        fail_to_pass = coord.executor.execute(str(manifest["test_command"]), cwd=str(coord.workspace), timeout=1800)
        oracle = coord.executor.execute(str(manifest["oracle_command"]), cwd=str(coord.workspace), timeout=1800)
        patch_legality = _patch_legality_diagnostic(coord, failed_state)
        usage_after = coord._usage_snapshot()
        actual_success = oracle["returncode"] == 0
        oracle_infra_error = _is_oracle_infra_error(
            int(oracle["returncode"]),
            str(oracle.get("output", "")),
        ) or _is_oracle_infra_error(
            int(fail_to_pass["returncode"]),
            str(fail_to_pass.get("output", "")),
        )
        legal_success = _is_legal_recovery_success(
            actual_success=actual_success,
            patch_legality=patch_legality,
            oracle_infra_error=oracle_infra_error,
        )
        evaluation_status = _evaluation_status(
            actual_success=actual_success,
            oracle_infra_error=oracle_infra_error,
        )
        success_legitimacy = _success_legitimacy_tier(
            actual_success=actual_success,
            patch_legality=patch_legality,
        )
        guarded_recover_success = _guarded_recover_success_from_legitimacy(success_legitimacy)
        raw_metadata = dict(raw_outcome.metadata)
        raw_ledger_after = dict(raw_metadata.get("ledger_after", {}) or {})
        raw_ledger_metadata = dict(raw_ledger_after.get("metadata", {}) or {})
        if raw_ledger_metadata.get("car_enabled"):
            ledger_after = RecoveryLedger.from_dict(raw_ledger_after)
            finalize_car_episodes(
                ledger_after,
                cell_succeeded=guarded_recover_success,
                oracle_clean=not oracle_infra_error,
                infra_clean=not bool(raw_ledger_metadata.get("substrate_error", False)),
                provider_clean=not bool(raw_ledger_metadata.get("provider_error_observed", False)),
                append_path=str(raw_ledger_metadata.get("car_output_episodes_path", "") or "") or None,
            )
            raw_metadata["ledger_after"] = ledger_after.to_dict()
            raw_outcome.metadata = raw_metadata
        replay_patcher_trace = _selected_replay_patcher_trace(raw_outcome)
        failure_mode = _derive_failure_mode(
            actual_success=actual_success,
            legal_success=legal_success,
            patch_legality=patch_legality,
            patcher_trace=replay_patcher_trace,
            oracle_infra_error=oracle_infra_error,
        )
        outcomes.append(
            ProgramOutcome(
                program_id=raw_outcome.program_id,
                recover_success=legal_success,
                official_resolved=legal_success,
                token_cost=_usage_delta(usage_before, usage_after),
                latency_sec=max(0.001, time.time() - overall_start),
                secondary_risk=raw_outcome.secondary_risk,
                milestone_gain=raw_outcome.milestone_gain,
                step_outcomes=raw_outcome.step_outcomes,
                notes=raw_outcome.notes,
                metadata={
                    **raw_metadata,
                    "internal_guarded_recover_success": bool(
                        raw_metadata.get("guarded_recover_success", False)
                    ),
                    "guarded_recover_success": guarded_recover_success,
                    "scope": _program_scope(program),
                    "family": _program_family(program),
                    "strategy": _program_strategy(program),
                    "fail_to_pass_returncode": fail_to_pass["returncode"],
                    "oracle_returncode": oracle["returncode"],
                    "oracle_returncode_success": actual_success,
                    "oracle_infra_error": oracle_infra_error,
                    "evaluation_status": evaluation_status,
                    "fail_to_pass_output": fail_to_pass["output"][:4000],
                    "oracle_output": oracle["output"][:4000],
                    "patch_legality": patch_legality,
                    "legal_success": legal_success,
                    "success_legitimacy": success_legitimacy,
                    "failure_mode": failure_mode,
                },
            )
        )

    coord.recorder.checkpoint_store.restore(polluted_checkpoint, coord.workspace)
    coord.shared_facts = baseline_shared_facts
    coord.stage_outputs = baseline_stage_outputs
    coord.stage_nodes = baseline_stage_nodes
    coord.recorder.graph = baseline_graph
    coord.recorder._checkpoints = baseline_checkpoints
    coord.recovery_calls = baseline_recovery_calls
    coord._recovery_enabled = baseline_recovery_enabled
    coord._restore_agent_configs(baseline_agent_configs)
    return outcomes


def _summarize_case(
    *,
    coord,
    manifest: dict[str, Any],
    failed_state: FailedState,
    programs: list[RecoveryProgram],
    outcomes: list[ProgramOutcome],
) -> dict[str, Any]:
    budget = coord.config.budget
    best_program, best_outcome, selection_mode = _best_program_for_pool(programs, outcomes, budget)
    program_by_id = {program.program_id: program for program in programs}
    outcome_by_id = {outcome.program_id: outcome for outcome in outcomes}
    synthesis_record = dict(coord.stage_outputs.get("atomic_program_synthesis", {}) or {})
    utility_program = max(programs, key=lambda program: _actual_utility(outcome_by_id[program.program_id], budget))
    utility_outcome = outcome_by_id[utility_program.program_id]
    best_patch_legality = dict(best_outcome.metadata.get("patch_legality", {}) or {})
    rollback_anchor_label = _selected_rollback_anchor_label(
        failed_state=failed_state,
        program=best_program,
        outcome=best_outcome,
    )
    replay_resume_stage = _selected_replay_resume_stage(best_program, best_outcome)
    replay_patcher_trace = _selected_replay_patcher_trace(best_outcome)
    selected_success_under_budget = bool(
        best_outcome.recover_success
        and float(best_outcome.token_cost) <= float(budget.token_budget)
        and float(best_outcome.latency_sec) <= float(budget.latency_budget_sec)
    )

    family_best: dict[str, dict[str, Any]] = {}
    family_mode: dict[str, str] = {}
    for family in sorted({_program_family(program) for program in programs}):
        family_programs = [program for program in programs if _program_family(program) == family]
        family_program, family_outcome, family_selection_mode = _best_program_for_pool(family_programs, outcomes, budget)
        utility = _actual_utility(family_outcome, budget)
        family_best[family] = {
            "program": _program_to_dict(family_program),
            "outcome": family_outcome.to_dict(),
            "utility": utility,
            "selection_mode": family_selection_mode,
        }
        family_mode[family] = family_selection_mode

    all_family_outcomes: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        program = program_by_id[outcome.program_id]
        family = _program_family(program)
        all_family_outcomes.setdefault(family, []).append(
            {
                "program": _program_to_dict(program),
                "outcome": outcome.to_dict(),
                "utility": _actual_utility(outcome, budget),
            }
        )

    global_candidates = [
        item
        for family, item in family_best.items()
        if _is_global_family(family)
    ]
    global_best = max(global_candidates, key=lambda item: item["utility"]) if global_candidates else None
    local_candidates = [
        item for family, item in family_best.items()
        if not _is_global_family(family)
    ]
    local_best = max(local_candidates, key=lambda item: item["utility"]) if local_candidates else None

    row = {
        "instance_id": manifest["instance_id"],
        "profile": manifest.get("_benchmark_profile", "unknown"),
        "program_space_version": manifest.get("_program_space_version", "legacy"),
        "diagnosis_mode": manifest.get("_diagnosis_mode", "none"),
        "execute_candidate_limit": manifest.get("_execute_candidate_limit", 0),
        "program_id_filter": list(manifest.get("_program_id_filter", []) or []),
        "program_online_selector": manifest.get("_program_online_selector", "none"),
        "online_selection_mode": manifest.get("_program_online_selector", "none"),
        "replay_verifier_mode": manifest.get("_replay_verifier_mode", "live"),
        "workspace_start_snapshot": manifest.get("_workspace_start_snapshot", "unknown"),
        "source_type": "controlled_perturbation",
        "fault_type": failed_state.metadata.get("fault_type", "unknown"),
        "run_id": coord.run_id,
        "oracle_selection_mode": selection_mode,
        "trigger": failed_state.trigger.to_dict(),
        "suspect_region": failed_state.suspect_region.to_dict(),
        "failure_observation": dict(failed_state.metadata.get("failure_observation", {})),
        "failed_state_metadata": {
            "checkpoint_depth": failed_state.metadata.get("checkpoint_depth"),
            "recovery_invocations": failed_state.metadata.get("recovery_invocations"),
            "checkpoint_candidates": failed_state.metadata.get("checkpoint_candidates", []),
            "latest_test_status": failed_state.metadata.get("latest_test_status", {}),
            "phase_outputs": failed_state.metadata.get("phase_outputs", {}),
            "fault_type": failed_state.metadata.get("fault_type"),
            "touched_paths": failed_state.metadata.get("touched_paths", []),
            "wrong_localized_path": failed_state.metadata.get("wrong_localized_path"),
            "correct_paths": failed_state.metadata.get("correct_paths", []),
            "partial_mode": failed_state.metadata.get("partial_mode"),
            "anchor_health": failed_state.metadata.get("anchor_health"),
        },
        "selected_program_id": best_program.program_id,
        "selected_strategy": _program_strategy(best_program),
        "selected_program_strategy": _program_strategy(best_program),
        "selected_family": _program_family(best_program),
        "selected_success": bool(best_outcome.recover_success),
        "selected_success_under_budget": selected_success_under_budget,
        "selected_official_resolved": bool(best_outcome.official_resolved),
        "selected_token_cost": float(best_outcome.token_cost),
        "selected_latency_sec": float(best_outcome.latency_sec),
        "selected_utility": _actual_utility(best_outcome, budget),
        "rollback_anchor_label": rollback_anchor_label,
        "replay_resume_stage": replay_resume_stage,
        "replay_patcher_trace": replay_patcher_trace,
        "replay_patcher_command_count": replay_patcher_trace.get("patcher_command_count"),
        "replay_patcher_write_command_count": replay_patcher_trace.get("patcher_write_command_count"),
        "replay_patcher_validation_command_count": replay_patcher_trace.get("patcher_validation_command_count"),
        "replay_patcher_readonly_command_count": replay_patcher_trace.get("patcher_readonly_command_count"),
        "replay_patcher_patch_present": bool(replay_patcher_trace.get("patcher_patch_present", False)),
        "patch_scope": best_patch_legality.get("patch_scope"),
        "changed_files": best_patch_legality.get("changed_files", []),
        "changed_file_classes": best_patch_legality.get("changed_file_classes", {}),
        "touches_suspect_path": bool(best_patch_legality.get("touches_suspect_path", False)),
        "failure_mode": best_outcome.metadata.get("failure_mode"),
        "success_legitimacy": best_outcome.metadata.get("success_legitimacy"),
        "evaluation_status": best_outcome.metadata.get("evaluation_status"),
        "oracle_infra_error": bool(best_outcome.metadata.get("oracle_infra_error", False)),
        "replay_model_infra_error": bool(replay_patcher_trace.get("patcher_infrastructure_error", False)),
        "fail_to_pass_returncode": best_outcome.metadata.get("fail_to_pass_returncode"),
        "oracle_returncode": best_outcome.metadata.get("oracle_returncode"),
        "executed_program_ids": [program.program_id for program in programs],
        "llm_recommended_program_id": synthesis_record.get("recommended_program_id", ""),
        "raw_llm_candidate_programs": synthesis_record.get("raw_llm_candidate_programs", []),
        "validated_program_ids": synthesis_record.get("validated_program_ids", []),
        "backfilled_program_ids": synthesis_record.get("backfilled_program_ids", []),
        "returned_program_ids": synthesis_record.get("returned_program_ids", []),
        "returned_candidate_programs": synthesis_record.get("returned_candidate_programs", []),
        "synthesis_diagnosis": synthesis_record.get("diagnosis", ""),
        "synthesis_raw_response_excerpt": synthesis_record.get("raw_response_excerpt", ""),
        "synthesizer_model": synthesis_record.get("synthesizer_model", ""),
        "semantic_execution_mode": best_outcome.metadata.get("execution_mode", "legacy"),
        "semantic_guard_mode": best_outcome.metadata.get("guard_mode", ""),
        "selected_execution_mode": best_outcome.metadata.get("execution_mode", "legacy"),
        "selected_guard_mode": best_outcome.metadata.get("guard_mode", ""),
        "selected_loop_unit": best_outcome.metadata.get("loop_unit", ""),
        "selected_guard_summary": dict(best_outcome.metadata.get("guard_summary", {}) or {}),
        "selected_guarded_recover_success": bool(best_outcome.metadata.get("guarded_recover_success", False)),
        "selected_budget_summary": dict(best_outcome.metadata.get("budget_summary", {}) or {}),
        "selected_semantic_program": dict(best_outcome.metadata.get("semantic_program", {}) or {}),
        "selected_compiled_primitive_program": dict(best_outcome.metadata.get("compiled_primitive_program", {}) or {}),
        "selected_ledger_before": dict(best_outcome.metadata.get("ledger_before", {}) or {}),
        "selected_ledger_after": dict(best_outcome.metadata.get("ledger_after", {}) or {}),
        "selected_semantic_trace": list(best_outcome.metadata.get("semantic_trace", []) or []),
        "selected_closed_loop_summary": dict(best_outcome.metadata.get("closed_loop_summary", {}) or {}),
        "selected_online_selector_debug": dict(best_program.metadata.get("online_selector_debug", {}) or {}),
        "program_online_selector_rankings": [
            {
                "program_id": str(program.program_id),
                "strategy": str(program.metadata.get("strategy", "")),
                **dict(program.metadata.get("online_selector_debug", {}) or {}),
            }
            for program in programs
            if dict(program.metadata.get("online_selector_debug", {}) or {})
        ],
        "programs": [_program_to_dict(program) for program in programs],
        "outcomes": [outcome.to_dict() for outcome in outcomes],
        "best_overall": {
            "program": _program_to_dict(best_program),
            "outcome": best_outcome.to_dict(),
            "utility": _actual_utility(best_outcome, budget),
            "selection_mode": selection_mode,
        },
        "best_utility_only": {
            "program": _program_to_dict(utility_program),
            "outcome": utility_outcome.to_dict(),
            "utility": _actual_utility(utility_outcome, budget),
        },
        "best_local": local_best,
        "best_global": global_best,
        "family_outcomes": all_family_outcomes,
        "local_beats_global": bool(
            local_best and (
                global_best is None
                or (
                    local_best["outcome"]["recover_success"]
                    and not global_best["outcome"]["recover_success"]
                )
                or (
                    local_best["outcome"]["recover_success"]
                    and global_best["outcome"]["recover_success"]
                    and (
                        float(local_best["outcome"]["token_cost"]),
                        float(local_best["outcome"]["latency_sec"]),
                        float(local_best["outcome"]["secondary_risk"]),
                    )
                    < (
                        float(global_best["outcome"]["token_cost"]),
                        float(global_best["outcome"]["latency_sec"]),
                        float(global_best["outcome"]["secondary_risk"]),
                    )
                )
                or (
                    not local_best["outcome"]["recover_success"]
                    and not global_best["outcome"]["recover_success"]
                    and local_best["utility"] > global_best["utility"]
                )
            )
        ),
        "local_success_beats_global": bool(
            local_best and local_best["outcome"]["recover_success"] and (
                global_best is None
                or not global_best["outcome"]["recover_success"]
                or local_best["outcome"]["token_cost"] < global_best["outcome"]["token_cost"]
            )
        ),
    }
    row.update(
        benchmark_row_governance_fields(
            source_type="controlled_perturbation",
            runtime=str(manifest.get("_resolved_runtime", "") or ""),
            related_payloads=[row],
        )
    )
    return row


def _summarize_fault_validation_case(
    *,
    coord,
    manifest: dict[str, Any],
    failed_state: FailedState,
) -> dict[str, Any]:
    row = {
        "instance_id": manifest["instance_id"],
        "profile": manifest.get("_benchmark_profile", "unknown"),
        "program_space_version": manifest.get("_program_space_version", "legacy"),
        "diagnosis_mode": manifest.get("_diagnosis_mode", "none"),
        "program_online_selector": manifest.get("_program_online_selector", "none"),
        "source_type": "controlled_perturbation",
        "fault_type": failed_state.metadata.get("fault_type", "unknown"),
        "run_id": coord.run_id,
        "fault_only": True,
        "trigger": failed_state.trigger.to_dict(),
        "suspect_region": failed_state.suspect_region.to_dict(),
        "failure_observation": dict(failed_state.metadata.get("failure_observation", {})),
        "failed_state_metadata": {
            "checkpoint_depth": failed_state.metadata.get("checkpoint_depth"),
            "recovery_invocations": failed_state.metadata.get("recovery_invocations"),
            "checkpoint_candidates": failed_state.metadata.get("checkpoint_candidates", []),
            "latest_test_status": failed_state.metadata.get("latest_test_status", {}),
            "phase_outputs": failed_state.metadata.get("phase_outputs", {}),
            "fault_type": failed_state.metadata.get("fault_type"),
            "touched_paths": failed_state.metadata.get("touched_paths", []),
            "wrong_localized_path": failed_state.metadata.get("wrong_localized_path"),
            "correct_paths": failed_state.metadata.get("correct_paths", []),
            "partial_mode": failed_state.metadata.get("partial_mode"),
            "anchor_health": failed_state.metadata.get("anchor_health"),
        },
        "programs": [],
        "outcomes": [],
        "fault_validation": {
            "official_test_failed": True,
            "has_trigger": failed_state.trigger is not None,
            "has_suspect_region": failed_state.suspect_region is not None,
            "has_checkpoint_id": bool(failed_state.checkpoint_id),
            "has_verifier_excerpt": bool(
                dict(failed_state.metadata.get("failure_observation", {})).get("verifier_output_excerpt")
            ),
        },
    }
    row.update(
        benchmark_row_governance_fields(
            source_type="controlled_perturbation",
            runtime=str(manifest.get("_resolved_runtime", "") or ""),
            related_payloads=[row],
        )
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("error")]
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_valid": len(valid_rows),
        "n_invalid_perturbation": 0,
        "best_strategy_counts": {},
        "family_metrics": {},
        "local_beats_global_count": 0,
        "local_success_beats_global_count": 0,
        "profile_counts": {},
        "program_space_version_counts": {},
        "diagnosis_mode_counts": {},
        "program_online_selector_counts": {},
        "error_type_counts": {},
        "failure_mode_counts": {},
        "patch_scope_counts": {},
        "success_legitimacy_counts": {},
        "evaluation_status_counts": {},
        "oracle_infra_error_count": 0,
        "selected_success_count": 0,
        "selected_guarded_recover_success_count": 0,
        "selected_guarded_success_count": 0,
        "selected_raw_success_not_guarded_count": 0,
        "semantic_guard_result_mode_counts": {},
        "semantic_guard_pending_official_count": 0,
        "recommendation_respected_count": 0,
        "recommendation_overridden_count": 0,
        "touches_suspect_path_count": 0,
        "source_edit_count": 0,
        "test_edit_count": 0,
        "replay_no_write_command_count": 0,
        "replay_patch_present_count": 0,
    }
    error_type_counts: dict[str, int] = {}
    for row in rows:
        if row.get("error"):
            error_type = str(row.get("error_type", "runtime_error"))
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    summary["error_type_counts"] = error_type_counts
    summary["n_invalid_perturbation"] = error_type_counts.get("invalid_perturbation", 0)
    if not valid_rows:
        return summary

    strategy_counts: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    version_counts: dict[str, int] = {}
    diagnosis_mode_counts: dict[str, int] = {}
    program_online_selector_counts: dict[str, int] = {}
    failure_mode_counts: dict[str, int] = {}
    patch_scope_counts: dict[str, int] = {}
    success_legitimacy_counts: dict[str, int] = {}
    evaluation_status_counts: dict[str, int] = {}
    semantic_guard_result_mode_counts: dict[str, int] = {}
    family_buckets: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        profile = str(row.get("profile", "unknown"))
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        version = str(row.get("program_space_version", "legacy"))
        version_counts[version] = version_counts.get(version, 0) + 1
        diagnosis_mode = str(row.get("diagnosis_mode", "none") or "none")
        diagnosis_mode_counts[diagnosis_mode] = diagnosis_mode_counts.get(diagnosis_mode, 0) + 1
        program_online_selector = str(row.get("program_online_selector", "heuristic_rerank") or "heuristic_rerank")
        program_online_selector_counts[program_online_selector] = (
            program_online_selector_counts.get(program_online_selector, 0) + 1
        )
        failure_mode = str(row.get("failure_mode", "") or "")
        if failure_mode:
            failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1
        patch_scope = str(row.get("patch_scope", "") or "")
        if patch_scope:
            patch_scope_counts[patch_scope] = patch_scope_counts.get(patch_scope, 0) + 1
        success_legitimacy = str(row.get("success_legitimacy", "") or "")
        if success_legitimacy:
            success_legitimacy_counts[success_legitimacy] = (
                success_legitimacy_counts.get(success_legitimacy, 0) + 1
            )
        evaluation_status = str(row.get("evaluation_status", "") or "")
        if evaluation_status:
            evaluation_status_counts[evaluation_status] = evaluation_status_counts.get(evaluation_status, 0) + 1
        if row.get("oracle_infra_error"):
            summary["oracle_infra_error_count"] += 1
        if row.get("selected_success"):
            summary["selected_success_count"] += 1
        if row.get("selected_guarded_recover_success"):
            summary["selected_guarded_recover_success_count"] += 1
            summary["selected_guarded_success_count"] += 1
        if row.get("selected_success") and not row.get("selected_guarded_recover_success"):
            summary["selected_raw_success_not_guarded_count"] += 1
        guard_summary = dict(row.get("selected_guard_summary", {}) or {})
        guard_result_mode_counts = dict(guard_summary.get("result_mode_counts", {}) or {})
        for mode, count in guard_result_mode_counts.items():
            normalized_mode = str(mode)
            semantic_guard_result_mode_counts[normalized_mode] = (
                semantic_guard_result_mode_counts.get(normalized_mode, 0) + int(count or 0)
            )
        summary["semantic_guard_pending_official_count"] += int(
            guard_summary.get("pending_official_count", 0) or 0
        )
        recommendation = str(row.get("llm_recommended_program_id", "") or "").strip()
        executed_ids = row.get("executed_program_ids", []) or []
        executed = str(executed_ids[0]).strip() if executed_ids else ""
        if recommendation:
            if recommendation == executed:
                summary["recommendation_respected_count"] += 1
            else:
                summary["recommendation_overridden_count"] += 1
        if row.get("touches_suspect_path"):
            summary["touches_suspect_path_count"] += 1
        changed_file_classes = row.get("changed_file_classes", {}) or {}
        if changed_file_classes.get("source_files"):
            summary["source_edit_count"] += 1
        if changed_file_classes.get("test_files"):
            summary["test_edit_count"] += 1
        write_count = row.get("replay_patcher_write_command_count")
        if write_count is not None and int(write_count or 0) == 0:
            summary["replay_no_write_command_count"] += 1
        if row.get("replay_patcher_patch_present"):
            summary["replay_patch_present_count"] += 1
        programs = [_program_from_payload(item) for item in row.get("programs", [])]
        outcomes = [ProgramOutcome.from_dict(item) for item in row.get("outcomes", [])]
        best_overall = dict(row.get("best_overall") or {})
        best_program_payload = dict(best_overall.get("program") or {})
        family_outcomes = row.get("family_outcomes") or {}
        if best_program_payload:
            strategy = str(
                best_program_payload.get("metadata", {}).get(
                    "strategy",
                    best_program_payload.get("program_id", ""),
                )
            )
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            row_local_beats_global = bool(row.get("local_beats_global"))
            row_local_success_beats_global = bool(row.get("local_success_beats_global"))
        else:
            if programs and outcomes and len(programs) == len(outcomes):
                best_program, _, _ = _best_program_for_pool(programs, outcomes, RecoveryBudget())
                strategy = str(best_program.metadata.get("strategy", best_program.program_id))
            else:
                strategy = "unknown"
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            row_local_beats_global = bool(row.get("local_beats_global"))
            row_local_success_beats_global = bool(row.get("local_success_beats_global"))

        if row_local_beats_global:
            summary["local_beats_global_count"] += 1
        if row_local_success_beats_global:
            summary["local_success_beats_global_count"] += 1
        if family_outcomes:
            for family, items in family_outcomes.items():
                family_buckets.setdefault(str(family), []).extend(items)
        elif programs and outcomes and len(programs) == len(outcomes):
            for program, outcome in zip(programs, outcomes):
                family = _program_family(program)
                family_buckets.setdefault(family, []).append(
                    {
                        "program": program.to_dict(),
                        "outcome": outcome.to_dict(),
                        "utility": _actual_utility(outcome, RecoveryBudget()),
                    }
                )

    summary["best_strategy_counts"] = strategy_counts
    summary["profile_counts"] = profile_counts
    summary["program_space_version_counts"] = version_counts
    summary["diagnosis_mode_counts"] = diagnosis_mode_counts
    summary["program_online_selector_counts"] = program_online_selector_counts
    summary["failure_mode_counts"] = failure_mode_counts
    summary["patch_scope_counts"] = patch_scope_counts
    summary["success_legitimacy_counts"] = success_legitimacy_counts
    summary["evaluation_status_counts"] = evaluation_status_counts
    summary["semantic_guard_result_mode_counts"] = semantic_guard_result_mode_counts
    summary["selected_guarded_success_count"] = summary["selected_guarded_recover_success_count"]
    for family, items in family_buckets.items():
        successes = [item for item in items if item["outcome"]["recover_success"]]
        summary["family_metrics"][family] = {
            "n": len(items),
            "success_rate": sum(1 for item in items if item["outcome"]["recover_success"]) / float(len(items)),
            "avg_token_cost": sum(float(item["outcome"]["token_cost"]) for item in items) / float(len(items)),
            "avg_latency_sec": sum(float(item["outcome"]["latency_sec"]) for item in items) / float(len(items)),
            "avg_token_cost_success_only": (
                sum(float(item["outcome"]["token_cost"]) for item in successes) / float(len(successes))
                if successes else None
            ),
        }
    summary.update(summarize_governance_rows(rows))
    return summary


def _write_payload_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("error") and not row.get("error_type"):
                row["error_type"] = _classify_error(str(row.get("error", "")))
        return rows
    return []


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("instance_id") or "").strip(),
        str(row.get("fault_type") or "").strip(),
        str(row.get("profile") or "").strip(),
        str(row.get("program_space_version") or "").strip(),
        str(row.get("diagnosis_mode") or "none").strip(),
        str(row.get("program_online_selector") or "heuristic_rerank").strip(),
    )


def _classify_error(error: str) -> str:
    lowered = str(error).lower()
    if "unexpectedly passed verifier" in lowered or "invalid perturbation" in lowered:
        return "invalid_perturbation"
    if "manual_fault_spec_not_found" in lowered:
        return "manual_fault_spec_not_found"
    if "manifest_not_found" in lowered:
        return "manifest_not_found"
    if "no_patch_paths" in lowered:
        return "no_patch_paths"
    return "runtime_error"


def _upsert_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    row_key = _row_identity(new_row)
    if not row_key[0]:
        rows.append(new_row)
        return rows
    for idx, row in enumerate(rows):
        if _row_identity(row) == row_key:
            rows[idx] = new_row
            return rows
    rows.append(new_row)
    return rows


def _is_terminal_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if not _row_identity(row)[0]:
        return False
    if not row.get("error"):
        return True
    error_type = str(row.get("error_type", ""))
    return error_type in {"invalid_perturbation", "manual_fault_spec_not_found", "manifest_not_found", "no_patch_paths"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recovery-only BCMR benchmark from polluted late-stage states.")
    parser.add_argument("--api-path", type=Path, default=Path(__file__).resolve().parents[2] / "api.md")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--strong-model", type=str, default=None)
    parser.add_argument("--strong-stages", type=str, default="planner,implementer")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--manifest-root", type=Path, default=DATA_ROOT / "artifacts")
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--workspace-root", type=Path, default=OUTPUT_ROOT / "recovery_benchmark_workspaces")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "recovery_only_benchmark.json")
    parser.add_argument("--runtime", type=str, default="auto", choices=["auto", "local", "harness"])
    parser.add_argument("--force-rebuild-harness", action="store_true")
    parser.add_argument(
        "--harness-setup-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for official harness repository initialization inside the container.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full_matrix",
        choices=["anchor_restore", "local_rebuild", "full_matrix"],
        help="Experiment profile: anchor_restore tests minimal rollback, local_rebuild tests local replay without post_patch restore, full_matrix runs both against global restart.",
    )
    parser.add_argument(
        "--program-space-version",
        type=str,
        default="semantic_dual_v1",
        choices=[
            "legacy",
            "intent_discovery_v1",
            "intent_discovery_v2",
            "intent_schema_v1",
            "intent_schema_v2",
            "intent_schema_v3",
            "rollback_only_v1",
            "atomic_actions_v1",
            "semantic_dual_v1",
            "semantic_closed_loop_v1",
            "semantic_action_loop_v1",
            "semantic_object_loop_v2",
        ],
        help="Candidate program version. recovery-only is a diagnostic line; the frozen baseline default is semantic_dual_v1, while semantic_action_loop_v1 and semantic_object_loop_v2 remain diagnosis-only alternatives.",
    )
    parser.add_argument(
        "--fault-family",
        type=str,
        default="post_patch_regression",
        choices=["post_patch_regression", "contaminated_post_patch", "localization_pollution", "partial_patch"],
        help="Controlled fault family. New starter families use manual specs from --manual-fault-specs.",
    )
    parser.add_argument(
        "--manual-fault-specs",
        type=Path,
        default=DEFAULT_MANUAL_FAULT_SPECS,
        help="Manual expert fault specs for localization_pollution and partial_patch starter cases.",
    )
    parser.add_argument("--locator-iters", type=int, default=12)
    parser.add_argument("--planner-iters", type=int, default=8)
    parser.add_argument("--patcher-iters", type=int, default=14)
    parser.add_argument("--verifier-iters", type=int, default=12)
    parser.add_argument("--max-candidate-programs", type=int, default=5)
    parser.add_argument(
        "--execute-candidate-limit",
        type=int,
        default=0,
        help="Online execution cap for candidate programs. 0 executes the full generated pool for offline oracle scan; 1 executes the top-1 runtime.",
    )
    parser.add_argument(
        "--program-online-selector",
        type=str,
        default="heuristic_rerank",
        choices=[
            "heuristic_rerank",
            "llm_recommended_top1",
            "raw_order_top1",
            "semantic_action_state_top1",
            "parc_lifecycle_frontier_top1",
            "car_counterexample_top1",
            "semantic_object_state_top1",
        ],
        help="How to choose the online top-k candidate programs after synthesis. semantic_action_state_top1 and parc_lifecycle_frontier_top1 are for action-level starters; semantic_object_state_top1 is for object-level macro actions.",
    )
    parser.add_argument(
        "--program-id-filter",
        nargs="+",
        default=[],
        help="Discovery-only filter for executing specific program IDs, used for action coverage probes without running the whole candidate pool.",
    )
    parser.add_argument(
        "--replay-verifier-mode",
        type=str,
        default="live",
        choices=["live", "official_only"],
        help="live replays the internal LLM verifier; official_only skips it and relies on official test/oracle commands after replay.",
    )
    parser.add_argument(
        "--diagnosis-mode",
        type=str,
        default="none",
        choices=["none", "contaminated_post_patch_v1"],
        help="Optional diagnosis-only execution mode. contaminated_post_patch_v1 freezes the contaminated family to two diagnostic recovery programs and emits richer failure diagnostics.",
    )
    parser.add_argument(
        "--seed-case-memory-dir",
        type=Path,
        default=None,
        help="Optional frozen case-memory bank directory. When set, atomic action synthesis retrieves similar cases from this bank.",
    )
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument(
        "--fault-only",
        action="store_true",
        help="Only construct and validate the failed state; skip recovery program execution.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output json if present.")
    args = parser.parse_args()

    os.environ.setdefault("SWE_MAS_MODEL_RETRY_ATTEMPTS", "1")

    resolved_model, resolved_strong_model = resolve_model_names(
        args.api_path,
        model_name=args.model,
        strong_model_name=args.strong_model,
    )
    if not args.skip_model_preflight:
        base_probe = run_model_preflight(
            args.api_path,
            model_name=resolved_model,
            label="base_model",
            request_timeout=min(args.request_timeout, 20),
        )
        print(f"[preflight] base model ok: {base_probe['model']} -> {base_probe['content']}", flush=True)
        if resolved_strong_model != resolved_model:
            strong_probe = run_model_preflight(
                args.api_path,
                model_name=resolved_strong_model,
                label="strong_model",
                request_timeout=min(args.request_timeout, 20),
            )
            print(f"[preflight] strong model ok: {strong_probe['model']} -> {strong_probe['content']}", flush=True)

    model = build_gemini_model(
        args.api_path,
        model_name=resolved_model,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    strong_model = None
    if resolved_strong_model:
        strong_model = build_gemini_model(
            args.api_path,
            model_name=resolved_strong_model,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )

    catalog = load_manifest_catalog(args.manifest_root)
    manual_fault_specs = _load_manual_fault_specs(args.manual_fault_specs)
    rows: list[dict[str, Any]] = _load_existing_rows(args.output) if args.resume else []
    current_row_prefix = {
        "fault_type": args.fault_family,
        "profile": args.profile,
        "program_space_version": args.program_space_version,
        "diagnosis_mode": args.diagnosis_mode,
        "program_online_selector": args.program_online_selector,
    }
    processed_row_keys = {
        _row_identity(row)
        for row in rows
        if _is_terminal_row(row)
    }

    for instance_id in args.instance_ids:
        current_row_key = (
            instance_id,
            current_row_prefix["fault_type"],
            current_row_prefix["profile"],
            current_row_prefix["program_space_version"],
            current_row_prefix["diagnosis_mode"],
            current_row_prefix["program_online_selector"],
        )
        if args.resume and current_row_key in processed_row_keys:
            print(f"[resume] skipping completed instance: {instance_id}")
            continue
        print(f"[instance] starting {instance_id}")
        entry = select_manifest_entry(catalog, instance_id=instance_id, fault="none")
        if entry is None:
            _upsert_row(
                rows,
                {
                    "instance_id": instance_id,
                    **current_row_prefix,
                    "error": "manifest_not_found",
                    "error_type": "manifest_not_found",
                },
            )
            payload = {
                "rows": rows,
                "summary": _aggregate(rows),
            }
            _write_payload_atomic(args.output, payload)
            print(f"[instance] {instance_id} -> error: manifest_not_found")
            continue
        manifest = entry["payload"]
        manifest = dict(manifest)
        manifest["_benchmark_profile"] = args.profile
        manifest["_program_space_version"] = args.program_space_version
        manifest["_execute_candidate_limit"] = args.execute_candidate_limit
        manifest["_replay_verifier_mode"] = args.replay_verifier_mode
        manifest["_workspace_start_snapshot"] = "source_snapshot"
        manifest["_program_online_selector"] = args.program_online_selector
        manifest["_diagnosis_mode"] = args.diagnosis_mode
        manifest["_program_id_filter"] = list(args.program_id_filter or [])
        manual_spec = _manual_injection_spec(
            manual_fault_specs,
            fault_family=args.fault_family,
            instance_id=instance_id,
        )
        if manual_spec:
            manifest["_manual_injection_spec"] = manual_spec
        suspicious_targets = _suspicious_manifest_test_targets(manifest)
        if suspicious_targets:
            _upsert_row(
                rows,
                {
                    "instance_id": instance_id,
                    **current_row_prefix,
                    "error": "invalid_manifest_targets",
                    "error_type": "invalid_manifest_targets",
                    "suspicious_test_targets": suspicious_targets,
                    "manifest_test_command": str(manifest.get("test_command", "") or ""),
                    "manifest_oracle_command": str(manifest.get("oracle_command", "") or ""),
                },
            )
            payload = {
                "rows": rows,
                "summary": _aggregate(rows),
            }
            _write_payload_atomic(args.output, payload)
            print(f"[instance] {instance_id} -> error: invalid_manifest_targets")
            continue
        source_snapshot = Path(str(manifest["source_snapshot"])).resolve()
        oracle_snapshot = Path(str(manifest.get("oracle_snapshot") or (str(manifest["source_snapshot"]) + "__oracle"))).resolve()
        patch_paths = _patch_paths(manifest)
        if not patch_paths:
            _upsert_row(
                rows,
                {
                    "instance_id": instance_id,
                    **current_row_prefix,
                    "error": "no_patch_paths",
                    "error_type": "no_patch_paths",
                },
            )
            payload = {
                "rows": rows,
                "summary": _aggregate(rows),
            }
            _write_payload_atomic(args.output, payload)
            print(f"[instance] {instance_id} -> error: no_patch_paths")
            continue

        resolved_runtime = resolve_runtime(args.runtime, manifest)
        workspace_strategy = workspace_strategy_for_runtime(args.runtime, manifest)
        workspace = materialize_workspace(
            source_snapshot,
            args.workspace_root,
            f"{instance_id}_recoverybench",
            strategy=workspace_strategy,
        )

        runtime_session = None
        try:
            executor, runtime_session = build_executor(
                workspace=str(workspace),
                runtime=args.runtime,
                manifest=manifest,
                force_rebuild_harness=args.force_rebuild_harness,
                harness_setup_timeout=args.harness_setup_timeout,
            )
            coordinator = build_coordinator(
                workspace=str(workspace),
                model=model,
                strong_model=strong_model,
                strong_stages=tuple(stage.strip() for stage in args.strong_stages.split(",") if stage.strip()),
                executor=executor,
                locator_max_iterations=args.locator_iters,
                planner_max_iterations=args.planner_iters,
                patcher_max_iterations=args.patcher_iters,
                verifier_max_iterations=args.verifier_iters,
            )
            coordinator._official_only_replay_verifier = args.replay_verifier_mode == "official_only"
            coordinator._recovery_diagnosis_mode = args.diagnosis_mode
            checkpoint_ids = _bootstrap_run_context(
                coordinator,
                manifest=manifest,
                workspace=workspace,
                executor=executor,
            )
            if args.seed_case_memory_dir is not None:
                seed_dir = args.seed_case_memory_dir.resolve()
                if not seed_dir.exists():
                    raise FileNotFoundError(f"seed case memory dir not found: {seed_dir}")
                seed_summary = _seed_case_memory_from_dir(
                    coordinator,
                    seed_dir=seed_dir,
                )
                coordinator.stage_outputs["seed_case_memory"] = seed_summary
            if args.fault_family == "post_patch_regression":
                _seed_healthy_anchor(
                    coordinator,
                    manifest=manifest,
                    source_snapshot=source_snapshot,
                    oracle_snapshot=oracle_snapshot,
                    patch_paths=patch_paths,
                    checkpoint_ids=checkpoint_ids,
                )
                failed_state = _inject_fault_and_build_failed_state(
                    coordinator,
                    manifest=manifest,
                    source_snapshot=source_snapshot,
                    workspace=workspace,
                    patch_paths=patch_paths,
                    checkpoint_ids=checkpoint_ids,
                )
            elif args.fault_family == "contaminated_post_patch":
                failed_state = _seed_contaminated_post_patch_and_build_failed_state(
                    coordinator,
                    manifest=manifest,
                    source_snapshot=source_snapshot,
                    oracle_snapshot=oracle_snapshot,
                    workspace=workspace,
                    patch_paths=patch_paths,
                    checkpoint_ids=checkpoint_ids,
                )
            elif args.fault_family == "localization_pollution":
                failed_state = _seed_localization_pollution_and_build_failed_state(
                    coordinator,
                    manifest=manifest,
                    source_snapshot=source_snapshot,
                    workspace=workspace,
                    patch_paths=patch_paths,
                    checkpoint_ids=checkpoint_ids,
                    manual_spec=manual_spec,
                )
            elif args.fault_family == "partial_patch":
                failed_state = _seed_partial_patch_and_build_failed_state(
                    coordinator,
                    manifest=manifest,
                    source_snapshot=source_snapshot,
                    oracle_snapshot=oracle_snapshot,
                    workspace=workspace,
                    patch_paths=patch_paths,
                    checkpoint_ids=checkpoint_ids,
                    manual_spec=manual_spec,
                )
            else:
                raise ValueError(f"Unsupported fault family: {args.fault_family}")
            if args.fault_only:
                _upsert_row(rows,
                    _summarize_fault_validation_case(
                        coord=coordinator,
                        manifest=manifest,
                        failed_state=failed_state,
                    )
                )
                payload = {
                    "rows": rows,
                    "summary": _aggregate(rows),
                }
                _write_payload_atomic(args.output, payload)
                print(f"[instance] {instance_id} -> fault validation completed")
                continue
            coordinator._recovery_failed_state = failed_state
            if str(args.program_space_version).strip().lower() == "atomic_actions_v1":
                if (
                    args.diagnosis_mode == "contaminated_post_patch_v1"
                    and str(args.fault_family).strip().lower() == "contaminated_post_patch"
                ):
                    programs = _contaminated_diagnosis_programs(failed_state)
                else:
                    programs = _synthesized_atomic_programs(
                        coordinator,
                        failed_state=failed_state,
                        max_candidate_programs=args.max_candidate_programs,
                        program_online_selector=args.program_online_selector,
                    )
            else:
                programs = _candidate_programs_for_profile(
                    checkpoint_ids,
                    profile=args.profile,
                    program_space_version=args.program_space_version,
                    fault_family=args.fault_family,
                )
            programs = _filter_programs_by_id(
                programs,
                program_id_filter=args.program_id_filter,
            )
            programs = _limit_programs_for_online_execution(
                programs,
                failed_state=failed_state,
                budget=coordinator.config.budget,
                execute_candidate_limit=args.execute_candidate_limit,
                program_online_selector=args.program_online_selector,
            )
            outcomes = _evaluate_programs(
                coordinator,
                failed_state=failed_state,
                programs=programs,
                manifest=manifest,
                polluted_checkpoint_id=checkpoint_ids["post_fault"],
            )
            _upsert_row(rows,
                _summarize_case(
                    coord=coordinator,
                    manifest=manifest,
                    failed_state=failed_state,
                    programs=programs,
                    outcomes=outcomes,
                )
            )
            print(f"[instance] {instance_id} -> completed")
        except Exception as exc:
            error_text = str(exc)
            error_type = _classify_error(error_text)
            _upsert_row(
                rows,
                {
                    "instance_id": instance_id,
                    **current_row_prefix,
                    "error": error_text,
                    "error_type": error_type,
                },
            )
            print(f"[instance] {instance_id} -> error({error_type}): {exc}")
        finally:
            if runtime_session is not None:
                runtime_session.close()
        payload = {
            "rows": rows,
            "summary": _aggregate(rows),
        }
        _write_payload_atomic(args.output, payload)

    payload = {
        "rows": rows,
        "summary": _aggregate(rows),
    }
    _write_payload_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
