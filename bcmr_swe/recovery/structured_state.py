"""Unified structured recovery state builders and projections for BCMR mainline."""

from __future__ import annotations

import json
import re
from typing import Any

from bcmr_swe.types import FailedState, PropagationObject, StructuredRecoveryState
from swe_mas.utils.path_filters import normalize_repo_path, parse_unified_diff_paths


SELECTION_FACT_KEYS = {
    "localized_path",
    "selected_target",
    "selected_file",
    "selected_module",
    "target_file",
    "target_module",
}


def _dedupe_strs(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _truncate(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    return " ".join(text.split())[:limit]


def _normalize_repo_path(path: str) -> str:
    return normalize_repo_path(path)


def _extract_paths_from_text(text: str) -> list[str]:
    return _dedupe_strs(
        [
            _normalize_repo_path(match)
            for match in re.findall(r"[A-Za-z0-9_./-]+\.py", str(text or ""))
        ]
    )


def _paths_from_test_ids(test_ids: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in test_ids:
        text = str(item or "").strip()
        if not text:
            continue
        path = text.split("::", 1)[0]
        if path.endswith(".py"):
            paths.append(_normalize_repo_path(path))
    return _dedupe_strs(paths)


def _list_field(payload: dict[str, Any], *keys: str) -> list[str]:
    values: list[Any] = []
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif isinstance(raw, tuple):
            values.extend(list(raw))
        elif raw:
            values.append(raw)
    return _dedupe_strs([_normalize_repo_path(str(item)) for item in values])


def _patcher_modified_files(patcher: dict[str, Any]) -> list[str]:
    patch_summary = dict(patcher.get("patch_summary", {}) or {})
    trace = dict(patcher.get("patcher_trace", {}) or {})
    trace_summary = dict(trace.get("patcher_patch_summary", {}) or {})
    changed = _list_field(
        patch_summary,
        "fresh_source_files",
        "source_files",
        "changed_files",
        "fresh_changed_files",
        "effective_files",
        "modified_files",
    )
    changed.extend(
        _list_field(
            trace_summary,
            "fresh_source_files",
            "source_files",
            "changed_files",
            "fresh_changed_files",
            "effective_files",
            "modified_files",
        )
    )
    if not changed:
        changed.extend(parse_unified_diff_paths(str(patcher.get("patch", "") or "")))
    if not changed:
        changed.extend(_extract_paths_from_text(str(patcher.get("patch", "") or "")))
    return _dedupe_strs(changed)


def _verifier_failed_files(failing_tests: list[Any], verifier_text: str) -> list[str]:
    return _dedupe_strs(_paths_from_test_ids(failing_tests) + _extract_paths_from_text(verifier_text))


def _verifier_assertion_count(verifier_text: str) -> int:
    text = str(verifier_text or "")
    if not text:
        return 0
    return len(re.findall(r"\bAssertionError\b|\bassert\b|E\s+assert", text, flags=re.IGNORECASE))


_KNOWN_EXCEPTION_TYPES = (
    "SyntaxError",
    "TypeError",
    "AttributeError",
    "KeyError",
    "ValueError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "IndexError",
    "RuntimeError",
    "AssertionError",
)


def _verifier_exception_type(verifier_text: str) -> str | None:
    text = str(verifier_text or "")
    if not text:
        return None
    for exception_type in _KNOWN_EXCEPTION_TYPES:
        if re.search(rf"\b{re.escape(exception_type)}\b", text):
            return exception_type
    match = re.search(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b", text)
    return match.group(1) if match else None


def _patcher_stop_reason(patcher: dict[str, Any]) -> str | None:
    patch_summary = dict(patcher.get("patch_summary", {}) or {})
    for key in ("stop_reason", "failure_mode", "status", "error", "exception_type"):
        value = str(patcher.get(key, "") or patch_summary.get(key, "") or "").strip()
        if value:
            return value
    return None


def _counterexample_evidence_fields(
    *,
    failing_tests: list[Any],
    verifier_text: str,
    patcher: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verifier_failed_files": _verifier_failed_files(failing_tests, verifier_text),
        "patcher_modified_files": _patcher_modified_files(patcher),
        "verifier_assertion_count": _verifier_assertion_count(verifier_text),
        "verifier_exception_type": _verifier_exception_type(verifier_text),
        "patcher_stop_reason": _patcher_stop_reason(patcher),
    }


def _checkpoint_candidates_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(metadata.get("checkpoint_candidates", []) or [])
        if isinstance(item, dict)
    ]


def _role_aggregate_view(phase_outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for role in ("locator", "planner", "patcher", "implementer", "verifier"):
        payload = dict(phase_outputs.get(role, {}) or {})
        if not payload:
            continue
        commands = payload.get("commands", []) or []
        messages = payload.get("messages", []) or []
        roles[role] = {
            "success": bool(payload.get("success", False)),
            "stop_reason": str(payload.get("stop_reason", "") or ""),
            "command_count": len(commands) if isinstance(commands, list) else 0,
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "evidence_delta_kind": str(payload.get("evidence_delta_kind", "") or ""),
            "inspected_regions_or_symbols": _dedupe_strs(list(payload.get("inspected_regions_or_symbols", []) or [])),
            "selected_target_candidates": _dedupe_strs(list(payload.get("selected_target_candidates", []) or [])),
            "located_files_excerpt": _truncate(payload.get("located_files", ""), 600),
            "plan_excerpt": _truncate(payload.get("plan", ""), 600),
            "patch_excerpt": _truncate(payload.get("patch", ""), 600),
            "verification_excerpt": _truncate(payload.get("verification", ""), 600),
            "status": str(payload.get("status", "") or ""),
        }
    return roles


def _propagation_object(
    *,
    instance_id: str,
    object_type: str,
    index: int,
    producer_stage: str,
    consumer_stage: str,
    contamination_status: str,
    evidence_anchor: str,
    replay_anchor: str,
    verifier_link: str,
    payload: dict[str, Any],
) -> PropagationObject:
    suffix = {
        "Selection": "selection",
        "SharedFact": "shared",
        "VerifierVerdict": "verifier",
    }.get(object_type, object_type.lower())
    return PropagationObject(
        object_type=object_type,
        object_id=f"{instance_id}::{suffix}::{index}",
        producer_stage=producer_stage,
        consumer_stage=consumer_stage,
        contamination_status=contamination_status,
        evidence_anchor=evidence_anchor,
        replay_anchor=replay_anchor,
        verifier_link=verifier_link,
        payload=dict(payload),
    )


def _selection_objects_from_phase_outputs(instance_id: str, phase_outputs: dict[str, Any]) -> list[PropagationObject]:
    locator = dict(phase_outputs.get("locator", {}) or {})
    candidates = _dedupe_strs(locator.get("selected_target_candidates", []) or [])
    if not candidates:
        candidates = _extract_paths_from_text(locator.get("located_files", ""))
    objects: list[PropagationObject] = []
    for index, path in enumerate(candidates[:6]):
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="Selection",
                index=index,
                producer_stage="locator",
                consumer_stage="patcher",
                contamination_status="suspicious",
                evidence_anchor=f"selection[{index}]={path}",
                replay_anchor="post_locate",
                verifier_link="",
                payload={
                    "fact_key": "selected_file",
                    "fact_value_excerpt": path,
                    "path": path,
                },
            )
        )
    return objects


def _shared_fact_objects_from_graph(instance_id: str, nodes: list[dict[str, Any]]) -> list[PropagationObject]:
    objects: list[PropagationObject] = []
    for node in nodes:
        kind = str(node.get("kind", "") or "")
        if kind != "SharedFact":
            continue
        payload = dict(node.get("payload", {}) or {})
        fact_key = str(payload.get("fact_key", "") or "")
        if fact_key in SELECTION_FACT_KEYS:
            continue
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="SharedFact",
                index=len(objects),
                producer_stage=str(node.get("role", "") or "patcher"),
                consumer_stage="verifier",
                contamination_status="stale_or_unverified" if str(node.get("status", "") or "") != "active" else "unknown",
                evidence_anchor=f"shared_fact[{len(objects)}]={fact_key or node.get('node_id', '')}",
                replay_anchor="post_patch",
                verifier_link="",
                payload={
                    "node_id": str(node.get("node_id", "") or ""),
                    "fact_key": fact_key,
                    "fact_value_excerpt": _truncate(payload.get("fact_value", "")),
                    "status": str(node.get("status", "") or ""),
                    "role": str(node.get("role", "") or ""),
                    "phase": str(node.get("phase", "") or ""),
                },
            )
        )
    return objects


def _verifier_objects_from_graph(instance_id: str, nodes: list[dict[str, Any]]) -> list[PropagationObject]:
    objects: list[PropagationObject] = []
    for node in nodes:
        kind = str(node.get("kind", "") or "")
        if kind != "VerifierResult":
            continue
        payload = dict(node.get("payload", {}) or {})
        verdict = str(payload.get("verdict", "") or "")
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="VerifierVerdict",
                index=len(objects),
                producer_stage=str(node.get("role", "") or "verifier"),
                consumer_stage="bcmr",
                contamination_status="contradiction" if payload.get("contradicted_fact_ids") else "observed",
                evidence_anchor=f"verifier[{len(objects)}]={verdict or node.get('node_id', '')}",
                replay_anchor="post_patch",
                verifier_link="self",
                payload={
                    "node_id": str(node.get("node_id", "") or ""),
                    "verdict": verdict,
                    "test_status": str(payload.get("test_status", "") or ""),
                    "failing_tests": list(payload.get("failing_tests", []) or []),
                    "output_excerpt": _truncate(payload.get("output_excerpt", ""), 600),
                },
            )
        )
    return objects


def _typed_assets_to_object_chain(
    *,
    instance_id: str,
    typed_assets: dict[str, Any],
) -> list[PropagationObject]:
    objects: list[PropagationObject] = []
    for index, item in enumerate(list(typed_assets.get("selection_objects", []) or [])):
        if not isinstance(item, dict):
            continue
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="Selection",
                index=index,
                producer_stage=str(item.get("role", "") or "locator"),
                consumer_stage="patcher",
                contamination_status="suspicious",
                evidence_anchor=f"selection_objects[{index}]={item.get('fact_value_excerpt', '')}",
                replay_anchor="post_locate",
                verifier_link="",
                payload=item,
            )
        )
    for index, item in enumerate(list(typed_assets.get("shared_fact_objects", []) or [])):
        if not isinstance(item, dict):
            continue
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="SharedFact",
                index=index,
                producer_stage=str(item.get("role", "") or "patcher"),
                consumer_stage="verifier",
                contamination_status="stale_or_unverified" if str(item.get("status", "") or "") != "active" else "unknown",
                evidence_anchor=f"shared_fact_objects[{index}]={item.get('fact_key', '')}",
                replay_anchor="post_patch",
                verifier_link="",
                payload=item,
            )
        )
    for index, item in enumerate(list(typed_assets.get("verifier_verdict_objects", []) or [])):
        if not isinstance(item, dict):
            continue
        objects.append(
            _propagation_object(
                instance_id=instance_id,
                object_type="VerifierVerdict",
                index=index,
                producer_stage=str(item.get("role", "") or "verifier"),
                consumer_stage="bcmr",
                contamination_status="contradiction" if str(item.get("verdict", "") or "") == "fail" else "observed",
                evidence_anchor=f"verifier_verdict_objects[{index}]={item.get('verdict', '')}",
                replay_anchor="post_patch",
                verifier_link="self",
                payload=item,
            )
        )
    return objects


def _negative_constraints(
    *,
    trigger_type: str,
    stop_reason: str,
    target_legitimacy: str,
    patch_legitimacy: str,
    object_chain: list[PropagationObject],
    phase_outputs: dict[str, Any],
) -> list[str]:
    constraints: list[str] = []
    if trigger_type == "fact_conflict":
        constraints.append("latest propagated belief may be stale or contradicted")
    if trigger_type == "verifier_contradiction":
        constraints.append("current verifier outcome contradicts the previously promoted local conclusion")
    if trigger_type == "no_progress_loop":
        constraints.append("avoid repeating the same local replay without state change")
    if stop_reason == "budget_exhausted_with_partial_evidence":
        constraints.append("reuse existing evidence before expanding search")
    if stop_reason == "true_no_progress":
        constraints.append("prefer changing anchor or object state before retrying")
    if target_legitimacy == "no_diff" or patch_legitimacy == "no_effective_patch":
        constraints.append("previous recovery attempt produced no effective source diff")
    if target_legitimacy in {"tests_only", "generated_only", "non_source_only"}:
        constraints.append("avoid non-canonical targets and prioritize source paths")
    patcher = dict(phase_outputs.get("patcher", {}) or {})
    if patcher.get("infrastructure_error"):
        constraints.append("previous patcher execution was affected by infrastructure noise")
    if any(obj.object_type == "Selection" for obj in object_chain):
        constraints.append("treat current localized target set as suspicious, not authoritative")
    return _dedupe_strs(constraints)


def build_structured_recovery_state_from_failed_state(failed_state: FailedState) -> StructuredRecoveryState:
    metadata = dict(failed_state.metadata or {})
    phase_outputs = dict(metadata.get("phase_outputs", {}) or {})
    trigger = failed_state.trigger.to_dict()
    suspect_summary = dict(failed_state.suspect_region.summary or {})
    checkpoint_candidates = _checkpoint_candidates_from_metadata(metadata)
    checkpoint_labels = _dedupe_strs(
        [
            candidate.get("label")
            or dict(candidate.get("metadata", {}) or {}).get("stage", "")
            for candidate in checkpoint_candidates
        ]
    )
    post_fault_labels = [label for label in checkpoint_labels if "fault" in label.lower()]
    healthy_anchor_candidates = [label for label in checkpoint_labels if label not in post_fault_labels] if post_fault_labels else list(checkpoint_labels)
    current_checkpoint_label = str(getattr(failed_state.checkpoint, "label", "") or "")

    suspect_graph = dict(metadata.get("suspect_graph", {}) or {})
    graph_nodes = []
    nodes = suspect_graph.get("nodes", {})
    if isinstance(nodes, dict):
        graph_nodes = [dict(item) for item in nodes.values() if isinstance(item, dict)]
    elif isinstance(nodes, list):
        graph_nodes = [dict(item) for item in nodes if isinstance(item, dict)]

    object_chain = _selection_objects_from_phase_outputs(failed_state.instance_id, phase_outputs)
    object_chain.extend(_shared_fact_objects_from_graph(failed_state.instance_id, graph_nodes))
    object_chain.extend(_verifier_objects_from_graph(failed_state.instance_id, graph_nodes))

    verifier = dict(phase_outputs.get("verifier", {}) or {})
    patcher = dict(phase_outputs.get("patcher", {}) or {})
    locator = dict(phase_outputs.get("locator", {}) or {})
    failing_tests = list(dict(metadata.get("latest_test_status", {}) or {}).get("failing_tests", []) or [])
    verifier_excerpt = _truncate(
        verifier.get("verification", "")
        or dict(metadata.get("latest_test_status", {}) or {}).get("verification", "")
        or dict(metadata.get("failure_observation", {}) or {}).get("verification_excerpt", ""),
        1200,
    )
    selected_target_candidates = _dedupe_strs(
        list(locator.get("selected_target_candidates", []) or [])
        + _extract_paths_from_text(locator.get("located_files", ""))
        + _extract_paths_from_text(patcher.get("plan", ""))
    )
    target_legitimacy = str(
        dict(patcher.get("patch_summary", {}) or {}).get("target_legitimacy", "")
        or metadata.get("target_legitimacy", "")
        or ""
    )
    patch_legitimacy = str(
        dict(patcher.get("patch_summary", {}) or {}).get("failure_mode", "")
        or metadata.get("patch_legitimacy", "")
        or ""
    )
    stop_reason = str(verifier.get("stop_reason") or patcher.get("stop_reason") or locator.get("stop_reason") or "")

    state = StructuredRecoveryState(
        local_region_view={
            "source_type": "natural_failed_state",
            "fault_type": str(trigger.get("trigger_type", "") or "unknown"),
            "trigger_type": str(trigger.get("trigger_type", "") or "unknown"),
            "trigger_reason": str(trigger.get("reason", "") or ""),
            "suspect_region_size": int(suspect_summary.get("size", len(failed_state.suspect_region.node_ids)) or 0),
            "role_chain": list(suspect_summary.get("role_chain", []) or []),
            "replay_anchor_role": str(suspect_summary.get("replay_anchor_role", "") or ""),
            "has_conflicting_fact": bool(suspect_summary.get("has_conflicting_fact", False)),
            "conflicting_fact": str(suspect_summary.get("conflicting_fact_key", "") or metadata.get("conflicting_fact", "") or ""),
            "selected_target_candidates": selected_target_candidates[:8],
            "suspect_paths": selected_target_candidates[:8],
        },
        role_aggregate_view=_role_aggregate_view(phase_outputs),
        object_chain_view=object_chain,
        replay_anchor_view={
            "current_checkpoint_id": failed_state.checkpoint_id,
            "current_checkpoint_label": current_checkpoint_label,
            "checkpoint_labels_available": checkpoint_labels[:8],
            "healthy_anchor_candidates": healthy_anchor_candidates[:6],
            "post_fault_checkpoint_labels": post_fault_labels[:4],
        },
        evidence_pack={
            "failing_tests": failing_tests,
            "failing_tests_count": len(failing_tests),
            "verifier_excerpt": verifier_excerpt,
            "target_legitimacy": target_legitimacy,
            "patch_legitimacy": patch_legitimacy,
            "stop_reason": stop_reason,
            "selected_target_candidates": selected_target_candidates[:8],
            "failure_family_manual": str(metadata.get("failure_family_manual", "") or ""),
            "typed_object_quality": str(metadata.get("typed_object_quality", "") or ""),
            "review_verifier_excerpt": str(dict(metadata.get("failure_observation", {}) or {}).get("verification_excerpt", "") or "")[:1200],
            **_counterexample_evidence_fields(
                failing_tests=failing_tests,
                verifier_text=verifier_excerpt,
                patcher=patcher,
            ),
        },
        state_numeric={
            str(key): float(value)
            for key, value in dict(failed_state.state_features.numeric or {}).items()
        },
        metadata={
            "instance_id": failed_state.instance_id,
            "source_type": "natural_failed_state",
            "phase_outputs": phase_outputs,
            "model_routes": dict(metadata.get("model_routes", {}) or {}),
        },
    )
    constraints = _negative_constraints(
        trigger_type=state.local_region_view["trigger_type"],
        stop_reason=stop_reason,
        target_legitimacy=target_legitimacy,
        patch_legitimacy=patch_legitimacy,
        object_chain=object_chain,
        phase_outputs=phase_outputs,
    )
    state.local_region_view["negative_constraints"] = constraints
    state.evidence_pack["negative_constraints"] = constraints
    return state


def build_structured_recovery_state_from_trajectory_artifacts(
    *,
    instance_id: str,
    natural_failure_family: str,
    stage_outputs: dict[str, Any],
    typed_assets: dict[str, Any],
    review_metadata: dict[str, Any],
) -> StructuredRecoveryState:
    object_chain = _typed_assets_to_object_chain(
        instance_id=instance_id,
        typed_assets=typed_assets,
    )
    replay_anchors = [
        dict(item)
        for item in list(typed_assets.get("replay_anchors", []) or [])
        if isinstance(item, dict)
    ]
    checkpoint_labels = _dedupe_strs([item.get("label", "") for item in replay_anchors])
    role_chain = []
    for obj in object_chain:
        if obj.producer_stage and obj.producer_stage not in role_chain:
            role_chain.append(obj.producer_stage)
        if obj.consumer_stage and obj.consumer_stage not in role_chain:
            role_chain.append(obj.consumer_stage)
    if not role_chain:
        role_chain = [role for role in ("locator", "planner", "patcher", "verifier") if stage_outputs.get(role)]
    selected_target_candidates = _dedupe_strs(
        list(review_metadata.get("selected_target_candidates", []) or [])
        + _extract_paths_from_text(dict(stage_outputs.get("locator", {}) or {}).get("located_files", ""))
    )
    verifier = dict(stage_outputs.get("verifier", {}) or {})
    patcher = dict(stage_outputs.get("patcher", {}) or {})
    failing_tests = list(review_metadata.get("failing_tests", []) or verifier.get("failing_tests", []) or [])
    verifier_excerpt = _truncate(verifier.get("verification", "") or review_metadata.get("verifier_excerpt", ""), 1200)
    state = StructuredRecoveryState(
        local_region_view={
            "source_type": "natural",
            "fault_type": str(natural_failure_family or "unknown"),
            "trigger_type": str(natural_failure_family or "unknown"),
            "trigger_reason": str(review_metadata.get("stop_reason", "") or ""),
            "suspect_region_size": len(object_chain),
            "role_chain": role_chain,
            "replay_anchor_role": role_chain[0] if role_chain else "",
            "has_conflicting_fact": any(obj.object_type == "SharedFact" and obj.contamination_status != "unknown" for obj in object_chain),
            "conflicting_fact": next((obj.payload.get("fact_key", "") for obj in object_chain if obj.object_type == "SharedFact"), ""),
            "selected_target_candidates": selected_target_candidates[:8],
            "suspect_paths": selected_target_candidates[:8],
            "negative_constraints": [],
        },
        role_aggregate_view=_role_aggregate_view(stage_outputs),
        object_chain_view=object_chain,
        replay_anchor_view={
            "current_checkpoint_id": replay_anchors[-1].get("checkpoint_id", "") if replay_anchors else "",
            "current_checkpoint_label": replay_anchors[-1].get("label", "") if replay_anchors else "",
            "checkpoint_labels_available": checkpoint_labels[:8],
            "healthy_anchor_candidates": [label for label in checkpoint_labels if label][:6],
            "post_fault_checkpoint_labels": [],
        },
        evidence_pack={
            "failing_tests": failing_tests,
            "failing_tests_count": len(failing_tests),
            "verifier_excerpt": verifier_excerpt,
            "target_legitimacy": str(review_metadata.get("target_legitimacy", "") or ""),
            "patch_legitimacy": str(review_metadata.get("patch_legitimacy", "") or ""),
            "stop_reason": str(review_metadata.get("stop_reason", "") or ""),
            "selected_target_candidates": selected_target_candidates[:8],
            "inspected_regions_or_symbols": _dedupe_strs(list(review_metadata.get("inspected_regions_or_symbols", []) or [])),
            **_counterexample_evidence_fields(
                failing_tests=failing_tests,
                verifier_text=verifier_excerpt,
                patcher=patcher,
            ),
        },
        state_numeric={
            "typed_object_count": float(len(object_chain)),
            "selection_count": float(len([obj for obj in object_chain if obj.object_type == "Selection"])),
            "shared_fact_count": float(len([obj for obj in object_chain if obj.object_type == "SharedFact"])),
            "verifier_verdict_count": float(len([obj for obj in object_chain if obj.object_type == "VerifierVerdict"])),
            "replay_anchor_count": float(len(replay_anchors)),
        },
        metadata={
            "instance_id": instance_id,
            "source_type": "natural",
            "phase_outputs": dict(stage_outputs or {}),
        },
    )
    constraints = _negative_constraints(
        trigger_type=str(natural_failure_family or "unknown"),
        stop_reason=str(review_metadata.get("stop_reason", "") or ""),
        target_legitimacy=str(review_metadata.get("target_legitimacy", "") or ""),
        patch_legitimacy=str(review_metadata.get("patch_legitimacy", "") or ""),
        object_chain=object_chain,
        phase_outputs=stage_outputs,
    )
    state.local_region_view["negative_constraints"] = constraints
    state.evidence_pack["negative_constraints"] = constraints
    return state


def project_structured_state_core(state: StructuredRecoveryState) -> dict[str, Any]:
    local = dict(state.local_region_view or {})
    replay = dict(state.replay_anchor_view or {})
    evidence = dict(state.evidence_pack or {})
    return {
        "core_recovery_state": {
            "source_type": str(local.get("source_type", "") or state.metadata.get("source_type", "")),
            "fault_type": str(local.get("fault_type", "") or "unknown"),
            "trigger_type": str(local.get("trigger_type", "") or "unknown"),
            "trigger_reason": str(local.get("trigger_reason", "") or ""),
            "suspect_region_size": int(local.get("suspect_region_size", 0) or 0),
            "role_chain": list(local.get("role_chain", []) or []),
            "has_conflicting_fact": bool(local.get("has_conflicting_fact", False)),
            "conflicting_fact": str(local.get("conflicting_fact", "") or ""),
            "checkpoint_labels_available": list(replay.get("checkpoint_labels_available", []) or []),
            "current_checkpoint_label": str(replay.get("current_checkpoint_label", "") or ""),
            "healthy_anchor_candidates": list(replay.get("healthy_anchor_candidates", []) or []),
            "failing_tests_count": int(evidence.get("failing_tests_count", 0) or 0),
            "selected_target_candidates": list(local.get("selected_target_candidates", []) or []),
            "negative_constraints": list(local.get("negative_constraints", []) or []),
            "target_legitimacy": str(evidence.get("target_legitimacy", "") or ""),
            "patch_legitimacy": str(evidence.get("patch_legitimacy", "") or ""),
        }
    }


def project_structured_state_with_evidence(state: StructuredRecoveryState) -> dict[str, Any]:
    payload = project_structured_state_core(state)
    payload["evidence_pack"] = dict(state.evidence_pack or {})
    payload["role_aggregate_view"] = {
        str(key): dict(value)
        for key, value in state.role_aggregate_view.items()
    }
    payload["object_chain_view"] = [item.to_dict() for item in state.object_chain_view]
    payload["replay_anchor_view"] = dict(state.replay_anchor_view or {})
    payload["state_numeric"] = {
        str(key): float(value)
        for key, value in dict(state.state_numeric or {}).items()
    }
    return payload


def project_structured_state_llm_payload(state: StructuredRecoveryState) -> dict[str, Any]:
    """Project the state into PARC's fixed-enum LLM-native schema."""

    return state.to_llm_payload()


def render_structured_state_llm_payload(state: StructuredRecoveryState) -> str:
    """Render PARC's LLM-native schema as stable compact JSON."""

    return json.dumps(state.to_llm_payload(), ensure_ascii=False, separators=(",", ":"))
