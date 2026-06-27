"""Build propagation-aware MAS failure graphs from normalized trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from bcmr_swe.mas_diagnosis.schema import MASDXTrajectoryRecord


STAGE_ORDER = ("locator", "planner", "implementer", "patcher", "verifier")


def build_propagation_graph(record: MASDXTrajectoryRecord | dict[str, Any]) -> dict[str, Any]:
    """Build a MAS-DX-R propagation graph.

    The graph is intentionally richer than the older MAS-DX diagnosis graph: it
    separates stage nodes from artifacts such as localization, patch, verifier
    verdict, oracle result, shared facts, test targets, and commands.
    """

    if not isinstance(record, MASDXTrajectoryRecord):
        record = MASDXTrajectoryRecord.from_dict(dict(record))

    evidence_spans = _evidence_spans(record)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    stage_ids = _add_stage_nodes(record, nodes, evidence_spans)
    artifact_ids = _add_artifact_nodes(record, nodes, evidence_spans)
    _add_command_nodes(record, nodes, edges, stage_ids, evidence_spans)
    _add_stage_handoffs(edges, stage_ids)
    _add_artifact_edges(record, edges, stage_ids, artifact_ids)

    semantic_edges = _build_semantic_edges(edges)
    summary = _summary(record, artifact_ids)
    summary["semantic_edge_counts"] = _count_edge_types(semantic_edges)
    summary["has_semantic_edges"] = bool(semantic_edges)
    return {
        "schema": "mas_dx_r_propagation_graph_v1",
        "case_id": record.case_id,
        "instance_id": record.instance_id,
        "nodes": nodes,
        "edges": edges,
        "semantic_edges": semantic_edges,
        "evidence_spans": evidence_spans,
        "summary": summary,
        "source_record_digest": _digest(record.to_dict()),
    }


def _add_stage_nodes(
    record: MASDXTrajectoryRecord,
    nodes: list[dict[str, Any]],
    evidence_spans: dict[str, str],
) -> dict[str, str]:
    stage_ids: dict[str, str] = {}
    for stage in STAGE_ORDER:
        output = dict(record.stage_outputs.get(stage, {}) or {})
        if not output:
            continue
        node_id = f"stage:{stage}"
        stage_ids[stage] = node_id
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "stage",
                "label": stage,
                "stage": stage,
                "artifact_type": "",
                "status": _stage_status(output),
                "payload": {
                    "success": output.get("success"),
                    "stop_reason": str(output.get("stop_reason", "") or ""),
                    "command_count": len(list(output.get("commands", []) or [])),
                },
                "evidence_span_ids": [f"span:stage:{stage}"],
            }
        )
    nodes.append(
        {
            "node_id": "stage:oracle",
            "node_type": "stage",
            "label": "oracle",
            "stage": "oracle",
            "artifact_type": "",
            "status": "success" if record.oracle.get("oracle_success") else "failed",
            "payload": {
                "oracle_success": bool(record.oracle.get("oracle_success", False)),
                "oracle_returncode": record.oracle.get("oracle_returncode"),
                "fail_to_pass_returncode": record.oracle.get("fail_to_pass_returncode"),
            },
            "evidence_span_ids": ["span:oracle"],
        }
    )
    return stage_ids


def _add_artifact_nodes(
    record: MASDXTrajectoryRecord,
    nodes: list[dict[str, Any]],
    evidence_spans: dict[str, str],
) -> dict[str, str]:
    artifact_ids: dict[str, str] = {}
    locator = dict(record.stage_outputs.get("locator", {}) or {})
    located_files = str(locator.get("located_files", "") or "").strip()
    selected_targets = _selected_targets(record)
    if located_files or selected_targets:
        artifact_ids["localization"] = "artifact:localization"
        nodes.append(
            _artifact_node(
                "artifact:localization",
                "localization",
                "locator",
                "available",
                {
                    "located_files_excerpt": located_files[:700],
                    "selected_targets": selected_targets[:20],
                },
                ["span:stage:locator"],
            )
        )

    for key, value in sorted(record.shared_facts.items()):
        span_id = f"span:shared_fact:{key}"
        node_id = f"artifact:shared_fact:{key}"
        artifact_ids[f"shared_fact:{key}"] = node_id
        nodes.append(
            _artifact_node(
                node_id,
                str(key),
                "",
                "available",
                {"value_excerpt": _clip(value, 500)},
                [span_id] if span_id in evidence_spans else ["span:shared_facts"],
                artifact_type="shared_fact",
            )
        )

    patch_summary = _patch_summary(record)
    patcher = dict(record.stage_outputs.get("patcher", {}) or {})
    if patch_summary or str(patcher.get("patch", "") or "").strip():
        artifact_ids["patch"] = "artifact:patch"
        patch_text = str(patcher.get("patch", "") or "")
        replayability = _patch_replayability(patch_text)
        nodes.append(
            _artifact_node(
                "artifact:patch",
                "patch",
                "patcher",
                "available",
                {
                    "patch_text_available": bool(patch_text.strip()),
                    "patch_line_count": len(patch_text.splitlines()),
                    "changed_files": list(patch_summary.get("changed_files", []) or []),
                    "changed_file_classes": dict(patch_summary.get("changed_file_classes", {}) or {}),
                    "target_legitimacy": str(
                        patch_summary.get("fresh_target_legitimacy")
                        or patch_summary.get("target_legitimacy")
                        or ""
                    ),
                    "source_patch_risk": dict(patch_summary.get("source_patch_risk", {}) or {}),
                    "patch_replayability": replayability,
                },
                ["span:diff"],
            )
        )
        artifact_ids["patch_state"] = "artifact:patch_state"
        nodes.append(
            _artifact_node(
                "artifact:patch_state",
                "patch replay state",
                "patcher",
                "replayable" if replayability["has_replayable_text_shape"] else "unknown",
                {
                    "requires_replay_before_nonpatch_verification": True,
                    "patch_text_available": bool(patch_text.strip()),
                    "patch_line_count": len(patch_text.splitlines()),
                    "changed_files": list(patch_summary.get("changed_files", []) or []),
                    "replayability": replayability,
                },
                ["span:diff"],
                artifact_type="patch_state",
            )
        )

    test_targets = _test_targets(record)
    if test_targets:
        artifact_ids["test_targets"] = "artifact:test_targets"
        nodes.append(
            _artifact_node(
                "artifact:test_targets",
                "test targets",
                "verifier",
                "available",
                {"targets": test_targets[:30]},
                ["span:commands", "span:verifier"],
                artifact_type="test",
            )
        )

    if record.verifier_evidence or record.stage_outputs.get("verifier"):
        artifact_ids["verifier_verdict"] = "artifact:verifier_verdict"
        nodes.append(
            _artifact_node(
                "artifact:verifier_verdict",
                "verifier verdict",
                "verifier",
                _verifier_status(record),
                {
                    "status": str(record.verifier_evidence.get("status", "") or ""),
                    "success": _verifier_success(record),
                    "fail_to_pass_returncode": record.verifier_evidence.get("fail_to_pass_returncode"),
                },
                ["span:verifier"],
            )
        )

    artifact_ids["oracle_result"] = "artifact:oracle_result"
    nodes.append(
        _artifact_node(
            "artifact:oracle_result",
            "oracle result",
            "oracle",
            "success" if record.oracle.get("oracle_success") else "failed",
            dict(record.oracle),
            ["span:oracle"],
        )
    )
    return artifact_ids


def _add_command_nodes(
    record: MASDXTrajectoryRecord,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    stage_ids: dict[str, str],
    evidence_spans: dict[str, str],
) -> None:
    for index, command in enumerate(record.commands[-30:]):
        node_id = f"command:{index}"
        stage = str(command.get("stage", "") or "")
        span_id = f"span:command:{index}"
        evidence_spans.setdefault(span_id, _clip(command, 900))
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "command",
                "label": str(command.get("command", "") or "")[:80],
                "stage": stage,
                "artifact_type": _command_artifact_type(command),
                "status": "success" if int(command.get("returncode", 0) or 0) == 0 else "failed",
                "payload": {
                    "command": str(command.get("command", "") or ""),
                    "returncode": command.get("returncode"),
                    "read_only_probe": bool(command.get("read_only_probe", False)),
                },
                "evidence_span_ids": [span_id],
            }
        )
        if stage in stage_ids:
            edges.append(
                _edge(
                    stage_ids[stage],
                    node_id,
                    "observes" if bool(command.get("read_only_probe", False)) else "executes",
                    [span_id],
                )
            )


def _add_stage_handoffs(edges: list[dict[str, Any]], stage_ids: dict[str, str]) -> None:
    chain = [stage for stage in STAGE_ORDER if stage in stage_ids]
    for source, target in zip(chain, chain[1:]):
        edges.append(
            _edge(
                stage_ids[source],
                stage_ids[target],
                "handoff",
                [f"span:stage:{source}", f"span:stage:{target}"],
            )
        )


def _add_artifact_edges(
    record: MASDXTrajectoryRecord,
    edges: list[dict[str, Any]],
    stage_ids: dict[str, str],
    artifact_ids: dict[str, str],
) -> None:
    if "localization" in artifact_ids and "locator" in stage_ids:
        edges.append(_edge(stage_ids["locator"], artifact_ids["localization"], "produces", ["span:stage:locator"]))
    if "localization" in artifact_ids and "patcher" in stage_ids:
        edges.append(_edge(artifact_ids["localization"], stage_ids["patcher"], "handoff", ["span:stage:locator"]))
    if "localization" in artifact_ids and "verifier" in stage_ids:
        edges.append(_edge(artifact_ids["localization"], stage_ids["verifier"], "handoff", ["span:stage:locator"]))
    for key, node_id in artifact_ids.items():
        if not key.startswith("shared_fact:"):
            continue
        if "locator" in stage_ids:
            edges.append(_edge(stage_ids["locator"], node_id, "produces", ["span:shared_facts"]))
        if "patcher" in stage_ids:
            edges.append(_edge(node_id, stage_ids["patcher"], "consumes", ["span:shared_facts"]))
        if "verifier" in stage_ids:
            edges.append(_edge(node_id, stage_ids["verifier"], "consumes", ["span:shared_facts"]))
    if "patch" in artifact_ids and "patcher" in stage_ids:
        edges.append(_edge(stage_ids["patcher"], artifact_ids["patch"], "produces", ["span:diff"]))
    if "patch" in artifact_ids and "patch_state" in artifact_ids:
        edges.append(_edge(artifact_ids["patch"], artifact_ids["patch_state"], "requires_replay", ["span:diff"]))
    if "patch_state" in artifact_ids and "verifier" in stage_ids:
        edges.append(_edge(artifact_ids["patch_state"], stage_ids["verifier"], "must_precede", ["span:diff", "span:verifier"]))
    if "patch" in artifact_ids and "verifier" in stage_ids:
        edges.append(_edge(artifact_ids["patch"], stage_ids["verifier"], "tests", ["span:diff", "span:verifier"]))
    if "test_targets" in artifact_ids and "verifier" in stage_ids:
        edges.append(_edge(artifact_ids["test_targets"], stage_ids["verifier"], "consumes", ["span:commands"]))
    if "verifier_verdict" in artifact_ids and "verifier" in stage_ids:
        edges.append(_edge(stage_ids["verifier"], artifact_ids["verifier_verdict"], "produces", ["span:verifier"]))
    if "oracle_result" in artifact_ids:
        edges.append(_edge("stage:oracle", artifact_ids["oracle_result"], "produces", ["span:oracle"]))
    if "verifier_verdict" in artifact_ids and "oracle_result" in artifact_ids:
        edge_type = "contradicts" if _oracle_verifier_contradiction(record) else "validates"
        edges.append(_edge(artifact_ids["oracle_result"], artifact_ids["verifier_verdict"], edge_type, ["span:oracle", "span:verifier"]))


def _summary(record: MASDXTrajectoryRecord, artifact_ids: dict[str, str]) -> dict[str, Any]:
    patch_summary = _patch_summary(record)
    changed_classes = dict(patch_summary.get("changed_file_classes", {}) or {})
    fresh_classes = dict(patch_summary.get("fresh_changed_file_classes", {}) or {})
    test_files = list(changed_classes.get("test_files", []) or fresh_classes.get("test_files", []) or [])
    source_files = list(changed_classes.get("source_files", []) or fresh_classes.get("source_files", []) or [])
    text = _all_failure_text(record)
    has_collect_blocker = _has_test_collection_blocker(record, text)
    has_external_dependency_blocker = _has_external_dependency_blocker(text)
    has_patch_introduced_import_regression = _has_patch_introduced_import_regression(record, text)
    has_patch_added_missing_import_symbol = _has_patch_added_missing_import_symbol(record)
    has_explicit_patch_regression_language = _has_explicit_patch_regression_language(
        _patch_regression_failure_text(record)
    )
    has_patch_collection_causality_conflict = bool(
        has_collect_blocker
        and has_patch_introduced_import_regression
        and not has_explicit_patch_regression_language
        and not has_patch_added_missing_import_symbol
    )
    has_missing_dep = _contains_any(
        text,
        (
            "modulenotfounderror",
            "no module named",
            "missing dependency",
            "xmlschema",
            "hypothesis",
            "缺少依赖",
        ),
    )
    has_invalid_target = _contains_any(
        text,
        (
            "not found:",
            "not found",
            "nonexistent",
            "invalid test",
            "no tests ran",
            "nodeid",
            "测试路径",
            "不存在",
        ),
    )
    has_syntax_failure = _contains_any(
        text,
        (
            "syntaxerror",
            "indentationerror",
            "nameerror",
            "importerror",
            "语法",
            "缩进",
        ),
    )
    risk = dict(patch_summary.get("source_patch_risk", {}) or {})
    oracle_success = bool(record.oracle.get("oracle_success", False))
    shared_fact_count = sum(1 for key in artifact_ids if key.startswith("shared_fact:"))
    has_handoff_edges = bool("localization" in artifact_ids or shared_fact_count)
    has_shared_fact_verifier_dependency = bool(shared_fact_count and "verifier" in record.stage_outputs)
    has_suspicious_shared_fact_signal = _suspicious_shared_fact_signal(record)
    has_patch_artifact = "patch" in artifact_ids
    has_reusable_localization = "localization" in artifact_ids
    has_selection_only_handoff_failure = bool(
        has_reusable_localization
        and not has_patch_artifact
        and not oracle_success
        and not has_collect_blocker
        and not has_missing_dep
        and not has_external_dependency_blocker
        and not has_invalid_target
    )
    return {
        "has_oracle_verifier_contradiction": _oracle_verifier_contradiction(record),
        "has_test_collection_blocker": has_collect_blocker,
        "has_missing_dependency_signal": has_missing_dep,
        "has_external_dependency_blocker_signal": has_external_dependency_blocker,
        "has_patch_introduced_import_regression_signal": has_patch_introduced_import_regression,
        "has_patch_added_missing_import_symbol_signal": has_patch_added_missing_import_symbol,
        "has_explicit_patch_regression_language": has_explicit_patch_regression_language,
        "has_patch_collection_causality_conflict": has_patch_collection_causality_conflict,
        "has_patch_syntax_or_collection_error": bool(has_syntax_failure and test_files),
        "has_source_mixed_patch": bool(source_files and test_files),
        "has_broad_source_change": str(risk.get("risk_level", "") or "") == "broad_source_change",
        "has_invalid_test_target_signal": has_invalid_target,
        "has_handoff_edges": has_handoff_edges,
        "has_shared_fact_artifact": bool(shared_fact_count),
        "shared_fact_count": shared_fact_count,
        "has_shared_fact_verifier_dependency": has_shared_fact_verifier_dependency,
        "has_suspicious_shared_fact_signal": has_suspicious_shared_fact_signal,
        "has_reusable_localization": has_reusable_localization,
        "has_patch_artifact": has_patch_artifact,
        "has_selection_only_handoff_failure": has_selection_only_handoff_failure,
        "has_patch_state_artifact": "patch_state" in artifact_ids,
        "patch_requires_replay_before_nonpatch_verification": "patch_state" in artifact_ids,
        "patch_text_replay_shape_ok": _patch_replayability(
            str(dict(record.stage_outputs.get("patcher", {}) or {}).get("patch", "") or "")
        )["has_replayable_text_shape"],
        "oracle_success": oracle_success,
        "verifier_success": bool(_verifier_success(record)),
        "candidate_recovery_actions": _candidate_actions(
            has_collect_blocker=has_collect_blocker,
            has_missing_dep=has_missing_dep,
            has_invalid_target=has_invalid_target,
            contradiction=_oracle_verifier_contradiction(record),
            has_patch_artifact=has_patch_artifact,
            has_selection_only_handoff_failure=has_selection_only_handoff_failure,
            has_handoff_invalid_target=bool(has_invalid_target and has_handoff_edges),
            has_suspicious_shared_fact_signal=has_suspicious_shared_fact_signal,
        ),
    }


def _candidate_actions(
    *,
    has_collect_blocker: bool,
    has_missing_dep: bool,
    has_invalid_target: bool,
    contradiction: bool,
    has_patch_artifact: bool,
    has_selection_only_handoff_failure: bool,
    has_handoff_invalid_target: bool,
    has_suspicious_shared_fact_signal: bool,
) -> list[str]:
    actions: list[str] = []
    if has_collect_blocker or has_missing_dep:
        actions.append("environment_preflight_then_verifier")
    if has_invalid_target or contradiction:
        actions.append("verifier_only_replay")
    if has_handoff_invalid_target:
        actions.append("handoff_correction_then_verifier")
    if has_suspicious_shared_fact_signal:
        actions.append("handoff_correction_then_verifier")
        actions.append("shared_fact_quarantine_then_repatch")
    if has_patch_artifact:
        actions.append("patcher_fixed_localization")
    if has_selection_only_handoff_failure:
        actions.append("patcher_fixed_localization")
    return list(dict.fromkeys(actions))


def _evidence_spans(record: MASDXTrajectoryRecord) -> dict[str, str]:
    spans = {
        "span:issue": record.issue[:1600],
        "span:shared_facts": _clip(record.shared_facts, 2000),
        "span:commands": _clip(record.commands[-30:], 3000),
        "span:diff": _clip(record.diff_summary, 2000),
        "span:verifier": _clip(record.verifier_evidence or record.stage_outputs.get("verifier", {}), 2200),
        "span:oracle": _clip(record.oracle, 1200),
    }
    for stage, output in record.stage_outputs.items():
        spans[f"span:stage:{stage}"] = _clip(output, 2200)
    for key, value in record.shared_facts.items():
        spans[f"span:shared_fact:{key}"] = _clip(value, 1200)
    return spans


def _suspicious_shared_fact_signal(record: MASDXTrajectoryRecord) -> bool:
    if not record.shared_facts:
        return False
    text = _clip(record.shared_facts, 4000).lower()
    return _contains_any(
        text,
        (
            "stale shared",
            "stale fact",
            "stale belief",
            "conflicting shared",
            "contradictory shared",
            "shared fact conflict",
            "belief conflict",
            "wrong shared",
            "污染",
            "过期",
            "矛盾",
        ),
    )


def _artifact_node(
    node_id: str,
    label: str,
    stage: str,
    status: str,
    payload: dict[str, Any],
    evidence_span_ids: list[str],
    *,
    artifact_type: str = "artifact",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": "artifact",
        "label": label,
        "stage": stage,
        "artifact_type": artifact_type,
        "status": status,
        "payload": payload,
        "evidence_span_ids": evidence_span_ids,
    }


def _edge(source: str, target: str, edge_type: str, evidence_span_ids: list[str]) -> dict[str, Any]:
    raw = f"{source}->{edge_type}->{target}"
    return {
        "edge_id": "edge:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "artifact_type": "",
        "confidence": "observed" if evidence_span_ids else "structural",
        "payload": {},
        "evidence_span_ids": evidence_span_ids,
    }


def _build_semantic_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic_edges: list[dict[str, Any]] = []
    for edge in edges:
        edge_type = str(edge.get("edge_type", "") or "")
        source = str(edge.get("source", "") or "")
        target = str(edge.get("target", "") or "")
        spans = list(edge.get("evidence_span_ids", []) or [])
        source_edge_id = str(edge.get("edge_id", "") or "")
        if edge_type == "produces":
            semantic_edges.append(
                _semantic_edge(
                    target,
                    source,
                    "derived_from",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Artifact or verdict was produced by an upstream MAS stage.",
                )
            )
        elif edge_type in {"consumes", "handoff"}:
            semantic_edges.append(
                _semantic_edge(
                    source,
                    target,
                    "trusted_by",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Downstream stage or artifact consumed/trusted upstream propagated evidence.",
                )
            )
        elif edge_type == "contradicts":
            semantic_edges.append(
                _semantic_edge(
                    target,
                    source,
                    "contradicted_by",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Oracle or verifier evidence contradicts an upstream verdict/artifact.",
                )
            )
        elif edge_type == "validates":
            semantic_edges.append(
                _semantic_edge(
                    target,
                    source,
                    "validated_by",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Oracle or verifier evidence validates the propagated verdict/artifact.",
                )
            )
        elif edge_type == "tests":
            semantic_edges.append(
                _semantic_edge(
                    source,
                    target,
                    "failed_by",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Patch or artifact is evaluated by verifier/test evidence that can fail it.",
                )
            )
        elif edge_type in {"requires_replay", "must_precede"}:
            semantic_edges.append(
                _semantic_edge(
                    target,
                    source,
                    "replayed_from",
                    spans,
                    source_edge_id=source_edge_id,
                    reason="Verification or replay state depends on a prior patch/artifact state.",
                )
            )
    return _dedupe_semantic_edges(semantic_edges)


def _semantic_edge(
    source: str,
    target: str,
    semantic_type: str,
    evidence_span_ids: list[str],
    *,
    source_edge_id: str = "",
    source_step_id: str = "",
    source_decision_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    raw = f"{source}->{semantic_type}->{target}:{source_edge_id}:{source_step_id}:{source_decision_id}"
    return {
        "semantic_edge_id": "sem_edge:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        "source": source,
        "target": target,
        "semantic_type": semantic_type,
        "source_edge_id": source_edge_id,
        "source_step_id": source_step_id,
        "source_decision_id": source_decision_id,
        "confidence": "observed" if evidence_span_ids else "planned",
        "evidence_span_ids": list(evidence_span_ids),
        "payload": {"reason": reason},
    }


def _dedupe_semantic_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for edge in edges:
        key = (
            str(edge.get("source", "") or ""),
            str(edge.get("target", "") or ""),
            str(edge.get("semantic_type", "") or ""),
            str(edge.get("source_edge_id", "") or ""),
            str(edge.get("source_step_id", "") or ""),
            str(edge.get("source_decision_id", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _count_edge_types(edges: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in edges:
        key = str(edge.get("semantic_type", "") or "")
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stage_status(output: dict[str, Any]) -> str:
    if output.get("success") is True:
        return "success"
    if output.get("success") is False:
        return "failed"
    return str(output.get("status", "") or "unknown")


def _verifier_status(record: MASDXTrajectoryRecord) -> str:
    if _verifier_success(record):
        return "success"
    return "failed"


def _verifier_success(record: MASDXTrajectoryRecord) -> bool:
    verifier = dict(record.stage_outputs.get("verifier", {}) or {})
    if verifier.get("success") is True:
        return True
    if record.verifier_evidence.get("success") is True:
        return True
    status = str(record.verifier_evidence.get("status", "") or verifier.get("status", "") or "").lower()
    return status in {"pass", "passed", "通过", "success"}


def _oracle_verifier_contradiction(record: MASDXTrajectoryRecord) -> bool:
    return bool(_verifier_success(record) and record.oracle.get("oracle_success") is False)


def _patch_summary(record: MASDXTrajectoryRecord) -> dict[str, Any]:
    patcher = dict(record.stage_outputs.get("patcher", {}) or {})
    return dict(patcher.get("patch_summary", {}) or record.diff_summary or {})


def _patch_replayability(patch_text: str) -> dict[str, Any]:
    text = str(patch_text or "")
    lines = text.splitlines()
    diff_headers = sum(1 for line in lines if line.startswith("diff --git "))
    hunk_headers = sum(1 for line in lines if line.startswith("@@ "))
    has_replayable_text_shape = bool(text.strip() and diff_headers and hunk_headers)
    return {
        "check_type": "static_text_shape",
        "has_replayable_text_shape": has_replayable_text_shape,
        "diff_header_count": diff_headers,
        "hunk_header_count": hunk_headers,
        "ends_with_newline": text.endswith("\n") if text else False,
        "requires_git_apply_check": True,
    }


def _selected_targets(record: MASDXTrajectoryRecord) -> list[str]:
    locator = dict(record.stage_outputs.get("locator", {}) or {})
    raw = locator.get("selected_target_candidates", []) or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip() for item in list(raw or []) if str(item).strip()]


def _test_targets(record: MASDXTrajectoryRecord) -> list[str]:
    targets: list[str] = []
    for command in record.commands:
        text = str(command.get("command", "") or "")
        if "pytest" not in text and "runtests" not in text:
            continue
        for token in text.split():
            cleaned = token.strip("'\"")
            if "::" in cleaned or cleaned.endswith(".py"):
                targets.append(cleaned)
    return list(dict.fromkeys(targets))


def _command_artifact_type(command: dict[str, Any]) -> str:
    text = str(command.get("command", "") or "").lower()
    if "pytest" in text or "runtests" in text:
        return "test_command"
    if text.startswith(("cat ", "sed ", "grep ", "rg ")):
        return "read_probe"
    if "apply_patch" in text or text.startswith(("python - <<", "cat >")):
        return "mutation"
    return "command"


def _all_failure_text(record: MASDXTrajectoryRecord) -> str:
    parts = [
        record.issue,
        _clip(record.verifier_evidence, 4000),
        _clip(record.stage_outputs.get("verifier", {}), 4000),
        _clip(record.oracle, 2000),
        _clip(record.commands[-20:], 5000),
    ]
    return "\n".join(parts).lower()


def _has_test_collection_blocker(record: MASDXTrajectoryRecord, text: str) -> bool:
    """Detect collection/protocol failures without treating normal test failures as blockers."""

    strong_collection_signal = _contains_any(
        text,
        (
            "error collecting",
            "failed collecting",
            "failed to collect",
            "collection failed",
            "errors during collection",
            "error during collection",
            "interrupted: 1 error during collection",
            "interrupted: 2 errors during collection",
            "collected 0 items",
            "collected 0 tests",
            "no tests ran",
            "no tests collected",
            "found no collectors",
            "empty suite",
            "not found:",
            "importerror while importing test module",
            "while importing test module",
            "import file mismatch",
            "收集失败",
            "收集错误",
            "未收集到测试",
            "没有运行测试",
            "无法完成有效验证",
        ),
    )
    if strong_collection_signal:
        return True

    if _has_pytest_collection_returncode(record):
        return True

    # Avoid false positives from ordinary test result summaries or prose that
    # happens to contain "collection"/"collect" without collection failure.
    concrete_test_failure = _contains_any(
        text,
        (
            "assertionerror",
            "failed:",
            " failed",
            " failures",
            "test failed",
            "tests failed",
            "个失败",
            "断言",
            "已运行相关测试",
            "运行相关测试",
        ),
    )
    if concrete_test_failure:
        return False

    return False


def _has_pytest_collection_returncode(record: MASDXTrajectoryRecord) -> bool:
    for command in record.commands:
        command_text = str(command.get("command", "") or "").lower()
        if "pytest" not in command_text and "runtests" not in command_text:
            continue
        try:
            returncode = int(command.get("returncode", 0) or 0)
        except (TypeError, ValueError):
            returncode = 0
        if returncode not in {4, 5}:
            continue
        command_blob = _clip(command, 1800).lower()
        if _contains_any(
            command_blob,
            (
                "error collecting",
                "failed to collect",
                "collection failed",
                "collected 0 items",
                "no tests ran",
                "no tests collected",
                "found no collectors",
                "not found:",
                "收集失败",
                "未收集到测试",
            ),
        ):
            return True
    return False


def _has_external_dependency_blocker(text: str) -> bool:
    """Detect missing external test dependencies separately from patch import regressions."""

    return _contains_any(
        text,
        (
            "no module named pytest",
            "no module named 'pytest'",
            "no module named hypothesis",
            "no module named 'hypothesis'",
            "no module named xmlschema",
            "no module named 'xmlschema'",
            "需要 pytest",
            "未安装 pytest",
            "环境中未安装 pytest",
            "缺少依赖 xmlschema",
            "缺少依赖 hypothesis",
            "unrecognized arguments: -n",
            "pytest-xdist",
            "xdist",
        ),
    )


def _has_patch_introduced_import_regression(record: MASDXTrajectoryRecord, text: str) -> bool:
    """Detect evidence that the generated patch itself broke imports/runtime.

    This is intentionally narrower than generic ImportError detection: missing
    external test dependencies remain environment/protocol blockers, while
    import errors naming changed source modules or explicit patch-regression
    language should drive fixed-localization repatching.
    """

    patch_summary = _patch_summary(record)
    changed_files = [
        str(item)
        for item in list(patch_summary.get("changed_files", []) or patch_summary.get("fresh_changed_files", []) or [])
        if str(item)
    ]
    changed_source_paths = {
        item.lower()
        for item in changed_files
        if item.endswith(".py") and "/tests/" not in item and not item.startswith("tests/")
    }
    changed_modules = {
        item[:-3].replace("/", ".")
        for item in changed_files
        if item.endswith(".py") and "/tests/" not in item and not item.startswith("tests/")
    }
    # Use only failure evidence for this detector. The broader graph text also
    # contains issue prose, command output, and diff text, which can mention the
    # changed module even when the actual import failure is elsewhere.
    failure_text = _patch_regression_failure_text(record).lower()
    explicit_patch_regression = _has_explicit_patch_regression_language(failure_text)
    import_runtime_failure = _contains_any(
        failure_text,
        (
            "importerror",
            "cannot import name",
            "nameerror",
            "导入阶段失败",
            "导入回归",
        ),
    )
    if explicit_patch_regression and (import_runtime_failure or changed_files):
        return True
    return bool(
        import_runtime_failure
        and not _has_external_dependency_blocker(failure_text)
        and (
            any(module and module.lower() in failure_text for module in changed_modules)
            or any(path and path in failure_text for path in changed_source_paths)
        )
    )


def _has_patch_added_missing_import_symbol(record: MASDXTrajectoryRecord) -> bool:
    """Detect direct causality between an added import symbol and ImportError.

    This signal is intentionally narrower than the generic patch/import
    regression detector. It requires verifier evidence to name a missing import
    symbol and patch text to add that same symbol inside an import from the same
    module, allowing relative imports to resolve through the changed file.
    """

    failure_text = _patch_regression_failure_text(record)
    missing_imports = _missing_import_symbols(failure_text)
    if not missing_imports:
        return False
    patch_text = _patch_text(record)
    if not patch_text.strip():
        return False
    patch_summary = _patch_summary(record)
    changed_source_files = [
        str(item)
        for item in list(patch_summary.get("changed_files", []) or patch_summary.get("fresh_changed_files", []) or [])
        if str(item).endswith(".py") and "/tests/" not in str(item) and not str(item).startswith("tests/")
    ]
    return any(
        _patch_adds_import_symbol(
            patch_text=patch_text,
            symbol=symbol,
            module=module,
            changed_source_files=changed_source_files,
        )
        for symbol, module in missing_imports
    )


def _missing_import_symbols(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(
        r"cannot\s+import\s+name\s+['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s+from\s+['\"]?([A-Za-z_][A-Za-z0-9_.]*)['\"]?",
        text,
        flags=re.IGNORECASE,
    ):
        pairs.append((match.group(1), match.group(2)))
    for match in re.finditer(
        r"从\s+([A-Za-z_][A-Za-z0-9_.]*)\s+导入\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
    ):
        pairs.append((match.group(2), match.group(1)))
    return _dedupe_pairs(pairs)


def _patch_adds_import_symbol(
    *,
    patch_text: str,
    symbol: str,
    module: str,
    changed_source_files: list[str],
) -> bool:
    lines = patch_text.splitlines()
    for index, raw_line in enumerate(lines):
        if not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue
        line = raw_line[1:]
        if not _contains_identifier(line, symbol):
            continue
        import_source = _nearby_import_source(lines, index)
        if not import_source:
            continue
        if _import_module_matches(
            import_source=import_source,
            failure_module=module,
            changed_source_files=changed_source_files,
        ):
            return True
    return False


def _nearby_import_source(lines: list[str], index: int) -> str:
    lower = max(0, index - 8)
    upper = min(len(lines), index + 1)
    for raw_line in reversed(lines[lower:upper]):
        line = _strip_diff_line_prefix(raw_line).strip()
        match = re.search(r"\bfrom\s+([A-Za-z_.][A-Za-z0-9_.]*)\s+import\b", line)
        if match:
            return match.group(1)
    return ""


def _import_module_matches(
    *,
    import_source: str,
    failure_module: str,
    changed_source_files: list[str],
) -> bool:
    normalized_failure = failure_module.strip().strip("'\"").lower()
    if not normalized_failure:
        return False
    candidates: set[str] = set()
    if not import_source.startswith("."):
        candidates.add(import_source.strip().lower())
    for changed_file in changed_source_files:
        resolved = _resolve_relative_import(import_source, changed_file)
        if resolved:
            candidates.add(resolved.lower())
    return normalized_failure in candidates


def _resolve_relative_import(import_source: str, changed_file: str) -> str:
    if not import_source.startswith("."):
        return import_source
    module_path = changed_file[:-3].replace("/", ".")
    package_parts = module_path.split(".")[:-1]
    leading_dots = len(import_source) - len(import_source.lstrip("."))
    suffix = import_source.lstrip(".")
    if leading_dots > 1:
        package_parts = package_parts[: max(0, len(package_parts) - leading_dots + 1)]
    if suffix:
        package_parts.extend(suffix.split("."))
    return ".".join(part for part in package_parts if part)


def _patch_text(record: MASDXTrajectoryRecord) -> str:
    parts: list[str] = []
    patcher = dict(record.stage_outputs.get("patcher", {}) or {})
    parts.append(str(patcher.get("patch", "") or ""))
    parts.extend(_shared_fact_text_values(record.shared_facts))
    return "\n".join(part for part in parts if part.strip())


def _shared_fact_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_shared_fact_text_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_shared_fact_text_values(item))
        return out
    return []


def _strip_diff_line_prefix(line: str) -> str:
    if line.startswith(("+", "-", " ")):
        return line[1:]
    return line


def _contains_identifier(text: str, identifier: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])", text))


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for symbol, module in pairs:
        item = (str(symbol or "").strip(), str(module or "").strip())
        if not item[0] or not item[1] or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _has_explicit_patch_regression_language(text: str) -> bool:
    return _contains_any(
        text,
        (
            "补丁引入",
            "直接的导入回归",
            "明显回归",
            "与本次改动相关的回归",
            "this patch introduced",
            "patch introduced",
            "introduced import regression",
        ),
    )


def _patch_regression_failure_text(record: MASDXTrajectoryRecord) -> str:
    parts = [
        _clip(record.verifier_evidence, 4000),
        _clip(record.stage_outputs.get("verifier", {}), 4000),
        _clip(record.oracle, 2000),
    ]
    return "\n".join(parts)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("\x00", "")[:limit]


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
