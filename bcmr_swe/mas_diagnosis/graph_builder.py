"""Build MAS-DX evidence graphs from normalized trajectory records."""

from __future__ import annotations

from typing import Any

from bcmr_swe.mas_diagnosis.schema import (
    MASDXEvidenceGraph,
    MASDXGraphEdge,
    MASDXGraphNode,
    MASDXTrajectoryRecord,
)


STAGE_CHAIN = ("locator", "patcher", "verifier")


def build_evidence_graph(record: MASDXTrajectoryRecord) -> MASDXEvidenceGraph:
    nodes: list[MASDXGraphNode] = []
    edges: list[MASDXGraphEdge] = []
    evidence_spans: dict[str, str] = {}

    stage_node_ids: dict[str, str] = {}
    for stage, output in record.stage_outputs.items():
        node_id = f"agent:{stage}"
        stage_node_ids[stage] = node_id
        nodes.append(
            MASDXGraphNode(
                node_id=node_id,
                node_type="agent_stage",
                label=stage,
                stage=stage,
                payload=_stage_payload_summary(output),
            )
        )
        span_id = f"span:stage:{stage}"
        evidence_spans[span_id] = _clip_text(output, 1200)
        edges.append(
            MASDXGraphEdge(
                source=node_id,
                target=node_id,
                edge_type="has_stage_evidence",
                evidence_span_ids=[span_id],
            )
        )

    for source, target in zip(STAGE_CHAIN, STAGE_CHAIN[1:]):
        if source in stage_node_ids and target in stage_node_ids:
            edges.append(
                MASDXGraphEdge(
                    source=stage_node_ids[source],
                    target=stage_node_ids[target],
                    edge_type="handoff",
                )
            )

    shared_nodes = _shared_fact_nodes(record.shared_facts)
    nodes.extend(shared_nodes)
    for node in shared_nodes:
        if "locator" in stage_node_ids:
            edges.append(
                MASDXGraphEdge(
                    source=stage_node_ids["locator"],
                    target=node.node_id,
                    edge_type="writes_shared_fact",
                )
            )
        if "patcher" in stage_node_ids:
            edges.append(
                MASDXGraphEdge(
                    source=node.node_id,
                    target=stage_node_ids["patcher"],
                    edge_type="read_by_downstream_agent",
                )
            )

    patch_node = _patch_artifact_node(record)
    if patch_node is not None:
        nodes.append(patch_node)
        if "patcher" in stage_node_ids:
            edges.append(
                MASDXGraphEdge(
                    source=stage_node_ids["patcher"],
                    target=patch_node.node_id,
                    edge_type="produces_artifact",
                )
            )
        if "verifier" in stage_node_ids:
            edge_type = "contradicted_by_verifier" if _verifier_failed(record) else "validated_by_verifier"
            edges.append(
                MASDXGraphEdge(
                    source=patch_node.node_id,
                    target=stage_node_ids["verifier"],
                    edge_type=edge_type,
                    evidence_span_ids=["span:verifier"],
                )
            )

    if record.verifier_evidence:
        evidence_spans["span:verifier"] = _clip_text(record.verifier_evidence, 1600)
    if record.diff_summary:
        evidence_spans["span:diff"] = _clip_text(record.diff_summary, 1200)
    if record.oracle:
        evidence_spans["span:oracle"] = _clip_text(record.oracle, 1200)

    return MASDXEvidenceGraph(
        case_id=record.case_id,
        nodes=nodes,
        edges=edges,
        evidence_spans=evidence_spans,
        metadata={
            "instance_id": record.instance_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "has_cross_agent_handoff": any(edge.edge_type == "handoff" for edge in edges),
            "has_verifier_contradiction": any(edge.edge_type == "contradicted_by_verifier" for edge in edges),
        },
    )


def _stage_payload_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": output.get("success"),
        "status": output.get("status", ""),
        "stop_reason": output.get("stop_reason", ""),
        "command_count": len(list(output.get("commands", []) or [])),
        "has_messages": bool(output.get("messages") or output.get("planner_messages")),
    }


def _shared_fact_nodes(shared_facts: dict[str, Any]) -> list[MASDXGraphNode]:
    nodes: list[MASDXGraphNode] = []
    for key, value in sorted(shared_facts.items()):
        fact = dict(value or {}) if isinstance(value, dict) else {"value": value}
        node_id = str(fact.get("node_id", "") or f"shared_fact:{key}")
        nodes.append(
            MASDXGraphNode(
                node_id=node_id,
                node_type="shared_fact",
                label=str(key),
                payload={"value_excerpt": _clip_text(fact.get("value", ""), 300)},
            )
        )
    return nodes


def _patch_artifact_node(record: MASDXTrajectoryRecord) -> MASDXGraphNode | None:
    patcher = dict(record.stage_outputs.get("patcher", {}) or {})
    patch_summary = dict(patcher.get("patch_summary", {}) or record.diff_summary or {})
    patch = str(patcher.get("patch", "") or "")
    if not patch and not patch_summary:
        return None
    return MASDXGraphNode(
        node_id="artifact:patch",
        node_type="artifact",
        label="patch",
        stage="patcher",
        payload={
            "patch_excerpt": patch[:500],
            "changed_files": list(patch_summary.get("changed_files", []) or []),
            "failure_mode": str(patch_summary.get("failure_mode", "") or ""),
        },
    )


def _verifier_failed(record: MASDXTrajectoryRecord) -> bool:
    verifier = dict(record.stage_outputs.get("verifier", {}) or {})
    if verifier and verifier.get("success") is False:
        return True
    if record.oracle.get("oracle_success") is False:
        return True
    return False


def _clip_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\x00", "")
    return text[:limit]
