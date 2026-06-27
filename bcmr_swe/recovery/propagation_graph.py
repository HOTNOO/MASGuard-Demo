"""PARC propagation graph for cross-agent MAS recovery state.

The graph is a deterministic view over `StructuredRecoveryState`.  It makes
the MAS-specific handoff edges explicit so later recovery control can reason
about polluted objects that moved across role/stage boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcmr_swe.types import PropagationObject, StructuredRecoveryState


@dataclass(frozen=True, slots=True)
class PropagationGraphNode:
    node_id: str
    node_type: str
    payload_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "payload_summary": self.payload_summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PropagationGraphEdge:
    src: str
    dst: str
    edge_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "edge_type": self.edge_type,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class PropagationGraph:
    nodes: dict[str, PropagationGraphNode] = field(default_factory=dict)
    edges: list[PropagationGraphEdge] = field(default_factory=list)

    def add_node(self, node: PropagationGraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: PropagationGraphEdge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            return
        key = (edge.src, edge.dst, edge.edge_type)
        if any((item.src, item.dst, item.edge_type) == key for item in self.edges):
            return
        self.edges.append(edge)

    def backward_slice(self, node_id: str) -> set[str]:
        """Return all upstream nodes that can reach `node_id`."""

        visited: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.dst != current or edge.src in visited:
                    continue
                visited.add(edge.src)
                frontier.append(edge.src)
        visited.discard(node_id)
        return visited

    def forward_taint(self, node_id: str) -> set[str]:
        """Return downstream stages tainted by a suspicious object."""

        visited: set[str] = set()
        tainted_stages: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.src != current or edge.dst in visited:
                    continue
                visited.add(edge.dst)
                dst = self.nodes.get(edge.dst)
                if dst is not None:
                    stage = str(dst.metadata.get("stage", "") or dst.metadata.get("consumer_stage", "") or "")
                    if stage:
                        tainted_stages.add(stage)
                frontier.append(edge.dst)
        return tainted_stages

    def edges_by_type(self, edge_type: str) -> list[PropagationGraphEdge]:
        return [edge for edge in self.edges if edge.edge_type == edge_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
        }


def build_propagation_graph(state: StructuredRecoveryState) -> PropagationGraph:
    """Build a deterministic MAS propagation graph from structured state."""

    graph = PropagationGraph()
    role_nodes = _add_role_nodes(graph, state)
    object_nodes = _add_object_nodes(graph, state)
    anchor_nodes = _add_replay_anchor_nodes(graph, state)

    for obj in state.object_chain_view:
        object_id = _object_node_id(obj)
        producer_id = role_nodes.get(str(obj.producer_stage or ""))
        consumer_id = role_nodes.get(str(obj.consumer_stage or ""))
        if producer_id:
            graph.add_edge(
                PropagationGraphEdge(
                    src=producer_id,
                    dst=object_id,
                    edge_type="produced_by",
                    metadata={"producer_stage": obj.producer_stage},
                )
            )
        if consumer_id:
            graph.add_edge(
                PropagationGraphEdge(
                    src=object_id,
                    dst=consumer_id,
                    edge_type="consumed_by",
                    metadata={"consumer_stage": obj.consumer_stage},
                )
            )
        if obj.object_type == "SharedFact" and producer_id:
            graph.add_edge(
                PropagationGraphEdge(
                    src=producer_id,
                    dst=object_id,
                    edge_type="promoted_to_shared",
                    metadata={"fact_key": str(obj.payload.get("fact_key", "") or "")},
                )
            )
        verifier_id = _verifier_node_id(state)
        if verifier_id and obj.object_type != "VerifierVerdict":
            if _object_contradicted_by_verifier(obj, state):
                graph.add_edge(
                    PropagationGraphEdge(
                        src=object_id,
                        dst=verifier_id,
                        edge_type="contradicted_by",
                        metadata={"verifier_link": obj.verifier_link},
                    )
                )
        anchor_id = anchor_nodes.get(str(obj.replay_anchor or ""))
        if anchor_id:
            graph.add_edge(
                PropagationGraphEdge(
                    src=object_id,
                    dst=anchor_id,
                    edge_type="anchored_at",
                    metadata={"replay_anchor": obj.replay_anchor},
                )
            )
    return graph


def _add_role_nodes(graph: PropagationGraph, state: StructuredRecoveryState) -> dict[str, str]:
    role_nodes: dict[str, str] = {}
    role_names = list((state.role_aggregate_view or {}).keys())
    for obj in state.object_chain_view:
        if obj.producer_stage:
            role_names.append(obj.producer_stage)
        if obj.consumer_stage:
            role_names.append(obj.consumer_stage)
    for role in _dedupe(role_names):
        node_id = f"role::{role}"
        role_nodes[role] = node_id
        payload = dict((state.role_aggregate_view or {}).get(role, {}) or {})
        graph.add_node(
            PropagationGraphNode(
                node_id=node_id,
                node_type="RoleOutput",
                payload_summary=_summary_from_payload(payload),
                metadata={
                    "role": role,
                    "stage": role,
                    "success": bool(payload.get("success", False)),
                    "status": str(payload.get("status", "") or ""),
                },
            )
        )
    return role_nodes


def _add_object_nodes(graph: PropagationGraph, state: StructuredRecoveryState) -> dict[str, str]:
    object_nodes: dict[str, str] = {}
    for obj in state.object_chain_view:
        node_id = _object_node_id(obj)
        object_nodes[obj.object_id] = node_id
        graph.add_node(
            PropagationGraphNode(
                node_id=node_id,
                node_type=obj.object_type,
                payload_summary=_summary_from_payload(obj.payload),
                metadata={
                    "object_id": obj.object_id,
                    "producer_stage": obj.producer_stage,
                    "consumer_stage": obj.consumer_stage,
                    "stage": obj.consumer_stage or obj.producer_stage,
                    "contamination_status": obj.contamination_status,
                    "replay_anchor": obj.replay_anchor,
                },
            )
        )
    return object_nodes


def _add_replay_anchor_nodes(graph: PropagationGraph, state: StructuredRecoveryState) -> dict[str, str]:
    anchors: dict[str, str] = {}
    replay = dict(state.replay_anchor_view or {})
    for label in _dedupe(
        list(replay.get("checkpoint_labels_available", []) or [])
        + list(replay.get("healthy_anchor_candidates", []) or [])
        + list(replay.get("post_fault_checkpoint_labels", []) or [])
        + [replay.get("current_checkpoint_label", "")]
    ):
        node_id = f"anchor::{label}"
        anchors[label] = node_id
        graph.add_node(
            PropagationGraphNode(
                node_id=node_id,
                node_type="ReplayAnchor",
                payload_summary=label,
                metadata={"stage": label, "label": label},
            )
        )
    return anchors


def _object_node_id(obj: PropagationObject) -> str:
    return f"object::{obj.object_id}"


def _verifier_node_id(state: StructuredRecoveryState) -> str:
    for obj in state.object_chain_view:
        if obj.object_type == "VerifierVerdict":
            return _object_node_id(obj)
    if "verifier" in (state.role_aggregate_view or {}):
        return "role::verifier"
    return ""


def _object_contradicted_by_verifier(obj: PropagationObject, state: StructuredRecoveryState) -> bool:
    if obj.contamination_status in {"stale_or_unverified", "suspicious", "contradiction"}:
        return True
    evidence = dict(state.evidence_pack or {})
    target_legitimacy = str(evidence.get("target_legitimacy", "") or "")
    patch_legitimacy = str(evidence.get("patch_legitimacy", "") or "")
    return target_legitimacy in {"no_diff", "tests_only"} or patch_legitimacy in {"no_effective_patch"}


def _summary_from_payload(payload: dict[str, Any]) -> str:
    for key in (
        "fact_key",
        "fact_value_excerpt",
        "path",
        "verdict",
        "status",
        "patch_excerpt",
        "verification_excerpt",
        "located_files_excerpt",
    ):
        value = payload.get(key)
        if value:
            return " ".join(str(value).split())[:240]
    return ""


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
