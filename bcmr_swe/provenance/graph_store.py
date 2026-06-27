"""Execution provenance graph storage."""

from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Iterable

from bcmr_swe.types import EdgeKind, ProvenanceEdge, ProvenanceNode


class ExecutionProvenanceGraph:
    """A lightweight, JSON-serializable execution provenance graph."""

    def __init__(self):
        self.nodes: dict[str, ProvenanceNode] = {}
        self.edges: dict[str, ProvenanceEdge] = {}
        self.outgoing: dict[str, list[str]] = defaultdict(list)
        self.incoming: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: ProvenanceNode) -> ProvenanceNode:
        self.nodes[node.node_id] = node
        self.outgoing.setdefault(node.node_id, [])
        self.incoming.setdefault(node.node_id, [])
        return node

    def add_edge(self, edge: ProvenanceEdge) -> ProvenanceEdge:
        if edge.edge_id in self.edges:
            return self.edges[edge.edge_id]
        self.edges[edge.edge_id] = edge
        self.outgoing.setdefault(edge.source_id, []).append(edge.edge_id)
        self.incoming.setdefault(edge.target_id, []).append(edge.edge_id)
        return edge

    def get_node(self, node_id: str) -> ProvenanceNode | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> ProvenanceEdge | None:
        return self.edges.get(edge_id)

    def latest(self, *, kind: str | None = None, role: str | None = None) -> ProvenanceNode | None:
        candidates = self.filter_nodes(kind=kind, role=role)
        return candidates[-1] if candidates else None

    def filter_nodes(self, *, kind: str | None = None, role: str | None = None) -> list[ProvenanceNode]:
        result = list(self.nodes.values())
        if kind:
            result = [node for node in result if node.kind.value == kind]
        if role:
            result = [node for node in result if node.role == role]
        result.sort(key=lambda item: item.timestamp)
        return result

    def incoming_edges(self, node_id: str, *, kinds: Iterable[EdgeKind] | None = None) -> list[ProvenanceEdge]:
        allowed = {item.value for item in kinds} if kinds else None
        edges = [self.edges[edge_id] for edge_id in self.incoming.get(node_id, [])]
        if allowed is not None:
            edges = [edge for edge in edges if edge.kind.value in allowed]
        return edges

    def outgoing_edges(self, node_id: str, *, kinds: Iterable[EdgeKind] | None = None) -> list[ProvenanceEdge]:
        allowed = {item.value for item in kinds} if kinds else None
        edges = [self.edges[edge_id] for edge_id in self.outgoing.get(node_id, [])]
        if allowed is not None:
            edges = [edge for edge in edges if edge.kind.value in allowed]
        return edges

    def upstream_closure(self, node_ids: Iterable[str], *, max_hops: int = 3) -> set[str]:
        queue = deque((node_id, 0) for node_id in node_ids if node_id in self.nodes)
        visited: set[str] = set()
        while queue:
            current, hops = queue.popleft()
            if current in visited or hops > max_hops:
                continue
            visited.add(current)
            for edge in self.incoming_edges(current):
                queue.append((edge.source_id, hops + 1))
        return visited

    def edge_ids_between(self, node_ids: Iterable[str]) -> list[str]:
        node_set = set(node_ids)
        return [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.source_id in node_set and edge.target_id in node_set
        ]

    def subgraph(self, node_ids: Iterable[str]) -> dict[str, object]:
        node_list = list(dict.fromkeys(node_ids))
        return {
            "nodes": [self.nodes[node_id].to_dict() for node_id in node_list if node_id in self.nodes],
            "edges": [self.edges[edge_id].to_dict() for edge_id in self.edge_ids_between(node_list)],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
