"""Recording helpers for BCMR provenance graphs."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any

from bcmr_swe.provenance.checkpoint_store import WorkspaceCheckpointStore
from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import CheckpointRecord, EdgeKind, NodeKind, ProvenanceEdge, ProvenanceNode


class ProvenanceRecorder:
    """Build a BCMR provenance graph and persist it alongside checkpoints."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        workspace: str | Path,
        graph: ExecutionProvenanceGraph | None = None,
        checkpoint_store: WorkspaceCheckpointStore | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace)
        self.graph = graph or ExecutionProvenanceGraph()
        self.checkpoint_store = checkpoint_store or WorkspaceCheckpointStore(self.run_dir / "checkpoints")
        self.graph_path = self.run_dir / "graph.json"
        self.checkpoints_index_path = self.run_dir / "checkpoints.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: dict[str, CheckpointRecord] = {}

    def _node_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def _edge_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def record_node(
        self,
        *,
        kind: NodeKind,
        role: str,
        phase: str,
        content: str,
        payload: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> ProvenanceNode:
        node = ProvenanceNode(
            node_id=self._node_id(kind.value.lower()),
            kind=kind,
            role=role,
            phase=phase,
            content=content,
            payload=payload or {},
        )
        self.graph.add_node(node)
        for parent_id in depends_on or []:
            self.add_edge(EdgeKind.DEPENDS_ON, parent_id, node.node_id)
        return node

    def add_edge(self, kind: EdgeKind, source_id: str, target_id: str, payload: dict[str, Any] | None = None) -> ProvenanceEdge:
        edge = ProvenanceEdge(
            edge_id=self._edge_id(kind.value),
            kind=kind,
            source_id=source_id,
            target_id=target_id,
            payload=payload or {},
        )
        return self.graph.add_edge(edge)

    def record_agent_step(
        self,
        *,
        role: str,
        phase: str,
        content: str,
        payload: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        reads: list[str] | None = None,
        writes: list[str] | None = None,
    ) -> ProvenanceNode:
        node = self.record_node(kind=NodeKind.AGENT_STEP, role=role, phase=phase, content=content, payload=payload, depends_on=depends_on)
        for node_id in reads or []:
            self.add_edge(EdgeKind.READS, node_id, node.node_id)
        for node_id in writes or []:
            self.add_edge(EdgeKind.WRITES, node.node_id, node_id)
        return node

    def record_message(
        self,
        *,
        role: str,
        phase: str,
        content: str,
        payload: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> ProvenanceNode:
        return self.record_node(kind=NodeKind.MESSAGE, role=role, phase=phase, content=content, payload=payload, depends_on=depends_on)

    def record_shared_fact(
        self,
        *,
        key: str,
        value: Any,
        role: str,
        phase: str,
        source_node_id: str = "",
        confidence: float = 0.5,
        payload: dict[str, Any] | None = None,
    ) -> ProvenanceNode:
        merged_payload = {"fact_key": key, "fact_value": value, "confidence": confidence}
        if payload:
            merged_payload.update(payload)
        node = self.record_node(
            kind=NodeKind.SHARED_FACT,
            role=role,
            phase=phase,
            content=f"{key}={value}",
            payload=merged_payload,
            depends_on=[source_node_id] if source_node_id else None,
        )
        if source_node_id:
            self.add_edge(EdgeKind.PRODUCES, source_node_id, node.node_id)
        return node

    def record_tool_call(
        self,
        *,
        role: str,
        phase: str,
        command: str,
        output: str,
        returncode: int,
        depends_on: list[str] | None = None,
        files_touched: list[str] | None = None,
    ) -> ProvenanceNode:
        payload = {
            "command": command,
            "returncode": returncode,
            "files_touched": files_touched or [],
            "output_excerpt": output[:2000],
        }
        return self.record_node(
            kind=NodeKind.TOOL_CALL,
            role=role,
            phase=phase,
            content=command,
            payload=payload,
            depends_on=depends_on,
        )

    def record_verifier_result(
        self,
        *,
        role: str,
        phase: str,
        verdict: str,
        test_status: str,
        failing_tests: list[str] | None,
        output: str,
        depends_on: list[str] | None = None,
        contradicted_fact_ids: list[str] | None = None,
        failure_signature: str = "",
    ) -> ProvenanceNode:
        payload = {
            "verdict": verdict,
            "test_status": test_status,
            "failing_tests": failing_tests or [],
            "failure_signature": failure_signature,
            "contradicted_fact_ids": contradicted_fact_ids or [],
            "output_excerpt": output[:3000],
        }
        node = self.record_node(
            kind=NodeKind.VERIFIER_RESULT,
            role=role,
            phase=phase,
            content=f"{verdict}:{test_status}",
            payload=payload,
            depends_on=depends_on,
        )
        for node_id in contradicted_fact_ids or []:
            self.add_edge(EdgeKind.VALIDATED_BY, node.node_id, node_id, payload={"result": "contradiction"})
        return node

    def create_checkpoint(
        self,
        *,
        label: str,
        metadata: dict[str, Any] | None = None,
        source_node_id: str = "",
    ) -> CheckpointRecord:
        checkpoint = self.checkpoint_store.create(
            self.workspace,
            label=label,
            metadata=metadata,
            source_node_id=source_node_id,
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        checkpoint_node = self.record_node(
            kind=NodeKind.CHECKPOINT,
            role="system",
            phase="checkpoint",
            content=label,
            payload=checkpoint.to_dict(),
            depends_on=[source_node_id] if source_node_id else None,
        )
        if source_node_id:
            self.add_edge(EdgeKind.PRODUCES, source_node_id, checkpoint_node.node_id)
        self.save()
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        return self._checkpoints.get(checkpoint_id)

    def latest_checkpoint(self) -> CheckpointRecord | None:
        if not self._checkpoints:
            return None
        values = sorted(self._checkpoints.values(), key=lambda item: item.created_at)
        return values[-1]

    def mark_fact_conflict(self, fact_node_id: str, *, reason: str, evidence_node_id: str = "") -> None:
        node = self.graph.get_node(fact_node_id)
        if node is None:
            return
        node.status = "conflicted"
        node.payload["conflict_reason"] = reason
        if evidence_node_id:
            node.payload.setdefault("conflict_evidence_node_ids", []).append(evidence_node_id)

    def save(self) -> None:
        self.graph.save(self.graph_path)
        payload = [record.to_dict() for record in sorted(self._checkpoints.values(), key=lambda item: item.created_at)]
        self.checkpoints_index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
