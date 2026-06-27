"""Failure trigger detection."""

from __future__ import annotations

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import NodeKind, TriggerDecision, TriggerType


class FailureTriggerDetector:
    """Detect the narrow set of BCMR recovery-mode triggers."""

    def detect(self, graph: ExecutionProvenanceGraph) -> TriggerDecision | None:
        verifier_nodes = graph.filter_nodes(kind=NodeKind.VERIFIER_RESULT.value)
        if not verifier_nodes:
            return self._stage_failure_trigger(graph)

        latest = verifier_nodes[-1]
        verdict = str(latest.payload.get("verdict", "")).lower()
        test_status = str(latest.payload.get("test_status", "")).lower()
        contradicted_fact_ids = list(latest.payload.get("contradicted_fact_ids", []))

        if verdict in {"promote", "pass", "success", "resolved"} and test_status == "fail":
            return TriggerDecision(
                trigger_type=TriggerType.VERIFIER_CONTRADICTION,
                trigger_node_id=latest.node_id,
                evidence_node_ids=contradicted_fact_ids,
                reason="Verifier promoted progress but tests still failed.",
            )

        if contradicted_fact_ids:
            return TriggerDecision(
                trigger_type=TriggerType.FACT_CONFLICT,
                trigger_node_id=latest.node_id,
                evidence_node_ids=contradicted_fact_ids,
                reason="A shared fact was contradicted by later verification evidence.",
            )

        if len(verifier_nodes) >= 2:
            prev = verifier_nodes[-2]
            prev_failures = tuple(prev.payload.get("failing_tests", []))
            latest_failures = tuple(latest.payload.get("failing_tests", []))
            prev_sig = str(prev.payload.get("failure_signature", ""))
            latest_sig = str(latest.payload.get("failure_signature", ""))
            if test_status == "fail" and str(prev.payload.get("test_status", "")).lower() == "fail":
                same_failures = latest_failures and latest_failures == prev_failures
                same_signature = latest_sig and latest_sig == prev_sig
                if same_failures or same_signature:
                    return TriggerDecision(
                        trigger_type=TriggerType.NO_PROGRESS_LOOP,
                        trigger_node_id=latest.node_id,
                        evidence_node_ids=[prev.node_id],
                        reason="Consecutive patch/verify cycles did not change the failure surface.",
                    )
        return self._stage_failure_trigger(graph)

    def _stage_failure_trigger(self, graph: ExecutionProvenanceGraph) -> TriggerDecision | None:
        latest_failed_stage = self._latest_failed_stage(graph)
        if latest_failed_stage is None:
            return None
        tool_evidence = self._recent_tool_calls_for_role(graph, latest_failed_stage.role, limit=3)
        repeated_commands = self._has_repeated_commands(tool_evidence)
        reason = (
            f"{latest_failed_stage.role} stage ended unsuccessfully after repeated exploration commands."
            if repeated_commands
            else f"{latest_failed_stage.role} stage ended unsuccessfully before verifier confirmation."
        )
        return TriggerDecision(
            trigger_type=TriggerType.NO_PROGRESS_LOOP,
            trigger_node_id=latest_failed_stage.node_id,
            evidence_node_ids=[node.node_id for node in tool_evidence[-2:]],
            reason=reason,
        )

    def _latest_failed_stage(self, graph: ExecutionProvenanceGraph):
        agent_nodes = graph.filter_nodes(kind=NodeKind.AGENT_STEP.value)
        for node in reversed(agent_nodes):
            if node.payload.get("success") is False:
                return node
        return None

    def _recent_tool_calls_for_role(self, graph: ExecutionProvenanceGraph, role: str, *, limit: int):
        matches = [node for node in graph.filter_nodes(kind=NodeKind.TOOL_CALL.value) if node.role == role]
        return matches[-limit:]

    def _has_repeated_commands(self, tool_nodes) -> bool:
        commands = [str(node.payload.get("command", "")).strip() for node in tool_nodes if node.payload.get("command")]
        if len(commands) < 2:
            return False
        return len(set(commands)) < len(commands)
