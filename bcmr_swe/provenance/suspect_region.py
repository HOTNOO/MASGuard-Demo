"""Suspect region extraction for failed states."""

from __future__ import annotations

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import NodeKind, SuspectRegion, TriggerDecision, TriggerType


class SuspectRegionExtractor:
    """Recover a small, replayable local graph slice around a failure trigger."""

    def extract(self, graph: ExecutionProvenanceGraph, trigger: TriggerDecision) -> SuspectRegion:
        region_node_ids: list[str] = []
        if trigger.trigger_node_id:
            region_node_ids.append(trigger.trigger_node_id)
        region_node_ids.extend(trigger.evidence_node_ids)

        verifier_node = graph.get_node(trigger.trigger_node_id) or graph.latest(kind=NodeKind.VERIFIER_RESULT.value)
        if verifier_node and verifier_node.node_id not in region_node_ids:
            region_node_ids.append(verifier_node.node_id)

        patch_node_id = self._nearest_upstream_role(
            graph,
            trigger.trigger_node_id,
            roles={"patcher", "implementer"},
        )
        if patch_node_id:
            region_node_ids.append(patch_node_id)

        conflicting_facts = self._collect_conflicting_facts(graph, trigger)
        region_node_ids.extend(conflicting_facts)

        needs_locator = trigger.trigger_type in {TriggerType.NO_PROGRESS_LOOP, TriggerType.FACT_CONFLICT}
        if needs_locator or not patch_node_id:
            locator_node_id = self._nearest_upstream_role(graph, trigger.trigger_node_id, roles={"locator"})
            if locator_node_id:
                region_node_ids.append(locator_node_id)

        ordered_node_ids = list(dict.fromkeys(node_id for node_id in region_node_ids if node_id))
        edge_ids = graph.edge_ids_between(ordered_node_ids)

        roles = []
        for node_id in ordered_node_ids:
            node = graph.get_node(node_id)
            if node and node.role not in roles:
                roles.append(node.role)

        return SuspectRegion(
            node_ids=ordered_node_ids,
            edge_ids=edge_ids,
            summary={
                "trigger_type": trigger.trigger_type.value,
                "role_chain": roles,
                "size": len(ordered_node_ids),
                "has_conflicting_fact": bool(conflicting_facts),
                "replay_anchor_role": roles[0] if roles else "patcher",
            },
        )

    def _nearest_upstream_role(self, graph: ExecutionProvenanceGraph, start_node_id: str, *, roles: set[str]) -> str:
        if not start_node_id:
            return ""
        visited = graph.upstream_closure([start_node_id], max_hops=4)
        candidates = []
        for node_id in visited:
            node = graph.get_node(node_id)
            if node and node.kind == NodeKind.AGENT_STEP and node.role in roles:
                candidates.append(node)
        candidates.sort(key=lambda item: item.timestamp, reverse=True)
        return candidates[0].node_id if candidates else ""

    def _collect_conflicting_facts(self, graph: ExecutionProvenanceGraph, trigger: TriggerDecision) -> list[str]:
        verifier_node = graph.get_node(trigger.trigger_node_id)
        payload = verifier_node.payload if verifier_node else {}
        ids = list(payload.get("contradicted_fact_ids", []))
        if ids:
            return ids
        return [
            node.node_id
            for node in graph.filter_nodes(kind=NodeKind.SHARED_FACT.value)
            if node.status == "conflicted"
        ][-2:]
