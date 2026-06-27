"""Recovery action enumeration."""

from __future__ import annotations

import uuid

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import ActionType, CandidateAction, FailedState, NodeKind, TriggerType


class RecoveryActionPlanner:
    """Enumerate the small, discrete BCMR action space."""

    def enumerate(self, graph: ExecutionProvenanceGraph, failed_state: FailedState) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        checkpoint = failed_state.checkpoint
        trigger = failed_state.trigger.trigger_type
        anchor_role = failed_state.suspect_region.summary.get("replay_anchor_role", "patcher")

        conflicting_facts = [
            node_id
            for node_id in failed_state.suspect_region.node_ids
            if graph.get_node(node_id) and graph.get_node(node_id).kind == NodeKind.SHARED_FACT
        ]
        for fact_node_id in conflicting_facts[:1]:
            actions.append(
                CandidateAction(
                    action_id=self._action_id(ActionType.QUARANTINE_FACT),
                    action_type=ActionType.QUARANTINE_FACT,
                    description="Quarantine the conflicting shared fact and rerun the affected downstream stage.",
                    payload={
                        "fact_node_id": fact_node_id,
                        "checkpoint_id": checkpoint.checkpoint_id if checkpoint else "",
                        "resume_from": "locator" if anchor_role == "locator" else "patcher",
                    },
                    estimated_recover_prob=0.72 if trigger == TriggerType.FACT_CONFLICT else 0.45,
                    estimated_token_cost=700.0,
                    estimated_latency_sec=25.0,
                    estimated_risk=0.12,
                )
            )

        if checkpoint:
            actions.append(
                CandidateAction(
                    action_id=self._action_id(ActionType.ROLLBACK_TO_CHECKPOINT),
                    action_type=ActionType.ROLLBACK_TO_CHECKPOINT,
                    description="Rollback the workspace to the nearest healthy checkpoint and resume from the next stage.",
                    payload={
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "resume_from": checkpoint.metadata.get("resume_from", "patcher"),
                    },
                    estimated_recover_prob=0.58 if trigger == TriggerType.VERIFIER_CONTRADICTION else 0.67,
                    estimated_token_cost=1200.0,
                    estimated_latency_sec=40.0,
                    estimated_risk=0.18,
                )
            )

        actions.append(
            CandidateAction(
                action_id=self._action_id(ActionType.REPLAY_SUBGRAPH),
                action_type=ActionType.REPLAY_SUBGRAPH,
                description="Replay only the suspect subgraph from its local anchor instead of rerunning the whole task.",
                payload={
                    "checkpoint_id": checkpoint.checkpoint_id if checkpoint else "",
                    "resume_from": anchor_role,
                    "suspect_node_ids": list(failed_state.suspect_region.node_ids),
                },
                estimated_recover_prob=0.70 if trigger == TriggerType.VERIFIER_CONTRADICTION else 0.62,
                estimated_token_cost=1800.0,
                estimated_latency_sec=65.0,
                estimated_risk=0.15,
            )
        )

        actions.append(
            CandidateAction(
                action_id=self._action_id(ActionType.INSERT_VERIFIER),
                action_type=ActionType.INSERT_VERIFIER,
                description="Insert an extra verification pass before trusting the current branch again.",
                payload={
                    "deep_verify": True,
                    "checkpoint_id": checkpoint.checkpoint_id if checkpoint else "",
                    "resume_from": "verifier",
                },
                estimated_recover_prob=0.74 if trigger == TriggerType.VERIFIER_CONTRADICTION else 0.40,
                estimated_token_cost=900.0,
                estimated_latency_sec=30.0,
                estimated_risk=0.07,
            )
        )

        actions.append(
            CandidateAction(
                action_id=self._action_id(ActionType.ESCALATE_NODE),
                action_type=ActionType.ESCALATE_NODE,
                description="Escalate the most suspect node to a stronger strategy or model setting.",
                payload={
                    "target_role": anchor_role,
                    "resume_from": anchor_role,
                    "checkpoint_id": checkpoint.checkpoint_id if checkpoint else "",
                    "escalation_level": 1,
                },
                estimated_recover_prob=0.68 if trigger == TriggerType.NO_PROGRESS_LOOP else 0.52,
                estimated_token_cost=2200.0,
                estimated_latency_sec=85.0,
                estimated_risk=0.16,
            )
        )

        return actions

    def _action_id(self, action_type: ActionType) -> str:
        return f"act_{action_type.value.lower()}_{uuid.uuid4().hex[:8]}"
