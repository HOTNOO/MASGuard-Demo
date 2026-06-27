"""Encode failed states into numeric and categorical features."""

from __future__ import annotations

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import FailedState, NodeKind, StateFeatures


class StateEncoder:
    """Build a compact feature view over the current failed state."""

    def encode(self, graph: ExecutionProvenanceGraph, failed_state: FailedState) -> StateFeatures:
        tool_calls = graph.filter_nodes(kind=NodeKind.TOOL_CALL.value)
        verifier_nodes = graph.filter_nodes(kind=NodeKind.VERIFIER_RESULT.value)
        shared_facts = graph.filter_nodes(kind=NodeKind.SHARED_FACT.value)
        conflicting_facts = [node for node in shared_facts if node.status == "conflicted"]

        latest_verifier = verifier_nodes[-1] if verifier_nodes else None
        failing_tests = latest_verifier.payload.get("failing_tests", []) if latest_verifier else []

        suspect_roles = failed_state.suspect_region.summary.get("role_chain", [])
        numeric = {
            "n_graph_nodes": float(len(graph.nodes)),
            "n_graph_edges": float(len(graph.edges)),
            "n_tool_calls": float(len(tool_calls)),
            "n_verifier_runs": float(len(verifier_nodes)),
            "n_conflicting_facts": float(len(conflicting_facts)),
            "suspect_region_size": float(len(failed_state.suspect_region.node_ids)),
            "failing_tests_count": float(len(failing_tests)),
            "checkpoint_depth": float(failed_state.metadata.get("checkpoint_depth", 0.0)),
            "recovery_invocations": float(failed_state.metadata.get("recovery_invocations", 0.0)),
            "suspect_has_locator": 1.0 if "locator" in suspect_roles else 0.0,
            "suspect_has_patcher": 1.0 if any(role in {"patcher", "implementer"} for role in suspect_roles) else 0.0,
        }
        categorical = {
            "trigger_type": failed_state.trigger.trigger_type.value,
            "current_phase": latest_verifier.phase if latest_verifier else "verify",
            "replay_anchor_role": failed_state.suspect_region.summary.get("replay_anchor_role", "patcher"),
        }
        return StateFeatures(numeric=numeric, categorical=categorical)
