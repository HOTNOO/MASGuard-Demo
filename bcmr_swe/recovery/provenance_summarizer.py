"""Convert execution provenance graph into structured text for LLM consumption."""

from __future__ import annotations

from typing import Any

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.types import (
    CandidateAction,
    FailedState,
    NodeKind,
    RecoveryBudget,
)


class ProvenanceSummarizer:
    """Produce a concise, structured natural-language summary of a failed state.

    The output is designed to fit within 800-1500 tokens so that an LLM can
    reason about candidate recovery actions without seeing the full raw graph.
    """

    def summarize(
        self,
        graph: ExecutionProvenanceGraph,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        *,
        used_recovery_calls: int = 0,
        used_tokens: float = 0.0,
    ) -> str:
        sections = [
            self._failed_state_section(failed_state, used_tokens),
            self._checkpoint_candidates_section(failed_state),
            self._suspect_region_section(graph, failed_state),
            self._dependency_chain_section(graph, failed_state),
            self._recent_verification_section(graph),
            self._candidate_actions_section(actions),
            self._budget_section(budget, used_recovery_calls, used_tokens),
        ]
        return "\n\n".join(section for section in sections if section)

    def _failed_state_section(self, fs: FailedState, used_tokens: float) -> str:
        trigger = fs.trigger
        numerics = fs.state_features.numeric
        cats = fs.state_features.categorical
        lines = [
            "## Failed State Summary",
            f"- Instance: {fs.instance_id}",
            f"- Trigger: {trigger.trigger_type.value}",
            f"- Trigger reason: {trigger.reason}",
            f"- Current phase: {cats.get('current_phase', 'unknown')}",
            f"- Steps executed: {int(numerics.get('n_graph_nodes', 0))}",
            f"- Tool calls so far: {int(numerics.get('n_tool_calls', 0))}",
            f"- Verifier runs: {int(numerics.get('n_verifier_runs', 0))}",
            f"- Tokens consumed: {int(used_tokens)}",
            f"- Failing tests: {int(numerics.get('failing_tests_count', 0))}",
            f"- Conflicting facts: {int(numerics.get('n_conflicting_facts', 0))}",
        ]
        return "\n".join(lines)

    def _checkpoint_candidates_section(self, fs: FailedState) -> str:
        candidates = fs.metadata.get("checkpoint_candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return ""

        lines = ["## Recovery Checkpoint Candidates"]
        for candidate in candidates[:8]:
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("label", "")).strip() or "unknown"
            checkpoint_id = str(candidate.get("checkpoint_id", "")).strip() or "unknown"
            metadata = candidate.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            resume_from = str(metadata.get("resume_from", "")).strip() or "unknown"
            stage = str(metadata.get("stage", "")).strip() or "unknown"
            anchor_health = str(metadata.get("anchor_health", "")).strip()
            touched_paths = metadata.get("touched_paths", [])
            if isinstance(touched_paths, list):
                touched = ", ".join(str(path) for path in touched_paths[:3])
            else:
                touched = str(touched_paths)

            description = [
                f"- {label}: checkpoint_id={checkpoint_id}",
                f"resume_from={resume_from}",
                f"stage={stage}",
            ]
            if anchor_health:
                description.append(f"anchor_health={anchor_health}")
            if touched:
                description.append(f"touched_paths={touched}")
            lines.append(" | ".join(description))

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _suspect_region_section(
        self, graph: ExecutionProvenanceGraph, fs: FailedState
    ) -> str:
        region = fs.suspect_region
        summary = region.summary
        role_chain = summary.get("role_chain", [])
        chain_str = " → ".join(role_chain) if role_chain else "unknown"

        conflicting_facts = self._collect_conflicting_fact_descriptions(graph, region.node_ids)

        lines = [
            "## Suspect Region",
            f"- Scope: {chain_str}",
            f"- Size: {summary.get('size', len(region.node_ids))} nodes, {len(region.edge_ids)} edges",
            f"- Replay anchor: {summary.get('replay_anchor_role', 'patcher')}",
            f"- Has conflicting fact: {summary.get('has_conflicting_fact', False)}",
        ]
        if conflicting_facts:
            lines.append("- Conflicting facts:")
            for desc in conflicting_facts[:3]:
                lines.append(f"  * {desc}")
        return "\n".join(lines)

    def _dependency_chain_section(
        self, graph: ExecutionProvenanceGraph, fs: FailedState
    ) -> str:
        lines = ["## Dependency Chain"]
        region_nodes = [
            graph.get_node(nid) for nid in fs.suspect_region.node_ids
        ]
        region_nodes = [n for n in region_nodes if n is not None]
        region_nodes.sort(key=lambda n: n.timestamp)

        for node in region_nodes:
            kind_label = node.kind.value
            role = node.role
            status = node.status

            detail = self._node_one_liner(node)
            lines.append(f"  {role}[{kind_label}] status={status} | {detail}")

            for edge_id in fs.suspect_region.edge_ids:
                edge = graph.get_edge(edge_id)
                if edge and edge.source_id == node.node_id:
                    target = graph.get_node(edge.target_id)
                    if target:
                        lines.append(
                            f"    --{edge.kind.value}--> {target.role}[{target.kind.value}]"
                        )

        if len(lines) == 1:
            lines.append("  (no nodes in suspect region)")
        return "\n".join(lines)

    def _recent_verification_section(self, graph: ExecutionProvenanceGraph) -> str:
        verifier_nodes = graph.filter_nodes(kind=NodeKind.VERIFIER_RESULT.value)
        if not verifier_nodes:
            return "## Recent Verification\n- No verification runs recorded."

        latest = verifier_nodes[-1]
        payload = latest.payload
        verdict = payload.get("verdict", "unknown")
        test_status = payload.get("test_status", "unknown")
        failing = payload.get("failing_tests", [])
        contradicted = payload.get("contradicted_fact_ids", [])

        changed = "unknown"
        if len(verifier_nodes) >= 2:
            prev_failing = set(verifier_nodes[-2].payload.get("failing_tests", []))
            curr_failing = set(failing)
            if curr_failing == prev_failing:
                changed = "no (same failures)"
            elif len(curr_failing) < len(prev_failing):
                changed = "yes (fewer failures)"
            else:
                changed = "yes (different failures)"

        lines = [
            "## Recent Verification",
            f"- Verdict: {verdict}",
            f"- Test status: {test_status}",
            f"- Failing tests ({len(failing)}): {', '.join(failing[:5])}{'...' if len(failing) > 5 else ''}",
            f"- Contradicted facts: {len(contradicted)}",
            f"- Changed since last round: {changed}",
        ]
        return "\n".join(lines)

    def _candidate_actions_section(self, actions: list[CandidateAction]) -> str:
        lines = ["## Candidate Recovery Actions"]
        for idx, action in enumerate(actions, 1):
            resume = action.payload.get("resume_from", "?")
            lines.append(
                f"[A{idx}] {action.action_type.value} — {action.description}"
            )
            lines.append(
                f"     Resume from: {resume} | "
                f"Est. cost: {int(action.estimated_token_cost)} tokens, "
                f"{action.estimated_latency_sec:.0f}s | "
                f"Est. risk: {action.estimated_risk:.2f}"
            )
        return "\n".join(lines)

    def _budget_section(
        self,
        budget: RecoveryBudget,
        used_calls: int,
        used_tokens: float,
    ) -> str:
        remaining_tokens = max(0, budget.token_budget - used_tokens)
        remaining_calls = max(0, budget.max_recovery_calls - used_calls)
        lines = [
            "## Budget Constraints",
            f"- Token budget: {int(budget.token_budget)} (remaining ≈ {int(remaining_tokens)})",
            f"- Max recovery calls: {budget.max_recovery_calls} (remaining: {remaining_calls})",
            f"- Latency budget: {budget.latency_budget_sec:.0f}s",
            f"- Preference weights: λ_token={budget.lambda_token}, λ_latency={budget.lambda_latency}, λ_risk={budget.lambda_risk}",
        ]
        return "\n".join(lines)

    def _collect_conflicting_fact_descriptions(
        self, graph: ExecutionProvenanceGraph, node_ids: list[str]
    ) -> list[str]:
        descriptions: list[str] = []
        for nid in node_ids:
            node = graph.get_node(nid)
            if node is None or node.kind != NodeKind.SHARED_FACT:
                continue
            if node.status not in ("conflicted", "quarantined"):
                continue
            key = node.payload.get("fact_key", "?")
            value = str(node.payload.get("fact_value", "?"))[:120]
            reason = node.payload.get("conflict_reason", "")
            desc = f"{key}={value}"
            if reason:
                desc += f" ({reason[:80]})"
            descriptions.append(desc)
        return descriptions

    def _node_one_liner(self, node: Any) -> str:
        payload = node.payload if isinstance(node.payload, dict) else {}

        if node.kind == NodeKind.VERIFIER_RESULT:
            verdict = payload.get("verdict", "?")
            test_status = payload.get("test_status", "?")
            n_failing = len(payload.get("failing_tests", []))
            return f"verdict={verdict}, test_status={test_status}, failing={n_failing}"

        if node.kind == NodeKind.SHARED_FACT:
            key = payload.get("fact_key", "?")
            value = str(payload.get("fact_value", "?"))[:60]
            conf = payload.get("confidence", "?")
            return f"fact: {key}={value} (confidence={conf})"

        if node.kind == NodeKind.TOOL_CALL:
            cmd = str(payload.get("command", "?"))[:80]
            rc = payload.get("returncode", "?")
            return f"cmd: {cmd} (rc={rc})"

        if node.kind == NodeKind.AGENT_STEP:
            success = payload.get("success", "?")
            return f"success={success}"

        if node.kind == NodeKind.CHECKPOINT:
            label = payload.get("label", "?")
            return f"checkpoint: {label}"

        content = str(node.content)[:80]
        return content
