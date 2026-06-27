"""Controller bridge from CFR frontier decisions to RecoveryPrograms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from bcmr_swe.recovery.contamination_frontier import (
    ContaminationFrontier,
    compute_contamination_frontier,
)
from bcmr_swe.recovery.propagation_graph import PropagationGraph, build_propagation_graph
from bcmr_swe.types import OpType, RecoveryProgram, RecoveryStep, StructuredRecoveryState


DEFAULT_MIN_REPLAY_CONFIDENCE = 0.42


@dataclass(frozen=True, slots=True)
class FrontierControllerDecision:
    """CFR controller output: frontier plus executable or probe program."""

    frontier: ContaminationFrontier
    program: RecoveryProgram
    decision_type: str
    reason: str
    fallback_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier": self.frontier.to_dict(),
            "program": self.program.to_dict(),
            "decision_type": self.decision_type,
            "reason": self.reason,
            "fallback_required": self.fallback_required,
            "metadata": dict(self.metadata),
        }


def decide_frontier_recovery(
    state: StructuredRecoveryState,
    *,
    graph: PropagationGraph | None = None,
    evidence: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    min_replay_confidence: float = DEFAULT_MIN_REPLAY_CONFIDENCE,
) -> FrontierControllerDecision:
    """Compute CFR frontier and compile it into an executable program.

    High-confidence frontiers compile to `SELECTIVE_REPLAY`. Low-confidence or
    empty frontiers compile to a cheap `INSPECT` probe program, preserving the
    existing bounded/contract fallback path for the caller.
    """

    graph = graph or build_propagation_graph(state)
    frontier = compute_contamination_frontier(
        state,
        graph=graph,
        evidence=evidence,
        budget=budget,
    )
    if frontier.invalidated_object_ids and frontier.confidence >= min_replay_confidence and frontier.replay_start_stage:
        program = frontier_to_recovery_program(frontier)
        return FrontierControllerDecision(
            frontier=frontier,
            program=program,
            decision_type="selective_replay",
            reason="cfr_frontier_confident",
            fallback_required=False,
            metadata={"min_replay_confidence": min_replay_confidence},
        )

    program = frontier_to_probe_program(frontier)
    return FrontierControllerDecision(
        frontier=frontier,
        program=program,
        decision_type="cheap_revalidation",
        reason="cfr_frontier_low_confidence",
        fallback_required=True,
        metadata={"min_replay_confidence": min_replay_confidence},
    )


def frontier_to_recovery_program(frontier: ContaminationFrontier) -> RecoveryProgram:
    """Compile a confident CFR frontier into a selective replay program."""

    role = str(frontier.replay_start_stage or "").strip().lower()
    replay_contract = _frontier_replay_contract(frontier)
    context_hint = _frontier_context_hint(frontier, replay_contract)
    return RecoveryProgram(
        program_id=f"cfr_selective_replay_{role or 'unknown'}",
        steps=[
            RecoveryStep(
                op=OpType.SELECTIVE_REPLAY,
                args={
                    "role": role,
                    "cache_upstream": bool(frontier.cache_upstream),
                    "scope": role,
                    "context_hint": context_hint,
                    "invalidated_object_ids": list(frontier.invalidated_object_ids),
                    "frontier_stage_boundaries": [
                        [producer, consumer]
                        for producer, consumer in frontier.frontier_stage_boundaries
                    ],
                    "clean_upstream_object_ids": list(frontier.clean_upstream_object_ids),
                    "negative_facts": list(frontier.evidence.negative_constraints),
                    "replay_contract": replay_contract,
                },
            )
        ],
        rationale=(
            "CFR invalidates the minimal contaminated object frontier and "
            "selectively replays only the earliest contaminated producer role."
        ),
        estimated_total_cost=frontier.estimated_replay_cost,
        estimated_recover_prob=frontier.confidence,
        estimated_risk=frontier.under_invalidation_risk,
        metadata={
            "family": "contamination_frontier",
            "strategy": "cfr_selective_replay_v0",
            "frontier": frontier.to_dict(),
            "invalidated_object_ids": list(frontier.invalidated_object_ids),
            "frontier_stage_boundaries": [
                [producer, consumer]
                for producer, consumer in frontier.frontier_stage_boundaries
            ],
            "replay_span": frontier.estimated_replay_span,
            "clean_work_preservation_ids": list(frontier.clean_upstream_object_ids),
            "replay_contract": replay_contract,
        },
    )


def frontier_to_probe_program(frontier: ContaminationFrontier) -> RecoveryProgram:
    """Compile a low-confidence CFR frontier into cheap revalidation probes."""

    probes = list(frontier.cheap_revalidation_probes)
    if probes:
        first_probe = probes[0]
        target = str(first_probe.get("target", "") or "test_output")
        depth = "quick" if first_probe.get("probe_type") != "focused_test" else "deep"
    else:
        target = "test_output"
        depth = "quick"
    return RecoveryProgram(
        program_id="cfr_cheap_revalidation_probe",
        steps=[
            RecoveryStep(
                op=OpType.INSPECT,
                args={
                    "target": target,
                    "depth": depth,
                    "context_hint": _frontier_context_hint(frontier),
                    "cheap_revalidation_probes": probes,
                },
            )
        ],
        rationale=(
            "CFR did not find a high-confidence minimal frontier; perform a "
            "cheap evidence probe before falling back to bounded recovery."
        ),
        estimated_total_cost=120.0,
        estimated_recover_prob=max(0.05, frontier.confidence * 0.5),
        estimated_risk=max(0.2, frontier.under_invalidation_risk),
        metadata={
            "family": "contamination_frontier_probe",
            "strategy": "cfr_cheap_revalidation_v0",
            "frontier": frontier.to_dict(),
            "fallback_required": True,
        },
    )


def attach_frontier_audit_to_row(row: dict[str, Any], decision: FrontierControllerDecision) -> dict[str, Any]:
    """Add compact CFR metrics to an experiment row or cell payload."""

    row = dict(row)
    frontier = decision.frontier
    row.update(
        {
            "cfr_decision_type": decision.decision_type,
            "cfr_fallback_required": bool(decision.fallback_required),
            "cfr_invalidated_object_ids": list(frontier.invalidated_object_ids),
            "cfr_frontier_stage_boundaries": [
                [producer, consumer]
                for producer, consumer in frontier.frontier_stage_boundaries
            ],
            "cfr_replay_start_stage": frontier.replay_start_stage,
            "cfr_confidence": frontier.confidence,
            "cfr_estimated_replay_span": frontier.estimated_replay_span,
            "cfr_clean_upstream_object_ids": list(frontier.clean_upstream_object_ids),
            "cfr_under_invalidation_risk": frontier.under_invalidation_risk,
            "cfr_audit_reasons": list(frontier.audit_reasons),
        }
    )
    return row


def _frontier_context_hint(
    frontier: ContaminationFrontier,
    replay_contract: dict[str, Any] | None = None,
) -> str:
    invalidated = ", ".join(frontier.invalidated_object_ids) or "none"
    clean = ", ".join(frontier.clean_upstream_object_ids) or "none"
    negatives = " | ".join(frontier.evidence.negative_constraints) or "none"
    boundaries = ", ".join(
        f"{producer}->{consumer}"
        for producer, consumer in frontier.frontier_stage_boundaries
    ) or "none"
    contract = dict(replay_contract or {})
    preferred = ", ".join(list(contract.get("preferred_source_paths", []) or [])[:5]) or "none"
    readonly = ", ".join(list(contract.get("read_only_evidence_paths", []) or [])[:5]) or "none"
    return (
        "CFR contamination frontier recovery. "
        f"Invalidate propagated objects: {invalidated}. "
        f"Frontier boundaries: {boundaries}. "
        f"Preserve clean upstream objects: {clean}. "
        f"Preferred source edit paths: {preferred}. "
        f"Read-only evidence/test paths: {readonly}. "
        f"Patch scope policy: {contract.get('patch_scope_policy', 'source_only_preferred')}. "
        f"Do not consume invalidated objects or repeat negative facts: {negatives}."
    )


def _frontier_replay_contract(frontier: ContaminationFrontier) -> dict[str, Any]:
    raw = dict(frontier.evidence.raw or {})
    failing_tests = _string_list(raw.get("failing_tests"))
    evidence_paths = _dedupe_strs(list(frontier.evidence.paths))
    selected_paths = _string_list(raw.get("selected_target_candidates"))
    patcher_paths = _string_list(raw.get("patcher_modified_files"))
    verifier_failed_paths = _string_list(raw.get("verifier_failed_files"))
    test_paths_from_ids = [
        _path_from_test_id(item)
        for item in failing_tests
        if _path_from_test_id(item)
    ]
    preferred_source_paths = _dedupe_strs(
        path
        for path in patcher_paths + selected_paths + evidence_paths
        if _is_source_path(path)
    )[:8]
    read_only_evidence_paths = _dedupe_strs(
        path
        for path in verifier_failed_paths + selected_paths + evidence_paths + test_paths_from_ids
        if _is_test_or_generated_path(path)
    )[:8]
    verifier_excerpt = str(
        raw.get("verifier_excerpt")
        or raw.get("review_verifier_excerpt")
        or ""
    ).strip()
    return {
        "contract_version": "cfr_replay_contract_v1",
        "replay_start_stage": frontier.replay_start_stage,
        "patch_scope_policy": "source_only_preferred",
        "invalidated_object_ids": list(frontier.invalidated_object_ids),
        "must_not_consume_object_ids": list(frontier.invalidated_object_ids),
        "clean_upstream_object_ids": list(frontier.clean_upstream_object_ids),
        "must_preserve_object_ids": list(frontier.clean_upstream_object_ids),
        "tainted_downstream_object_ids": list(frontier.tainted_downstream_object_ids),
        "frontier_stage_boundaries": [
            [producer, consumer]
            for producer, consumer in frontier.frontier_stage_boundaries
        ],
        "preferred_source_paths": preferred_source_paths,
        "read_only_evidence_paths": read_only_evidence_paths,
        "all_evidence_paths": evidence_paths[:12],
        "failing_tests": failing_tests[:8],
        "negative_facts": list(frontier.evidence.negative_constraints),
        "verifier_excerpt": verifier_excerpt[:1200],
        "forbidden_edit_patterns": [
            "tests/**",
            "**/tests/**",
            "build/**",
            "dist/**",
            "generated/**",
        ],
        "success_criteria": [
            "produce a source-code diff touching a preferred source path when one is available",
            "do not edit read-only evidence or test paths",
            "do not reuse invalidated objects as authoritative facts",
            "run focused verification or use the verifier failure text before declaring success",
            "if verification still fails, revise the source diff instead of repeating the polluted patch belief",
        ],
        "audit_reasons": list(frontier.audit_reasons[:6]),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    return _dedupe_strs(str(item).replace("\\", "/") for item in raw_values if str(item or "").strip())


def _dedupe_strs(values: Iterable[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip().replace("\\", "/")
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _path_from_test_id(test_id: str) -> str:
    text = str(test_id or "").strip().replace("\\", "/")
    if not text:
        return ""
    path = text.split("::", 1)[0]
    return path if path.endswith(".py") else ""


def _is_source_path(path: str) -> bool:
    text = str(path or "").strip().replace("\\", "/")
    return (
        text.endswith(".py")
        and "/" in text
        and not text.startswith("/")
        and not _is_test_or_generated_path(text)
    )


def _is_test_or_generated_path(path: str) -> bool:
    text = str(path or "").strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    basename = parts[-1] if parts else text
    return (
        text.startswith("tests/")
        or "/tests/" in text
        or basename.startswith("test_")
        or text.startswith("build/")
        or text.startswith("dist/")
        or "/generated/" in text
        or text.startswith("generated/")
    )
