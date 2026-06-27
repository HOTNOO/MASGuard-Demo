"""Contamination Frontier Recovery (CFR) core.

This module is the deterministic v0 implementation of the CFR pivot: given a
failed MAS recovery state and a propagation graph, score propagated objects,
choose a minimal invalidation frontier, and expose an auditable selective
replay plan surface.  It deliberately does not call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable

from bcmr_swe.recovery.object_lifecycle import lifecycle_state_for_payload
from bcmr_swe.recovery.propagation_graph import (
    PropagationGraph,
    PropagationGraphNode,
    build_propagation_graph,
)
from bcmr_swe.types import PropagationObject, StructuredRecoveryState


STAGE_ORDER = ("locator", "planner", "critic", "patcher", "implementer", "verifier", "bcmr")
CONTAMINATION_WEIGHTS = {
    "contradiction": 80.0,
    "stale_or_unverified": 45.0,
    "suspicious": 28.0,
    "unknown": 4.0,
}
LIFECYCLE_WEIGHTS = {
    "invalidated": 32.0,
    "suspicious": 12.0,
    "exhausted": -28.0,
    "resolved": -80.0,
}


@dataclass(frozen=True, slots=True)
class ContaminationEvidence:
    """Compact deterministic evidence packet used by CFR scoring."""

    raw: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    tokens: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    failed_stages: set[str] = field(default_factory=set)
    target_node_ids: set[str] = field(default_factory=set)
    negative_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": dict(self.raw),
            "text_excerpt": self.text[:1000],
            "tokens": sorted(self.tokens),
            "paths": sorted(self.paths),
            "failed_stages": sorted(self.failed_stages),
            "target_node_ids": sorted(self.target_node_ids),
            "negative_constraints": list(self.negative_constraints),
        }


@dataclass(frozen=True, slots=True)
class ContaminantScore:
    """Score for one propagated object candidate."""

    object_id: str
    node_id: str
    object_type: str
    producer_stage: str
    consumer_stage: str
    raw_score: float
    adjusted_score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    forward_tainted_stages: list[str] = field(default_factory=list)
    forward_tainted_object_ids: list[str] = field(default_factory=list)
    upstream_node_ids: list[str] = field(default_factory=list)
    replay_start_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "node_id": self.node_id,
            "object_type": self.object_type,
            "producer_stage": self.producer_stage,
            "consumer_stage": self.consumer_stage,
            "raw_score": round(float(self.raw_score), 4),
            "adjusted_score": round(float(self.adjusted_score), 4),
            "confidence": round(float(self.confidence), 4),
            "reasons": list(self.reasons),
            "forward_tainted_stages": list(self.forward_tainted_stages),
            "forward_tainted_object_ids": list(self.forward_tainted_object_ids),
            "upstream_node_ids": list(self.upstream_node_ids),
            "replay_start_stage": self.replay_start_stage,
        }


@dataclass(frozen=True, slots=True)
class ContaminationFrontier:
    """Minimal invalidation frontier selected by CFR v0."""

    invalidated_object_ids: list[str] = field(default_factory=list)
    frontier_stage_boundaries: list[tuple[str, str]] = field(default_factory=list)
    replay_start_stage: str = ""
    cache_upstream: bool = True
    confidence: float = 0.0
    score: float = 0.0
    clean_upstream_object_ids: list[str] = field(default_factory=list)
    tainted_downstream_stages: list[str] = field(default_factory=list)
    tainted_downstream_object_ids: list[str] = field(default_factory=list)
    cheap_revalidation_probes: list[dict[str, Any]] = field(default_factory=list)
    estimated_replay_span: int = 0
    estimated_replay_cost: float = 0.0
    under_invalidation_risk: float = 0.0
    candidate_scores: list[ContaminantScore] = field(default_factory=list)
    audit_reasons: list[str] = field(default_factory=list)
    evidence: ContaminationEvidence = field(default_factory=ContaminationEvidence)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidated_object_ids": list(self.invalidated_object_ids),
            "frontier_stage_boundaries": [
                [producer, consumer]
                for producer, consumer in self.frontier_stage_boundaries
            ],
            "replay_start_stage": self.replay_start_stage,
            "cache_upstream": self.cache_upstream,
            "confidence": round(float(self.confidence), 4),
            "score": round(float(self.score), 4),
            "clean_upstream_object_ids": list(self.clean_upstream_object_ids),
            "tainted_downstream_stages": list(self.tainted_downstream_stages),
            "tainted_downstream_object_ids": list(self.tainted_downstream_object_ids),
            "cheap_revalidation_probes": [dict(item) for item in self.cheap_revalidation_probes],
            "estimated_replay_span": int(self.estimated_replay_span),
            "estimated_replay_cost": round(float(self.estimated_replay_cost), 4),
            "under_invalidation_risk": round(float(self.under_invalidation_risk), 4),
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
            "audit_reasons": list(self.audit_reasons),
            "evidence": self.evidence.to_dict(),
            "metadata": dict(self.metadata),
        }


def normalize_evidence(
    state: StructuredRecoveryState,
    evidence: dict[str, Any] | None = None,
) -> ContaminationEvidence:
    """Merge state evidence and caller evidence into a compact packet."""

    merged: dict[str, Any] = {}
    merged.update(dict(state.evidence_pack or {}))
    if evidence:
        merged.update(dict(evidence or {}))

    paths: set[str] = set()
    for key in (
        "verifier_failed_files",
        "patcher_modified_files",
        "selected_target_candidates",
        "failing_files",
        "suspect_paths",
    ):
        paths.update(_path_tokens(merged.get(key)))
    paths.update(_paths_from_tests(merged.get("failing_tests", [])))

    text_parts = [
        merged.get("verifier_excerpt", ""),
        merged.get("review_verifier_excerpt", ""),
        merged.get("failure_family_manual", ""),
        merged.get("target_legitimacy", ""),
        merged.get("patch_legitimacy", ""),
        merged.get("stop_reason", ""),
        merged.get("verifier_exception_type", ""),
        " ".join(str(item) for item in merged.get("negative_constraints", []) or []),
        " ".join(paths),
    ]
    text = " ".join(str(item or "") for item in text_parts)
    tokens = _text_tokens(text)
    for path in paths:
        tokens.update(_text_tokens(path.replace("/", " ")))

    failed_stages = {
        _normalize_stage(stage)
        for stage in merged.get("failed_stages", []) or []
        if str(stage or "").strip()
    }
    if str(merged.get("patch_legitimacy", "") or "").strip():
        failed_stages.add("patcher")
    if str(merged.get("verifier_excerpt", "") or "").strip() or merged.get("failing_tests"):
        failed_stages.add("verifier")

    target_node_ids = {
        _node_id_for_object_id(str(item))
        for item in list(merged.get("evidence_node_ids", []) or [])
        + list(merged.get("target_node_ids", []) or [])
        if str(item or "").strip()
    }
    target_node_ids.update(
        f"object::{item}"
        for item in list(merged.get("target_object_ids", []) or [])
        if str(item or "").strip()
    )

    negative_constraints = [
        str(item)
        for item in list(merged.get("negative_constraints", []) or [])
        if str(item or "").strip()
    ]
    return ContaminationEvidence(
        raw=merged,
        text=text,
        tokens=tokens,
        paths=paths,
        failed_stages=failed_stages,
        target_node_ids=target_node_ids,
        negative_constraints=negative_constraints,
    )


def score_contaminants(
    state: StructuredRecoveryState,
    graph: PropagationGraph | None = None,
    evidence: dict[str, Any] | ContaminationEvidence | None = None,
) -> list[ContaminantScore]:
    """Score propagated objects as possible contamination-frontier members."""

    graph = graph or build_propagation_graph(state)
    ev = evidence if isinstance(evidence, ContaminationEvidence) else normalize_evidence(state, evidence)
    object_by_id = {obj.object_id: obj for obj in state.object_chain_view}
    target_nodes = _evidence_target_nodes(graph, ev)
    upstream_by_target = {node_id: graph.backward_slice(node_id) for node_id in target_nodes}
    scores: list[ContaminantScore] = []

    for node_id, node in sorted(graph.nodes.items()):
        if not node_id.startswith("object::"):
            continue
        object_id = str(node.metadata.get("object_id", "") or node_id.removeprefix("object::"))
        obj = object_by_id.get(object_id)
        if obj is None:
            continue
        score, reasons = _score_object(obj, node, graph, ev, upstream_by_target)
        tainted_stages = sorted(graph.forward_taint(node_id), key=_stage_sort_key)
        tainted_objects = sorted(_forward_object_ids(graph, node_id))
        upstream_nodes = sorted(_union_slices(upstream_by_target, node_id))
        replay_stage = _replay_stage_for_object(obj)
        replay_span = _estimate_replay_span(replay_stage, graph)
        lost_clean_work = _lost_clean_work_penalty(obj, graph, state)
        adjusted = score - (6.0 * replay_span) - (2.5 * lost_clean_work)
        confidence = _confidence_from_score(adjusted)
        scores.append(
            ContaminantScore(
                object_id=object_id,
                node_id=node_id,
                object_type=obj.object_type,
                producer_stage=obj.producer_stage,
                consumer_stage=obj.consumer_stage,
                raw_score=score,
                adjusted_score=adjusted,
                confidence=confidence,
                reasons=reasons,
                forward_tainted_stages=tainted_stages,
                forward_tainted_object_ids=tainted_objects,
                upstream_node_ids=upstream_nodes,
                replay_start_stage=replay_stage,
            )
        )
    return sorted(scores, key=lambda item: (-item.adjusted_score, item.object_id))


def compute_contamination_frontier(
    state: StructuredRecoveryState,
    graph: PropagationGraph | None = None,
    evidence: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    *,
    score_threshold: float = 52.0,
    max_frontier_size: int = 3,
) -> ContaminationFrontier:
    """Compute a deterministic minimal invalidation frontier."""

    graph = graph or build_propagation_graph(state)
    ev = normalize_evidence(state, evidence)
    scores = score_contaminants(state, graph, ev)
    selected = _select_minimal_frontier(scores, threshold=score_threshold, limit=max_frontier_size)
    selected = _expand_selection_set_frontier(state, scores, selected)
    invalidated_ids = [item.object_id for item in selected]
    boundaries = _frontier_boundaries(state.object_chain_view, invalidated_ids)
    replay_stage = _earliest_replay_stage([item.replay_start_stage for item in selected])
    tainted_stages = _dedupe(
        stage
        for item in selected
        for stage in item.forward_tainted_stages
        if stage
    )
    tainted_objects = _dedupe(
        object_id
        for item in selected
        for object_id in item.forward_tainted_object_ids
        if object_id
    )
    clean_upstream = _clean_upstream_object_ids(state, graph, selected)
    replay_span = _estimate_replay_span(replay_stage, graph) if replay_stage else 0
    replay_cost = _estimate_replay_cost(replay_stage, budget or {}, replay_span)
    confidence = min((item.confidence for item in selected), default=0.0)
    score = sum(item.adjusted_score for item in selected)
    audit = _frontier_audit_reasons(selected, scores, ev)
    probes = _cheap_revalidation_probes(state, scores, ev)
    under_risk = _under_invalidation_risk(selected, scores, score_threshold)
    return ContaminationFrontier(
        invalidated_object_ids=invalidated_ids,
        frontier_stage_boundaries=boundaries,
        replay_start_stage=replay_stage,
        cache_upstream=True,
        confidence=confidence,
        score=score,
        clean_upstream_object_ids=clean_upstream,
        tainted_downstream_stages=tainted_stages,
        tainted_downstream_object_ids=tainted_objects,
        cheap_revalidation_probes=probes,
        estimated_replay_span=replay_span,
        estimated_replay_cost=replay_cost,
        under_invalidation_risk=under_risk,
        candidate_scores=scores,
        audit_reasons=audit,
        evidence=ev,
        metadata={
            "method": "cfr_v0_deterministic",
            "score_threshold": score_threshold,
            "max_frontier_size": max_frontier_size,
            "budget": dict(budget or {}),
        },
    )


def evaluate_frontier_against_oracle(
    frontier: ContaminationFrontier,
    *,
    oracle_invalidated_object_ids: Iterable[str],
    all_object_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return object-level invalidation metrics for synthetic/manual labels."""

    predicted = set(frontier.invalidated_object_ids)
    oracle = {str(item) for item in oracle_invalidated_object_ids if str(item).strip()}
    universe = {str(item) for item in all_object_ids or (predicted | oracle) if str(item).strip()}
    tp = len(predicted & oracle)
    fp = len(predicted - oracle)
    fn = len(oracle - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0 if not oracle else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    clean = universe - oracle
    clean_preserved = len(clean - predicted)
    clean_work_preservation = clean_preserved / len(clean) if clean else 1.0
    return {
        "InvalidationPrecision": precision,
        "InvalidationRecall": recall,
        "InvalidationF1": f1,
        "OverInvalidationRate": fp / len(universe) if universe else 0.0,
        "UnderInvalidationRate": fn / len(universe) if universe else 0.0,
        "CleanWorkPreservation": clean_work_preservation,
    }


def render_frontier_audit(frontier: ContaminationFrontier) -> str:
    """Render stable JSON for run logs and case audit cards."""

    return json.dumps(frontier.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)


def _score_object(
    obj: PropagationObject,
    node: PropagationGraphNode,
    graph: PropagationGraph,
    evidence: ContaminationEvidence,
    upstream_by_target: dict[str, set[str]],
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    status = str(obj.contamination_status or "unknown").strip().lower()
    score = CONTAMINATION_WEIGHTS.get(status, 0.0)
    if score:
        reasons.append(f"contamination_status:{status}+{score:.0f}")

    node_id = f"object::{obj.object_id}"
    direct_target = node_id in evidence.target_node_ids
    overlap = _payload_evidence_overlap(obj.payload, evidence)
    producer_success = _role_success(graph, obj.producer_stage)
    consumer_success = _role_success(graph, obj.consumer_stage)
    endpoint_failed = producer_success is False or consumer_success is False

    if direct_target:
        score += 90.0
        reasons.append("direct_evidence_target+90")

    explicit_lifecycle = _explicit_lifecycle_state(obj.payload)
    if explicit_lifecycle:
        lifecycle_weight = LIFECYCLE_WEIGHTS.get(explicit_lifecycle, 0.0)
        if lifecycle_weight:
            score += lifecycle_weight
            reasons.append(f"lifecycle:{explicit_lifecycle}{lifecycle_weight:+.0f}")
    unproductive = _unproductive_count(obj.payload)
    if unproductive:
        bonus = min(18.0, 6.0 * unproductive)
        score += bonus
        reasons.append(f"unproductive_count:{unproductive}+{bonus:.0f}")

    evidence_linked = direct_target or bool(overlap) or endpoint_failed or status == "contradiction"
    if _has_outgoing_edge(graph, node_id, "contradicted_by"):
        if evidence_linked:
            score += 95.0
            reasons.append("direct_verifier_contradiction+95")
        else:
            score += 10.0
            reasons.append("weak_verifier_contradiction_without_evidence+10")

    for target, upstream in sorted(upstream_by_target.items()):
        if node_id in upstream and evidence_linked:
            score += 24.0
            reasons.append(f"on_backward_slice_to:{target}+24")
            break

    if producer_success is False:
        score += 18.0
        reasons.append(f"producer_failed:{obj.producer_stage}+18")
    if consumer_success is False:
        score += 14.0
        reasons.append(f"consumer_failed:{obj.consumer_stage}+14")

    if overlap:
        bonus = min(36.0, 8.0 * len(overlap))
        score += bonus
        reasons.append(f"payload_evidence_overlap:{','.join(overlap[:4])}+{bonus:.0f}")

    if obj.object_type == "VerifierVerdict" and status not in {"contradiction", "stale_or_unverified"}:
        score -= 45.0
        reasons.append("preserve_observed_verifier_evidence-45")
    if obj.object_type == "Selection" and obj.consumer_stage == "patcher":
        if any(path in str(obj.payload) for path in evidence.paths):
            score += 8.0
            reasons.append("selection_mentions_failed_path+8")
    if _normalize_stage(obj.consumer_stage) in evidence.failed_stages:
        score += 10.0
        reasons.append(f"consumer_is_failed_stage:{obj.consumer_stage}+10")

    return max(0.0, score), reasons


def _select_minimal_frontier(
    scores: list[ContaminantScore],
    *,
    threshold: float,
    limit: int,
) -> list[ContaminantScore]:
    eligible = [item for item in scores if item.adjusted_score >= threshold]
    selected: list[ContaminantScore] = []
    for item in eligible:
        if len(selected) >= max(1, limit):
            break
        if _is_dominated_by_selected(item, selected):
            continue
        selected.append(item)
    return selected


def _expand_selection_set_frontier(
    state: StructuredRecoveryState,
    scores: list[ContaminantScore],
    selected: list[ContaminantScore],
) -> list[ContaminantScore]:
    """Expand locator frontier when the contaminated object is a candidate set.

    In pre-patch natural failures the propagated object is often not one
    individual selected path but the locator's whole candidate set. Treating
    only the highest-overlap path as polluted under-invalidates the selection
    frontier and gives replay permission to consume sibling stale candidates.
    """

    if not selected or not _selection_set_mode(state):
        return selected
    if not any(item.object_type == "Selection" for item in selected):
        return selected
    selected_ids = {item.object_id for item in selected}
    selection_scores = [
        item
        for item in scores
        if item.object_type == "Selection"
        and _normalize_stage(item.producer_stage) == "locator"
        and _normalize_stage(item.consumer_stage) == "patcher"
    ]
    expanded = list(selected)
    for item in selection_scores:
        if item.object_id in selected_ids:
            continue
        if item.adjusted_score <= 0 and "suspicious" not in " ".join(item.reasons):
            continue
        expanded.append(item)
        selected_ids.add(item.object_id)
    return sorted(expanded, key=lambda item: (-item.adjusted_score, item.object_id))


def _selection_set_mode(state: StructuredRecoveryState) -> bool:
    local = dict(state.local_region_view or {})
    evidence = dict(state.evidence_pack or {})
    text = " ".join(
        str(value or "").lower()
        for value in (
            local.get("fault_type", ""),
            local.get("trigger_type", ""),
            evidence.get("failure_family_manual", ""),
            evidence.get("target_legitimacy", ""),
            evidence.get("patch_legitimacy", ""),
            evidence.get("typed_object_quality", ""),
            evidence.get("patcher_stop_reason", ""),
            evidence.get("stop_reason", ""),
        )
    )
    if "wrong selection" not in text and "selection" not in text:
        return False
    if any(
        marker in text
        for marker in (
            "pre_patch_selection_only",
            "patcher_failed_without_effective_patch",
            "no_effective_patch",
            "no_diff",
            "typed_object_quality selection",
        )
    ):
        return True
    object_types = {obj.object_type for obj in state.object_chain_view}
    return bool(object_types) and object_types.issubset({"Selection"})


def _is_dominated_by_selected(
    item: ContaminantScore,
    selected: list[ContaminantScore],
) -> bool:
    item_taint = set(item.forward_tainted_stages)
    for chosen in selected:
        chosen_taint = set(chosen.forward_tainted_stages)
        if chosen.object_id == item.object_id:
            return True
        if chosen_taint and item_taint and item_taint.issubset(chosen_taint):
            if chosen.adjusted_score >= item.adjusted_score * 0.75:
                return True
        if item.node_id in set(chosen.upstream_node_ids):
            # Preserve upstream work when a much stronger downstream object is
            # directly contradicted; otherwise CFR degenerates into rollback.
            if chosen.adjusted_score >= item.adjusted_score * 2.0:
                return True
        if _stage_sort_key(chosen.replay_start_stage) >= _stage_sort_key(item.replay_start_stage):
            if chosen.adjusted_score >= item.adjusted_score * 0.8:
                return True
    return False


def _frontier_boundaries(
    objects: list[PropagationObject],
    invalidated_ids: list[str],
) -> list[tuple[str, str]]:
    selected = set(invalidated_ids)
    boundaries: list[tuple[str, str]] = []
    for obj in objects:
        if obj.object_id not in selected:
            continue
        producer = _normalize_stage(obj.producer_stage)
        consumer = _normalize_stage(obj.consumer_stage)
        if producer and consumer and (producer, consumer) not in boundaries:
            boundaries.append((producer, consumer))
    return boundaries


def _clean_upstream_object_ids(
    state: StructuredRecoveryState,
    graph: PropagationGraph,
    selected: list[ContaminantScore],
) -> list[str]:
    selected_nodes = {item.node_id for item in selected}
    tainted_nodes = set(selected_nodes)
    for item in selected:
        tainted_nodes.update(f"object::{object_id}" for object_id in item.forward_tainted_object_ids)
    clean: list[str] = []
    for obj in state.object_chain_view:
        node_id = f"object::{obj.object_id}"
        if node_id in tainted_nodes:
            continue
        if any(node_id in graph.backward_slice(item.node_id) for item in selected):
            clean.append(obj.object_id)
    return clean


def _frontier_audit_reasons(
    selected: list[ContaminantScore],
    all_scores: list[ContaminantScore],
    evidence: ContaminationEvidence,
) -> list[str]:
    if not selected:
        top = all_scores[0] if all_scores else None
        if top is None:
            return ["no_propagation_objects_available_for_cfr"]
        return [
            "no_candidate_crossed_confidence_threshold",
            f"top_candidate={top.object_id}",
            f"top_adjusted_score={top.adjusted_score:.2f}",
            "cheap_revalidation_required_before_selective_replay",
        ]
    reasons = [
        "selected_minimal_object_frontier",
        f"invalidated={','.join(item.object_id for item in selected)}",
        f"replay_start={_earliest_replay_stage(item.replay_start_stage for item in selected)}",
    ]
    for item in selected:
        reasons.append(f"{item.object_id}: " + "; ".join(item.reasons[:5]))
    if evidence.negative_constraints:
        reasons.append("negative_constraints=" + " | ".join(evidence.negative_constraints[:3]))
    return reasons


def _cheap_revalidation_probes(
    state: StructuredRecoveryState,
    scores: list[ContaminantScore],
    evidence: ContaminationEvidence,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    if evidence.paths:
        probes.append(
            {
                "probe_type": "path_existence_or_hash",
                "target": sorted(evidence.paths)[0],
                "reason": "verify evidence-linked path before invalidating upstream objects",
            }
        )
    if state.evidence_pack.get("failing_tests"):
        probes.append(
            {
                "probe_type": "focused_test",
                "target": str(list(state.evidence_pack.get("failing_tests", []) or [""])[0]),
                "reason": "confirm current verifier evidence is still active",
            }
        )
    if scores:
        probes.append(
            {
                "probe_type": "object_payload_recheck",
                "target": scores[0].object_id,
                "reason": "top CFR candidate needs cheap confirmation when confidence is low",
            }
        )
    return probes[:3]


def _under_invalidation_risk(
    selected: list[ContaminantScore],
    scores: list[ContaminantScore],
    threshold: float,
) -> float:
    if not scores:
        return 1.0
    if not selected:
        return min(1.0, scores[0].confidence + 0.25)
    selected_ids = {item.object_id for item in selected}
    near_misses = [
        item
        for item in scores
        if item.object_id not in selected_ids and item.adjusted_score >= threshold * 0.75
    ]
    return min(1.0, 0.12 + 0.10 * len(near_misses))


def _evidence_target_nodes(graph: PropagationGraph, evidence: ContaminationEvidence) -> set[str]:
    targets = {node_id for node_id in evidence.target_node_ids if node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        if node.node_type == "VerifierVerdict":
            targets.add(node_id)
        if node_id == "role::verifier":
            targets.add(node_id)
    return targets


def _union_slices(upstream_by_target: dict[str, set[str]], node_id: str) -> set[str]:
    out: set[str] = set()
    for values in upstream_by_target.values():
        if node_id in values:
            out.update(values)
    return out


def _forward_object_ids(graph: PropagationGraph, node_id: str) -> set[str]:
    visited: set[str] = set()
    objects: set[str] = set()
    frontier = [node_id]
    while frontier:
        current = frontier.pop()
        for edge in graph.edges:
            if edge.src != current or edge.dst in visited:
                continue
            visited.add(edge.dst)
            if edge.dst.startswith("object::"):
                objects.add(edge.dst.removeprefix("object::"))
            frontier.append(edge.dst)
    return objects


def _has_outgoing_edge(graph: PropagationGraph, node_id: str, edge_type: str) -> bool:
    return any(edge.src == node_id and edge.edge_type == edge_type for edge in graph.edges)


def _role_success(graph: PropagationGraph, stage: str) -> bool | None:
    node = graph.nodes.get(f"role::{_normalize_stage(stage)}")
    if node is None:
        return None
    if "success" not in node.metadata:
        return None
    return bool(node.metadata.get("success", False))


def _payload_evidence_overlap(payload: dict[str, Any], evidence: ContaminationEvidence) -> list[str]:
    text = _payload_text(payload).lower()
    overlaps: list[str] = []
    for path in sorted(evidence.paths):
        if path and path.lower() in text:
            overlaps.append(path)
    payload_tokens = _text_tokens(text)
    for token in sorted(evidence.tokens & payload_tokens):
        if len(token) >= 4 and token not in overlaps:
            overlaps.append(token)
    return overlaps[:8]


def _payload_text(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(payload)


def _unproductive_count(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    values = []
    if isinstance(metadata, dict):
        values.append(metadata.get("unproductive_count", 0))
    values.append(payload.get("unproductive_count", 0) if isinstance(payload, dict) else 0)
    for value in values:
        try:
            count = int(value or 0)
            if count:
                return count
        except (TypeError, ValueError):
            continue
    return 0


def _explicit_lifecycle_state(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if isinstance(metadata, dict):
        value = str(metadata.get("lifecycle_state", "") or "").strip()
        if value:
            return lifecycle_state_for_payload(payload)
    value = str(payload.get("lifecycle_state", "") if isinstance(payload, dict) else "").strip()
    if value:
        return lifecycle_state_for_payload(payload)
    return ""


def _replay_stage_for_object(obj: PropagationObject) -> str:
    producer = _normalize_stage(obj.producer_stage)
    consumer = _normalize_stage(obj.consumer_stage)
    if producer:
        return producer
    return consumer


def _earliest_replay_stage(stages: Iterable[str]) -> str:
    values = [_normalize_stage(stage) for stage in stages if str(stage or "").strip()]
    if not values:
        return ""
    return sorted(values, key=_stage_sort_key)[0]


def _estimate_replay_span(stage: str, graph: PropagationGraph) -> int:
    stage = _normalize_stage(stage)
    if not stage:
        return 0
    graph_stages = {
        _normalize_stage(str(node.metadata.get("stage", "") or node.metadata.get("role", "") or ""))
        for node in graph.nodes.values()
        if node.node_type == "RoleOutput"
    }
    graph_stages.discard("")
    graph_stages.discard("bcmr")
    graph_stages = {item for item in graph_stages if item in STAGE_ORDER}
    if stage and stage not in graph_stages:
        graph_stages.add(stage)
    if graph_stages:
        ordered = sorted(graph_stages, key=_stage_sort_key)
    else:
        ordered = list(STAGE_ORDER)
    start_index = min((_stage_sort_key(item) for item in ordered if item == stage), default=_stage_sort_key(stage))
    return max(1, len([item for item in ordered if _stage_sort_key(item) >= start_index]))


def _estimate_replay_cost(stage: str, budget: dict[str, Any], replay_span: int) -> float:
    if not stage:
        return 0.0
    per_stage = float(budget.get("estimated_tokens_per_stage", 900.0) or 900.0)
    fixed = float(budget.get("fixed_replay_overhead", 120.0) or 120.0)
    return fixed + per_stage * max(1, replay_span)


def _lost_clean_work_penalty(
    obj: PropagationObject,
    graph: PropagationGraph,
    state: StructuredRecoveryState,
) -> int:
    node_id = f"object::{obj.object_id}"
    upstream = graph.backward_slice(node_id)
    upstream_objects = [item for item in upstream if item.startswith("object::")]
    # Penalize early objects that force more upstream work to be questioned.
    if not upstream_objects and _normalize_stage(obj.producer_stage) == "locator":
        return max(1, len(state.object_chain_view) // 2)
    return len(upstream_objects)


def _confidence_from_score(score: float) -> float:
    if score <= 0:
        return 0.0
    return max(0.0, min(0.99, score / 140.0))


def _stage_sort_key(stage: str) -> int:
    norm = _normalize_stage(stage)
    if norm in STAGE_ORDER:
        return STAGE_ORDER.index(norm)
    return len(STAGE_ORDER) + (sum(ord(ch) for ch in norm) % 1000)


def _normalize_stage(stage: str) -> str:
    return str(stage or "").strip().lower()


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _path_tokens(value: Any) -> set[str]:
    values: list[str] = []
    if isinstance(value, (list, tuple, set)):
        values = [str(item or "") for item in value]
    elif value:
        values = [str(value)]
    paths: set[str] = set()
    for item in values:
        if not item:
            continue
        paths.update(match.replace("\\", "/") for match in re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+", item))
    return paths


def _paths_from_tests(value: Any) -> set[str]:
    paths: set[str] = set()
    items = list(value or []) if isinstance(value, (list, tuple, set)) else [value]
    for item in items:
        text = str(item or "")
        candidate = text.split("::", 1)[0].strip()
        if "." in candidate:
            paths.add(candidate.replace("\\", "/"))
    return paths


def _text_tokens(text: Any) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]{3,}", str(text or ""))
        if token
    }


def _node_id_for_object_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("object::", "role::", "anchor::")):
        return text
    return f"object::{text}"
