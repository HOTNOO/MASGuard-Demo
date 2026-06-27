"""Budget-aware PARC controller.

This controller is deliberately small and deterministic.  It consumes the
structured MAS state, action cards, and mutation predictor; it does not call an
LLM and does not encode benchmark-instance special cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcmr_swe.recovery.action_cards import ActionCard, available_action_cards
from bcmr_swe.recovery.mutation_predictor import expensive_action_needs_escape, predict_mutations
from bcmr_swe.recovery.object_lifecycle import (
    LifecycleState,
    initialize_lifecycle_state,
    lifecycle_state_for_payload,
)
from bcmr_swe.types import RecoveryLedger, SemanticActionType, StructuredRecoveryState


@dataclass(slots=True)
class RecoveryFrontier:
    active_object_id: str = ""
    unresolved_objects: list[str] = field(default_factory=list)
    exhausted_objects: list[str] = field(default_factory=list)
    resolved_objects: list[str] = field(default_factory=list)
    viable_replay_anchors: list[str] = field(default_factory=list)
    cheap_actions_available: list[str] = field(default_factory=list)
    expensive_actions_available: list[str] = field(default_factory=list)
    precondition_changed_since_last_replay: bool = False
    predicted_state_mutations: dict[str, list[str]] = field(default_factory=dict)
    object_states: dict[str, str] = field(default_factory=dict)
    recovery_done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_object_id": self.active_object_id,
            "unresolved_objects": list(self.unresolved_objects),
            "exhausted_objects": list(self.exhausted_objects),
            "resolved_objects": list(self.resolved_objects),
            "viable_replay_anchors": list(self.viable_replay_anchors),
            "cheap_actions_available": list(self.cheap_actions_available),
            "expensive_actions_available": list(self.expensive_actions_available),
            "precondition_changed_since_last_replay": self.precondition_changed_since_last_replay,
            "predicted_state_mutations": {
                key: list(value)
                for key, value in self.predicted_state_mutations.items()
            },
            "object_states": dict(self.object_states),
            "recovery_done": bool(self.recovery_done),
        }


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    action: str
    reason: str
    score: float
    frontier: RecoveryFrontier
    rejected: list[dict[str, Any]] = field(default_factory=list)
    escape_granted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "score": self.score,
            "frontier": self.frontier.to_dict(),
            "rejected": [dict(item) for item in self.rejected],
            "escape_granted": self.escape_granted,
        }


def compute_recovery_frontier(ledger: RecoveryLedger) -> RecoveryFrontier:
    state = _structured_state_from_ledger(ledger)
    object_states: dict[str, str] = {}
    unresolved: list[str] = []
    exhausted: list[str] = []
    resolved: list[str] = []
    active = str(ledger.active_object_id or "").strip()
    if state is not None:
        initialize_lifecycle_state(state)
        for obj in state.object_chain_view:
            value = lifecycle_state_for_payload(obj.payload)
            object_states[obj.object_id] = value
            if value in {LifecycleState.SUSPICIOUS.value, LifecycleState.INVALIDATED.value}:
                unresolved.append(obj.object_id)
            elif value == LifecycleState.EXHAUSTED.value:
                exhausted.append(obj.object_id)
            elif value == LifecycleState.RESOLVED.value:
                resolved.append(obj.object_id)
        if not active and unresolved:
            active = unresolved[0]
        replay = dict(state.replay_anchor_view or {})
        anchors = _dedupe(
            list(replay.get("healthy_anchor_candidates", []) or [])
            + list(replay.get("checkpoint_labels_available", []) or [])
        )
    else:
        anchors = []
        if active:
            object_states[active] = "suspicious"
            unresolved.append(active)

    frontier = RecoveryFrontier(
        active_object_id=active,
        unresolved_objects=unresolved,
        exhausted_objects=exhausted,
        resolved_objects=resolved,
        viable_replay_anchors=anchors,
        object_states=object_states,
        precondition_changed_since_last_replay=bool(
            ledger.metadata.get("last_anchor")
            or ledger.invalidated_object_ids
            or ledger.invalidated_targets
            or ledger.key_evidence
        ),
    )
    frontier.recovery_done = bool(object_states) and not unresolved
    for card in available_action_cards(ledger):
        mutations = predict_mutations(card.action, ledger=ledger, frontier=frontier)
        if not mutations and not expensive_action_needs_escape(card.action, ledger=ledger, frontier=frontier):
            continue
        frontier.predicted_state_mutations[card.action.value] = mutations
        if card.cost_class == "expensive":
            frontier.expensive_actions_available.append(card.action.value)
        else:
            frontier.cheap_actions_available.append(card.action.value)
    ledger.metadata["recovery_frontier"] = frontier.to_dict()
    if state is not None:
        ledger.structured_state = state.to_dict()
    return frontier


def suggest_next_action(
    ledger: RecoveryLedger,
    *,
    current_suggestion: str = "",
) -> ControllerDecision | None:
    frontier = compute_recovery_frontier(ledger)
    if frontier.recovery_done:
        decision = ControllerDecision(
            action="",
            reason="recovery_goal_satisfied",
            score=float("inf"),
            frontier=frontier,
        )
        _record_controller_event(ledger, decision)
        return decision
    candidates = available_action_cards(ledger)
    if not candidates:
        return None

    best: tuple[float, ActionCard, list[str], bool] | None = None
    rejected: list[dict[str, Any]] = []
    for card in candidates:
        mutations = predict_mutations(card.action, ledger=ledger, frontier=frontier)
        escape = False
        if not mutations:
            if expensive_action_needs_escape(card.action, ledger=ledger, frontier=frontier):
                escape = True
            else:
                rejected.append(
                    {
                        "action": card.action.value,
                        "reason": "no_predicted_mutation",
                    }
                )
                continue
        score = _score_candidate(
            card,
            mutations,
            ledger=ledger,
            frontier=frontier,
            current_suggestion=current_suggestion,
            escape=escape,
        )
        if best is None or score > best[0]:
            best = (score, card, mutations, escape)

    if best is None:
        return ControllerDecision(
            action="",
            reason="no_viable_action",
            score=float("-inf"),
            frontier=frontier,
            rejected=rejected,
        )

    score, card, mutations, escape = best
    frontier.predicted_state_mutations[card.action.value] = mutations
    decision = ControllerDecision(
        action=card.action.value,
        reason="parc_budgeted_controller",
        score=score,
        frontier=frontier,
        rejected=rejected,
        escape_granted=escape,
    )
    _record_controller_event(ledger, decision)
    if escape:
        ledger.metadata["escape_grants_used"] = int(ledger.metadata.get("escape_grants_used", 0) or 0) + 1
    return decision


def _record_controller_event(ledger: RecoveryLedger, decision: ControllerDecision) -> None:
    events = [
        dict(item)
        for item in list(ledger.metadata.get("parc_controller_events", []) or [])
        if isinstance(item, dict)
    ]
    events.append(decision.to_dict())
    ledger.metadata["parc_controller_events"] = events[-12:]


def _score_candidate(
    card: ActionCard,
    mutations: list[str],
    *,
    ledger: RecoveryLedger,
    frontier: RecoveryFrontier,
    current_suggestion: str,
    escape: bool,
) -> float:
    score = 0.0
    action = card.action
    active_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
    active_state = _frontier_active_state(frontier, ledger)
    last_action = str(ledger.last_action or "")
    tried = [str(item) for item in list(ledger.tried_actions or []) if str(item).strip()]
    last_mode = _last_result_mode(ledger)

    if "object.lifecycle_state" in mutations:
        score += 500.0
    if action == SemanticActionType.REVOKE_OBJECT and active_type == "shared_fact":
        score += 280.0
        if _has_focused_source_boundary(ledger):
            score -= 320.0
    if action == SemanticActionType.REVOKE_OBJECT and active_type == "selection":
        score += 40.0
    if action == SemanticActionType.RECHECK_OBJECT and active_type in {"verifier_verdict", "selection"}:
        score += 260.0
    if action == SemanticActionType.EVIDENCE_RECHECK and active_type in {"verifier_verdict", ""}:
        score += 200.0
    if (
        action == SemanticActionType.EVIDENCE_RECHECK
        and active_type == "shared_fact"
        and _has_focused_source_boundary(ledger)
    ):
        score -= 220.0
    if action == SemanticActionType.TARGET_RESET and active_type == "selection":
        score += 180.0
    if action == SemanticActionType.TARGET_RESET and active_type == "shared_fact":
        score += 620.0 if _has_focused_source_boundary(ledger) else 90.0

    if active_state == LifecycleState.INVALIDATED.value:
        if action == SemanticActionType.LOCAL_REPAIR:
            score += 760.0
        elif card.cost_class == "cheap" and action in {
            SemanticActionType.EVIDENCE_RECHECK,
            SemanticActionType.RECHECK_OBJECT,
            SemanticActionType.REVOKE_OBJECT,
        }:
            score -= 220.0

    if last_action == SemanticActionType.REVOKE_OBJECT.value and action == SemanticActionType.LOCAL_REPAIR:
        score += 500.0
    if last_action == SemanticActionType.RECHECK_OBJECT.value and active_type == "selection":
        if action == SemanticActionType.TARGET_RESET:
            score += 720.0
        elif action == SemanticActionType.LOCAL_REPAIR:
            score += 360.0
        elif action == SemanticActionType.RECHECK_OBJECT:
            score -= 260.0

    if last_action in {SemanticActionType.LOCAL_REPAIR.value, SemanticActionType.REPAIR_LOCAL.value}:
        if last_mode in {"no_diff", "wrong_edit_target", "unproductive_no_state_change"}:
            if action == SemanticActionType.SCOPE_EXPAND:
                score += 820.0
            elif action == SemanticActionType.CAPABILITY_BOOST:
                score += 260.0
            elif action in {
                SemanticActionType.RECHECK_OBJECT,
                SemanticActionType.REVOKE_OBJECT,
                SemanticActionType.EVIDENCE_RECHECK,
            }:
                score -= 360.0
        elif last_mode == "oracle_failed_after_source_edit":
            if action == SemanticActionType.CAPABILITY_BOOST:
                score += 420.0
            elif action == SemanticActionType.LOCAL_REPAIR:
                score += 220.0

    if _consecutive_preparatory_actions(tried) >= 2:
        if action == SemanticActionType.LOCAL_REPAIR:
            score += 250.0
        elif card.cost_class == "cheap":
            score -= 80.0

    if card.cost_class == "cheap":
        score += 120.0
    elif card.cost_class == "medium":
        score += 70.0
    else:
        score += 30.0
    if mutations:
        score += 40.0
    if card.action.value == current_suggestion:
        score += 20.0
    if card.action.value == last_action:
        score -= 30.0
    if escape:
        score -= 15.0
    return score


def _frontier_active_state(frontier: RecoveryFrontier, ledger: RecoveryLedger) -> str:
    active = str(frontier.active_object_id or ledger.active_object_id or "")
    states = dict(frontier.object_states or {})
    value = str(states.get(active, "") or "").strip()
    return value or LifecycleState.SUSPICIOUS.value


def _consecutive_preparatory_actions(tried: list[str]) -> int:
    preparatory = {
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.RECHECK_OBJECT.value,
        SemanticActionType.REVOKE_OBJECT.value,
        SemanticActionType.TARGET_RESET.value,
    }
    count = 0
    for action in reversed(tried):
        if action not in preparatory:
            break
        count += 1
    return count


def _has_focused_source_boundary(ledger: RecoveryLedger) -> bool:
    """True when the structured MAS state already exposes a narrow source handoff.

    This is a generic stage-boundary signal: if the failed state has a concrete
    canonical source path, the cheapest recovery boundary is often to reset the
    downstream patch handoff before globally invalidating the propagated patch
    object.  It does not inspect benchmark ids or repository-specific names.
    """

    paths = [
        str(item).replace("\\", "/")
        for item in list(ledger.suspect_paths or [])
        if str(item).strip()
    ]
    if not paths:
        paths = [
            str(ledger.active_target or "").replace("\\", "/")
        ] if str(ledger.active_target or "").strip() else []
    source_paths = [
        path
        for path in paths
        if path.endswith(".py") and "/tests/" not in f"/{path}" and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    return 0 < len(source_paths) <= 2


def _last_result_mode(ledger: RecoveryLedger) -> str:
    guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
    if guard:
        return str(guard.get("result_mode", "") or "")
    guards = [
        dict(item)
        for item in list(ledger.metadata.get("guard_history", []) or [])
        if isinstance(item, dict)
    ]
    if guards:
        return str(guards[-1].get("result_mode", "") or "")
    return ""


def _structured_state_from_ledger(ledger: RecoveryLedger) -> StructuredRecoveryState | None:
    if not ledger.structured_state:
        return None
    try:
        return StructuredRecoveryState.from_dict(ledger.structured_state)
    except Exception:
        return None


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
