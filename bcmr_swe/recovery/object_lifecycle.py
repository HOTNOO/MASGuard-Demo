"""PARC object lifecycle state machine.

The lifecycle is a small deterministic layer over propagation objects.  It
does not know about SWE-Bench cases or repository contents; it only maps
MAS recovery actions and observed result modes into object-state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from bcmr_swe.types import ActionObservation, RecoveryLedger, SemanticActionType, StateDelta, StructuredRecoveryState


class LifecycleState(str, Enum):
    SUSPICIOUS = "suspicious"
    INVALIDATED = "invalidated"
    RESOLVED = "resolved"
    EXHAUSTED = "exhausted"


TERMINAL_LIFECYCLE_STATES = {
    LifecycleState.RESOLVED.value,
    LifecycleState.EXHAUSTED.value,
}

UNPRODUCTIVE_RESULT_MODES = {
    "no_diff",
    "wrong_edit_target",
    "source_edit_but_not_suspect",
    "budget_exhausted",
    "unproductive_no_state_change",
}

RESOLVED_RESULT_MODES = {
    "strong_source_success",
    "mixed_source_test_success",
    "checkpoint_or_no_diff_success",
}


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    object_id: str
    before: str
    after: str
    reason: str
    unproductive_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "unproductive_count": self.unproductive_count,
        }


def lifecycle_state_for_payload(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if isinstance(metadata, dict):
        value = str(metadata.get("lifecycle_state", "") or "").strip()
        if value:
            return _normalize_lifecycle(value)
    value = str(payload.get("lifecycle_state", "") if isinstance(payload, dict) else "").strip()
    if value:
        return _normalize_lifecycle(value)
    return LifecycleState.SUSPICIOUS.value


def initialize_lifecycle_state(state: StructuredRecoveryState) -> StructuredRecoveryState:
    """Ensure every propagation object has a v1 lifecycle label."""

    for obj in state.object_chain_view:
        metadata = _payload_metadata(obj.payload)
        metadata.setdefault("lifecycle_state", _initial_object_state(obj.contamination_status))
        metadata.setdefault("unproductive_count", 0)
    return state


def apply_lifecycle_transition(
    state: StructuredRecoveryState,
    *,
    action: str | SemanticActionType,
    active_object_id: str,
    result_mode: str = "",
    produced_fresh_source_diff: bool = False,
    unproductive_threshold: int = 2,
) -> LifecycleTransition | None:
    """Apply one deterministic lifecycle transition to `state`.

    The rules follow the current PARC v1 plan:
    - REVOKE_OBJECT invalidates the active object.
    - LOCAL_REPAIR/SCOPE_EXPAND resolve an invalidated or suspicious object
      only when they produce a fresh source diff and a success-like result.
    - Repeated unproductive attempts exhaust the object.
    """

    action_value = _action_value(action)
    result = str(result_mode or "").strip()
    target = _find_object_payload(state, active_object_id)
    if target is None:
        return None

    payload = target
    metadata = _payload_metadata(payload)
    before = lifecycle_state_for_payload(payload)
    after = before
    reason = "no_transition"
    unproductive_count = int(metadata.get("unproductive_count", 0) or 0)

    if before in TERMINAL_LIFECYCLE_STATES:
        return None

    if action_value == SemanticActionType.REVOKE_OBJECT.value:
        after = LifecycleState.INVALIDATED.value
        reason = "revoked_active_object"
        unproductive_count = 0
    elif action_value in {
        SemanticActionType.LOCAL_REPAIR.value,
        SemanticActionType.REPAIR_LOCAL.value,
        SemanticActionType.SCOPE_EXPAND.value,
        SemanticActionType.EXPAND_SCOPE.value,
    }:
        if produced_fresh_source_diff and result in RESOLVED_RESULT_MODES:
            after = LifecycleState.RESOLVED.value
            reason = "fresh_source_diff_resolved"
            unproductive_count = 0
        elif result in UNPRODUCTIVE_RESULT_MODES or not produced_fresh_source_diff:
            unproductive_count += 1
            if unproductive_count >= max(1, int(unproductive_threshold)):
                after = LifecycleState.EXHAUSTED.value
                reason = "unproductive_replay_threshold"
            else:
                reason = "unproductive_replay_observed"
    elif action_value in {
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.RECHECK_OBJECT.value,
    }:
        if result in RESOLVED_RESULT_MODES:
            after = LifecycleState.RESOLVED.value
            reason = "evidence_recheck_resolved"
            unproductive_count = 0

    metadata["lifecycle_state"] = after
    metadata["unproductive_count"] = unproductive_count
    if after == before and reason == "no_transition":
        return None
    return LifecycleTransition(
        object_id=active_object_id,
        before=before,
        after=after,
        reason=reason,
        unproductive_count=unproductive_count,
    )


def apply_lifecycle_transition_to_ledger(
    ledger: RecoveryLedger,
    *,
    action_observation: ActionObservation,
    state_delta: StateDelta,
    unproductive_threshold: int = 2,
) -> LifecycleTransition | None:
    """Update the serialized structured state carried by a recovery ledger."""

    if not ledger.structured_state:
        return None
    try:
        state = StructuredRecoveryState.from_dict(ledger.structured_state)
    except Exception:
        return None
    initialize_lifecycle_state(state)
    active_object_id = (
        str(state_delta.metadata.get("lifecycle_target_object_id", "") or "")
        or str(state_delta.active_object_id or "")
        or str(action_observation.active_object_id_after or "")
        or str(action_observation.active_object_id_before or "")
        or str(ledger.active_object_id or "")
    )
    if not active_object_id:
        return None
    result_mode = (
        str(action_observation.status or "")
        or str(state_delta.metadata.get("result_mode", "") or "")
    )
    produced_fresh_source_diff = bool(
        action_observation.metadata.get("fresh_source_diff", False)
        or action_observation.metadata.get("fresh_source_files", [])
        or action_observation.touched_paths
        and result_mode in RESOLVED_RESULT_MODES | {"oracle_failed_after_source_edit"}
    )
    transition = apply_lifecycle_transition(
        state,
        action=action_observation.action_type,
        active_object_id=active_object_id,
        result_mode=result_mode,
        produced_fresh_source_diff=produced_fresh_source_diff,
        unproductive_threshold=unproductive_threshold,
    )
    if transition is None:
        ledger.structured_state = state.to_dict()
        return None
    transitions = [
        dict(item)
        for item in list(ledger.metadata.get("lifecycle_transitions", []) or [])
        if isinstance(item, dict)
    ]
    transitions.append(transition.to_dict())
    ledger.metadata["lifecycle_transitions"] = transitions[-12:]
    ledger.structured_state = state.to_dict()
    return transition


def lifecycle_summary(state: StructuredRecoveryState) -> dict[str, Any]:
    initialize_lifecycle_state(state)
    by_state: dict[str, list[str]] = {
        LifecycleState.SUSPICIOUS.value: [],
        LifecycleState.INVALIDATED.value: [],
        LifecycleState.RESOLVED.value: [],
        LifecycleState.EXHAUSTED.value: [],
    }
    for obj in state.object_chain_view:
        state_value = lifecycle_state_for_payload(obj.payload)
        by_state.setdefault(state_value, []).append(obj.object_id)
    return {
        "by_state": by_state,
        "unresolved_objects": by_state.get(LifecycleState.SUSPICIOUS.value, [])
        + by_state.get(LifecycleState.INVALIDATED.value, []),
        "terminal_objects": by_state.get(LifecycleState.RESOLVED.value, [])
        + by_state.get(LifecycleState.EXHAUSTED.value, []),
        "recovery_done": not (
            by_state.get(LifecycleState.SUSPICIOUS.value, [])
            or by_state.get(LifecycleState.INVALIDATED.value, [])
        ),
    }


def _payload_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    return metadata


def _find_object_payload(state: StructuredRecoveryState, object_id: str) -> dict[str, Any] | None:
    for obj in state.object_chain_view:
        if str(obj.object_id or "") == str(object_id or ""):
            return obj.payload
    return None


def _initial_object_state(contamination_status: str) -> str:
    status = str(contamination_status or "").strip().lower()
    if status in {"resolved", "revalidated"}:
        return LifecycleState.RESOLVED.value
    if status in {"exhausted"}:
        return LifecycleState.EXHAUSTED.value
    if status in {"invalidated", "revoked"}:
        return LifecycleState.INVALIDATED.value
    return LifecycleState.SUSPICIOUS.value


def _normalize_lifecycle(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "unknown": LifecycleState.SUSPICIOUS.value,
        "revalidated": LifecycleState.RESOLVED.value,
        "repaired": LifecycleState.RESOLVED.value,
        "revoked": LifecycleState.INVALIDATED.value,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {item.value for item in LifecycleState}:
        return normalized
    return LifecycleState.SUSPICIOUS.value


def _action_value(action: str | SemanticActionType) -> str:
    if isinstance(action, SemanticActionType):
        return action.value
    return str(action or "")
