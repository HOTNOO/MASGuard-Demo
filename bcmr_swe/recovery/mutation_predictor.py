"""Zero-token mutation prediction for PARC action gating."""

from __future__ import annotations

from typing import Any

from bcmr_swe.recovery.action_cards import get_action_card
from bcmr_swe.recovery.object_lifecycle import TERMINAL_LIFECYCLE_STATES
from bcmr_swe.types import RecoveryLedger, SemanticActionType


def predict_mutations(
    action: str | SemanticActionType,
    *,
    ledger: RecoveryLedger,
    frontier: Any | None = None,
    graph: Any | None = None,
) -> list[str]:
    """Predict which state fields an action can change without calling an LLM."""

    card = get_action_card(action)
    action_value = card.action.value
    active_state = _active_lifecycle_state(ledger, frontier)
    tried = [str(item) for item in list(ledger.tried_actions or []) if str(item).strip()]

    if active_state in TERMINAL_LIFECYCLE_STATES and action_value not in {
        SemanticActionType.TARGET_RESET.value,
        SemanticActionType.EVIDENCE_RECHECK.value,
    }:
        return []

    if action_value == SemanticActionType.REVOKE_OBJECT.value:
        if active_state == "invalidated" or not str(ledger.active_object_id or "").strip():
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.RECHECK_OBJECT.value:
        if not str(ledger.active_object_id or "").strip():
            return []
        if _last_action_result_mode(ledger) in {"observed", ""} and tried[-1:] == [action_value]:
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.EVIDENCE_RECHECK.value:
        if tried[-1:] == [action_value] and _last_action_result_mode(ledger) in {"observed", "no_diff", ""}:
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.TARGET_RESET.value:
        if tried.count(action_value) >= 1 and not _has_untried_anchor(ledger):
            return []
        if not _has_viable_anchor(ledger) and not ledger.suspect_paths:
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.CAPABILITY_BOOST.value:
        if str(ledger.execution_profile or "") == "boosted" or tried.count(action_value) >= 1:
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.LOCAL_REPAIR.value:
        if _same_action_unproductive_replays(ledger, action_value) >= 3:
            return []
        if not str(ledger.active_object_id or ledger.active_target or "").strip():
            return []
        return list(card.predicted_mutations)

    if action_value == SemanticActionType.SCOPE_EXPAND.value:
        if _scope_already_global(ledger):
            return []
        if _same_action_unproductive_replays(ledger, action_value) >= 2:
            return []
        return list(card.predicted_mutations)

    return list(card.predicted_mutations)


def expensive_action_needs_escape(
    action: str | SemanticActionType,
    *,
    ledger: RecoveryLedger,
    frontier: Any | None = None,
) -> bool:
    card = get_action_card(action)
    if card.cost_class != "expensive":
        return False
    if predict_mutations(card.action, ledger=ledger, frontier=frontier):
        return False
    return _escape_hatch_conditions(ledger, frontier)


def _escape_hatch_conditions(ledger: RecoveryLedger, frontier: Any | None) -> bool:
    active_id = str(ledger.active_object_id or "").strip()
    if not active_id:
        return False
    if int(ledger.metadata.get("escape_grants_used", 0) or 0) >= 1:
        return False
    if _consecutive_cheap_actions(ledger) < 3:
        return False
    if frontier is not None:
        unresolved = list(getattr(frontier, "unresolved_objects", []) or [])
        resolved = list(getattr(frontier, "resolved_objects", []) or [])
        exhausted = list(getattr(frontier, "exhausted_objects", []) or [])
        if not unresolved or resolved or exhausted:
            return False
    return True


def _active_lifecycle_state(ledger: RecoveryLedger, frontier: Any | None) -> str:
    if frontier is not None:
        states = dict(getattr(frontier, "object_states", {}) or {})
        active = str(getattr(frontier, "active_object_id", "") or ledger.active_object_id or "")
        value = str(states.get(active, "") or "").strip()
        if value:
            return value
    structured = dict(ledger.structured_state or {})
    active = str(ledger.active_object_id or "").strip()
    for item in list(structured.get("object_chain_view", []) or []):
        if not isinstance(item, dict):
            continue
        if active and str(item.get("object_id", "") or "") != active:
            continue
        payload = dict(item.get("payload", {}) or {})
        metadata = dict(payload.get("metadata", {}) or {})
        return str(metadata.get("lifecycle_state", payload.get("lifecycle_state", "suspicious")) or "suspicious")
    return "suspicious"


def _last_action_result_mode(ledger: RecoveryLedger) -> str:
    guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
    if guard:
        return str(guard.get("result_mode", "") or "")
    guards = [dict(item) for item in list(ledger.metadata.get("guard_history", []) or []) if isinstance(item, dict)]
    if guards:
        return str(guards[-1].get("result_mode", "") or "")
    return ""


def _same_action_unproductive_replays(ledger: RecoveryLedger, action_value: str) -> int:
    count = 0
    histories = [
        dict(item)
        for item in list(ledger.metadata.get("replay_cost_history", []) or [])
        if isinstance(item, dict)
    ]
    guards = [
        dict(item)
        for item in list(ledger.metadata.get("guard_history", []) or [])
        if isinstance(item, dict)
    ]
    for index, item in enumerate(histories):
        if str(item.get("action", "") or "") != action_value:
            continue
        guard = guards[index] if index < len(guards) else {}
        mode = str(guard.get("result_mode", "") or "")
        fresh = dict(guard.get("fresh_changed_file_classes", {}) or {}).get("source_files", [])
        if mode in {"no_diff", "unproductive_no_state_change", "wrong_edit_target"} and not fresh:
            count += 1
    return count


def _has_viable_anchor(ledger: RecoveryLedger) -> bool:
    replay = dict(ledger.structured_state.get("replay_anchor_view", {}) or {}) if ledger.structured_state else {}
    return bool(
        replay.get("healthy_anchor_candidates")
        or replay.get("checkpoint_labels_available")
        or ledger.metadata.get("checkpoint_candidates")
    )


def _has_untried_anchor(ledger: RecoveryLedger) -> bool:
    replay = dict(ledger.structured_state.get("replay_anchor_view", {}) or {}) if ledger.structured_state else {}
    anchors = [
        str(item)
        for item in list(replay.get("healthy_anchor_candidates", []) or [])
        if str(item).strip()
    ]
    last_anchor = str(ledger.metadata.get("last_anchor", "") or "")
    return bool([anchor for anchor in anchors if anchor != last_anchor])


def _scope_already_global(ledger: RecoveryLedger) -> bool:
    if str(ledger.metadata.get("last_replay_scope", "") or "").strip().lower() == "full":
        return True
    for item in reversed(list(ledger.metadata.get("replay_cost_history", []) or [])):
        if not isinstance(item, dict):
            continue
        scope = str(item.get("scope", "") or "").strip().lower()
        if scope:
            return scope == "full"
    return False


def _consecutive_cheap_actions(ledger: RecoveryLedger) -> int:
    cheap = {
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.RECHECK_OBJECT.value,
        SemanticActionType.REVOKE_OBJECT.value,
        SemanticActionType.TARGET_RESET.value,
    }
    count = 0
    for action in reversed(list(ledger.tried_actions or [])):
        if str(action) not in cheap:
            break
        count += 1
    return count
