"""Intra-episode structured belief revision for CAR.

This module keeps CAR's reflection signal inside the current recovery run.  It
does not retrieve cross-instance history and it does not call an LLM.  Instead,
it records typed events from the ledger, guard outcome, and episode memory so
the controller can avoid repeating invalidated beliefs and explain each action.
"""

from __future__ import annotations

import time
from typing import Any

from bcmr_swe.recovery.action_cards import action_cost_class
from bcmr_swe.types import RecoveryLedger, SemanticActionType


CHEAP_BELIEF_ACTIONS = {
    SemanticActionType.EVIDENCE_RECHECK.value,
    SemanticActionType.RECHECK_OBJECT.value,
    SemanticActionType.REVOKE_OBJECT.value,
    SemanticActionType.TARGET_RESET.value,
}

EXPENSIVE_ACTIONS = {
    SemanticActionType.LOCAL_REPAIR.value,
    SemanticActionType.REPAIR_LOCAL.value,
    SemanticActionType.SCOPE_EXPAND.value,
    SemanticActionType.EXPAND_SCOPE.value,
}

NO_PROGRESS_MODES = {
    "no_diff",
    "contract_violation_no_fresh_source",
    "intent_violation_no_fresh_source",
    "wrong_edit_target",
    "budget_exhausted",
    "unproductive_no_state_change",
}

PROGRESS_MODES = {
    "strong_source_success",
    "oracle_failed_after_source_edit",
    "contract_violation_after_source_edit",
    "intent_violation_missed_target",
    "intent_violation_revoked_target",
    "intent_violation_too_broad",
    "intent_violation_after_source_edit",
    "source_edit_pending_official",
}

SOURCE_BOUNDARY_FAILURE_MODES = {
    "no_diff",
    "contract_violation_no_fresh_source",
    "intent_violation_no_fresh_source",
    "wrong_edit_target",
}


def update_recovery_ledgers_from_guard(
    ledger: RecoveryLedger,
    *,
    action_taken: str,
    guard: dict[str, Any],
) -> dict[str, Any]:
    """Update in-run negative/unknown ledgers from one action outcome.

    The entries are deliberately compact and enum-like so they can be surfaced
    to the controller and LLM prompts without re-reading full traces.
    """

    action = _canonical_action(action_taken)
    mode = str(guard.get("result_mode", "") or "")
    now = time.time()
    source_files = _guard_source_files(guard)
    target_paths = _guard_intent_target_paths(guard)
    intent_flags = _guard_intent_flags(guard)
    negative_entries = _ledger_entries(ledger, "negative_ledger")
    unknown_entries = _ledger_entries(ledger, "unknown_ledger")
    added_negative: list[dict[str, Any]] = []
    added_unknown: list[dict[str, Any]] = []

    if mode in NO_PROGRESS_MODES or "intent_no_fresh_source_diff" in intent_flags:
        added_negative.append(
            _ledger_event(
                created_at=now,
                kind="route_without_fresh_source",
                action=action,
                result_mode=mode,
                paths=target_paths,
                reason="action produced no fresh source change",
            )
        )
        if target_paths:
            added_unknown.append(
                _ledger_event(
                    created_at=now,
                    kind="target_boundary_unverified",
                    action=action,
                    result_mode=mode,
                    paths=target_paths,
                    reason="target path was selected but not changed",
                )
            )
    if mode in {
        "oracle_failed_after_source_edit",
        "contract_violation_after_source_edit",
        "intent_violation_after_source_edit",
        "intent_violation_missed_target",
        "intent_violation_too_broad",
        "source_edit_pending_official",
    }:
        paths = source_files or target_paths
        added_unknown.append(
            _ledger_event(
                created_at=now,
                kind="source_candidate_unresolved",
                action=action,
                result_mode=mode,
                paths=paths,
                reason="source candidate changed state but is not validated",
            )
        )
    if "intent_missed_target_path" in intent_flags:
        added_negative.append(
            _ledger_event(
                created_at=now,
                kind="off_target_source_edit",
                action=action,
                result_mode=mode,
                paths=source_files,
                reason="patch changed source outside intended target",
            )
        )
        if target_paths:
            added_unknown.append(
                _ledger_event(
                    created_at=now,
                    kind="intended_target_untried",
                    action=action,
                    result_mode=mode,
                    paths=target_paths,
                    reason="intended target still needs direct validation",
                )
            )
    guard_flags = {
        str(item)
        for item in list(guard.get("guard_flags", []) or [])
        if str(item).strip()
    }
    validation_not_actionable = bool(
        {
            "intent_missing_focused_validation",
            "focused_validation_not_target_related",
            "focused_validation_missing_result",
        }
        & (intent_flags | guard_flags)
    )
    if validation_not_actionable:
        added_unknown.append(
            _ledger_event(
                created_at=now,
                kind="validation_pending",
                action=action,
                result_mode=mode,
                paths=source_files or target_paths,
                reason="source change needs target-related focused validation",
            )
        )
    if mode == "strong_source_success":
        added_unknown = []

    for event in added_negative:
        _append_unique_event(negative_entries, event)
    for event in added_unknown:
        _append_unique_event(unknown_entries, event)
    ledger.metadata["negative_ledger"] = negative_entries[-24:]
    ledger.metadata["unknown_ledger"] = _filter_unknown_against_negative(
        unknown_entries,
        negative_entries,
    )[-24:]
    latest = {
        "schema": "car_recovery_ledgers_update_v1",
        "action": action,
        "result_mode": mode,
        "added_negative_count": len(added_negative),
        "added_unknown_count": len(added_unknown),
        "negative_kinds": [event["kind"] for event in added_negative],
        "unknown_kinds": [event["kind"] for event in added_unknown],
    }
    ledger.metadata["latest_recovery_ledgers_update"] = latest
    return latest


def record_belief_revision_event(
    ledger: RecoveryLedger,
    *,
    action_taken: str,
    result_mode: str,
    state_changed: bool,
    counterexample_type: str = "",
    token_cost: float = 0.0,
    latency_sec: float = 0.0,
) -> dict[str, Any]:
    """Append a typed in-run reflection event to the ledger."""

    action = _canonical_action(action_taken)
    mode = str(result_mode or "")
    object_id = str(ledger.active_object_id or "")
    object_type = str(ledger.active_object_type or "")
    target = str(ledger.active_target or "")
    revision_type = _revision_type(
        action=action,
        result_mode=mode,
        state_changed=state_changed,
    )
    initial_budget = float(ledger.metadata.get("initial_token_budget", 0.0) or 0.0)
    event = {
        "schema": "car_belief_revision_event_v1",
        "created_at": time.time(),
        "action": action,
        "counterexample_type": str(counterexample_type or ""),
        "active_object_id": object_id,
        "active_object_type": object_type,
        "active_target": target,
        "result_mode": mode,
        "revision_type": revision_type,
        "state_changed": bool(state_changed),
        "token_cost": float(token_cost or 0.0),
        "latency_sec": float(latency_sec or 0.0),
        "expensive_action": action in EXPENSIVE_ACTIONS,
        "high_cost_action": bool(initial_budget > 0.0 and float(token_cost or 0.0) >= 0.3 * initial_budget),
        "next_action_hint": next_action_hint_for_revision(revision_type),
    }
    history = _belief_revision_history(ledger)
    history.append(event)
    ledger.metadata["belief_revision_history"] = history[-16:]
    ledger.metadata["latest_belief_revision_event"] = event
    if revision_type in {"revoked", "invalidated_no_progress"} and object_id:
        invalidated = [str(item) for item in list(ledger.invalidated_object_ids or [])]
        if object_id not in invalidated:
            invalidated.append(object_id)
            ledger.invalidated_object_ids = invalidated
    return event


def belief_revision_signal(ledger: RecoveryLedger) -> dict[str, Any]:
    """Return a compact controller signal from this run's revision history."""

    history = _belief_revision_history(ledger)
    no_progress_events = [
        event for event in history if str(event.get("revision_type", "")) == "invalidated_no_progress"
    ]
    recently_revoked = [
        event for event in history if str(event.get("revision_type", "")) in {"revoked", "invalidated_no_progress"}
    ]
    repeated_expensive_no_progress = _repeated_expensive_no_progress(history)
    budget_fraction_used = _budget_fraction_used(ledger)
    latest = dict(history[-1]) if history else {}
    latest_revision = str(latest.get("revision_type", "") or "")
    latest_action = str(latest.get("action", "") or "")
    latest_requires_evidence = latest_revision in {
        "candidate_preserved_unverified",
        "belief_retargeted",
        "invalidated_no_progress",
    } and latest_action not in {
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.RECHECK_OBJECT.value,
    }
    expensive_nonterminal_count = sum(
        1
        for event in history
        if bool(event.get("expensive_action", False))
        and str(event.get("revision_type", "") or "") != "reinforced_source_candidate"
    )
    high_cost_nonterminal = bool(
        latest.get("high_cost_action", False)
        and latest_revision != "reinforced_source_candidate"
    )
    pause_expensive_replay = (
        latest_requires_evidence
        or high_cost_nonterminal
        or (expensive_nonterminal_count >= 2 and budget_fraction_used >= 0.45)
    )
    source_boundary_signal = _source_boundary_signal(
        ledger,
        history=history,
        latest_revision=latest_revision,
        latest_requires_evidence=latest_requires_evidence,
    )
    ledger_signal = _recovery_ledger_signal(ledger)
    stop_expensive_replay = (
        repeated_expensive_no_progress >= 2
        and (budget_fraction_used >= 0.7 or int(ledger.remaining_step_budget or 0) <= 1)
    )
    return {
        "event_count": len(history),
        "latest_event": latest,
        "latest_revision_type": latest_revision,
        "recently_revoked_object_id": str(recently_revoked[-1].get("active_object_id", "") if recently_revoked else ""),
        "no_progress_event_count": len(no_progress_events),
        "repeated_expensive_no_progress": repeated_expensive_no_progress,
        "expensive_nonterminal_count": expensive_nonterminal_count,
        "needs_cheap_evidence_before_expensive": latest_requires_evidence,
        "pause_expensive_replay": pause_expensive_replay,
        "budget_fraction_used": budget_fraction_used,
        "stop_expensive_replay": stop_expensive_replay,
        **source_boundary_signal,
        **ledger_signal,
        "preferred_cheap_actions": preferred_cheap_actions(ledger),
    }


def score_adjustment_for_action(action: str | SemanticActionType, signal: dict[str, Any]) -> float:
    action_value = _canonical_action(action)
    if not signal:
        return 0.0
    score = 0.0
    hint = str(dict(signal.get("latest_event", {}) or {}).get("next_action_hint", "") or "")
    if hint and action_value == hint:
        score += 920.0 if bool(signal.get("needs_cheap_evidence_before_expensive", False)) else 180.0
    if bool(signal.get("stop_expensive_replay", False)) or bool(signal.get("pause_expensive_replay", False)):
        if action_value in CHEAP_BELIEF_ACTIONS:
            score += 260.0
        elif action_value in EXPENSIVE_ACTIONS:
            score -= 900.0
    elif int(signal.get("no_progress_event_count", 0) or 0) > 0:
        if action_value in CHEAP_BELIEF_ACTIONS:
            score += 80.0
    if bool(signal.get("source_boundary_retarget_ready", False)):
        if action_value == SemanticActionType.SCOPE_EXPAND.value:
            score += 780.0
        elif action_value == SemanticActionType.LOCAL_REPAIR.value:
            score -= 520.0
        elif action_value in {
            SemanticActionType.EVIDENCE_RECHECK.value,
            SemanticActionType.RECHECK_OBJECT.value,
        }:
            score -= 350.0
    elif bool(signal.get("source_boundary_invalidated", False)):
        if action_value == SemanticActionType.SCOPE_EXPAND.value:
            score += 120.0
    if bool(signal.get("has_unresolved_source_candidate", False)):
        if action_value == SemanticActionType.LOCAL_REPAIR.value:
            score += 260.0
        elif action_value == SemanticActionType.SCOPE_EXPAND.value:
            score -= 220.0
    if bool(signal.get("has_pending_validation", False)):
        if action_value in {
            SemanticActionType.EVIDENCE_RECHECK.value,
            SemanticActionType.RECHECK_OBJECT.value,
        }:
            score += 520.0
        elif action_value in {
            SemanticActionType.TARGET_RESET.value,
            SemanticActionType.REVOKE_OBJECT.value,
        }:
            score -= 220.0
        elif action_value in EXPENSIVE_ACTIONS:
            score -= 180.0
    if bool(signal.get("has_route_without_fresh_source", False)):
        if action_value == SemanticActionType.LOCAL_REPAIR.value:
            score -= 260.0
        elif action_value == SemanticActionType.SCOPE_EXPAND.value:
            score += 180.0
        elif action_value in CHEAP_BELIEF_ACTIONS:
            score += 90.0
    return score


def should_reject_action_for_budget(
    action: str | SemanticActionType,
    signal: dict[str, Any],
) -> bool:
    if not (
        bool(signal.get("stop_expensive_replay", False))
        or bool(signal.get("pause_expensive_replay", False))
    ):
        return False
    return action_cost_class(action) == "expensive"


def should_reject_action_for_recovery_ledger(
    action: str | SemanticActionType,
    signal: dict[str, Any],
) -> bool:
    action_value = _canonical_action(action)
    if not bool(signal.get("has_pending_validation", False)):
        return False
    preferred = {
        str(item)
        for item in list(signal.get("preferred_cheap_actions", []) or [])
        if str(item).strip()
    }
    validation_actions = {
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.RECHECK_OBJECT.value,
    }
    if not preferred.intersection(validation_actions):
        return False
    return action_value not in validation_actions


def preferred_cheap_actions(ledger: RecoveryLedger) -> list[str]:
    tried = {str(item) for item in list(ledger.tried_actions or []) if str(item).strip()}
    ordered = [
        SemanticActionType.EVIDENCE_RECHECK.value,
        SemanticActionType.REVOKE_OBJECT.value,
        SemanticActionType.TARGET_RESET.value,
        SemanticActionType.RECHECK_OBJECT.value,
    ]
    return [action for action in ordered if action not in tried]


def next_action_hint_for_revision(revision_type: str) -> str:
    if revision_type == "invalidated_no_progress":
        return SemanticActionType.EVIDENCE_RECHECK.value
    if revision_type == "candidate_preserved_unverified":
        return SemanticActionType.EVIDENCE_RECHECK.value
    if revision_type == "belief_retargeted":
        return SemanticActionType.EVIDENCE_RECHECK.value
    if revision_type == "revoked":
        return SemanticActionType.TARGET_RESET.value
    if revision_type == "reinforced_source_candidate":
        return SemanticActionType.LOCAL_REPAIR.value
    if revision_type == "observed_evidence":
        return SemanticActionType.LOCAL_REPAIR.value
    return ""


def _revision_type(*, action: str, result_mode: str, state_changed: bool) -> str:
    if action == SemanticActionType.REVOKE_OBJECT.value:
        return "revoked"
    if action == SemanticActionType.TARGET_RESET.value and state_changed:
        return "belief_retargeted"
    if action in {SemanticActionType.EVIDENCE_RECHECK.value, SemanticActionType.RECHECK_OBJECT.value}:
        return "observed_evidence" if state_changed else "evidence_unchanged"
    if result_mode in NO_PROGRESS_MODES:
        return "invalidated_no_progress"
    if result_mode == "strong_source_success":
        return "reinforced_source_candidate"
    if result_mode in PROGRESS_MODES:
        return "candidate_preserved_unverified"
    if state_changed:
        return "observed_evidence"
    return "unresolved"


def _belief_revision_history(ledger: RecoveryLedger) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(ledger.metadata.get("belief_revision_history", []) or [])
        if isinstance(item, dict)
    ]


def _recovery_ledger_signal(ledger: RecoveryLedger) -> dict[str, Any]:
    negative = _ledger_entries(ledger, "negative_ledger")
    unknown = _filter_unknown_against_negative(
        _ledger_entries(ledger, "unknown_ledger"),
        negative,
    )
    unknown_kinds = {
        str(item.get("kind", "") or "")
        for item in unknown
        if str(item.get("kind", "") or "")
    }
    negative_kinds = {
        str(item.get("kind", "") or "")
        for item in negative
        if str(item.get("kind", "") or "")
    }
    pending_paths = []
    for event in unknown:
        for path in list(event.get("paths", []) or []):
            if str(path).strip() and str(path) not in pending_paths:
                pending_paths.append(str(path))
    return {
        "negative_ledger_count": len(negative),
        "unknown_ledger_count": len(unknown),
        "negative_ledger_kinds": sorted(negative_kinds),
        "unknown_ledger_kinds": sorted(unknown_kinds),
        "has_route_without_fresh_source": "route_without_fresh_source" in negative_kinds,
        "has_unresolved_source_candidate": "source_candidate_unresolved" in unknown_kinds,
        "has_pending_validation": "validation_pending" in unknown_kinds,
        "pending_unknown_paths": pending_paths[:6],
    }


def _source_boundary_signal(
    ledger: RecoveryLedger,
    *,
    history: list[dict[str, Any]],
    latest_revision: str,
    latest_requires_evidence: bool,
) -> dict[str, Any]:
    boundary_events = [
        event
        for event in history
        if str(event.get("result_mode", "") or "") in SOURCE_BOUNDARY_FAILURE_MODES
        and str(event.get("action", "") or "") in EXPENSIVE_ACTIONS
    ]
    latest_guard = _latest_guard(ledger)
    latest_guard_mode = str(latest_guard.get("result_mode", "") or "")
    latest_guard_suggestion = str(latest_guard.get("next_action_suggestion", "") or "")
    guard_points_to_boundary = (
        latest_guard_mode in SOURCE_BOUNDARY_FAILURE_MODES
        or latest_guard_suggestion == SemanticActionType.SCOPE_EXPAND.value
    )
    source_candidate_available = _has_source_candidate(ledger)
    invalidated = bool(boundary_events or guard_points_to_boundary) and not source_candidate_available
    retarget_ready = bool(
        invalidated
        and boundary_events
        and not latest_requires_evidence
        and latest_revision in {"observed_evidence", "evidence_unchanged", ""}
    )
    return {
        "source_boundary_invalidated": invalidated,
        "source_boundary_no_progress_count": len(boundary_events),
        "source_boundary_retarget_ready": retarget_ready,
        "preferred_expensive_action": SemanticActionType.SCOPE_EXPAND.value if retarget_ready else "",
    }


def _repeated_expensive_no_progress(history: list[dict[str, Any]]) -> int:
    count = 0
    for event in reversed(history):
        action = str(event.get("action", "") or "")
        revision = str(event.get("revision_type", "") or "")
        if action in EXPENSIVE_ACTIONS and revision == "invalidated_no_progress":
            count += 1
            continue
        if revision in {"observed_evidence", "evidence_unchanged"}:
            continue
        if count:
            break
    return count


def _budget_fraction_used(ledger: RecoveryLedger) -> float:
    initial = float(ledger.metadata.get("initial_token_budget", 0.0) or 0.0)
    remaining = float(ledger.remaining_token_budget or 0.0)
    if initial <= 0.0:
        return 0.0
    used = max(0.0, min(initial, initial - remaining))
    return used / initial


def _latest_guard(ledger: RecoveryLedger) -> dict[str, Any]:
    guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
    if guard:
        return guard
    guards = [
        dict(item)
        for item in list(ledger.metadata.get("guard_history", []) or [])
        if isinstance(item, dict)
    ]
    return guards[-1] if guards else {}


def _has_source_candidate(ledger: RecoveryLedger) -> bool:
    for item in list(ledger.metadata.get("source_candidate_memory", []) or []):
        if not isinstance(item, dict):
            continue
        if list(item.get("fresh_source_files", []) or item.get("source_files", []) or []):
            return True
    return False


def _ledger_entries(ledger: RecoveryLedger, key: str) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(ledger.metadata.get(key, []) or [])
        if isinstance(item, dict)
    ]


def _ledger_event(
    *,
    created_at: float,
    kind: str,
    action: str,
    result_mode: str,
    paths: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "car_recovery_ledger_event_v1",
        "created_at": float(created_at),
        "kind": kind,
        "action": action,
        "result_mode": result_mode,
        "paths": _dedupe(paths)[:8],
        "reason": reason,
    }


def _append_unique_event(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    fingerprint = (
        str(event.get("kind", "") or ""),
        str(event.get("action", "") or ""),
        str(event.get("result_mode", "") or ""),
        tuple(str(path) for path in list(event.get("paths", []) or [])),
    )
    for existing in events:
        existing_fingerprint = (
            str(existing.get("kind", "") or ""),
            str(existing.get("action", "") or ""),
            str(existing.get("result_mode", "") or ""),
            tuple(str(path) for path in list(existing.get("paths", []) or [])),
        )
        if existing_fingerprint == fingerprint:
            existing["created_at"] = event["created_at"]
            return
    events.append(event)


def _filter_unknown_against_negative(
    unknown: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    negative_keys = {
        (
            str(item.get("kind", "") or ""),
            tuple(str(path) for path in list(item.get("paths", []) or [])),
        )
        for item in negative
    }
    filtered = []
    for event in unknown:
        key = (
            str(event.get("kind", "") or ""),
            tuple(str(path) for path in list(event.get("paths", []) or [])),
        )
        if key not in negative_keys:
            filtered.append(event)
    return filtered


def _guard_source_files(guard: dict[str, Any]) -> list[str]:
    classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
    paths = list(classes.get("source_files", []) or [])
    if not paths:
        classes = dict(guard.get("changed_file_classes", {}) or {})
        paths = list(classes.get("source_files", []) or [])
    return _dedupe(str(path) for path in paths if str(path).strip())


def _guard_intent_target_paths(guard: dict[str, Any]) -> list[str]:
    intent = dict(guard.get("patch_intent", {}) or {})
    paths = list(dict(intent.get("paths", {}) or {}).get("target_paths", []) or [])
    if not paths:
        paths = list(guard.get("suspect_paths", []) or [])
    return _dedupe(str(path) for path in paths if str(path).strip())


def _guard_intent_flags(guard: dict[str, Any]) -> set[str]:
    audit = dict(guard.get("patch_intent_audit", {}) or {})
    flags = {
        str(item)
        for item in list(audit.get("flags", []) or [])
        if str(item).strip()
    }
    flags.update(
        str(item)
        for item in list(audit.get("hard_flags", []) or [])
        if str(item).strip()
    )
    flags.update(
        str(item)
        for item in list(guard.get("guard_flags", []) or [])
        if str(item).strip()
    )
    return flags


def _dedupe(values: Any) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _canonical_action(action: str | SemanticActionType) -> str:
    if isinstance(action, SemanticActionType):
        return action.value
    text = str(action or "")
    if text == SemanticActionType.REPAIR_LOCAL.value:
        return SemanticActionType.LOCAL_REPAIR.value
    if text == SemanticActionType.EXPAND_SCOPE.value:
        return SemanticActionType.SCOPE_EXPAND.value
    return text
