"""CAR action selector.

This selector is the first executable slice of Counterexample-Aware Recovery:
classify the current counterexample, then rank the same bounded PARC action
space by type-specific priorities and zero-token mutation prediction.
"""

from __future__ import annotations

from typing import Any

from bcmr_swe.recovery.action_cards import ActionCard, available_action_cards
from bcmr_swe.recovery.belief_revision import (
    belief_revision_signal,
    record_belief_revision_event,
    score_adjustment_for_action,
    should_reject_action_for_budget,
    should_reject_action_for_recovery_ledger,
    update_recovery_ledgers_from_guard,
)
from bcmr_swe.recovery.budgeted_controller import ControllerDecision, compute_recovery_frontier
from bcmr_swe.recovery.counterexample import (
    ACTION_PRIORITY,
    NEVER_FIRST,
    CounterexampleClassification,
    CounterexampleType,
    classify_counterexample_from_ledger,
)
from bcmr_swe.recovery.episode_memory import (
    ActionPrior,
    history_action_priors_from_ledger,
    local_action_prior,
    make_recovery_episode,
    merge_priors,
    remember_episode,
)
from bcmr_swe.recovery.mutation_predictor import expensive_action_needs_escape, predict_mutations
from bcmr_swe.types import RecoveryLedger, SemanticActionType


CONVERGENCE_ACTIONS = {
    SemanticActionType.LOCAL_REPAIR,
    SemanticActionType.EVIDENCE_RECHECK,
    SemanticActionType.CAPABILITY_BOOST,
}


def select_car_action(
    ledger: RecoveryLedger,
    *,
    current_suggestion: str = "",
) -> ControllerDecision | None:
    frontier = compute_recovery_frontier(ledger)
    classification = classify_counterexample_from_ledger(ledger)
    ledger.metadata["latest_car_counterexample"] = classification.to_dict()
    if frontier.recovery_done:
        decision = ControllerDecision(
            action="",
            reason="recovery_goal_satisfied",
            score=float("inf"),
            frontier=frontier,
        )
        _record_car_event(ledger, decision, classification, runner_up={})
        return decision

    cards = available_action_cards(ledger)
    if not cards:
        return None

    candidate_actions = [card.action.value for card in cards]
    cross_sample_enabled = bool(ledger.metadata.get("car_cross_sample_prior_enabled", False))
    historical_priors = (
        history_action_priors_from_ledger(
            ledger,
            counterexample_type=classification.counterexample_type.value,
            candidate_actions=candidate_actions,
        )
        if cross_sample_enabled
        else {}
    )
    local_priors = {
        action: local_action_prior(ledger, action)
        for action in candidate_actions
    }
    priors = {
        action: merge_priors(
            local_priors.get(action, ActionPrior(action=action)),
            historical_priors.get(action, ActionPrior(action=action)),
        )
        for action in candidate_actions
    }
    convergence = _source_candidate_convergence_state(ledger)
    revision_signal = belief_revision_signal(ledger)
    ledger.metadata["latest_belief_revision_signal"] = revision_signal
    best: tuple[float, ActionCard, list[str], bool] | None = None
    runner_up: dict[str, Any] = {}
    rejected: list[dict[str, Any]] = []
    first_decision = not any(str(item).strip() for item in list(ledger.tried_actions or []))
    never_first = NEVER_FIRST.get(classification.counterexample_type, set()) if first_decision else set()

    for card in cards:
        if should_reject_action_for_budget(card.action, revision_signal):
            if _candidate_refine_slot_overrides_budget_halt(card.action, convergence=convergence):
                pass
            elif _candidate_exhausted_boundary_switch_overrides_budget_halt(card.action, convergence=convergence):
                pass
            elif (
                card.action == SemanticActionType.SCOPE_EXPAND
                and revision_signal.get("source_boundary_retarget_ready")
            ):
                mutations = predict_mutations(card.action, ledger=ledger, frontier=frontier)
                if mutations:
                    prior = priors.get(card.action.value, ActionPrior(action=card.action.value))
                    score = _score_car_candidate(
                        card,
                        mutations,
                        classification=classification,
                        current_suggestion=current_suggestion,
                        escape=False,
                        prior=prior,
                        convergence=convergence,
                        belief_revision=revision_signal,
                    )
                    score += 1200.0
                    if best is None or score > best[0]:
                        if best is not None:
                            runner_up = {
                                "action": best[1].action.value,
                                "score": best[0],
                                "mutations": list(best[2]),
                                "escape_granted": best[3],
                                "prior": priors.get(
                                    best[1].action.value,
                                    ActionPrior(action=best[1].action.value),
                                ).to_dict(),
                            }
                        best = (score, card, mutations, False)
                    elif not runner_up or score > float(runner_up.get("score", float("-inf"))):
                        runner_up = {
                            "action": card.action.value,
                            "score": score,
                            "mutations": list(mutations),
                            "escape_granted": False,
                            "prior": prior.to_dict(),
                        }
                    continue
                _reject_budget_halt(rejected, card=card, revision_signal=revision_signal)
                continue
            else:
                _reject_budget_halt(rejected, card=card, revision_signal=revision_signal)
                continue
        if should_reject_action_for_recovery_ledger(
            card.action,
            revision_signal,
        ) and not _candidate_refine_slot_overrides_validation_gate(
            card.action,
            convergence=convergence,
        ):
            rejected.append(
                {
                    "action": card.action.value,
                    "reason": "recovery_ledger_requires_validation",
                    "belief_revision_signal": revision_signal,
                }
            )
            continue
        if card.action in never_first:
            rejected.append(
                {
                    "action": card.action.value,
                    "reason": "never_first_for_counterexample_type",
                    "counterexample_type": classification.counterexample_type.value,
                }
            )
            continue
        allowed_convergence_actions = _allowed_convergence_actions(convergence)
        if convergence and card.action not in allowed_convergence_actions:
            rejected.append(
                {
                    "action": card.action.value,
                    "reason": "source_candidate_convergence_prefers_refine",
                    "convergence": convergence,
                }
            )
            continue
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
        prior = priors.get(card.action.value, ActionPrior(action=card.action.value))
        prior_rejection = _action_prior_rejection(
            card,
            local_prior=local_priors.get(card.action.value, ActionPrior(action=card.action.value)),
            historical_prior=historical_priors.get(card.action.value, ActionPrior(action=card.action.value)),
            merged_prior=prior,
        )
        if prior_rejection and not _candidate_refine_slot_overrides_negative_prior(
            card.action,
            convergence=convergence,
        ):
            rejected.append(
                {
                    "action": card.action.value,
                    "reason": "negative_episode_prior",
                    "prior_scope": prior_rejection,
                    "prior": prior.to_dict(),
                }
            )
            continue
        score = _score_car_candidate(
            card,
            mutations,
            classification=classification,
            current_suggestion=current_suggestion,
            escape=escape,
            prior=prior,
            convergence=convergence,
            belief_revision=revision_signal,
        )
        if best is None or score > best[0]:
            if best is not None:
                runner_up = {
                    "action": best[1].action.value,
                    "score": best[0],
                    "mutations": list(best[2]),
                    "escape_granted": best[3],
                    "prior": priors.get(
                        best[1].action.value,
                        ActionPrior(action=best[1].action.value),
                    ).to_dict(),
                }
            best = (score, card, mutations, escape)
        elif not runner_up or score > float(runner_up.get("score", float("-inf"))):
            runner_up = {
                "action": card.action.value,
                "score": score,
                "mutations": list(mutations),
                "escape_granted": escape,
                "prior": prior.to_dict(),
            }

    if best is None:
        decision = ControllerDecision(
            action="",
            reason="no_viable_action",
            score=float("-inf"),
            frontier=frontier,
            rejected=rejected,
        )
        _record_car_event(ledger, decision, classification, runner_up={})
        return decision

    score, card, mutations, escape = best
    frontier.predicted_state_mutations[card.action.value] = mutations
    decision = ControllerDecision(
        action=card.action.value,
        reason="car_counterexample_controller",
        score=score,
        frontier=frontier,
        rejected=rejected,
        escape_granted=escape,
    )
    _record_car_event(ledger, decision, classification, runner_up=runner_up)
    _record_car_prior_audit(
        ledger,
        priors=priors,
        historical_priors=historical_priors,
        candidate_actions=candidate_actions,
        cross_sample_enabled=cross_sample_enabled,
    )
    if escape:
        ledger.metadata["escape_grants_used"] = int(ledger.metadata.get("escape_grants_used", 0) or 0) + 1
    return decision


def _score_car_candidate(
    card: ActionCard,
    mutations: list[str],
    *,
    classification: CounterexampleClassification,
    current_suggestion: str,
    escape: bool,
    prior: ActionPrior | None = None,
    convergence: dict[str, Any] | None = None,
    belief_revision: dict[str, Any] | None = None,
) -> float:
    prior = prior or ActionPrior(action=card.action.value)
    convergence = convergence or {}
    priority = ACTION_PRIORITY.get(
        classification.counterexample_type,
        ACTION_PRIORITY[CounterexampleType.UNKNOWN],
    )
    if card.action in priority:
        index = priority.index(card.action)
        score = 1000.0 - float(index * 120)
    else:
        score = 100.0
    if mutations:
        score += 40.0
    if card.action.value == current_suggestion:
        score += 20.0
    score += float(prior.success_rate) * 300.0
    score += float(prior.trajectory_success_rate) * 220.0
    if prior.trajectory_success_sample_count <= 0:
        score -= float(prior.trajectory_failure_rate) * 180.0
    state_change_weight = 120.0 if prior.trajectory_failure_rate < 0.5 else 20.0
    score += float(prior.state_change_rate) * state_change_weight
    score -= float(prior.no_diff_rate) * 420.0
    if prior.sample_count > 0 and prior.last_verifier_delta in {"same", "worse"}:
        score -= 80.0
    if convergence:
        if card.action == SemanticActionType.LOCAL_REPAIR:
            score += 180.0
        elif card.action == SemanticActionType.EVIDENCE_RECHECK:
            score += 220.0 if convergence.get("needs_evidence_recheck") else 60.0
        if str(convergence.get("policy", "") or "") == "oracle_failure_evidence_refine_slot":
            if card.action == SemanticActionType.LOCAL_REPAIR:
                score += 1700.0
            elif card.action == SemanticActionType.EVIDENCE_RECHECK:
                score -= 1200.0
    score += score_adjustment_for_action(card.action, dict(belief_revision or {}))
    if card.cost_class == "cheap":
        score += 30.0
    elif card.cost_class == "medium":
        score -= 20.0
    else:
        score -= min(90.0, float(card.base_token_estimate) / 1000.0)
    if escape:
        score -= 50.0
    return score


def action_prior_from_episode_memory(
    ledger: RecoveryLedger,
    action: str | SemanticActionType,
) -> ActionPrior:
    action_value = str(action.value if isinstance(action, SemanticActionType) else action)
    counterexample_type = str(
        dict(ledger.metadata.get("latest_car_counterexample", {}) or {}).get(
            "counterexample_type",
            "",
        )
        or ""
    )
    historical_priors = history_action_priors_from_ledger(
        ledger,
        counterexample_type=counterexample_type,
        candidate_actions=[action_value],
    ) if bool(ledger.metadata.get("car_cross_sample_prior_enabled", False)) else {}
    return merge_priors(
        local_action_prior(ledger, action_value),
        historical_priors.get(action_value, ActionPrior(action=action_value)),
    )


def remember_car_episode(
    ledger: RecoveryLedger,
    *,
    action_taken: str,
    result_mode: str,
    token_cost: float = 0.0,
    latency_sec: float = 0.0,
    state_changed: bool | None = None,
    verifier_delta: str = "",
) -> dict[str, Any]:
    latest_decision = dict(ledger.metadata.get("latest_car_controller_decision", {}) or {})
    episode = make_recovery_episode(
        ledger,
        action_taken=action_taken,
        result_mode=result_mode,
        token_cost=token_cost,
        latency_sec=latency_sec,
        state_changed=state_changed,
        verifier_delta=verifier_delta,
        action_score=float(latest_decision.get("score", 0.0) or 0.0),
        action_was_first_choice=str(action_taken or "") == str(latest_decision.get("action", "") or ""),
    )
    remember_episode(
        ledger,
        episode,
        append_path=str(ledger.metadata.get("car_output_episodes_path", "") or "") or None,
    )
    guard = _latest_guard(ledger)
    ledger_update: dict[str, Any] = {}
    if guard:
        ledger_update = update_recovery_ledgers_from_guard(
            ledger,
            action_taken=action_taken,
            guard=guard,
        )
    record_belief_revision_event(
        ledger,
        action_taken=action_taken,
        result_mode=result_mode,
        state_changed=episode.action_changed_state,
        counterexample_type=episode.counterexample_type,
        token_cost=token_cost,
        latency_sec=latency_sec,
    )
    if ledger_update:
        latest = dict(ledger.metadata.get("latest_belief_revision_event", {}) or {})
        latest["recovery_ledgers_update"] = ledger_update
        ledger.metadata["latest_belief_revision_event"] = latest
    return episode.to_dict()


def _fresh_source_files_from_latest_guard(ledger: RecoveryLedger) -> list[str]:
    guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
    if not guard:
        guards = [
            dict(item)
            for item in list(ledger.metadata.get("guard_history", []) or [])
            if isinstance(item, dict)
        ]
        guard = guards[-1] if guards else {}
    classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
    return [
        str(item)
        for item in list(classes.get("source_files", []) or [])
        if str(item).strip()
    ]


def _verifier_delta_from_result_mode(result_mode: str) -> str:
    mode = str(result_mode or "")
    if mode == "strong_source_success":
        return "improved"
    if mode in {
        "oracle_failed_after_source_edit",
        "contract_violation_after_source_edit",
        "intent_violation_missed_target",
        "intent_violation_revoked_target",
        "intent_violation_too_broad",
        "intent_violation_after_source_edit",
    }:
        return "same"
    if mode in {"no_diff", "contract_violation_no_fresh_source", "intent_violation_no_fresh_source", "wrong_edit_target"}:
        return "same"
    return "unknown"


def _action_prior_rejection(
    card: ActionCard,
    *,
    local_prior: ActionPrior,
    historical_prior: ActionPrior,
    merged_prior: ActionPrior,
) -> str:
    if card.cost_class == "cheap":
        return ""
    if _local_prior_rejects(local_prior):
        return "local_episode"
    if _historical_prior_rejects(historical_prior):
        return "historical_episode"
    if (
        local_prior.sample_count > 0
        and historical_prior.sample_count > 0
        and _historical_prior_rejects(merged_prior)
    ):
        return "local_plus_history"
    return ""


def _local_prior_rejects(prior: ActionPrior) -> bool:
    if prior.sample_count <= 0:
        return False
    if prior.success_rate > 0.0:
        return False
    if prior.no_diff_lower_bound >= 0.5:
        return True
    if prior.no_diff_rate >= 1.0:
        return True
    if prior.sample_count >= 2 and prior.state_change_rate <= 0.25:
        return True
    return False


def _historical_prior_rejects(prior: ActionPrior) -> bool:
    if prior.sample_count <= 0:
        return False
    if prior.success_rate > 0.0:
        return False
    return prior.no_diff_lower_bound >= 0.5


def _record_car_event(
    ledger: RecoveryLedger,
    decision: ControllerDecision,
    classification: CounterexampleClassification,
    *,
    runner_up: dict[str, Any],
) -> None:
    events = [
        dict(item)
        for item in list(ledger.metadata.get("car_controller_events", []) or [])
        if isinstance(item, dict)
    ]
    event = {
        **decision.to_dict(),
        "counterexample": classification.to_dict(),
        "runner_up": dict(runner_up or {}),
        "priority_order": [
            action.value
            for action in ACTION_PRIORITY.get(
                classification.counterexample_type,
                ACTION_PRIORITY[CounterexampleType.UNKNOWN],
            )
        ],
        "source_candidate_convergence": _source_candidate_convergence_state(ledger),
    }
    events.append(event)
    ledger.metadata["car_controller_events"] = events[-12:]
    ledger.metadata["latest_car_controller_decision"] = event


def _record_car_prior_audit(
    ledger: RecoveryLedger,
    *,
    priors: dict[str, ActionPrior],
    historical_priors: dict[str, ActionPrior],
    candidate_actions: list[str],
    cross_sample_enabled: bool,
) -> None:
    history_actions = [
        action
        for action in candidate_actions
        if historical_priors.get(action, ActionPrior(action=action)).sample_count > 0
    ]
    history_sources = {
        action: historical_priors.get(action, ActionPrior(action=action)).source
        for action in history_actions
    }
    if not history_actions and not bool(ledger.metadata.get("car_full_audit_without_history", False)):
        audit = {
            "history_prior_count": 0,
            "history_prior_actions": [],
            "history_prior_sources": {},
            "history_prior_rejections": [],
            "candidate_priors": {},
            "audit_mode": "light_no_history_prior",
            "cross_sample_prior_enabled": bool(cross_sample_enabled),
            "main_prior_scope": "intra_episode_belief_revision",
            "belief_revision_signal": dict(ledger.metadata.get("latest_belief_revision_signal", {}) or {}),
        }
        ledger.metadata["car_prior_audit"] = audit
        latest = dict(ledger.metadata.get("latest_car_controller_decision", {}) or {})
        latest["prior_audit"] = audit
        ledger.metadata["latest_car_controller_decision"] = latest
        events = [
            dict(item)
            for item in list(ledger.metadata.get("car_controller_events", []) or [])
            if isinstance(item, dict)
        ]
        if events:
            events[-1]["prior_audit"] = audit
            ledger.metadata["car_controller_events"] = events[-12:]
        return
    rejections = [
        {
            "action": action,
            "sample_count": prior.sample_count,
            "no_diff_rate": prior.no_diff_rate,
            "no_diff_lower_bound": prior.no_diff_lower_bound,
            "success_rate": prior.success_rate,
            "source": prior.source,
        }
        for action, prior in priors.items()
        if prior.sample_count > 0 and prior.no_diff_lower_bound >= 0.5
    ]
    audit = {
        "history_prior_count": len(history_actions),
        "history_prior_actions": history_actions,
        "history_prior_sources": history_sources,
        "history_prior_rejections": rejections,
        "audit_mode": "full_history_prior",
        "cross_sample_prior_enabled": bool(cross_sample_enabled),
        "main_prior_scope": (
            "cross_sample_prior_ablation"
            if cross_sample_enabled
            else "intra_episode_belief_revision"
        ),
        "belief_revision_signal": dict(ledger.metadata.get("latest_belief_revision_signal", {}) or {}),
        "candidate_priors": {
            action: priors.get(action, ActionPrior(action=action)).to_dict()
            for action in candidate_actions
        },
    }
    ledger.metadata["car_prior_audit"] = audit
    latest = dict(ledger.metadata.get("latest_car_controller_decision", {}) or {})
    latest["prior_audit"] = audit
    ledger.metadata["latest_car_controller_decision"] = latest
    events = [
        dict(item)
        for item in list(ledger.metadata.get("car_controller_events", []) or [])
        if isinstance(item, dict)
    ]
    if events:
        events[-1]["prior_audit"] = audit
        ledger.metadata["car_controller_events"] = events[-12:]


def _source_candidate_convergence_state(ledger: RecoveryLedger) -> dict[str, Any]:
    candidate = _best_source_candidate_from_memory(ledger)
    if not candidate:
        return {}
    latest_mode = _latest_guard_result_mode(ledger)
    candidate_mode = str(candidate.get("result_mode", "") or "")
    fresh_source_files = [
        str(item)
        for item in list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or [])
        if str(item).strip()
    ]
    if not fresh_source_files:
        return {}
    actionable_modes = {
        "oracle_failed_after_source_edit",
        "contract_violation_after_source_edit",
        "intent_violation_missed_target",
        "intent_violation_revoked_target",
        "intent_violation_too_broad",
        "intent_violation_after_source_edit",
        "source_edit_pending_official",
    }
    if latest_mode not in actionable_modes and candidate_mode not in actionable_modes:
        return {}
    replay_count = _candidate_preserving_replay_count(ledger)
    touches_suspect_path = bool(candidate.get("touches_suspect_path", False))
    suspect_paths = [
        str(item)
        for item in list(getattr(ledger, "suspect_paths", []) or [])
        if str(item).strip()
    ]
    status = _source_candidate_status(
        candidate=candidate,
        latest_mode=latest_mode,
        replay_count=replay_count,
        touches_suspect_path=touches_suspect_path,
        suspect_paths=suspect_paths,
    )
    if status == "candidate_exhausted_after_refine":
        return {
            "candidate_mode": candidate_mode,
            "latest_mode": latest_mode,
            "fresh_source_files": fresh_source_files[:6],
            "touches_suspect_path": touches_suspect_path,
            "candidate_preserving_replay_count": replay_count,
            "candidate_status": status,
            "needs_evidence_recheck": False,
            "policy": "candidate_refine_exhausted_switch_boundary",
        }
    oracle_failure_evidence = _has_target_oracle_failure_evidence(ledger, candidate=candidate)
    needs_evidence_recheck = False if oracle_failure_evidence else _candidate_needs_evidence_recheck(ledger)
    return {
        "candidate_mode": candidate_mode,
        "latest_mode": latest_mode,
        "fresh_source_files": fresh_source_files[:6],
        "touches_suspect_path": touches_suspect_path,
        "candidate_preserving_replay_count": replay_count,
        "candidate_status": status,
        "needs_evidence_recheck": needs_evidence_recheck,
        "oracle_failure_evidence": oracle_failure_evidence,
        "policy": (
            "suspect_candidate_boundary_recheck"
            if status == "candidate_off_target"
            else "oracle_failure_evidence_refine_slot"
            if oracle_failure_evidence and status == "candidate_needs_refine"
            else "prefer_candidate_refine_before_new_exploration"
        ),
    }


def _best_source_candidate_from_memory(ledger: RecoveryLedger) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in list(ledger.metadata.get("source_candidate_memory", []) or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}

    def _score(candidate: dict[str, Any]) -> tuple[int, int, float]:
        mode = str(candidate.get("result_mode", "") or "")
        fresh_source_files = list(candidate.get("fresh_source_files", []) or [])
        source_files = list(candidate.get("source_files", []) or [])
        mode_score = 0
        if mode == "source_edit_pending_official":
            mode_score = 6
        elif mode in {
            "oracle_failed_after_source_edit",
            "contract_violation_after_source_edit",
            "intent_violation_missed_target",
            "intent_violation_revoked_target",
            "intent_violation_too_broad",
            "intent_violation_after_source_edit",
        }:
            mode_score = 4
        elif mode == "source_edit_but_not_suspect":
            mode_score = 2
        return (
            mode_score,
            1 if fresh_source_files or source_files else 0,
            float(candidate.get("created_at", 0.0) or 0.0),
        )

    return max(candidates, key=_score)


def _latest_guard_result_mode(ledger: RecoveryLedger) -> str:
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


def _candidate_preserving_replay_count(ledger: RecoveryLedger) -> int:
    count = 0
    for item in list(ledger.metadata.get("replay_cost_history", []) or []):
        if not isinstance(item, dict):
            continue
        repair_mode = str(item.get("repair_mode", "") or "")
        replay_precondition = str(item.get("replay_precondition", "") or "")
        if repair_mode.startswith("candidate_preserving") or replay_precondition == "source_candidate_refine":
            count += 1
    return count


def _allowed_convergence_actions(convergence: dict[str, Any]) -> set[SemanticActionType]:
    status = str(convergence.get("candidate_status", "") or "")
    if status == "candidate_exhausted_after_refine":
        return {SemanticActionType.SCOPE_EXPAND, SemanticActionType.TARGET_RESET, SemanticActionType.REVOKE_OBJECT}
    if status == "candidate_off_target":
        if convergence.get("needs_evidence_recheck"):
            return {SemanticActionType.EVIDENCE_RECHECK, SemanticActionType.SCOPE_EXPAND, SemanticActionType.TARGET_RESET}
        return {SemanticActionType.SCOPE_EXPAND, SemanticActionType.TARGET_RESET, SemanticActionType.REVOKE_OBJECT}
    if convergence.get("needs_evidence_recheck"):
        return {SemanticActionType.EVIDENCE_RECHECK}
    return set(CONVERGENCE_ACTIONS)


def _candidate_refine_slot_overrides_budget_halt(
    action: SemanticActionType,
    *,
    convergence: dict[str, Any],
) -> bool:
    return (
        action == SemanticActionType.LOCAL_REPAIR
        and str(convergence.get("policy", "") or "") == "oracle_failure_evidence_refine_slot"
        and str(convergence.get("candidate_status", "") or "") == "candidate_needs_refine"
        and not bool(convergence.get("needs_evidence_recheck", False))
        and int(convergence.get("candidate_preserving_replay_count", 0) or 0) <= 0
    )


def _candidate_refine_slot_overrides_validation_gate(
    action: SemanticActionType,
    *,
    convergence: dict[str, Any],
) -> bool:
    return _candidate_refine_slot_overrides_budget_halt(action, convergence=convergence)


def _candidate_refine_slot_overrides_negative_prior(
    action: SemanticActionType,
    *,
    convergence: dict[str, Any],
) -> bool:
    return _candidate_refine_slot_overrides_budget_halt(action, convergence=convergence)


def _candidate_exhausted_boundary_switch_overrides_budget_halt(
    action: SemanticActionType,
    *,
    convergence: dict[str, Any],
) -> bool:
    return (
        action == SemanticActionType.SCOPE_EXPAND
        and str(convergence.get("policy", "") or "") == "candidate_refine_exhausted_switch_boundary"
        and str(convergence.get("candidate_status", "") or "") == "candidate_exhausted_after_refine"
    )


def _reject_budget_halt(
    rejected: list[dict[str, Any]],
    *,
    card: ActionCard,
    revision_signal: dict[str, Any],
) -> None:
    rejected.append(
        {
            "action": card.action.value,
            "reason": "belief_revision_budget_halt",
            "belief_revision_signal": revision_signal,
        }
    )


def _source_candidate_status(
    *,
    candidate: dict[str, Any],
    latest_mode: str,
    replay_count: int,
    touches_suspect_path: bool,
    suspect_paths: list[str],
) -> str:
    if replay_count > 0 and latest_mode in {
        "oracle_failed_after_source_edit",
        "contract_violation_after_source_edit",
        "intent_violation_after_source_edit",
        "intent_violation_missed_target",
        "intent_violation_too_broad",
        "intent_violation_revoked_target",
        "no_diff",
        "intent_violation_no_fresh_source",
        "contract_violation_no_fresh_source",
        "budget_exhausted",
    }:
        return "candidate_exhausted_after_refine"
    candidate_mode = str(candidate.get("result_mode", "") or "")
    if suspect_paths and not touches_suspect_path:
        return "candidate_off_target"
    if latest_mode in {"", "source_edit_pending_official"} or candidate_mode == "source_edit_pending_official":
        return "candidate_pending_validation"
    if latest_mode == "oracle_failed_after_source_edit" or candidate_mode == "oracle_failed_after_source_edit":
        return "candidate_needs_refine"
    return "candidate_unresolved"


def _candidate_needs_evidence_recheck(ledger: RecoveryLedger) -> bool:
    tried = [str(item) for item in list(ledger.tried_actions or []) if str(item).strip()]
    if SemanticActionType.EVIDENCE_RECHECK.value in tried:
        return False
    latest_guard = _latest_guard(ledger)
    flags = {
        str(item)
        for item in list(latest_guard.get("guard_flags", []) or [])
        if str(item).strip()
    }
    focused_status = str(latest_guard.get("closed_loop_focused_eval_status", "") or "")
    has_focused_returncode = "closed_loop_fail_to_pass_returncode" in latest_guard
    if focused_status or has_focused_returncode:
        return bool(
            {
                "focused_validation_not_target_related",
                "focused_validation_missing_result",
            }
            & flags
        )
    return bool(
        {
            "contract_missing_focused_validation",
            "intent_missing_focused_validation",
            "focused_validation_not_target_related",
            "focused_validation_missing_result",
        }
        & flags
    ) or bool(_latest_guard_result_mode(ledger))


def _has_target_oracle_failure_evidence(
    ledger: RecoveryLedger,
    *,
    candidate: dict[str, Any],
) -> bool:
    latest_guard = _latest_guard(ledger)
    latest_mode = str(latest_guard.get("result_mode", "") or "")
    candidate_mode = str(candidate.get("result_mode", "") or "")
    if latest_mode != "oracle_failed_after_source_edit" and candidate_mode != "oracle_failed_after_source_edit":
        return False
    if not list(candidate.get("fresh_source_files", []) or candidate.get("source_files", []) or []):
        return False
    if not bool(candidate.get("touches_suspect_path", False)):
        return False
    focused_status = str(latest_guard.get("closed_loop_focused_eval_status", "") or "")
    if focused_status == "failed":
        return True
    if latest_guard.get("closed_loop_fail_to_pass_returncode") is not None:
        return True
    official_status = str(latest_guard.get("closed_loop_official_eval_status", "") or "")
    oracle_returncode = latest_guard.get("closed_loop_oracle_returncode")
    if official_status in {"resolved", "failed"} and oracle_returncode not in {None, 0, "0"}:
        return True
    return bool(str(latest_guard.get("closed_loop_fail_to_pass_output_excerpt", "") or "").strip())


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
