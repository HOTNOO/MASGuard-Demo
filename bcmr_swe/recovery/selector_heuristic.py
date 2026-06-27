"""Heuristic recovery selector."""

from __future__ import annotations

from bcmr_swe.types import ActionScore, CandidateAction, FailedState, RecoveryBudget, TriggerType


class HeuristicRecoverySelector:
    """Score actions with a budget-aware heuristic utility function."""

    def rank(self, failed_state: FailedState, actions: list[CandidateAction], budget: RecoveryBudget) -> list[ActionScore]:
        ranked: list[ActionScore] = []
        trigger = failed_state.trigger.trigger_type
        state = failed_state.state_features.numeric
        region_size = max(1.0, state.get("suspect_region_size", 1.0))
        conflicting = state.get("n_conflicting_facts", 0.0)

        for action in actions:
            recover = action.estimated_recover_prob + self._trigger_bonus(trigger, action.action_type.value)
            recover += min(0.08, 0.01 * conflicting) if action.action_type.value == "QUARANTINE_FACT" else 0.0
            recover += 0.05 if action.action_type.value == "REPLAY_SUBGRAPH" and region_size <= 4 else 0.0
            recover = max(0.01, min(0.99, recover))

            utility = (
                recover
                - budget.lambda_token * action.estimated_token_cost
                - budget.lambda_latency * action.estimated_latency_sec
                - budget.lambda_risk * action.estimated_risk
            )
            ranked.append(
                ActionScore(
                    action_id=action.action_id,
                    utility=utility,
                    estimated_recover_prob=recover,
                    estimated_token_cost=action.estimated_token_cost,
                    estimated_latency_sec=action.estimated_latency_sec,
                    estimated_risk=action.estimated_risk,
                    explanation=f"trigger={trigger.value}, action={action.action_type.value}, region={region_size:.0f}",
                )
            )

        ranked.sort(key=lambda item: item.utility, reverse=True)
        return ranked

    def _trigger_bonus(self, trigger: TriggerType, action_name: str) -> float:
        if trigger == TriggerType.VERIFIER_CONTRADICTION:
            return {
                "INSERT_VERIFIER": 0.08,
                "REPLAY_SUBGRAPH": 0.05,
                "ROLLBACK_TO_CHECKPOINT": -0.02,
            }.get(action_name, 0.0)
        if trigger == TriggerType.NO_PROGRESS_LOOP:
            return {
                "ESCALATE_NODE": 0.10,
                "ROLLBACK_TO_CHECKPOINT": 0.06,
                "INSERT_VERIFIER": -0.05,
            }.get(action_name, 0.0)
        if trigger == TriggerType.FACT_CONFLICT:
            return {
                "QUARANTINE_FACT": 0.12,
                "ROLLBACK_TO_CHECKPOINT": 0.03,
                "INSERT_VERIFIER": -0.03,
            }.get(action_name, 0.0)
        return 0.0
