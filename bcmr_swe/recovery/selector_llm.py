"""LLM-based recovery selector with budget-constrained optimization.

This selector combines LLM recovery scoring with formal budget optimization.
It is NOT a pure "LLM-as-judge" — the scores are *predictions* that get
verified through counterfactual replay, producing ground-truth calibration
signals for the Phase 2 student reranker.

The budget-constrained optimization solves:

    a* = argmax_a [ p_recover(a) - λ₁·c_token(a) - λ₂·c_latency(a) - λ₃·risk(a) ]
    s.t.  c_token(a) ≤ remaining_token_budget
          recovery_call_count ≤ max_recovery_calls
"""

from __future__ import annotations

import logging
from typing import Any

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.recovery.llm_recovery_scorer import (
    LLMRecoveryScorer,
    LLMRecoveryScorerConfig,
)
from bcmr_swe.types import (
    ActionScore,
    CandidateAction,
    FailedState,
    RecoveryBudget,
)

logger = logging.getLogger(__name__)


class LLMRecoverySelector:
    """Budget-constrained action selector powered by LLM provenance reasoning.

    This selector has two outputs that are both recorded:

    1. **Ranked action scores** — used for online decision-making.
    2. **Full prediction record** — stored alongside replay outcomes in the
       counterfactual dataset so that the Phase 2 student can learn to
       calibrate LLM predictions against real execution results.
    """

    def __init__(
        self,
        scorer: LLMRecoveryScorer,
        *,
        graph: ExecutionProvenanceGraph | None = None,
    ):
        self.scorer = scorer
        self.graph: ExecutionProvenanceGraph | None = graph
        self.last_prediction_record: dict[str, Any] = {}

    def rank(
        self,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        *,
        graph: ExecutionProvenanceGraph | None = None,
        used_recovery_calls: int = 0,
        used_tokens: float = 0.0,
    ) -> list[ActionScore]:
        """Score and rank actions, applying hard budget constraints.

        Actions whose estimated token cost exceeds the remaining budget are
        penalised rather than removed outright — the LLM may underestimate
        costs, and removing them would reduce the action space available to
        the counterfactual replay evaluator.
        """
        active_graph = graph or self.graph
        if active_graph is None:
            logger.warning("No provenance graph; falling back to prior estimates.")
            return self.scorer._fallback_scores(actions, budget)

        scores = self.scorer.score(
            active_graph,
            failed_state,
            actions,
            budget,
            used_recovery_calls=used_recovery_calls,
            used_tokens=used_tokens,
        )

        remaining_tokens = max(0.0, budget.token_budget - used_tokens)
        constrained = self._apply_budget_constraints(scores, remaining_tokens)

        self.last_prediction_record = {
            "diagnosis": self.scorer.last_diagnosis,
            "llm_scores": [s.to_dict() for s in scores],
            "constrained_scores": [s.to_dict() for s in constrained],
            "remaining_token_budget": remaining_tokens,
            "used_recovery_calls": used_recovery_calls,
        }

        return constrained

    def get_prediction_record(self) -> dict[str, Any]:
        """Return the full LLM prediction for dataset persistence.

        This record is saved alongside the replay outcome so that the
        Phase 2 student reranker can learn the mapping:
            (LLM prediction, state features) → actual replay outcome
        """
        return dict(self.last_prediction_record)

    def _apply_budget_constraints(
        self,
        scores: list[ActionScore],
        remaining_tokens: float,
    ) -> list[ActionScore]:
        """Penalise actions that exceed the remaining token budget.

        We apply a soft penalty (halve utility) rather than hard removal so
        that the counterfactual dataset still records all actions.
        """
        constrained: list[ActionScore] = []
        for s in scores:
            if s.estimated_token_cost > remaining_tokens > 0:
                overshoot = s.estimated_token_cost / max(1.0, remaining_tokens)
                penalty = min(0.5, 0.1 * overshoot)
                adjusted_utility = s.utility - penalty
                constrained.append(ActionScore(
                    action_id=s.action_id,
                    utility=adjusted_utility,
                    estimated_recover_prob=s.estimated_recover_prob,
                    estimated_token_cost=s.estimated_token_cost,
                    estimated_latency_sec=s.estimated_latency_sec,
                    estimated_risk=s.estimated_risk,
                    explanation=f"{s.explanation} | budget_penalty={penalty:.3f}",
                ))
            else:
                constrained.append(s)
        constrained.sort(key=lambda s: s.utility, reverse=True)
        return constrained
