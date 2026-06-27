"""Selector wrapper that routes to heuristic, learned, or LLM selectors."""

from __future__ import annotations

from typing import Any

from bcmr_swe.recovery.selector_heuristic import HeuristicRecoverySelector
from bcmr_swe.types import ActionScore, CandidateAction, FailedState, RecoveryBudget


class RecoverySelector:
    """Unified selector interface for all BCMR selector variants.

    Priority: LLM selector > learned model > heuristic.
    """

    def __init__(
        self,
        model: object | None = None,
        llm_selector: object | None = None,
    ):
        self.model = model
        self.llm_selector = llm_selector
        self.heuristic = HeuristicRecoverySelector()

    def rank(
        self,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        **kwargs: Any,
    ) -> list[ActionScore]:
        if self.llm_selector is not None:
            return self.llm_selector.rank(failed_state, actions, budget, **kwargs)
        if self.model is not None:
            return self.model.rank(failed_state, actions, budget)
        return self.heuristic.rank(failed_state, actions, budget)

    def get_prediction_record(self) -> dict[str, Any]:
        """Return the LLM prediction record if the LLM selector is active."""
        getter = getattr(self.llm_selector, "get_prediction_record", None)
        if callable(getter):
            return getter()
        return {}
