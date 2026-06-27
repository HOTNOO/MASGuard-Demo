"""Counterfactual replay execution."""

from __future__ import annotations

import time
from typing import Protocol

from bcmr_swe.types import CandidateAction, ReplayOutcome


class RecoveryRuntimeProtocol(Protocol):
    def apply_recovery_action(self, action: CandidateAction) -> ReplayOutcome:
        ...


class ReplayEngine:
    """Thin dispatcher over the runtime's recovery action executor."""

    def execute(self, runtime: RecoveryRuntimeProtocol, action: CandidateAction) -> ReplayOutcome:
        started = time.time()
        outcome = runtime.apply_recovery_action(action)
        if outcome.latency_sec <= 0:
            outcome.latency_sec = max(0.001, time.time() - started)
        return outcome
