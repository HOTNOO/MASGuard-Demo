"""Student reranker that calibrates LLM teacher predictions using replay evidence.

This model is the core of BCMR-Hybrid Phase 2.  It does NOT learn recovery
policy from scratch.  Instead it learns to *calibrate* the LLM teacher's
predictions against ground-truth counterfactual replay outcomes.

Training signal for each (failed_state, action) sample:
- Input:  state features + region features + action features
        + LLM teacher's (p_recover, cost, latency, risk, utility)
- Target: replay-verified (actual_success, actual_cost, actual_latency, actual_risk)

The calibration objective has three components:
1. **Success calibration** — correct the teacher's recover probability
   toward the binary replay outcome.
2. **Cost calibration** — correct the teacher's token/latency estimates
   toward actual measurements.
3. **Pairwise ranking** — within the same failed-state group, enforce that
   actions with better replay outcomes rank higher.

At inference time the student operates as a fast reranker:
- Score all actions without LLM calls (cheap, ~1ms).
- Optionally route the top-K uncertain candidates to the LLM teacher
  for a second opinion (the "hybrid cascade").
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from bcmr_swe.types import (
    ActionScore,
    CandidateAction,
    FailedState,
    RecoveryBudget,
)


class StudentReranker:
    """Lightweight reranker trained on (LLM prediction, replay outcome) pairs.

    The model is a multi-head sparse linear scorer (same efficient form as
    the existing GCRV) but with a critical difference: its features include
    the *LLM teacher's predictions*, so it learns the teacher → reality
    calibration mapping rather than raw recovery policy.
    """

    def __init__(self):
        self.w_utility: dict[str, float] = defaultdict(float)
        self.w_success: dict[str, float] = defaultdict(float)
        self.w_cost: dict[str, float] = defaultdict(float)
        self.bias = {"utility": 0.0, "success": 0.0, "cost": 0.0}

    def fit(
        self,
        samples: list[dict[str, Any]],
        *,
        epochs: int = 30,
        lr: float = 0.01,
    ) -> dict[str, float]:
        """Train on samples extracted from counterfactual dataset groups.

        Each sample dict must contain:
        - ``features``: dict[str, float] from ``extract_features``
        - ``target_success``: float (0 or 1)
        - ``target_utility``: float
        - ``target_cost``: float (normalised token cost)
        - ``group_id``: str (for pairwise grouping)
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in samples:
            groups[s.get("group_id", "")].append(s)

        for _epoch in range(epochs):
            for group_samples in groups.values():
                for s in group_samples:
                    feats = s["features"]
                    self._sgd(self.w_success, feats, s["target_success"], lr, "success")
                    self._sgd(self.w_utility, feats, s["target_utility"], lr * 0.5, "utility")
                    self._sgd(self.w_cost, feats, s["target_cost"], lr * 0.001, "cost")

                for i, a in enumerate(group_samples):
                    for b in group_samples[i + 1:]:
                        if abs(a["target_utility"] - b["target_utility"]) < 1e-9:
                            continue
                        better, worse = (a, b) if a["target_utility"] > b["target_utility"] else (b, a)
                        self._pairwise(better["features"], worse["features"], lr * 0.3)

        return self._training_summary(samples)

    def rank(
        self,
        failed_state: FailedState,
        actions: list[CandidateAction],
        budget: RecoveryBudget,
        *,
        teacher_scores: dict[str, dict[str, Any]] | None = None,
    ) -> list[ActionScore]:
        results: list[ActionScore] = []
        for action in actions:
            feats = self.extract_features(
                failed_state, action, teacher_scores=teacher_scores
            )
            p_success = _sigmoid(
                self._dot(self.w_success, feats) + self.bias["success"]
            )
            cost_correction = self._dot(self.w_cost, feats) + self.bias["cost"]
            adjusted_cost = max(0.0, action.estimated_token_cost + cost_correction)

            utility = (
                p_success
                - budget.lambda_token * adjusted_cost
                - budget.lambda_latency * action.estimated_latency_sec
                - budget.lambda_risk * action.estimated_risk
                + 0.15 * (self._dot(self.w_utility, feats) + self.bias["utility"])
            )
            results.append(ActionScore(
                action_id=action.action_id,
                utility=utility,
                estimated_recover_prob=p_success,
                estimated_token_cost=adjusted_cost,
                estimated_latency_sec=action.estimated_latency_sec,
                estimated_risk=action.estimated_risk,
                explanation="student_reranker",
            ))
        results.sort(key=lambda s: s.utility, reverse=True)
        return results

    def extract_features(
        self,
        failed_state: FailedState,
        action: CandidateAction,
        *,
        teacher_scores: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Build feature vector including LLM teacher predictions.

        The teacher prediction features are what distinguish this from a
        plain tabular model — they let the student learn *when* the teacher
        is reliable and when it needs correction.
        """
        feats: dict[str, float] = {}

        for k, v in failed_state.state_features.numeric.items():
            feats[f"state::{k}"] = float(v)
        for k, v in failed_state.state_features.categorical.items():
            feats[f"cat::{k}::{v}"] = 1.0

        region = failed_state.suspect_region.summary
        for k, v in region.items():
            if isinstance(v, (int, float)):
                feats[f"region::{k}"] = float(v)
            elif isinstance(v, str):
                feats[f"region::{k}::{v}"] = 1.0
            elif isinstance(v, bool):
                feats[f"region::{k}"] = 1.0 if v else 0.0

        feats[f"action::{action.action_type.value}"] = 1.0
        feats["action::prior_prob"] = action.estimated_recover_prob
        feats["action::prior_cost"] = action.estimated_token_cost / 1000.0
        feats["action::prior_latency"] = action.estimated_latency_sec / 100.0
        feats["action::prior_risk"] = action.estimated_risk

        teacher = (teacher_scores or {}).get(action.action_id, {})
        if teacher:
            t_prob = float(teacher.get("estimated_recover_prob", 0.5))
            t_cost = float(teacher.get("estimated_token_cost", 0)) / 1000.0
            t_latency = float(teacher.get("estimated_latency_sec", 0)) / 100.0
            t_risk = float(teacher.get("estimated_risk", 0))
            t_utility = float(teacher.get("utility", 0))

            feats["teacher::prob"] = t_prob
            feats["teacher::cost"] = t_cost
            feats["teacher::latency"] = t_latency
            feats["teacher::risk"] = t_risk
            feats["teacher::utility"] = t_utility
            feats["teacher::available"] = 1.0

            feats["gap::prob_vs_prior"] = t_prob - action.estimated_recover_prob
            feats["gap::cost_vs_prior"] = t_cost - action.estimated_token_cost / 1000.0
        else:
            feats["teacher::available"] = 0.0

        return feats

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "model_type": "student_reranker",
            "w_utility": dict(self.w_utility),
            "w_success": dict(self.w_success),
            "w_cost": dict(self.w_cost),
            "bias": dict(self.bias),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StudentReranker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model.w_utility.update(data.get("w_utility", {}))
        model.w_success.update(data.get("w_success", {}))
        model.w_cost.update(data.get("w_cost", {}))
        model.bias.update(data.get("bias", {}))
        return model

    def _sgd(self, weights: dict[str, float], feats: dict[str, float], target: float, lr: float, head: str) -> None:
        pred = self._dot(weights, feats) + self.bias[head]
        error = target - pred
        for k, v in feats.items():
            weights[k] += lr * error * v
        self.bias[head] += lr * error

    def _pairwise(self, better_feats: dict[str, float], worse_feats: dict[str, float], lr: float, margin: float = 0.2) -> None:
        b_score = self._dot(self.w_utility, better_feats) + self.bias["utility"]
        w_score = self._dot(self.w_utility, worse_feats) + self.bias["utility"]
        if b_score - w_score >= margin:
            return
        diff = {k: better_feats.get(k, 0.0) - worse_feats.get(k, 0.0) for k in set(better_feats) | set(worse_feats)}
        for k, v in diff.items():
            self.w_utility[k] += lr * v
        self.bias["utility"] += lr * 0.1

    def _dot(self, weights: dict[str, float], feats: dict[str, float]) -> float:
        return sum(weights.get(k, 0.0) * v for k, v in feats.items())

    def _training_summary(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        n = len(samples)
        n_success = sum(1 for s in samples if s.get("target_success", 0) > 0.5)
        n_with_teacher = sum(1 for s in samples if s["features"].get("teacher::available", 0) > 0.5)
        return {
            "n_samples": float(n),
            "n_success": float(n_success),
            "n_with_teacher": float(n_with_teacher),
            "success_rate": n_success / max(1, n),
            "teacher_coverage": n_with_teacher / max(1, n),
        }


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
