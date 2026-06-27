from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class ArmStats:
    """Statistics for a single arm in Beta-Bernoulli bandit."""
    successes: float = 0.0
    failures: float = 0.0

    @property
    def total(self) -> int:
        return self.successes + self.failures


class BetaBandit:
    """Thompson Sampling for discrete arms with binary rewards."""

    def select_arm(self, arms: Dict[str, ArmStats]) -> str:
        if not arms:
            raise ValueError("No arms provided")

        unexplored = [arm_id for arm_id, s in arms.items() if s.total == 0]
        if unexplored:
            return random.choice(unexplored)

        samples: Dict[str, float] = {}
        for arm_id, stats in arms.items():
            alpha = stats.successes + 1.0
            beta = stats.failures + 1.0
            samples[arm_id] = random.betavariate(alpha, beta)

        return max(samples.items(), key=lambda kv: kv[1])[0]

    def update(self, arms: Dict[str, ArmStats], arm_id: str, reward: float, weight: float = 1.0) -> None:
        """reward > 0 视为 success，reward <= 0 视为 failure。weight 用于加权计数。"""
        if arm_id not in arms:
            arms[arm_id] = ArmStats()
        if reward > 0:
            arms[arm_id].successes += weight
        else:
            arms[arm_id].failures += weight
