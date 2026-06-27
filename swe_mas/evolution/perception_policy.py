from __future__ import annotations

from dataclasses import dataclass

from .bandit import BetaBandit, ArmStats
from .policy_store import PolicyStore
from .perception_templates import PERCEPTION_POLICIES
from swe_mas.perception.graph_analyzer import PerceptionPolicyConfig


@dataclass
class PerceptionPolicy:
    store: PolicyStore
    bandit: BetaBandit

    def bucket_id(self, loop_score: float, test_score: float, churn_score: float) -> str:
        def bucket(v: float) -> str:
            if v < 0.2:
                return "L"
            if v < 0.5:
                return "M"
            return "H"

        return f"LOOP={bucket(loop_score)}|TEST={bucket(test_score)}|CHURN={bucket(churn_score)}"

    def select_policy_id(self, loop_score: float, test_score: float, churn_score: float) -> str:
        bucket = self.bucket_id(loop_score, test_score, churn_score)

        if bucket not in self.store.perception.buckets:
            self.store.perception.buckets[bucket] = {pid: ArmStats() for pid in PERCEPTION_POLICIES.keys()}

        arms = self.store.perception.buckets[bucket]
        return self.bandit.select_arm(arms)

    def select_config(self, loop_score: float, test_score: float, churn_score: float) -> tuple[str, PerceptionPolicyConfig]:
        pid = self.select_policy_id(loop_score, test_score, churn_score)
        template = PERCEPTION_POLICIES[pid]
        return pid, template.config
