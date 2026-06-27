from __future__ import annotations

from dataclasses import dataclass

from swe_mas.memory.recorder import MemoryRecorder
from swe_mas.recovery.models import RecoveryStrategy, RecoveryRequest
from .bandit import BetaBandit, ArmStats
from .policy_store import PolicyStore
from .summary_templates import SUMMARY_STRATEGIES


@dataclass
class SummaryPolicy:
    recorder: MemoryRecorder
    store: PolicyStore
    bandit: BetaBandit

    def bucket_id(self, request: RecoveryRequest) -> str:
        fault = getattr(request, "fault_type", None)
        role = getattr(request, "target_role", None) or "unknown"
        resource = getattr(request, "resource", None)

        fault_name = getattr(fault, "name", None) or (str(fault) if fault is not None else "NONE")
        cognitive = {"HALLUCINATION", "MISALIGNMENT", "LOOP", "INJECTED"}
        fault_class = "cognitive" if fault_name in cognitive else "external"
        bandwidth = getattr(resource, "bandwidth", None)
        bandwidth_name = getattr(bandwidth, "name", "UNKNOWN")

        token_left = getattr(resource, "token_left", None)
        if token_left is None:
            token_bucket = "UNK"
        elif token_left <= 800:
            token_bucket = "LOW"
        elif token_left <= 2000:
            token_bucket = "MID"
        else:
            token_bucket = "HIGH"

        return f"FCLASS={fault_class}|FAULT={fault_name}|ROLE={role}|BW={bandwidth_name}|TOK={token_bucket}"

    def select_strategy_id(self, request: RecoveryRequest) -> str:
        bucket = self.bucket_id(request)

        if bucket not in self.store.summary.buckets:
            self.store.summary.buckets[bucket] = {sid: ArmStats() for sid in SUMMARY_STRATEGIES.keys()}

        arms = self.store.summary.buckets[bucket]
        return self.bandit.select_arm(arms)

    def select_strategy(self, request: RecoveryRequest) -> tuple[str, RecoveryStrategy]:
        strategy_id = self.select_strategy_id(request)
        template = SUMMARY_STRATEGIES[strategy_id]
        return strategy_id, template.strategy
