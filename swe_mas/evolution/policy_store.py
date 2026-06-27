from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

from .bandit import ArmStats


@dataclass
class SummaryPolicyState:
    buckets: Dict[str, Dict[str, ArmStats]]


@dataclass
class PerceptionPolicyState:
    buckets: Dict[str, Dict[str, ArmStats]]


@dataclass
class PolicyStore:
    summary: SummaryPolicyState
    perception: PerceptionPolicyState
    path: Path

    @classmethod
    def load(cls, path: Path) -> "PolicyStore":
        if not path.exists():
            return cls(
                summary=SummaryPolicyState(buckets={}),
                perception=PerceptionPolicyState(buckets={}),
                path=path,
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        summary_buckets: Dict[str, Dict[str, ArmStats]] = {}
        for bucket, stats_dict in data.get("summary", {}).items():
            summary_buckets[bucket] = {sid: ArmStats(**s) for sid, s in stats_dict.items()}

        perception_buckets: Dict[str, Dict[str, ArmStats]] = {}
        for bucket, stats_dict in data.get("perception", {}).items():
            perception_buckets[bucket] = {pid: ArmStats(**s) for pid, s in stats_dict.items()}

        return cls(
            summary=SummaryPolicyState(buckets=summary_buckets),
            perception=PerceptionPolicyState(buckets=perception_buckets),
            path=path,
        )

    def save(self) -> None:
        data = {
            "summary": {
                bucket: {sid: asdict(stats) for sid, stats in stats_dict.items()}
                for bucket, stats_dict in self.summary.buckets.items()
            },
            "perception": {
                bucket: {pid: asdict(stats) for pid, stats in stats_dict.items()}
                for bucket, stats_dict in self.perception.buckets.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
