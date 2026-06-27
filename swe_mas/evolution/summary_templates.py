from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from swe_mas.recovery.models import RecoveryStrategy, Granularity


@dataclass(frozen=True)
class SummaryStrategyTemplate:
    """A named, discrete summary strategy that bandit can choose."""
    id: str
    strategy: RecoveryStrategy


def _make_strategies() -> Dict[str, SummaryStrategyTemplate]:
    s1 = SummaryStrategyTemplate(
        id="S1_concise_keep_branch",
        strategy=RecoveryStrategy(
            granularity=Granularity.CONCISE,
            focus_areas=["intent", "last_state"],
            discard_recent_branch=False,
            reuse_healthy_intent=True,
            max_history=20,
        ),
    )

    s2 = SummaryStrategyTemplate(
        id="S2_standard_keep_branch",
        strategy=RecoveryStrategy(
            granularity=Granularity.STANDARD,
            focus_areas=["intent", "actions", "state"],
            discard_recent_branch=False,
            reuse_healthy_intent=True,
            max_history=40,
        ),
    )

    s3 = SummaryStrategyTemplate(
        id="S3_verbose_drop_branch",
        strategy=RecoveryStrategy(
            granularity=Granularity.VERBOSE,
            focus_areas=["intent", "actions", "state", "branches"],
            discard_recent_branch=True,
            reuse_healthy_intent=True,
            max_history=80,
        ),
    )

    s4 = SummaryStrategyTemplate(
        id="S4_verbose_keep_branch_large_ctx",
        strategy=RecoveryStrategy(
            granularity=Granularity.VERBOSE,
            focus_areas=["intent", "actions", "state", "branches"],
            discard_recent_branch=False,
            reuse_healthy_intent=True,
            max_history=120,
        ),
    )

    return {s.id: s for s in (s1, s2, s3, s4)}


SUMMARY_STRATEGIES: Dict[str, SummaryStrategyTemplate] = _make_strategies()
