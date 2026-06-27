from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from swe_mas.perception.graph_analyzer import PerceptionPolicyConfig


@dataclass(frozen=True)
class PerceptionTemplate:
    id: str
    config: PerceptionPolicyConfig


def _make_policies() -> Dict[str, PerceptionTemplate]:
    p0 = PerceptionTemplate(
        id="P0_balanced",
        config=PerceptionPolicyConfig(
            w_loop=0.4,
            w_test=0.3,
            w_churn=0.3,
            threshold=0.5,
        ),
    )

    p1 = PerceptionTemplate(
        id="P1_aggressive_loop",
        config=PerceptionPolicyConfig(
            w_loop=0.6,
            w_test=0.2,
            w_churn=0.2,
            threshold=0.4,
        ),
    )

    p2 = PerceptionTemplate(
        id="P2_test_focused",
        config=PerceptionPolicyConfig(
            w_loop=0.3,
            w_test=0.5,
            w_churn=0.2,
            threshold=0.6,
        ),
    )

    p3 = PerceptionTemplate(
        id="P3_conservative",
        config=PerceptionPolicyConfig(
            w_loop=0.3,
            w_test=0.3,
            w_churn=0.4,
            threshold=0.7,
        ),
    )

    return {p.id: p for p in (p0, p1, p2, p3)}


PERCEPTION_POLICIES: Dict[str, PerceptionTemplate] = _make_policies()
