"""Adapters that expose the legacy swe_mas stages as a BCMR substrate bundle."""

from __future__ import annotations

from typing import Any

from bcmr_swe.agent.locator import LegacyLocatorAdapter
from bcmr_swe.agent.patcher import PlannerPatcherAdapter
from bcmr_swe.agent.verifier import LegacyVerifierAdapter
from bcmr_swe.substrate.stage_bundle import StageBundle


def build_swe_mas_stage_bundle(
    *,
    model: Any,
    executor: Any,
    planner_model: Any,
    implementer_model: Any,
    verifier_model: Any,
    locator_model: Any,
    recorder: Any | None = None,
    session_id: str | None = None,
    locator_max_iterations: int = 8,
    planner_max_iterations: int = 4,
    patcher_max_iterations: int = 8,
    verifier_max_iterations: int = 6,
) -> StageBundle:
    """Expose the existing swe_mas agents as a substrate bundle.

    This keeps BCMR's core logic independent from the legacy MAS runtime while
    preserving backward-compatible experiments on the current base system.
    """

    locator = LegacyLocatorAdapter(
        model=locator_model,
        executor=executor,
        recorder=recorder,
        session_id=session_id,
        max_iterations=locator_max_iterations,
    )
    patcher = PlannerPatcherAdapter(
        model=implementer_model,
        planner_model=planner_model,
        implementer_model=implementer_model,
        executor=executor,
        recorder=recorder,
        session_id=session_id,
        max_plan_iterations=planner_max_iterations,
        max_patch_iterations=patcher_max_iterations,
    )
    verifier = LegacyVerifierAdapter(
        model=verifier_model,
        executor=executor,
        recorder=recorder,
        session_id=session_id,
        max_iterations=verifier_max_iterations,
    )
    return StageBundle(
        locator=locator,
        patcher=patcher,
        verifier=verifier,
        backend_name="swe_mas_legacy",
    )
