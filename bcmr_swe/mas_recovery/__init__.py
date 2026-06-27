"""MAS-DX-R recovery artifacts for propagation-aware MAS failure recovery."""

from bcmr_swe.mas_recovery.checkpoint_policy import build_checkpoint_recovery_plan
from bcmr_swe.mas_recovery.propagation_graph import build_propagation_graph
from bcmr_swe.mas_recovery.hypothesis import generate_failure_hypotheses
from bcmr_swe.mas_recovery.intervention_loop import build_intervention_plan
from bcmr_swe.mas_recovery.observation import (
    apply_observations_to_hypotheses,
    observations_from_action_outputs,
)
from bcmr_swe.mas_recovery.recovery_blueprint import build_recovery_blueprint
from bcmr_swe.mas_recovery.recovery_controller import select_recovery_decision
from bcmr_swe.mas_recovery.recovery_policy_v2 import select_harm_aware_recovery_decision
from bcmr_swe.mas_recovery.fusion_controller import select_fusion_recovery_decision
from bcmr_swe.mas_recovery.preference_design import (
    build_graph_calibrated_preference_profile,
    build_mas_evidence_fail_closed_preference_profile,
)

__all__ = [
    "build_propagation_graph",
    "build_checkpoint_recovery_plan",
    "build_recovery_blueprint",
    "generate_failure_hypotheses",
    "build_intervention_plan",
    "observations_from_action_outputs",
    "apply_observations_to_hypotheses",
    "select_recovery_decision",
    "select_harm_aware_recovery_decision",
    "select_fusion_recovery_decision",
    "build_graph_calibrated_preference_profile",
    "build_mas_evidence_fail_closed_preference_profile",
]
