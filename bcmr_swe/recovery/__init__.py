"""Recovery pipeline modules for BCMR-SWE.

v3 three-layer architecture:
  Layer 1 – Atomic operators (defined in types.py as OpType)
  Layer 2 – Program synthesis (program_grammar + program_synthesizer)
  Layer 3 – Case memory and skill induction (case_memory)
"""

from bcmr_swe.recovery.actions import RecoveryActionPlanner
from bcmr_swe.recovery.case_memory import CaseMemory
from bcmr_swe.recovery.contamination_frontier import (
    ContaminationFrontier,
    ContaminantScore,
    compute_contamination_frontier,
    evaluate_frontier_against_oracle,
    render_frontier_audit,
    score_contaminants,
)
from bcmr_swe.recovery.dataset_builder import CounterfactualDatasetBuilder
from bcmr_swe.recovery.frontier_controller import (
    FrontierControllerDecision,
    decide_frontier_recovery,
    frontier_to_probe_program,
    frontier_to_recovery_program,
)
from bcmr_swe.recovery.llm_recovery_scorer import LLMRecoveryScorer, LLMRecoveryScorerConfig
from bcmr_swe.recovery.program_executor import ProgramExecutor
from bcmr_swe.recovery.program_grammar import (
    format_grammar_for_prompt,
    validate_program,
)
from bcmr_swe.recovery.program_synthesizer import ProgramSynthesizer
from bcmr_swe.recovery.provenance_summarizer import ProvenanceSummarizer
from bcmr_swe.recovery.replay_engine import ReplayEngine
from bcmr_swe.recovery.semantic_executor import SemanticProgramExecutor
from bcmr_swe.recovery.semantic_language import (
    bootstrap_recovery_ledger,
    compile_semantic_program,
    semantic_programs_v1,
)
from bcmr_swe.recovery.selector_heuristic import HeuristicRecoverySelector
from bcmr_swe.recovery.selector_llm import LLMRecoverySelector
from bcmr_swe.recovery.selector_model import RecoverySelector
from bcmr_swe.recovery.state_encoder import StateEncoder
from bcmr_swe.recovery.structured_state import (
    build_structured_recovery_state_from_failed_state,
    build_structured_recovery_state_from_trajectory_artifacts,
    project_structured_state_core,
    project_structured_state_with_evidence,
)
from bcmr_swe.recovery.triggers import FailureTriggerDetector

__all__ = [
    "CaseMemory",
    "ContaminantScore",
    "ContaminationFrontier",
    "CounterfactualDatasetBuilder",
    "FailureTriggerDetector",
    "FrontierControllerDecision",
    "HeuristicRecoverySelector",
    "LLMRecoveryScorer",
    "LLMRecoveryScorerConfig",
    "LLMRecoverySelector",
    "ProgramExecutor",
    "ProgramSynthesizer",
    "ProvenanceSummarizer",
    "RecoveryActionPlanner",
    "RecoverySelector",
    "ReplayEngine",
    "SemanticProgramExecutor",
    "StateEncoder",
    "bootstrap_recovery_ledger",
    "build_structured_recovery_state_from_failed_state",
    "build_structured_recovery_state_from_trajectory_artifacts",
    "compile_semantic_program",
    "compute_contamination_frontier",
    "decide_frontier_recovery",
    "evaluate_frontier_against_oracle",
    "format_grammar_for_prompt",
    "frontier_to_probe_program",
    "frontier_to_recovery_program",
    "project_structured_state_core",
    "project_structured_state_with_evidence",
    "render_frontier_audit",
    "score_contaminants",
    "semantic_programs_v1",
    "validate_program",
]
