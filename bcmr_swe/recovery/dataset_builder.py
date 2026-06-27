"""Counterfactual dataset builder for recovery programs.

Each dataset *group* captures one failed state and records EVERYTHING about
the recovery attempts made on it.  A group contains four layers of data:

Layer 1 — Failed State Context
    Failure signature, state features, suspect region, provenance subgraph.
    Used for: case retrieval, student model input features.

Layer 2 — LLM Synthesis Record
    The LLM's diagnosis, all candidate programs it proposed, and its
    probability/cost estimates for each.  This is the LLM's *prediction*.
    Used for: calibration signal computation, understanding LLM biases.

Layer 3 — Program Execution Traces
    For each candidate program, the full step-by-step execution trace:
    what each operator did, intermediate outputs, per-step latency.
    Used for: fine-grained analysis, skill induction, debugging.

Layer 4 — Replay-Verified Outcomes
    Ground-truth results: did the recovery succeed? Actual token cost?
    Actual latency?  Test results?
    Used for: the ultimate training signal, pairwise ranking labels.

The gap between Layer 2 predictions and Layer 4 outcomes is the
*calibration signal* that makes this framework more than LLM-as-judge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from bcmr_swe.types import (
    FailedState,
    ProgramOutcome,
    RecoveryProgram,
)


class CounterfactualDatasetBuilder:
    """Build and persist counterfactual recovery datasets."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_program_group(
        self,
        failed_state: FailedState,
        programs: list[RecoveryProgram],
        outcomes: list[ProgramOutcome],
        *,
        synthesis_record: dict[str, Any] | None = None,
    ) -> Path:
        """Write a complete program-level counterfactual group.

        Parameters
        ----------
        failed_state :
            The captured failure context (Layer 1).
        programs :
            All candidate recovery programs (from LLM or default).
        outcomes :
            Ground-truth replay results for each program (Layer 4).
        synthesis_record :
            LLM's diagnosis + predictions (Layer 2).
        """
        outcome_map = {o.program_id: o for o in outcomes}

        program_entries: list[dict[str, Any]] = []
        for prog in programs:
            entry: dict[str, Any] = {
                "program": prog.to_dict(),
                "llm_prediction": {
                    "estimated_recover_prob": prog.estimated_recover_prob,
                    "estimated_total_cost": prog.estimated_total_cost,
                    "estimated_risk": prog.estimated_risk,
                    "rationale": prog.rationale,
                },
            }
            outcome = outcome_map.get(prog.program_id)
            if outcome:
                entry["replay_outcome"] = outcome.to_dict()
                entry["step_traces"] = outcome.step_outcomes
                entry["calibration_signal"] = _calibration_signal(
                    prog, outcome
                )
            program_entries.append(entry)

        has_success = any(o.recover_success for o in outcomes)
        has_failure = any(not o.recover_success for o in outcomes)

        payload: dict[str, Any] = {
            "format_version": "v3_program",
            "group_id": failed_state.group_id,
            "created_at": time.time(),

            "failed_state": failed_state.to_dict(),

            "candidate_programs": program_entries,

            "group_quality": {
                "n_programs": len(programs),
                "n_outcomes": len(outcomes),
                "has_synthesis": synthesis_record is not None,
                "has_success": has_success,
                "has_failure": has_failure,
                "has_differentiation": has_success and has_failure,
                "skeletons": list({p.skeleton for p in programs}),
            },
        }

        if synthesis_record:
            payload["synthesis_record"] = {
                "diagnosis": synthesis_record.get("diagnosis", ""),
                "synthesizer_model": synthesis_record.get("synthesizer_model", ""),
                "raw_response_excerpt": str(
                    synthesis_record.get("raw_response_excerpt", "")
                )[:2000],
            }

        path = self.output_dir / f"{failed_state.group_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    # ------------------------------------------------------------------
    # Backward-compatible: old single-action write (for existing callers)
    # ------------------------------------------------------------------

    def write_group(
        self,
        failed_state: FailedState,
        actions: list,
        outcomes: list,
        *,
        teacher_predictions: dict[str, Any] | None = None,
    ) -> Path:
        """Legacy interface for v2 single-action datasets."""
        payload: dict[str, Any] = {
            "format_version": "v2_action",
            "group_id": failed_state.group_id,
            "created_at": time.time(),
            "failed_state": failed_state.to_dict(),
            "candidate_actions": [a.to_dict() for a in actions],
            "replay_results": [o.to_dict() for o in outcomes],
        }
        if teacher_predictions:
            payload["teacher_predictions"] = teacher_predictions

        path = self.output_dir / f"{failed_state.group_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path


def _calibration_signal(
    program: RecoveryProgram,
    outcome: ProgramOutcome,
) -> dict[str, float]:
    """Compute prediction-vs-reality gap for student training.

    Positive = LLM was optimistic; negative = LLM was pessimistic.
    """
    actual_success = 1.0 if outcome.recover_success else 0.0
    return {
        "prob_error": program.estimated_recover_prob - actual_success,
        "cost_error": program.estimated_total_cost - outcome.token_cost,
        "cost_ratio": (
            program.estimated_total_cost / max(1.0, outcome.token_cost)
        ) - 1.0,
        "risk_error": program.estimated_risk - outcome.secondary_risk,
        "prob_error_abs": abs(program.estimated_recover_prob - actual_success),
    }
