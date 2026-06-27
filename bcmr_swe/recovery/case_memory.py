"""Recovery Case Memory: store, retrieve, and induce skills from experiences.

This module implements Layer 3 of the BCMR-Hybrid-PM architecture.

Case Memory stores every replay-verified recovery attempt as a structured
case.  At runtime, similar cases are retrieved and injected into the LLM
synthesis prompt as few-shot examples — this is how the system improves
over time without retraining.

Skill Induction periodically clusters similar successful cases and
extracts generalised program skeletons with statistical metadata (success
rate, average cost).  Skills are a higher-level abstraction used for
offline analysis and future student model training.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from bcmr_swe.types import (
    FailedState,
    FailureSignature,
    ProgramOutcome,
    RecoveryCase,
    RecoveryProgram,
    RecoverySkill,
)

logger = logging.getLogger(__name__)


class CaseMemory:
    """Persistent store of recovery cases with similarity-based retrieval."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.cases: list[RecoveryCase] = []
        self._load_existing()

    def commit(
        self,
        failed_state: FailedState,
        program: RecoveryProgram,
        outcome: ProgramOutcome,
    ) -> RecoveryCase:
        """Create and persist a new recovery case (COMMIT_EXPERIENCE)."""
        signature = self._extract_signature(failed_state)
        state_summary = self._extract_state_summary(failed_state)

        case = RecoveryCase(
            case_id=f"case_{uuid.uuid4().hex[:10]}",
            instance_id=failed_state.instance_id,
            failure_signature=signature,
            state_summary=state_summary,
            program=program,
            outcome=outcome,
        )
        self.cases.append(case)
        self._persist_case(case)
        return case

    def retrieve(
        self,
        failed_state: FailedState,
        *,
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> list[RecoveryCase]:
        """Find the most similar past cases for a given failed state."""
        query_sig = self._extract_signature(failed_state)
        scored: list[tuple[float, RecoveryCase]] = []
        for case in self.cases:
            sim = query_sig.similarity(case.failure_signature)
            if sim >= min_similarity:
                scored.append((sim, case))
        scored.sort(key=lambda t: (-t[0], -t[1].outcome.recover_success))
        return [case for _, case in scored[:top_k]]

    def successful_cases(self) -> list[RecoveryCase]:
        return [c for c in self.cases if c.outcome.recover_success]

    def induce_skills(self, *, min_cases: int = 3) -> list[RecoverySkill]:
        """Cluster successful cases by failure signature + skeleton and
        extract generalised skills.

        A skill represents a program skeleton that has worked for a
        particular class of failures.  It includes:
        - Applicability condition (failure signature)
        - Program skeleton (op sequence)
        - Success rate and average cost from past cases
        """
        successful = self.successful_cases()
        if len(successful) < min_cases:
            return []

        groups: dict[str, list[RecoveryCase]] = defaultdict(list)
        for case in successful:
            key = (
                f"{case.failure_signature.trigger_type}"
                f"|{case.failure_signature.failed_stage}"
                f"|{case.program.skeleton}"
            )
            groups[key].append(case)

        skills: list[RecoverySkill] = []
        for key, group_cases in groups.items():
            if len(group_cases) < min_cases:
                continue
            representative = group_cases[0]
            all_cases_for_sig = [
                c for c in self.cases
                if (c.failure_signature.trigger_type == representative.failure_signature.trigger_type
                    and c.failure_signature.failed_stage == representative.failure_signature.failed_stage
                    and c.program.skeleton == representative.program.skeleton)
            ]
            n_total = len(all_cases_for_sig)
            n_success = sum(1 for c in all_cases_for_sig if c.outcome.recover_success)
            avg_cost = (
                sum(c.outcome.token_cost for c in all_cases_for_sig) / n_total
                if n_total > 0 else 0.0
            )

            skills.append(RecoverySkill(
                skill_name=f"skill_{representative.failure_signature.trigger_type}_{representative.program.skeleton.replace('→', '_')}",
                applicability=representative.failure_signature,
                program_skeleton=[s.op.value for s in representative.program.steps],
                n_cases=n_total,
                success_rate=n_success / max(1, n_total),
                avg_cost=avg_cost,
            ))

        skills.sort(key=lambda s: (-s.success_rate, s.avg_cost))
        return skills

    def summary(self) -> dict[str, Any]:
        n = len(self.cases)
        n_success = sum(1 for c in self.cases if c.outcome.recover_success)
        triggers = defaultdict(int)
        skeletons = defaultdict(int)
        for c in self.cases:
            triggers[c.failure_signature.trigger_type] += 1
            skeletons[c.program.skeleton] += 1
        return {
            "total_cases": n,
            "successful_cases": n_success,
            "success_rate": n_success / max(1, n),
            "trigger_distribution": dict(triggers),
            "skeleton_distribution": dict(skeletons),
            "unique_instances": len({c.instance_id for c in self.cases}),
        }

    # ------------------------------------------------------------------
    # Signature extraction
    # ------------------------------------------------------------------

    def _extract_signature(self, fs: FailedState) -> FailureSignature:
        region = fs.suspect_region.summary
        role_chain = region.get("role_chain", [])
        region_shape = "→".join(role_chain) if role_chain else "unknown"
        numerics = fs.state_features.numeric

        return FailureSignature(
            trigger_type=fs.trigger.trigger_type.value,
            failed_stage=role_chain[-1] if role_chain else "unknown",
            region_shape=region_shape,
            n_failing_tests=int(numerics.get("failing_tests_count", 0)),
            has_conflicting_fact=bool(region.get("has_conflicting_fact", False)),
        )

    def _extract_state_summary(self, fs: FailedState) -> dict[str, Any]:
        n = fs.state_features.numeric
        return {
            "n_steps": int(n.get("n_graph_nodes", 0)),
            "n_tool_calls": int(n.get("n_tool_calls", 0)),
            "n_verifier_runs": int(n.get("n_verifier_runs", 0)),
            "checkpoint_depth": float(n.get("checkpoint_depth", 0)),
            "recovery_invocations": float(n.get("recovery_invocations", 0)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_case(self, case: RecoveryCase) -> None:
        path = self.storage_dir / f"{case.case_id}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(case.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            logger.warning("Failed to persist case %s", case.case_id)

    def _load_existing(self) -> None:
        for path in sorted(self.storage_dir.glob("case_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.cases.append(RecoveryCase.from_dict(data))
            except Exception:
                logger.warning("Failed to load case from %s", path)

    def save_skills(self, skills: list[RecoverySkill]) -> Path:
        path = self.storage_dir / "skills.json"
        payload = [s.to_dict() for s in skills]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load_skills(self) -> list[RecoverySkill]:
        path = self.storage_dir / "skills.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [RecoverySkill.from_dict(s) for s in data]
        except Exception:
            return []
