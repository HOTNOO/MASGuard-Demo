"""BCMR clean-start and recovery coordinator (v3: program-level recovery)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import time
import uuid

from bcmr_swe import OUTPUT_ROOT
from bcmr_swe.provenance import ProvenanceRecorder, SuspectRegionExtractor
from bcmr_swe.recovery import (
    CounterfactualDatasetBuilder,
    FailureTriggerDetector,
    RecoveryActionPlanner,
    RecoverySelector,
    ReplayEngine,
    StateEncoder,
)
from bcmr_swe.recovery.case_memory import CaseMemory
from bcmr_swe.recovery.program_executor import (
    ProgramExecutor,
    locator_result_allows_recovery_replay,
    locator_result_is_recovery_best_effort,
)
from bcmr_swe.recovery.program_synthesizer import ProgramSynthesizer
from bcmr_swe.types import (
    ActionType,
    CandidateAction,
    FailedState,
    ProgramOutcome,
    RecoveryBudget,
    RecoveryProgram,
    ReplayOutcome,
)

logger = logging.getLogger(__name__)


def extract_failing_tests(verification_text: str) -> list[str]:
    text = verification_text or ""
    candidates = re.findall(r"[A-Za-z0-9_./\\-]+::[A-Za-z0-9_./\\-]+", text)
    unittest_cases = re.findall(r"\b(test_[A-Za-z0-9_]+)\s+\(([A-Za-z0-9_.]+)\)", text)
    candidates.extend(f"{case}.{test}" for test, case in unittest_cases)
    return list(dict.fromkeys(candidates))[:8]


def validation_failure_signature(*, command: str, output: str, verification: str = "") -> dict[str, object]:
    text = "\n".join(part for part in (command or "", output or "", verification or "") if part)
    tool_missing = bool(
        re.search(r"No module named (pytest|nose|nosetests)\b|(?:pytest|nosetests): command not found", text)
    )
    exception_classes = re.findall(
        r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception))\b",
        text,
    )
    traceback_files = re.findall(r'File "([^"]+\.py)"', text)
    project_traceback_files = [
        path
        for path in traceback_files
        if "/site-packages/" not in path and "/lib/python" not in path
    ]
    return {
        "validation_tool_missing": tool_missing,
        "exception_classes": list(dict.fromkeys(exception_classes))[:6],
        "failing_tests": extract_failing_tests(text),
        "traceback_source_files": list(dict.fromkeys(project_traceback_files))[:8],
    }


@dataclass
class BCMRCoordinatorConfig:
    output_root: Path = OUTPUT_ROOT
    budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    persist_counterfactuals: bool = True
    capture_full_counterfactual_group: bool = False
    capture_counterfactual_outcomes: bool = True
    execute_live_after_full_capture: bool = True
    continue_after_full_capture: bool = True
    max_captured_failed_state_groups: int | None = 1
    counterfactual_followup_recovery_calls: int = 0
    initial_recovery_context: str = ""


class BCMRCoordinator:
    """Run a SWE task and invoke BCMR when the run enters a failed state.

    v3 architecture: when a ``ProgramSynthesizer`` is provided the
    coordinator uses the three-layer recovery path:

        Layer 1 – atomic operators (INSPECT/REVOKE/ROLLBACK/REPLAY/ESCALATE)
        Layer 2 – LLM synthesises candidate programs → counterfactual replay
        Layer 3 – case memory stores verified experiences

    Without a synthesiser it falls back to the v2 single-action path for
    backward compatibility.
    """

    STAGE_ORDER = ("locator", "patcher", "verifier")
    CHECKPOINT_PROGRESS = {
        "initial": 0.0,
        "post_locate": 1.0,
        "post_patch": 2.0,
        "post_verify": 3.0,
    }

    def __init__(
        self,
        *,
        locator,
        patcher,
        verifier,
        selector: RecoverySelector | None = None,
        synthesizer: ProgramSynthesizer | None = None,
        config: BCMRCoordinatorConfig | None = None,
    ):
        self.locator = locator
        self.patcher = patcher
        self.verifier = verifier
        self.config = config or BCMRCoordinatorConfig()
        self.trigger_detector = FailureTriggerDetector()
        self.region_extractor = SuspectRegionExtractor()
        self.state_encoder = StateEncoder()

        self.synthesizer = synthesizer
        self.program_executor = ProgramExecutor() if synthesizer else None
        self.case_memory: CaseMemory | None = None

        self.action_planner = RecoveryActionPlanner()
        self.selector = selector or RecoverySelector()
        self.replay_engine = ReplayEngine()
        self.dataset_builder: CounterfactualDatasetBuilder | None = None

        self.issue = ""
        self.workspace = ""
        self.instance_id = ""
        self.run_id = ""
        self.run_dir: Path | None = None
        self.recorder: ProvenanceRecorder | None = None
        self.shared_facts: dict[str, dict] = {}
        self.stage_outputs: dict[str, dict] = {}
        self.stage_nodes: dict[str, str] = {}
        self.recovery_calls = 0
        self.captured_failed_state_groups = 0
        self.selected_actions: list[dict] = []
        self.executed_programs: list[dict] = []
        self._pending_verifier_failure_feedback = ""
        self._recovery_enabled = True
        self.required_verifier_command = ""

    def run(self, *, issue: str, workspace: str, instance_id: str = "unknown", run_id: str | None = None) -> dict:
        self.issue = issue
        self.workspace = workspace
        self.instance_id = instance_id
        self.run_id = run_id or f"bcmr_{uuid.uuid4().hex[:8]}"
        self.run_dir = self.config.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = ProvenanceRecorder(run_dir=self.run_dir, workspace=workspace)
        self.dataset_builder = CounterfactualDatasetBuilder(self.run_dir / "failed_states")
        if self.synthesizer is not None:
            self.case_memory = CaseMemory(self.run_dir / "case_memory")

        self.recorder.create_checkpoint(label="initial", metadata={"resume_from": "locator"})
        success = self._resume_from(
            "locator",
            recovery_context=str(self.config.initial_recovery_context or ""),
            escalation_level=0,
            deep_verify=False,
        )

        result = {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "success": success,
            "recovery_mode": "v3_program" if self.synthesizer else "v2_action",
            "captured_failed_state_groups": self.captured_failed_state_groups,
            "selected_actions": list(self.selected_actions),
            "executed_programs": list(self.executed_programs),
            "shared_facts": dict(self.shared_facts),
            "stage_outputs": dict(self.stage_outputs),
        }
        if self.case_memory:
            result["case_memory_summary"] = self.case_memory.summary()
        (self.run_dir / "run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def apply_recovery_action(self, action: CandidateAction) -> ReplayOutcome:
        started = time.time()
        usage_before = self._usage_snapshot()
        checkpoint_before = self.recorder.latest_checkpoint()
        progress_before = self._progress_score()
        stage_status_before = self._stage_status_snapshot()
        if action.action_type == ActionType.QUARANTINE_FACT:
            fact_id = str(action.payload.get("fact_node_id", ""))
            fact_node = self.recorder.graph.get_node(fact_id)
            if fact_node:
                fact_node.status = "quarantined"
                self.shared_facts.pop(str(fact_node.payload.get("fact_key", "")), None)
            success = self._restore_and_resume(action, recovery_context="A conflicting shared fact was quarantined.")
            return self._finalize_replay_outcome(
                action=action,
                success=success,
                started=started,
                usage_before=usage_before,
                checkpoint_before=checkpoint_before,
                progress_before=progress_before,
                stage_status_before=stage_status_before,
                notes="Quarantined conflicting fact before replay.",
            )

        if action.action_type == ActionType.ROLLBACK_TO_CHECKPOINT:
            success = self._restore_and_resume(action, recovery_context="Resuming from a healthy checkpoint after rollback.")
            return self._finalize_replay_outcome(
                action=action,
                success=success,
                started=started,
                usage_before=usage_before,
                checkpoint_before=checkpoint_before,
                progress_before=progress_before,
                stage_status_before=stage_status_before,
                notes="Restored checkpoint and resumed downstream stages.",
            )

        if action.action_type == ActionType.REPLAY_SUBGRAPH:
            success = self._restore_and_resume(action, recovery_context="Replay the suspect subgraph only and avoid repeating prior errors.")
            return self._finalize_replay_outcome(
                action=action,
                success=success,
                started=started,
                usage_before=usage_before,
                checkpoint_before=checkpoint_before,
                progress_before=progress_before,
                stage_status_before=stage_status_before,
                notes="Subgraph replay completed.",
            )

        if action.action_type == ActionType.INSERT_VERIFIER:
            success = self._resume_from("verifier", recovery_context="Perform an additional, deeper verification before trusting the current patch.", escalation_level=0, deep_verify=True)
            return self._finalize_replay_outcome(
                action=action,
                success=success,
                started=started,
                usage_before=usage_before,
                checkpoint_before=checkpoint_before,
                progress_before=progress_before,
                stage_status_before=stage_status_before,
                notes="Inserted an extra verification pass.",
            )

        success = self._restore_and_resume(action, recovery_context="Escalate the suspect node with a stronger strategy and continue.", escalation_level=int(action.payload.get("escalation_level", 1)))
        return self._finalize_replay_outcome(
            action=action,
            success=success,
            started=started,
            usage_before=usage_before,
            checkpoint_before=checkpoint_before,
            progress_before=progress_before,
            stage_status_before=stage_status_before,
            notes="Escalated the suspect stage and replayed from its anchor.",
        )

    def _finalize_replay_outcome(
        self,
        *,
        action: CandidateAction,
        success: bool,
        started: float,
        usage_before: dict[str, float],
        checkpoint_before,
        progress_before: float,
        stage_status_before: dict[str, bool],
        notes: str,
    ) -> ReplayOutcome:
        usage_after = self._usage_snapshot()
        token_cost = self._usage_delta(usage_before, usage_after)
        latency_sec = max(0.001, time.time() - started)
        checkpoint_after = self.recorder.latest_checkpoint()
        rollback_depth = self._rollback_depth(checkpoint_before, action)
        progress_after = self._progress_score()
        milestone_gain = max(0.0, progress_after - progress_before)
        metadata = {
            "token_cost_label_source": "usage_delta" if token_cost > 0 else "estimated_fallback",
            "estimated_token_cost": action.estimated_token_cost,
            "estimated_latency_sec": action.estimated_latency_sec,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "checkpoint_before": checkpoint_before.to_dict() if checkpoint_before else None,
            "checkpoint_after": checkpoint_after.to_dict() if checkpoint_after else None,
            "progress_before": progress_before,
            "progress_after": progress_after,
            "stage_status_before": stage_status_before,
            "stage_status_after": self._stage_status_snapshot(),
            "last_failed_stage": self._last_failed_stage(),
            "last_failure_reason": self._last_failure_reason(),
        }
        return ReplayOutcome(
            action_id=action.action_id,
            action_type=action.action_type,
            recover_success=success,
            official_resolved=success,
            token_cost=token_cost if token_cost > 0 else action.estimated_token_cost,
            latency_sec=latency_sec,
            rollback_depth=rollback_depth,
            secondary_risk=action.estimated_risk,
            milestone_gain=milestone_gain,
            notes=notes,
            metadata=metadata,
        )

    def _restore_and_resume(self, action: CandidateAction, *, recovery_context: str, escalation_level: int = 0) -> bool:
        checkpoint = self.recorder.get_checkpoint(str(action.payload.get("checkpoint_id", ""))) or self.recorder.latest_checkpoint()
        if checkpoint:
            self.recorder.checkpoint_store.restore(checkpoint, self.workspace)
        resume_from = str(action.payload.get("resume_from", "patcher"))
        recovery_context = self._attach_pending_verifier_failure_feedback(
            resume_from=resume_from,
            recovery_context=recovery_context,
        )
        self._clear_stage_outputs_from(resume_from)
        return self._resume_from(resume_from, recovery_context=recovery_context, escalation_level=escalation_level, deep_verify=False)

    def _resume_from(self, stage: str, *, recovery_context: str, escalation_level: int, deep_verify: bool) -> bool:
        stage = stage.lower()
        logger.info(
            "[BCMR-RESUME] stage=%s recovery_context=%s escalation_level=%s deep_verify=%s",
            stage,
            "yes" if recovery_context.strip() else "no",
            escalation_level,
            deep_verify,
        )
        if stage == "locator":
            locate_result = self._run_locator(recovery_context=recovery_context, escalation_level=escalation_level)
            if not locator_result_allows_recovery_replay(
                locate_result,
                recovery_context=recovery_context,
            ):
                logger.info("[BCMR-FAILURE] locator returned unsuccessful; delegating to failure handler")
                return self._handle_failure()
            if locator_result_is_recovery_best_effort(
                locate_result,
                recovery_context=recovery_context,
            ):
                self._accept_locator_best_effort_for_recovery(locate_result)
            stage = "patcher"
        if stage == "patcher":
            patch_result = self._run_patcher(recovery_context=recovery_context, escalation_level=escalation_level)
            if not patch_result.get("success"):
                logger.info("[BCMR-FAILURE] patcher returned unsuccessful; delegating to failure handler")
                return self._handle_failure()
            stage = "verifier"
        if stage == "verifier":
            verify_result = self._run_verifier(recovery_context=recovery_context, deep_verify=deep_verify)
            if verify_result.get("success"):
                return True
            logger.info("[BCMR-FAILURE] verifier returned unsuccessful; delegating to failure handler")
            return self._handle_failure()
        return False

    def _run_locator(self, *, recovery_context: str, escalation_level: int) -> dict:
        result = self.locator.locate(self.issue, self.workspace, recovery_context=recovery_context, escalation_level=escalation_level)
        summary = result.get("located_files", "")
        success = bool(result.get("success"))
        node = self.recorder.record_agent_step(
            role="locator",
            phase="locate",
            content=summary or "locator_result",
            payload={"success": success, "recovery_context": recovery_context},
        )
        self.stage_nodes["locator"] = node.node_id
        self.stage_outputs["locator"] = result

        path_fact = self._extract_primary_path(summary) if success else ""
        if path_fact:
            fact_node = self.recorder.record_shared_fact(
                key="localized_path",
                value=path_fact,
                role="locator",
                phase="locate",
                source_node_id=node.node_id,
                confidence=0.75,
            )
            self.shared_facts["localized_path"] = {"value": path_fact, "node_id": fact_node.node_id}

        for command in result.get("commands", []):
            self.recorder.record_tool_call(
                role="locator",
                phase="locate",
                command=command.get("command", ""),
                output=command.get("output", ""),
                returncode=int(command.get("returncode", 0)),
                depends_on=[node.node_id],
            )

        if success:
            self.recorder.create_checkpoint(
                label="post_locate",
                metadata={"resume_from": "patcher", "stage": "locator"},
                source_node_id=node.node_id,
            )
        self.recorder.save()
        return result

    def _accept_locator_best_effort_for_recovery(self, locate_result: dict) -> None:
        """Promote low-confidence locator candidates only within recovery replay."""

        located_files = str(locate_result.get("located_files", "") or "").strip()
        if not located_files:
            return
        locate_result["accepted_for_recovery_replay"] = True
        locate_result["locator_confidence"] = "best_effort_recovery"
        stage_locator = self.stage_outputs.get("locator")
        if isinstance(stage_locator, dict):
            stage_locator["accepted_for_recovery_replay"] = True
            stage_locator["locator_confidence"] = "best_effort_recovery"

        path_fact = self._extract_primary_path(located_files)
        if not path_fact or self.recorder is None:
            return
        source_node_id = self.stage_nodes.get("locator", "")
        fact_node = self.recorder.record_shared_fact(
            key="localized_path",
            value=path_fact,
            role="locator",
            phase="locate",
            source_node_id=source_node_id,
            confidence=0.45,
            payload={
                "best_effort_only": True,
                "accepted_for_recovery_replay": True,
            },
        )
        self.shared_facts["localized_path"] = {
            "value": path_fact,
            "node_id": fact_node.node_id,
            "confidence": 0.45,
            "best_effort_only": True,
            "accepted_for_recovery_replay": True,
        }
        self.recorder.save()

    def _run_patcher(self, *, recovery_context: str, escalation_level: int) -> dict:
        located_files = self.stage_outputs.get("locator", {}).get("located_files", "")
        result = self.patcher.patch(
            self.issue,
            self.workspace,
            located_files=located_files,
            recovery_context=recovery_context,
            escalation_level=escalation_level,
        )
        node = self.recorder.record_agent_step(
            role="patcher",
            phase="patch",
            content=result.get("plan", "") or result.get("patch", "") or "patcher_result",
            payload={"success": bool(result.get("success")), "patch": result.get("patch", "")[:4000]},
            depends_on=[self.stage_nodes["locator"]] if "locator" in self.stage_nodes else None,
            reads=[self.shared_facts["localized_path"]["node_id"]] if "localized_path" in self.shared_facts else None,
        )
        self.stage_nodes["patcher"] = node.node_id
        self.stage_outputs["patcher"] = result

        for command in result.get("commands", []):
            self.recorder.record_tool_call(
                role="patcher",
                phase="patch",
                command=command.get("command", ""),
                output=command.get("output", ""),
                returncode=int(command.get("returncode", 0)),
                depends_on=[node.node_id],
            )

        if result.get("patch"):
            patch_fact = self.recorder.record_shared_fact(
                key="latest_patch",
                value=result.get("patch", "")[:400],
                role="patcher",
                phase="patch",
                source_node_id=node.node_id,
                confidence=0.65,
            )
            self.shared_facts["latest_patch"] = {"value": result.get("patch", ""), "node_id": patch_fact.node_id}

        if result.get("success"):
            self.recorder.create_checkpoint(
                label="post_patch",
                metadata={"resume_from": "verifier", "stage": "patcher"},
                source_node_id=node.node_id,
            )
        self.recorder.save()
        return result

    def _run_verifier(self, *, recovery_context: str, deep_verify: bool) -> dict:
        patch = self.stage_outputs.get("patcher", {}).get("patch", "")
        recovery_context = self._attach_required_verifier_command(recovery_context)
        result = self.verifier.verify(
            self.issue,
            self.workspace,
            patch=patch,
            recovery_context=recovery_context,
            deep_verify=deep_verify,
        )

        verification_text = result.get("verification", "")
        failing_tests = self._extract_failing_tests(verification_text)
        contradicted_fact_ids = []
        verdict = "pass" if result.get("success") else "fail"
        status_text = str(result.get("status", "")).lower()
        if not result.get("success") and "latest_patch" in self.shared_facts:
            contradicted_fact_ids.append(self.shared_facts["latest_patch"]["node_id"])
            self.recorder.mark_fact_conflict(
                self.shared_facts["latest_patch"]["node_id"],
                reason="Patch fact contradicted by verifier failure.",
            )
        if status_text in {"通过", "pass", "passed"} and failing_tests:
            verdict = "promote"

        node = self.recorder.record_verifier_result(
            role="verifier",
            phase="verify",
            verdict=verdict,
            test_status="pass" if result.get("success") else "fail",
            failing_tests=failing_tests,
            output=verification_text,
            depends_on=[self.stage_nodes["patcher"]] if "patcher" in self.stage_nodes else None,
            contradicted_fact_ids=contradicted_fact_ids,
            failure_signature="|".join(failing_tests[:3]),
        )
        self.stage_nodes["verifier"] = node.node_id
        self.stage_outputs["verifier"] = result
        if not result.get("success"):
            self._pending_verifier_failure_feedback = self._build_verifier_failure_feedback(result)
        self.recorder.save()
        return result

    def _attach_required_verifier_command(self, recovery_context: str) -> str:
        command = str(getattr(self, "required_verifier_command", "") or "").strip()
        if not command:
            return recovery_context
        instruction = (
            "[REQUIRED VERIFICATION SCOPE]\n"
            "Before declaring the patch successful, run this exact non-interactive "
            f"validation command and base the final verdict on it:\n{command}"
        )
        base = str(recovery_context or "").strip()
        if instruction in base:
            return base
        return f"{instruction}\n\n{base}".strip()

    def _handle_failure(self) -> bool:
        if not self._recovery_enabled:
            return False
        trigger = self.trigger_detector.detect(self.recorder.graph)
        if trigger is None or self.recovery_calls >= self.config.budget.max_recovery_calls:
            return False

        self.recovery_calls += 1
        suspect_region = self.region_extractor.extract(self.recorder.graph, trigger)
        checkpoint = self.recorder.latest_checkpoint()
        failed_state_metadata = self._build_failed_state_metadata(suspect_region, checkpoint)
        failed_state = FailedState(
            group_id=f"fs_{uuid.uuid4().hex[:10]}",
            run_id=self.run_id,
            instance_id=self.instance_id,
            trigger=trigger,
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
            checkpoint=checkpoint,
            suspect_region=suspect_region,
            state_features=self.state_encoder.encode(
                self.recorder.graph,
                FailedState(
                    group_id="tmp",
                    run_id=self.run_id,
                    instance_id=self.instance_id,
                    trigger=trigger,
                    checkpoint_id=checkpoint.checkpoint_id if checkpoint else "",
                    checkpoint=checkpoint,
                    suspect_region=suspect_region,
                    state_features=None,  # type: ignore[arg-type]
                    metadata=failed_state_metadata,
                ),
            ),
            metadata=failed_state_metadata,
        )

        if self.synthesizer is not None:
            return self._handle_failure_v3(failed_state)
        return self._handle_failure_v2(failed_state)

    # ------------------------------------------------------------------
    # v3 path: program synthesis → counterfactual replay → case memory
    # ------------------------------------------------------------------

    def _handle_failure_v3(self, failed_state: FailedState) -> bool:
        similar_cases = []
        if self.case_memory:
            similar_cases = self.case_memory.retrieve(failed_state, top_k=3)

        candidate_programs = self.synthesizer.synthesize(
            self.recorder.graph,
            failed_state,
            self.config.budget,
            similar_cases=similar_cases,
            used_recovery_calls=self.recovery_calls,
            used_tokens=self._total_tokens_used(),
        )
        if not candidate_programs:
            logger.warning("No candidate programs synthesised; skipping recovery.")
            return False

        synthesis_record = self.synthesizer.get_synthesis_record()

        if self.config.capture_full_counterfactual_group:
            logger.info(
                "[BCMR-CAPTURE] Failure detected at stage=%s; capturing counterfactual program group from checkpoint=%s.",
                self._last_failed_stage(),
                failed_state.checkpoint_id or "<none>",
            )
            if self.config.capture_counterfactual_outcomes:
                outcomes = self._evaluate_programs_counterfactual(
                    failed_state, candidate_programs
                )
            else:
                logger.info(
                    "[BCMR-CAPTURE] Counterfactual outcome execution is disabled; writing failed-state group with candidate programs only."
                )
                outcomes = []
            if self.dataset_builder and self.config.persist_counterfactuals:
                self.dataset_builder.write_program_group(
                    failed_state, candidate_programs, outcomes,
                    synthesis_record=synthesis_record,
                )
            self.captured_failed_state_groups += 1

            if not self.config.execute_live_after_full_capture:
                logger.info(
                    "[BCMR-CAPTURE] Counterfactual group captured; live recovery execution is disabled, so the run will stop after capture."
                )
                return False

            best = self._select_best_program(candidate_programs, outcomes)

            live_outcome = self.program_executor.execute(self, best)

            if self.case_memory:
                self.case_memory.commit(failed_state, best, live_outcome)
            self.executed_programs.append({
                "failed_state": failed_state.to_dict(),
                "programs": [p.to_dict() for p in candidate_programs],
                "counterfactual_outcomes": [o.to_dict() for o in outcomes],
                "chosen": best.to_dict(),
                "live_outcome": live_outcome.to_dict(),
            })

            if not self.config.continue_after_full_capture or self._reached_capture_limit():
                return live_outcome.recover_success
            return live_outcome.recover_success

        best = self._select_best_program(candidate_programs, [])
        outcome = self.program_executor.execute(self, best)

        if self.dataset_builder and self.config.persist_counterfactuals:
            self.dataset_builder.write_program_group(
                failed_state, candidate_programs, [outcome],
                synthesis_record=synthesis_record,
            )
        if self.case_memory:
            self.case_memory.commit(failed_state, best, outcome)

        self.executed_programs.append({
            "failed_state": failed_state.to_dict(),
            "chosen": best.to_dict(),
            "live_outcome": outcome.to_dict(),
        })
        return outcome.recover_success

    def _evaluate_programs_counterfactual(
        self,
        failed_state: FailedState,
        programs: list[RecoveryProgram],
    ) -> list[ProgramOutcome]:
        """Execute each candidate program on a restored baseline and collect outcomes.

        After each program, ALL mutable state is restored, including agent
        configs that ESCALATE may have mutated.
        """
        baseline_shared_facts = copy.deepcopy(self.shared_facts)
        baseline_stage_outputs = copy.deepcopy(self.stage_outputs)
        baseline_stage_nodes = copy.deepcopy(self.stage_nodes)
        baseline_graph = copy.deepcopy(self.recorder.graph)
        baseline_checkpoints = copy.deepcopy(self.recorder._checkpoints)
        baseline_recovery_calls = self.recovery_calls
        baseline_recovery_enabled = self._recovery_enabled
        baseline_agent_configs = self._snapshot_agent_configs()

        outcomes: list[ProgramOutcome] = []
        for program in programs:
            checkpoint = self.recorder.latest_checkpoint()
            if checkpoint:
                self.recorder.checkpoint_store.restore(checkpoint, self.workspace)
            self.shared_facts = copy.deepcopy(baseline_shared_facts)
            self.stage_outputs = copy.deepcopy(baseline_stage_outputs)
            self.stage_nodes = copy.deepcopy(baseline_stage_nodes)
            self.recorder.graph = copy.deepcopy(baseline_graph)
            self.recorder._checkpoints = copy.deepcopy(baseline_checkpoints)
            self.recovery_calls = baseline_recovery_calls
            self._recovery_enabled = False
            self._restore_agent_configs(baseline_agent_configs)

            outcome = self.program_executor.execute(self, program)
            outcomes.append(outcome)

        checkpoint = self.recorder.latest_checkpoint()
        if checkpoint:
            self.recorder.checkpoint_store.restore(checkpoint, self.workspace)
        self.shared_facts = baseline_shared_facts
        self.stage_outputs = baseline_stage_outputs
        self.stage_nodes = baseline_stage_nodes
        self.recorder.graph = baseline_graph
        self.recorder._checkpoints = baseline_checkpoints
        self.recovery_calls = baseline_recovery_calls
        self._recovery_enabled = baseline_recovery_enabled
        self._restore_agent_configs(baseline_agent_configs)
        return outcomes

    def _snapshot_agent_configs(self) -> dict[str, dict[str, object]]:
        """Capture mutable configs and model routes that recovery actions may change."""
        snapshot: dict[str, dict[str, object]] = {}
        for name in ("locator", "patcher", "verifier"):
            agent = getattr(self, name, None)
            if agent is None:
                continue
            cfg: dict[str, object] = {}
            for attr in ("max_iterations", "max_patch_iterations", "max_plan_iterations"):
                val = getattr(agent, attr, None)
                if isinstance(val, int):
                    cfg[attr] = val
            for attr in ("model", "planner_model", "implementer_model"):
                if hasattr(agent, attr):
                    cfg[attr] = getattr(agent, attr)
            agent_config = getattr(agent, "config", None)
            if agent_config is not None:
                val = getattr(agent_config, "max_iterations", None)
                if isinstance(val, int):
                    cfg["config.max_iterations"] = val
            snapshot[name] = cfg
        return snapshot

    def _restore_agent_configs(self, snapshot: dict[str, dict[str, object]]) -> None:
        for name, cfg in snapshot.items():
            agent = getattr(self, name, None)
            if agent is None:
                continue
            for attr, val in cfg.items():
                if attr.startswith("config."):
                    agent_config = getattr(agent, "config", None)
                    if agent_config is not None:
                        setattr(agent_config, attr.split(".", 1)[1], val)
                else:
                    setattr(agent, attr, val)

    def _select_best_program(
        self,
        programs: list[RecoveryProgram],
        outcomes: list[ProgramOutcome],
    ) -> RecoveryProgram:
        """Pick the best program using budget-constrained utility.

        If replay outcomes are available, rank by actual results.
        Otherwise rank by LLM estimates.
        """
        budget = self.config.budget
        if outcomes:
            outcome_map = {o.program_id: o for o in outcomes}
            def actual_utility(prog: RecoveryProgram) -> float:
                o = outcome_map.get(prog.program_id)
                if o is None:
                    return -999.0
                return (
                    (1.0 if o.recover_success else 0.0)
                    - budget.lambda_token * o.token_cost
                    - budget.lambda_latency * o.latency_sec
                    - budget.lambda_risk * o.secondary_risk
                )
            return max(programs, key=actual_utility)

        def estimated_utility(prog: RecoveryProgram) -> float:
            return (
                prog.estimated_recover_prob
                - budget.lambda_token * prog.estimated_total_cost
                - budget.lambda_risk * prog.estimated_risk
            )
        return max(programs, key=estimated_utility)

    # ------------------------------------------------------------------
    # v2 path: single-action selection (backward compatible)
    # ------------------------------------------------------------------

    def _handle_failure_v2(self, failed_state: FailedState) -> bool:
        actions = self.action_planner.enumerate(self.recorder.graph, failed_state)
        teacher_predictions = self._collect_teacher_predictions(failed_state, actions)

        ranked = self.selector.rank(
            failed_state,
            actions,
            self.config.budget,
        )
        action_by_id = {action.action_id: action for action in actions}
        chosen = action_by_id[ranked[0].action_id]
        self.selected_actions.append({
            "failed_state": failed_state.to_dict(),
            "scores": [item.to_dict() for item in ranked],
            "chosen": chosen.to_dict(),
        })

        if self.config.capture_full_counterfactual_group:
            outcomes = self._evaluate_counterfactual_group(actions)
            if self.dataset_builder and self.config.persist_counterfactuals:
                self.dataset_builder.write_group(
                    failed_state,
                    actions,
                    outcomes,
                    teacher_predictions=teacher_predictions,
                )
            self.captured_failed_state_groups += 1
            outcome = self.replay_engine.execute(self, chosen)
            return outcome.recover_success

        outcome = self.replay_engine.execute(self, chosen)
        if self.dataset_builder and self.config.persist_counterfactuals:
            self.dataset_builder.write_group(
                failed_state,
                actions,
                [outcome],
                teacher_predictions=teacher_predictions,
            )
        return outcome.recover_success

    def _reached_capture_limit(self) -> bool:
        limit = self.config.max_captured_failed_state_groups
        return limit is not None and limit > 0 and self.captured_failed_state_groups >= limit

    def _collect_teacher_predictions(
        self,
        failed_state: FailedState,
        actions: list[CandidateAction],
    ) -> dict | None:
        """Ask the LLM selector (if active) for predictions *before* replay.

        The predictions are recorded alongside the real replay outcomes so
        that the Phase 2 student can learn the calibration mapping:
            (LLM prediction, state features) → actual outcome
        """
        getter = getattr(self.selector, "get_prediction_record", None)
        if not callable(getter):
            return None
        llm_sel = getattr(self.selector, "llm_selector", None)
        if llm_sel is None:
            return None
        try:
            self.selector.rank(
                failed_state,
                actions,
                self.config.budget,
                graph=self.recorder.graph,
                used_recovery_calls=self.recovery_calls,
                used_tokens=self._total_tokens_used(),
            )
            record = getter()
            return record if record else None
        except Exception:
            return None

    def _total_tokens_used(self) -> float:
        snapshot = self._usage_snapshot()
        return float(snapshot.get("total_tokens", 0.0))

    def _model_name(self, candidate: object | None) -> str:
        if candidate is None:
            return ""
        base = getattr(candidate, "base_model", candidate)
        for probe in (base, candidate):
            config = getattr(probe, "config", None)
            name = str(getattr(config, "model", "") or "").strip()
            if name:
                return name
        return ""

    def _model_routes_snapshot(self) -> dict[str, object]:
        routes = {
            "locator_model": self._model_name(getattr(self.locator, "model", None)),
            "patcher_planner_model": self._model_name(getattr(self.patcher, "planner_model", None)),
            "patcher_implementer_model": self._model_name(getattr(self.patcher, "implementer_model", None)),
            "verifier_model": self._model_name(getattr(self.verifier, "model", None)),
        }
        active_models = sorted({value for value in routes.values() if value})
        return {
            **routes,
            "active_models": active_models,
        }

    def _build_failed_state_metadata(self, suspect_region, checkpoint) -> dict[str, object]:
        checkpoint_candidates = sorted(self.recorder._checkpoints.values(), key=lambda item: item.created_at)
        return {
            "checkpoint_depth": 1.0 if checkpoint else 0.0,
            "recovery_invocations": float(self.recovery_calls),
            "suspect_graph": self.recorder.graph.subgraph(suspect_region.node_ids),
            "checkpoint_candidates": [record.to_dict() for record in checkpoint_candidates[-4:]],
            "phase_outputs": copy.deepcopy(self.stage_outputs),
            "latest_test_status": copy.deepcopy(self.stage_outputs.get("verifier", {})),
            "stage_nodes": copy.deepcopy(self.stage_nodes),
            "model_routes": self._model_routes_snapshot(),
        }

    def _evaluate_counterfactual_group(self, actions: list[CandidateAction]) -> list[ReplayOutcome]:
        baseline_shared_facts = copy.deepcopy(self.shared_facts)
        baseline_stage_outputs = copy.deepcopy(self.stage_outputs)
        baseline_stage_nodes = copy.deepcopy(self.stage_nodes)
        baseline_graph = copy.deepcopy(self.recorder.graph)
        baseline_checkpoints = copy.deepcopy(self.recorder._checkpoints)
        baseline_selected_actions = copy.deepcopy(self.selected_actions)
        baseline_recovery_calls = self.recovery_calls
        baseline_recovery_enabled = self._recovery_enabled
        baseline_capture_full = self.config.capture_full_counterfactual_group
        baseline_continue_after_full = self.config.continue_after_full_capture
        baseline_persist_counterfactuals = self.config.persist_counterfactuals
        baseline_budget_max = self.config.budget.max_recovery_calls

        outcomes: list[ReplayOutcome] = []
        for action in actions:
            checkpoint = self.recorder.get_checkpoint(str(action.payload.get("checkpoint_id", ""))) or self.recorder.latest_checkpoint()
            if checkpoint:
                self.recorder.checkpoint_store.restore(checkpoint, self.workspace)
            self.shared_facts = copy.deepcopy(baseline_shared_facts)
            self.stage_outputs = copy.deepcopy(baseline_stage_outputs)
            self.stage_nodes = copy.deepcopy(baseline_stage_nodes)
            self.recorder.graph = copy.deepcopy(baseline_graph)
            self.recorder._checkpoints = copy.deepcopy(baseline_checkpoints)
            self.selected_actions = copy.deepcopy(baseline_selected_actions)
            self.recovery_calls = baseline_recovery_calls
            self._recovery_enabled = self.config.counterfactual_followup_recovery_calls > 0
            self.config.capture_full_counterfactual_group = False
            self.config.continue_after_full_capture = False
            self.config.persist_counterfactuals = False
            self.config.budget.max_recovery_calls = baseline_recovery_calls + max(
                0,
                int(self.config.counterfactual_followup_recovery_calls),
            )
            outcomes.append(self.apply_recovery_action(action))

        checkpoint = self.recorder.latest_checkpoint()
        if checkpoint:
            self.recorder.checkpoint_store.restore(checkpoint, self.workspace)
        self.shared_facts = baseline_shared_facts
        self.stage_outputs = baseline_stage_outputs
        self.stage_nodes = baseline_stage_nodes
        self.recorder.graph = baseline_graph
        self.recorder._checkpoints = baseline_checkpoints
        self.selected_actions = baseline_selected_actions
        self.recovery_calls = baseline_recovery_calls
        self._recovery_enabled = baseline_recovery_enabled
        self.config.capture_full_counterfactual_group = baseline_capture_full
        self.config.continue_after_full_capture = baseline_continue_after_full
        self.config.persist_counterfactuals = baseline_persist_counterfactuals
        self.config.budget.max_recovery_calls = baseline_budget_max
        self.recorder.save()
        return outcomes

    def _clear_stage_outputs_from(self, stage: str) -> None:
        if stage not in self.STAGE_ORDER:
            return
        index = self.STAGE_ORDER.index(stage)
        for stale_stage in self.STAGE_ORDER[index:]:
            self.stage_outputs.pop(stale_stage, None)
            self.stage_nodes.pop(stale_stage, None)

    def _extract_primary_path(self, text: str) -> str:
        matches = re.findall(r"[A-Za-z0-9_./\\-]+\.py", text or "")
        return matches[0] if matches else ""

    def _extract_failing_tests(self, verification_text: str) -> list[str]:
        return extract_failing_tests(verification_text)

    def _validation_failure_signature(self, *, command: str, output: str, verification: str = "") -> dict[str, object]:
        return validation_failure_signature(command=command, output=output, verification=verification)

    def _build_verifier_failure_feedback(self, verify_result: dict) -> str:
        commands = [dict(item) for item in list(verify_result.get("commands", []) or []) if isinstance(item, dict)]
        validation_commands = [
            item
            for item in commands
            if any(
                token in str(item.get("command", "")).lower()
                for token in ("pytest", "nosetests", "unittest", "runtests.py", "manage.py test")
            )
        ]
        command_row = validation_commands[-1] if validation_commands else (commands[-1] if commands else {})
        command = str(command_row.get("command", "") or "").strip()
        output = str(command_row.get("output", "") or "").strip()
        returncode = command_row.get("returncode", "")
        verification = str(verify_result.get("verification", "") or "").strip()
        status = str(verify_result.get("status", "") or "").strip()
        stop_reason = str(verify_result.get("stop_reason", "") or "").strip()
        signature_commands = validation_commands or ([command_row] if command_row else [])
        signature = self._validation_failure_signature(
            command="\n".join(str(item.get("command", "") or "") for item in signature_commands),
            output="\n".join(str(item.get("output", "") or "") for item in signature_commands),
            verification=verification,
        )
        feedback_lines = [
            "[VERIFIER FAILURE FEEDBACK]",
            "The previous patch failed bounded validation; the next patcher attempt must revise the source diff to address this evidence instead of repeating the same edit.",
        ]
        if status:
            feedback_lines.append(f"Verifier status: {status[:200]}")
        if stop_reason:
            feedback_lines.append(f"Verifier stop_reason: {stop_reason[:200]}")
        if command:
            feedback_lines.append(f"Validation command: {command[:300]}")
        if returncode != "":
            feedback_lines.append(f"Validation returncode: {returncode}")
        if output:
            feedback_lines.append(f"Validation output excerpt:\n{output[:1200]}")
        elif verification:
            feedback_lines.append(f"Verifier explanation excerpt:\n{verification[:1200]}")
        if any(signature.values()):
            feedback_lines.append("[STRUCTURED VALIDATION FAILURE]")
            if signature["validation_tool_missing"]:
                feedback_lines.append(
                    "validation_tool_missing: true; use the repository-native focused validation entrypoint when the requested test runner is unavailable."
                )
            if signature["failing_tests"]:
                feedback_lines.append("failing_tests: " + ", ".join(str(item) for item in signature["failing_tests"]))
            if signature["exception_classes"]:
                feedback_lines.append(
                    "exception_classes: " + ", ".join(str(item) for item in signature["exception_classes"])
                )
            if signature["traceback_source_files"]:
                feedback_lines.append(
                    "traceback_source_files: " + ", ".join(str(item) for item in signature["traceback_source_files"])
                )
        feedback_lines.append(
            "Hard recovery rule: keep the patch source-only and make the next edit explain how it addresses the validation evidence above."
        )
        feedback_lines.append(
            "Hard recovery rule: if the same exception class or traceback source path remains after a patch, treat the semantic repair hypothesis as wrong and revise the mechanism instead of widening or bypassing validation."
        )
        return "\n".join(feedback_lines)

    def _attach_pending_verifier_failure_feedback(self, *, resume_from: str, recovery_context: str) -> str:
        feedback = self._pending_verifier_failure_feedback.strip()
        if not feedback:
            return recovery_context
        if str(resume_from or "").lower() not in {"locator", "patcher"}:
            return recovery_context
        self._pending_verifier_failure_feedback = ""
        base = str(recovery_context or "").strip()
        if not base:
            return feedback
        return f"{base}\n\n{feedback}"

    def _usage_snapshot(self) -> dict[str, float]:
        snapshots: list[dict[str, float]] = []
        for component in (self.locator, self.patcher, self.verifier):
            for attr in ("model", "planner_model", "implementer_model"):
                candidate = getattr(component, attr, None)
                getter = getattr(candidate, "get_usage_snapshot", None)
                if callable(getter):
                    snapshots.append({str(k): float(v) for k, v in dict(getter()).items()})
        if not snapshots:
            return {"n_calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
        totals = {"n_calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
        seen_ids: set[int] = set()
        for component in (self.locator, self.patcher, self.verifier):
            for attr in ("model", "planner_model", "implementer_model"):
                candidate = getattr(component, attr, None)
                if candidate is None:
                    continue
                base = getattr(candidate, "base_model", candidate)
                marker = id(base)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                getter = getattr(candidate, "get_usage_snapshot", None)
                if not callable(getter):
                    getter = getattr(base, "get_usage_snapshot", None)
                if not callable(getter):
                    continue
                snapshot = dict(getter())
                for key in totals:
                    totals[key] += float(snapshot.get(key, 0.0) or 0.0)
        return totals

    def _usage_delta(self, before: dict[str, float], after: dict[str, float]) -> float:
        return max(0.0, float(after.get("total_tokens", 0.0) or 0.0) - float(before.get("total_tokens", 0.0) or 0.0))

    def _stage_status_snapshot(self) -> dict[str, bool]:
        snapshot: dict[str, bool] = {}
        for stage in self.STAGE_ORDER:
            snapshot[stage] = bool(self.stage_outputs.get(stage, {}).get("success"))
        return snapshot

    def _progress_score(self) -> float:
        score = 0.0
        checkpoint = self.recorder.latest_checkpoint()
        if checkpoint:
            score = max(score, self.CHECKPOINT_PROGRESS.get(checkpoint.label, 0.0))
        for index, stage in enumerate(self.STAGE_ORDER, start=1):
            if self.stage_outputs.get(stage):
                score = max(score, float(index))
            if self.stage_outputs.get(stage, {}).get("success"):
                score = max(score, float(index))
        return score

    def _rollback_depth(self, checkpoint_before, action: CandidateAction) -> int:
        before_rank = self.CHECKPOINT_PROGRESS.get(getattr(checkpoint_before, "label", ""), 0.0)
        target = self.recorder.get_checkpoint(str(action.payload.get("checkpoint_id", ""))) or self.recorder.latest_checkpoint()
        target_rank = self.CHECKPOINT_PROGRESS.get(getattr(target, "label", ""), 0.0)
        return max(0, int(before_rank - target_rank))

    def _last_failed_stage(self) -> str:
        for stage in reversed(self.STAGE_ORDER):
            result = self.stage_outputs.get(stage, {})
            if result and not result.get("success"):
                return stage
        return ""

    def _last_failure_reason(self) -> str:
        stage = self._last_failed_stage()
        if not stage:
            return ""
        result = self.stage_outputs.get(stage, {})
        for key in ("error", "located_files", "verification", "patch", "plan", "status"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value[:500]
        return ""
