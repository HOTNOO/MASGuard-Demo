"""LLM-based recovery program synthesizer under grammar constraints.

The synthesizer asks an LLM to compose short recovery programs (1-3 steps)
from the five atomic operators.  It receives:

1. Structured provenance context (from ProvenanceSummarizer)
2. The program grammar (valid operators and transitions)
3. Retrieved similar Recovery Cases (from Case Memory)
4. Budget constraints

The LLM output is parsed, validated against the grammar, and returned as
a list of candidate RecoveryPrograms for counterfactual replay evaluation.

This is NOT "LLM-as-judge" — every candidate program will be executed via
real replay, producing ground-truth outcomes.  The LLM's role is to
*propose* promising programs; the replay *verifies* them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

from bcmr_swe.provenance.graph_store import ExecutionProvenanceGraph
from bcmr_swe.recovery.program_grammar import (
    format_grammar_for_prompt,
    validate_program,
)
from bcmr_swe.recovery.provenance_summarizer import ProvenanceSummarizer
from bcmr_swe.types import (
    FailedState,
    OpType,
    RecoveryBudget,
    RecoveryCase,
    RecoveryProgram,
    RecoveryStep,
)

logger = logging.getLogger(__name__)


class LLMBackendProtocol(Protocol):
    def query(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]: ...


SYSTEM_PROMPT = """\
You are a recovery program synthesizer for a multi-stage SWE agent system.

The agent runs a pipeline: locator → patcher → verifier.  It has entered a
failed state where context, shared facts, or workspace may be polluted.
Your job is to compose up to {max_candidate_programs} candidate recovery program(s).

Each program is a sequence of 1-5 atomic operators.  You decide the
sequence — there are no fixed templates.  Think about what actually went
wrong and design programs that address the root cause.

{grammar}

RULES:
- Include at least one observable state-changing operator in each program:
  ROLLBACK or REPLAY.
- A one-step ROLLBACK is valid when it restores a known healthy checkpoint;
  the benchmark will run official tests after the program.
- Don't end a program with ESCALATE (it must be followed by REPLAY).
- You may use REPLAY multiple times for iterative recovery.
- Shorter programs are preferred when equally effective.
- Prefer the minimal action that can plausibly restore correctness.  If a
  healthy patch anchor is explicitly available and the current failure looks
  like a regression from that anchor, choose ROLLBACK to that checkpoint
  instead of re-running patcher.
- REVOKE and INSPECT do not change files.  Do not pair them with
  REPLAY(verifier) as the only replay step, because that only re-checks the
  same broken workspace.  If you use REVOKE/INSPECT for repair, follow with
  REPLAY(patcher+verifier), REPLAY(locator+patcher+verifier), or REPLAY(full).
- Consider the token budget.
- Do NOT only vary rollback scope.  Cover distinct recovery intents when possible:
  - evidence re-check
  - belief cleanup / stale-fact removal
  - local rebuild from the current workspace
  - capability boost before replay
  - re-localization when the target itself may be stale

{cases_section}

Respond with valid JSON only (no markdown fences).  Schema:
{{
  "diagnosis": "<one-paragraph root-cause analysis>",
  "candidate_programs": [
    {{
      "program_id": "P1",
      "steps": [
        {{"op": "OPERATOR_NAME", "args": {{...}} }}
      ],
      "rationale": "<why this program addresses the root cause>",
      "estimated_total_cost": <int tokens>,
      "estimated_recover_prob": <float 0-1>,
      "estimated_risk": <float 0-1>
    }}
  ],
  "recommended_program": "P1"
}}
"""

USER_TEMPLATE = """\
Analyse the following failed state and compose recovery programs.

{provenance_context}

Generate up to {max_candidate_programs} candidate recovery program(s) with distinct strategies.
Think step by step: first diagnose the root cause, then design programs
that address it.  Vary your programs from conservative (cheap, safe) to
aggressive (expensive, higher chance of success).
"""


class ProgramSynthesizer:
    """Synthesise candidate recovery programs via LLM under grammar constraints."""

    def __init__(
        self,
        model: LLMBackendProtocol,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
        request_timeout: int = 120,
        cache_dir: Path | None = None,
        max_candidate_programs: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.request_timeout = request_timeout
        self.cache_dir = cache_dir
        self.max_candidate_programs = max(1, int(max_candidate_programs))
        self.summarizer = ProvenanceSummarizer()
        self.last_diagnosis: str = ""
        self.last_raw_response: str = ""
        self.last_recommended_program_id: str = ""
        self.last_raw_llm_candidate_programs: list[dict[str, Any]] = []
        self.last_validated_program_ids: list[str] = []
        self.last_backfilled_program_ids: list[str] = []
        self.last_returned_program_ids: list[str] = []
        self.total_calls: int = 0
        self.total_tokens: float = 0.0

    def synthesize(
        self,
        graph: ExecutionProvenanceGraph,
        failed_state: FailedState,
        budget: RecoveryBudget,
        *,
        similar_cases: list[RecoveryCase] | None = None,
        used_recovery_calls: int = 0,
        used_tokens: float = 0.0,
        preserve_recommended: bool = False,
    ) -> list[RecoveryProgram]:
        """Generate candidate recovery programs for a failed state.

        Returns validated programs sorted by estimated utility (descending).
        """
        self.last_diagnosis = ""
        self.last_raw_response = ""
        self.last_recommended_program_id = ""
        self.last_raw_llm_candidate_programs = []
        self.last_validated_program_ids = []
        self.last_backfilled_program_ids = []
        self.last_returned_program_ids = []

        actions_placeholder: list[Any] = []
        provenance_text = self.summarizer.summarize(
            graph, failed_state, actions_placeholder, budget,
            used_recovery_calls=used_recovery_calls,
            used_tokens=used_tokens,
        )

        cases_section = self._format_cases(similar_cases)
        grammar_text = format_grammar_for_prompt()

        system_msg = SYSTEM_PROMPT.format(
            grammar=grammar_text,
            cases_section=cases_section,
            max_candidate_programs=self.max_candidate_programs,
        )
        user_msg = USER_TEMPLATE.format(
            provenance_context=provenance_text,
            max_candidate_programs=self.max_candidate_programs,
        )

        raw_programs = self._call_llm(system_msg, user_msg)
        if raw_programs is None:
            logger.warning("LLM synthesis failed; returning default programs.")
            defaults = self._default_programs(failed_state)
            self.last_returned_program_ids = [program.program_id for program in defaults]
            return defaults

        validated = []
        for prog in raw_programs:
            ok, reason = validate_program(prog)
            if ok:
                prog.metadata.setdefault("source", "llm")
                if prog.program_id == self.last_recommended_program_id:
                    prog.metadata["llm_recommended"] = True
                validated.append(prog)
            else:
                logger.debug("Rejected program %s: %s", prog.program_id, reason)

        if not validated:
            logger.warning("No valid programs from LLM; using defaults.")
            defaults = self._default_programs(failed_state)
            self.last_returned_program_ids = [program.program_id for program in defaults]
            return defaults

        self.last_validated_program_ids = [program.program_id for program in validated]
        if len(validated) < self.max_candidate_programs:
            before_backfill = {program.program_id for program in validated}
            validated = self._backfill_with_defaults(validated, failed_state)
            self.last_backfilled_program_ids = [
                program.program_id
                for program in validated
                if program.program_id not in before_backfill
            ]

        validated.sort(
            key=lambda p: self._utility(p, budget), reverse=True
        )
        if self.max_candidate_programs == 1 and not preserve_recommended:
            guarded = self._guarded_single_program(validated, failed_state)
            if guarded is not None:
                self.last_returned_program_ids = [guarded.program_id]
                return [guarded]
        pruned = self._prune_programs(validated, failed_state)
        if preserve_recommended and self.last_recommended_program_id:
            recommended = next(
                (
                    program
                    for program in validated
                    if program.program_id == self.last_recommended_program_id
                ),
                None,
            )
            if recommended is not None:
                pruned = [recommended] + [
                    program
                    for program in pruned
                    if program.program_id != recommended.program_id
                ]
                pruned = pruned[: self.max_candidate_programs]
        self.last_returned_program_ids = [program.program_id for program in pruned]
        return pruned

    def get_synthesis_record(self) -> dict[str, Any]:
        """Full synthesis metadata for dataset persistence."""
        return {
            "diagnosis": self.last_diagnosis,
            "raw_response_excerpt": self.last_raw_response[:2000],
            "recommended_program_id": self.last_recommended_program_id,
            "raw_llm_candidate_programs": list(self.last_raw_llm_candidate_programs),
            "validated_program_ids": list(self.last_validated_program_ids),
            "backfilled_program_ids": list(self.last_backfilled_program_ids),
            "returned_program_ids": list(self.last_returned_program_ids),
            "synthesizer_model": getattr(
                getattr(self.model, "config", None), "model", "unknown"
            ),
        }

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self, system_msg: str, user_msg: str
    ) -> list[RecoveryProgram] | None:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            result = self.model.query(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                request_timeout=self.request_timeout,
            )
            raw = result.get("content", "")
            self.last_raw_response = raw
            self.total_calls += 1
            usage = result.get("extra", {}).get("usage", {})
            self.total_tokens += float(
                usage.get("total_tokens", 0)
                or usage.get("totalTokenCount", 0)
                or 0
            )
            return self._parse_response(raw)
        except Exception:
            logger.exception("LLM program synthesis call failed")
            return None

    def _parse_response(self, raw: str) -> list[RecoveryProgram] | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        self.last_diagnosis = str(data.get("diagnosis", ""))
        self.last_recommended_program_id = str(data.get("recommended_program", "")).strip()
        raw_programs = data.get("candidate_programs", [])
        if not isinstance(raw_programs, list):
            return None

        programs: list[RecoveryProgram] = []
        for entry in raw_programs[:8]:
            if not isinstance(entry, dict):
                continue
            try:
                steps = self._parse_steps(entry.get("steps", []))
                if not steps:
                    continue
                programs.append(RecoveryProgram(
                    program_id=str(entry.get("program_id", f"P{len(programs)+1}")),
                    steps=steps,
                    rationale=str(entry.get("rationale", "")),
                    estimated_total_cost=max(0.0, float(entry.get("estimated_total_cost", 1000))),
                    estimated_recover_prob=_clamp(float(entry.get("estimated_recover_prob", 0.5)), 0.0, 1.0),
                    estimated_risk=_clamp(float(entry.get("estimated_risk", 0.1)), 0.0, 1.0),
                ))
            except Exception:
                continue
        self.last_raw_llm_candidate_programs = [program.to_dict() for program in programs]
        return programs if programs else None

    def _parse_steps(self, raw_steps: list) -> list[RecoveryStep]:
        steps: list[RecoveryStep] = []
        for entry in raw_steps:
            if not isinstance(entry, dict):
                continue
            op_str = str(entry.get("op", "")).upper()
            try:
                op = OpType(op_str)
            except ValueError:
                continue
            steps.append(RecoveryStep(op=op, args=dict(entry.get("args", {}))))
        return steps

    # ------------------------------------------------------------------
    # Cases formatting
    # ------------------------------------------------------------------

    def _format_cases(self, cases: list[RecoveryCase] | None) -> str:
        if not cases:
            return ""
        lines = ["## Similar Recovery Cases from Memory", ""]
        for i, case in enumerate(cases[:3], 1):
            sig = case.failure_signature
            outcome = case.outcome
            skel = case.program.skeleton
            lines.append(f"Case {i}: trigger={sig.trigger_type}, stage={sig.failed_stage}, shape={sig.region_shape}")
            lines.append(f"  Program: {skel}")
            lines.append(f"  Outcome: {'SUCCESS' if outcome.recover_success else 'FAILED'}, cost={int(outcome.token_cost)} tokens")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Default programs (when LLM fails)
    # ------------------------------------------------------------------

    def _default_programs(self, fs: FailedState) -> list[RecoveryProgram]:
        """Deterministic fallback programs based on trigger type."""
        trigger = fs.trigger.trigger_type.value
        region = fs.suspect_region.summary
        anchor = region.get("replay_anchor_role", "patcher")
        scope = f"{anchor}+verifier" if anchor != "verifier" else "verifier"
        needs_relocalize = anchor in {"locator", "patcher"} or trigger == "fact_conflict"
        post_patch_id = self._checkpoint_id_for_label(fs, "post_patch", fs.checkpoint_id)
        post_locate_id = self._checkpoint_id_for_label(fs, "post_locate", fs.checkpoint_id)
        initial_id = self._checkpoint_id_for_label(fs, "initial", fs.checkpoint_id)

        programs = [
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": post_patch_id}),
                ],
                rationale="Restore the patch anchor and verify",
                estimated_total_cost=80.0,
                estimated_recover_prob=0.95,
                estimated_risk=0.01,
                metadata={"family": "local_minimal"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": post_patch_id}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": "verifier", "context_hint": "Restore the latest patch anchor and verify it before doing more work."}),
                ],
                rationale="Restore the patch anchor and verify",
                estimated_total_cost=900.0,
                estimated_recover_prob=0.52,
                estimated_risk=0.08,
                metadata={"family": "local_minimal_verify"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[RecoveryStep(op=OpType.REPLAY, args={"scope": scope, "context_hint": f"Previous attempt failed due to {trigger}"})],
                rationale=f"Direct replay from {anchor}",
                estimated_total_cost=2000.0,
                estimated_recover_prob=0.4,
                estimated_risk=0.15,
                metadata={"family": "direct_replay"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": post_locate_id}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": "patcher+verifier", "context_hint": "Return to the localization anchor and rebuild patch plus verification."}),
                ],
                rationale="Rollback to localization anchor, then replay patch and verifier",
                estimated_total_cost=2400.0,
                estimated_recover_prob=0.50,
                estimated_risk=0.11,
                metadata={"family": "local_broader"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.INSPECT, args={"target": "test_output", "depth": "deep"}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": scope, "context_hint": f"Re-check the failing evidence, then replay from {anchor}"}),
                ],
                rationale="Evidence re-check then replay",
                estimated_total_cost=2500.0,
                estimated_recover_prob=0.45,
                estimated_risk=0.12,
                metadata={"family": "evidence_recheck"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.REVOKE, args={"fact_id": "fact:latest_patch"}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": scope, "context_hint": "The promoted patch fact may be stale; remove it and derive a fresh local repair."}),
                ],
                rationale="Belief cleanup then replay",
                estimated_total_cost=2200.0,
                estimated_recover_prob=0.5,
                estimated_risk=0.10,
                metadata={"family": "belief_cleanup"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.ESCALATE, args={"scope": "patcher", "strategy": "stronger_prompt", "escalation_level": 1}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": "patcher+verifier", "context_hint": "Retry local repair with stronger patching capability."}),
                ],
                rationale="Capability boost before local replay",
                estimated_total_cost=2800.0,
                estimated_recover_prob=0.48,
                estimated_risk=0.14,
                metadata={"family": "capability_boost"},
            ),
            RecoveryProgram(
                program_id=f"default_{uuid.uuid4().hex[:6]}",
                steps=[
                    RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": initial_id}),
                    RecoveryStep(op=OpType.REPLAY, args={"scope": "full", "context_hint": f"After rollback, re-execute from scratch"}),
                ],
                rationale="Rollback and full replay",
                estimated_total_cost=4000.0,
                estimated_recover_prob=0.55,
                estimated_risk=0.20,
                metadata={"family": "global"},
            ),
        ]
        if needs_relocalize:
            programs.append(
                RecoveryProgram(
                    program_id=f"default_{uuid.uuid4().hex[:6]}",
                    steps=[
                        RecoveryStep(op=OpType.REVOKE, args={"fact_id": "fact:localized_path"}),
                        RecoveryStep(op=OpType.ESCALATE, args={"scope": "locator", "strategy": "broader_search", "escalation_level": 1}),
                        RecoveryStep(op=OpType.REPLAY, args={"scope": "locator+patcher+verifier", "context_hint": "The localized target may be stale; re-localize and rebuild."}),
                    ],
                    rationale="Re-localize then rebuild",
                    estimated_total_cost=3200.0,
                    estimated_recover_prob=0.44,
                    estimated_risk=0.16,
                    metadata={"family": "relocalize"},
                )
            )
        return self._prune_programs(programs, fs)

    def _backfill_with_defaults(
        self,
        programs: list[RecoveryProgram],
        fs: FailedState,
    ) -> list[RecoveryProgram]:
        """Add deterministic atomic-action fallbacks when the LLM under-generates.

        The LLM remains the primary composer.  The fallback only ensures the
        counterfactual replay pool is not too small to diagnose the action
        language.
        """
        merged = list(programs)
        seen: set[str] = {program.skeleton for program in merged}
        for program in self._default_programs(fs):
            if len(merged) >= self.max_candidate_programs:
                break
            if program.skeleton in seen:
                continue
            program.metadata.setdefault("source", "deterministic_backfill")
            merged.append(program)
            seen.add(program.skeleton)
        return merged

    def _checkpoint_id_for_label(
        self,
        fs: FailedState,
        label: str,
        fallback: str,
    ) -> str:
        candidates = fs.metadata.get("checkpoint_candidates", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if str(candidate.get("label", "")).strip() == label:
                    checkpoint_id = str(candidate.get("checkpoint_id", "")).strip()
                    if checkpoint_id:
                        return checkpoint_id
        return fallback

    def _healthy_checkpoint_id_for_label(
        self,
        fs: FailedState,
        label: str,
    ) -> str | None:
        candidates = fs.metadata.get("checkpoint_candidates", [])
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("label", "")).strip() != label:
                continue
            metadata = candidate.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            health = str(metadata.get("anchor_health", "")).strip().lower()
            if health in {"healthy_oracle_patch", "healthy_patch", "verified_healthy"}:
                checkpoint_id = str(candidate.get("checkpoint_id", "")).strip()
                if checkpoint_id:
                    return checkpoint_id
        return None

    def _guarded_single_program(
        self,
        programs: list[RecoveryProgram],
        fs: FailedState,
    ) -> RecoveryProgram | None:
        """Apply a minimal-recovery guard for known healthy anchors.

        This is intentionally narrow: it only fires when the failed state
        explicitly exposes a healthy post-patch checkpoint.  In that case a
        single rollback is the most controlled recovery; re-running patcher
        would add cost and variance without new evidence.
        """
        healthy_post_patch_id = self._healthy_checkpoint_id_for_label(fs, "post_patch")
        if not healthy_post_patch_id:
            return None

        return RecoveryProgram(
            program_id="guarded_local_anchor_restore",
            steps=[
                RecoveryStep(
                    op=OpType.ROLLBACK,
                    args={"checkpoint_id": healthy_post_patch_id},
                )
            ],
            rationale=(
                "Selector guard: the failed state exposes a known healthy "
                "post-patch checkpoint, so use the minimal observable recovery "
                "instead of regenerating the patch."
            ),
            estimated_total_cost=80.0,
            estimated_recover_prob=0.95,
            estimated_risk=0.01,
            metadata={
                "family": "local_minimal",
                "strategy": "guarded_rollback_post_patch_restore",
                "selection_guard": "healthy_post_patch_anchor",
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _utility(self, prog: RecoveryProgram, budget: RecoveryBudget) -> float:
        return (
            prog.estimated_recover_prob
            - budget.lambda_token * prog.estimated_total_cost
            - budget.lambda_risk * prog.estimated_risk
        )

    def _prune_programs(
        self,
        programs: list[RecoveryProgram],
        failed_state: FailedState,
    ) -> list[RecoveryProgram]:
        """Keep a small, diverse candidate set aligned with minimal recovery.

        Current runtime logic:
        - Deduplicate near-identical programs.
        - Prefer local replay scopes over ``full`` replay.
        - Allow at most one ``full`` replay fallback.
        - For verifier/patcher-side failures, require local programs to be
          considered before global restart candidates.
        """
        if not programs:
            return []

        trigger = failed_state.trigger.trigger_type.value
        anchor = str(
            failed_state.suspect_region.summary.get("replay_anchor_role", "")
        ).lower()
        prefer_local_only = trigger in {"fact_conflict", "verifier_contradiction"} or anchor in {"verifier", "patcher"}

        deduped: list[RecoveryProgram] = []
        seen: set[tuple[str, str]] = set()
        for prog in programs:
            scope = self._replay_scope(prog)
            key = (prog.skeleton, scope)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(prog)

        local_programs = [prog for prog in deduped if self._replay_scope(prog) != "full"]
        full_programs = [prog for prog in deduped if self._replay_scope(prog) == "full"]

        selected: list[RecoveryProgram] = []
        seen_families: set[str] = set()
        for prog in local_programs:
            if len(selected) >= self.max_candidate_programs:
                break
            family = self._program_family(prog)
            if family in seen_families:
                continue
            selected.append(prog)
            seen_families.add(family)

        for prog in local_programs:
            if len(selected) >= self.max_candidate_programs:
                break
            if prog in selected:
                continue
            selected.append(prog)

        if len(selected) < self.max_candidate_programs and full_programs:
            selected.append(full_programs[0])

        if not selected:
            selected = deduped[: self.max_candidate_programs]

        return selected[: self.max_candidate_programs]

    def _replay_scope(self, program: RecoveryProgram) -> str:
        for step in reversed(program.steps):
            if step.op == OpType.REPLAY:
                return str(step.args.get("scope", "")).strip().lower()
        return ""

    def _program_family(self, program: RecoveryProgram) -> str:
        explicit = str(program.metadata.get("family", "")).strip().lower()
        if explicit:
            return explicit
        ops = [step.op for step in program.steps]
        if OpType.REVOKE in ops and OpType.ESCALATE in ops:
            return "relocalize"
        if OpType.REVOKE in ops:
            return "belief_cleanup"
        if OpType.INSPECT in ops:
            return "evidence_recheck"
        if OpType.ESCALATE in ops:
            return "capability_boost"
        if OpType.ROLLBACK in ops:
            return "rollback_rebuild"
        return "direct_replay"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
