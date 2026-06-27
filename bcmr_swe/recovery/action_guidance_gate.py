"""BCMR-CAR action guidance gate.

The PROBE-like baseline has a diagnosis-to-guidance gate.  BCMR-CAR needs a
separate method-internal gate: the CAR controller has already selected a typed
recovery action, and this module turns the ledger state behind that action into
auditable replay guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from bcmr_swe.types import RecoveryLedger, SemanticActionType
from swe_mas.utils.path_filters import normalize_repo_path


SCHEMA_VERSION = "bcmr.car_action_guidance_gate.v1"
PROMPT_BEGIN = "[BCMR-CAR ACTION GUIDANCE GATE]"
PROMPT_END = "[/BCMR-CAR ACTION GUIDANCE GATE]"


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_paths(values: list[Any]) -> list[str]:
    return _dedupe([normalize_repo_path(str(value)) for value in values])


def _stable_id(parts: list[Any]) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _enum_token(prefix: str, value: Any) -> str:
    raw = str(value or "unknown").strip().replace("-", "_").replace(" ", "_")
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw).strip("_").upper()
    while "__" in token:
        token = token.replace("__", "_")
    return f"<{prefix}:{token or 'UNKNOWN'}>"


def _canonical_action(action: str) -> str:
    text = str(action or "").strip()
    if text == SemanticActionType.REPAIR_LOCAL.value:
        return SemanticActionType.LOCAL_REPAIR.value
    if text == SemanticActionType.EXPAND_SCOPE.value:
        return SemanticActionType.SCOPE_EXPAND.value
    return text


def _latest_revision_type(ledger: RecoveryLedger) -> str:
    event = dict(ledger.metadata.get("latest_belief_revision_event", {}) or {})
    signal = dict(ledger.metadata.get("latest_belief_revision_signal", {}) or {})
    history = [
        dict(item)
        for item in list(ledger.metadata.get("belief_revision_history", []) or [])
        if isinstance(item, dict)
    ]
    return str(
        event.get("revision_type", "")
        or signal.get("latest_revision_type", "")
        or (history[-1].get("revision_type", "") if history else "")
        or ""
    )


def _latest_counterexample_type(ledger: RecoveryLedger) -> str:
    latest = dict(ledger.metadata.get("latest_car_counterexample", {}) or {})
    return str(latest.get("counterexample_type", "") or ledger.trigger_reason or "")


def _latest_guard(ledger: RecoveryLedger) -> dict[str, Any]:
    history = [
        dict(item)
        for item in list(ledger.metadata.get("guard_history", []) or [])
        if isinstance(item, dict)
    ]
    if history:
        return history[-1]
    return dict(ledger.last_action_result.get("semantic_guard", {}) or {})


def _best_source_candidate(ledger: RecoveryLedger) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in list(ledger.metadata.get("source_candidate_memory", []) or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}

    def _score(candidate: dict[str, Any]) -> tuple[int, int, float]:
        mode = str(candidate.get("result_mode", "") or "")
        mode_score = {
            "source_edit_pending_official": 6,
            "oracle_failed_after_source_edit": 5,
            "contract_violation_after_source_edit": 4,
            "source_edit_but_not_suspect": 2,
        }.get(mode, 0)
        return (
            mode_score,
            1 if bool(candidate.get("touches_suspect_path", False)) else 0,
            float(candidate.get("created_at", 0.0) or 0.0),
        )

    return max(candidates, key=_score)


def _candidate_paths(candidate: dict[str, Any]) -> list[str]:
    return _dedupe_paths(
        list(candidate.get("fresh_source_files", []) or [])
        + list(candidate.get("source_files", []) or [])
    )


def _evidence_summary(ledger: RecoveryLedger, latest_guard: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    if ledger.failing_tests_summary:
        evidence.append("failing_tests=" + ", ".join(ledger.failing_tests_summary[:4]))
    test_command = str(ledger.metadata.get("test_command", "") or "").strip()
    if test_command:
        evidence.append(f"focused_test={test_command}")
    if latest_guard:
        result_mode = str(latest_guard.get("result_mode", "") or "")
        if result_mode:
            evidence.append(f"last_guard_result={result_mode}")
        flags = [
            str(item)
            for item in list(latest_guard.get("guard_flags", []) or [])
            if str(item).strip()
        ]
        if flags:
            evidence.append("last_guard_flags=" + ", ".join(flags[:5]))
    if ledger.key_evidence:
        evidence.append("latest_evidence=" + " ".join(str(ledger.key_evidence[0]).split())[:360])
    return evidence[:6]


@dataclass(frozen=True)
class CARFailureDiagnosis:
    """Why the selected typed action is being replayed now."""

    selected_action: str
    counterexample_type: str = ""
    latest_revision_type: str = ""
    failure_anchor: str = ""
    evidence_summary: list[str] = field(default_factory=list)
    candidate_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "counterexample_type": self.counterexample_type,
            "latest_revision_type": self.latest_revision_type,
            "failure_anchor": self.failure_anchor,
            "evidence_summary": list(self.evidence_summary),
            "candidate_state": dict(self.candidate_state),
        }


@dataclass(frozen=True)
class CARActionGuidanceDecision:
    """Actionability decision and replay-facing directives."""

    injectable: bool
    mode: str
    score: float
    reasons: list[str] = field(default_factory=list)
    replay_directives: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    required_evidence_before_edit: list[str] = field(default_factory=list)
    required_edit_boundary: list[str] = field(default_factory=list)
    verification_target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "injectable": bool(self.injectable),
            "mode": self.mode,
            "score": round(float(self.score), 3),
            "reasons": list(self.reasons),
            "replay_directives": list(self.replay_directives),
            "guardrails": list(self.guardrails),
            "required_evidence_before_edit": list(self.required_evidence_before_edit),
            "required_edit_boundary": list(self.required_edit_boundary),
            "verification_target": self.verification_target,
        }


@dataclass(frozen=True)
class CARActionGuidancePacket:
    """Complete CAR action guidance artifact."""

    guidance_id: str
    diagnosis: CARFailureDiagnosis
    actionability: CARActionGuidanceDecision
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "guidance_id": self.guidance_id,
            "diagnosis": self.diagnosis.to_dict(),
            "actionability": self.actionability.to_dict(),
            "context": dict(self.context),
            "method_boundary": {
                "uses_bcmr_car_actions": True,
                "uses_belief_ledger": True,
                "uses_patch_contract": True,
                "role": "main_method_action_conditioning",
            },
        }

    def to_llm_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "schema_version": SCHEMA_VERSION,
            "guidance_id": self.guidance_id,
            "diagnosis": {
                "selected_action": _enum_token("ACTION", self.diagnosis.selected_action),
                "counterexample": _enum_token("CE", self.diagnosis.counterexample_type),
                "latest_revision": _enum_token("REV", self.diagnosis.latest_revision_type),
                "failure_anchor": self.diagnosis.failure_anchor[:320],
                "candidate_state": dict(self.diagnosis.candidate_state),
            },
            "actionability": {
                "injectable": bool(self.actionability.injectable),
                "mode": _enum_token("MODE", self.actionability.mode),
                "score": round(float(self.actionability.score), 3),
                "reasons": [_enum_token("WHY", item) for item in self.actionability.reasons[:8]],
                "replay_directives": [
                    _enum_token("DIR", item)
                    for item in self.actionability.replay_directives[:10]
                ],
                "guardrails": [
                    _enum_token("GUARD", item)
                    for item in self.actionability.guardrails[:10]
                ],
                "required_evidence_before_edit": list(
                    self.actionability.required_evidence_before_edit[:6]
                ),
                "required_edit_boundary": list(self.actionability.required_edit_boundary[:8]),
                "verification_target": self.actionability.verification_target[:240],
            },
            "method_boundary": payload["method_boundary"],
        }

    def to_prompt_text(self) -> str:
        compact_json = json.dumps(self.to_llm_payload(), ensure_ascii=False, separators=(",", ":"))
        return f"{PROMPT_BEGIN}\n{compact_json}\n{PROMPT_END}"


def build_car_action_guidance_packet(
    ledger: RecoveryLedger,
    *,
    selected_action: str,
    scope: str,
    repair_mode: str,
    execution_profile: str = "normal",
) -> CARActionGuidancePacket:
    """Build deterministic CAR action guidance from the current ledger."""

    action = _canonical_action(selected_action or ledger.last_action or "")
    latest_guard = _latest_guard(ledger)
    candidate = _best_source_candidate(ledger)
    candidate_source_paths = _candidate_paths(candidate)
    suspect_paths = _dedupe_paths(list(ledger.suspect_paths or []))
    active_target = normalize_repo_path(str(ledger.active_target or ""))
    target_paths = _dedupe_paths(([active_target] if active_target else []) + suspect_paths)
    if candidate_source_paths and (
        str(repair_mode or "").startswith("candidate_preserving")
        or str(candidate.get("result_mode", "") or "") == "oracle_failed_after_source_edit"
    ):
        target_paths = _dedupe_paths(candidate_source_paths + target_paths)
    counterexample = _latest_counterexample_type(ledger)
    latest_revision = _latest_revision_type(ledger)
    replay_precondition = str(ledger.metadata.get("current_replay_precondition", "") or "").strip()
    test_command = str(ledger.metadata.get("test_command", "") or "").strip()
    failure_anchor = _failure_anchor(
        ledger=ledger,
        latest_guard=latest_guard,
        target_paths=target_paths,
        candidate_source_paths=candidate_source_paths,
    )
    diagnosis = CARFailureDiagnosis(
        selected_action=action,
        counterexample_type=counterexample,
        latest_revision_type=latest_revision,
        failure_anchor=failure_anchor,
        evidence_summary=_evidence_summary(ledger, latest_guard),
        candidate_state={
            "has_candidate": bool(candidate_source_paths),
            "candidate_result_mode": str(candidate.get("result_mode", "") or ""),
            "candidate_source_paths": candidate_source_paths[:6],
            "touches_suspect_path": bool(candidate.get("touches_suspect_path", False)),
        },
    )
    actionability = _actionability_decision(
        ledger=ledger,
        selected_action=action,
        scope=scope,
        repair_mode=repair_mode,
        execution_profile=execution_profile,
        replay_precondition=replay_precondition,
        target_paths=target_paths,
        candidate_source_paths=candidate_source_paths,
        test_command=test_command,
    )
    guidance_id = _stable_id(
        [
            action,
            counterexample,
            latest_revision,
            scope,
            repair_mode,
            replay_precondition,
            ",".join(target_paths[:4]),
            len(list(ledger.tried_actions or [])),
        ]
    )
    return CARActionGuidancePacket(
        guidance_id=guidance_id,
        diagnosis=diagnosis,
        actionability=actionability,
        context={
            "replay_scope": str(scope or ""),
            "repair_mode": str(repair_mode or ""),
            "execution_profile": str(execution_profile or "normal"),
            "replay_precondition": replay_precondition,
            "active_object_type": str(ledger.active_object_type or ""),
            "active_object_id": str(ledger.active_object_id or ""),
            "negative_constraints": [
                " ".join(str(item or "").split())[:180]
                for item in list(ledger.negative_constraints or [])[:6]
                if str(item).strip()
            ],
        },
    )


def _failure_anchor(
    *,
    ledger: RecoveryLedger,
    latest_guard: dict[str, Any],
    target_paths: list[str],
    candidate_source_paths: list[str],
) -> str:
    if candidate_source_paths:
        return "source_candidate=" + ", ".join(candidate_source_paths[:4])
    if target_paths:
        return "target_boundary=" + ", ".join(target_paths[:4])
    if latest_guard:
        result_mode = str(latest_guard.get("result_mode", "") or "")
        if result_mode:
            return f"last_guard_result={result_mode}"
    if ledger.failing_tests_summary:
        return "failing_tests=" + ", ".join(ledger.failing_tests_summary[:4])
    return str(ledger.trigger_reason or "")


def _actionability_decision(
    *,
    ledger: RecoveryLedger,
    selected_action: str,
    scope: str,
    repair_mode: str,
    execution_profile: str,
    replay_precondition: str,
    target_paths: list[str],
    candidate_source_paths: list[str],
    test_command: str,
) -> CARActionGuidanceDecision:
    reasons: list[str] = []
    directives: list[str] = []
    guardrails: list[str] = [
        "do_not_convert_diagnosis_into_new_action",
        "do_not_edit_tests_or_generated_outputs",
        "keep_recovery_patch_small",
    ]
    required_evidence: list[str] = []
    required_boundary = list(target_paths)
    mode = "action_guidance"
    score = 0.55
    injectable = True

    if selected_action in {
        SemanticActionType.LOCAL_REPAIR.value,
        SemanticActionType.SCOPE_EXPAND.value,
    }:
        reasons.append("selected_action_requires_replay")
        directives.extend(["produce_fresh_source_diff", "run_focused_validation"])
        score += 0.15
    else:
        mode = "evidence_only"
        reasons.append("selected_action_is_not_replay_action")
        directives.append("preserve_action_diagnosis_for_followup_replay")
        score = 0.45
        injectable = bool(replay_precondition or ledger.key_evidence or ledger.negative_constraints)

    if selected_action == SemanticActionType.LOCAL_REPAIR.value:
        directives.extend(["edit_intended_source_boundary", "avoid_broad_relocalization"])
        if required_boundary:
            guardrails.append("retarget_only_with_explicit_contradictory_evidence")
            score += 0.1
        else:
            required_evidence.append("identify_source_boundary_before_edit")
            reasons.append("missing_target_boundary")
            score -= 0.15
    elif selected_action == SemanticActionType.SCOPE_EXPAND.value:
        directives.extend(["refresh_localization_before_patch", "justify_new_target_with_failure_evidence"])
        guardrails.append("expand_to_smallest_adjacent_source_boundary")
        if not required_boundary:
            required_evidence.append("use_latest_failure_to_select_smallest_source_boundary")

    if candidate_source_paths:
        directives.append("preserve_and_refine_source_candidate")
        guardrails.append("do_not_discard_candidate_without_verifier_evidence")
        required_boundary = _dedupe_paths(candidate_source_paths + required_boundary)
        reasons.append("source_candidate_available")
        score += 0.1

    if replay_precondition == "post_evidence_source_repair":
        directives.extend(["post_evidence_source_repair", "do_not_spend_replay_on_readonly_diagnosis"])
        reasons.append("evidence_recheck_already_completed")
        score += 0.1
    elif replay_precondition == "source_candidate_refine":
        directives.extend(["candidate_preserving_refine", "repair_latest_oracle_failure"])
        reasons.append("candidate_refinement_after_oracle_failure")
        score += 0.1
    elif replay_precondition == "evidence_bounded_scope_expand":
        directives.append("evidence_bounded_scope_expand")
        reasons.append("scope_expand_requires_fresh_localization")

    if str(execution_profile or "").strip().lower() == "compact":
        directives.append("compact_replay_edit_or_validate")
        guardrails.append("avoid_repeated_readonly_probe_loop")

    if test_command:
        verification_target = test_command
    elif ledger.failing_tests_summary:
        verification_target = ", ".join(ledger.failing_tests_summary[:3])
    else:
        verification_target = "focused verifier evidence from ledger"
        required_evidence.append("focused_verification_result")

    if not injectable:
        mode = "skip"
    score = max(0.0, min(1.0, score))
    return CARActionGuidanceDecision(
        injectable=injectable,
        mode=mode,
        score=score,
        reasons=_dedupe(reasons),
        replay_directives=_dedupe(directives),
        guardrails=_dedupe(guardrails),
        required_evidence_before_edit=_dedupe(required_evidence),
        required_edit_boundary=_dedupe_paths(required_boundary)[:8],
        verification_target=verification_target,
    )


def audit_car_action_guidance_packet(
    packet: dict[str, Any],
    *,
    guard: dict[str, Any],
) -> dict[str, Any]:
    """Post-replay audit for the guidance packet's observable requirements."""

    if not packet or str(packet.get("schema_version", "") or "") != SCHEMA_VERSION:
        return {
            "schema_version": "bcmr.car_action_guidance_gate_audit.v1",
            "guidance_present": False,
            "satisfied": True,
            "flags": [],
        }

    actionability = dict(packet.get("actionability", {}) or {})
    target_paths = _dedupe_paths(list(actionability.get("required_edit_boundary", []) or []))
    directives = {
        str(item)
        for item in list(actionability.get("replay_directives", []) or [])
        if str(item).strip()
    }
    fresh_classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
    fresh_source_files = _dedupe_paths(list(fresh_classes.get("source_files", []) or []))
    focused_validation = dict(guard.get("focused_validation", {}) or {})
    flags: list[str] = []
    target_overlap = _path_overlap(fresh_source_files, target_paths)
    if "produce_fresh_source_diff" in directives and not fresh_source_files:
        flags.append("guidance_no_fresh_source_diff")
    if (
        ("edit_intended_source_boundary" in directives or "preserve_and_refine_source_candidate" in directives)
        and fresh_source_files
        and target_paths
        and not target_overlap
    ):
        flags.append("guidance_missed_required_boundary")
    if "run_focused_validation" in directives and fresh_source_files:
        if not bool(focused_validation.get("has_result", False)):
            flags.append("guidance_missing_focused_validation_result")
        elif not bool(focused_validation.get("target_related", False)):
            flags.append("guidance_validation_not_target_related")

    return {
        "schema_version": "bcmr.car_action_guidance_gate_audit.v1",
        "guidance_present": True,
        "guidance_id": str(packet.get("guidance_id", "") or ""),
        "satisfied": not flags,
        "flags": _dedupe(flags),
        "required_edit_boundary": target_paths,
        "fresh_source_files": fresh_source_files,
        "fresh_required_boundary_overlap": target_overlap,
        "directives": sorted(directives),
    }


def _path_overlap(paths: list[str], targets: list[str]) -> list[str]:
    overlap: list[str] = []
    for path in _dedupe_paths(paths):
        for target in _dedupe_paths(targets):
            if path == target or path.endswith(f"/{target}") or target.endswith(f"/{path}"):
                overlap.append(path)
                break
    return _dedupe(overlap)
