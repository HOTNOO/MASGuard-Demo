"""PROBE-like diagnosis-to-guidance gate for strong structured baselines.

This module deliberately does not use BCMR-CAR's typed action space, ledgers,
episode priors, or patch contracts.  It implements the baseline idea we need
for comparison: turn failure-anchored evidence into bounded, actionable guidance
only when the diagnosis is concrete enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


SCHEMA_VERSION = "bcmr.probe_like_guidance_gate.v1"

_INFRA_TERMS = (
    "timeout",
    "docker",
    "container",
    "kind",
    "helm",
    "oom",
    "out of memory",
    "provider",
    "upstream_error",
)

_OPERATIONAL_TERMS = (
    "connection refused",
    "connection reset",
    "kubernetes",
    "kubectl",
    "targetport",
    "service",
    "endpoint",
    "permission_denied",
    "permission denied",
    "missing required args",
)


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    """Failure-anchored diagnosis separated from next-run guidance."""

    primary_cause: str = ""
    failure_anchor: str = ""
    behavioral_mistake: str = ""
    scope: str = "behavioral"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    contributing_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_cause": self.primary_cause,
            "failure_anchor": self.failure_anchor,
            "behavioral_mistake": self.behavioral_mistake,
            "scope": self.scope,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
            "contributing_factors": list(self.contributing_factors),
        }


@dataclass(frozen=True, slots=True)
class GuidanceGateResult:
    """Actionability decision and bounded retry guidance."""

    injectable: bool
    mode: str
    score: float
    reasons: list[str]
    recommended_hints: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    strategy_template: str = ""
    next_step_template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "injectable": bool(self.injectable),
            "mode": self.mode,
            "score": round(float(self.score), 3),
            "reasons": list(self.reasons),
            "recommended_hints": list(self.recommended_hints),
            "guardrails": list(self.guardrails),
            "strategy_template": self.strategy_template,
            "next_step_template": self.next_step_template,
        }


@dataclass(frozen=True, slots=True)
class ProbeLikeGuidancePacket:
    """Complete baseline artifact rendered into the recovery prompt."""

    diagnosis: FailureDiagnosis
    gate: GuidanceGateResult
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "diagnosis": self.diagnosis.to_dict(),
            "guidance_gate": self.gate.to_dict(),
            "context": dict(self.context),
            "method_boundary": {
                "uses_bcmr_car_actions": False,
                "uses_belief_ledger": False,
                "uses_patch_contract": False,
                "role": "strong_structured_recovery_baseline",
            },
        }

    def to_prompt_text(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "diagnosis": self.diagnosis.to_dict(),
            "guidance_gate": self.gate.to_dict(),
            "method_boundary": self.to_dict()["method_boundary"],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def build_probe_like_guidance_packet(context: dict[str, Any]) -> ProbeLikeGuidancePacket:
    """Build a deterministic PROBE-like diagnosis/guidance packet.

    The output is intentionally conservative.  It can say "do not inject" for
    infra-dominant or vague failures, preserving the separation between knowing
    something went wrong and telling the agent what to do next.
    """

    compact_context = _compact_context(context)
    diagnosis = diagnose_failure_anchor(compact_context)
    gate = assess_guidance_actionability(diagnosis, compact_context)
    return ProbeLikeGuidancePacket(
        diagnosis=diagnosis,
        gate=gate,
        context=compact_context,
    )


def diagnose_failure_anchor(context: dict[str, Any]) -> FailureDiagnosis:
    failure = _failure_observation(context)
    core = _core_state(context)
    evidence_pack = dict(context.get("evidence_pack", {}) or {})
    report_meta = dict(context.get("probe_report_meta", {}) or {})
    role_view = dict(context.get("role_aggregate_view", {}) or {})
    object_chain = list(context.get("object_chain_view", []) or [])

    verification = _first_text(
        failure.get("verification_excerpt"),
        failure.get("verifier_output_excerpt"),
        evidence_pack.get("verifier_excerpt"),
        evidence_pack.get("review_verifier_excerpt"),
    )
    failing_tests = list(
        failure.get("failing_tests", [])
        or evidence_pack.get("failing_tests", [])
        or []
    )
    selected_targets = list(
        core.get("selected_target_candidates", [])
        or evidence_pack.get("selected_target_candidates", [])
        or []
    )
    changed_files = list(evidence_pack.get("patcher_modified_files", []) or [])
    failed_files = list(evidence_pack.get("verifier_failed_files", []) or [])
    exception_type = str(evidence_pack.get("verifier_exception_type", "") or "")
    patch_legitimacy = str(core.get("patch_legitimacy") or evidence_pack.get("patch_legitimacy") or "")
    target_legitimacy = str(core.get("target_legitimacy") or evidence_pack.get("target_legitimacy") or "")
    stop_reason = str(evidence_pack.get("stop_reason", "") or core.get("trigger_reason", "") or "")
    negative_constraints = list(core.get("negative_constraints", []) or evidence_pack.get("negative_constraints", []) or [])
    guidance_paths = _paths_from_texts(negative_constraints)
    if guidance_paths:
        selected_targets = _dedupe(guidance_paths + [str(item) for item in selected_targets])
    probe_guidance_mode = str(report_meta.get("probe_guidance_mode", "") or "")
    resolved_outcome = _has_resolved_outcome(report_meta)
    rca_available = bool(report_meta.get("rca_available", True))
    probe_guidance_reasons = [
        str(item)
        for item in list(report_meta.get("probe_guidance_reasons", []) or [])
        if str(item).strip()
    ]

    evidence: list[str] = []
    if failing_tests:
        evidence.append(f"failing_tests={_join_limited(failing_tests, 4)}")
    if exception_type:
        evidence.append(f"exception_type={exception_type}")
    if failed_files:
        evidence.append(f"verifier_failed_files={_join_limited(failed_files, 4)}")
    if changed_files:
        evidence.append(f"patcher_modified_files={_join_limited(changed_files, 4)}")
    if verification:
        evidence.append(f"verification_excerpt={verification[:500]}")

    scope = "behavioral"
    cause = ""
    anchor = ""
    mistake = "The failed run did not turn the strongest failure evidence into a bounded verify-after-edit plan."
    confidence = 0.55
    factors: list[str] = []

    haystack = " ".join(
        [verification, stop_reason, patch_legitimacy, target_legitimacy, " ".join(str(item) for item in negative_constraints)]
    ).lower()
    if resolved_outcome and not rca_available and probe_guidance_mode == "skip":
        scope = "resolved"
        cause = "Report outcome is already resolved"
        anchor = verification[:240] or "resolved outcome evidence"
        mistake = "No next-run recovery guidance is needed after a resolved outcome."
        confidence = 0.8
    elif not rca_available and not negative_constraints:
        scope = "unknown"
        cause = "No RCA result available"
        anchor = ""
        mistake = "The report does not contain a concrete diagnosis safe enough for bounded guidance."
        confidence = 0.25
        if probe_guidance_reasons:
            factors.append(f"probe_guidance_reasons={_join_limited(probe_guidance_reasons, 2)}")
    elif _is_substrate_or_provider_noise(haystack) and not _has_actionable_guidance(negative_constraints):
        scope = "infra"
        cause = "Infrastructure or runtime-dominant failure evidence"
        anchor = verification[:240] or stop_reason[:240] or cause
        mistake = "The failure evidence is not yet safe to convert into source-edit guidance."
        confidence = 0.7
    elif _is_operational_recovery(haystack) and _has_actionable_guidance(negative_constraints):
        cause = "Operational failure with bounded recovery guidance"
        anchor = verification[:240] or _join_limited(negative_constraints, 1)
        mistake = "The next attempt should follow the bounded operational check/fix loop before broad exploration."
        confidence = 0.72
    elif patch_legitimacy in {"no_effective_patch", "no_diff"} or target_legitimacy == "no_diff":
        cause = "Previous attempt produced no effective source change"
        anchor = patch_legitimacy or target_legitimacy
        mistake = "The next attempt must first find a concrete source target instead of repeating a no-diff route."
        confidence = 0.75
    elif failed_files and changed_files and not _path_overlap(failed_files, changed_files):
        cause = "Touched files do not overlap the verifier failure boundary"
        anchor = f"failed={_join_limited(failed_files, 3)}; changed={_join_limited(changed_files, 3)}"
        mistake = "The previous repair likely followed the wrong source boundary."
        confidence = 0.72
    elif selected_targets:
        cause = "Failure anchored on selected source targets and verifier evidence"
        anchor = _join_limited(selected_targets, 4)
        confidence = 0.62
    elif exception_type:
        cause = f"Verifier failed with {exception_type}"
        anchor = verification[:240] or exception_type
        confidence = 0.62
    elif failing_tests:
        cause = "External verifier reported unresolved failing tests"
        anchor = _join_limited(failing_tests, 4)
        confidence = 0.58
    elif _has_actionable_guidance(negative_constraints):
        cause = "Available structured guidance contains bounded next-step evidence"
        anchor = _join_limited(negative_constraints, 1)
        mistake = "The next attempt should execute the bounded evaluator/tool clue and verify the result."
        confidence = 0.58
    elif role_view:
        cause = "Stage trace indicates a failed or incomplete repair attempt"
        anchor = _role_anchor(role_view)
        confidence = 0.5
    else:
        scope = "unknown"
        cause = "No concrete failure anchor available"
        anchor = ""
        mistake = "The available context is too sparse for bounded guidance."
        confidence = 0.2

    if object_chain:
        factors.append(f"typed_object_count={len(object_chain)}")
    if negative_constraints:
        factors.append(f"negative_constraints={_join_limited(negative_constraints, 3)}")
    if target_legitimacy:
        factors.append(f"target_legitimacy={target_legitimacy}")
    if patch_legitimacy:
        factors.append(f"patch_legitimacy={patch_legitimacy}")

    return FailureDiagnosis(
        primary_cause=cause,
        failure_anchor=anchor,
        behavioral_mistake=mistake,
        scope=scope,
        confidence=confidence,
        evidence=evidence[:6],
        contributing_factors=factors[:6],
    )


def assess_guidance_actionability(
    diagnosis: FailureDiagnosis,
    context: dict[str, Any],
) -> GuidanceGateResult:
    reasons: list[str] = []
    hints: list[str] = []

    if diagnosis.scope == "infra":
        return GuidanceGateResult(
            injectable=False,
            mode="infra",
            score=0.2,
            reasons=["Diagnosis is infrastructure/runtime-dominant; source-edit guidance would be unsafe."],
            recommended_hints=[],
            guardrails=_generic_guardrails(),
        )
    if diagnosis.scope == "resolved":
        return GuidanceGateResult(
            injectable=False,
            mode="skip",
            score=0.0,
            reasons=["The report outcome is already resolved; next-run guidance is not needed."],
            recommended_hints=[],
            guardrails=_generic_guardrails(),
        )
    if diagnosis.scope == "unknown" or not diagnosis.failure_anchor:
        return GuidanceGateResult(
            injectable=False,
            mode="skip",
            score=0.0,
            reasons=["No concrete failure anchor was available for bounded next-run guidance."],
            recommended_hints=[],
            guardrails=_generic_guardrails(),
        )

    core = _core_state(context)
    evidence_pack = dict(context.get("evidence_pack", {}) or {})
    failed_files = list(evidence_pack.get("verifier_failed_files", []) or [])
    changed_files = list(evidence_pack.get("patcher_modified_files", []) or [])
    negative_constraints = list(core.get("negative_constraints", []) or evidence_pack.get("negative_constraints", []) or [])
    selected_targets = list(
        core.get("selected_target_candidates", [])
        or evidence_pack.get("selected_target_candidates", [])
        or []
    )
    guidance_paths = _paths_from_texts(negative_constraints)
    if guidance_paths:
        selected_targets = _dedupe(guidance_paths + [str(item) for item in selected_targets])
    failing_tests = list(_failure_observation(context).get("failing_tests", []) or evidence_pack.get("failing_tests", []) or [])

    hints.append(f"Start from this failure anchor: {diagnosis.failure_anchor[:260]}.")
    if failed_files and changed_files and not _path_overlap(failed_files, changed_files):
        hints.append("Re-check localization against the verifier failing files before editing new code.")
    elif selected_targets:
        hints.append(f"Inspect the smallest selected target first: {_join_limited(selected_targets, 2)}.")
    if failing_tests:
        hints.append(f"Run focused validation for: {_join_limited(failing_tests, 2)}.")
    if negative_constraints:
        hints.append(f"Follow this bounded recovery clue before broad exploration: {_join_limited(negative_constraints, 1)}.")
    hints.append("Make the smallest task-relevant repair that directly addresses the anchored failure.")

    score = 0.3
    if diagnosis.primary_cause:
        score += 0.25
    if diagnosis.evidence:
        score += 0.2
    if hints:
        score += 0.25
    score = min(1.0, score)
    reasons.append("Diagnosis has a concrete failure anchor and bounded next-run hints.")

    guardrails = _generic_guardrails()
    strategy = _strategy_template(
        diagnosis=diagnosis,
        hints=hints[:4],
        guardrails=guardrails,
    )
    return GuidanceGateResult(
        injectable=True,
        mode="behavior",
        score=score,
        reasons=reasons,
        recommended_hints=hints[:4],
        guardrails=guardrails,
        strategy_template=strategy,
        next_step_template=_next_step_template(),
    )


def render_probe_like_guidance_prompt(context: dict[str, Any]) -> str:
    packet = build_probe_like_guidance_packet(context)
    context_json = json.dumps(packet.context, ensure_ascii=False, indent=2)
    lines = [
        "You are recovering a failed repository-level software repair attempt.",
        "Use failure-anchored structured diagnosis before editing.",
        "Do not use any BCMR-CAR typed recovery action language, belief ledger, episode prior, or patch contract.",
        "",
        "PROBE-like diagnosis and guidance gate:",
        packet.to_prompt_text(),
        "",
        "Protocol:",
        "1. Use the diagnosis only as a failure anchor; do not treat it as proof.",
        "2. If guidance_gate.injectable is false, gather the smallest missing evidence before source edits.",
        "3. If guidance_gate.injectable is true, follow only the bounded hints and guardrails.",
        "4. Make a source-only repair; avoid tests, generated files, broad rewrites, and unrelated files.",
        "5. Run focused validation when possible and stop after the smallest defensible fix.",
        "",
        "Structured failure context:",
        context_json,
    ]
    return "\n".join(lines)


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    payload = dict(context or {})
    if "role_aggregate_view" in payload:
        payload["role_aggregate_view"] = {
            str(key): _truncate_nested(dict(value), 600)
            for key, value in dict(payload.get("role_aggregate_view", {}) or {}).items()
            if isinstance(value, dict)
        }
    if "evidence_pack" in payload:
        payload["evidence_pack"] = _truncate_nested(dict(payload.get("evidence_pack", {}) or {}), 1200)
    if "object_chain_view" in payload:
        payload["object_chain_view"] = [
            _truncate_nested(dict(item), 500)
            for item in list(payload.get("object_chain_view", []) or [])[:8]
            if isinstance(item, dict)
        ]
    return payload


def _truncate_nested(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_truncate_nested(item, limit) for item in value[:12]]
    if isinstance(value, dict):
        return {str(key): _truncate_nested(item, limit) for key, item in value.items()}
    return value


def _failure_observation(context: dict[str, Any]) -> dict[str, Any]:
    failure = context.get("failure_observation")
    if isinstance(failure, dict):
        return dict(failure)
    return {}


def _core_state(context: dict[str, Any]) -> dict[str, Any]:
    core = context.get("core_recovery_state")
    if isinstance(core, dict):
        return dict(core)
    nested = context.get("structured_state_core")
    if isinstance(nested, dict) and isinstance(nested.get("core_recovery_state"), dict):
        return dict(nested["core_recovery_state"])
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _join_limited(values: list[Any], limit: int) -> str:
    return ", ".join(str(item) for item in values[:limit] if str(item).strip())


def _path_overlap(left: list[Any], right: list[Any]) -> bool:
    left_paths = [str(item).strip() for item in left if str(item).strip()]
    right_paths = [str(item).strip() for item in right if str(item).strip()]
    for a in left_paths:
        for b in right_paths:
            if a == b or a.endswith(f"/{b}") or b.endswith(f"/{a}"):
                return True
    return False


def _role_anchor(role_view: dict[str, Any]) -> str:
    failed = []
    for role, payload in role_view.items():
        if isinstance(payload, dict) and not bool(payload.get("success", False)):
            failed.append(str(role))
    return "failed_or_incomplete_roles=" + (_join_limited(failed, 5) if failed else _join_limited(list(role_view), 5))


def _has_actionable_guidance(values: list[Any]) -> bool:
    text = " ".join(str(item or "") for item in values).lower()
    return any(
        marker in text
        for marker in (
            "run ",
            "call ",
            "retrieve ",
            "examine ",
            "verify",
            "check",
            "fix",
            "ensure",
            "before",
            "after",
        )
    )


def _is_substrate_or_provider_noise(text: str) -> bool:
    return any(term in text for term in _INFRA_TERMS)


def _is_operational_recovery(text: str) -> bool:
    return any(term in text for term in _OPERATIONAL_TERMS)


def _has_resolved_outcome(report_meta: dict[str, Any]) -> bool:
    for item in list(report_meta.get("outcome_evidence", []) or []):
        if not isinstance(item, dict):
            continue
        if item.get("resolved") is True:
            return True
        summary = str(item.get("summary", "") or "").lower()
        if "success=true" in summary or "resolved=true" in summary:
            return True
    return False


def _paths_from_texts(values: list[Any]) -> list[str]:
    import re

    matches: list[str] = []
    for value in values:
        matches.extend(
            item
            for item in re.findall(
                r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_./-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|yaml|yml|json|toml)",
                str(value or ""),
            )
        )
    return _dedupe(matches)


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _generic_guardrails() -> list[str]:
    return [
        "Do not edit tests, generated files, or benchmark harness files unless the issue explicitly requires it.",
        "Do not add temporary settings, bootstrap files, or environment hacks just to make verification run.",
        "Do not broaden the repair beyond the failure anchor without new evidence.",
        "After one targeted edit, rerun focused validation and report pass/fail evidence.",
    ]


def _strategy_template(
    *,
    diagnosis: FailureDiagnosis,
    hints: list[str],
    guardrails: list[str],
) -> str:
    parts = [
        "### Required Plan For This Iteration",
        f"[Failure Anchor]: {diagnosis.failure_anchor}",
        f"[Primary Cause]: {diagnosis.primary_cause}",
        "[Bounded Guidance]:",
    ]
    parts.extend(f"- {hint}" for hint in hints)
    parts.append("[Verification Guardrails]:")
    parts.extend(f"- {item}" for item in guardrails)
    return "\n".join(parts)


def _next_step_template() -> str:
    return "\n".join(
        [
            "OBSERVATION:",
            "{{observation}}",
            "",
            "## Plan Check",
            "If the latest observation weakens the failure anchor, restate the updated hypothesis and run one concrete verification step before editing.",
        ]
    )
