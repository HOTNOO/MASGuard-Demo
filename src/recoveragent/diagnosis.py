"""Deterministic failure diagnosis for the MASGuard demo."""

from __future__ import annotations

from recoveragent.models import Diagnosis, EvidenceBundle


def diagnose(bundle: EvidenceBundle) -> Diagnosis:
    """Classify a failed LLM repair-agent trajectory using extracted evidence."""

    signals = bundle.signals
    evidence = _evidence_items(bundle)

    if signals.get("has_collection_or_target_error"):
        return Diagnosis(
            failure_type="validation_target_failure",
            responsible_stage="validation",
            confidence="high",
            rationale="The agent appears to have run an invalid or uncollectable test target.",
            evidence=evidence,
        )
    if signals.get("patcher_no_effective_diff"):
        return Diagnosis(
            failure_type="no_effective_patch",
            responsible_stage="patch",
            confidence="high",
            rationale="The MAS reached the patching stage but did not produce an effective repository diff.",
            evidence=evidence,
        )
    if signals.get("patch_touches_source") and signals.get("has_test_failure"):
        return Diagnosis(
            failure_type="patch_semantic_failure",
            responsible_stage="patch",
            confidence="medium",
            rationale="The agent patched source code, but targeted validation still fails, indicating an incomplete or semantically wrong patch.",
            evidence=evidence,
        )
    if signals.get("stack_trace_not_touched") and signals.get("has_patch"):
        return Diagnosis(
            failure_type="fault_localization_failure",
            responsible_stage="localization_or_patch",
            confidence="medium",
            rationale="The stack trace names files that were not touched by the patch.",
            evidence=evidence,
        )
    if signals.get("repeated_tool_failure"):
        return Diagnosis(
            failure_type="context_drift_or_repeated_mistake",
            responsible_stage="agent_control",
            confidence="medium",
            rationale="The trajectory repeats a failing command or tool action.",
            evidence=evidence,
        )
    if signals.get("has_environment_error"):
        return Diagnosis(
            failure_type="environment_or_tool_failure",
            responsible_stage="validation",
            confidence="high",
            rationale="The test/build log contains dependency, timeout, or tool-environment errors.",
            evidence=evidence,
        )
    if signals.get("patch_touches_tests") and not signals.get("patch_touches_source"):
        return Diagnosis(
            failure_type="patch_generation_failure",
            responsible_stage="patch",
            confidence="medium",
            rationale="The patch changes tests but does not modify source files implicated by the failure.",
            evidence=evidence,
        )
    if signals.get("has_test_failure") and signals.get("has_patch"):
        return Diagnosis(
            failure_type="patch_semantic_failure",
            responsible_stage="patch",
            confidence="low",
            rationale="The patch was produced but tests still fail; more targeted evidence is needed before mutation.",
            evidence=evidence,
        )
    return Diagnosis(
        failure_type="ambiguous_failure",
        responsible_stage="unknown",
        confidence="low",
        rationale="The available evidence is insufficient for a precise diagnosis.",
        evidence=evidence,
    )


def _evidence_items(bundle: EvidenceBundle) -> list[str]:
    items: list[str] = []
    if bundle.failure_lines:
        items.append(f"log failure excerpt: {bundle.failure_lines[0]}")
    if bundle.stack_trace_files:
        items.append(f"stack trace files: {', '.join(bundle.stack_trace_files[:5])}")
    if bundle.touched_files:
        items.append(f"patch touched files: {', '.join(bundle.touched_files[:5])}")
    if bundle.test_targets:
        items.append(f"test targets: {', '.join(bundle.test_targets[:5])}")
    if bundle.tool_calls:
        items.append(f"tool calls observed: {len(bundle.tool_calls)}")
    return items
