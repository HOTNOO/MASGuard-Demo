"""Counterexample-aware recovery primitives for CAR.

The classifier is deterministic and zero-token.  It turns the structured MAS
failure evidence into a typed counterexample that can drive action selection
without adding another LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bcmr_swe.types import RecoveryLedger, SemanticActionType, StructuredRecoveryState


class CounterexampleType(str, Enum):
    SYNTAX_LEVEL = "syntax_level"
    SEMANTIC_LEVEL = "semantic_level"
    INVARIANT_LEVEL = "invariant_level"
    COVERAGE_GAP = "coverage_gap"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CounterexampleClassification:
    counterexample_type: CounterexampleType
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterexample_type": self.counterexample_type.value,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


ACTION_PRIORITY: dict[CounterexampleType, list[SemanticActionType]] = {
    CounterexampleType.SYNTAX_LEVEL: [
        SemanticActionType.LOCAL_REPAIR,
        SemanticActionType.EVIDENCE_RECHECK,
        SemanticActionType.TARGET_RESET,
    ],
    CounterexampleType.SEMANTIC_LEVEL: [
        SemanticActionType.EVIDENCE_RECHECK,
        SemanticActionType.LOCAL_REPAIR,
        SemanticActionType.TARGET_RESET,
        SemanticActionType.SCOPE_EXPAND,
    ],
    CounterexampleType.INVARIANT_LEVEL: [
        SemanticActionType.REVOKE_OBJECT,
        SemanticActionType.TARGET_RESET,
        SemanticActionType.LOCAL_REPAIR,
    ],
    CounterexampleType.COVERAGE_GAP: [
        SemanticActionType.SCOPE_EXPAND,
        SemanticActionType.TARGET_RESET,
        SemanticActionType.LOCAL_REPAIR,
    ],
    CounterexampleType.UNKNOWN: [
        SemanticActionType.EVIDENCE_RECHECK,
        SemanticActionType.LOCAL_REPAIR,
    ],
}

NEVER_FIRST: dict[CounterexampleType, set[SemanticActionType]] = {
    CounterexampleType.SYNTAX_LEVEL: {
        SemanticActionType.SCOPE_EXPAND,
        SemanticActionType.CAPABILITY_BOOST,
    },
    CounterexampleType.SEMANTIC_LEVEL: {
        SemanticActionType.CAPABILITY_BOOST,
    },
    CounterexampleType.INVARIANT_LEVEL: {
        SemanticActionType.SCOPE_EXPAND,
    },
    CounterexampleType.COVERAGE_GAP: {
        SemanticActionType.EVIDENCE_RECHECK,
    },
    CounterexampleType.UNKNOWN: set(),
}


def classify_counterexample(
    state: StructuredRecoveryState,
    ledger: RecoveryLedger | None = None,
) -> CounterexampleClassification:
    ev = dict(state.evidence_pack or {})
    verifier_excerpt = str(ev.get("verifier_excerpt", "") or "")
    patch_legitimacy = str(ev.get("patch_legitimacy", "") or "").strip()
    patcher_stop_reason = str(ev.get("patcher_stop_reason", "") or ev.get("stop_reason", "") or "").strip()
    exception_type = str(ev.get("verifier_exception_type", "") or "").strip()
    failure_family = str(ev.get("failure_family_manual", "") or "").strip().lower()
    typed_object_quality = str(ev.get("typed_object_quality", "") or "").strip().lower()
    failed_files = _string_set(ev.get("verifier_failed_files", []))
    modified_files = _string_set(ev.get("patcher_modified_files", []))
    assertion_count = _safe_int(ev.get("verifier_assertion_count", 0))
    failing_tests_count = _safe_int(ev.get("failing_tests_count", 0))

    evidence = {
        "patch_legitimacy": patch_legitimacy,
        "patcher_stop_reason": patcher_stop_reason,
        "verifier_exception_type": exception_type,
        "verifier_failed_files": sorted(failed_files),
        "patcher_modified_files": sorted(modified_files),
        "verifier_assertion_count": assertion_count,
        "failing_tests_count": failing_tests_count,
        "failure_family_manual": failure_family,
        "typed_object_quality": typed_object_quality,
    }

    if exception_type == "SyntaxError":
        return CounterexampleClassification(
            CounterexampleType.SYNTAX_LEVEL,
            "verifier_reported_syntax_error",
            evidence,
        )
    if patch_legitimacy in {"patch_apply_failed", "no_effective_patch"} and patcher_stop_reason in {
        "apply_failed",
        "parse_error",
        "patch_apply_failed",
    }:
        return CounterexampleClassification(
            CounterexampleType.SYNTAX_LEVEL,
            "patch_structure_failed_before_semantic_validation",
            evidence,
        )

    if failed_files and modified_files and not (failed_files & modified_files):
        return CounterexampleClassification(
            CounterexampleType.COVERAGE_GAP,
            "verifier_failure_outside_modified_patch_scope",
            evidence,
        )

    if exception_type.lower() in {"typeerror", "attributeerror", "keyerror", "valueerror"} and "assert" not in verifier_excerpt.lower():
        return CounterexampleClassification(
            CounterexampleType.INVARIANT_LEVEL,
            "non_assertion_runtime_contract_exception",
            evidence,
        )

    if assertion_count > 0 or "assert" in verifier_excerpt.lower() or failing_tests_count > 0:
        return CounterexampleClassification(
            CounterexampleType.SEMANTIC_LEVEL,
            "assertion_or_test_failure_counterexample",
            evidence,
        )

    if "verifier object missing" in failure_family or "verifier missing" in failure_family:
        return CounterexampleClassification(
            CounterexampleType.SEMANTIC_LEVEL,
            "reviewed_verifier_evidence_gap_counterexample",
            evidence,
        )
    if "wrong selection" in failure_family or (
        "selection" in typed_object_quality
        and patch_legitimacy in {
            "no_effective_patch",
            "patcher_failed_without_effective_patch",
        }
    ):
        return CounterexampleClassification(
            CounterexampleType.COVERAGE_GAP,
            "reviewed_selection_scope_counterexample",
            evidence,
        )
    if "stale" in failure_family or "contaminated" in failure_family or "shared_fact" in typed_object_quality:
        return CounterexampleClassification(
            CounterexampleType.INVARIANT_LEVEL,
            "reviewed_shared_belief_or_contamination_counterexample",
            evidence,
        )
    if patch_legitimacy in {"no_effective_patch", "patcher_failed_without_effective_patch"}:
        return CounterexampleClassification(
            CounterexampleType.COVERAGE_GAP,
            "reviewed_no_effective_patch_without_low_level_trace",
            evidence,
        )

    return CounterexampleClassification(
        CounterexampleType.UNKNOWN,
        "insufficient_counterexample_evidence",
        evidence,
    )


def classify_counterexample_from_ledger(ledger: RecoveryLedger) -> CounterexampleClassification:
    try:
        state = StructuredRecoveryState.from_dict(dict(ledger.structured_state or {}))
    except Exception:
        return CounterexampleClassification(
            CounterexampleType.UNKNOWN,
            "ledger_structured_state_unavailable",
            {},
        )
    return classify_counterexample(state, ledger=ledger)


def priority_actions_for_type(counterexample_type: CounterexampleType | str) -> list[str]:
    ce_type = _coerce_type(counterexample_type)
    return [action.value for action in ACTION_PRIORITY[ce_type]]


def never_first_actions_for_type(counterexample_type: CounterexampleType | str) -> set[str]:
    ce_type = _coerce_type(counterexample_type)
    return {action.value for action in NEVER_FIRST.get(ce_type, set())}


def _coerce_type(value: CounterexampleType | str) -> CounterexampleType:
    if isinstance(value, CounterexampleType):
        return value
    try:
        return CounterexampleType(str(value))
    except ValueError:
        return CounterexampleType.UNKNOWN


def _string_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value or "").strip()
    return {text} if text else set()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
