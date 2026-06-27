"""Classify bounded-observation failure spans for recovery routing.

The classifier is intentionally instance-agnostic. It reads only the focused
failure excerpts produced by bounded observations and converts recurring
software-failure shapes into recovery constraints.
"""

from __future__ import annotations

import re
from typing import Any


SEMANTIC_INVARIANT_GATE_FAMILIES = {
    "api_surface_mismatch",
    "missing_attribute",
    "missing_symbol",
    "metadata_preservation",
    "image_regression",
    "warning_policy",
    "dimension_shape_mismatch",
    "formatter_output_semantics",
}


def infer_failure_span_family(
    observations: list[dict[str, Any]] | None,
    *,
    top_updated: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer a generic failure-span family from bounded observations."""

    observations = [dict(item) for item in list(observations or []) if isinstance(item, dict)]
    text = _observation_text(observations)
    span = classify_failure_span_text(text)
    hypothesis_family = str(dict(top_updated or {}).get("failure_type", "") or "")
    if hypothesis_family:
        span["hypothesis_failure_type"] = hypothesis_family
    return span


def classify_failure_span_text(text: str) -> dict[str, Any]:
    normalized = _strip_ansi(str(text or ""))
    lowered = normalized.lower()
    invariants: list[str] = []
    expected: list[str] = []
    forbidden: list[str] = []
    api_surface_mismatch: dict[str, Any] = {}
    undefined_symbol = ""
    family = "unknown_clean_failure"
    subtype = ""
    exception_type = _exception_type(normalized)
    evidence_excerpt = _first_failure_line(normalized)
    verified_failure_cause = ""

    keyword_match = re.search(
        r"TypeError:\s+([\w.<>]+|\w+\(\)|__init__\(\))\s+got an unexpected keyword argument ['\"]([^'\"]+)['\"]",
        normalized,
    )
    if keyword_match:
        callable_name = keyword_match.group(1)
        parameter = keyword_match.group(2)
        family = "api_surface_mismatch"
        subtype = "unexpected_keyword_argument"
        api_surface_mismatch = {
            "exception_type": subtype,
            "callable": callable_name,
            "parameter": parameter,
            "evidence": f"TypeError: {callable_name} got an unexpected keyword argument '{parameter}'",
        }
        invariants.append(
            f"The focused constructor/API path must accept or correctly forward the '{parameter}' parameter."
        )
        expected.append(
            f"Map '{parameter}' to the existing implementation path or add the minimal API surface required by the focused test."
        )
        forbidden.append(
            f"Do not leave '{parameter}' unhandled or return no-diff when the bounded observation proves an API-surface mismatch."
        )
        verified_failure_cause = f"unexpected keyword argument '{parameter}'"

    attribute_match = re.search(
        r"AttributeError:\s+['\"]([^'\"]+)['\"] object has no attribute ['\"]([^'\"]+)['\"]",
        normalized,
    )
    if not api_surface_mismatch and attribute_match:
        object_type = attribute_match.group(1)
        attribute = attribute_match.group(2)
        family = "missing_attribute"
        subtype = "attribute_not_initialized_or_exposed"
        invariants.append(
            f"Objects of type '{object_type}' must expose '{attribute}' before the focused path reads it."
        )
        expected.append(
            f"Initialize, preserve, or correctly route the '{attribute}' attribute inside the reusable source boundary."
        )
        forbidden.append(
            f"Do not mask the AttributeError without preserving the expected '{attribute}' behavior."
        )
        verified_failure_cause = f"missing attribute '{attribute}' on '{object_type}'"

    name_match = re.search(r"NameError:\s+name ['\"]([^'\"]+)['\"] is not defined", normalized)
    if family == "unknown_clean_failure" and name_match:
        undefined_symbol = name_match.group(1)
        family = "missing_symbol"
        subtype = "undefined_helper_or_name"
        invariants.append(
            f"The retry patch must be self-contained for symbol '{undefined_symbol}'."
        )
        expected.append(
            f"Define, import, or remove the unresolved '{undefined_symbol}' reference within the changed source boundary."
        )
        forbidden.append(
            f"Do not introduce or keep a call to '{undefined_symbol}' unless it is defined or imported by the source patch."
        )
        verified_failure_cause = f"undefined symbol '{undefined_symbol}'"

    if family == "unknown_clean_failure" and _mentions_metadata_preservation(lowered):
        family = "metadata_preservation"
        subtype = "attrs_or_metadata_lost"
        invariants.append("Metadata/attrs carried by the input object must be preserved on the focused result.")
        expected.append("Preserve attrs/metadata through the operation under the same keep-attrs or object-lifecycle contract.")
        forbidden.append("Do not fix numeric values while dropping attrs/metadata expected by the focused test.")
        verified_failure_cause = "metadata or attrs not preserved"

    if family == "unknown_clean_failure" and _mentions_image_regression(lowered):
        family = "image_regression"
        subtype = "image_comparison_mismatch"
        invariants.append("Rendered image output must remain visually equivalent to the expected baseline under the focused test.")
        expected.append("Repair the source behavior that changes drawing geometry, style, or renderer output; keep image tests untouched.")
        forbidden.append("Do not weaken image comparison tolerance, skip image checks, or edit expected images.")
        verified_failure_cause = "image comparison mismatch"

    if family == "unknown_clean_failure" and _mentions_warning_policy(lowered):
        family = "warning_policy"
        subtype = "unexpected_or_missing_warning"
        invariants.append("The focused path must preserve the expected warning policy.")
        expected.append("Emit, silence, or route warnings only where the focused test expects that behavior.")
        forbidden.append("Do not broadly suppress warnings or edit warning tests to hide the policy mismatch.")
        verified_failure_cause = "warning policy mismatch"

    if family == "unknown_clean_failure" and _mentions_dimension_shape(lowered):
        family = "dimension_shape_mismatch"
        subtype = "dimension_or_ndim_contract"
        invariants.append("Dimension names and ndim semantics must remain consistent for the focused object construction path.")
        expected.append("Repair the source dimension/shape handling so the observed dims and data ndim agree.")
        forbidden.append("Do not bypass dimension validation or mask the failure with a broad exception handler.")
        verified_failure_cause = "dimension or ndim contract mismatch"

    if family == "unknown_clean_failure" and _mentions_formatter_output(lowered):
        family = "formatter_output_semantics"
        subtype = "formatted_text_mismatch"
        invariants.append("Formatted output must preserve the exact semantic separators and escaping expected by the focused formatter test.")
        expected.append("Repair the formatter/escaping path that produced the observed text mismatch.")
        forbidden.append("Do not edit expected strings or broadly normalize formatted output outside the focused source path.")
        verified_failure_cause = "formatted output mismatch"

    if family == "unknown_clean_failure" and "assertionerror" in lowered:
        family = "assertion_semantic_mismatch"
        subtype = "focused_assertion_mismatch"
        expected.append("Repair the source behavior that makes the focused assertion mismatch persist.")
        forbidden.append("Do not edit tests or broaden localization without new bounded-observation evidence.")
        verified_failure_cause = "focused assertion mismatch"
    elif family == "unknown_clean_failure" and "valueerror" in lowered:
        family = "value_semantic_mismatch"
        subtype = "focused_value_error"
        expected.append("Repair the source behavior that raises the observed ValueError on the focused path.")
        forbidden.append("Do not replace the observed semantic error with a broader generic failure.")
        verified_failure_cause = "focused ValueError"

    routed_operator_gate = (
        "semantic_invariant_guarded_repatch"
        if family in SEMANTIC_INVARIANT_GATE_FAMILIES
        else "patch_contract_nonregression_repatch"
    )
    return {
        "schema": "masguard_failure_span_family_v1",
        "family": family,
        "subtype": subtype,
        "exception_type": exception_type,
        "evidence_excerpt": evidence_excerpt,
        "semantic_retry_reason": _semantic_retry_reason(family, subtype),
        "semantic_invariants": _dedupe(invariants),
        "expected_behavior_constraints": _dedupe(expected),
        "forbidden_patch_directions": _dedupe(forbidden),
        "api_surface_mismatch": api_surface_mismatch,
        "undefined_symbol": undefined_symbol,
        "verified_failure_cause": verified_failure_cause,
        "routed_operator_gate": routed_operator_gate,
        "route_reason": (
            "bounded_observation_span_family_requires_semantic_invariant_gate"
            if routed_operator_gate == "semantic_invariant_guarded_repatch"
            else "bounded_observation_span_family_uses_nonregression_repatch"
        ),
        "requires_semantic_invariant_gate": routed_operator_gate == "semantic_invariant_guarded_repatch",
    }


def _observation_text(observations: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for observation in observations:
        chunks.extend(str(line) for line in list(observation.get("failure_output_excerpt", []) or []))
        chunks.append(str(observation.get("output_excerpt", "") or ""))
        chunks.append(str(observation.get("observed_signal", "") or ""))
    return "\n".join(chunk for chunk in chunks if str(chunk).strip())


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))


def _exception_type(text: str) -> str:
    match = re.search(
        r"\b(AssertionError|AttributeError|ValueError|TypeError|NameError|ImportError|ModuleNotFoundError|IndexError|FutureWarning|ImageComparisonFailure)\b",
        str(text or ""),
    )
    return match.group(1) if match else ""


def _first_failure_line(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "...":
            continue
        lowered = stripped.lower()
        if (
            stripped.startswith("E")
            or stripped.startswith(">")
            or "failed " in lowered
            or "error:" in lowered
            or "warning:" in lowered
            or "assert" in lowered
        ):
            return stripped[:240]
    return next((line.strip()[:240] for line in str(text or "").splitlines() if line.strip()), "")


def _mentions_metadata_preservation(lowered: str) -> bool:
    return (
        "actual.attrs" in lowered
        or ".attrs ==" in lowered
        or "ordereddict() ==" in lowered
        or "right contains" in lowered and "attr" in lowered
        or "keep_attrs" in lowered
    )


def _mentions_image_regression(lowered: str) -> bool:
    return (
        "imagecomparisonfailure" in lowered
        or "images not close" in lowered
        or "failed-diff.png" in lowered
        or "rms " in lowered and "result_images" in lowered
    )


def _mentions_warning_policy(lowered: str) -> bool:
    return (
        "futurewarning" in lowered
        or "pytest.warns" in lowered
        or "did not warn" in lowered
        or "warning" in lowered and "expected" in lowered
    )


def _mentions_dimension_shape(lowered: str) -> bool:
    return (
        "dimensions" in lowered
        and ("ndim" in lowered or "same length" in lowered or "_dims" in lowered)
    )


def _mentions_formatter_output(lowered: str) -> bool:
    return (
        "formatter" in lowered
        and ("mathdefault" in lowered or "usetex" in lowered or "format_ticks" in lowered)
    )


def _semantic_retry_reason(family: str, subtype: str) -> str:
    if family == "api_surface_mismatch":
        return "localized_type_error"
    if family == "missing_attribute":
        return "localized_attribute_error"
    if family == "missing_symbol":
        return "localized_name_error"
    if family in {"metadata_preservation", "image_regression", "formatter_output_semantics"}:
        return "localized_assertion_regression"
    if family == "dimension_shape_mismatch":
        return "localized_value_error_regression"
    if family == "warning_policy":
        return "localized_warning_policy_regression"
    if family == "assertion_semantic_mismatch":
        return "localized_assertion_mismatch"
    if family == "value_semantic_mismatch":
        return "localized_value_error"
    return subtype


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
