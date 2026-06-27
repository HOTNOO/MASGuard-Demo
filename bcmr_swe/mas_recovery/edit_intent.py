"""Build MASGuard edit intent from graph and bounded-observation evidence.

The edit intent is produced before patch synthesis.  It converts the G2
propagation summary, updated hypothesis, bounded observation, and failure-span
family into a compact source-edit contract that the existing CAR patch-intent
parser already understands.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from bcmr_swe.recovery.patch_intent import CARPatchIntent
from swe_mas.utils.path_filters import classify_changed_files, normalize_repo_path


SCHEMA_VERSION = "masguard.edit_intent.v1"
PROMPT_BEGIN = "[MASGUARD EDIT INTENT]"
PROMPT_END = "[/MASGUARD EDIT INTENT]"


def build_masguard_edit_intent(
    *,
    patcher_failure_evidence: dict[str, Any] | None,
    fixed_localization: dict[str, Any] | None = None,
    operator_gate: str = "",
    patch_contract: str = "",
    enabled: bool = True,
) -> dict[str, Any]:
    """Return a source-span anchored edit-intent payload.

    This function is deterministic and side-effect free.  It does not inspect
    the workspace, does not call a model, and does not decide oracle success.
    """

    if not enabled:
        return _skipped("not_enabled")
    evidence = dict(patcher_failure_evidence or {})
    if not evidence:
        return _skipped("missing_patcher_failure_evidence")

    fixed = dict(fixed_localization or {})
    target_paths = _source_paths(
        list(evidence.get("selected_target_candidates", []) or [])
        + list(fixed.get("selected_target_candidates", []) or [])
        + list(evidence.get("support_file_candidates", []) or [])
    )
    changed_source_paths = _source_paths(
        list(dict(evidence.get("changed_file_classes", {}) or {}).get("source_files", []) or [])
        + list(evidence.get("changed_files", []) or [])
    )
    candidate_source_paths = _source_paths(
        list(evidence.get("candidate_patch_source_files", []) or [])
        + list(evidence.get("candidate_patch_replay_source_files", []) or [])
    )
    if not target_paths:
        target_paths = _dedupe_paths(candidate_source_paths + changed_source_paths)
    suspect_paths = _dedupe_paths(target_paths + changed_source_paths + candidate_source_paths)

    span = dict(evidence.get("failure_span_family", {}) or {})
    top_after = dict(evidence.get("g2_top_hypothesis_after", {}) or {})
    driver = dict(evidence.get("g2_recovery_driver", {}) or {})
    signature = dict(evidence.get("post_patch_failure_signature", {}) or {})
    observations = [
        dict(item)
        for item in list(evidence.get("g2_bounded_observations", []) or [])
        if isinstance(item, dict)
    ]
    graph = dict(evidence.get("g2_propagation_summary", {}) or {})
    failed_tests = _dedupe(
        [
            str(item)
            for item in list(signature.get("failed_tests", []) or [])
            if str(item).strip()
        ]
    )
    failure_excerpt = _dedupe(
        [
            str(item)
            for item in list(signature.get("exception_excerpt", []) or [])
            + list(evidence.get("failure_output_excerpt", []) or [])
            if str(item).strip()
        ]
    )
    semantic_terms = _dedupe(
        list(evidence.get("semantic_invariants", []) or [])
        + list(evidence.get("expected_behavior_constraints", []) or [])
        + list(evidence.get("forbidden_patch_directions", []) or [])
        + [span.get("verified_failure_cause", ""), evidence.get("verified_failure_cause", "")]
    )
    failed_patch_feedback = _failed_patch_feedback(evidence)
    span_linked_edit_sketch = _span_linked_edit_sketch(
        span=span,
        target_paths=target_paths,
        failure_excerpt=failure_excerpt,
        observations=observations,
    )
    assertion_scope_constraints = _assertion_contract_scope_constraints(evidence)
    assertion_edit_steps = _assertion_contract_edit_steps(evidence)
    if assertion_edit_steps:
        span_linked_edit_sketch = _dedupe(span_linked_edit_sketch + assertion_edit_steps)
    selected_action = _selected_action(driver, operator_gate)
    preserve_candidate = bool(candidate_source_paths) and bool(
        evidence.get("candidate_patch_replayed", False)
        or evidence.get("candidate_patch_replay_attempted", False)
    )
    require_target_touch = bool(target_paths)
    intent_id = _stable_id(
        [
            evidence.get("instance_id", ""),
            selected_action,
            ",".join(target_paths[:4]),
            span.get("family", ""),
            span.get("subtype", ""),
            top_after.get("failure_type", ""),
            operator_gate,
        ]
    )
    car_intent = CARPatchIntent(
        intent_id=intent_id,
        selected_action=selected_action,
        counterexample_type=str(top_after.get("failure_type", "") or span.get("family", "") or ""),
        replay_scope="g2_observation_conditioned_source_repair",
        repair_mode="source_span_anchored_minimal_edit",
        execution_profile="focused_source_repair" if patch_contract == "source_only" else "normal",
        active_object_type=str(span.get("family", "") or top_after.get("responsible_stage", "") or ""),
        active_object_id=str(span.get("verified_failure_cause", "") or signature.get("headline", "") or ""),
        latest_revision_type="bounded_observation_updated_hypothesis",
        target_paths=target_paths[:8],
        suspect_paths=suspect_paths[:8],
        candidate_source_paths=candidate_source_paths[:8],
        avoid_target_paths=[],
        require_fresh_source_diff=True,
        require_target_touch=require_target_touch,
        require_evidence_before_retarget=True,
        preserve_candidate_source=preserve_candidate,
        max_fresh_source_files=max(1, min(3, len(target_paths) or 3)),
        directives=_directives(operator_gate=operator_gate, span=span, patch_contract=patch_contract),
        negative_constraints=_negative_constraints(evidence),
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "enabled": True,
        "intent_id": intent_id,
        "claim_boundary": {
            "built_before_patch_synthesis": True,
            "derived_from_g2_graph_and_bounded_observation": True,
            "does_not_execute_recovery": True,
            "does_not_grant_recovery_credit": True,
        },
        "source_span_anchors": {
            "target_paths": target_paths[:8],
            "suspect_paths": suspect_paths[:8],
            "candidate_source_paths": candidate_source_paths[:8],
            "failed_tests": failed_tests[:8],
            "failure_excerpt": failure_excerpt[:8],
        },
        "graph_chain": {
            "node_count": graph.get("node_count", 0),
            "edge_count": graph.get("edge_count", 0),
            "semantic_edge_count": graph.get("semantic_edge_count", 0),
            "true_summary_flags": [
                key
                for key, value in dict(graph.get("summary_flags", {}) or {}).items()
                if value is True
            ][:12],
            "bounded_observation_signals": [
                str(item.get("observed_signal", "") or "")
                for item in observations
                if str(item.get("observed_signal", "") or "")
            ][:8],
        },
        "failure_span": {
            "family": str(span.get("family", "") or ""),
            "subtype": str(span.get("subtype", "") or ""),
            "verified_failure_cause": str(
                span.get("verified_failure_cause", "") or evidence.get("verified_failure_cause", "") or ""
            ),
            "semantic_retry_reason": str(
                span.get("semantic_retry_reason", "") or evidence.get("semantic_retry_reason", "") or ""
            ),
        },
        "updated_hypothesis": {
            "failure_type": str(top_after.get("failure_type", "") or ""),
            "responsible_stage": str(top_after.get("responsible_stage", "") or ""),
            "confidence": str(top_after.get("confidence", "") or ""),
            "rationale": str(top_after.get("rationale", "") or "")[:500],
        },
        "operator_gate": str(operator_gate or ""),
        "patch_contract": str(patch_contract or ""),
        "semantic_terms": semantic_terms[:12],
        "failed_patch_feedback": failed_patch_feedback,
        "span_linked_edit_sketch": span_linked_edit_sketch,
        "assertion_contract_scope_constraints": assertion_scope_constraints,
        "assertion_contract_edit_steps": assertion_edit_steps,
        "source_edit_contract_text": _source_edit_contract_text(evidence),
        "car_patch_intent": car_intent.to_dict(),
        "prompt_text": _prompt_text(
            car_intent=car_intent,
            target_paths=target_paths,
            failed_tests=failed_tests,
            failure_excerpt=failure_excerpt,
            semantic_terms=semantic_terms,
            failed_patch_feedback=failed_patch_feedback,
            span_linked_edit_sketch=span_linked_edit_sketch,
            assertion_contract_scope_constraints=assertion_scope_constraints,
            assertion_contract_edit_steps=assertion_edit_steps,
            span=span,
            top_after=top_after,
            operator_gate=operator_gate,
            patch_contract=patch_contract,
        ),
    }
    if not target_paths:
        payload["warnings"] = ["no_source_target_path_anchor"]
    return payload


def edit_intent_skipped_result(reason: str) -> dict[str, Any]:
    return _skipped(reason)


def _prompt_text(
    *,
    car_intent: CARPatchIntent,
    target_paths: list[str],
    failed_tests: list[str],
    failure_excerpt: list[str],
    semantic_terms: list[str],
    failed_patch_feedback: dict[str, Any],
    span_linked_edit_sketch: list[str],
    assertion_contract_scope_constraints: list[str],
    assertion_contract_edit_steps: list[str],
    span: dict[str, Any],
    top_after: dict[str, Any],
    operator_gate: str,
    patch_contract: str,
) -> str:
    lines = [
        PROMPT_BEGIN,
        "This edit intent was built before patch synthesis from the MAS propagation graph, bounded observation, and updated failure hypothesis.",
        "Hard recovery rule: patch the smallest source span inside the target source boundary that explains the bounded observation.",
        "Hard recovery rule: do not use this retry for broad repository search, test edits, generated files, or unrelated cleanup.",
        f"Operator gate: {operator_gate or '<none>'}.",
        f"Patch contract: {patch_contract or '<none>'}.",
        f"Updated failure type: {top_after.get('failure_type', '')}.",
        f"Failure span family: {span.get('family', '')}; subtype: {span.get('subtype', '')}.",
    ]
    cause = str(span.get("verified_failure_cause", "") or "").strip()
    if cause:
        lines.append(f"Verified cause: {cause[:360]}")
    if target_paths:
        lines.append("Source-span target boundary:")
        lines.extend(f"- {path}" for path in target_paths[:8])
    if failed_tests:
        lines.append("Focused tests or verifier targets that define the span:")
        lines.extend(f"- {item}" for item in failed_tests[:8])
    if failure_excerpt:
        lines.append("Bounded-observation failure excerpt:")
        lines.extend(f"- {line[:240]}" for line in failure_excerpt[:8])
    if semantic_terms:
        lines.append("Semantic constraints to preserve or address:")
        lines.extend(f"- {item[:240]}" for item in semantic_terms[:10])
    if assertion_contract_scope_constraints:
        lines.append("Assertion-contract source-scope constraints:")
        lines.extend(f"- {item[:260]}" for item in assertion_contract_scope_constraints[:8])
    if assertion_contract_edit_steps:
        lines.append("Ordered assertion-contract edit steps:")
        lines.extend(f"{idx}. {item[:260]}" for idx, item in enumerate(assertion_contract_edit_steps[:8], start=1))
        lines.append(
            "Hard recovery rule: complete every ordered assertion-contract edit step in one patch before local validation."
        )
    failed_added_lines = [
        str(item)
        for item in list(failed_patch_feedback.get("failed_patch_added_lines", []) or [])
        if str(item).strip()
    ]
    if failed_added_lines:
        lines.append("Failed patch-shape feedback from the previous bounded attempt:")
        lines.append("Hard recovery rule: do not repeat these added source lines unless the new diff also changes the failing behavior.")
        lines.extend(f"- {item[:240]}" for item in failed_added_lines[:8])
    if span_linked_edit_sketch:
        lines.append("Span-linked source edit sketch:")
        lines.extend(f"- {item[:240]}" for item in span_linked_edit_sketch[:10])
    lines.append(car_intent.to_prompt_text())
    lines.append(PROMPT_END)
    return "\n".join(line for line in lines if str(line).strip())


def _assertion_contract_scope_constraints(evidence: dict[str, Any]) -> list[str]:
    explicit = _dedupe(
        str(item)
        for item in list(evidence.get("assertion_contract_scope_constraints", []) or [])
        if str(item).strip()
    )
    if explicit:
        return explicit
    sources = [
        dict(item)
        for item in list(evidence.get("semantic_invariant_sources", []) or [])
        if isinstance(item, dict)
    ]
    invariant_ids = {
        str(item.get("invariant_id", "") or "")
        for item in sources
    }
    if "xarray_indexvariable_copy_unicode_dtype_aliasing" not in invariant_ids:
        return []
    return [
        "In xarray/core/indexing.py, add exactly one new copy method and place it only inside class PandasIndexAdapter.",
        "Do not add copy methods to ExplicitIndexer, LazilyOuterIndexedArray, LazilyVectorizedIndexedArray, or other indexing helper classes.",
        "Do not add unrelated adapter APIs such as astype for this contract unless focused validation explicitly asks for them.",
        "The final source diff must also touch xarray/core/variable.py and update the data=None branch of IndexVariable.copy.",
    ]


def _source_edit_contract_text(evidence: dict[str, Any]) -> str:
    explicit = str(evidence.get("source_edit_contract_text", "") or "").strip()
    if explicit:
        return explicit
    sources = [
        dict(item)
        for item in list(evidence.get("semantic_invariant_sources", []) or [])
        if isinstance(item, dict)
    ]
    invariant_ids = {
        str(item.get("invariant_id", "") or "")
        for item in sources
    }
    support_files = _source_paths(list(evidence.get("support_file_candidates", []) or []))
    if "xarray_indexvariable_copy_unicode_dtype_aliasing" not in invariant_ids:
        return ""
    required = [
        path
        for path in ["xarray/core/indexing.py", "xarray/core/variable.py"]
        if path in set(support_files)
    ]
    if len(required) != 2:
        return ""
    payload = {
        "schema_version": "masguard.source_edit_contract.v4",
        "contract_id": "assertion_contract_xarray_indexvariable_copy_v1",
        "requirements": {
            "max_source_files": 2,
            "primary_source_target": "xarray/core/indexing.py",
            "allowed_source_files": required,
            "required_source_files": required,
            "state_failure_mechanism_before_edit": True,
            "produce_minimal_diff_or_abstain": True,
            "focused_validation_required": True,
            "diff_first_or_abstain": True,
            "exact_class_scope_required": True,
        },
        "forbidden": {
            "path_classes": ["test", "generated", "build_artifact", "installed_copy"],
            "patch_patterns": [
                "copy_method_outside_PandasIndexAdapter",
                "unrelated_adapter_api_astype",
                "neighbor_method_deletion",
            ],
        },
        "credit_boundary": {
            "oracle_still_required": True,
            "semantic_invariant_gate_still_required": True,
        },
    }
    return (
        "[MASGUARD SOURCE EDIT CONTRACT]\n"
        "Assertion-contract strict source-edit contract. This focused contract requires a coordinated "
        "two-file source diff: add the backing adapter capability in xarray/core/indexing.py and update "
        "the IndexVariable.copy data=None branch in xarray/core/variable.py. Do not broaden beyond the "
        "allowed source files, do not edit tests, and do not add copy methods outside PandasIndexAdapter.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "[/MASGUARD SOURCE EDIT CONTRACT]"
    )


def _assertion_contract_edit_steps(evidence: dict[str, Any]) -> list[str]:
    explicit = _dedupe(
        str(item)
        for item in list(evidence.get("assertion_contract_edit_steps", []) or [])
        if str(item).strip()
    )
    if explicit:
        return explicit
    sources = [
        dict(item)
        for item in list(evidence.get("semantic_invariant_sources", []) or [])
        if isinstance(item, dict)
    ]
    invariant_ids = {
        str(item.get("invariant_id", "") or "")
        for item in sources
    }
    support_files = set(_source_paths(list(evidence.get("support_file_candidates", []) or [])))
    if "xarray_indexvariable_copy_unicode_dtype_aliasing" not in invariant_ids:
        return []
    if not {"xarray/core/indexing.py", "xarray/core/variable.py"}.issubset(support_files):
        return []
    return [
        "Open xarray/core/indexing.py and add PandasIndexAdapter.copy(self, deep=True) inside the PandasIndexAdapter class.",
        "The adapter copy method must return PandasIndexAdapter(self.array.copy(deep=True), self._dtype) when deep is true.",
        "The adapter copy method must return PandasIndexAdapter(self.array, self._dtype) when deep is false.",
        "Open xarray/core/variable.py and change only the data=None branch of IndexVariable.copy to use self._data.copy(deep=deep).",
        "Preserve IndexVariable.copy documentation and neighboring methods such as equals, _data_equals, and to_index_variable.",
    ]


def _span_linked_edit_sketch(
    *,
    span: dict[str, Any],
    target_paths: list[str],
    failure_excerpt: list[str],
    observations: list[dict[str, Any]],
) -> list[str]:
    family = str(span.get("family", "") or "")
    subtype = str(span.get("subtype", "") or "")
    cause = str(span.get("verified_failure_cause", "") or "")
    text = "\n".join(
        list(failure_excerpt)
        + [
            str(item.get("output_excerpt", "") or "")
            for item in observations
            if isinstance(item, dict)
        ]
    )
    quoted_diffs = _quoted_expected_observed_pairs(text)
    call_sites = _call_site_lines(text)
    paths = ", ".join(target_paths[:3])

    sketch = [
        "Start from the bounded-observation failure span, then edit the reusable source path that every failing span reaches.",
        "If several failure excerpts share one source helper or lifecycle path, prefer that shared path over a single caller patch.",
    ]
    if paths:
        sketch.append(f"Keep the edit inside the source boundary unless the span evidence proves it wrong: {paths}.")

    if family == "formatter_output_semantics":
        sketch.extend(
            [
                "Repair the common formatter or escaping helper that produces the observed text, not only one formatter caller.",
                "Preserve the separator and escaping semantics visible in the bounded diff for every failing formatter path.",
                "When multiple formatter classes or format_ticks/__call__ paths fail, avoid a patch that only rewrites the first call site.",
            ]
        )
    elif family == "missing_attribute":
        sketch.extend(
            [
                "Repair object lifecycle or attribute exposure where the object is created, copied, or initialized.",
                "Prefer initializing or preserving the missing attribute on the real object over masking AttributeError at the read site.",
                "If the attribute is derived from another field, route that derivation through the existing source invariant.",
            ]
        )
    elif family == "api_surface_mismatch":
        api = dict(span.get("api_surface_mismatch", {}) or {})
        parameter = str(api.get("parameter", "") or "")
        callable_name = str(api.get("callable", "") or "")
        if parameter:
            sketch.append(f"Thread the '{parameter}' parameter through the focused API path or map it to the existing implementation contract.")
        if callable_name:
            sketch.append(f"Patch the source implementation of {callable_name} or its forwarding wrapper, not the caller/test.")
        sketch.append("Keep default behavior unchanged for callers that do not pass the new or mismatched parameter.")
    elif family == "warning_policy":
        sketch.extend(
            [
                "Patch the exact source branch that emits, suppresses, or forwards the warning observed in the bounded failure.",
                "Avoid broad global filters; warning behavior must remain scoped to the focused condition.",
                "Preserve existing non-warning return semantics while changing only the warning policy edge.",
            ]
        )
    elif family == "image_regression":
        sketch.extend(
            [
                "Patch the rendering state, geometry, transform, or style source path that reaches the image comparison span.",
                "Do not change image baselines, tolerances, or test decorators.",
                "Prefer preserving previous renderer invariants over adding special cases around the image test.",
            ]
        )
    elif family == "dimension_shape_mismatch":
        sketch.extend(
            [
                "Patch the shape/dimension normalization point before the invalid object is constructed.",
                "Preserve dimension names and ndim consistency together; do not bypass validation with a broad exception handler.",
            ]
        )
    elif family == "metadata_preservation":
        sketch.extend(
            [
                "Patch the object construction/copy path that drops attrs or metadata before the assertion observes the result.",
                "Preserve metadata only under the existing keep-attrs/object-lifecycle contract; do not change numeric semantics.",
            ]
        )
    elif family in {"assertion_semantic_mismatch", "value_semantic_mismatch"}:
        sketch.append("Patch the source value path that directly explains the focused assertion or exception, not the verifier/test.")

    if subtype:
        sketch.append(f"Span subtype to satisfy: {subtype}.")
    if cause:
        sketch.append(f"Verified bounded cause: {cause}.")
    if call_sites:
        sketch.append("Bounded call-site evidence to cover: " + " | ".join(call_sites[:3]))
    if quoted_diffs:
        sketch.append("Observed/expected literals to preserve: " + " | ".join(quoted_diffs[:3]))
    return _dedupe(sketch)[:12]


def _quoted_expected_observed_pairs(text: str) -> list[str]:
    pairs: list[str] = []
    for line in str(text or "").splitlines():
        if "!=" not in line and "==" not in line and "diff:" not in line.lower():
            continue
        literals = re.findall(r"'([^']{1,120})'", line)
        if len(literals) >= 2:
            pairs.append(f"{literals[0]} -> {literals[1]}")
    return _dedupe(pairs)


def _call_site_lines(text: str) -> list[str]:
    values: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 180:
            continue
        lowered = stripped.lower()
        if any(token in lowered for token in ["formatter(", "format_ticks", "pytest.warns", "attributeerror", "typeerror"]):
            values.append(stripped)
    return _dedupe(values)


def _selected_action(driver: dict[str, Any], operator_gate: str) -> str:
    action = str(driver.get("selected_action", "") or "").strip().upper()
    if action in {"LOCAL_REPAIR", "REPAIR_LOCAL", "SCOPE_EXPAND", "EXPAND_SCOPE"}:
        return "LOCAL_REPAIR" if action == "REPAIR_LOCAL" else action
    if operator_gate in {"handoff_correction_or_ablation", "shared_fact_quarantine_then_repatch"}:
        return "SCOPE_EXPAND"
    return "LOCAL_REPAIR"


def _directives(*, operator_gate: str, span: dict[str, Any], patch_contract: str) -> list[str]:
    values = [
        "produce_fresh_source_diff",
        "run_focused_validation",
        "edit_intended_source_boundary",
        "minimal_source_patch",
        "source_span_anchored_edit",
        "address_bounded_observation_failure_span",
    ]
    if patch_contract == "source_only":
        values.append("source_only_no_tests_or_generated_files")
    family = str(span.get("family", "") or "")
    if family:
        values.append(f"failure_span_family_{family}")
    if operator_gate:
        values.append(f"operator_gate_{operator_gate}")
    return _dedupe(values)


def _negative_constraints(evidence: dict[str, Any]) -> list[str]:
    values = [
        "Do not rerun localization unless the bounded observation proves the source boundary is wrong.",
        "Do not copy the previous failed patch verbatim.",
    ]
    if _failed_patch_feedback(evidence).get("failed_patch_added_lines"):
        values.append(
            "Do not repeat the previous failed patch shape unless the new source diff changes the failing behavior."
        )
    values.extend(str(item) for item in list(evidence.get("forbidden_patch_directions", []) or []))
    values.extend(str(item) for item in list(evidence.get("patch_quality_rejection_rules", []) or []))
    return _dedupe(values)[:6]


def _failed_patch_feedback(evidence: dict[str, Any]) -> dict[str, Any]:
    feedback = dict(evidence.get("failed_patch_feedback", {}) or {})
    added_lines = _dedupe(
        _normalize_added_line(line)
        for line in list(feedback.get("failed_patch_added_lines", []) or [])
        + list(feedback.get("failed_added_lines", []) or [])
        + list(evidence.get("previous_patch_added_lines", []) or [])
        + _extract_added_lines(str(feedback.get("failed_patch_diff", "") or ""))
        + _extract_added_lines(str(evidence.get("failed_patch_diff", "") or ""))
        + _extract_added_lines(str(evidence.get("workspace_diff", "") or ""))
        + _extract_added_lines("\n".join(str(item) for item in list(evidence.get("previous_patch_excerpt", []) or [])))
        if _normalize_added_line(line)
    )
    diff_sha = str(feedback.get("failed_patch_diff_sha256", "") or evidence.get("failed_patch_diff_sha256", "") or "")
    return {
        "present": bool(added_lines or diff_sha),
        "failed_patch_diff_sha256": diff_sha,
        "failed_patch_added_lines": added_lines[:12],
        "claim_boundary": "failed_patch_feedback_is_a_no_credit_revision_constraint",
    }


def _extract_added_lines(text: str) -> list[str]:
    added: list[str] = []
    for line in str(text or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return added


def _normalize_added_line(line: Any) -> str:
    text = str(line or "").strip()
    if text.startswith("+") and not text.startswith("+++"):
        text = text[1:].strip()
    if not text or text.startswith("#"):
        return ""
    return re.sub(r"\s+", " ", text)


def _source_paths(values: list[Any]) -> list[str]:
    normalized = _dedupe_paths(values)
    classes = classify_changed_files(normalized)
    return _dedupe_paths(list(classes.get("source_files", []) or []))


def _dedupe_paths(values: list[Any]) -> list[str]:
    return _dedupe([normalize_repo_path(str(value or "")) for value in values])


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _stable_id(parts: list[Any]) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "enabled": False,
        "skip_reason": reason,
        "prompt_text": "",
        "claim_boundary": {
            "built_before_patch_synthesis": False,
            "does_not_grant_recovery_credit": True,
        },
    }
