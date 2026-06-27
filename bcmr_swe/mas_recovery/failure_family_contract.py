"""Failure-family contracts for evidence-conditioned MASGuard recovery.

The contracts are deterministic and instance-agnostic.  They translate a
bounded-observation failure span into a small recovery program: what source
semantics must be preserved, what patch directions are forbidden, which sanity
checks should happen before spending oracle budget, and when the recovery
attempt must stop.
"""

from __future__ import annotations

from typing import Any


SCHEMA = "masguard.failure_family_contract.v1"


def build_failure_family_contract(
    *,
    failure_span: dict[str, Any] | None,
    target_paths: list[str] | None = None,
    operator_gate: str = "",
    patch_contract: str = "",
) -> dict[str, Any]:
    """Build a generic recovery contract from a failure-span family.

    The result is safe to place in a recovery sidecar.  It does not inspect a
    workspace, call a model, or use oracle outcome labels.
    """

    span = dict(failure_span or {})
    family = str(span.get("family", "") or "unknown_clean_failure")
    subtype = str(span.get("subtype", "") or "")
    targets = _dedupe(str(path) for path in list(target_paths or []) if str(path).strip())
    base = _base_contract(
        family=family,
        subtype=subtype,
        targets=targets,
        operator_gate=operator_gate,
        patch_contract=patch_contract,
    )
    family_specific = _family_specific_contract(family=family, span=span)
    merged = _merge_contracts(base, family_specific)
    merged["schema"] = SCHEMA
    merged["family"] = family
    merged["subtype"] = subtype
    merged["operator_gate"] = str(operator_gate or "")
    merged["patch_contract"] = str(patch_contract or "")
    merged["target_paths"] = targets[:10]
    merged["claim_boundary"] = {
        "deterministic_family_contract": True,
        "built_from_failure_span_and_source_boundary": True,
        "does_not_call_models": True,
        "does_not_execute_recovery": True,
        "does_not_use_oracle_success_label": True,
        "does_not_add_recovery_credit": True,
    }
    return merged


def _base_contract(
    *,
    family: str,
    subtype: str,
    targets: list[str],
    operator_gate: str,
    patch_contract: str,
) -> dict[str, Any]:
    source_only = patch_contract == "source_only"
    validation_steps = [
        "inspect_bounded_failure_span_before_editing",
        "produce_fresh_minimal_source_diff",
        "run_focused_fail_to_pass_or_equivalent_validation",
        "record_post_patch_failure_span_if_validation_fails",
    ]
    if source_only:
        validation_steps.insert(2, "reject_test_generated_or_eval_target_edits")
    if any(path.endswith(".py") for path in targets):
        validation_steps.insert(2, "run_python_syntax_or_import_sanity_on_touched_python_sources")

    hard_constraints = [
        "contract_v17_use_failure_family_not_instance_specific_patch_recipe",
        "contract_v17_patch_only_the_source_path_explaining_the_bounded_span",
        "contract_v17_keep_reusable_localization_as_boundary_not_ground_truth",
        "contract_v17_do_not_copy_previous_failed_patch_verbatim",
    ]
    if source_only:
        hard_constraints.append("contract_v17_source_only_patch_required")
    if operator_gate:
        hard_constraints.append(f"contract_v17_operator_gate:{operator_gate}")
    if family:
        hard_constraints.append(f"contract_v17_failure_family:{family}")
    if subtype:
        hard_constraints.append(f"contract_v17_failure_subtype:{subtype}")

    return {
        "mode": "family_conditioned_source_recovery",
        "semantic_invariants": [],
        "expected_behavior_constraints": [
            "The patch must directly remove the bounded-observation failure, not merely change an unrelated source path."
        ],
        "forbidden_patch_directions": [
            "Do not edit tests, generated files, snapshots, or evaluation targets for recovery credit.",
            "Do not broaden localization unless the bounded observation contradicts the source boundary.",
        ],
        "patch_quality_rejection_rules": [
            "Reject no-diff outputs when the bounded observation is a clean source failure.",
            "Reject patches that introduce Python syntax, import, or test collection failures.",
            "Reject patches that touch only callers while the span evidence points to a shared source helper.",
        ],
        "recovery_program_steps": validation_steps,
        "hard_constraints": hard_constraints,
        "stop_conditions": [
            "stop_after_one_family_conditioned_repatch",
            "stop_if_no_fresh_source_diff_can_be_produced_inside_the_target_boundary",
            "stop_if_syntax_import_or_collection_sanity_fails_after_one_bounded_revision",
            "stop_if_the_same_failure_family_and_subtype_remain_after_validation",
        ],
        "risk_flags": [],
    }


def _family_specific_contract(*, family: str, span: dict[str, Any]) -> dict[str, Any]:
    cause = str(span.get("verified_failure_cause", "") or "")
    api = dict(span.get("api_surface_mismatch", {}) or {})
    parameter = str(api.get("parameter", "") or "")
    callable_name = str(api.get("callable", "") or "")
    undefined = str(span.get("undefined_symbol", "") or "")

    if family == "missing_attribute":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["The attribute must be provided through the object's normal lifecycle, not hidden by a broad exception handler."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch initialization, copying, delegation, or attribute exposure where the object state is created."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not catch AttributeError and return a dummy value without preserving object semantics."],
            "patch_quality_rejection_rules": [
                "Reject patches that only change __getattr__/__getattribute__ fallback behavior without preserving the named attribute.",
                "Reject patches that special-case the test name or expected exception text.",
            ],
            "recovery_program_steps": [
                "trace_object_lifecycle_to_attribute_read",
                "patch_minimal_initializer_or_forwarding_path",
                "validate_original_missing_attribute_span_is_removed",
            ],
        }

    if family == "missing_symbol":
        symbol = undefined or "the undefined symbol"
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + [f"Every new reference to {symbol} must be locally defined, imported, or removed."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + [f"Resolve {symbol} inside the changed source boundary without introducing a new global dependency."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + [f"Do not leave {symbol} reachable on the focused validation path."],
            "patch_quality_rejection_rules": [
                "Reject patches that add helper calls without definitions or imports.",
                "Reject patches that move the NameError to a different branch of the same focused path.",
            ],
            "recovery_program_steps": [
                "scan_patch_for_new_identifiers_before_validation",
                "define_import_or_inline_the_required_helper",
                "run_import_or_collection_sanity_before_oracle",
            ],
            "risk_flags": ["high_collection_breakage_risk"],
        }

    if family == "api_surface_mismatch":
        parts = []
        if parameter:
            parts.append(f"parameter '{parameter}'")
        if callable_name:
            parts.append(f"callable '{callable_name}'")
        target = " and ".join(parts) or "the focused API surface"
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + [f"{target} must remain compatible with existing callers and the focused failing call."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Thread, map, or explicitly reject the mismatched API argument at the narrow source boundary."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not remove the caller-visible API path or ignore the unexpected argument silently."],
            "patch_quality_rejection_rules": [
                "Reject patches that only update documentation, tests, or call sites.",
                "Reject patches that accept arbitrary **kwargs without routing or validation.",
            ],
            "recovery_program_steps": [
                "identify_public_wrapper_and_internal_implementation_boundary",
                "thread_or_map_the_mismatched_parameter_minimally",
                "validate_default_callers_and_focused_keyword_call",
            ],
        }

    if family == "warning_policy":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["Warning emission, silence, and stacklevel behavior must stay scoped to the focused condition."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch the branch that controls the observed warning policy while preserving return semantics."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not install broad warning filters or edit pytest warning assertions."],
            "patch_quality_rejection_rules": [
                "Reject patches that globally suppress warning classes.",
                "Reject patches that make unrelated warning paths noisier.",
            ],
            "recovery_program_steps": [
                "locate_warning_emission_or_suppression_branch",
                "patch_the_narrow_policy_condition",
                "validate_warning_and_non_warning_paths",
            ],
        }

    if family == "formatter_output_semantics":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["Formatted text must preserve separators, escaping, precision, and structural delimiters expected by the focused formatter."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch the shared formatter or escaping path reached by the bounded span."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not normalize away the expected literal difference or edit expected output assets."],
            "patch_quality_rejection_rules": [
                "Reject patches that change only one caller when the span reaches a shared formatter helper.",
                "Reject patches that broaden string normalization outside the focused source path.",
            ],
            "recovery_program_steps": [
                "identify_common_formatter_helper_from_span_chain",
                "patch_formatter_semantics_at_shared_source_path",
                "validate_multiple_formatter_call_shapes_when_collectable",
            ],
        }

    if family == "metadata_preservation":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["Metadata, attrs, and named container state must follow the existing object lifecycle contract."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch the construction, copy, or wrapper path that drops metadata before the assertion observes it."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not preserve metadata by mutating unrelated global state or changing numeric semantics."],
            "patch_quality_rejection_rules": [
                "Reject patches that hard-code a metadata key from the test.",
                "Reject patches that preserve attrs only on one special-case class while the span shows a lifecycle issue.",
            ],
            "recovery_program_steps": [
                "trace_metadata_from_input_to_result_object",
                "patch_copy_or_constructor_semantics",
                "validate_value_and_metadata_together",
            ],
        }

    if family == "dimension_shape_mismatch":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["Dimension names, shape, and ndim must be consistent at object construction and validation boundaries."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch shape normalization before invalid state reaches the focused constructor or validator."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not bypass dimension validation with a broad exception or unconditional coercion."],
            "patch_quality_rejection_rules": [
                "Reject patches that coerce dimensions without preserving names.",
                "Reject patches that only silence validation errors.",
            ],
            "recovery_program_steps": [
                "trace_shape_and_dimension_normalization_path",
                "patch_minimal_pre_validation_normalizer",
                "validate_dim_name_and_ndim_contracts",
            ],
        }

    if family == "image_regression":
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or [])
            + ["Rendering state, geometry, transform, and style changes must preserve baseline image semantics."],
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch the source rendering path, not test tolerance or expected image files."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not change image baselines, tolerance thresholds, decorators, or skip logic."],
            "patch_quality_rejection_rules": [
                "Reject patches that touch baseline images or image comparison configuration.",
                "Reject patches that hide the image path behind a backend-specific special case without span evidence.",
            ],
            "recovery_program_steps": [
                "trace_rendering_state_to_image_difference",
                "patch_minimal_geometry_style_or_transform_source_path",
                "validate_focused_image_or_nearest_collectable_rendering_check",
            ],
        }

    if family in {"assertion_semantic_mismatch", "value_semantic_mismatch"}:
        return {
            "semantic_invariants": list(span.get("semantic_invariants", []) or []),
            "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or [])
            + ["Patch the source value path that explains the focused assertion or exception."],
            "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or [])
            + ["Do not convert the observed semantic failure into a broader protocol or collection failure."],
            "patch_quality_rejection_rules": [
                "Reject patches that only change error messages while the value path remains wrong.",
                "Reject patches that make the focused failure disappear by skipping validation.",
            ],
            "recovery_program_steps": [
                "trace_value_from_source_path_to_focused_failure",
                "patch_smallest_shared_value_or_exception_branch",
                "validate_the_original_assertion_or_exception_span",
            ],
        }

    fallback_expected = []
    if cause:
        fallback_expected.append(f"Address the verified cause: {cause}.")
    return {
        "expected_behavior_constraints": list(span.get("expected_behavior_constraints", []) or []) + fallback_expected,
        "forbidden_patch_directions": list(span.get("forbidden_patch_directions", []) or []),
        "patch_quality_rejection_rules": [
            "Reject broad rewrites when the family is unknown; keep the patch inside the source boundary.",
            "Reject patches that increase touched source files beyond the bounded source boundary without new evidence.",
        ],
        "recovery_program_steps": [
            "treat_unknown_family_as_minimal_nonregression_repatch",
            "patch_only_source_paths_supported_by_span_chain",
            "validate_that_failure_output_changes_or_resolves",
        ],
        "risk_flags": ["unknown_family_lower_confidence"],
    }


def _merge_contracts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in base.items()}
    for key, value in extra.items():
        if key in {
            "semantic_invariants",
            "expected_behavior_constraints",
            "forbidden_patch_directions",
            "patch_quality_rejection_rules",
            "recovery_program_steps",
            "hard_constraints",
            "stop_conditions",
            "risk_flags",
        }:
            merged[key] = _dedupe(list(merged.get(key, []) or []) + list(value or []))
        else:
            merged[key] = value
    return merged


def _dedupe(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
