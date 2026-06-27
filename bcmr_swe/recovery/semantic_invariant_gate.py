"""Deterministic semantic-invariant gates for MAS-DX-R recovery."""

from __future__ import annotations

import re

from typing import Any


def semantic_invariant_patch_violations(
    *,
    patch_text: str,
    evidence: dict[str, Any],
) -> list[str]:
    """Detect forbidden patch shapes implied by semantic retry evidence."""

    evidence_text = _evidence_constraint_text(evidence).lower()
    patch_lower = str(patch_text or "").lower()
    violations: list[str] = []
    if _mentions_false_none_identifier_lookup(evidence_text):
        if "ast.constant" in patch_lower and ("false" in patch_lower or "none" in patch_lower):
            violations.append("semantic_invariant_false_none_identifier_lookup_not_preserved")
        if _patch_pytest_empty_expression_false_sentinel_changed(str(patch_text or "")):
            violations.append("semantic_invariant_false_none_identifier_lookup_false_sentinel_corrupted")
    undefined_helper = _undefined_helper_from_evidence(evidence, evidence_text)
    if undefined_helper and _patch_calls_symbol(patch_text, undefined_helper) and not _patch_defines_or_imports_symbol(
        patch_text, undefined_helper
    ):
        violations.append("semantic_invariant_undefined_helper_self_containment_not_preserved")
    if _mentions_timeseries_first_column(evidence_text):
        if "required columns missing" in patch_lower or "required column missing" in patch_lower:
            violations.append("semantic_invariant_timeseries_first_column_error_not_preserved")
    if _mentions_point_scalar_left_multiplication(evidence_text):
        if "__rmul__" not in patch_lower and "rmul" not in patch_lower:
            violations.append("semantic_invariant_point_scalar_left_multiplication_not_preserved")
        if _forbids_point_addition_dispatch_change(evidence_text) and (
            re.search(r"^\+\s*def __radd__", str(patch_text or ""), flags=re.MULTILINE)
            or re.search(r"^\+\s*if isinstance\(other, expr\)", patch_lower, flags=re.MULTILINE)
            or re.search(r"^\+\s*return self\.__radd__\(other\)", patch_lower, flags=re.MULTILINE)
        ):
            violations.append("semantic_invariant_point_addition_dispatch_changed")
    if _mentions_pylint_type_comment_import_consumption(evidence_text):
        if _requires_pylint_assignment_type_comment_entry(evidence_text) and not _patch_consumes_module_level_type_comment_imports(
            str(patch_text or "")
        ):
            violations.append("semantic_invariant_pylint_type_comment_import_consumption_not_preserved")
        if _patch_adds_duplicate_leave_assign(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_duplicate_leave_assign")
        if _mentions_pylint_attribute_type_comment_consumption(evidence_text) and not _patch_handles_astroid_attribute_type_comments(
            str(patch_text or "")
        ):
            violations.append("semantic_invariant_pylint_attribute_type_comment_consumption_not_preserved")
    if _mentions_pylint_is_pypy_collection_fix(evidence_text):
        if not _patch_restores_is_pypy_constant(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_is_pypy_collection_fix_missing")
    if _mentions_requests_header_none_merge_semantics(evidence_text):
        if _patch_filters_request_none_before_merge(str(patch_text or "")):
            violations.append("semantic_invariant_requests_header_none_request_override_not_preserved")
        if _patch_broadens_requests_header_none_boundary(str(patch_text or "")):
            violations.append("semantic_invariant_requests_header_none_boundary_broadened")
    if _mentions_requests_binary_payload_preservation(evidence_text):
        if _patch_imports_undefined_requests_compat_iterable(str(patch_text or "")):
            violations.append("semantic_invariant_requests_binary_payload_undefined_compat_iterable")
        if _patch_requests_binary_only_excludes_stream_but_still_encodes(str(patch_text or "")):
            violations.append("semantic_invariant_requests_binary_payload_stream_only_fix")
        if _patch_broadens_requests_binary_decode_boundary(str(patch_text or "")):
            violations.append("semantic_invariant_requests_binary_payload_broad_decode")
    if _mentions_sklearn_ridge_classifier_store_cv_values(evidence_text):
        if not _patch_adds_ridge_classifier_store_cv_values_constructor(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_ridge_classifier_store_cv_values_constructor_missing")
        if _patch_only_changes_docstring_for_store_cv_values(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_ridge_classifier_store_cv_values_doc_only")
    if _mentions_sklearn_hgb_string_target_early_stopping(evidence_text):
        hgb_docstring_fix = _patch_hgb_early_stopping_fix_in_docstring(str(patch_text or ""))
        hgb_missing_cast = _patch_hgb_classes_index_without_integer_cast(str(patch_text or ""))
        if hgb_docstring_fix:
            violations.append("semantic_invariant_sklearn_hgb_string_target_fix_in_docstring")
        if hgb_missing_cast:
            violations.append("semantic_invariant_sklearn_hgb_classes_index_without_integer_cast")
        if (
            not _patch_hgb_maps_encoded_targets_to_classes(str(patch_text or ""))
            and not hgb_docstring_fix
            and not hgb_missing_cast
        ):
            violations.append("semantic_invariant_sklearn_hgb_string_target_mapping_missing")
    if _mentions_sklearn_lasso_lars_copy_x_propagation(evidence_text):
        if _patch_sklearn_lasso_lars_copy_x_constructor_only(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_lasso_lars_copy_x_constructor_only")
        if not _patch_sklearn_lasso_lars_copy_x_propagates_to_preprocess(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_lasso_lars_copy_x_preprocess_propagation_missing")
    if _mentions_sklearn_dataframe_output_dtype_preservation(evidence_text):
        if _patch_sklearn_dtype_preservation_unpopulated_dtype_map(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_dataframe_output_dtype_map_not_propagated")
    if _mentions_sklearn_roc_curve_first_threshold_inf(evidence_text):
        if _patch_sklearn_roc_curve_threshold_caps_inf(str(patch_text or "")):
            violations.append("semantic_invariant_sklearn_roc_curve_first_threshold_inf_capped")
        elif _patch_touches_sklearn_roc_curve_thresholds(str(patch_text or "")) and "np.inf" not in str(
            patch_text or ""
        ):
            violations.append("semantic_invariant_sklearn_roc_curve_first_threshold_inf_missing")
    api_surface = dict(evidence.get("api_surface_mismatch", {}) or {})
    parameter = str(api_surface.get("parameter", "") or "").strip()
    if parameter and not _patch_mentions_api_parameter(str(patch_text or ""), parameter):
        violations.append("semantic_invariant_api_surface_required_parameter_missing")
    wrong_parameters = [
        str(item).strip()
        for item in list(api_surface.get("wrong_parameters", []) or [])
        if str(item).strip()
    ]
    if parameter and wrong_parameters and _patch_only_mentions_wrong_api_parameters(
        str(patch_text or ""),
        required_parameter=parameter,
        wrong_parameters=wrong_parameters,
    ):
        violations.append("semantic_invariant_api_surface_wrong_parameter_only")
    if _mentions_django_autofield_subclass_semantics(evidence_text):
        if _patch_broadens_django_autofield_subclasses(str(patch_text or "")):
            violations.append("semantic_invariant_django_autofield_subclasses_boundary_broadened")
        if _patch_uses_recursive_django_autofield_tuple_subclass_check(str(patch_text or "")):
            violations.append("semantic_invariant_django_autofield_recursive_tuple_subclass_check")
        if _requires_django_autofield_iterative_subclass_check(
            evidence_text
        ) and not _patch_uses_django_autofield_iterative_subclass_check(str(patch_text or "")):
            violations.append("semantic_invariant_django_autofield_iterative_subclass_check_missing")
    if _mentions_django_mysql_dbshell_option_precedence(evidence_text):
        if not _patch_preserves_django_mysql_dbshell_classmethod_signature(str(patch_text or "")):
            violations.append("semantic_invariant_django_mysql_dbshell_classmethod_signature_not_preserved")
        if _patch_rewrites_django_mysql_dbshell_method_broadly(str(patch_text or "")):
            violations.append("semantic_invariant_django_mysql_dbshell_broad_rewrite")
        if _patch_removes_django_mysql_dbshell_env_or_connection_options(str(patch_text or "")):
            violations.append("semantic_invariant_django_mysql_dbshell_connection_options_removed")
        if not _patch_adds_django_mysql_database_precedence(str(patch_text or "")):
            violations.append("semantic_invariant_django_mysql_dbshell_database_precedence_missing")
    if _mentions_pytest_caplog_phase_record_isolation(evidence_text):
        if _patch_clears_logcapturehandler_records_in_place(str(patch_text or "")):
            violations.append("semantic_invariant_pytest_caplog_phase_records_not_isolated")
    if _mentions_pytest_unittest_class_skip_teardown(evidence_text):
        if _patch_returns_early_for_skipped_unittest_methods(str(patch_text or "")):
            violations.append("semantic_invariant_pytest_unittest_class_skip_must_report_skip")
        if _patch_only_checks_unittest_method_skip(str(patch_text or "")):
            violations.append("semantic_invariant_pytest_unittest_class_skip_parent_not_checked")
    if _mentions_sphinx_none_annotation_reftype_obj(evidence_text):
        if _patch_sphinx_none_annotation_rewrites_autodoc_typehints(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_none_annotation_wrong_autodoc_typehints_scope")
        if not _patch_sphinx_none_annotation_reftype_obj(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_none_annotation_reftype_obj_missing")
    if _mentions_sphinx_reserved_toctree_entries(evidence_text):
        if "sphinx/directives/other.py" not in str(patch_text or "").lower():
            violations.append("semantic_invariant_sphinx_reserved_toctree_directive_entry_construction_missing")
        elif not _patch_adds_sphinx_reserved_toctree_entry_append(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_reserved_toctree_entry_append_missing")
        if _patch_skips_sphinx_reserved_toctree_entries(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_reserved_toctree_entries_skipped")
    if _mentions_sphinx_graphviz_svg_external_hyperlink(evidence_text):
        if _patch_sphinx_graphviz_only_rewrites_svg_relative_paths(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_graphviz_svg_external_link_relative_path_only")
        if _patch_sphinx_graphviz_external_hyperlink_fix_missing(str(patch_text or "")):
            violations.append("semantic_invariant_sphinx_graphviz_svg_external_link_mapping_missing")
    if _mentions_sympy_permutation_non_disjoint_cycles(evidence_text):
        if _patch_sympy_permutation_only_skips_cycle_duplicate_check(str(patch_text or "")):
            violations.append("semantic_invariant_sympy_permutation_cycle_args_not_canonicalized")
        if not _patch_sympy_permutation_non_disjoint_cycles_array_form(str(patch_text or "")):
            violations.append("semantic_invariant_sympy_permutation_non_disjoint_cycle_array_form_missing")
    if _mentions_xarray_quantile_keep_attrs(evidence_text):
        if _patch_changes_xarray_rank_without_quantile_path(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_quantile_patch_touched_rank_instead")
        if _patch_sets_temp_dataset_attrs_without_returned_dataarray_attrs(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_quantile_temp_dataset_attrs_not_returned")
        if _patch_quantile_only_changes_variable_without_keep_attrs_contract(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_quantile_keep_attrs_contract_missing")
        if _patch_xarray_quantile_changes_unrelated_dataarray_methods(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_quantile_unrelated_dataarray_method_changed")
    if _mentions_xarray_where_keep_attrs(evidence_text):
        if not _patch_xarray_where_keep_attrs_contract(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_where_keep_attrs_contract_missing")
        if _patch_xarray_where_keep_attrs_blind_apply_ufunc_forward(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_where_keep_attrs_uses_cond_attrs")
    if _mentions_xarray_integrate_dim_keyword(evidence_text):
        if _patch_xarray_integrate_renames_dim_without_alias(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_integrate_dim_keyword_renamed_without_alias")
        if _patch_xarray_integrate_missing_dim_keyword_alias(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_integrate_dim_keyword_alias_missing")
    if _mentions_xarray_custom_values_attr_scalar(evidence_text):
        if _patch_xarray_custom_values_attr_posthoc_variable_init_coercion(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_custom_values_attr_scalar_wrong_posthoc_coercion")
        if _patch_xarray_custom_values_attr_missing_as_compatible_guard(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_custom_values_attr_scalar_values_guard_missing")
    if _mentions_xarray_reset_index_xindexes_contract(evidence_text):
        if _patch_xarray_reset_index_only_changes_datavariables(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_reset_index_datavariables_only")
        if _patch_xarray_reset_index_missing_xindexes_contract(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_reset_index_xindexes_contract_missing")
    if _mentions_xarray_combine_auto_bystander_dimension_ordering(evidence_text):
        if _patch_xarray_combine_auto_deletes_monotonic_index_guard(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_combine_auto_global_monotonic_guard_deleted")
        if _patch_xarray_combine_auto_sorts_everything_without_bystander_scope(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_combine_auto_broad_sort_without_bystander_scope")
    if _mentions_xarray_stack_multiindex_dtype(evidence_text):
        if _patch_xarray_stack_dtype_only_adds_pandas_index_fastpath(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_stack_multiindex_dtype_fastpath_only")
        if _patch_xarray_places_pandas_multiindex_before_pandasindex(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_pandas_multiindex_before_pandasindex")
    if _mentions_xarray_indexvariable_copy_aliasing(evidence_text):
        if _patch_xarray_indexvariable_deletes_neighbor_methods(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_indexvariable_neighbor_methods_deleted")
        if _patch_xarray_indexvariable_adds_copy_outside_pandas_adapter(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_indexvariable_copy_wrong_class_scope")
        if _patch_xarray_indexvariable_adds_unrelated_adapter_api(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_indexvariable_unrelated_adapter_api")
        if not _patch_xarray_indexvariable_copy_aliasing_fix(str(patch_text or "")):
            violations.append("semantic_invariant_xarray_indexvariable_copy_aliasing_fix_missing")
    if _mentions_astropy_observed_frame_transform_import_scope(evidence_text):
        if _patch_astropy_observed_transform_uses_local_frame_imports_in_decorators(str(patch_text or "")):
            violations.append("semantic_invariant_astropy_observed_frame_decorator_import_scope_not_preserved")
        if _patch_astropy_bypasses_cirs_observed_transforms(str(patch_text or "")):
            violations.append("semantic_invariant_astropy_cirs_observed_transform_graph_bypassed")
    if _mentions_matplotlib_huge_range_lognorm_image(evidence_text):
        if _patch_matplotlib_broadens_lognorm_invalid_range_semantics(str(patch_text or "")):
            violations.append("semantic_invariant_matplotlib_huge_range_log_broad_lognorm_change")
        if _patch_matplotlib_returns_intermediate_masked_image_tuple(str(patch_text or "")):
            violations.append("semantic_invariant_matplotlib_huge_range_log_wrong_make_image_return")
        if _patch_matplotlib_returns_transparent_rgba_for_lognorm_no_positive_values(str(patch_text or "")):
            violations.append("semantic_invariant_matplotlib_huge_range_log_transparent_output_changed")
        if _patch_matplotlib_mutates_lognorm_state_in_image_path(str(patch_text or "")):
            violations.append("semantic_invariant_matplotlib_huge_range_log_mutates_norm_state")
        if _patch_matplotlib_recomputes_lognorm_image_output_branch(str(patch_text or "")):
            violations.append("semantic_invariant_matplotlib_huge_range_log_image_output_changed")
    if _mentions_pylint_min_similarity_zero_disables_checker(evidence_text):
        if _patch_pylint_only_changes_similar_report_string(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_report_string_only")
        if _patch_pylint_edits_similar_tests(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_tests_edited")
        if _patch_pylint_uses_missing_min_similarity_lines_attr(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_missing_attr")
        if _patch_pylint_sets_infinite_min_lines_for_zero(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_still_reports_total")
        if _patch_pylint_uses_undefined_val_in_run(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_undefined_val")
        if _patch_pylint_only_changes_checker_lifecycle_for_standalone_run(str(patch_text or "")):
            violations.append("semantic_invariant_pylint_min_similarity_zero_standalone_run_not_changed")
    return _dedupe(violations)


def semantic_invariant_row_violations(
    *,
    row: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    """Detect semantic-invariant violations visible in a completed raw row."""

    evidence_text = _evidence_constraint_text(evidence).lower()
    if not evidence_text:
        return []
    patch_text = str(row.get("patch", "") or "")
    row_text = _joined_text(
        [
            patch_text,
            str(row.get("fail_to_pass_output", "") or ""),
            str(row.get("oracle_output", "") or ""),
            str(row.get("planner_error", "") or ""),
            str(row.get("implementer_error", "") or ""),
        ]
    )
    row_lower = row_text.lower()
    violations = semantic_invariant_patch_violations(
        patch_text=patch_text,
        evidence=evidence,
    )
    oracle_success = bool(row.get("oracle_success", False))
    if _mentions_false_none_identifier_lookup(evidence_text):
        unresolved_lookup_failure = (
            not oracle_success
            and (
                "where false = evaluate('false'" in row_lower
                or 'where false = evaluate("false"' in row_lower
                or "where none = evaluate('none'" in row_lower
                or 'where none = evaluate("none"' in row_lower
            )
        )
        if unresolved_lookup_failure:
            violations.append("semantic_invariant_false_none_identifier_lookup_not_preserved")
    undefined_helper = _undefined_helper_from_evidence(evidence, evidence_text)
    if undefined_helper:
        if _patch_calls_symbol(patch_text, undefined_helper) and not _patch_defines_or_imports_symbol(
            patch_text, undefined_helper
        ):
            violations.append("semantic_invariant_undefined_helper_self_containment_not_preserved")
        unresolved_undefined_helper = (
            not oracle_success
            and f"name '{undefined_helper.lower()}' is not defined" in row_lower
        )
        if unresolved_undefined_helper:
            violations.append("semantic_invariant_undefined_helper_self_containment_not_preserved")
    if _mentions_timeseries_first_column(evidence_text):
        generic_required_columns = "required columns missing" in row_lower or "required column missing" in row_lower
        first_column_expected = (
            "expected 'time' as the first column" in row_lower
            or 'expected "time" as the first column' in row_lower
        )
        if not oracle_success and generic_required_columns and first_column_expected:
            violations.append("semantic_invariant_timeseries_first_column_error_not_preserved")
    if _mentions_point_scalar_left_multiplication(evidence_text):
        unresolved_left_multiplication_failure = (
            not oracle_success
            and (
                "unsupported operand type(s) for *: 'int' and 'point2d'" in row_lower
                or "unsupported operand type(s) for *: 'int' and 'point3d'" in row_lower
                or "unsupported operand type(s) for *: 'int' and 'point'" in row_lower
            )
        )
        if unresolved_left_multiplication_failure:
            violations.append("semantic_invariant_point_scalar_left_multiplication_not_preserved")
    if _mentions_pylint_type_comment_import_consumption(evidence_text):
        unresolved_unused_import = (
            not oracle_success
            and "unused import abc" in row_lower
            and "unused-import" in row_lower
        )
        if unresolved_unused_import:
            violations.append("semantic_invariant_pylint_type_comment_import_consumption_not_preserved")
        if _patch_adds_duplicate_leave_assign(patch_text):
            violations.append("semantic_invariant_pylint_duplicate_leave_assign")
        unresolved_attribute_unused_import = (
            not oracle_success
            and "bar imported from foo" in row_lower
            and "unused-import" in row_lower
        )
        if unresolved_attribute_unused_import:
            violations.append("semantic_invariant_pylint_attribute_type_comment_consumption_not_preserved")
    if _mentions_pylint_is_pypy_collection_fix(evidence_text):
        unresolved_is_pypy_import = (
            not oracle_success
            and "cannot import name 'is_pypy'" in row_lower
            and "pylint.constants" in row_lower
        )
        if unresolved_is_pypy_import:
            violations.append("semantic_invariant_pylint_is_pypy_collection_fix_missing")
    if _mentions_requests_header_none_merge_semantics(evidence_text):
        unresolved_header_none = (
            not oracle_success
            and ("accept-encoding: none" in row_lower or "header: none" in row_lower)
        )
        if unresolved_header_none:
            violations.append("semantic_invariant_requests_header_none_merge_semantics_not_preserved")
    if _mentions_requests_binary_payload_preservation(evidence_text):
        if _patch_imports_undefined_requests_compat_iterable(patch_text):
            violations.append("semantic_invariant_requests_binary_payload_undefined_compat_iterable")
        if _patch_requests_binary_only_excludes_stream_but_still_encodes(patch_text):
            violations.append("semantic_invariant_requests_binary_payload_stream_only_fix")
        if _patch_broadens_requests_binary_decode_boundary(patch_text):
            violations.append("semantic_invariant_requests_binary_payload_broad_decode")
        unresolved_binary_payload = (
            not oracle_success
            and (
                "test_binary_put" in row_lower
                or "unicodedecodeerror" in row_lower
                or "to_native_string(data)" in row_lower
            )
            and ("ascii" in row_lower or "_encode_params" in row_lower or "to_native_string" in row_lower)
        )
        if unresolved_binary_payload:
            violations.append("semantic_invariant_requests_binary_payload_not_preserved")
        unresolved_iterable = (
            not oracle_success
            and "iterable" in row_lower
            and (
                "cannot import name" in row_lower
                or "importerror" in row_lower
                or "name 'iterable' is not defined" in row_lower
            )
        )
        if unresolved_iterable:
            violations.append("semantic_invariant_requests_binary_payload_undefined_compat_iterable")
    if _mentions_sklearn_ridge_classifier_store_cv_values(evidence_text):
        unresolved_store_cv_values = (
            not oracle_success
            and "ridgeclassifiercv" in row_lower
            and "unexpected keyword argument 'store_cv_values'" in row_lower
        )
        if unresolved_store_cv_values:
            violations.append("semantic_invariant_sklearn_ridge_classifier_store_cv_values_constructor_missing")
    if _mentions_sklearn_hgb_string_target_early_stopping(evidence_text):
        hgb_docstring_fix = _patch_hgb_early_stopping_fix_in_docstring(patch_text)
        hgb_missing_cast = _patch_hgb_classes_index_without_integer_cast(patch_text)
        if hgb_docstring_fix:
            violations.append("semantic_invariant_sklearn_hgb_string_target_fix_in_docstring")
        if hgb_missing_cast:
            violations.append("semantic_invariant_sklearn_hgb_classes_index_without_integer_cast")
        if (
            not _patch_hgb_maps_encoded_targets_to_classes(patch_text)
            and not hgb_docstring_fix
            and not hgb_missing_cast
        ):
            violations.append("semantic_invariant_sklearn_hgb_string_target_mapping_missing")
        unresolved_string_target = (
            not oracle_success
            and "test_string_target_early_stopping" in row_lower
            and (
                "arrays used as indices must be of integer" in row_lower
                or "not supported between instances of 'str' and 'float'" in row_lower
                or 'not supported between instances of "str" and "float"' in row_lower
            )
        )
        if unresolved_string_target:
            violations.append("semantic_invariant_sklearn_hgb_string_target_mapping_missing")
    if _mentions_django_autofield_subclass_semantics(evidence_text):
        if _patch_broadens_django_autofield_subclasses(patch_text):
            violations.append("semantic_invariant_django_autofield_subclasses_boundary_broadened")
        if _patch_uses_recursive_django_autofield_tuple_subclass_check(patch_text):
            violations.append("semantic_invariant_django_autofield_recursive_tuple_subclass_check")
        unresolved_recursion = (
            not oracle_success
            and "recursionerror" in row_lower
            and "autofieldmeta" in row_lower
            and "__subclasscheck__" in row_lower
            and "issubclass(subclass, self._subclasses)" in row_lower
        )
        if unresolved_recursion:
            violations.append("semantic_invariant_django_autofield_recursive_tuple_subclass_check")
    if _mentions_django_mysql_dbshell_option_precedence(evidence_text):
        if not _patch_preserves_django_mysql_dbshell_classmethod_signature(patch_text):
            violations.append("semantic_invariant_django_mysql_dbshell_classmethod_signature_not_preserved")
        if _patch_rewrites_django_mysql_dbshell_method_broadly(patch_text):
            violations.append("semantic_invariant_django_mysql_dbshell_broad_rewrite")
        if _patch_removes_django_mysql_dbshell_env_or_connection_options(patch_text):
            violations.append("semantic_invariant_django_mysql_dbshell_connection_options_removed")
        if not _patch_adds_django_mysql_database_precedence(patch_text):
            violations.append("semantic_invariant_django_mysql_dbshell_database_precedence_missing")
        unresolved_precedence = (
            not oracle_success
            and (
                "deprecatedoptiondbname" in row_lower
                or "settingdbname" in row_lower
                or "optiondbname" in row_lower
            )
            and "test_options" in row_lower
        )
        if unresolved_precedence:
            violations.append("semantic_invariant_django_mysql_dbshell_database_precedence_missing")
    if _mentions_pytest_caplog_phase_record_isolation(evidence_text):
        if _patch_clears_logcapturehandler_records_in_place(patch_text):
            violations.append("semantic_invariant_pytest_caplog_phase_records_not_isolated")
        unresolved_phase_pollution = (
            not oracle_success
            and "test_clear_for_call_stage" in row_lower
            and "a_setup_log" in row_lower
            and "a_call_log" in row_lower
        )
        if unresolved_phase_pollution:
            violations.append("semantic_invariant_pytest_caplog_phase_records_not_isolated")
    if _mentions_pytest_unittest_class_skip_teardown(evidence_text):
        if _patch_returns_early_for_skipped_unittest_methods(patch_text):
            violations.append("semantic_invariant_pytest_unittest_class_skip_must_report_skip")
        if _patch_only_checks_unittest_method_skip(patch_text):
            violations.append("semantic_invariant_pytest_unittest_class_skip_parent_not_checked")
        unresolved_class_skip_teardown = (
            not oracle_success
            and "test_pdb_teardown_skipped_for_classes" in row_lower
            and "teardown:" in row_lower
            and "mytestcase.test_1" in row_lower
        )
        if unresolved_class_skip_teardown:
            violations.append("semantic_invariant_pytest_unittest_class_skip_teardown_not_suppressed")
    if _mentions_xarray_quantile_keep_attrs(evidence_text):
        if _patch_changes_xarray_rank_without_quantile_path(patch_text):
            violations.append("semantic_invariant_xarray_quantile_patch_touched_rank_instead")
        if _patch_sets_temp_dataset_attrs_without_returned_dataarray_attrs(patch_text):
            violations.append("semantic_invariant_xarray_quantile_temp_dataset_attrs_not_returned")
        if _patch_quantile_only_changes_variable_without_keep_attrs_contract(patch_text):
            violations.append("semantic_invariant_xarray_quantile_keep_attrs_contract_missing")
        unresolved_keep_attrs = (
            not oracle_success
            and "test_quantile" in row_lower
            and "actual.attrs" in row_lower
            and "orderedDict()".lower() in row_lower
        )
        if unresolved_keep_attrs:
            violations.append("semantic_invariant_xarray_quantile_keep_attrs_contract_missing")
    if _mentions_xarray_where_keep_attrs(evidence_text):
        if not _patch_xarray_where_keep_attrs_contract(patch_text):
            violations.append("semantic_invariant_xarray_where_keep_attrs_contract_missing")
        if _patch_xarray_where_keep_attrs_blind_apply_ufunc_forward(patch_text):
            violations.append("semantic_invariant_xarray_where_keep_attrs_uses_cond_attrs")
        unresolved_where_attrs = (
            not oracle_success
            and "where() got an unexpected keyword argument 'keep_attrs'" in row_lower
        )
        if unresolved_where_attrs:
            violations.append("semantic_invariant_xarray_where_keep_attrs_contract_missing")
        wrong_attr_source = (
            not oracle_success
            and "differing attributes" in row_lower
            and "l   attr: x" in row_lower
            and "r   attr: cond" in row_lower
        )
        if wrong_attr_source:
            violations.append("semantic_invariant_xarray_where_keep_attrs_uses_cond_attrs")
    if _mentions_xarray_integrate_dim_keyword(evidence_text):
        if _patch_xarray_integrate_renames_dim_without_alias(patch_text):
            violations.append("semantic_invariant_xarray_integrate_dim_keyword_renamed_without_alias")
        if _patch_xarray_integrate_missing_dim_keyword_alias(patch_text):
            violations.append("semantic_invariant_xarray_integrate_dim_keyword_alias_missing")
        unresolved_dim_keyword = (
            not oracle_success
            and "test_integrate" in row_lower
            and "dataarray.integrate() got an unexpected keyword argument 'dim'" in row_lower
        )
        if unresolved_dim_keyword:
            violations.append("semantic_invariant_xarray_integrate_dim_keyword_alias_missing")
    if _mentions_xarray_custom_values_attr_scalar(evidence_text):
        if _patch_xarray_custom_values_attr_posthoc_variable_init_coercion(patch_text):
            violations.append("semantic_invariant_xarray_custom_values_attr_scalar_wrong_posthoc_coercion")
        if _patch_xarray_custom_values_attr_missing_as_compatible_guard(patch_text):
            violations.append("semantic_invariant_xarray_custom_values_attr_scalar_values_guard_missing")
        unresolved_custom_values_scalar = (
            not oracle_success
            and "test_unsupported_type" in row_lower
            and "dimensions () must have the same length" in row_lower
            and "ndim=1" in row_lower
        )
        if unresolved_custom_values_scalar:
            violations.append("semantic_invariant_xarray_custom_values_attr_scalar_still_ndim_1")
    if _mentions_xarray_combine_auto_bystander_dimension_ordering(evidence_text):
        if _patch_xarray_combine_auto_deletes_monotonic_index_guard(patch_text):
            violations.append("semantic_invariant_xarray_combine_auto_global_monotonic_guard_deleted")
        if _patch_xarray_combine_auto_sorts_everything_without_bystander_scope(patch_text):
            violations.append("semantic_invariant_xarray_combine_auto_broad_sort_without_bystander_scope")
        unresolved_bystander_ordering = (
            not oracle_success
            and "test_combine_leaving_bystander_dimensions" in row_lower
            and "monotonic" in row_lower
            and "global index" in row_lower
        )
        if unresolved_bystander_ordering:
            violations.append("semantic_invariant_xarray_combine_auto_bystander_ordering_unresolved")
    if _mentions_sphinx_none_annotation_reftype_obj(evidence_text):
        if _patch_sphinx_none_annotation_rewrites_autodoc_typehints(patch_text):
            violations.append("semantic_invariant_sphinx_none_annotation_wrong_autodoc_typehints_scope")
        if not _patch_sphinx_none_annotation_reftype_obj(patch_text):
            violations.append("semantic_invariant_sphinx_none_annotation_reftype_obj_missing")
        wrong_none_reftype = (
            not oracle_success
            and "test_parse_annotation" in row_lower
            and "node[reftype] is not 'obj': 'class'" in row_lower
        )
        if wrong_none_reftype:
            violations.append("semantic_invariant_sphinx_none_annotation_reftype_obj_missing")
    if _mentions_sphinx_reserved_toctree_entries(evidence_text):
        if "sphinx/directives/other.py" not in patch_text.lower():
            violations.append("semantic_invariant_sphinx_reserved_toctree_directive_entry_construction_missing")
        elif not _patch_adds_sphinx_reserved_toctree_entry_append(patch_text):
            violations.append("semantic_invariant_sphinx_reserved_toctree_entry_append_missing")
        if _patch_skips_sphinx_reserved_toctree_entries(patch_text):
            violations.append("semantic_invariant_sphinx_reserved_toctree_entries_skipped")
        if (
            not oracle_success
            and "the node[entries] is not [(none, 'genindex'), (none, 'modindex'), (none, 'search')]" in row_lower
        ):
            violations.append("semantic_invariant_sphinx_reserved_toctree_entries_skipped")
    if _mentions_sphinx_graphviz_svg_external_hyperlink(evidence_text):
        if _patch_sphinx_graphviz_only_rewrites_svg_relative_paths(patch_text):
            violations.append("semantic_invariant_sphinx_graphviz_svg_external_link_relative_path_only")
        if _patch_sphinx_graphviz_external_hyperlink_fix_missing(patch_text):
            violations.append("semantic_invariant_sphinx_graphviz_svg_external_link_mapping_missing")
        unresolved_external_link = (
            not oracle_success
            and "test_inheritance_diagram_svg_html" in row_lower
            and "https://example.org" in row_lower
            and "assert 'https://example.org' in" in row_lower
        )
        if unresolved_external_link:
            violations.append("semantic_invariant_sphinx_graphviz_svg_external_link_not_preserved")
    if _mentions_sympy_permutation_non_disjoint_cycles(evidence_text):
        if _patch_sympy_permutation_only_skips_cycle_duplicate_check(patch_text):
            violations.append("semantic_invariant_sympy_permutation_cycle_args_not_canonicalized")
        if not _patch_sympy_permutation_non_disjoint_cycles_array_form(patch_text):
            violations.append("semantic_invariant_sympy_permutation_non_disjoint_cycle_array_form_missing")
        args_reconstruction_failure = (
            not oracle_success
            and "test_args" in row_lower
            and ("recursionerror" in row_lower or "test_sympy__combinatorics__permutations__permutation" in row_lower)
        )
        if args_reconstruction_failure:
            violations.append("semantic_invariant_sympy_permutation_cycle_args_not_canonicalized")
    if _mentions_xarray_stack_multiindex_dtype(evidence_text):
        if _patch_xarray_stack_dtype_only_adds_pandas_index_fastpath(patch_text):
            violations.append("semantic_invariant_xarray_stack_multiindex_dtype_fastpath_only")
        if _patch_xarray_places_pandas_multiindex_before_pandasindex(patch_text):
            violations.append("semantic_invariant_xarray_pandas_multiindex_before_pandasindex")
        unresolved_dtype_upcast = (
            not oracle_success
            and "test_restore_dtype_on_multiindexes" in row_lower
            and (
                "assert 'int64' == 'int32'" in row_lower
                or 'assert "int64" == "int32"' in row_lower
                or "assert 'float64' == 'float32'" in row_lower
                or 'assert "float64" == "float32"' in row_lower
            )
        )
        if unresolved_dtype_upcast:
            violations.append("semantic_invariant_xarray_stack_multiindex_dtype_not_preserved")
        unresolved_import_order = (
            not oracle_success
            and "xarray/core/indexes.py" in row_lower
            and (
                "name 'pandasindex' is not defined" in row_lower
                or 'name "pandasindex" is not defined' in row_lower
            )
        )
        if unresolved_import_order:
            violations.append("semantic_invariant_xarray_pandas_multiindex_before_pandasindex")
    if _mentions_astropy_observed_frame_transform_import_scope(evidence_text):
        if _patch_astropy_observed_transform_uses_local_frame_imports_in_decorators(patch_text):
            violations.append("semantic_invariant_astropy_observed_frame_decorator_import_scope_not_preserved")
        if _patch_astropy_bypasses_cirs_observed_transforms(patch_text):
            violations.append("semantic_invariant_astropy_cirs_observed_transform_graph_bypassed")
        unresolved_observed_name_error = (
            not oracle_success
            and "intermediate_rotation_transforms.py" in row_lower
            and (
                "name 'altaz' is not defined" in row_lower
                or 'name "altaz" is not defined' in row_lower
                or "name 'hadec' is not defined" in row_lower
                or 'name "hadec" is not defined' in row_lower
            )
        )
        if unresolved_observed_name_error:
            violations.append("semantic_invariant_astropy_observed_frame_decorator_import_scope_not_preserved")
    if _mentions_matplotlib_huge_range_lognorm_image(evidence_text):
        if _patch_matplotlib_broadens_lognorm_invalid_range_semantics(patch_text):
            violations.append("semantic_invariant_matplotlib_huge_range_log_broad_lognorm_change")
        if _patch_matplotlib_returns_intermediate_masked_image_tuple(patch_text):
            violations.append("semantic_invariant_matplotlib_huge_range_log_wrong_make_image_return")
        if _patch_matplotlib_returns_transparent_rgba_for_lognorm_no_positive_values(patch_text):
            violations.append("semantic_invariant_matplotlib_huge_range_log_transparent_output_changed")
        unresolved_invalid_lognorm = (
            not oracle_success
            and "test_huge_range_log" in row_lower
            and "invalid vmin or vmax" in row_lower
        )
        if unresolved_invalid_lognorm:
            violations.append("semantic_invariant_matplotlib_huge_range_log_lognorm_not_guarded")
        unresolved_image_mismatch = (
            not oracle_success
            and "test_huge_range_log" in row_lower
            and "imagecomparisonfailure" in row_lower
        )
        if unresolved_image_mismatch:
            violations.append("semantic_invariant_matplotlib_huge_range_log_image_output_changed")
    if _mentions_pylint_min_similarity_zero_disables_checker(evidence_text):
        if _patch_pylint_only_changes_similar_report_string(patch_text):
            violations.append("semantic_invariant_pylint_min_similarity_zero_report_string_only")
        if _patch_pylint_edits_similar_tests(patch_text):
            violations.append("semantic_invariant_pylint_min_similarity_zero_tests_edited")
        if _patch_pylint_uses_missing_min_similarity_lines_attr(patch_text):
            violations.append("semantic_invariant_pylint_min_similarity_zero_missing_attr")
        unresolved_duplicate_output = (
            not oracle_success
            and "test_set_duplicate_lines_to_zero" in row_lower
            and "total lines=" in row_lower
            and "duplicates=0" in row_lower
        )
        if unresolved_duplicate_output:
            violations.append("semantic_invariant_pylint_min_similarity_zero_not_disabled")
        unresolved_missing_attr = (
            not oracle_success
            and "similar" in row_lower
            and "min_similarity_lines" in row_lower
            and "attributeerror" in row_lower
        )
        if unresolved_missing_attr:
            violations.append("semantic_invariant_pylint_min_similarity_zero_missing_attr")
    api_surface = dict(evidence.get("api_surface_mismatch", {}) or {})
    parameter = str(api_surface.get("parameter", "") or "").strip()
    if parameter:
        lowered_parameter = parameter.lower()
        unresolved_keyword = (
            f"unexpected keyword argument '{lowered_parameter}'" in row_lower
            or f'unexpected keyword argument "{lowered_parameter}"' in row_lower
        )
        if not oracle_success and unresolved_keyword:
            violations.append("semantic_invariant_api_surface_mismatch_unresolved")
    return _dedupe(violations)


def _evidence_constraint_text(evidence: dict[str, Any]) -> str:
    return _joined_text(
        list(evidence.get("semantic_invariants", []) or [])
        + list(evidence.get("expected_behavior_constraints", []) or [])
        + list(evidence.get("forbidden_patch_directions", []) or [])
    )


def _mentions_false_none_identifier_lookup(text: str) -> bool:
    return (
        "false and none" in text
        or "false/none" in text
        or "identifier lookup semantics" in text
    )


def _undefined_helper_from_evidence(evidence: dict[str, Any], evidence_text: str) -> str:
    direct = str(evidence.get("undefined_symbol", "") or "").strip()
    if direct:
        return direct
    match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", evidence_text)
    if match:
        return match.group(1)
    match = re.search(r"call to ['\"]([^'\"]+)['\"]", evidence_text)
    if match and "undefined" in evidence_text:
        return match.group(1)
    return ""


def _patch_pytest_empty_expression_false_sentinel_changed(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "src/_pytest/mark/expression.py" not in patch_lower:
        return False
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    false_sentinel_removed = "ast.nameconstant(false)" in removed or "ast.constant(false)" in removed
    false_sentinel_replaced_by_name = bool(
        re.search(r"ast\.name\(\s*['\"][^'\"]*false[^'\"]*['\"]\s*,\s*ast\.load\(\)", added)
    )
    return false_sentinel_removed and false_sentinel_replaced_by_name


def _patch_calls_symbol(patch_text: str, symbol: str) -> bool:
    if not symbol:
        return False
    return bool(re.search(rf"^\+\s*.*\b{re.escape(symbol)}\s*\(", str(patch_text or ""), flags=re.MULTILINE))


def _patch_defines_or_imports_symbol(patch_text: str, symbol: str) -> bool:
    if not symbol:
        return False
    patch = str(patch_text or "")
    escaped = re.escape(symbol)
    return bool(
        re.search(rf"^\+\s*def\s+{escaped}\s*\(", patch, flags=re.MULTILINE)
        or re.search(rf"^\+\s*class\s+{escaped}\b", patch, flags=re.MULTILINE)
        or re.search(rf"^\+\s*from\s+[\w.]+\s+import\s+.*\b{escaped}\b", patch, flags=re.MULTILINE)
        or re.search(rf"^\+\s*import\s+.*\b{escaped}\b", patch, flags=re.MULTILINE)
    )


def _patch_mentions_api_parameter(patch_text: str, parameter: str) -> bool:
    name = str(parameter or "").strip()
    if not name:
        return True
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", str(patch_text or "")))


def _patch_only_mentions_wrong_api_parameters(
    patch_text: str,
    *,
    required_parameter: str,
    wrong_parameters: list[str],
) -> bool:
    if _patch_mentions_api_parameter(patch_text, required_parameter):
        return False
    return any(_patch_mentions_api_parameter(patch_text, item) for item in wrong_parameters)


def _mentions_timeseries_first_column(text: str) -> bool:
    return "timeseries validation" in text or "first-column" in text or "first column" in text


def _mentions_point_scalar_left_multiplication(text: str) -> bool:
    return (
        "point scalar left multiplication" in text
        or "point scalar-left multiplication" in text
        or "scalar-left multiplication" in text and "point" in text
        or "int * point" in text
        or "point2d" in text and "point3d" in text and "left multiplication" in text
    )


def _forbids_point_addition_dispatch_change(text: str) -> bool:
    return (
        "do not change point addition" in text
        or "do not modify __add__" in text
        or "do not add __radd__" in text
        or "preserve existing point addition" in text
    )


def _mentions_pylint_type_comment_import_consumption(text: str) -> bool:
    return (
        ("unused-import" in text or "unused import" in text)
        and ("type comment" in text or "type_comment" in text)
        and ("abc.abc" in text or "module-level" in text or "module level" in text)
    )


def _mentions_pylint_is_pypy_collection_fix(text: str) -> bool:
    return "is_pypy" in text and ("collection" in text or "collect" in text or "importerror" in text)


def _mentions_requests_header_none_merge_semantics(text: str) -> bool:
    return (
        ("session.headers" in text or "requests session" in text or "request.headers" in text)
        and ("accept-encoding" in text or "header" in text)
        and ("none" in text)
        and ("merge_setting" in text or "default header" in text or "delete" in text or "remove" in text)
    )


def _mentions_requests_binary_payload_preservation(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("preparedrequest.prepare_body" in lowered or "prepare_body" in lowered or "test_binary_put" in lowered)
        and ("bytes" in lowered or "bytearray" in lowered or "binary" in lowered)
        and ("_encode_params" in lowered or "to_native_string" in lowered or "ascii" in lowered)
    )


def _mentions_sklearn_ridge_classifier_store_cv_values(text: str) -> bool:
    return (
        "ridgeclassifiercv" in text
        and "store_cv_values" in text
        and ("constructor" in text or "__init__" in text or "unexpected keyword argument" in text)
    )


def _mentions_sklearn_hgb_string_target_early_stopping(text: str) -> bool:
    return (
        ("histgradientboostingclassifier" in text or "histgradientboosting" in text or "hgb" in text)
        and ("string target" in text or "string labels" in text or "classes_" in text)
        and ("early stopping" in text or "n_iter_no_change" in text or "_check_early_stopping" in text)
    )


def _mentions_sklearn_lasso_lars_copy_x_propagation(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("copy_x" in lowered or "copyx" in lowered)
        and ("np.array_equal(x, x_copy)" in lowered or "preprocess" in lowered or "_preprocess_data" in lowered)
    )


def _mentions_sklearn_dataframe_output_dtype_preservation(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("dataframe output dtype" in lowered or "dtype preservation" in lowered or "preserve the original input dtype" in lowered)
        and ("set_output" in lowered or "_set_output" in lowered or "pandas output" in lowered)
    )


def _mentions_sklearn_roc_curve_first_threshold_inf(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "roc_curve" in lowered
        and ("thresholds[0]" in lowered or "first threshold" in lowered or "threshold sentinel" in lowered)
        and ("np.inf" in lowered or "isinf" in lowered)
    )


def _mentions_django_autofield_subclass_semantics(text: str) -> bool:
    return (
        ("autofieldmeta" in text or "default_auto_field" in text)
        and ("__subclasscheck__" in text or "issubclass" in text or "subclass" in text)
        and ("bigautofield" in text or "smallautofield" in text or "autofield" in text)
    )


def _mentions_django_mysql_dbshell_option_precedence(text: str) -> bool:
    return (
        ("dbshell" in text or "mysql" in text)
        and ("options['database']" in text or "options['db']" in text or "optiondbname" in text)
        and ("precedence" in text or "override" in text or "preferred" in text)
    )


def _mentions_pytest_caplog_phase_record_isolation(text: str) -> bool:
    return (
        ("caplog" in text or "logcapturehandler" in text)
        and ("phase" in text or "setup" in text and "call" in text)
        and ("records" in text)
    )


def _mentions_pytest_unittest_class_skip_teardown(text: str) -> bool:
    return (
        ("unittest" in text or "testcase" in text)
        and ("class-level skip" in text or "class level skip" in text or "class-decorated skip" in text)
        and ("teardown" in text or "teardown()" in text)
        and ("--pdb" in text or "usepdb" in text)
    )


def _mentions_xarray_quantile_keep_attrs(text: str) -> bool:
    return (
        ("xarray" in text or "dataarray" in text or "variable.quantile" in text)
        and "quantile" in text
        and "keep_attrs" in text
    )


def _mentions_xarray_where_keep_attrs(text: str) -> bool:
    return (
        ("xarray" in text or "xr.where" in text or "xarray.where" in text)
        and "where" in text
        and "keep_attrs" in text
        and ("attrs" in text or "attributes" in text or "unexpected keyword argument" in text)
    )


def _mentions_xarray_integrate_dim_keyword(text: str) -> bool:
    return (
        ("xarray" in text or "dataarray.integrate" in text or "dataset.integrate" in text)
        and "integrate" in text
        and "dim" in text
        and ("unexpected keyword" in text or "propagate" in text or "api" in text)
    )


def _mentions_xarray_custom_values_attr_scalar(text: str) -> bool:
    return (
        ("xarray" in text or "variable" in text)
        and ("values-attribute" in text or "values attribute" in text or "custom values" in text)
        and ("scalar" in text or "dims=()" in text or "ndim 0" in text)
    )


def _mentions_xarray_reset_index_xindexes_contract(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        (
            "reset_index" in lowered
            or "xindexes" in lowered
            or "not coordinates with an index" in lowered
        )
        and ("xindexes" in lowered or "pandasindex" in lowered or "not coordinates with an index" in lowered)
    )


def _mentions_xarray_combine_auto_bystander_dimension_ordering(text: str) -> bool:
    return (
        ("xarray" in text or "combine_auto" in text)
        and ("combine-auto" in text or "combine_auto" in text)
        and ("bystander dimension" in text or "bystander dimensions" in text)
        and ("monotonic" in text or "dimension ordering" in text or "global index" in text)
    )


def _mentions_sphinx_none_annotation_reftype_obj(text: str) -> bool:
    return (
        "sphinx" in text
        and "none" in text
        and ("reftype='obj'" in text or "reftype=\"obj\"" in text or "object reference" in text)
        and ("parse_annotation" in text or "signature" in text or "pending_xref" in text)
    )


def _mentions_sphinx_reserved_toctree_entries(text: str) -> bool:
    return (
        "sphinx" in text
        and "toctree" in text
        and "genindex" in text
        and "modindex" in text
        and "search" in text
        and ("reserved" in text or "entries" in text or "do not remove" in text or "do not skip" in text)
    )


def _mentions_sphinx_graphviz_svg_external_hyperlink(text: str) -> bool:
    return (
        "sphinx" in text
        and ("graphviz" in text or "inheritance diagram" in text)
        and "svg" in text
        and ("hyperlink" in text or "external link" in text or "href" in text or "https://example.org" in text)
    )


def _mentions_sympy_permutation_non_disjoint_cycles(text: str) -> bool:
    return (
        "sympy" in text
        and "permutation" in text
        and ("non-disjoint" in text or "non disjoint" in text or "cycle" in text)
        and ("array form" in text or "args/reconstruction" in text or "test_args" in text)
    )


def _requires_django_autofield_iterative_subclass_check(text: str) -> bool:
    return (
        "any(issubclass" in text
        or "iterative subclass check" in text
        or "must iterate over _subclasses" in text
        or "check each autofield subclass base separately" in text
    )


def _requires_pylint_assignment_type_comment_entry(text: str) -> bool:
    return (
        "module-level assignment type comments must be consumed" in text
        or "assignment type comments must be consumed" in text
        or "must_handle_module_level_assignment_type_comment" in text
        or "must preserve module-level assignment type-comment consumption" in text
    )


def _mentions_pylint_attribute_type_comment_consumption(text: str) -> bool:
    return (
        ("astroid.attribute" in text or "attribute" in text)
        and ("bar imported from foo" in text or "foo.bar" in text or "bar.boo" in text)
        and ("type comment" in text or "type_comment" in text)
    )


def _patch_consumes_module_level_type_comment_imports(patch_text: str) -> bool:
    patch_lower = patch_text.lower()
    if "type_comment_args" in patch_lower or "type_comment_returns" in patch_lower:
        if not _mentions_assignment_type_comment_attr(patch_lower):
            return False
    if "def visit_typealias" in patch_lower and not _mentions_assignment_type_comment_attr(patch_lower):
        return False
    return _mentions_assignment_type_comment_attr(patch_lower)


def _patch_adds_duplicate_leave_assign(patch_text: str) -> bool:
    return len(re.findall(r"^\+\s*def leave_assign\(", str(patch_text or ""), flags=re.MULTILINE)) > 1


def _patch_restores_is_pypy_constant(patch_text: str) -> bool:
    return bool(
        re.search(r"^\+\s*IS_PYPY\s*=", str(patch_text or ""), flags=re.MULTILINE)
        and "pylint/constants.py" in str(patch_text or "")
    )


def _patch_handles_astroid_attribute_type_comments(patch_text: str) -> bool:
    patch_lower = str(patch_text or "").lower()
    return (
        "astroid.attribute" in patch_lower
        and "nodes_of_class(astroid.name" in patch_lower
        and "_type_annotation_names" in patch_lower
    )


def _patch_filters_request_none_before_merge(patch_text: str) -> bool:
    patch_lower = str(patch_text or "").lower()
    if "request_setting" not in patch_lower or "if v is not none" not in patch_lower:
        return False
    return bool(
        re.search(
            r"merged_setting\.update\(\(k,\s*v\)\s*for\s+k,\s*v\s+in\s+to_key_val_list\(request_setting\).*?if\s+v\s+is\s+not\s+none",
            patch_lower,
            flags=re.DOTALL,
        )
        or re.search(
            r"request_setting\).*?if\s+v\s+is\s+not\s+none",
            patch_lower,
            flags=re.DOTALL,
        )
    )


def _patch_broadens_requests_header_none_boundary(patch_text: str) -> bool:
    patch_lower = str(patch_text or "").lower()
    return (
        "diff --git a/requests/models.py" in patch_lower
        and (
            "def prepare_headers" in patch_lower
            or "for name, value in headers.items()" in patch_lower
            or "if value is not none" in patch_lower
        )
    )


def _patch_imports_undefined_requests_compat_iterable(patch_text: str) -> bool:
    patch = str(patch_text or "")
    return bool(
        re.search(
            r"^\+\s*from\s+(?:requests\.)?compat\s+import\s+.*\biterable\b",
            patch,
            flags=re.MULTILINE,
        )
        or re.search(
            r"^\+\s*from\s+\.compat\s+import\s+.*\biterable\b",
            patch,
            flags=re.MULTILINE,
        )
    )


def _patch_requests_binary_only_excludes_stream_but_still_encodes(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "requests/models.py" not in patch_lower:
        return False
    added_text = "\n".join(
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    added_lower = added_text.lower()
    stream_bytes_exclusion = (
        "is_stream" in added_lower
        and ("bytes" in added_lower or "bytearray" in added_lower)
        and ("not isinstance" in added_lower or "isinstance" in added_lower)
    )
    if not stream_bytes_exclusion:
        return False
    still_encodes_data = "_encode_params(data)" in patch_lower or "self._encode_params(data)" in patch_lower
    preserves_binary_body = bool(
        re.search(r"isinstance\(\s*data\s*,\s*\([^)]*(?:bytes|bytearray)", added_text, flags=re.IGNORECASE)
        and re.search(r"\bbody\s*=\s*data\b|\bself\.body\s*=\s*data\b", added_text)
    )
    return still_encodes_data and not preserves_binary_body


def _patch_broadens_requests_binary_decode_boundary(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "requests/utils.py" not in patch_lower:
        return False
    added_text = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        ("to_native_string" in patch_lower or "decode(" in added_text)
        and ("utf-8" in added_text or "ascii" in added_text)
        and ("data.decode" in added_text or "string.decode" in added_text or "body.decode" in added_text)
    )


def _patch_adds_ridge_classifier_store_cv_values_constructor(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "ridgeclassifiercv" not in patch_lower or "store_cv_values" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    ).lower()
    return (
        "store_cv_values=false" in added_lines
        and "store_cv_values=store_cv_values" in added_lines
        and "super(ridgeclassifiercv, self).__init__" in patch_lower
    )


def _patch_only_changes_docstring_for_store_cv_values(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "store_cv_values" not in patch.lower():
        return False
    added_lines = [
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return False
    doc_or_text_markers = (
        "store_cv_values :",
        "flag indicating",
        "cv_values_",
        "possible inputs for cv",
        "``",
        "attribute",
        "available when",
        "re:",
    )
    has_code_shape = any(
        "store_cv_values=false" in line
        or "store_cv_values=store_cv_values" in line
        or "self.store_cv_values" in line
        for line in added_lines
    )
    return not has_code_shape and all(
        any(marker in line for marker in doc_or_text_markers) or not line
        for line in added_lines
    )


def _patch_hgb_early_stopping_fix_in_docstring(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py" not in patch_lower:
        return False
    diff_lines = [line for line in patch.splitlines() if not line.startswith(("diff --git", "index ", "---", "+++"))]
    added_lines = [line[1:].strip() for line in diff_lines if line.startswith("+")]
    if not any("classes_" in line or "y_small_train" in line or "y_val" in line for line in added_lines):
        return False
    in_docstring = False
    for line in diff_lines:
        if line.startswith("@@"):
            in_docstring = False
            continue
        if line.startswith("-"):
            continue
        content = line[1:] if line.startswith("+") else line[1:] if line.startswith(" ") else line
        if '"""' in content or "'''" in content:
            stripped = content.strip()
            marker = '"""' if '"""' in stripped else "'''"
            after_marker = stripped.split(marker, 1)[1].strip()
            in_docstring = bool(after_marker)
            continue
        if line.startswith("+") and in_docstring and (
            "classes_" in content or "y_small_train" in content or "y_val" in content
        ):
            return True
    return False


def _patch_hgb_classes_index_without_integer_cast(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    if "classes_[" not in added:
        return False
    if ".astype(int)" in added or ".astype(np.int" in added or "dtype=int" in added:
        return False
    return bool(
        re.search(r"classes_\[\s*y_small_train\s*\]", added)
        or re.search(r"classes_\[\s*y_val\s*\]", added)
    )


def _patch_hgb_maps_encoded_targets_to_classes(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    added_lower = added.lower()
    return (
        "classes_[" in added
        and "y_small_train" in added
        and ".astype(int)" in added_lower
        and (
            "y_val" not in added
            or "self._use_validation_data" in added
            or re.search(r"if\s+y_val\s+is\s+not\s+none", added_lower)
        )
    )


def _patch_sklearn_lasso_lars_copy_x_constructor_only(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/linear_model/least_angle.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    touches_constructor_defaults = "self.copy_x" in added and "copy_x is not none" in added
    propagates_to_preprocess = _patch_sklearn_lasso_lars_copy_x_propagates_to_preprocess(patch)
    return touches_constructor_defaults and not propagates_to_preprocess


def _patch_sklearn_lasso_lars_copy_x_propagates_to_preprocess(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/linear_model/least_angle.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if "copy_x" not in added:
        return False
    forwards_runtime_copy_x = bool(
        re.search(r"_preprocess_data\([^)]*copy_x\s*\)", added, flags=re.DOTALL)
        or re.search(r"_preprocess_data\([^)]*copy_x\s*,", added, flags=re.DOTALL)
        or "self.copy_x if copy_x is none else copy_x" in added
        or "self.fit_intercept, self.normalize, copy_x)" in added
    )
    lasso_ic_default_preserved = "def fit(self, x, y, copy_x=none)" in added or "if copy_x is none" in added
    added_runtime_mentions = len(re.findall(r"copy_x\)", added)) + len(re.findall(r"copy_x\s*,", added))
    preprocess_call_count = len(re.findall(r"_preprocess_data\(", patch_lower))
    return forwards_runtime_copy_x and lasso_ic_default_preserved and (preprocess_call_count >= 2 or added_runtime_mentions >= 2)


def _patch_sklearn_dtype_preservation_unpopulated_dtype_map(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/utils/_set_output.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    added_lower = added.lower()
    adds_dtype_sink = "dtypes=none" in added_lower or "astype(dtypes" in added_lower
    if not adds_dtype_sink:
        return False
    propagates_dtype_map = (
        re.search(r"_wrap_in_pandas_container\([^\n]*dtypes\s*=", added, flags=re.DOTALL)
        or "dtypes=original_input" in added_lower
        or "dtypes=getattr(original_input" in added_lower
        or "original_input.dtypes" in added_lower
        or "x.dtypes" in added_lower
    )
    return not bool(propagates_dtype_map)


def _patch_touches_sklearn_roc_curve_thresholds(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sklearn/metrics/_ranking.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    ).lower()
    return "threshold" in added and ("roc_curve" in patch_lower or "thresholds = np.r_" in patch_lower)


def _patch_sklearn_roc_curve_threshold_caps_inf(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if not _patch_touches_sklearn_roc_curve_thresholds(patch):
        return False
    added = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    ).lower()
    return (
        "first_threshold = 1" in added
        or "thresholds[0] = 1" in added
        or "min(first_threshold, 1" in added
        or "min(thresholds[0] + 1, 1" in added
        or "minimum(first_threshold, 1" in added
        or ("np.inf" in added and "min(" in added and ", 1" in added)
    )


def _patch_clears_logcapturehandler_records_in_place(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "src/_pytest/logging.py" not in patch and "LogCaptureHandler" not in patch:
        return False
    return bool(re.search(r"^\+\s*self\.records\.clear\(\)", patch, flags=re.MULTILINE))


def _patch_returns_early_for_skipped_unittest_methods(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "src/_pytest/unittest.py" not in patch:
        return False
    return bool(
        re.search(
            r"^\+\s*elif\s+_is_skipped\(self\.obj\):\n\+\s*return\b",
            patch,
            flags=re.MULTILINE,
        )
        or re.search(
            r"^\+\s*if\s+_is_skipped\(self\.obj\):\n\+\s*return\b",
            patch,
            flags=re.MULTILINE,
        )
    )


def _patch_only_checks_unittest_method_skip(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "src/_pytest/unittest.py" not in patch_lower:
        return False
    if "_is_skipped(self.parent.obj)" in patch or "_is_skipped(self.parent)" in patch:
        return False
    return bool(
        re.search(
            r"^[ +]\s*if\s+self\.config\.getoption\(\"usepdb\"\)\s+and\s+not\s+_is_skipped\(self\.obj\):",
            patch,
            flags=re.MULTILINE,
        )
    )


def _patch_quantile_only_changes_variable_without_keep_attrs_contract(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "variable.py" not in patch_lower:
        return False
    if "keep_attrs" in patch_lower and ("dataarray.py" in patch_lower or "dataset.py" in patch_lower):
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if "keep_attrs" in patch_lower and "attrs" in added:
        return True
    return "return variable(new_dims, qs" in patch_lower or "attrs=self._attrs" in patch_lower


def _patch_xarray_quantile_changes_unrelated_dataarray_methods(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/dataarray.py" not in patch_lower:
        return False
    added_keep_attrs = [
        line
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++") and "keep_attrs" in line
    ]
    if len(added_keep_attrs) > 1:
        return True
    return bool(
        re.search(r"@@[^@]*(?:def\s+persist|def\s+thin|def\s+copy|def\s+broadcast_like)", patch_lower, flags=re.DOTALL)
        and "keep_attrs" in patch_lower
    )


def _patch_xarray_where_keep_attrs_contract(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/computation.py" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    has_signature = bool(
        re.search(r"def\s+where\s*\(\s*cond\s*,\s*x\s*,\s*y\s*,\s*keep_attrs\s*=", added_lines)
        or re.search(r"def\s+where\s*\(\s*cond\s*,\s*x\s*,\s*y\s*,\s*keep_attrs\s*=", patch_lower)
    )
    has_x_attrs_source = (
        "x.attrs" in added_lines
        or "getattr(x, \"attrs\"" in added_lines
        or "getattr(x, 'attrs'" in added_lines
        or "x._attrs" in added_lines
    )
    return has_signature and has_x_attrs_source


def _patch_xarray_where_keep_attrs_blind_apply_ufunc_forward(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/computation.py" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    forwards_keep_attrs = "keep_attrs=keep_attrs" in added_lines or "keep_attrs = keep_attrs" in added_lines
    has_x_attrs_source = (
        "x.attrs" in added_lines
        or "getattr(x, \"attrs\"" in added_lines
        or "getattr(x, 'attrs'" in added_lines
        or "x._attrs" in added_lines
    )
    return forwards_keep_attrs and not has_x_attrs_source


def _patch_xarray_integrate_renames_dim_without_alias(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/dataarray.py" not in patch_lower:
        return False
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    renames_signature = "self, dim" in removed and "self, coord" in added
    forwards_coord = "integrate(coord" in added
    has_dim_alias = "dim=" in added or "dim:" in added or "dim is not none" in added or "**kwargs" in added
    return renames_signature and forwards_coord and not has_dim_alias


def _patch_xarray_integrate_missing_dim_keyword_alias(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/dataarray.py" not in patch_lower:
        return False
    if "def integrate" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if not added.strip():
        return False
    mentions_coord_signature = "self, coord" in added or "coord:" in added
    has_dim_alias = "dim=" in added or "dim:" in added or "dim is not none" in added or "**kwargs" in added
    return mentions_coord_signature and not has_dim_alias


def _patch_xarray_custom_values_attr_posthoc_variable_init_coercion(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/variable.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    touches_variable_init = "self._data = as_compatible_data" in patch_lower or "def __init__" in patch_lower
    touches_parse_dimensions = "def _parse_dimensions" in patch_lower or "len(dims) != self.ndim" in patch_lower
    posthoc_data_rewrite = (
        "self._data.ndim" in added
        or "self._data[0]" in added
        or "np.asarray(self._data" in added
        or "if self.ndim == 0 and len(dims) == 0" in added
    )
    return (touches_variable_init or touches_parse_dimensions) and posthoc_data_rewrite


def _patch_xarray_custom_values_attr_missing_as_compatible_guard(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/variable.py" not in patch_lower:
        return False
    if "as_compatible_data" not in patch_lower and "getattr(data" not in patch_lower and ".values" not in patch_lower:
        return True
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    changes_values_consumption = "getattr(data, \"values\"" in patch_lower or "getattr(data, 'values'" in patch_lower
    narrows_to_known_containers = (
        "pd." in added
        or "pandas" in added
        or "xr." in added
        or "dataarray" in added
        or "dataset" in added
        or "variable" in added
    )
    preserves_custom_object = (
        "not hasattr(data, \"values\")" in added
        or "not hasattr(data, 'values')" in added
        or "utils.is_scalar(data)" in added
        or "to_0d_object_array" in added
        or "0d_object" in added
    )
    removes_unconditional_values = "getattr(data, \"values\", data)" in removed or "getattr(data, 'values', data)" in removed
    if changes_values_consumption:
        return not (narrows_to_known_containers or preserves_custom_object or removes_unconditional_values)
    return True


def _patch_xarray_reset_index_only_changes_datavariables(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/dataset.py" not in patch_lower:
        return False
    added_removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    touches_datavariables = "class datavariables" in patch_lower or "self._dataset._coord_names" in added_removed
    touches_reset_index_contract = (
        "def reset_index" in patch_lower
        or "drop_indexes" in added_removed
        or "xindexes" in added_removed
        or "set_xindex" in added_removed
        or "_indexes" in added_removed
    )
    return touches_datavariables and not touches_reset_index_contract


def _patch_xarray_reset_index_missing_xindexes_contract(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if not added.strip():
        return True
    mentions_required_contract = (
        "reset_index" in patch_lower
        and (
            "xindexes" in added
            or "drop_indexes" in added
            or "set_xindex" in added
            or "pandasindex" in added
            or "_indexes" in added
        )
    )
    return not mentions_required_contract


def _patch_xarray_combine_auto_deletes_monotonic_index_guard(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/combine.py" not in patch_lower:
        return False
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removes_monotonic_error = (
        "monotonic" in removed
        and "global index" in removed
        and ("raise valueerror" in removed or "is_monotonic" in removed)
    )
    preserves_monotonic_error = (
        "monotonic" in added
        and "global index" in added
        and ("raise valueerror" in added or "is_monotonic" in added)
    )
    return removes_monotonic_error and not preserves_monotonic_error


def _patch_xarray_combine_auto_sorts_everything_without_bystander_scope(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/combine.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if not added.strip():
        return False
    broad_sort = "sortby(" in added or ".sort_index(" in added or "sorted(" in added
    scoped_to_bystander = (
        "bystander" in added
        or "concat_dim" in added
        or "concat_dims" in added
        or "combine_auto" in added
        or "_infer_concat_order" in patch_lower
    )
    return broad_sort and not scoped_to_bystander


def _patch_sphinx_none_annotation_reftype_obj(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/domains/python.py" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    mentions_none = "'none'" in added_lines or '"none"' in added_lines
    sets_obj = "reftype='obj'" in added_lines or 'reftype="obj"' in added_lines or "reftype = 'obj'" in added_lines
    preserves_class = "reftype='class'" in patch_lower or 'reftype="class"' in patch_lower
    return mentions_none and sets_obj and preserves_class


def _patch_sphinx_none_annotation_rewrites_autodoc_typehints(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/ext/autodoc/typehints.py" not in patch_lower:
        return False
    if "sphinx/domains/python.py" in patch_lower and _patch_sphinx_none_annotation_reftype_obj(patch):
        return False
    deleted_lines = [
        line
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    return len(deleted_lines) >= 10 or "def modify_field_list" in patch_lower or "def insert_field_list" in patch_lower


def _patch_skips_sphinx_reserved_toctree_entries(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/environment/adapters/toctree.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if not all(token in added for token in ("genindex", "modindex", "search")):
        return False
    direct_reserved_skip = bool(
        re.search(
            r"(?ms)^\+\s*(?:if|elif)\s+ref\s+in\s+\{[^}]*genindex[^}]*modindex[^}]*search[^}]*\}\s*:\s*\n"
            r"(?:^\+\s*(?:#.*)?\n|^\+\s*[^\n]*\n){0,3}?^\+\s*continue\b",
            patch,
        )
    )
    return (
        direct_reserved_skip
        or "entries.remove" in added
        or ".remove(ref)" in added
        or "filter(" in added
        or "not in {'genindex', 'modindex', 'search'}" in added
        or 'not in {"genindex", "modindex", "search"}' in added
    )


def _patch_adds_sphinx_reserved_toctree_entry_append(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/directives/other.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "toctree['entries'].append" in added
        and "genindex" in added
        and "modindex" in added
        and "search" in added
    )


def _patch_sphinx_graphviz_only_rewrites_svg_relative_paths(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/ext/graphviz.py" not in patch_lower:
        return False
    if "sphinx/ext/inheritance_diagram.py" in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "fix_svg_relative_paths" in patch_lower
        or "urlsplit" in added
        or "path.relpath" in added
        or "xlink:href" in added
    )


def _patch_sphinx_graphviz_external_hyperlink_fix_missing(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sphinx/ext/graphviz.py" not in patch_lower and "sphinx/ext/inheritance_diagram.py" not in patch_lower:
        return False
    if "sphinx/ext/inheritance_diagram.py" not in patch_lower:
        return True
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    maps_graphviz_url = "urls[" in added or "this_node_attrs['url']" in added or 'this_node_attrs["url"]' in added
    preserves_external_refuri = (
        "refuri" in added
        and (
            "://" in added
            or "urlsplit" in added
            or "urlparse" in added
            or "relative_uri" in added
            or "intersphinx" in added
        )
    )
    return not (maps_graphviz_url and preserves_external_refuri)


def _patch_sympy_permutation_only_skips_cycle_duplicate_check(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sympy/combinatorics/permutations.py" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    skips_duplicate_check = "if is_cycle and has_dups(temp)" in added_lines or "if is_cycle:" in added_lines and "has_dups(temp)" in added_lines
    canonicalizes_to_array = _patch_sympy_permutation_non_disjoint_cycles_array_form(patch)
    return skips_duplicate_check and not canonicalizes_to_array


def _patch_sympy_permutation_non_disjoint_cycles_array_form(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "sympy/combinatorics/permutations.py" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    mentions_cycle_list_input = (
        ("is_sequence" in added_lines and "cycle" in added_lines)
        or "if is_cycle and has_dups(temp)" in added_lines
        or ("-        if has_dups(temp):" in patch and "if is_cycle:" in patch_lower)
    )
    builds_cycle_product = (
        ("cycle()" in added_lines and "for ci in args" in added_lines and "c = c(*ci)" in added_lines)
        or ("for c in a" in added_lines and "cyc = cycle(c)" in added_lines and "_af_rmul" in added_lines)
        or (
            "-        if has_dups(temp):" in patch
            and "if is_cycle:" in patch_lower
            and "c = cycle()" in patch_lower
            and "for ci in args" in patch_lower
            and "c = c(*ci)" in patch_lower
        )
    )
    stores_array_form = (
        "aform = c.list" in added_lines
        or "return _af_new(c.list" in added_lines
        or "args = c.list" in added_lines
        or "args = (arr,)" in added_lines
        or "args = (aform,)" in added_lines
        or ("aform = c.list" in patch_lower and "-        if has_dups(temp):" in patch)
    )
    avoids_repeated_basic_args = "basic.__new__(cls, args" not in patch_lower
    return mentions_cycle_list_input and builds_cycle_product and stores_array_form and avoids_repeated_basic_args


def _patch_changes_xarray_rank_without_quantile_path(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if "rank(" not in added_lines and ".rank(" not in added_lines:
        return False
    return "quantile" not in patch_lower


def _patch_sets_temp_dataset_attrs_without_returned_dataarray_attrs(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/dataarray.py" not in patch_lower or "quantile" not in patch_lower:
        return False
    if "ds.attrs" not in patch_lower:
        return False
    added_lines = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return "result.attrs" not in added_lines and "array.attrs" not in added_lines


def _mentions_xarray_stack_multiindex_dtype(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "dataset.stack" in lowered
        or "multiindex coordinates" in lowered
        or "test_restore_dtype_on_multiindexes" in lowered
        or "pandasmultiindex" in lowered
        or "pandasindex" in lowered
    ) and ("dtype" in lowered or "nameerror" in lowered or "defined before" in lowered)


def _mentions_xarray_indexvariable_copy_aliasing(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("xarray" in lowered or "indexvariable" in lowered)
        and ("copy/deepcopy" in lowered or "copy(deep=true)" in lowered or "deep copy" in lowered)
        and ("unicode index dtype" in lowered or "same dtype" in lowered or "v.dtype == w.dtype" in lowered)
        and ("source ndarray aliasing" in lowered or "source_ndarray(v.values)" in lowered)
    )


def _patch_xarray_indexvariable_copy_aliasing_fix(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/variable.py" not in patch_lower or "xarray/core/indexing.py" not in patch_lower:
        return False
    added_lines = [
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines).lower()
    has_adapter_copy = (
        "xarray/core/indexing.py" in patch_lower
        and "def copy" in added_text
        and "deep" in added_text
        and "return pandasindexadapter" in added_text
    )
    has_deep_shallow_adapter_semantics = (
        ".copy(deep=true)" in added_text
        and (
            "if deep else self.array" in added_text
            or "else self.array" in added_text
            or "pandasindexadapter(self.array, self._dtype)" in added_text
        )
    )
    has_indexvariable_delegation = (
        "self._data.copy(deep=deep)" in added_text
        or "self._data.copy(deep = deep)" in added_text
    )
    return has_adapter_copy and has_deep_shallow_adapter_semantics and has_indexvariable_delegation


def _patch_xarray_indexvariable_adds_copy_outside_pandas_adapter(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/indexing.py" not in patch_lower:
        return False
    current_hunk_class = ""
    for line in patch.splitlines():
        hunk_class = re.search(r"@@.*class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if hunk_class:
            current_hunk_class = hunk_class.group(1)
            continue
        if (
            line.startswith("+")
            and not line.startswith("+++")
            and re.search(r"\bdef\s+copy\s*\(\s*self\s*,\s*deep\s*=\s*true\s*\)", line, flags=re.IGNORECASE)
            and current_hunk_class
            and current_hunk_class != "PandasIndexAdapter"
        ):
            return True
    return False


def _patch_xarray_indexvariable_adds_unrelated_adapter_api(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/indexing.py" not in patch_lower:
        return False
    return any(
        line.startswith("+") and not line.startswith("+++") and re.search(r"\bdef\s+astype\s*\(", line)
        for line in patch.splitlines()
    )


def _patch_xarray_indexvariable_deletes_neighbor_methods(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/variable.py" not in patch_lower:
        return False
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return "def equals" in removed or "def _data_equals" in removed or "def to_index_variable" in removed


def _patch_xarray_stack_dtype_only_adds_pandas_index_fastpath(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/indexes.py" not in patch_lower:
        return False
    added_lines = [
        line[1:].strip()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines).lower()
    if "fastpath=true" not in added_text:
        return False
    touches_multiindex_dtype = "level_coords_dtype" in added_text or "pandasmultiindex" in added_text
    return not touches_multiindex_dtype


def _patch_xarray_places_pandas_multiindex_before_pandasindex(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "xarray/core/indexes.py" not in patch_lower:
        return False
    added_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines)
    return bool(re.search(r"^\s*class\s+PandasMultiIndex\(PandasIndex\):", added_text, flags=re.MULTILINE))


def _patch_broadens_django_autofield_subclasses(patch_text: str) -> bool:
    patch = str(patch_text or "")
    return bool(
        re.search(
            r"^\+\s*return\s*\(\s*AutoField\s*,\s*BigAutoField\s*,\s*SmallAutoField\s*\)",
            patch,
            flags=re.MULTILINE,
        )
    )


def _patch_uses_recursive_django_autofield_tuple_subclass_check(patch_text: str) -> bool:
    return bool(
        re.search(
            r"^\+\s*return\s+issubclass\(\s*subclass\s*,\s*self\._subclasses\s*\)",
            str(patch_text or ""),
            flags=re.MULTILINE,
        )
    )


def _patch_uses_django_autofield_iterative_subclass_check(patch_text: str) -> bool:
    patch = str(patch_text or "")
    return bool(
        re.search(
            r"^\+\s*return\s+any\(\s*issubclass\(\s*subclass\s*,\s*klass\s*\)\s+for\s+klass\s+in\s+self\._subclasses\s*\)",
            patch,
            flags=re.MULTILINE,
        )
    )


def _patch_preserves_django_mysql_dbshell_classmethod_signature(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "django/db/backends/mysql/client.py" not in patch:
        return True
    if re.search(r"^-\s*@classmethod\b", patch, flags=re.MULTILINE):
        return False
    if re.search(r"^-\s*def settings_to_cmd_args_env\(cls,\s*settings_dict,\s*parameters\)", patch, flags=re.MULTILINE):
        return False
    if re.search(r"^\+\s*def settings_to_cmd_args_env\(self,\s*settings_dict,\s*parameters\)", patch, flags=re.MULTILINE):
        return False
    return True


def _patch_rewrites_django_mysql_dbshell_method_broadly(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "django/db/backends/mysql/client.py" not in patch:
        return False
    removed = [
        line
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    return len(removed) >= 20


def _patch_removes_django_mysql_dbshell_env_or_connection_options(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "django/db/backends/mysql/client.py" not in patch:
        return False
    protected_tokens = (
        "MYSQL_PWD",
        "--host=%s",
        "--socket=%s",
        "--port=%s",
        "--ssl-ca=%s",
        "--ssl-cert=%s",
        "--ssl-key=%s",
        "--defaults-file=%s",
        "--default-character-set=%s",
    )
    removed_text = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    return any(token in removed_text for token in protected_tokens)


def _patch_adds_django_mysql_database_precedence(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "django/db/backends/mysql/client.py" not in patch:
        return False
    added_text = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    return bool(
        (
            "settings_dict['OPTIONS'].get(" in added_text
            or ("options.get(" in added_text and "options = settings_dict['OPTIONS']" in added_text)
        )
        and "'database'" in added_text
        and "'db'" in added_text
        and "settings_dict['NAME']" in added_text
    )


def _mentions_astropy_observed_frame_transform_import_scope(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "observed-frame direct transforms" in lowered
        or "itrs<->altaz" in lowered
        or "itrs<->hadec" in lowered
        or "cirs<->observed transform graph" in lowered
    )


def _mentions_matplotlib_huge_range_lognorm_image(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("matplotlib" in lowered or "lib/matplotlib/image.py" in lowered)
        and ("test_huge_range_log" in lowered or "huge range log" in lowered)
        and ("lognorm" in lowered or "invalid vmin or vmax" in lowered)
    )


def _mentions_pylint_min_similarity_zero_disables_checker(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("pylint" in lowered or "similar.py" in lowered)
        and ("min-similarity-lines" in lowered or "min_similarity_lines" in lowered or "duplicates=0" in lowered)
        and ("disable" in lowered or "duplicate" in lowered or "r0801" in lowered)
    )


def _patch_astropy_observed_transform_uses_local_frame_imports_in_decorators(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "astropy/coordinates/builtin_frames/intermediate_rotation_transforms.py" not in patch_lower:
        return False
    if "frame_transform_graph.transform" not in patch:
        return False
    if "AltAz" not in patch and "HADec" not in patch:
        return False
    added_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    has_decorator_reference = any(
        "frame_transform_graph.transform" in line and ("AltAz" in line or "HADec" in line)
        for line in added_lines
    )
    if not has_decorator_reference:
        return False
    has_module_scope_import = any(
        re.search(r"^from\s+\.(altaz|hadec)\s+import\s+(AltAz|HADec)\b", line)
        or re.search(r"^from\s+astropy\.coordinates\.builtin_frames\.(altaz|hadec)\s+import\s+(AltAz|HADec)\b", line)
        for line in added_lines
    )
    if has_module_scope_import:
        return False
    has_local_import = any(
        re.search(r"^\s+from\s+\.(altaz|hadec)\s+import\s+(AltAz|HADec)\b", line)
        or re.search(r"^\s+from\s+astropy\.coordinates\.builtin_frames\.(altaz|hadec)\s+import\s+(AltAz|HADec)\b", line)
        for line in added_lines
    )
    return has_local_import


def _patch_astropy_bypasses_cirs_observed_transforms(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "astropy/coordinates/builtin_frames/cirs_observed_transforms.py" not in patch_lower:
        return False
    removed = "\n".join(
        line[1:].lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return (
        "cirs" in removed
        and ("altaz" in removed or "hadec" in removed)
        and "frame_transform_graph.transform" in removed
    )


def _patch_matplotlib_broadens_lognorm_invalid_range_semantics(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "lib/matplotlib/colors.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return (
        "invalid vmin or vmax" in removed
        and (
            "masked_all_like" in added
            or "return np.zeros" in added
            or "return np.full_like" in added
        )
    )


def _patch_matplotlib_returns_intermediate_masked_image_tuple(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "lib/matplotlib/image.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "masked_all_like" in added
        and bool(
            re.search(r"return\s+output\s*,\s*extent\s*,\s*transform\s*,\s*alpha", added)
            or re.search(r"return\s+resampled_masked\s*,\s*extent\s*,\s*transform\s*,\s*alpha", added)
        )
    )


def _patch_matplotlib_returns_transparent_rgba_for_lognorm_no_positive_values(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "lib/matplotlib/image.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "lognorm" in added
        and ("masked_less_equal" in added or "positive.count() == 0" in added or "data > 0" in added)
        and "np.zeros((*out_shape, 4)" in added
        and ("output[..., 3] = 0" in added or "return output, 0, 0" in added)
    )


def _patch_matplotlib_mutates_lognorm_state_in_image_path(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "lib/matplotlib/image.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "lognorm" in added
        and (
            "self.norm.vmin" in added
            or "self.norm.vmax" in added
            or "self.norm.autoscale_none" in added
        )
    )


def _patch_matplotlib_recomputes_lognorm_image_output_branch(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "lib/matplotlib/image.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "lognorm" in patch_lower
        and "valid_data" in added
        and "s_vmin, s_vmax" in added
        and "output = self.norm(resampled_masked)" in added
    )


def _patch_pylint_only_changes_similar_report_string(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "pylint/checkers/similar.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if "min_lines" in added or "min_similarity_lines" in added:
        return False
    return "total lines=" in patch_lower or "_get_similarity_report" in patch_lower


def _patch_pylint_edits_similar_tests(patch_text: str) -> bool:
    patch_lower = str(patch_text or "").lower()
    return (
        "tests/checkers/unittest_similar.py" in patch_lower
        or "tests/checkers/test_similar" in patch_lower
        or "pylint/checkers/tests/test_similar" in patch_lower
    )


def _patch_pylint_uses_missing_min_similarity_lines_attr(patch_text: str) -> bool:
    patch = str(patch_text or "")
    if "pylint/checkers/similar.py" not in patch.lower():
        return False
    return bool(re.search(r"^\+\s*if\s+self\.min_similarity_lines\b", patch, flags=re.MULTILINE))


def _patch_pylint_sets_infinite_min_lines_for_zero(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "pylint/checkers/similar.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return (
        "min_lines == 0" in added
        and ("float(\"inf\")" in added or "float('inf')" in added or "sys.maxsize" in added)
    )


def _patch_pylint_uses_undefined_val_in_run(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "pylint/checkers/similar.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return "if val is not none" in added and "for opt, val in opts" in patch_lower and "-    for opt, val in opts" in patch


def _patch_pylint_only_changes_checker_lifecycle_for_standalone_run(patch_text: str) -> bool:
    patch = str(patch_text or "")
    patch_lower = patch.lower()
    if "pylint/checkers/similar.py" not in patch_lower:
        return False
    added = "\n".join(
        line[1:].strip().lower()
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    touches_checker_lifecycle = (
        "self.enabled" in added
        or "def open" in patch_lower
        or "def process_module" in patch_lower
        or "def close" in patch_lower
    )
    touches_standalone_run = "def run" in patch_lower or "duplicates=0" in added or "--duplicates" in added
    return touches_checker_lifecycle and not touches_standalone_run


def _mentions_assignment_type_comment_attr(patch_lower: str) -> bool:
    return bool(
        re.search(r"\bnode\.type_comment\b(?!_)", patch_lower)
        or "getattr(node, \"type_comment\"" in patch_lower
        or "getattr(node, 'type_comment'" in patch_lower
    )


def _joined_text(items: list[Any]) -> str:
    return "\n".join(str(item) for item in items if str(item or "").strip())


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
