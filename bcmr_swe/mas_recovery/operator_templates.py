"""Operator-template library for MAS-DX-R recovery planning.

The templates in this module are intentionally runtime-safe: matching uses
only pre-oracle fields from strict80 proposal/material records, trajectory
digests, and changed-file hints. Development oracle-green examples can support
template provenance, but they are never required as runtime features.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any


RUNTIME_FEATURE_KEYS = {
    "instance_id",
    "project",
    "changed_files",
    "problem_terms",
    "proposal_failure_type",
    "proposal_primary_failure_type",
    "proposal_run_scope",
    "proposal_recovery_action",
    "responsible_stage",
    "review_priority",
    "proposal_confidence",
}

ORACLE_FIELD_RE = re.compile(
    r"(oracle|gold|patch_diff|candidate_patch|ground_truth|test_patch|solution)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OperatorTemplate:
    template_id: str
    operator_name: str
    project_families: tuple[str, ...]
    changed_file_patterns: tuple[str, ...]
    problem_terms: tuple[str, ...]
    failure_type_hints: tuple[str, ...]
    run_scope: str
    selected_action: str
    operator_gate: str
    patch_contract: str
    expected_runtime_evidence: tuple[str, ...]
    development_support_instance_ids: tuple[str, ...]
    claim_boundary: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_operator_templates() -> list[OperatorTemplate]:
    """Return the current curated MAS-DX-R operator templates."""

    return [
        OperatorTemplate(
            template_id="sphinx_classmethod_property_descriptor",
            operator_name="class-property descriptor recovery",
            project_families=("sphinx-doc",),
            changed_file_patterns=(
                "sphinx/ext/autodoc/",
                "sphinx/util/inspect.py",
                "sphinx/domains/python.py",
            ),
            problem_terms=("classmethod", "property", "autodoc", "descriptor"),
            failure_type_hints=("verifier_acceptance_gap", "patch_incomplete", "handoff_information_loss"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_descriptor_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed Sphinx autodoc/inspect/python-domain source",
                "problem or trace mentions classmethod/property/descriptor behavior",
            ),
            development_support_instance_ids=("sphinx-doc__sphinx-9461",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=10,
        ),
        OperatorTemplate(
            template_id="sklearn_kmeans_center_ordering",
            operator_name="k-means center ordering invariant recovery",
            project_families=("scikit-learn",),
            changed_file_patterns=("sklearn/preprocessing/_discretization.py",),
            problem_terms=("kmeans", "k-means", "kbins", "bin", "center", "sort"),
            failure_type_hints=("patch_incomplete", "verifier_acceptance_gap"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_numerical_ordering_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed discretization source",
                "problem or trace mentions KMeans/KBins/bin edge ordering",
            ),
            development_support_instance_ids=("scikit-learn__scikit-learn-13135",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=20,
        ),
        OperatorTemplate(
            template_id="sklearn_mixture_final_estep_placement",
            operator_name="mixture final E-step placement recovery",
            project_families=("scikit-learn",),
            changed_file_patterns=("sklearn/mixture/",),
            problem_terms=("mixture", "e-step", "fit_predict", "predict", "converged", "label"),
            failure_type_hints=("patch_incomplete", "local_patch_regression"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_state_update_order_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed sklearn mixture source",
                "problem or trace mentions fit_predict/predict consistency or final E-step",
            ),
            development_support_instance_ids=("scikit-learn__scikit-learn-13142",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=21,
        ),
        OperatorTemplate(
            template_id="sklearn_logistic_no_refit_parameter_selection",
            operator_name="logistic no-refit parameter selection recovery",
            project_families=("scikit-learn",),
            changed_file_patterns=("sklearn/linear_model/",),
            problem_terms=("logistic", "refit", "cv", "coef", "parameter", "selection"),
            failure_type_hints=("patch_incomplete", "local_patch_regression"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_model_selection_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed sklearn linear_model source",
                "problem or trace mentions logistic/CV/refit/model-selection behavior",
            ),
            development_support_instance_ids=("scikit-learn__scikit-learn-14087",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=22,
        ),
        OperatorTemplate(
            template_id="sklearn_dataframe_preserving_feature_selection",
            operator_name="DataFrame-preserving feature selection recovery",
            project_families=("scikit-learn",),
            changed_file_patterns=("sklearn/feature_selection/",),
            problem_terms=("dataframe", "pandas", "feature", "selection", "column", "support"),
            failure_type_hints=("patch_incomplete", "verifier_acceptance_gap"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_container_preservation_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed sklearn feature_selection source",
                "problem or trace mentions pandas/DataFrame/feature names or columns",
            ),
            development_support_instance_ids=("scikit-learn__scikit-learn-25102",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=23,
        ),
        OperatorTemplate(
            template_id="sympy_homomorphism_relator_mapping",
            operator_name="homomorphism relator mapping recovery",
            project_families=("sympy",),
            changed_file_patterns=("sympy/combinatorics/", "sympy/groups/"),
            problem_terms=("homomorphism", "relator", "fpgroup", "presentation", "permutation"),
            failure_type_hints=("patch_incomplete", "handoff_information_loss", "verifier_acceptance_gap"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_algebraic_mapping_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed SymPy combinatorics/group source",
                "problem or trace mentions homomorphism/relator/presentation/permutation mapping",
            ),
            development_support_instance_ids=("sympy__sympy-24443",),
            claim_boundary="template derived from development verifier success; runtime match must not use oracle diff",
            priority=30,
        ),
        OperatorTemplate(
            template_id="sympy_permutation_cycle_composition",
            operator_name="permutation cycle composition recovery",
            project_families=("sympy",),
            changed_file_patterns=("sympy/combinatorics/permutations.py",),
            problem_terms=("permutation", "cycle", "non-disjoint", "identity", "josephus"),
            failure_type_hints=("handoff_information_loss", "patch_incomplete"),
            run_scope="patcher_fixed_localization",
            selected_action="template_guided_permutation_cycle_repatch",
            operator_gate="semantic_invariant_guarded_repatch",
            patch_contract="source_only",
            expected_runtime_evidence=(
                "changed SymPy permutation source",
                "problem or trace mentions non-disjoint cycles or permutation constructor behavior",
            ),
            development_support_instance_ids=("sympy__sympy-12481",),
            claim_boundary="template is development-supported but not yet oracle-green under current broad test_args protocol",
            priority=31,
        ),
        OperatorTemplate(
            template_id="sphinx_linkcheck_local_uri_protocol",
            operator_name="linkcheck local URI protocol recovery",
            project_families=("sphinx-doc",),
            changed_file_patterns=("sphinx/builders/linkcheck.py",),
            problem_terms=("linkcheck", "anchor", "local", "uri", "ftp", "mailto"),
            failure_type_hints=("handoff_information_loss", "verifier_acceptance_gap", "test_environment_blocker"),
            run_scope="verifier_only_replay",
            selected_action="protocol_first_linkcheck_validation",
            operator_gate="",
            patch_contract="",
            expected_runtime_evidence=(
                "changed Sphinx linkcheck source",
                "problem or trace mentions linkcheck anchors/local URI handling",
            ),
            development_support_instance_ids=("sphinx-doc__sphinx-7985",),
            claim_boundary="protocol-sensitive template; use for quarantine/validation before source recovery credit",
            priority=90,
        ),
    ]


def runtime_features_from_row(row: dict[str, Any], *, problem_statement: str = "") -> dict[str, Any]:
    """Extract runtime-safe matching features from a proposal/material row."""

    digest = dict(row.get("trajectory_digest", {}) or {})
    changed_files = [
        str(path)
        for path in list(digest.get("changed_files", []) or row.get("changed_files", []) or [])
        if str(path).strip()
    ]
    text = " ".join(
        str(value or "")
        for value in (
            problem_statement,
            row.get("problem_statement", ""),
            row.get("proposed_graph_failure_type", ""),
            row.get("proposed_primary_failure_type", ""),
            row.get("proposed_recovery_action", ""),
            row.get("proposed_responsible_stage", ""),
            digest.get("natural_failure_family", ""),
        )
    ).lower()
    return {
        "instance_id": str(row.get("instance_id", "") or ""),
        "project": str(row.get("project", "") or _project_from_instance(str(row.get("instance_id", "") or ""))),
        "changed_files": changed_files,
        "problem_terms": sorted(set(_tokens(text))),
        "proposal_failure_type": str(row.get("proposed_graph_failure_type", "") or ""),
        "proposal_primary_failure_type": str(row.get("proposed_primary_failure_type", "") or ""),
        "proposal_run_scope": str(row.get("proposed_run_scope", "") or ""),
        "proposal_recovery_action": str(row.get("proposed_recovery_action", "") or ""),
        "responsible_stage": str(row.get("proposed_responsible_stage", "") or ""),
        "review_priority": str(row.get("review_priority", "") or ""),
        "proposal_confidence": str(row.get("proposal_confidence", "") or ""),
    }


def match_operator_templates(
    features: dict[str, Any],
    *,
    templates: list[OperatorTemplate] | None = None,
    min_score: float = 3.0,
) -> list[dict[str, Any]]:
    """Rank templates for runtime-safe features."""

    rows = []
    for template in templates or default_operator_templates():
        score, reasons = _score_template(template, features)
        if score >= min_score:
            rows.append(
                {
                    "template_id": template.template_id,
                    "operator_name": template.operator_name,
                    "score": round(score, 3),
                    "match_reasons": reasons,
                    "run_scope": template.run_scope,
                    "selected_action": template.selected_action,
                    "operator_gate": template.operator_gate,
                    "patch_contract": template.patch_contract,
                    "development_support_instance_ids": list(template.development_support_instance_ids),
                    "claim_boundary": template.claim_boundary,
                    "runtime_feature_keys": sorted(RUNTIME_FEATURE_KEYS),
                    "runtime_oracle_leakage_check": runtime_oracle_leakage_check(features),
                }
            )
    return sorted(rows, key=lambda row: (-float(row["score"]), row["template_id"]))


def build_template_queue_rows(
    proposal_rows: list[dict[str, Any]],
    *,
    problem_statements_by_instance: dict[str, str] | None = None,
    min_score: float = 3.0,
) -> list[dict[str, Any]]:
    """Build template-conditioned planning rows from strict80 proposals."""

    problems = dict(problem_statements_by_instance or {})
    rows = []
    for row in proposal_rows:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id", "") or "")
        features = runtime_features_from_row(row, problem_statement=problems.get(instance_id, ""))
        matches = match_operator_templates(features, min_score=min_score)
        best = matches[0] if matches else {}
        rows.append(
            {
                "instance_id": instance_id,
                "project": features["project"],
                "template_selected": bool(best),
                "selected_template_id": str(best.get("template_id", "") or ""),
                "selected_operator_name": str(best.get("operator_name", "") or ""),
                "selected_run_scope": str(best.get("run_scope", "") or row.get("proposed_run_scope", "") or ""),
                "selected_action": str(best.get("selected_action", "") or row.get("proposed_recovery_action", "") or ""),
                "operator_gate": str(best.get("operator_gate", "") or ""),
                "patch_contract": str(best.get("patch_contract", "") or ""),
                "template_score": float(best.get("score", 0.0) or 0.0),
                "template_match_reasons": list(best.get("match_reasons", []) or []),
                "all_template_matches": matches,
                "runtime_features": features,
                "claim_boundary": (
                    "template_queue_row_not_label_denominator_or_recovery_credit;"
                    "runtime_features_exclude_oracle_patch_fields"
                ),
            }
        )
    return rows


def runtime_oracle_leakage_check(features: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(_flatten_keys(features))
    suspicious = [key for key in keys if ORACLE_FIELD_RE.search(key)]
    return {
        "runtime_features_oracle_safe": not suspicious,
        "suspicious_feature_keys": suspicious,
    }


def template_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("template_selected")]
    return {
        "row_count": len(rows),
        "template_selected_count": len(selected),
        "template_selected_instance_ids": [str(row.get("instance_id", "") or "") for row in selected],
        "selected_template_counts": dict(
            sorted(Counter(str(row.get("selected_template_id", "") or "") for row in selected).items())
        ),
        "selected_run_scope_counts": dict(
            sorted(Counter(str(row.get("selected_run_scope", "") or "") for row in selected).items())
        ),
        "oracle_safe_feature_row_count": sum(
            1
            for row in rows
            if bool(
                dict(row.get("runtime_features", {}) or {})
                and runtime_oracle_leakage_check(dict(row.get("runtime_features", {}) or {}))[
                    "runtime_features_oracle_safe"
                ]
            )
        ),
    }


def _score_template(template: OperatorTemplate, features: dict[str, Any]) -> tuple[float, list[str]]:
    project = str(features.get("project", "") or "")
    changed_files = [str(path) for path in list(features.get("changed_files", []) or [])]
    terms = {str(term).lower() for term in list(features.get("problem_terms", []) or [])}
    failure = {
        str(features.get("proposal_failure_type", "") or ""),
        str(features.get("proposal_primary_failure_type", "") or ""),
    }

    score = 0.0
    reasons: list[str] = []
    if project in template.project_families:
        score += 2.0
        reasons.append(f"project:{project}")
    matched_files = [
        pattern
        for pattern in template.changed_file_patterns
        if any(_path_matches(path, pattern) for path in changed_files)
    ]
    if matched_files:
        score += 3.0
        reasons.append("changed_file:" + ",".join(matched_files[:3]))
    matched_terms = [
        term for term in template.problem_terms if _term_matches(term, terms)
    ]
    if matched_terms:
        score += min(2.0, 0.5 * len(matched_terms))
        reasons.append("term:" + ",".join(matched_terms[:5]))
    matched_failures = [hint for hint in template.failure_type_hints if hint in failure]
    if matched_failures:
        score += 1.0
        reasons.append("failure:" + ",".join(matched_failures[:3]))
    if str(features.get("instance_id", "") or "") in template.development_support_instance_ids:
        score += 0.5
        reasons.append("development_support_instance")
    return score, reasons


def _path_matches(path: str, pattern: str) -> bool:
    normalized_path = str(path or "").strip()
    normalized_pattern = str(pattern or "").strip()
    if not normalized_path or not normalized_pattern:
        return False
    return normalized_path == normalized_pattern or normalized_pattern.rstrip("/") in normalized_path


def _term_matches(term: str, terms: set[str]) -> bool:
    normalized = str(term or "").lower()
    if not normalized:
        return False
    pieces = [piece for piece in re.split(r"[^a-z0-9]+", normalized) if piece]
    return normalized in terms or all(piece in terms for piece in pieces)


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(text or "").lower()) if token]


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            keys.extend(_flatten_keys(item, child))
        return keys
    if isinstance(value, list):
        keys = []
        for index, item in enumerate(value):
            keys.extend(_flatten_keys(item, f"{prefix}[{index}]"))
        return keys
    return [prefix] if prefix else []


def _project_from_instance(instance_id: str) -> str:
    return instance_id.split("__", 1)[0] if "__" in instance_id else ""
