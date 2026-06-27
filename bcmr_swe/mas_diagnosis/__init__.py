"""MAS-DX schemas and utilities.

This package is intentionally isolated from the older BCMR-CAR recovery
controller.  It contains non-invasive diagnosis data structures for
mini-swe-agent-derived MAS trajectories.
"""

from __future__ import annotations

from bcmr_swe.mas_diagnosis.schema import (
    MASDXCaseAudit,
    MASDXDiagnosisLabel,
    MASDXEvidenceGraph,
    MASDXGraphEdge,
    MASDXGraphNode,
    MASDXTrajectoryRecord,
    MASFailureType,
    RecoveryActionLabel,
    ResponsibleStage,
)
from bcmr_swe.mas_diagnosis.normalizer import (
    load_manifest_by_instance,
    normalize_natural_row,
    normalize_records_from_file,
    normalize_run_result,
)
from bcmr_swe.mas_diagnosis.graph_builder import build_evidence_graph
from bcmr_swe.mas_diagnosis.diagnosis_inputs import (
    build_mas_graph_input,
    build_probe_flat_input,
)

__all__ = [
    "MASDXCaseAudit",
    "MASDXDiagnosisLabel",
    "MASDXEvidenceGraph",
    "MASDXGraphEdge",
    "MASDXGraphNode",
    "MASDXTrajectoryRecord",
    "MASFailureType",
    "RecoveryActionLabel",
    "ResponsibleStage",
    "load_manifest_by_instance",
    "normalize_natural_row",
    "normalize_records_from_file",
    "normalize_run_result",
    "build_evidence_graph",
    "build_mas_graph_input",
    "build_probe_flat_input",
]
