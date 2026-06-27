"""Data schemas for the MAS-DX non-invasive diagnosis line."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MASFailureType(str, Enum):
    HANDOFF_INFORMATION_LOSS = "handoff_information_loss"
    STALE_SHARED_BELIEF = "stale_shared_belief"
    WRONG_TRUST_IN_UPSTREAM_ARTIFACT = "wrong_trust_in_upstream_artifact"
    VERIFIER_FEEDBACK_NOT_PROPAGATED = "verifier_feedback_not_propagated"
    ROLE_BOUNDARY_MISMATCH = "role_boundary_mismatch"
    COORDINATOR_RETRY_ERROR = "coordinator_retry_error"
    DOWNSTREAM_OVERCOMMITMENT = "downstream_overcommitment"
    SINGLE_AGENT_LOCAL_FAILURE = "single_agent_local_failure"
    ENVIRONMENT_OR_INFRA_FAILURE = "environment_or_infra_failure"
    AMBIGUOUS = "ambiguous"


class ResponsibleStage(str, Enum):
    LOCATOR = "locator"
    REPRODUCER = "reproducer"
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    PATCHER = "patcher"
    VERIFIER = "verifier"
    COORDINATOR = "coordinator"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class RecoveryActionLabel(str, Enum):
    FULL_RERUN = "full_rerun"
    RELOCALIZE = "relocalize"
    REPLAN = "replan"
    REPATCH_WITH_EXISTING_LOCALIZATION = "repatch_with_existing_localization"
    RESET_SHARED_STATE = "reset_shared_state"
    PROPAGATE_VERIFIER_FEEDBACK = "propagate_verifier_feedback"
    RERUN_VERIFIER = "rerun_verifier"
    ESCALATE_MODEL_OR_BUDGET = "escalate_model_or_budget"
    ENVIRONMENT_REPAIR = "environment_repair"
    HUMAN_REVIEW = "human_review"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MASDXTrajectoryRecord:
    """A normalized record for one failed MAS SWE trajectory."""

    case_id: str
    instance_id: str
    issue: str = ""
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    shared_facts: dict[str, Any] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)
    diff_summary: dict[str, Any] = field(default_factory=dict)
    verifier_evidence: dict[str, Any] = field(default_factory=dict)
    oracle: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "instance_id": self.instance_id,
            "issue": self.issue,
            "stage_outputs": dict(self.stage_outputs),
            "shared_facts": dict(self.shared_facts),
            "commands": [dict(item) for item in self.commands],
            "diff_summary": dict(self.diff_summary),
            "verifier_evidence": dict(self.verifier_evidence),
            "oracle": dict(self.oracle),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MASDXTrajectoryRecord":
        return cls(
            case_id=str(data.get("case_id", "") or ""),
            instance_id=str(data.get("instance_id", "") or ""),
            issue=str(data.get("issue", "") or ""),
            stage_outputs=dict(data.get("stage_outputs", {}) or {}),
            shared_facts=dict(data.get("shared_facts", {}) or {}),
            commands=[
                dict(item)
                for item in list(data.get("commands", []) or [])
                if isinstance(item, dict)
            ],
            diff_summary=dict(data.get("diff_summary", {}) or {}),
            verifier_evidence=dict(data.get("verifier_evidence", {}) or {}),
            oracle=dict(data.get("oracle", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class MASDXGraphNode:
    node_id: str
    node_type: str
    label: str
    stage: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "stage": self.stage,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class MASDXGraphEdge:
    source: str
    target: str
    edge_type: str
    evidence_span_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "evidence_span_ids": list(self.evidence_span_ids),
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class MASDXEvidenceGraph:
    case_id: str
    nodes: list[MASDXGraphNode] = field(default_factory=list)
    edges: list[MASDXGraphEdge] = field(default_factory=list)
    evidence_spans: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "evidence_spans": dict(self.evidence_spans),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MASDXDiagnosisLabel:
    case_id: str
    primary_failure_type: MASFailureType = MASFailureType.AMBIGUOUS
    is_cross_agent_failure: bool = False
    responsible_stage: ResponsibleStage = ResponsibleStage.UNKNOWN
    faulty_artifact: str = ""
    faulty_handoff_edge: str = ""
    evidence_span_ids: list[str] = field(default_factory=list)
    recommended_recovery_action: RecoveryActionLabel = RecoveryActionLabel.UNKNOWN
    recommended_rerun_scope: str = ""
    confidence: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "primary_failure_type": self.primary_failure_type.value,
            "is_cross_agent_failure": self.is_cross_agent_failure,
            "responsible_stage": self.responsible_stage.value,
            "faulty_artifact": self.faulty_artifact,
            "faulty_handoff_edge": self.faulty_handoff_edge,
            "evidence_span_ids": list(self.evidence_span_ids),
            "recommended_recovery_action": self.recommended_recovery_action.value,
            "recommended_rerun_scope": self.recommended_rerun_scope,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(slots=True)
class MASDXCaseAudit:
    case_id: str
    complete: bool
    main_ready: bool = False
    failed_main_ready: bool = False
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    main_blockers: list[str] = field(default_factory=list)
    failed_main_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "complete": self.complete,
            "main_ready": self.main_ready,
            "failed_main_ready": self.failed_main_ready,
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "main_blockers": list(self.main_blockers),
            "failed_main_blockers": list(self.failed_main_blockers),
        }


def audit_trajectory_record(record: MASDXTrajectoryRecord) -> MASDXCaseAudit:
    """Check whether a trajectory has enough material for main-set use."""

    missing: list[str] = []
    warnings: list[str] = []
    if not record.case_id:
        missing.append("case_id")
    if not record.instance_id:
        missing.append("instance_id")
    if not record.issue.strip():
        missing.append("issue")
    if not record.stage_outputs:
        missing.append("stage_outputs")
    for stage in ("locator", "patcher", "verifier"):
        if stage not in record.stage_outputs:
            missing.append(f"stage_outputs.{stage}")
    if not record.shared_facts:
        warnings.append("shared_facts_empty")
    if not record.verifier_evidence:
        missing.append("verifier_evidence")
    if not record.oracle:
        warnings.append("oracle_missing")
    if not record.commands:
        warnings.append("commands_missing")
    if not record.diff_summary:
        warnings.append("diff_summary_missing")
    main_blockers = list(missing)
    if not record.oracle:
        main_blockers.append("oracle_missing")
    elif "oracle_success" not in record.oracle and "oracle_returncode" not in record.oracle:
        main_blockers.append("oracle_outcome_missing")
    if not record.commands:
        main_blockers.append("commands_missing")
    if not record.diff_summary:
        main_blockers.append("diff_summary_missing")
    failed_main_blockers = list(main_blockers)
    if _record_looks_resolved(record):
        failed_main_blockers.append("trajectory_resolved_not_failed")
    return MASDXCaseAudit(
        case_id=record.case_id,
        complete=not missing,
        main_ready=not main_blockers,
        failed_main_ready=not failed_main_blockers,
        missing_fields=missing,
        warnings=warnings,
        main_blockers=main_blockers,
        failed_main_blockers=failed_main_blockers,
    )


def _record_looks_resolved(record: MASDXTrajectoryRecord) -> bool:
    if record.oracle.get("oracle_success") is True:
        return True
    if record.oracle.get("oracle_returncode") == 0:
        return True
    if record.oracle.get("oracle_success") is False:
        return False
    if "oracle_returncode" in record.oracle:
        return False
    if "fail_to_pass_returncode" in record.oracle:
        return False
    family = str(record.metadata.get("natural_failure_family", "") or "").strip().lower()
    return family == "resolved"
