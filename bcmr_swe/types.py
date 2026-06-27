"""Shared BCMR data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import re
import time
from typing import Any


class NodeKind(str, Enum):
    AGENT_STEP = "AgentStep"
    MESSAGE = "Message"
    SHARED_FACT = "SharedFact"
    TOOL_CALL = "ToolCall"
    VERIFIER_RESULT = "VerifierResult"
    CHECKPOINT = "Checkpoint"


class EdgeKind(str, Enum):
    PRODUCES = "produces"
    READS = "reads"
    WRITES = "writes"
    DEPENDS_ON = "depends_on"
    VALIDATED_BY = "validated_by"


class TriggerType(str, Enum):
    VERIFIER_CONTRADICTION = "verifier_contradiction"
    NO_PROGRESS_LOOP = "no_progress_loop"
    FACT_CONFLICT = "fact_conflict"


class ActionType(str, Enum):
    QUARANTINE_FACT = "QUARANTINE_FACT"
    ROLLBACK_TO_CHECKPOINT = "ROLLBACK_TO_CHECKPOINT"
    REPLAY_SUBGRAPH = "REPLAY_SUBGRAPH"
    INSERT_VERIFIER = "INSERT_VERIFIER"
    ESCALATE_NODE = "ESCALATE_NODE"


class OpType(str, Enum):
    """Atomic recovery operators.

    The first five operators (INSPECT/REVOKE/ROLLBACK/REPLAY/ESCALATE) form
    the Phase-0 program-language core. SELECTIVE_REPLAY is the Path-B
    MAS-native primitive: re-run one role while honoring cached upstream
    outputs for the rest of the stage chain. It names a recovery intent
    that a single-agent retry substrate cannot express — the executor
    must walk the propagation chain to decide which typed objects its
    replay invalidates.
    """
    INSPECT = "INSPECT"
    REVOKE = "REVOKE"
    ROLLBACK = "ROLLBACK"
    REPLAY = "REPLAY"
    ESCALATE = "ESCALATE"
    SELECTIVE_REPLAY = "SELECTIVE_REPLAY"


class SemanticActionType(str, Enum):
    """Upper-layer recovery semantics used by the redesigned action language."""

    EVIDENCE_RECHECK = "EVIDENCE_RECHECK"
    TARGET_RESET = "TARGET_RESET"
    LOCAL_REPAIR = "LOCAL_REPAIR"
    SCOPE_EXPAND = "SCOPE_EXPAND"
    CAPABILITY_BOOST = "CAPABILITY_BOOST"
    RECHECK_OBJECT = "RECHECK_OBJECT"
    REVOKE_OBJECT = "REVOKE_OBJECT"
    REPAIR_LOCAL = "REPAIR_LOCAL"
    EXPAND_SCOPE = "EXPAND_SCOPE"


class PrimitiveOpType(str, Enum):
    """Lower-layer execution primitives compiled from semantic actions."""

    READ_EVIDENCE = "READ_EVIDENCE"
    CLEAR_LOCAL_STATE = "CLEAR_LOCAL_STATE"
    ROLLBACK_ANCHOR = "ROLLBACK_ANCHOR"
    CONSTRAINED_REPLAY = "CONSTRAINED_REPLAY"
    FOCUSED_VERIFY = "FOCUSED_VERIFY"
    BOOST_EXECUTION = "BOOST_EXECUTION"


@dataclass(slots=True)
class RecoveryStep:
    """One atomic operator invocation inside a recovery program."""
    op: OpType
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op.value, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryStep":
        return cls(
            op=OpType(str(data.get("op", "REPLAY"))),
            args=dict(data.get("args", {})),
        )


@dataclass(slots=True)
class RecoveryProgram:
    """A short sequence of recovery operators (1-3 core steps)."""
    program_id: str
    steps: list[RecoveryStep]
    rationale: str = ""
    estimated_total_cost: float = 0.0
    estimated_recover_prob: float = 0.0
    estimated_risk: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skeleton(self) -> str:
        """Compact string like ``INSPECT→REVOKE→REPLAY``."""
        return "→".join(s.op.value for s in self.steps)

    @property
    def terminal_op(self) -> "OpType | None":
        return self.steps[-1].op if self.steps else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "steps": [s.to_dict() for s in self.steps],
            "rationale": self.rationale,
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_recover_prob": self.estimated_recover_prob,
            "estimated_risk": self.estimated_risk,
            "skeleton": self.skeleton,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryProgram":
        return cls(
            program_id=str(data.get("program_id", "")),
            steps=[RecoveryStep.from_dict(s) for s in data.get("steps", [])],
            rationale=str(data.get("rationale", "")),
            estimated_total_cost=float(data.get("estimated_total_cost", 0)),
            estimated_recover_prob=float(data.get("estimated_recover_prob", 0)),
            estimated_risk=float(data.get("estimated_risk", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SemanticRecoveryStep:
    """One upper-layer semantic recovery action."""

    action: SemanticActionType
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticRecoveryStep":
        return cls(
            action=SemanticActionType(str(data.get("action", SemanticActionType.LOCAL_REPAIR.value))),
            args=dict(data.get("args", {})),
        )


@dataclass(slots=True)
class SemanticRecoveryProgram:
    """A short program over upper-layer recovery semantics."""

    program_id: str
    steps: list[SemanticRecoveryStep]
    rationale: str = ""
    estimated_total_cost: float = 0.0
    estimated_recover_prob: float = 0.0
    estimated_risk: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skeleton(self) -> str:
        return "→".join(step.action.value for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "steps": [step.to_dict() for step in self.steps],
            "rationale": self.rationale,
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_recover_prob": self.estimated_recover_prob,
            "estimated_risk": self.estimated_risk,
            "skeleton": self.skeleton,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticRecoveryProgram":
        return cls(
            program_id=str(data.get("program_id", "")),
            steps=[SemanticRecoveryStep.from_dict(step) for step in data.get("steps", [])],
            rationale=str(data.get("rationale", "")),
            estimated_total_cost=float(data.get("estimated_total_cost", 0)),
            estimated_recover_prob=float(data.get("estimated_recover_prob", 0)),
            estimated_risk=float(data.get("estimated_risk", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class PrimitiveStep:
    """One lower-layer execution primitive."""

    op: PrimitiveOpType
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op.value, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitiveStep":
        return cls(
            op=PrimitiveOpType(str(data.get("op", PrimitiveOpType.CONSTRAINED_REPLAY.value))),
            args=dict(data.get("args", {})),
        )


@dataclass(slots=True)
class PrimitiveProgram:
    """Compiled lower-layer primitive program."""

    program_id: str
    steps: list[PrimitiveStep]
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skeleton(self) -> str:
        return "→".join(step.op.value for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "steps": [step.to_dict() for step in self.steps],
            "rationale": self.rationale,
            "skeleton": self.skeleton,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimitiveProgram":
        return cls(
            program_id=str(data.get("program_id", "")),
            steps=[PrimitiveStep.from_dict(step) for step in data.get("steps", [])],
            rationale=str(data.get("rationale", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class PropagationObject:
    """Typed propagation object shared across MAS stages."""

    object_type: str
    object_id: str
    producer_stage: str
    consumer_stage: str
    contamination_status: str = "unknown"
    evidence_anchor: str = ""
    replay_anchor: str = ""
    verifier_link: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "object_id": self.object_id,
            "producer_stage": self.producer_stage,
            "consumer_stage": self.consumer_stage,
            "contamination_status": self.contamination_status,
            "evidence_anchor": self.evidence_anchor,
            "replay_anchor": self.replay_anchor,
            "verifier_link": self.verifier_link,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropagationObject":
        return cls(
            object_type=str(data.get("object_type", "")),
            object_id=str(data.get("object_id", "")),
            producer_stage=str(data.get("producer_stage", "")),
            consumer_stage=str(data.get("consumer_stage", "")),
            contamination_status=str(data.get("contamination_status", "unknown") or "unknown"),
            evidence_anchor=str(data.get("evidence_anchor", "")),
            replay_anchor=str(data.get("replay_anchor", "")),
            verifier_link=str(data.get("verifier_link", "")),
            payload=dict(data.get("payload", {})),
        )


@dataclass(slots=True)
class NaturalCaseCard:
    """Manual expert review card for natural failed-state pool admission."""

    case_id: str
    instance_id: str
    system_variant: str
    source_output: str
    run_id: str
    run_dir: str
    source_type: str = "natural"
    accept_status: str = "candidate"
    report_use: str = "exclude"
    evaluation_split: str = ""
    failure_family_manual: str = ""
    mas_object_chain: list[PropagationObject] = field(default_factory=list)
    recovery_anchor_policy: str = ""
    target_legitimacy: str = ""
    patch_legitimacy: str = ""
    typed_object_quality: str = ""
    verifier_excerpt: str = ""
    rejection_reason: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "instance_id": self.instance_id,
            "system_variant": self.system_variant,
            "source_type": self.source_type,
            "source_output": self.source_output,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "accept_status": self.accept_status,
            "report_use": self.report_use,
            "evaluation_split": self.evaluation_split,
            "failure_family_manual": self.failure_family_manual,
            "mas_object_chain": [item.to_dict() for item in self.mas_object_chain],
            "recovery_anchor_policy": self.recovery_anchor_policy,
            "target_legitimacy": self.target_legitimacy,
            "patch_legitimacy": self.patch_legitimacy,
            "typed_object_quality": self.typed_object_quality,
            "verifier_excerpt": self.verifier_excerpt,
            "rejection_reason": self.rejection_reason,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NaturalCaseCard":
        return cls(
            case_id=str(data.get("case_id", "")),
            instance_id=str(data.get("instance_id", "")),
            system_variant=str(data.get("system_variant", "base") or "base"),
            source_type=str(data.get("source_type", "natural") or "natural"),
            source_output=str(data.get("source_output", "")),
            run_id=str(data.get("run_id", "")),
            run_dir=str(data.get("run_dir", "")),
            accept_status=str(data.get("accept_status", "candidate") or "candidate"),
            report_use=str(data.get("report_use", data.get("pa" + "per_use", "exclude")) or "exclude"),
            evaluation_split=str(data.get("evaluation_split", "")),
            failure_family_manual=str(data.get("failure_family_manual", "")),
            mas_object_chain=[
                PropagationObject.from_dict(item)
                for item in data.get("mas_object_chain", [])
                if isinstance(item, dict)
            ],
            recovery_anchor_policy=str(data.get("recovery_anchor_policy", "")),
            target_legitimacy=str(data.get("target_legitimacy", "")),
            patch_legitimacy=str(data.get("patch_legitimacy", "")),
            typed_object_quality=str(data.get("typed_object_quality", "")),
            verifier_excerpt=str(data.get("verifier_excerpt", "")),
            rejection_reason=str(data.get("rejection_reason", "")),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class FrozenMethodSpec:
    """Frozen method specification for BCMR main evaluation."""

    spec_id: str
    state_schema_version: str
    program_space_version: str
    selector_family: str
    baseline_names: list[str] = field(default_factory=list)
    max_candidate_programs: int = 0
    execute_candidate_limit: int = 0
    recovery_budget: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "state_schema_version": self.state_schema_version,
            "program_space_version": self.program_space_version,
            "selector_family": self.selector_family,
            "baseline_names": list(self.baseline_names),
            "max_candidate_programs": self.max_candidate_programs,
            "execute_candidate_limit": self.execute_candidate_limit,
            "recovery_budget": dict(self.recovery_budget),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenMethodSpec":
        return cls(
            spec_id=str(data.get("spec_id", "")),
            state_schema_version=str(data.get("state_schema_version", "")),
            program_space_version=str(data.get("program_space_version", "")),
            selector_family=str(data.get("selector_family", "")),
            baseline_names=[str(item) for item in data.get("baseline_names", [])],
            max_candidate_programs=int(data.get("max_candidate_programs", 0)),
            execute_candidate_limit=int(data.get("execute_candidate_limit", 0)),
            recovery_budget=dict(data.get("recovery_budget", {})),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class StructuredRecoveryState:
    """Unified MAS-aware recovery state consumed by mainline runners and actions."""

    local_region_view: dict[str, Any] = field(default_factory=dict)
    role_aggregate_view: dict[str, dict[str, Any]] = field(default_factory=dict)
    object_chain_view: list[PropagationObject] = field(default_factory=list)
    replay_anchor_view: dict[str, Any] = field(default_factory=dict)
    evidence_pack: dict[str, Any] = field(default_factory=dict)
    state_numeric: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_region_view": dict(self.local_region_view),
            "role_aggregate_view": {
                str(key): dict(value)
                for key, value in self.role_aggregate_view.items()
            },
            "object_chain_view": [item.to_dict() for item in self.object_chain_view],
            "replay_anchor_view": dict(self.replay_anchor_view),
            "evidence_pack": dict(self.evidence_pack),
            "state_numeric": {str(key): float(value) for key, value in self.state_numeric.items()},
            "metadata": dict(self.metadata),
        }

    def to_llm_payload(self) -> dict[str, Any]:
        """Return PARC's fixed-schema, enum-token LLM payload.

        This payload is intentionally not a prose rendering of the existing
        views.  It keeps stable top-level keys and turns state labels into
        fixed enum tokens so the representation is friendlier to an LLM than
        free-form markdown while remaining deterministic and small-sample safe.
        """

        local = dict(self.local_region_view or {})
        replay = dict(self.replay_anchor_view or {})
        evidence = dict(self.evidence_pack or {})
        role_chain = _ordered_strs(list(local.get("role_chain", []) or []))
        suspect_paths = _ordered_strs(
            list(local.get("suspect_paths", []) or [])
            + list(local.get("selected_target_candidates", []) or [])
            + list(evidence.get("selected_target_candidates", []) or [])
        )
        active_object_type = (
            self.object_chain_view[0].object_type
            if self.object_chain_view
            else "unknown"
        )
        failed_stage = (
            role_chain[0]
            if role_chain
            else replay.get("current_checkpoint_label")
            or "unknown"
        )
        objects = [
            _llm_object_payload(obj)
            for obj in list(self.object_chain_view or [])[:16]
        ]
        return {
            "schema_version": "parc.structured_state.v1",
            "source": {
                "source_type": _enum_token("SRC", local.get("source_type") or self.metadata.get("source_type")),
                "instance_id_hash": _stable_hash(self.metadata.get("instance_id", "")),
            },
            "failure_signature": {
                "trigger": _enum_token("TRIGGER", local.get("trigger_type") or local.get("fault_type")),
                "failed_stage": _enum_token("STAGE", failed_stage),
                "active_object_type": _enum_token("OBJ_TYPE", active_object_type),
            },
            "mas_handoff": {
                "role_chain": [_enum_token("ROLE", role) for role in role_chain],
                "cross_agent_edges": _llm_cross_agent_edges(self.object_chain_view),
                "shared_promotions": [
                    edge
                    for edge in _llm_cross_agent_edges(self.object_chain_view)
                    if edge["edge_type"] == _enum_token("EDGE", "promoted_to_shared")
                ],
            },
            "objects": objects,
            "replay": {
                "current_checkpoint": _enum_token("ANCHOR", replay.get("current_checkpoint_label")),
                "healthy_anchors": [
                    _enum_token("ANCHOR", item)
                    for item in list(replay.get("healthy_anchor_candidates", []) or [])[:8]
                ],
                "post_fault_anchors": [
                    _enum_token("ANCHOR", item)
                    for item in list(replay.get("post_fault_checkpoint_labels", []) or [])[:8]
                ],
            },
            "evidence": {
                "failing_tests_bucket": _count_bucket(evidence.get("failing_tests_count", 0)),
                "target_legitimacy": _enum_token("TARGET", evidence.get("target_legitimacy")),
                "patch_legitimacy": _enum_token("PATCH", evidence.get("patch_legitimacy")),
                "stop_reason": _enum_token("STOP", evidence.get("stop_reason")),
                "negative_constraints": [
                    _enum_token("NEG", item)
                    for item in list(evidence.get("negative_constraints", []) or local.get("negative_constraints", []) or [])[:8]
                ],
            },
            "paths": {
                "suspect_paths": suspect_paths[:8],
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredRecoveryState":
        role_aggregate = {
            str(key): dict(value)
            for key, value in dict(data.get("role_aggregate_view", {})).items()
            if isinstance(value, dict)
        }
        return cls(
            local_region_view=dict(data.get("local_region_view", {})),
            role_aggregate_view=role_aggregate,
            object_chain_view=[
                PropagationObject.from_dict(item)
                for item in data.get("object_chain_view", [])
                if isinstance(item, dict)
            ],
            replay_anchor_view=dict(data.get("replay_anchor_view", {})),
            evidence_pack=dict(data.get("evidence_pack", {})),
            state_numeric={
                str(key): float(value)
                for key, value in dict(data.get("state_numeric", {})).items()
            },
            metadata=dict(data.get("metadata", {})),
        )


def _stable_hash(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _enum_token(prefix: str, value: Any) -> str:
    raw = str(value or "unknown").strip()
    if not raw:
        raw = "unknown"
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    token = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    token = re.sub(r"_+", "_", token) or "UNKNOWN"
    return f"<{prefix}:{token}>"


def _ordered_strs(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _count_bucket(value: Any) -> str:
    try:
        count = int(float(value or 0))
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        bucket = "ZERO"
    elif count == 1:
        bucket = "ONE"
    elif count <= 3:
        bucket = "LOW"
    elif count <= 10:
        bucket = "MED"
    else:
        bucket = "HIGH"
    return f"<COUNT:{bucket}>"


def _llm_object_payload(obj: PropagationObject) -> dict[str, Any]:
    lifecycle = ""
    metadata = obj.payload.get("metadata") if isinstance(obj.payload, dict) else {}
    if isinstance(metadata, dict):
        lifecycle = str(metadata.get("lifecycle_state", "") or "")
    lifecycle = lifecycle or str(obj.payload.get("lifecycle_state", "") if isinstance(obj.payload, dict) else "")
    payload_keys = sorted(str(key) for key in dict(obj.payload or {}).keys())[:8]
    return {
        "id_hash": _stable_hash(obj.object_id),
        "type": _enum_token("OBJ_TYPE", obj.object_type),
        "producer": _enum_token("STAGE", obj.producer_stage),
        "consumer": _enum_token("STAGE", obj.consumer_stage),
        "handoff": _enum_token("HANDOFF", f"{obj.producer_stage}_to_{obj.consumer_stage}"),
        "contamination": _enum_token("CONTAM", obj.contamination_status),
        "lifecycle": _enum_token("LIFE", lifecycle or "suspicious"),
        "replay_anchor": _enum_token("ANCHOR", obj.replay_anchor),
        "payload_keys": [_enum_token("KEY", key) for key in payload_keys],
    }


def _llm_cross_agent_edges(objects: list[PropagationObject]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for obj in objects:
        producer = str(obj.producer_stage or "unknown")
        consumer = str(obj.consumer_stage or "unknown")
        if not producer or not consumer or producer == consumer:
            continue
        edge_type = "promoted_to_shared" if obj.object_type == "SharedFact" else "consumed_by"
        key = (edge_type, producer, consumer)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "edge_type": _enum_token("EDGE", edge_type),
                "from": _enum_token("STAGE", producer),
                "to": _enum_token("STAGE", consumer),
            }
        )
    return edges


@dataclass(slots=True)
class ActionObservation:
    """Standardized result surface for one recovery action."""

    action_type: str
    status: str = ""
    active_target_before: str = ""
    active_target_after: str = ""
    active_object_type_before: str = ""
    active_object_type_after: str = ""
    active_object_id_before: str = ""
    active_object_id_after: str = ""
    replay_scope: str = ""
    touched_paths: list[str] = field(default_factory=list)
    invalidated_targets: list[str] = field(default_factory=list)
    invalidated_object_ids: list[str] = field(default_factory=list)
    verifier_excerpt: str = ""
    target_legitimacy: str = ""
    patch_legitimacy: str = ""
    negative_constraints: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "status": self.status,
            "active_target_before": self.active_target_before,
            "active_target_after": self.active_target_after,
            "active_object_type_before": self.active_object_type_before,
            "active_object_type_after": self.active_object_type_after,
            "active_object_id_before": self.active_object_id_before,
            "active_object_id_after": self.active_object_id_after,
            "replay_scope": self.replay_scope,
            "touched_paths": list(self.touched_paths),
            "invalidated_targets": list(self.invalidated_targets),
            "invalidated_object_ids": list(self.invalidated_object_ids),
            "verifier_excerpt": self.verifier_excerpt,
            "target_legitimacy": self.target_legitimacy,
            "patch_legitimacy": self.patch_legitimacy,
            "negative_constraints": list(self.negative_constraints),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionObservation":
        return cls(
            action_type=str(data.get("action_type", "")),
            status=str(data.get("status", "")),
            active_target_before=str(data.get("active_target_before", "")),
            active_target_after=str(data.get("active_target_after", "")),
            active_object_type_before=str(data.get("active_object_type_before", "")),
            active_object_type_after=str(data.get("active_object_type_after", "")),
            active_object_id_before=str(data.get("active_object_id_before", "")),
            active_object_id_after=str(data.get("active_object_id_after", "")),
            replay_scope=str(data.get("replay_scope", "")),
            touched_paths=[str(item) for item in data.get("touched_paths", [])],
            invalidated_targets=[str(item) for item in data.get("invalidated_targets", [])],
            invalidated_object_ids=[str(item) for item in data.get("invalidated_object_ids", [])],
            verifier_excerpt=str(data.get("verifier_excerpt", "")),
            target_legitimacy=str(data.get("target_legitimacy", "")),
            patch_legitimacy=str(data.get("patch_legitimacy", "")),
            negative_constraints=[str(item) for item in data.get("negative_constraints", [])],
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class StateDelta:
    """Structured state change emitted by one recovery action."""

    active_target: str = ""
    active_object_type: str = ""
    active_object_id: str = ""
    added_invalidated_targets: list[str] = field(default_factory=list)
    added_invalidated_object_ids: list[str] = field(default_factory=list)
    added_negative_constraints: list[str] = field(default_factory=list)
    latest_verifier_verdict: str = ""
    latest_shared_fact_key: str = ""
    touches_suspect_path: bool = False
    consumed_step_budget: int = 0
    consumed_token_budget: float = 0.0
    consumed_latency_budget_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_target": self.active_target,
            "active_object_type": self.active_object_type,
            "active_object_id": self.active_object_id,
            "added_invalidated_targets": list(self.added_invalidated_targets),
            "added_invalidated_object_ids": list(self.added_invalidated_object_ids),
            "added_negative_constraints": list(self.added_negative_constraints),
            "latest_verifier_verdict": self.latest_verifier_verdict,
            "latest_shared_fact_key": self.latest_shared_fact_key,
            "touches_suspect_path": self.touches_suspect_path,
            "consumed_step_budget": self.consumed_step_budget,
            "consumed_token_budget": self.consumed_token_budget,
            "consumed_latency_budget_sec": self.consumed_latency_budget_sec,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateDelta":
        return cls(
            active_target=str(data.get("active_target", "")),
            active_object_type=str(data.get("active_object_type", "")),
            active_object_id=str(data.get("active_object_id", "")),
            added_invalidated_targets=[str(item) for item in data.get("added_invalidated_targets", [])],
            added_invalidated_object_ids=[str(item) for item in data.get("added_invalidated_object_ids", [])],
            added_negative_constraints=[str(item) for item in data.get("added_negative_constraints", [])],
            latest_verifier_verdict=str(data.get("latest_verifier_verdict", "")),
            latest_shared_fact_key=str(data.get("latest_shared_fact_key", "")),
            touches_suspect_path=bool(data.get("touches_suspect_path", False)),
            consumed_step_budget=int(data.get("consumed_step_budget", 0)),
            consumed_token_budget=float(data.get("consumed_token_budget", 0.0)),
            consumed_latency_budget_sec=float(data.get("consumed_latency_budget_sec", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class RecoveryLedger:
    """Structured recovery state carried across actions."""

    trigger_reason: str = ""
    failing_tests_summary: list[str] = field(default_factory=list)
    active_target: str = ""
    active_object_type: str = ""
    active_object_id: str = ""
    active_object_excerpt: str = ""
    suspect_paths: list[str] = field(default_factory=list)
    invalidated_targets: list[str] = field(default_factory=list)
    invalidated_object_ids: list[str] = field(default_factory=list)
    key_evidence: list[str] = field(default_factory=list)
    latest_verifier_verdict: str = ""
    latest_shared_fact_key: str = ""
    negative_constraints: list[str] = field(default_factory=list)
    execution_profile: str = "normal"
    last_action: str = ""
    last_action_result: dict[str, Any] = field(default_factory=dict)
    last_source_edit_summary: dict[str, Any] = field(default_factory=dict)
    touches_suspect_path: bool = False
    tried_actions: list[str] = field(default_factory=list)
    remaining_step_budget: int = 0
    remaining_token_budget: float = 0.0
    remaining_latency_budget_sec: float = 0.0
    structured_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_reason": self.trigger_reason,
            "failing_tests_summary": list(self.failing_tests_summary),
            "active_target": self.active_target,
            "active_object_type": self.active_object_type,
            "active_object_id": self.active_object_id,
            "active_object_excerpt": self.active_object_excerpt,
            "suspect_paths": list(self.suspect_paths),
            "invalidated_targets": list(self.invalidated_targets),
            "invalidated_object_ids": list(self.invalidated_object_ids),
            "key_evidence": list(self.key_evidence),
            "latest_verifier_verdict": self.latest_verifier_verdict,
            "latest_shared_fact_key": self.latest_shared_fact_key,
            "negative_constraints": list(self.negative_constraints),
            "execution_profile": self.execution_profile,
            "last_action": self.last_action,
            "last_action_result": dict(self.last_action_result),
            "last_source_edit_summary": dict(self.last_source_edit_summary),
            "touches_suspect_path": self.touches_suspect_path,
            "tried_actions": list(self.tried_actions),
            "remaining_step_budget": self.remaining_step_budget,
            "remaining_token_budget": self.remaining_token_budget,
            "remaining_latency_budget_sec": self.remaining_latency_budget_sec,
            "structured_state": dict(self.structured_state),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryLedger":
        return cls(
            trigger_reason=str(data.get("trigger_reason", "")),
            failing_tests_summary=[str(item) for item in data.get("failing_tests_summary", [])],
            active_target=str(data.get("active_target", "")),
            active_object_type=str(data.get("active_object_type", "")),
            active_object_id=str(data.get("active_object_id", "")),
            active_object_excerpt=str(data.get("active_object_excerpt", "")),
            suspect_paths=[str(item) for item in data.get("suspect_paths", [])],
            invalidated_targets=[str(item) for item in data.get("invalidated_targets", [])],
            invalidated_object_ids=[str(item) for item in data.get("invalidated_object_ids", [])],
            key_evidence=[str(item) for item in data.get("key_evidence", [])],
            latest_verifier_verdict=str(data.get("latest_verifier_verdict", "")),
            latest_shared_fact_key=str(data.get("latest_shared_fact_key", "")),
            negative_constraints=[str(item) for item in data.get("negative_constraints", [])],
            execution_profile=str(data.get("execution_profile", "normal") or "normal"),
            last_action=str(data.get("last_action", "")),
            last_action_result=dict(data.get("last_action_result", {})),
            last_source_edit_summary=dict(data.get("last_source_edit_summary", {})),
            touches_suspect_path=bool(data.get("touches_suspect_path", False)),
            tried_actions=[str(item) for item in data.get("tried_actions", [])],
            remaining_step_budget=int(data.get("remaining_step_budget", 0)),
            remaining_token_budget=float(data.get("remaining_token_budget", 0.0)),
            remaining_latency_budget_sec=float(data.get("remaining_latency_budget_sec", 0.0)),
            structured_state=dict(data.get("structured_state", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ProgramOutcome:
    """Ground-truth result from executing a recovery program via replay."""
    program_id: str
    recover_success: bool
    official_resolved: bool
    token_cost: float
    latency_sec: float
    secondary_risk: float
    milestone_gain: float
    step_outcomes: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "recover_success": self.recover_success,
            "official_resolved": self.official_resolved,
            "token_cost": self.token_cost,
            "latency_sec": self.latency_sec,
            "secondary_risk": self.secondary_risk,
            "milestone_gain": self.milestone_gain,
            "step_outcomes": list(self.step_outcomes),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgramOutcome":
        return cls(
            program_id=str(data.get("program_id", "")),
            recover_success=bool(data.get("recover_success", False)),
            official_resolved=bool(data.get("official_resolved", False)),
            token_cost=float(data.get("token_cost", 0)),
            latency_sec=float(data.get("latency_sec", 0)),
            secondary_risk=float(data.get("secondary_risk", 0)),
            milestone_gain=float(data.get("milestone_gain", 0)),
            step_outcomes=list(data.get("step_outcomes", [])),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Layer 3: Recovery Case Memory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FailureSignature:
    """Compact fingerprint of a failure for case retrieval."""
    trigger_type: str
    failed_stage: str
    region_shape: str
    n_failing_tests: int = 0
    has_conflicting_fact: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureSignature":
        return cls(
            trigger_type=str(data.get("trigger_type", "")),
            failed_stage=str(data.get("failed_stage", "")),
            region_shape=str(data.get("region_shape", "")),
            n_failing_tests=int(data.get("n_failing_tests", 0)),
            has_conflicting_fact=bool(data.get("has_conflicting_fact", False)),
            extra=dict(data.get("extra", {})),
        )

    def similarity(self, other: "FailureSignature") -> float:
        """Simple feature-overlap similarity for retrieval."""
        score = 0.0
        if self.trigger_type == other.trigger_type:
            score += 0.35
        if self.failed_stage == other.failed_stage:
            score += 0.25
        if self.region_shape == other.region_shape:
            score += 0.20
        if self.has_conflicting_fact == other.has_conflicting_fact:
            score += 0.10
        test_diff = abs(self.n_failing_tests - other.n_failing_tests)
        score += 0.10 * max(0.0, 1.0 - test_diff / 5.0)
        return score


@dataclass(slots=True)
class RecoveryCase:
    """A complete record of one recovery attempt — the minimal experience unit."""
    case_id: str
    instance_id: str
    failure_signature: FailureSignature
    state_summary: dict[str, Any]
    program: RecoveryProgram
    outcome: ProgramOutcome
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "instance_id": self.instance_id,
            "failure_signature": self.failure_signature.to_dict(),
            "state_summary": dict(self.state_summary),
            "program": self.program.to_dict(),
            "outcome": self.outcome.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryCase":
        return cls(
            case_id=str(data.get("case_id", "")),
            instance_id=str(data.get("instance_id", "")),
            failure_signature=FailureSignature.from_dict(data.get("failure_signature", {})),
            state_summary=dict(data.get("state_summary", {})),
            program=RecoveryProgram.from_dict(data.get("program", {})),
            outcome=ProgramOutcome.from_dict(data.get("outcome", {})),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass(slots=True)
class RecoverySkill:
    """Generalised recovery pattern induced from multiple similar cases."""
    skill_id: str
    name: str
    applicability: FailureSignature
    program_skeleton: list[str]
    n_cases: int = 0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "applicability": self.applicability.to_dict(),
            "program_skeleton": list(self.program_skeleton),
            "n_cases": self.n_cases,
            "success_rate": self.success_rate,
            "avg_cost": self.avg_cost,
            "case_ids": list(self.case_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoverySkill":
        return cls(
            skill_id=str(data.get("skill_id", "")),
            name=str(data.get("name", "")),
            applicability=FailureSignature.from_dict(data.get("applicability", {})),
            program_skeleton=list(data.get("program_skeleton", [])),
            n_cases=int(data.get("n_cases", 0)),
            success_rate=float(data.get("success_rate", 0)),
            avg_cost=float(data.get("avg_cost", 0)),
            case_ids=list(data.get("case_ids", [])),
        )


@dataclass(slots=True)
class ProvenanceNode:
    node_id: str
    kind: NodeKind
    role: str
    phase: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    validity: str = "valid"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(slots=True)
class ProvenanceEdge:
    edge_id: str
    kind: EdgeKind
    source_id: str
    target_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(slots=True)
class CheckpointRecord:
    checkpoint_id: str
    label: str
    workspace: str
    archive_path: str
    metadata_path: str
    env_snapshot_hash: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_node_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointRecord":
        return cls(
            checkpoint_id=str(data.get("checkpoint_id", "")),
            label=str(data.get("label", "")),
            workspace=str(data.get("workspace", "")),
            archive_path=str(data.get("archive_path", "")),
            metadata_path=str(data.get("metadata_path", "")),
            env_snapshot_hash=str(data.get("env_snapshot_hash", "")),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
            source_node_id=str(data.get("source_node_id", "")),
        )


@dataclass(slots=True)
class SuspectRegion:
    node_ids: list[str]
    edge_ids: list[str]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuspectRegion":
        return cls(
            node_ids=[str(item) for item in data.get("node_ids", [])],
            edge_ids=[str(item) for item in data.get("edge_ids", [])],
            summary=dict(data.get("summary", {})),
        )


@dataclass(slots=True)
class StateFeatures:
    numeric: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": dict(self.numeric), "categorical": dict(self.categorical)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateFeatures":
        numeric = {str(key): float(value) for key, value in dict(data.get("numeric", {})).items()}
        categorical = {str(key): str(value) for key, value in dict(data.get("categorical", {})).items()}
        return cls(numeric=numeric, categorical=categorical)


@dataclass(slots=True)
class TriggerDecision:
    trigger_type: TriggerType
    trigger_node_id: str
    evidence_node_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trigger_type"] = self.trigger_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TriggerDecision":
        return cls(
            trigger_type=TriggerType(str(data.get("trigger_type", TriggerType.VERIFIER_CONTRADICTION.value))),
            trigger_node_id=str(data.get("trigger_node_id", "")),
            evidence_node_ids=[str(item) for item in data.get("evidence_node_ids", [])],
            reason=str(data.get("reason", "")),
        )


@dataclass(slots=True)
class FailedState:
    group_id: str
    run_id: str
    instance_id: str
    trigger: TriggerDecision
    checkpoint_id: str
    checkpoint: CheckpointRecord | None
    suspect_region: SuspectRegion
    state_features: StateFeatures
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "trigger": self.trigger.to_dict(),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "suspect_region": self.suspect_region.to_dict(),
            "state_features": self.state_features.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedState":
        checkpoint_payload = data.get("checkpoint")
        checkpoint = CheckpointRecord.from_dict(checkpoint_payload) if isinstance(checkpoint_payload, dict) else None
        return cls(
            group_id=str(data.get("group_id", "")),
            run_id=str(data.get("run_id", "")),
            instance_id=str(data.get("instance_id", "")),
            trigger=TriggerDecision.from_dict(dict(data.get("trigger", {}))),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            checkpoint=checkpoint,
            suspect_region=SuspectRegion.from_dict(dict(data.get("suspect_region", {}))),
            state_features=StateFeatures.from_dict(dict(data.get("state_features", {}))),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class CandidateAction:
    action_id: str
    action_type: ActionType
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    estimated_recover_prob: float = 0.0
    estimated_token_cost: float = 0.0
    estimated_latency_sec: float = 0.0
    estimated_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateAction":
        return cls(
            action_id=str(data.get("action_id", "")),
            action_type=ActionType(str(data.get("action_type", ActionType.REPLAY_SUBGRAPH.value))),
            description=str(data.get("description", "")),
            payload=dict(data.get("payload", {})),
            estimated_recover_prob=float(data.get("estimated_recover_prob", 0.0)),
            estimated_token_cost=float(data.get("estimated_token_cost", 0.0)),
            estimated_latency_sec=float(data.get("estimated_latency_sec", 0.0)),
            estimated_risk=float(data.get("estimated_risk", 0.0)),
        )


@dataclass(slots=True)
class ActionScore:
    action_id: str
    utility: float
    estimated_recover_prob: float
    estimated_token_cost: float
    estimated_latency_sec: float
    estimated_risk: float
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplayOutcome:
    action_id: str
    action_type: ActionType
    recover_success: bool
    official_resolved: bool
    token_cost: float
    latency_sec: float
    rollback_depth: int
    secondary_risk: float
    milestone_gain: float
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayOutcome":
        return cls(
            action_id=str(data.get("action_id", "")),
            action_type=ActionType(str(data.get("action_type", ActionType.REPLAY_SUBGRAPH.value))),
            recover_success=bool(data.get("recover_success", False)),
            official_resolved=bool(data.get("official_resolved", False)),
            token_cost=float(data.get("token_cost", 0.0)),
            latency_sec=float(data.get("latency_sec", 0.0)),
            rollback_depth=int(data.get("rollback_depth", 0)),
            secondary_risk=float(data.get("secondary_risk", 0.0)),
            milestone_gain=float(data.get("milestone_gain", 0.0)),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class RecoveryBudget:
    max_recovery_calls: int = 3
    token_budget: float = 120000.0
    latency_budget_sec: float = 1800.0
    lambda_token: float = 1e-4
    lambda_latency: float = 5e-4
    lambda_risk: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryBudget":
        return cls(
            max_recovery_calls=int(data.get("max_recovery_calls", 3)),
            token_budget=float(data.get("token_budget", 120000.0)),
            latency_budget_sec=float(data.get("latency_budget_sec", 1800.0)),
            lambda_token=float(data.get("lambda_token", 1e-4)),
            lambda_latency=float(data.get("lambda_latency", 5e-4)),
            lambda_risk=float(data.get("lambda_risk", 0.2)),
        )
