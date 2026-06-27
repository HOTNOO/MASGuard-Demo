"""Parameterized recovery action schemas for discovery and baseline comparison.

This module is the bridge between:

- fixed family templates used in early discovery
- and the future grammar-constrained free composition benchmark

The key idea is to make recovery intents explicit as schemas with slots,
instead of treating every family as a one-off handwritten template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcmr_swe.types import OpType, RecoveryProgram, RecoveryStep


@dataclass(slots=True)
class ActionSchema:
    """A parameterized recovery-intent schema."""

    schema_id: str
    family: str
    description: str
    slots: dict[str, Any] = field(default_factory=dict)
    steps: list[RecoveryStep] = field(default_factory=list)
    rationale: str = ""
    estimated_total_cost: float = 0.0
    estimated_recover_prob: float = 0.0
    estimated_risk: float = 0.0
    strategy: str = ""

    def to_program(self, *, program_id: str) -> RecoveryProgram:
        return RecoveryProgram(
            program_id=program_id,
            steps=list(self.steps),
            rationale=self.rationale or self.description,
            estimated_total_cost=self.estimated_total_cost,
            estimated_recover_prob=self.estimated_recover_prob,
            estimated_risk=self.estimated_risk,
            metadata={
                "family": self.family,
                "strategy": self.strategy or program_id,
                "schema_id": self.schema_id,
                "schema_slots": dict(self.slots),
            },
        )


def _local_minimal_schema(*, checkpoint_id: str) -> ActionSchema:
    return ActionSchema(
        schema_id="LOCAL_MINIMAL_RESTORE",
        family="local_minimal",
        description="Restore the latest healthy patch anchor without further regeneration.",
        slots={"checkpoint_id": checkpoint_id},
        steps=[RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_id})],
        rationale="Minimal recovery: restore the last known healthy patch checkpoint and let the official tests judge the outcome.",
        estimated_total_cost=80.0,
        estimated_recover_prob=0.95,
        estimated_risk=0.01,
        strategy="rollback_post_patch_restore",
    )


def _local_replan_schema(*, checkpoint_id: str, replay_scope: str) -> ActionSchema:
    return ActionSchema(
        schema_id="LOCAL_REPLAN",
        family="local_broader",
        description="Rollback to a broader local anchor and regenerate downstream work.",
        slots={
            "checkpoint_id": checkpoint_id,
            "replay_scope": replay_scope,
            "anchor": "post_locate",
        },
        steps=[
            RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_id}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": replay_scope,
                    "context_hint": "Rebuild only the downstream patch-and-verify region from the localization anchor.",
                },
            ),
        ],
        rationale="Broader local recovery from the localization checkpoint.",
        estimated_total_cost=2200.0,
        estimated_recover_prob=0.80,
        estimated_risk=0.10,
        strategy="rollback_post_locate_patch_verify",
    )


def _global_restart_schema(*, checkpoint_id: str) -> ActionSchema:
    return ActionSchema(
        schema_id="GLOBAL_RESTART",
        family="global",
        description="Restart from the initial clean checkpoint and replay the full pipeline.",
        slots={"checkpoint_id": checkpoint_id, "replay_scope": "full", "anchor": "initial"},
        steps=[
            RecoveryStep(op=OpType.ROLLBACK, args={"checkpoint_id": checkpoint_id}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": "full",
                    "context_hint": "Discard all intermediate progress and restart the full SWE pipeline.",
                },
            ),
        ],
        rationale="Coarse baseline: restart the full pipeline.",
        estimated_total_cost=6000.0,
        estimated_recover_prob=0.75,
        estimated_risk=0.25,
        strategy="rollback_initial_full_replay",
    )


def _evidence_recheck_schema(*, replay_scope: str) -> ActionSchema:
    return ActionSchema(
        schema_id="EVIDENCE_RECHECK",
        family="evidence_recheck",
        description="Re-check failing evidence before local replay.",
        slots={"target": "test_output", "depth": "deep", "replay_scope": replay_scope},
        steps=[
            RecoveryStep(op=OpType.INSPECT, args={"target": "test_output", "depth": "deep"}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": replay_scope,
                    "context_hint": "Re-check the failing evidence and rebuild only the patch/verify region without trusting the previous local conclusion.",
                },
            ),
        ],
        rationale="Evidence-oriented local recovery: re-check the failure signal before a local patch replay.",
        estimated_total_cost=1800.0,
        estimated_recover_prob=0.68,
        estimated_risk=0.08,
        strategy="inspect_test_then_patch_replay",
    )


def _belief_cleanup_schema(*, replay_scope: str, fact_id: str) -> ActionSchema:
    return ActionSchema(
        schema_id="BELIEF_CLEANUP",
        family="belief_cleanup",
        description="Remove a contradicted working belief, then locally regenerate.",
        slots={"fact_id": fact_id, "replay_scope": replay_scope},
        steps=[
            RecoveryStep(op=OpType.REVOKE, args={"fact_id": fact_id}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": replay_scope,
                    "context_hint": "The promoted patch fact was contradicted; discard it and derive a fresh local repair from the current failing workspace.",
                },
            ),
        ],
        rationale="Belief-cleanup recovery: remove the stale promoted patch fact before local replay.",
        estimated_total_cost=1700.0,
        estimated_recover_prob=0.70,
        estimated_risk=0.07,
        strategy="revoke_latest_patch_then_patch_replay",
    )


def _capability_boost_schema(*, replay_scope: str, scope: str = "patcher", level: int = 1) -> ActionSchema:
    return ActionSchema(
        schema_id="CAPABILITY_BOOST",
        family="capability_boost",
        description="Increase local repair capability, then retry a bounded local replay.",
        slots={"scope": scope, "level": level, "replay_scope": replay_scope},
        steps=[
            RecoveryStep(op=OpType.ESCALATE, args={"scope": scope, "strategy": "stronger_prompt", "escalation_level": level}),
            RecoveryStep(
                op=OpType.REPLAY,
                args={
                    "scope": replay_scope,
                    "context_hint": "Keep the local target but retry with stronger local repair reasoning.",
                },
            ),
        ],
        rationale="Capability-boost recovery: improve local repair capability before replaying patch/verify.",
        estimated_total_cost=2400.0,
        estimated_recover_prob=0.72,
        estimated_risk=0.11,
        strategy="escalate_patcher_then_patch_replay",
    )


def _selective_replay_schema(*, role: str, cache_upstream: bool = True) -> ActionSchema:
    """MAS-native selective-replay primitive (Path-B).

    Re-runs one role while honoring cached upstream outputs for the rest of
    the stage chain. Unlike `_local_replan_schema`, this does NOT issue a
    ROLLBACK — the upstream propagation objects survive untouched, only the
    chosen role's downstream consumers are invalidated.
    """

    role_norm = str(role or "").strip().lower() or "patcher"
    return ActionSchema(
        schema_id=f"SELECTIVE_REPLAY::{role_norm}",
        family="selective_replay",
        description=(
            "Re-run one MAS role with cached upstream outputs. Invalidates "
            "only the typed propagation objects that role produces; "
            "surviving upstream facts are preserved."
        ),
        slots={
            "role": role_norm,
            "cache_upstream": bool(cache_upstream),
        },
        steps=[
            RecoveryStep(
                op=OpType.SELECTIVE_REPLAY,
                args={
                    "role": role_norm,
                    "cache_upstream": bool(cache_upstream),
                    "scope": role_norm,
                    "context_hint": (
                        f"Selective replay of the {role_norm} role. Upstream "
                        "stage outputs are cached and must not be regenerated. "
                        "Invalidate only this role's downstream consumers."
                    ),
                },
            ),
        ],
        rationale=(
            f"MAS-native primitive: re-run {role_norm} while keeping other "
            "roles' typed outputs cached. Keeps the next repair attempt tied "
            "to propagation-object evidence."
        ),
        estimated_total_cost=1500.0,
        estimated_recover_prob=0.65,
        estimated_risk=0.10,
        strategy=f"selective_replay_{role_norm}",
    )


def _checkpoint_id(checkpoint_ids: dict[str, str], label: str) -> str | None:
    value = checkpoint_ids.get(label)
    return str(value) if value else None


def discovery_schemas_v1(
    checkpoint_ids: dict[str, str],
    *,
    fault_family: str | None = None,
    include_selective_replay: bool = False,
) -> list[ActionSchema]:
    """Return parameterized recovery schemas for the current discovery phase.

    This intentionally mirrors `intent_discovery_v2`, but expresses candidates
    as schemas with slots so the next step can move toward constrained
    composition instead of remaining template-only.

    `include_selective_replay` is a Path-B gate. When True, a
    `_selective_replay_schema(role="patcher")` entry is appended. This is
    intentionally off by default so legacy benchmarks keep the exact program
    list they have always used.
    """

    normalized_fault_family = str(fault_family or "").strip().lower()
    schemas: list[ActionSchema] = []
    post_patch = _checkpoint_id(checkpoint_ids, "post_patch")
    post_locate = _checkpoint_id(checkpoint_ids, "post_locate")
    initial = _checkpoint_id(checkpoint_ids, "initial")
    if post_patch:
        schemas.append(_local_minimal_schema(checkpoint_id=post_patch))
    schemas.extend(
        [
            _belief_cleanup_schema(
                replay_scope="patcher+verifier",
                fact_id="fact:latest_patch",
            ),
            _evidence_recheck_schema(replay_scope="patcher+verifier"),
        ]
    )
    if normalized_fault_family == "contaminated_post_patch":
        schemas.append(
            _capability_boost_schema(
                replay_scope="patcher+verifier",
                scope="patcher",
                level=1,
            ),
        )
    if post_locate:
        schemas.append(
            _local_replan_schema(
                checkpoint_id=post_locate,
                replay_scope="patcher+verifier",
            )
        )
    if initial:
        schemas.append(_global_restart_schema(checkpoint_id=initial))
    if include_selective_replay:
        schemas.append(_selective_replay_schema(role="patcher", cache_upstream=True))
    return schemas


def rollback_only_schemas_v1(
    checkpoint_ids: dict[str, str],
) -> list[ActionSchema]:
    """Return the checkpoint-only rollback language baseline.

    This is intentionally strong enough to be a meaningful baseline, but it
    contains no non-rollback recovery intent.
    """

    schemas: list[ActionSchema] = []
    post_patch = _checkpoint_id(checkpoint_ids, "post_patch")
    post_locate = _checkpoint_id(checkpoint_ids, "post_locate")
    initial = _checkpoint_id(checkpoint_ids, "initial")
    if post_patch:
        schemas.append(_local_minimal_schema(checkpoint_id=post_patch))
    if post_locate:
        schemas.append(
            _local_replan_schema(
                checkpoint_id=post_locate,
                replay_scope="patcher+verifier",
            )
        )
    if initial:
        schemas.append(_global_restart_schema(checkpoint_id=initial))
    return schemas


def intent_schemas_v2(
    checkpoint_ids: dict[str, str],
) -> list[ActionSchema]:
    """Return the current frozen-core candidate language for baseline runs.

    `intent_schema_v2` is intentionally narrower than discovery:
    - keep the current stable core
    - include `evidence_recheck`, which now has incremental harder-family wins
    - exclude observe-only families until they earn promotion
    """

    schemas: list[ActionSchema] = []
    post_patch = _checkpoint_id(checkpoint_ids, "post_patch")
    post_locate = _checkpoint_id(checkpoint_ids, "post_locate")
    initial = _checkpoint_id(checkpoint_ids, "initial")
    if post_patch:
        schemas.append(_local_minimal_schema(checkpoint_id=post_patch))
    schemas.append(_evidence_recheck_schema(replay_scope="patcher+verifier"))
    if post_locate:
        schemas.append(
            _local_replan_schema(
                checkpoint_id=post_locate,
                replay_scope="patcher+verifier",
            )
        )
    if initial:
        schemas.append(_global_restart_schema(checkpoint_id=initial))
    return schemas


def intent_schemas_v3(
    checkpoint_ids: dict[str, str],
    *,
    fault_family: str | None = None,
) -> list[ActionSchema]:
    """Return the current promoted-core structured language candidate.

    `intent_schema_v3` promotes the latest live evidence:
    - keep `belief_cleanup` because it now succeeds when global fails
    - keep `evidence_recheck` because it has repeated best-family wins
    - keep `capability_boost` for harder contaminated cases only
    - retain `local_minimal`, `local_broader`, and `global`
    """

    normalized_fault_family = str(fault_family or "").strip().lower()
    schemas: list[ActionSchema] = []
    post_patch = _checkpoint_id(checkpoint_ids, "post_patch")
    post_locate = _checkpoint_id(checkpoint_ids, "post_locate")
    initial = _checkpoint_id(checkpoint_ids, "initial")
    if post_patch:
        schemas.append(_local_minimal_schema(checkpoint_id=post_patch))
    schemas.extend(
        [
            _belief_cleanup_schema(
                replay_scope="patcher+verifier",
                fact_id="fact:latest_patch",
            ),
            _evidence_recheck_schema(replay_scope="patcher+verifier"),
        ]
    )
    if normalized_fault_family == "contaminated_post_patch":
        schemas.append(
            _capability_boost_schema(
                replay_scope="patcher+verifier",
                scope="patcher",
                level=1,
            ),
        )
    if post_locate:
        schemas.append(
            _local_replan_schema(
                checkpoint_id=post_locate,
                replay_scope="patcher+verifier",
            )
        )
    if initial:
        schemas.append(_global_restart_schema(checkpoint_id=initial))
    return schemas


def discovery_programs_v1(
    checkpoint_ids: dict[str, str],
    *,
    fault_family: str | None = None,
    include_selective_replay: bool = False,
) -> list[RecoveryProgram]:
    """Materialize v1 schema-backed discovery programs."""
    schemas = discovery_schemas_v1(
        checkpoint_ids,
        fault_family=fault_family,
        include_selective_replay=include_selective_replay,
    )
    program_ids = {
        "local_minimal": "local_anchor_restore",
        "belief_cleanup": "belief_cleanup_patch_replay",
        "evidence_recheck": "evidence_recheck_patch_replay",
        "capability_boost": "capability_boost_patch_replay",
        "local_broader": "local_patch_replay",
        "global": "global_full_restart",
        "selective_replay": "selective_replay_patcher",
    }
    return [
        schema.to_program(program_id=program_ids.get(schema.family, schema.family))
        for schema in schemas
    ]


def rollback_only_programs_v1(
    checkpoint_ids: dict[str, str],
) -> list[RecoveryProgram]:
    """Materialize the rollback-only baseline language."""

    schemas = rollback_only_schemas_v1(checkpoint_ids)
    program_ids = {
        "local_minimal": "local_anchor_restore",
        "local_broader": "local_patch_replay",
        "global": "global_full_restart",
    }
    return [
        schema.to_program(program_id=program_ids.get(schema.family, schema.family))
        for schema in schemas
    ]


def intent_programs_v2(
    checkpoint_ids: dict[str, str],
) -> list[RecoveryProgram]:
    """Materialize the current frozen-core structured language candidate."""

    schemas = intent_schemas_v2(checkpoint_ids)
    program_ids = {
        "local_minimal": "local_anchor_restore",
        "evidence_recheck": "evidence_recheck_patch_replay",
        "local_broader": "local_patch_replay",
        "global": "global_full_restart",
    }
    return [
        schema.to_program(program_id=program_ids.get(schema.family, schema.family))
        for schema in schemas
    ]


def intent_programs_v3(
    checkpoint_ids: dict[str, str],
    *,
    fault_family: str | None = None,
) -> list[RecoveryProgram]:
    """Materialize the promoted-core structured language candidate."""

    schemas = intent_schemas_v3(checkpoint_ids, fault_family=fault_family)
    program_ids = {
        "local_minimal": "local_anchor_restore",
        "belief_cleanup": "belief_cleanup_patch_replay",
        "evidence_recheck": "evidence_recheck_patch_replay",
        "capability_boost": "capability_boost_patch_replay",
        "local_broader": "local_patch_replay",
        "global": "global_full_restart",
    }
    return [
        schema.to_program(program_id=program_ids.get(schema.family, schema.family))
        for schema in schemas
    ]
