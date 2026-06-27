"""Named ablation cells for Phase-0.

Each cell is a fully-specified config. Benchmark runners resolve a cfg_id
string to its cell via `resolve_projection_cfg` / `resolve_action_space_cfg`.

Rules:

- `CFG_FULL` is the legacy-compatible default. All Phase-0 scripts must
  produce byte-identical outputs under `CFG_FULL` vs. the pre-refactor
  behaviour, because the legacy benchmarks still reference those numbers.
- Leave-one-out cells isolate exactly one view or one family. Nothing else
  changes — same `token_budget`, same `object_chain_max`.
- Content-controlled cells hold content identical and vary only render
  format. This is the C1 ablation that tests whether structure has causal
  value beyond prose organization.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from bcmr_swe.ablation.action_space import ActionSpaceConfig
from bcmr_swe.ablation.state_projection import StateProjectionConfig


CFG_FULL: StateProjectionConfig = StateProjectionConfig(cfg_id="FULL")

# Leave-one-out view ablations (C1 mechanism isolation).
CFG_NO_OBJECT_CHAIN: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="NO_OBJECT_CHAIN", include_object_chain_view=False
)
CFG_NO_ROLE_AGGREGATE: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="NO_ROLE_AGGREGATE", include_role_aggregate_view=False
)
CFG_NO_REPLAY_ANCHOR: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="NO_REPLAY_ANCHOR", include_replay_anchor_view=False
)
CFG_NO_EVIDENCE_PACK: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="NO_EVIDENCE_PACK", include_evidence_pack=False
)
CFG_NO_NEG_CONSTRAINTS: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="NO_NEG_CONSTRAINTS", include_negative_constraints=False
)

# Keep-only-one cells (for object-chain vs. role-summary at equal tokens).
CFG_OBJECT_CHAIN_ONLY: StateProjectionConfig = StateProjectionConfig(
    cfg_id="OBJECT_CHAIN_ONLY",
    include_local_region_view=False,
    include_role_aggregate_view=False,
    include_object_chain_view=True,
    include_replay_anchor_view=False,
    include_evidence_pack=False,
    include_negative_constraints=False,
)
CFG_ROLE_SUMMARY_ONLY: StateProjectionConfig = StateProjectionConfig(
    cfg_id="ROLE_SUMMARY_ONLY",
    include_local_region_view=False,
    include_role_aggregate_view=True,
    include_object_chain_view=False,
    include_replay_anchor_view=False,
    include_evidence_pack=False,
    include_negative_constraints=False,
)

# Content-controlled structure ablation (C1 causal test).
CFG_CONTENT_JSON: StateProjectionConfig = replace(CFG_FULL, cfg_id="CONTENT_CTRL_JSON", render_format="json")
CFG_CONTENT_BULLETS: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="CONTENT_CTRL_BULLETS", render_format="bullets"
)
CFG_CONTENT_PROSE: StateProjectionConfig = replace(
    CFG_FULL, cfg_id="CONTENT_CTRL_PROSE", render_format="prose"
)


PROJECTION_CELLS: dict[str, StateProjectionConfig] = {
    cfg.cfg_id: cfg
    for cfg in (
        CFG_FULL,
        CFG_NO_OBJECT_CHAIN,
        CFG_NO_ROLE_AGGREGATE,
        CFG_NO_REPLAY_ANCHOR,
        CFG_NO_EVIDENCE_PACK,
        CFG_NO_NEG_CONSTRAINTS,
        CFG_OBJECT_CHAIN_ONLY,
        CFG_ROLE_SUMMARY_ONLY,
        CFG_CONTENT_JSON,
        CFG_CONTENT_BULLETS,
        CFG_CONTENT_PROSE,
    )
}


ACT_FULL: ActionSpaceConfig = ActionSpaceConfig(cfg_id="ACT_FULL", base="discovery_v1")

ACT_NO_BELIEF_CLEANUP: ActionSpaceConfig = replace(
    ACT_FULL, cfg_id="ACT_NO_BELIEF_CLEANUP", include_belief_cleanup=False
)
ACT_NO_EVIDENCE_RECHECK: ActionSpaceConfig = replace(
    ACT_FULL, cfg_id="ACT_NO_EVIDENCE_RECHECK", include_evidence_recheck=False
)
ACT_NO_CAPABILITY_BOOST: ActionSpaceConfig = replace(
    ACT_FULL, cfg_id="ACT_NO_CAPABILITY_BOOST", include_capability_boost=False
)
ACT_NO_GLOBAL: ActionSpaceConfig = replace(
    ACT_FULL, cfg_id="ACT_NO_GLOBAL", include_global_restart=False
)
ACT_ROLLBACK_ONLY: ActionSpaceConfig = ActionSpaceConfig(
    cfg_id="ACT_ROLLBACK_ONLY",
    base="rollback_only_v1",
    include_belief_cleanup=False,
    include_evidence_recheck=False,
    include_capability_boost=False,
)
ACT_INTENT_V2: ActionSpaceConfig = ActionSpaceConfig(cfg_id="ACT_INTENT_V2", base="intent_v2")
ACT_INTENT_V3: ActionSpaceConfig = ActionSpaceConfig(cfg_id="ACT_INTENT_V3", base="intent_v3")

# Path-B MAS-native primitive cells.
# `ACT_SELECTIVE_REPLAY_ON` keeps the legacy Phase-0 families and adds the new
# `selective_replay` family. Used when R2a decides Path-B is worth pursuing.
ACT_SELECTIVE_REPLAY_ON: ActionSpaceConfig = replace(
    ACT_FULL,
    cfg_id="ACT_SELECTIVE_REPLAY_ON",
    include_selective_replay=True,
)
# `ACT_SELECTIVE_REPLAY_ONLY` isolates the primitive for leave-one-in
# attribution — everything else is off.
ACT_SELECTIVE_REPLAY_ONLY: ActionSpaceConfig = ActionSpaceConfig(
    cfg_id="ACT_SELECTIVE_REPLAY_ONLY",
    base="discovery_v1",
    include_local_minimal=False,
    include_belief_cleanup=False,
    include_evidence_recheck=False,
    include_capability_boost=False,
    include_local_broader=False,
    include_global_restart=False,
    include_selective_replay=True,
)


ACTION_SPACE_CELLS: dict[str, ActionSpaceConfig] = {
    cfg.cfg_id: cfg
    for cfg in (
        ACT_FULL,
        ACT_NO_BELIEF_CLEANUP,
        ACT_NO_EVIDENCE_RECHECK,
        ACT_NO_CAPABILITY_BOOST,
        ACT_NO_GLOBAL,
        ACT_ROLLBACK_ONLY,
        ACT_INTENT_V2,
        ACT_INTENT_V3,
        ACT_SELECTIVE_REPLAY_ON,
        ACT_SELECTIVE_REPLAY_ONLY,
    )
}


def resolve_projection_cfg(cfg_id: str) -> StateProjectionConfig:
    """Resolve a cfg_id string to a named StateProjectionConfig cell."""
    if cfg_id not in PROJECTION_CELLS:
        raise KeyError(
            f"unknown projection cfg_id {cfg_id!r}; "
            f"known = {sorted(PROJECTION_CELLS)}"
        )
    return PROJECTION_CELLS[cfg_id]


def resolve_action_space_cfg(cfg_id: str) -> ActionSpaceConfig:
    """Resolve a cfg_id string to a named ActionSpaceConfig cell."""
    if cfg_id not in ACTION_SPACE_CELLS:
        raise KeyError(
            f"unknown action-space cfg_id {cfg_id!r}; "
            f"known = {sorted(ACTION_SPACE_CELLS)}"
        )
    return ACTION_SPACE_CELLS[cfg_id]


def list_projection_cell_ids() -> list[str]:
    return sorted(PROJECTION_CELLS.keys())


def list_action_space_cell_ids() -> list[str]:
    return sorted(ACTION_SPACE_CELLS.keys())
