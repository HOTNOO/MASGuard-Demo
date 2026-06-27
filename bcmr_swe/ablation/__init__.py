"""BCMR ablation subsystem.

Composable dataclasses that gate state views and action families, plus a
registry of named experiment cells. See:

- `state_projection.py` — StateProjectionConfig
- `action_space.py` — ActionSpaceConfig
- `configs.py` — named cells and resolver helpers
"""

from __future__ import annotations

from bcmr_swe.ablation.action_space import ActionSpaceConfig
from bcmr_swe.ablation.configs import (
    ACT_FULL,
    ACT_INTENT_V2,
    ACT_INTENT_V3,
    ACT_NO_BELIEF_CLEANUP,
    ACT_NO_CAPABILITY_BOOST,
    ACT_NO_EVIDENCE_RECHECK,
    ACT_NO_GLOBAL,
    ACT_ROLLBACK_ONLY,
    ACTION_SPACE_CELLS,
    CFG_CONTENT_BULLETS,
    CFG_CONTENT_JSON,
    CFG_CONTENT_PROSE,
    CFG_FULL,
    CFG_NO_EVIDENCE_PACK,
    CFG_NO_NEG_CONSTRAINTS,
    CFG_NO_OBJECT_CHAIN,
    CFG_NO_REPLAY_ANCHOR,
    CFG_NO_ROLE_AGGREGATE,
    CFG_OBJECT_CHAIN_ONLY,
    CFG_ROLE_SUMMARY_ONLY,
    PROJECTION_CELLS,
    list_action_space_cell_ids,
    list_projection_cell_ids,
    resolve_action_space_cfg,
    resolve_projection_cfg,
)
from bcmr_swe.ablation.state_projection import StateProjectionConfig

__all__ = [
    "ActionSpaceConfig",
    "StateProjectionConfig",
    "CFG_FULL",
    "CFG_NO_OBJECT_CHAIN",
    "CFG_NO_ROLE_AGGREGATE",
    "CFG_NO_REPLAY_ANCHOR",
    "CFG_NO_EVIDENCE_PACK",
    "CFG_NO_NEG_CONSTRAINTS",
    "CFG_OBJECT_CHAIN_ONLY",
    "CFG_ROLE_SUMMARY_ONLY",
    "CFG_CONTENT_JSON",
    "CFG_CONTENT_BULLETS",
    "CFG_CONTENT_PROSE",
    "ACT_FULL",
    "ACT_INTENT_V2",
    "ACT_INTENT_V3",
    "ACT_NO_BELIEF_CLEANUP",
    "ACT_NO_EVIDENCE_RECHECK",
    "ACT_NO_CAPABILITY_BOOST",
    "ACT_NO_GLOBAL",
    "ACT_ROLLBACK_ONLY",
    "PROJECTION_CELLS",
    "ACTION_SPACE_CELLS",
    "resolve_projection_cfg",
    "resolve_action_space_cfg",
    "list_projection_cell_ids",
    "list_action_space_cell_ids",
]
