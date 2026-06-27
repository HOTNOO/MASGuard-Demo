"""Static PARC action cards.

Action cards keep the semantic action space small and auditable.  They are
method-level metadata, not case-specific policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bcmr_swe.types import RecoveryLedger, SemanticActionType


@dataclass(frozen=True, slots=True)
class ActionCard:
    action: SemanticActionType
    cost_class: str
    base_token_estimate: int
    preconditions: tuple[str, ...]
    predicted_mutations: tuple[str, ...]
    repeat_limit: int
    stop_if: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "cost_class": self.cost_class,
            "base_token_estimate": self.base_token_estimate,
            "preconditions": list(self.preconditions),
            "predicted_mutations": list(self.predicted_mutations),
            "repeat_limit": self.repeat_limit,
            "stop_if": list(self.stop_if),
        }


ACTION_CARDS: dict[SemanticActionType, ActionCard] = {
    SemanticActionType.EVIDENCE_RECHECK: ActionCard(
        action=SemanticActionType.EVIDENCE_RECHECK,
        cost_class="cheap",
        base_token_estimate=200,
        preconditions=("failure_evidence_available",),
        predicted_mutations=("ledger.suspect_paths", "object.lifecycle_state"),
        repeat_limit=2,
        stop_if=("last_evidence_recheck_no_change",),
    ),
    SemanticActionType.RECHECK_OBJECT: ActionCard(
        action=SemanticActionType.RECHECK_OBJECT,
        cost_class="cheap",
        base_token_estimate=200,
        preconditions=("active_object.state in [suspicious, invalidated]",),
        predicted_mutations=("object.lifecycle_state",),
        repeat_limit=2,
        stop_if=("object_terminal",),
    ),
    SemanticActionType.TARGET_RESET: ActionCard(
        action=SemanticActionType.TARGET_RESET,
        cost_class="cheap",
        base_token_estimate=0,
        preconditions=("viable_replay_anchor",),
        predicted_mutations=("ledger.active_object_id", "ledger.replay_anchor"),
        repeat_limit=1,
        stop_if=("no_alternative_anchor",),
    ),
    SemanticActionType.REVOKE_OBJECT: ActionCard(
        action=SemanticActionType.REVOKE_OBJECT,
        cost_class="cheap",
        base_token_estimate=0,
        preconditions=("active_object.state == suspicious",),
        predicted_mutations=("object.lifecycle_state",),
        repeat_limit=1,
        stop_if=("object_already_invalidated",),
    ),
    SemanticActionType.LOCAL_REPAIR: ActionCard(
        action=SemanticActionType.LOCAL_REPAIR,
        cost_class="expensive",
        base_token_estimate=30000,
        preconditions=("active_object.state in [suspicious, invalidated]", "replay_anchor_available"),
        predicted_mutations=("patch_candidate", "source_diff"),
        repeat_limit=3,
        stop_if=("local_repair_repeated_no_diff",),
    ),
    SemanticActionType.SCOPE_EXPAND: ActionCard(
        action=SemanticActionType.SCOPE_EXPAND,
        cost_class="expensive",
        base_token_estimate=50000,
        preconditions=("localization_not_global",),
        predicted_mutations=("ledger.localization", "patch_candidate"),
        repeat_limit=2,
        stop_if=("scope_already_global",),
    ),
    SemanticActionType.CAPABILITY_BOOST: ActionCard(
        action=SemanticActionType.CAPABILITY_BOOST,
        cost_class="medium",
        base_token_estimate=10000,
        preconditions=("execution_profile != boosted",),
        predicted_mutations=("ledger.execution_profile",),
        repeat_limit=1,
        stop_if=("capability_already_boosted",),
    ),
}


CANONICAL_PARC_ACTIONS: tuple[SemanticActionType, ...] = (
    SemanticActionType.EVIDENCE_RECHECK,
    SemanticActionType.RECHECK_OBJECT,
    SemanticActionType.TARGET_RESET,
    SemanticActionType.REVOKE_OBJECT,
    SemanticActionType.LOCAL_REPAIR,
    SemanticActionType.SCOPE_EXPAND,
    SemanticActionType.CAPABILITY_BOOST,
)


def get_action_card(action: str | SemanticActionType) -> ActionCard:
    normalized = SemanticActionType(str(action.value if isinstance(action, SemanticActionType) else action))
    if normalized in {SemanticActionType.REPAIR_LOCAL}:
        normalized = SemanticActionType.LOCAL_REPAIR
    if normalized in {SemanticActionType.EXPAND_SCOPE}:
        normalized = SemanticActionType.SCOPE_EXPAND
    return ACTION_CARDS[normalized]


def action_cost_class(action: str | SemanticActionType) -> str:
    return get_action_card(action).cost_class


def action_repeat_count(ledger: RecoveryLedger, action: str | SemanticActionType) -> int:
    action_value = str(action.value if isinstance(action, SemanticActionType) else action)
    return sum(1 for item in list(ledger.tried_actions or []) if str(item) == action_value)


def repeat_limit_exceeded(ledger: RecoveryLedger, action: str | SemanticActionType) -> bool:
    card = get_action_card(action)
    return action_repeat_count(ledger, card.action) >= card.repeat_limit


def available_action_cards(ledger: RecoveryLedger) -> list[ActionCard]:
    cards: list[ActionCard] = []
    for action in CANONICAL_PARC_ACTIONS:
        card = ACTION_CARDS[action]
        if repeat_limit_exceeded(ledger, action):
            continue
        if action == SemanticActionType.CAPABILITY_BOOST and str(ledger.execution_profile or "") == "boosted":
            continue
        cards.append(card)
    return cards
