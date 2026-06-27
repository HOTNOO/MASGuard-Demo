"""Action-space ablation config.

Gates each action family in the bounded recovery program space. Phase-0
exposes the six established families from `recovery.action_schema` plus
placeholder gates for the three Phase-4 MAS-native actions
(`inject_shared_fact`, `role_substitute`, `selective_replay`).

The Phase-4 gates are intentionally off by default and raise a clear error if
turned on in Phase-0. This lets the code wire through the execution path
today without silently shipping unimplemented actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcmr_swe.recovery.action_schema import (
    discovery_programs_v1,
    intent_programs_v2,
    intent_programs_v3,
    rollback_only_programs_v1,
)
from bcmr_swe.types import RecoveryProgram


_SUPPORTED_BASES: tuple[str, ...] = (
    "discovery_v1",
    "intent_v2",
    "intent_v3",
    "rollback_only_v1",
)


@dataclass(frozen=True, slots=True)
class ActionSpaceConfig:
    """Composable gate set for the bounded recovery program space."""

    cfg_id: str
    base: str = "discovery_v1"
    # Phase-0 family gates — default matches legacy behaviour.
    include_local_minimal: bool = True
    include_belief_cleanup: bool = True
    include_evidence_recheck: bool = True
    include_capability_boost: bool = True
    include_local_broader: bool = True
    include_global_restart: bool = True
    # Phase-4 MAS-native actions (not yet wired into the executor).
    include_inject_shared_fact: bool = False
    include_role_substitute: bool = False
    include_selective_replay: bool = False

    def __post_init__(self) -> None:
        assert self.cfg_id and isinstance(self.cfg_id, str), "cfg_id required"
        assert self.base in _SUPPORTED_BASES, f"unknown base: {self.base!r}"
        # Path-B note: `include_selective_replay` has a wired executor contract
        # (bcmr_swe.recovery.semantic_executor.CONSTRAINED_REPLAY with
        # cache_upstream=True) and is safe to enable. The other two Phase-4
        # gates (`inject_shared_fact`, `role_substitute`) remain unwired and
        # must continue to raise — silently accepting them would ship a
        # promise we cannot execute.
        if self.include_inject_shared_fact or self.include_role_substitute:
            raise NotImplementedError(
                "Phase-4 gates `inject_shared_fact` / `role_substitute` are "
                "not yet wired into the semantic executor. Enable them only "
                "after the executor contract is extended."
            )

    def build_programs(
        self,
        checkpoint_ids: dict[str, str],
        *,
        fault_family: str | None = None,
    ) -> list[RecoveryProgram]:
        """Return the filtered bounded program list.

        Selection proceeds in two stages:

        1. Materialize the base program list from the chosen family template.
        2. Keep only programs whose `metadata.family` matches an enabled gate.

        The post-filter is invariant-checked: the output must be non-empty,
        since an empty action space is always a protocol bug.
        """

        if self.base == "discovery_v1":
            programs = discovery_programs_v1(
                checkpoint_ids,
                fault_family=fault_family,
                include_selective_replay=self.include_selective_replay,
            )
        elif self.base == "intent_v2":
            programs = intent_programs_v2(checkpoint_ids)
        elif self.base == "intent_v3":
            programs = intent_programs_v3(checkpoint_ids, fault_family=fault_family)
        elif self.base == "rollback_only_v1":
            programs = rollback_only_programs_v1(checkpoint_ids)
        else:  # pragma: no cover -- guarded in __post_init__
            raise ValueError(f"unsupported base: {self.base}")

        enabled = self._enabled_families()
        filtered = [
            program
            for program in programs
            if str(program.metadata.get("family", "")) in enabled
        ]
        assert filtered, (
            f"ActionSpaceConfig {self.cfg_id!r} produced empty program list from base "
            f"{self.base!r}; enable at least one family gate"
        )
        return filtered

    def _enabled_families(self) -> frozenset[str]:
        mapping = {
            "local_minimal": self.include_local_minimal,
            "belief_cleanup": self.include_belief_cleanup,
            "evidence_recheck": self.include_evidence_recheck,
            "capability_boost": self.include_capability_boost,
            "local_broader": self.include_local_broader,
            "global": self.include_global_restart,
            "selective_replay": self.include_selective_replay,
        }
        return frozenset({name for name, flag in mapping.items() if flag})

    def summary_for_event_log(self) -> dict[str, Any]:
        return {
            "cfg_id": self.cfg_id,
            "base": self.base,
            "enabled_families": sorted(self._enabled_families()),
            "phase4_reserved": {
                "inject_shared_fact": self.include_inject_shared_fact,
                "role_substitute": self.include_role_substitute,
                "selective_replay": self.include_selective_replay,
            },
        }
