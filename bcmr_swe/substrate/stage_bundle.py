"""Stage-bundle interfaces for plugging BCMR into different execution substrates.

BCMR should depend on a thin stage-level substrate contract instead of a
specific MAS implementation.  A concrete MAS/framework can expose locator,
patcher, and verifier stages through this bundle, while BCMR remains the
sidecar/middleware layer that drives recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LocatorStage(Protocol):
    def locate(
        self,
        issue: str,
        workspace: str,
        *,
        recovery_context: str = "",
        escalation_level: int = 0,
    ) -> dict[str, Any]:
        ...


class PatcherStage(Protocol):
    def patch(
        self,
        issue: str,
        workspace: str,
        *,
        located_files: str,
        recovery_context: str = "",
        escalation_level: int = 0,
    ) -> dict[str, Any]:
        ...


class VerifierStage(Protocol):
    def verify(
        self,
        issue: str,
        workspace: str,
        *,
        patch: str,
        recovery_context: str = "",
        deep_verify: bool = False,
    ) -> dict[str, Any]:
        ...


@dataclass
class StageBundle:
    """A substrate-specific bundle of stages exposed to BCMR."""

    locator: LocatorStage
    patcher: PatcherStage
    verifier: VerifierStage
    backend_name: str = "unknown"
