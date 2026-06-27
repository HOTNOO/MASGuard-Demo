"""Locator adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LocatorProtocol(Protocol):
    def locate(self, issue: str, workspace: str, *, recovery_context: str = "", escalation_level: int = 0) -> dict[str, Any]:
        ...


@dataclass
class LegacyLocatorAdapter:
    """Reuse the existing swe_mas locator when available."""

    model: Any
    executor: Any
    recorder: Any | None = None
    session_id: str | None = None
    max_iterations: int = 8

    def locate(self, issue: str, workspace: str, *, recovery_context: str = "", escalation_level: int = 0) -> dict[str, Any]:
        from swe_mas.agents.locator import LocatorAgent

        agent = LocatorAgent(
            model=self.model,
            executor=self.executor,
            recorder=self.recorder,
            session_id=self.session_id,
        )
        agent.config.max_iterations = max(self.max_iterations, self.max_iterations + escalation_level)
        analysis = issue.strip()
        if recovery_context:
            analysis = f"{recovery_context}\n\n---\n\n{analysis}"
        return agent.run(problem_analysis=analysis, cwd=workspace)
