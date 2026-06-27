"""Verifier adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class VerifierProtocol(Protocol):
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
class LegacyVerifierAdapter:
    """Reuse the existing swe_mas verifier."""

    model: Any
    executor: Any
    recorder: Any | None = None
    session_id: str | None = None
    max_iterations: int = 6

    def verify(
        self,
        issue: str,
        workspace: str,
        *,
        patch: str,
        recovery_context: str = "",
        deep_verify: bool = False,
    ) -> dict[str, Any]:
        from swe_mas.agents.verifier import VerifierAgent

        verifier = VerifierAgent(
            model=self.model,
            executor=self.executor,
            recorder=self.recorder,
            session_id=self.session_id,
        )
        verifier.config.max_iterations = self.max_iterations + (1 if deep_verify else 0)
        issue_with_context = issue.strip()
        if recovery_context:
            issue_with_context = f"{recovery_context}\n\n---\n\n{issue_with_context}"
        return verifier.run(
            problem_statement=issue_with_context,
            implementation=patch,
            cwd=workspace,
        )
