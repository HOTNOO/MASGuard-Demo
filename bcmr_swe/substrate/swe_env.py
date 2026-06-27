"""SWE workspace environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcmr_swe.provenance.checkpoint_store import WorkspaceCheckpointStore
from swe_mas.utils.env_executor import LocalExecutor


@dataclass
class SWEEnvironment:
    """Simple environment wrapper that reuses the existing local executor."""

    workspace: str
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 120

    def __post_init__(self) -> None:
        self.workspace = str(Path(self.workspace).resolve())
        self.executor = LocalExecutor(cwd=self.workspace, env=self.env, timeout=self.timeout)
        self.checkpoints = WorkspaceCheckpointStore(Path(self.workspace) / ".bcmr_checkpoints")

    def run(self, command: str, *, timeout: int | None = None) -> dict[str, Any]:
        return self.executor.execute(command, timeout=timeout)
