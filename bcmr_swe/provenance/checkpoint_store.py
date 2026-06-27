"""Workspace checkpoint store backed by tar archives."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import uuid

from bcmr_swe.types import CheckpointRecord
from swe_mas.utils.git_utils import get_env_snapshot


class WorkspaceCheckpointStore:
    """Create and restore workspace snapshots without touching .git state."""

    DEFAULT_EXCLUDES = {
        ".git",
        ".venv",
        ".pydeps",
        "__pycache__",
        ".pytest_cache",
        "outputs",
        "data",
    }

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.archives_dir = self.root_dir / "archives"
        self.metadata_dir = self.root_dir / "metadata"
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        workspace: str | Path,
        *,
        label: str,
        metadata: dict | None = None,
        source_node_id: str = "",
    ) -> CheckpointRecord:
        workspace = Path(workspace).resolve()
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:10]}"
        archive_path = self.archives_dir / f"{checkpoint_id}.tar.gz"
        metadata_path = self.metadata_dir / f"{checkpoint_id}.json"

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in workspace.iterdir():
                if item.name in self.DEFAULT_EXCLUDES:
                    continue
                tar.add(item, arcname=item.name)

        payload = {
            "checkpoint_id": checkpoint_id,
            "label": label,
            "workspace": str(workspace),
            "env_snapshot_hash": get_env_snapshot(str(workspace)),
            "created_at": time.time(),
            "metadata": metadata or {},
            "source_node_id": source_node_id,
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            label=label,
            workspace=str(workspace),
            archive_path=str(archive_path),
            metadata_path=str(metadata_path),
            env_snapshot_hash=payload["env_snapshot_hash"],
            created_at=payload["created_at"],
            metadata=payload["metadata"],
            source_node_id=source_node_id,
        )

    def restore(self, checkpoint: CheckpointRecord, workspace: str | Path) -> None:
        workspace = Path(workspace).resolve()
        archive_path = Path(checkpoint.archive_path)
        self._clear_workspace(workspace)
        with tarfile.open(archive_path, "r:gz") as tar:
            # Checkpoints are local artifacts created by this store.  Python's
            # data filter rejects absolute-path symlinks (for example tox
            # virtualenv links), which turns otherwise valid failed-state
            # replays into substrate errors.  Use the trusted filter for this
            # closed-loop provenance archive.
            tar.extractall(path=workspace, filter="fully_trusted")

    def _clear_workspace(self, workspace: Path) -> None:
        protected_names = set(self.DEFAULT_EXCLUDES)
        try:
            relative_root = self.root_dir.resolve().relative_to(workspace.resolve())
            if relative_root.parts:
                protected_names.add(relative_root.parts[0])
        except ValueError:
            pass

        for item in workspace.iterdir():
            if item.name in protected_names:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
