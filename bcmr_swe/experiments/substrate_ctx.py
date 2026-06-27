"""Substrate contract for live recovery runs.

`CaseSubstrateCtx` is the shared setup protocol that every live-replay cell
of the 2x2 matrix uses. It guarantees by construction, not by convention:

- same fault checkpoint restored into each cell,
- same workspace-materialization strategy,
- same `test_command` / `oracle_command`,
- same model and runtime lock.

Phase-1 drivers (`live_program_driver`, `reflexion_k_driver`) use this
context instead of re-implementing the freeform setup surface. The coordinator
surface (`coord._resume_from(...)`) is reused verbatim — Phase-1 adds no new
executor code.

Design note: this is NOT a general-purpose harness. It is a narrow contract
scoped to what the 2x2 matrix needs. Keeping it narrow lets the invariants
actually fire; a general-purpose harness would have to relax them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bcmr_swe.experiments.common import (
    build_coordinator,
    build_executor,
    materialize_workspace,
    workspace_strategy_for_runtime,
)
from bcmr_swe.observability import (
    RunEventLogger,
    assert_checkpoint_restored,
    assert_workspace_reset,
    assert_manifest_lock,
)
from bcmr_swe.types import CheckpointRecord


def _chat_model_identifier(model: Any) -> str:
    """Return the model-name string for an OpenAI / Anthropic / Gemini
    compatible chat model built by `build_chat_model`.

    The surface layer uses `model.config.model` for all provider wrappers; a
    top-level `model_name` attribute is not reliably present. The invariant
    in `assert_manifest_lock` compares against this string exactly, so every
    caller must agree on one accessor.
    """
    if model is None:
        return ""
    for attr_chain in (("config", "model"), ("model",), ("model_name",)):
        cursor: Any = model
        ok = True
        for attr in attr_chain:
            cursor = getattr(cursor, attr, None)
            if cursor is None:
                ok = False
                break
        if ok and isinstance(cursor, str) and cursor.strip():
            return cursor.strip()
    return ""


@dataclass(slots=True)
class LiveRunResult:
    """Outcome of a single live-replay attempt."""

    recover_success: bool
    agent_reported_success: bool
    fail_to_pass_returncode: int
    oracle_returncode: int
    fail_to_pass_output: str
    oracle_output: str
    workspace_path: str
    used_checkpoint: dict[str, Any]
    token_cost: float
    latency_sec: float
    stage_trace: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recover_success": self.recover_success,
            "agent_reported_success": self.agent_reported_success,
            "fail_to_pass_returncode": self.fail_to_pass_returncode,
            "oracle_returncode": self.oracle_returncode,
            "fail_to_pass_output": self.fail_to_pass_output,
            "oracle_output": self.oracle_output,
            "workspace_path": self.workspace_path,
            "used_checkpoint": dict(self.used_checkpoint),
            "token_cost": self.token_cost,
            "latency_sec": self.latency_sec,
            "stage_trace": dict(self.stage_trace),
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class CaseSubstrateCtx:
    """Shared setup contract for a single live recovery cell.

    Construct once per (case, cell) pair, call `setup()`, then hand the
    resulting coordinator / executor to a driver. Call `close()` in a finally
    block. Invariants fire on every boundary — dirty workspace, commit drift,
    model/runtime mismatch will raise `InvariantViolation`.
    """

    case_id: str
    manifest: dict[str, Any]
    fault_checkpoint: CheckpointRecord
    located_files: str
    workspace_root: Path
    workspace_suffix: str
    runtime: str
    model: Any
    strong_model: Any
    expected_model_name: str = ""
    expected_strong_model_name: str = ""
    force_rebuild_harness: bool = False
    harness_setup_timeout: int | None = None
    harness_container_start_timeout: int | None = None
    harness_container_cleanup_timeout: int | None = None
    locator_iters: int = 1
    planner_iters: int = 2
    patcher_iters: int = 2
    verifier_iters: int = 1
    event_logger: RunEventLogger | None = None
    _coord: Any = field(default=None, init=False)
    _executor: Any = field(default=None, init=False)
    _runtime_session: Any = field(default=None, init=False)
    _workspace: Path | None = field(default=None, init=False)
    _is_open: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert self.case_id, "CaseSubstrateCtx requires a non-empty case_id"
        assert isinstance(self.manifest, dict) and self.manifest, "manifest required"
        assert isinstance(self.fault_checkpoint, CheckpointRecord), "fault_checkpoint required"
        assert self.workspace_root, "workspace_root required"
        assert self.runtime, "runtime required"
        assert self.model is not None, "model required"

    @property
    def coord(self) -> Any:
        assert self._is_open, "CaseSubstrateCtx not yet open — call setup() first"
        return self._coord

    @property
    def executor(self) -> Any:
        assert self._is_open, "CaseSubstrateCtx not yet open — call setup() first"
        return self._executor

    @property
    def workspace(self) -> Path:
        assert self._workspace is not None, "CaseSubstrateCtx has no workspace yet"
        return self._workspace

    def setup(self) -> None:
        if self._is_open:
            raise RuntimeError("CaseSubstrateCtx already open")

        # Governance / lock invariants before we touch disk.
        assert_manifest_lock(
            expected_model=self.expected_model_name,
            built_model_name=_chat_model_identifier(self.model),
            expected_strong_model=self.expected_strong_model_name,
            built_strong_model_name=_chat_model_identifier(self.strong_model),
            runtime=self.runtime,
            logger=self.event_logger,
        )

        oracle_snapshot = Path(
            str(
                self.manifest.get("oracle_snapshot")
                or (str(self.manifest["source_snapshot"]) + "__oracle")
            )
        ).resolve()
        instance_id = str(self.manifest.get("instance_id") or self.case_id)
        self._workspace = materialize_workspace(
            oracle_snapshot,
            self.workspace_root,
            f"{instance_id}_{self.workspace_suffix}",
            strategy=workspace_strategy_for_runtime(self.runtime, self.manifest),
        )

        assert_workspace_reset(self._workspace, allow_nonempty=True, logger=self.event_logger)

        self._executor, self._runtime_session = build_executor(
            workspace=str(self._workspace),
            runtime=self.runtime,
            manifest=self.manifest,
            force_rebuild_harness=self.force_rebuild_harness,
            harness_setup_timeout=self.harness_setup_timeout,
            harness_container_start_timeout=self.harness_container_start_timeout,
            harness_container_cleanup_timeout=self.harness_container_cleanup_timeout,
        )
        self._coord = build_coordinator(
            workspace=str(self._workspace),
            model=self.model,
            strong_model=self.strong_model,
            strong_stages=("planner", "implementer"),
            executor=self._executor,
            recovery_mode="v3_program",
            locator_max_iterations=self.locator_iters,
            planner_max_iterations=self.planner_iters,
            patcher_max_iterations=self.patcher_iters,
            verifier_max_iterations=self.verifier_iters,
        )

        from bcmr_swe.experiments.run_recovery_only_benchmark import _bootstrap_run_context

        _bootstrap_run_context(
            self._coord,
            manifest=self.manifest,
            workspace=self._workspace,
            executor=self._executor,
        )
        self._coord._official_only_replay_verifier = True
        self._coord._recovery_diagnosis_mode = "natural_failed_state_v1"

        self._coord._recovery_enabled = False
        self._coord.stage_outputs["locator"] = {
            "success": True,
            "located_files": self.located_files,
        }
        self._coord.recorder.checkpoint_store.restore(self.fault_checkpoint, self._coord.workspace)

        assert_checkpoint_restored(
            self._workspace,
            expected_commit=str(self.fault_checkpoint.metadata.get("commit") or "") or None,
            allow_dirty=True,  # fault checkpoints intentionally carry contamination
            logger=self.event_logger,
        )

        self._is_open = True

        if self.event_logger is not None:
            self.event_logger.emit(
                "checkpoint_restored",
                payload={
                    "label": self.fault_checkpoint.label,
                    "commit": str(self.fault_checkpoint.metadata.get("commit") or ""),
                    "workspace": str(self._workspace),
                    "located_files_excerpt": (self.located_files or "")[:400],
                },
            )

    def run_oracle(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the canonical `test_command` + `oracle_command` pair.

        The two commands come from the manifest; this is the single source of
        truth for "did the cell succeed" across every cell of the matrix.
        """

        assert self._is_open, "CaseSubstrateCtx.setup() must be called first"
        fail_to_pass = self._executor.execute(
            str(self.manifest["test_command"]),
            cwd=str(self._workspace),
            timeout=1800,
        )
        oracle = self._executor.execute(
            str(self.manifest["oracle_command"]),
            cwd=str(self._workspace),
            timeout=1800,
        )
        return fail_to_pass, oracle

    def harness_metadata(self) -> dict[str, Any]:
        session = self._runtime_session
        if session is None or not hasattr(session, "image_metadata"):
            return {}
        metadata = session.image_metadata()
        return dict(metadata or {}) if isinstance(metadata, dict) else {}

    def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        try:
            if self._runtime_session is not None:
                self._runtime_session.close()
        except Exception:
            pass

    def __enter__(self) -> "CaseSubstrateCtx":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def summarize_compact_stage_trace(stage_outputs: dict[str, Any]) -> dict[str, Any]:
    """Compact stage trace for event-log and benchmark row usage.

    Duplicates the compaction used in `run_structured_vs_freeform_recovery.py`
    so Phase-1 drivers do not depend on a private helper.
    """

    trace: dict[str, Any] = {}
    for stage_name in ("locator", "patcher", "verifier"):
        payload = dict((stage_outputs or {}).get(stage_name, {}) or {})
        if not payload:
            continue
        trace[stage_name] = {
            "success": bool(payload.get("success")),
            "commands": list(payload.get("commands", []) or []),
            "messages": list(payload.get("messages", []) or []),
            "planner_messages": list(payload.get("planner_messages", []) or []),
            "status": payload.get("status"),
        }
    return trace


def workspace_patch_summary(
    workspace: Path,
    suspect_paths: list[str],
    *,
    is_test_path_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Compute `patch_scope` / `touches_suspect_path` for a live workspace.

    Uses `git diff --numstat` twice (unstaged + staged) plus untracked
    files. Pure-mode-only entries are ignored so oracle snapshot mode flips
    don't pollute patch legitimacy.
    """

    def _run_git(*args: str) -> list[str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def _content_changed_files(*args: str) -> set[str]:
        changed: set[str] = set()
        for line in _run_git(*args):
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added, deleted, path = (part.strip() for part in parts)
            if (added, deleted) in {("0", "0"), ("-", "-")}:
                continue
            if path:
                changed.add(path)
        return changed

    def _default_is_test(path: str) -> bool:
        normalized = path.replace("\\", "/")
        name = normalized.rsplit("/", 1)[-1]
        return (
            normalized.startswith("tests/")
            or "/tests/" in normalized
            or name.startswith("test_")
            or name.endswith("_test.py")
        )

    is_test = is_test_path_fn or _default_is_test

    changed = _content_changed_files("diff", "--numstat")
    changed.update(_content_changed_files("diff", "--cached", "--numstat"))
    changed.update(_run_git("ls-files", "--others", "--exclude-standard"))
    changed_files = sorted(changed)
    test_files = [path for path in changed_files if is_test(path)]
    non_test_files = [path for path in changed_files if not is_test(path)]
    normalized_suspects = {p.replace("\\", "/") for p in suspect_paths}
    changed_normalized = {p.replace("\\", "/") for p in changed_files}
    overlap = sorted(changed_normalized.intersection(normalized_suspects))

    if not changed_files:
        patch_scope = "no_diff"
    elif test_files and not non_test_files:
        patch_scope = "tests_only"
    elif non_test_files and not test_files:
        patch_scope = "non_test_only"
    else:
        patch_scope = "mixed"

    return {
        "patch_scope": patch_scope,
        "changed_files": changed_files,
        "test_files": test_files,
        "non_test_files": non_test_files,
        "suspect_paths": list(suspect_paths),
        "suspect_path_overlap": overlap,
        "touches_suspect_path": bool(overlap),
    }
