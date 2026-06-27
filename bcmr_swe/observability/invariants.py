"""Research-grade invariant checks for BCMR critical paths.

Every check raises `InvariantViolation` with enough context to reproduce. We
intentionally do not swallow or log-and-continue: a silent-bug in a recovery
experiment pollutes empirical data and is strictly worse than a crash.

Callers are expected to register an optional `RunEventLogger`, which will
emit an `invariant_violated` event before the exception propagates. The
exception still raises; the event is an audit trail, not a suppression.

Typical placement:

- `assert_workspace_reset(...)` — after materializing a workspace and before
  restoring a checkpoint.
- `assert_checkpoint_restored(...)` — after `checkpoint_store.restore(...)`.
- `assert_budget_monotonic(...)` — on every `usage_after - usage_before` delta.
- `assert_manifest_lock(...)` — before running a natural-pool case.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from bcmr_swe.observability import RunEventLogger


class InvariantViolation(AssertionError):
    """Raised when a BCMR critical-path invariant is violated."""


@dataclass(slots=True, frozen=True)
class Invariant:
    name: str
    details: dict[str, Any]


def _emit_and_raise(
    logger: RunEventLogger | None,
    inv: Invariant,
) -> None:
    if logger is not None:
        try:
            logger.emit(
                "invariant_violated",
                payload={"invariant": inv.name, "details": inv.details},
            )
        except Exception:
            # Never let observability failure mask the invariant failure.
            pass
    raise InvariantViolation(f"[{inv.name}] {inv.details}")


# ---------- Workspace / checkpoint ------------------------------------------

def assert_workspace_reset(
    workspace: Path,
    *,
    allow_nonempty: bool = False,
    logger: RunEventLogger | None = None,
) -> None:
    workspace = Path(workspace)
    if not workspace.exists():
        return
    if allow_nonempty:
        return
    contents = [p for p in workspace.iterdir() if p.name not in {".git"}]
    if contents:
        _emit_and_raise(
            logger,
            Invariant(
                name="workspace_not_reset",
                details={
                    "workspace": str(workspace),
                    "leftover_entries": [p.name for p in contents][:16],
                },
            ),
        )


def assert_checkpoint_restored(
    workspace: Path,
    *,
    expected_commit: str | None = None,
    allow_dirty: bool = False,
    logger: RunEventLogger | None = None,
) -> None:
    workspace = Path(workspace)
    if not workspace.exists():
        _emit_and_raise(
            logger,
            Invariant(
                name="workspace_missing_after_restore",
                details={"workspace": str(workspace)},
            ),
        )
    if not (workspace / ".git").exists():
        return  # non-git workspaces are out of scope for this invariant
    actual_commit = _git_head(workspace)
    if expected_commit and actual_commit and expected_commit != actual_commit:
        _emit_and_raise(
            logger,
            Invariant(
                name="checkpoint_commit_mismatch",
                details={
                    "workspace": str(workspace),
                    "expected_commit": expected_commit,
                    "actual_commit": actual_commit,
                },
            ),
        )
    if not allow_dirty:
        status = _git_status_porcelain(workspace)
        if status:
            _emit_and_raise(
                logger,
                Invariant(
                    name="checkpoint_restore_left_dirty_workspace",
                    details={
                        "workspace": str(workspace),
                        "actual_commit": actual_commit,
                        "status_excerpt": status[:12],
                    },
                ),
            )


def _git_head(workspace: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_status_porcelain(workspace: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


# ---------- Budget accounting -----------------------------------------------

def assert_budget_monotonic(
    usage_before: dict[str, float],
    usage_after: dict[str, float],
    *,
    min_delta_tokens: float = 0.0,
    label: str = "budget",
    logger: RunEventLogger | None = None,
) -> float:
    before = float((usage_before or {}).get("total_tokens", 0.0) or 0.0)
    after = float((usage_after or {}).get("total_tokens", 0.0) or 0.0)
    if after < before:
        _emit_and_raise(
            logger,
            Invariant(
                name="budget_counter_went_backwards",
                details={
                    "label": label,
                    "usage_before_total_tokens": before,
                    "usage_after_total_tokens": after,
                },
            ),
        )
    delta = after - before
    if delta < min_delta_tokens:
        _emit_and_raise(
            logger,
            Invariant(
                name="budget_delta_below_floor",
                details={
                    "label": label,
                    "delta_tokens": delta,
                    "min_delta_tokens": min_delta_tokens,
                },
            ),
        )
    return delta


# ---------- Governance ------------------------------------------------------

def assert_manifest_lock(
    *,
    expected_model: str,
    built_model_name: str,
    expected_strong_model: str = "",
    built_strong_model_name: str = "",
    runtime: str,
    logger: RunEventLogger | None = None,
) -> None:
    if expected_model and built_model_name != expected_model:
        _emit_and_raise(
            logger,
            Invariant(
                name="model_lock_mismatch",
                details={
                    "expected_model": expected_model,
                    "built_model_name": built_model_name,
                },
            ),
        )
    if expected_strong_model and built_strong_model_name != expected_strong_model:
        _emit_and_raise(
            logger,
            Invariant(
                name="strong_model_lock_mismatch",
                details={
                    "expected_strong_model": expected_strong_model,
                    "built_strong_model_name": built_strong_model_name,
                },
            ),
        )
    if str(runtime).strip().lower() == "auto":
        _emit_and_raise(
            logger,
            Invariant(
                name="runtime_auto_not_allowed_for_natural_mainline",
                details={"runtime": runtime},
            ),
        )


# ---------- Candidate-set symmetry ------------------------------------------

def assert_same_checkpoint_across_methods(
    method_checkpoint_map: dict[str, str],
    *,
    logger: RunEventLogger | None = None,
) -> None:
    unique_values = {v for v in method_checkpoint_map.values() if v}
    if len(unique_values) > 1:
        _emit_and_raise(
            logger,
            Invariant(
                name="method_checkpoint_divergence",
                details={"method_checkpoint_map": dict(method_checkpoint_map)},
            ),
        )


def assert_recoverable_denominator_present(
    row: dict[str, Any],
    *,
    logger: RunEventLogger | None = None,
) -> None:
    if "recoverable" not in row:
        _emit_and_raise(
            logger,
            Invariant(
                name="row_missing_recoverable_flag",
                details={"keys": sorted(row.keys())},
            ),
        )
