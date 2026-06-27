"""Standard failed-run directory contract for MASGuard integrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunLayout:
    run_dir: Path
    repo: Path
    trajectory: Path
    log: Path
    failed_patch: Path
    output_dir: Path
    report_json: Path
    report_html: Path
    recovery_prompt: Path


def layout_for(run_dir: Path) -> RunLayout:
    root = run_dir.resolve()
    output_dir = root / "recoveragent"
    return RunLayout(
        run_dir=root,
        repo=root / "repo",
        trajectory=root / "trajectory.json",
        log=root / "logs" / "failing.log",
        failed_patch=root / "patches" / "failed.patch",
        output_dir=output_dir,
        report_json=output_dir / "report.json",
        report_html=output_dir / "report.html",
        recovery_prompt=output_dir / "recovery_prompt.md",
    )


def init_run_dir(run_dir: Path, *, force: bool = False) -> RunLayout:
    layout = layout_for(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.repo.mkdir(parents=True, exist_ok=True)
    layout.log.parent.mkdir(parents=True, exist_ok=True)
    layout.failed_patch.parent.mkdir(parents=True, exist_ok=True)
    layout.output_dir.mkdir(parents=True, exist_ok=True)

    _write_if_missing(layout.trajectory, _sample_trajectory(), force=force)
    _write_if_missing(layout.log, _sample_log(), force=force)
    _write_if_missing(layout.failed_patch, _sample_patch(), force=force)
    _write_if_missing(layout.run_dir / "README.md", _contract_readme(layout), force=force)
    return layout


def validate_run_dir(run_dir: Path) -> RunLayout:
    layout = layout_for(run_dir)
    missing = [
        path
        for path in (layout.repo, layout.trajectory, layout.log, layout.failed_patch)
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"MASGuard run directory is missing required inputs:\n{joined}")
    return layout


def print_layout(layout: RunLayout) -> str:
    return (
        f"run_dir:         {layout.run_dir}\n"
        f"repo snapshot:   {layout.repo}\n"
        f"trajectory:      {layout.trajectory}\n"
        f"failing log:     {layout.log}\n"
        f"failed patch:    {layout.failed_patch}\n"
        f"outputs:         {layout.output_dir}"
    )


def _write_if_missing(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sample_trajectory() -> str:
    payload = {
        "issue": "Replace this with the user issue or repair task.",
        "tool_calls": [
            {
                "role": "verifier",
                "command": "replace with validation command",
                "status": "failed",
                "summary": "Initial failure before the MAS patch.",
            },
            {
                "role": "patcher",
                "command": "replace with the MAS patch command or tool call",
                "status": "ok",
                "summary": "The MAS produced a candidate patch.",
            },
            {
                "role": "verifier",
                "command": "replace with validation command",
                "status": "failed",
                "summary": "Validation still failed after the MAS patch.",
            },
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def _sample_log() -> str:
    return (
        "Replace this file with the failing test/build log after the MAS patch.\n"
        "MASGuard uses this log to extract stack traces, failed tests, and environment errors.\n"
    )


def _sample_patch() -> str:
    return (
        "diff --git a/path/to/file.py b/path/to/file.py\n"
        "--- a/path/to/file.py\n"
        "+++ b/path/to/file.py\n"
        "@@ -1 +1 @@\n"
        "-replace this with the failed MAS patch\n"
        "+replace this with the failed MAS patch\n"
    )


def _contract_readme(layout: RunLayout) -> str:
    return (
        "# MASGuard Failed-Run Directory\n\n"
        "This directory is the integration boundary between an upstream MAS repair system and MASGuard.\n\n"
        "Required inputs:\n\n"
        "- `repo/`: repository snapshot after the failed MAS attempt, or a workspace that can be validated.\n"
        "- `trajectory.json`: MAS tool calls, roles, commands, and statuses.\n"
        "- `logs/failing.log`: validation log after the MAS candidate patch failed.\n"
        "- `patches/failed.patch`: the failed MAS candidate patch.\n\n"
        "Analyze this run:\n\n"
        "```bash\n"
        f"masguard analyze-run --run-dir {layout.run_dir}\n"
        "```\n\n"
        "Then pass `recoveragent/recovery_prompt.md` back to the same MAS patcher.\n"
    )
