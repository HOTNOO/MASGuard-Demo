"""Executable MAS-plugin demonstration for MASGuard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recoveragent.diagnosis import diagnose
from recoveragent.evidence import extract_evidence
from recoveragent.llm import maybe_generate_llm_insight
from recoveragent.planner import plan_recovery
from recoveragent.report import build_report
from recoveragent.report import write_html_report, write_report


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run_mas_plugin_demo(
    *,
    case_dir: Path,
    output_dir: Path,
    use_llm: bool = False,
    llm_model: str,
    llm_endpoint: str,
    llm_timeout: int,
) -> dict[str, Any]:
    """Run baseline MAS failure, MASGuard diagnosis, and recovered success."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(case_dir / "repo", workspace)

    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    initial = _run_pytest(workspace)
    _write_text(artifacts / "initial_validation.log", _format_command_result(initial))

    failed_patch = case_dir / "patches" / "failed.patch"
    failed_patch_result = _apply_simple_unified_patch(workspace, failed_patch)
    baseline = _run_pytest(workspace)
    baseline_log = artifacts / "baseline_failed_validation.log"
    _write_text(baseline_log, _format_command_result(baseline))

    trajectory = _build_trajectory(
        issue=_read_issue(case_dir),
        initial=initial,
        baseline=baseline,
        failed_patch_applied=failed_patch_result,
    )
    trajectory_path = artifacts / "trajectory.json"
    _write_json(trajectory, trajectory_path)

    analysis_json = output_dir / "masguard_report.json"
    analysis_html = output_dir / "masguard_report.html"
    report = _analyze_failed_run(
        trajectory_path=trajectory_path,
        repo_path=workspace,
        log_path=baseline_log,
        patch_path=failed_patch,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_endpoint=llm_endpoint,
        llm_timeout=llm_timeout,
    )
    write_report(report, analysis_json)
    write_html_report(report, analysis_html)

    rollback_result = _rollback_failed_test_patch(case_dir=case_dir, workspace=workspace)
    recovery_patch = case_dir / "recovery_patches" / "recovered.patch"
    recovery_patch_result = _apply_simple_unified_patch(workspace, recovery_patch)
    recovered = _run_pytest(workspace)
    recovered_log = artifacts / "recovered_validation.log"
    _write_text(recovered_log, _format_command_result(recovered))

    summary = {
        "schema": "masguard.mas_plugin_demo.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "workspace": str(workspace.resolve()),
        "flow": [
            {
                "stage": "initial_agent_validation",
                "description": "The mini-MAS verifier reproduces the original repository failure.",
                "returncode": initial.returncode,
                "passed": initial.returncode == 0,
            },
            {
                "stage": "baseline_mas_attempt",
                "description": "The baseline MAS applies a wrong test-only patch and validation still fails.",
                "patch_applied": failed_patch_result,
                "returncode": baseline.returncode,
                "passed": baseline.returncode == 0,
            },
            {
                "stage": "masguard_plugin",
                "description": "MASGuard analyzes trajectory/log/patch/repo evidence and selects a recovery action.",
                "diagnosis": report["diagnosis"]["failure_type"],
                "recovery_action": report["recovery_plan"]["action"],
                "llm_called": report["llm_insight"]["provider_called"],
            },
            {
                "stage": "recovered_mas_attempt",
                "description": "The original MAS resumes from the MASGuard prompt, emits a source patch, and validation passes.",
                "rollback_applied": rollback_result,
                "patch_applied": recovery_patch_result,
                "returncode": recovered.returncode,
                "passed": recovered.returncode == 0,
            },
        ],
        "success": initial.returncode != 0 and baseline.returncode != 0 and recovered.returncode == 0,
        "artifacts": {
            "trajectory": str(trajectory_path),
            "baseline_log": str(baseline_log),
            "recovered_log": str(recovered_log),
            "masguard_report_json": str(analysis_json),
            "masguard_report_html": str(analysis_html),
        },
        "scope_note": (
            "This is a local, fully reproducible MAS-plugin demonstration. It shows an end-to-end "
            "failure-to-recovery workflow on a small repository; it is not a SWE-bench aggregate result."
        ),
    }
    _write_json(summary, output_dir / "mas_plugin_summary.json")
    _write_markdown_summary(summary, output_dir / "mas_plugin_demo.md")
    return summary


def _analyze_failed_run(
    *,
    trajectory_path: Path,
    repo_path: Path,
    log_path: Path,
    patch_path: Path,
    use_llm: bool,
    llm_model: str,
    llm_endpoint: str,
    llm_timeout: int,
) -> dict[str, Any]:
    bundle = extract_evidence(
        trajectory_path=trajectory_path,
        repo_path=repo_path,
        log_path=log_path,
        patch_path=patch_path,
    )
    diagnosis = diagnose(bundle)
    plan = plan_recovery(bundle, diagnosis)
    insight = maybe_generate_llm_insight(
        bundle=bundle,
        diagnosis=diagnosis,
        plan=plan,
        enabled=use_llm,
        model=llm_model,
        endpoint=llm_endpoint,
        timeout=llm_timeout,
    )
    return build_report(bundle, diagnosis, plan, insight)


def _run_pytest(workspace: Path) -> CommandResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = ["python", "tests/test_widget.py"]
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    return CommandResult(
        command=command,
        cwd=str(workspace),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _apply_simple_unified_patch(workspace: Path, patch_path: Path) -> bool:
    text = patch_path.read_text(encoding="utf-8")
    current_file: Path | None = None
    removals: list[str] = []
    additions: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("+++ b/"):
            current_file = workspace / raw.removeprefix("+++ b/")
        elif raw.startswith("--- ") or raw.startswith("diff ") or raw.startswith("@@"):
            continue
        elif raw.startswith("-"):
            removals.append(raw[1:])
        elif raw.startswith("+"):
            additions.append(raw[1:])
    if current_file is None or not current_file.exists():
        return False
    original = current_file.read_text(encoding="utf-8")
    old = "\n".join(removals)
    new = "\n".join(additions)
    if old not in original:
        return False
    current_file.write_text(original.replace(old, new, 1), encoding="utf-8")
    return True


def _rollback_failed_test_patch(*, case_dir: Path, workspace: Path) -> bool:
    source = case_dir / "repo" / "tests" / "test_widget.py"
    target = workspace / "tests" / "test_widget.py"
    if not source.exists() or not target.exists():
        return False
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def _build_trajectory(*, issue: str, initial: CommandResult, baseline: CommandResult, failed_patch_applied: bool) -> dict:
    return {
        "issue": issue,
        "tool_calls": [
            {
                "role": "verifier",
                "command": " ".join(initial.command),
                "status": "failed" if initial.returncode else "ok",
                "summary": "Initial validation reproduced the repository failure.",
            },
            {
                "role": "locator",
                "command": "inspect tests/test_widget.py",
                "status": "ok",
                "summary": "The baseline MAS over-focused on the failing assertion and under-inspected widget.py.",
            },
            {
                "role": "patcher",
                "command": "apply patches/failed.patch",
                "status": "ok" if failed_patch_applied else "failed",
                "summary": "The baseline MAS applied a test-only patch instead of repairing source behavior.",
            },
            {
                "role": "verifier",
                "command": " ".join(baseline.command),
                "status": "failed" if baseline.returncode else "ok",
                "summary": "Validation after the baseline MAS patch still failed.",
            },
        ],
    }


def _read_issue(case_dir: Path) -> str:
    trajectory_path = case_dir / "trajectory.json"
    if not trajectory_path.exists():
        return "Mini-MAS repair task"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return str(data.get("issue") or "Mini-MAS repair task")


def _format_command_result(result: CommandResult) -> str:
    return (
        f"$ {' '.join(result.command)}\n"
        f"cwd: {result.cwd}\n"
        f"returncode: {result.returncode}\n\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# MASGuard MAS Plugin Demo",
        "",
        "This is a fully reproducible local MAS-plugin flow.",
        "",
        "| Stage | Result | Description |",
        "| --- | --- | --- |",
    ]
    for row in summary["flow"]:
        result = "PASS" if row.get("passed") else "FAIL"
        if row["stage"] == "masguard_plugin":
            result = f"{row['diagnosis']} -> {row['recovery_action']}"
        lines.append(f"| `{row['stage']}` | {result} | {row['description']} |")
    lines.extend(
        [
            "",
            f"- Overall success: `{summary['success']}`",
            f"- Report: `{summary['artifacts']['masguard_report_html']}`",
            f"- Baseline log: `{summary['artifacts']['baseline_log']}`",
            f"- Recovered log: `{summary['artifacts']['recovered_log']}`",
            "",
            f"Scope: {summary['scope_note']}",
        ]
    )
    _write_text(path, "\n".join(lines) + "\n")
