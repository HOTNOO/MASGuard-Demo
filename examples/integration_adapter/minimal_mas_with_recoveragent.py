"""Minimal example showing how a MAS repair pipeline calls MASGuard.

This file is intentionally small and readable. It is not the main tool
implementation. It demonstrates where a user's MAS would call MASGuard:

1. baseline MAS attempts a patch and fails validation;
2. adapter exports trajectory/log/patch/repo artifacts;
3. MASGuard analyzes the failed run;
4. adapter passes the recovery prompt back to the same MAS patcher;
5. the resumed MAS emits a recovered patch and validation passes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from recoveragent.diagnosis import diagnose
from recoveragent.evidence import extract_evidence
from recoveragent.patching import apply_simple_unified_patch, run_validation
from recoveragent.planner import plan_recovery
from recoveragent.prompt import build_recovery_prompt
from recoveragent.report import build_report, write_html_report, write_report


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "mas_plugin_case"
OUT = ROOT / "demo_outputs" / "adapter_example"
WORKSPACE = OUT / "workspace"
ARTIFACTS = OUT / "artifacts"


def main() -> int:
    reset_workspace()

    print("1. User's original MAS validates the repository.")
    initial = run_command(["python", "tests/test_widget.py"], WORKSPACE)
    write_log(ARTIFACTS / "initial_validation.log", initial)
    print(f"   returncode={initial.returncode}")

    print("2. Baseline MAS applies a wrong test-only patch.")
    baseline_patch = CASE / "patches" / "failed.patch"
    subprocess.run(["patch", "-p1", "-i", str(baseline_patch)], cwd=WORKSPACE, check=True)
    baseline = run_command(["python", "tests/test_widget.py"], WORKSPACE)
    baseline_log = ARTIFACTS / "baseline_failed_validation.log"
    write_log(baseline_log, baseline)
    print(f"   returncode={baseline.returncode}")

    print("3. Adapter exports failed-run artifacts and calls MASGuard.")
    trajectory = export_trajectory()
    trajectory_path = ARTIFACTS / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

    bundle = extract_evidence(
        trajectory_path=trajectory_path,
        repo_path=WORKSPACE,
        log_path=baseline_log,
        patch_path=baseline_patch,
    )
    diagnosis = diagnose(bundle)
    plan = plan_recovery(bundle, diagnosis)
    report = build_report(bundle, diagnosis, plan)
    write_report(report, OUT / "masguard_report.json")
    write_html_report(report, OUT / "masguard_report.html")
    (OUT / "recovery_prompt.md").write_text(build_recovery_prompt(report), encoding="utf-8")
    print(f"   diagnosis={diagnosis.failure_type}")
    print(f"   action={plan.action}")

    print("4. Original MAS consumes the recovery prompt and resumes repair.")
    print("   For reproducibility, the resumed MAS emits the checked-in recovered.patch.")
    shutil.copy2(CASE / "repo" / "tests" / "test_widget.py", WORKSPACE / "tests" / "test_widget.py")
    shutil.copy2(CASE / "repo" / "widget.py", WORKSPACE / "widget.py")
    apply_result = apply_simple_unified_patch(WORKSPACE, CASE / "recovery_patches" / "recovered.patch")
    print(f"   source_patch_applied={apply_result.applied}")

    print("5. Adapter asks MASGuard to validate recovered workspace.")
    recovered = run_validation(WORKSPACE, "python tests/test_widget.py")
    write_log(ARTIFACTS / "recovered_validation.log", completed_like(recovered))
    print(f"   recovered_passed={recovered.passed}")

    summary = {
        "baseline_failed": baseline.returncode != 0,
        "masguard_diagnosis": diagnosis.failure_type,
        "masguard_action": plan.action,
        "recovered_passed": recovered.passed,
        "report": str(OUT / "masguard_report.html"),
        "prompt": str(OUT / "recovery_prompt.md"),
    }
    (OUT / "adapter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(summary, indent=2))
    return 0 if recovered.passed else 1


def reset_workspace() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE / "repo", WORKSPACE)


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        f"$ {' '.join(completed.args if isinstance(completed.args, list) else [str(completed.args)])}\n"
        f"returncode: {completed.returncode}\n\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )


def completed_like(result):
    return subprocess.CompletedProcess(
        args=result.command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def export_trajectory() -> dict:
    return {
        "issue": "Fix widget.normalize so None normalizes to 'unknown' while preserving text normalization.",
        "tool_calls": [
            {"role": "verifier", "command": "python tests/test_widget.py", "status": "failed"},
            {"role": "locator", "command": "inspect tests/test_widget.py", "status": "ok"},
            {"role": "patcher", "command": "patch -p1 < failed.patch", "status": "ok"},
            {"role": "verifier", "command": "python tests/test_widget.py", "status": "failed"},
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
