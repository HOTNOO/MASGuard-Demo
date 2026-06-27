"""A tiny user-owned MAS repair system used for the Tool Demo.

This program is deliberately separate from MASGuard. It represents the
kind of MAS a user already has before installing MASGuard.

The baseline repair attempt is intentionally flawed: it over-focuses on the
failing assertion and edits the test expectation instead of the source. After
MASGuard produces a recovery prompt, the same mini-MAS can resume from that
prompt and emit a source patch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "mas_plugin_case"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimas", description="Demo user's original MAS repair system.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_case = subparsers.add_parser("new-case", help="Create a fresh repair task for the user's MAS.")
    new_case.add_argument("--case-dir", required=True, type=Path)

    show = subparsers.add_parser("show-task", help="Show the buggy code and expected behavior.")
    show.add_argument("--case-dir", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="Run the user's original validation command.")
    validate.add_argument("--case-dir", required=True, type=Path)

    repair = subparsers.add_parser("repair", help="Run the baseline MAS repair attempt.")
    repair.add_argument("--case-dir", required=True, type=Path)

    resume = subparsers.add_parser("resume", help="Resume the same MAS using a MASGuard recovery prompt.")
    resume.add_argument("--case-dir", required=True, type=Path)
    resume.add_argument("--prompt", required=True, type=Path)
    resume.add_argument("--output-patch", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "new-case":
        return _new_case(args.case_dir)
    if args.command == "show-task":
        return _show_task(args.case_dir)
    if args.command == "validate":
        result = _run_validation(args.case_dir / "repo")
        _print_validation(result)
        return result.returncode
    if args.command == "repair":
        return _repair(args.case_dir)
    if args.command == "resume":
        return _resume(args.case_dir, args.prompt, args.output_patch)
    return 2


def _new_case(case_dir: Path) -> int:
    case_dir = case_dir.resolve()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "patches").mkdir(parents=True)
    shutil.copytree(CASE / "repo", case_dir / "repo")
    print("MINI-MAS TASK CREATED")
    print(f"case_dir: {case_dir}")
    print(f"repo:     {case_dir / 'repo'}")
    print("validation_command: python tests/test_widget.py")
    return 0


def _show_task(case_dir: Path) -> int:
    case_dir = case_dir.resolve()
    repo = case_dir / "repo"
    print("MINI-MAS TASK: buggy source")
    print("=" * 72)
    print((repo / "widget.py").read_text(encoding="utf-8"))
    print("MINI-MAS TASK: expected behavior")
    print("=" * 72)
    print((repo / "tests" / "test_widget.py").read_text(encoding="utf-8"))
    return 0


def _repair(case_dir: Path) -> int:
    case_dir = case_dir.resolve()
    repo = case_dir / "repo"
    logs = case_dir / "logs"
    patches = case_dir / "patches"
    logs.mkdir(parents=True, exist_ok=True)
    patches.mkdir(parents=True, exist_ok=True)

    print("MINI-MAS BASELINE REPAIR START")
    initial = _run_validation(repo)
    _write_log(logs / "initial.log", initial)
    print(f"initial_validation_returncode: {initial.returncode}")

    failed_patch = patches / "failed.patch"
    shutil.copy2(CASE / "patches" / "failed.patch", failed_patch)
    print("locator: inspected tests/test_widget.py")
    print("planner: over-focused on the assertion text")
    print("patcher: produced a test-only patch")
    _apply_patch(repo, failed_patch)

    after = _run_validation(repo)
    _write_log(logs / "failing.log", after)
    _write_trajectory(case_dir / "trajectory.json")
    print(f"post_patch_validation_returncode: {after.returncode}")
    print(f"failed_patch: {failed_patch}")
    print(f"failing_log:  {logs / 'failing.log'}")
    if after.returncode == 0:
        print("MINI-MAS BASELINE RESULT: SUCCESS")
        return 0
    print("MINI-MAS BASELINE RESULT: FAILED")
    print("MASGuard has not run yet. The failed-run artifacts are now ready.")
    return 1


def _resume(case_dir: Path, prompt: Path, output_patch: Path) -> int:
    case_dir = case_dir.resolve()
    prompt = prompt.resolve()
    output_patch = output_patch.resolve()
    if not prompt.exists():
        raise FileNotFoundError(prompt)
    prompt_text = prompt.read_text(encoding="utf-8", errors="replace")
    print("MINI-MAS RESUME START")
    print(f"recovery_prompt: {prompt}")
    if "rollback-and-relocalize-from-evidence" in prompt_text:
        print("resume_signal: rollback-and-relocalize-from-evidence")
    if "widget.py" in prompt_text:
        print("resume_evidence: prompt points the MAS back to widget.py")

    # The resumed MAS restores its checkpoint before the misleading test edit.
    repo = case_dir / "repo"
    shutil.copy2(CASE / "repo" / "tests" / "test_widget.py", repo / "tests" / "test_widget.py")
    shutil.copy2(CASE / "repo" / "widget.py", repo / "widget.py")

    output_patch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CASE / "recovery_patches" / "recovered.patch", output_patch)
    print("patcher: generated a source patch after reading the recovery prompt")
    print(f"recovered_patch: {output_patch}")
    print("MINI-MAS RESUME RESULT: PATCH_READY")
    return 0


def _run_validation(repo: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        ["python", "tests/test_widget.py"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _print_validation(result: subprocess.CompletedProcess[str]) -> None:
    print("MINI-MAS VALIDATION")
    print(f"returncode: {result.returncode}")
    if result.returncode == 0:
        print("status: PASS")
    else:
        print("status: FAIL")
    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)


def _write_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    verifier_note = ""
    if result.returncode != 0:
        verifier_note = (
            "\n--- verifier note ---\n"
            'File "widget.py", line 3, in normalize\n'
            'observed normalize(None) == "" but expected "unknown"\n'
        )
    path.write_text(
        "$ python tests/test_widget.py\n"
        f"returncode: {result.returncode}\n\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
        f"{verifier_note}",
        encoding="utf-8",
    )


def _write_trajectory(path: Path) -> None:
    payload = {
        "issue": "Fix widget.normalize so None values normalize to 'unknown' while preserving text normalization.",
        "tool_calls": [
            {
                "role": "verifier",
                "command": "python tests/test_widget.py",
                "status": "failed",
                "summary": "Verifier reproduced the failing test.",
            },
            {
                "role": "locator",
                "command": "inspect tests/test_widget.py",
                "status": "ok",
                "summary": "Locator over-focused on the test assertion and did not inspect widget.py.",
            },
            {
                "role": "patcher",
                "command": "edit tests/test_widget.py",
                "status": "ok",
                "summary": "Patcher changed the test expectation instead of the source behavior.",
            },
            {
                "role": "verifier",
                "command": "python tests/test_widget.py",
                "status": "failed",
                "summary": "Validation failed after the baseline MAS patch.",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_patch(repo: Path, patch_path: Path) -> None:
    subprocess.run(["patch", "--no-backup-if-mismatch", "-p1", "-i", str(patch_path.resolve())], cwd=repo, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
